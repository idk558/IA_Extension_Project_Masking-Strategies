import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_only import instantiate_from_config, load_checkpoint
from generate_scene_desc_predictions import (
    build_dataset,
    decode_text,
    prepare_encoder_inputs,
    trim_special_tokens,
)
from score_scene_desc_predictions import (
    bleu_scores,
    build_document_frequency,
    cider_score,
    exact_match,
    rouge_l,
    tokenize,
)


DEFAULT_TOKENIZER_IDS = {
    "tok_rgb": "EPFL-VILAB/4M_tokenizers_rgb_16k_224-448",
    "tok_depth": "EPFL-VILAB/4M_tokenizers_depth_8k_224-448",
    "tok_normal": "EPFL-VILAB/4M_tokenizers_normal_8k_224-448",
}

DEFAULT_MODELS = [
    ("baseline", ""),
    ("v1", ""),
    ("v2", ""),
    ("v3", ""),
    ("v4", ""),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline/V1/V2/V3/V4 on one dataset sample, reconstruct the visual modalities, "
            "and save a markdown report with predictions and scores."
        )
    )
    parser.add_argument("--root-dir", required=True, help="Path to clevr_com_304.")
    parser.add_argument("--split", default="val", help="Dataset split.")
    parser.add_argument("--index", type=int, required=True, help="Dataset index to inspect.")
    parser.add_argument("--device", default="cuda", help="Device for nano4M models.")
    parser.add_argument("--num-steps", type=int, default=8, help="ROAR decoding steps.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=float, default=0.0, help="Top-k.")
    parser.add_argument("--top-p", type=float, default=0.0, help="Top-p.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--decode-image-size", type=int, default=224, help="Image size for 4M detokenization.")
    parser.add_argument("--decode-timesteps", type=int, default=19, help="Detokenizer timesteps for RGB/depth/normal.")
    parser.add_argument(
        "--skip-image-reconstruction",
        action="store_true",
        help="Skip RGB/depth/normal reconstruction and only compare text outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="sample_comparisons",
        help="Directory where images, JSON and markdown report will be written.",
    )
    for model_name, _ in DEFAULT_MODELS:
        parser.add_argument(
            f"--{model_name}-checkpoint",
            required=True,
            help=f"Path to the {model_name} .pth checkpoint.",
        )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_and_dataset(checkpoint_path: str, root_dir: str, split: str, device: str):
    state_dict, model_config, eval_loader_config, _ = load_checkpoint(checkpoint_path)
    if model_config is None or eval_loader_config is None:
        raise ValueError(f"{checkpoint_path} is missing model/eval configs in the checkpoint.")

    model = instantiate_from_config(model_config)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    dataset = build_dataset(eval_loader_config, root_dir=root_dir, split=split)
    return model, dataset, eval_loader_config


def get_sample_prediction(
    model,
    sample: Dict[str, torch.Tensor],
    tokenizer,
    device: str,
    num_steps: int,
    temperature: float,
    top_k: float,
    top_p: float,
) -> Tuple[str, int]:
    target_mod = "scene_desc"
    enc_tokens, enc_positions, enc_modalities = prepare_encoder_inputs(sample, model, target_mod=target_mod)
    enc_tokens = enc_tokens.to(device)
    enc_positions = enc_positions.to(device)
    enc_modalities = enc_modalities.to(device)

    pred_tokens, _, _, _ = model.generate_one_modality_roar(
        enc_input_tokens=enc_tokens,
        enc_input_positions=enc_positions,
        enc_input_modalities=enc_modalities,
        target_mod=target_mod,
        num_steps=num_steps,
        temp=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    predicted_tokens = trim_special_tokens(pred_tokens[0], tokenizer)
    return decode_text(predicted_tokens, tokenizer), int(predicted_tokens.numel())


def load_visual_tokenizers(device: str):
    try:
        from fourm.vq.vqvae import DiVAE
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Could not import 4M visual tokenizers. Run this script in the environment where the 4M "
            "package is installed."
        ) from exc

    tokenizers = {}
    for modality, repo_id in DEFAULT_TOKENIZER_IDS.items():
        tokenizers[modality] = DiVAE.from_pretrained(repo_id).eval().to(device)
    return tokenizers


def get_tokenizer_device(tokenizer_model) -> str:
    try:
        return str(next(tokenizer_model.parameters()).device)
    except StopIteration:
        return "cpu"


def tensor_to_image_array(decoded: torch.Tensor, modality: str) -> np.ndarray:
    if isinstance(decoded, torch.Tensor):
        arr = decoded.detach().cpu()
    else:
        arr = torch.as_tensor(decoded)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        arr = arr.permute(1, 2, 0)
    arr = arr.float().numpy()

    if modality == "tok_rgb":
        if arr.min() < 0.0 or arr.max() > 1.0:
            arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
        return arr

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def save_modalities_from_tokens(
    sample: Dict[str, torch.Tensor],
    output_dir: Path,
    image_size: int,
    timesteps: int,
):
    from PIL import Image

    tokenizers = load_visual_tokenizers("cuda" if torch.cuda.is_available() else "cpu")
    saved_paths = {}

    for modality in ("tok_rgb@256", "tok_depth@256", "tok_normal@256"):
        short_name = modality.split("@")[0]
        if modality not in sample:
            continue
        flat_tokens = sample[modality].flatten().long()
        side = int(round(flat_tokens.numel() ** 0.5))
        if side * side != flat_tokens.numel():
            raise ValueError(
                f"Expected a square token grid for {modality}, got {flat_tokens.numel()} tokens."
            )
        decode_device = get_tokenizer_device(tokenizers[short_name])
        token_tensor = flat_tokens.view(1, side, side).to(decode_device)
        try:
            decoded = tokenizers[short_name].decode_tokens(
                token_tensor,
                image_size=image_size,
                timesteps=timesteps,
            )
        except Exception as exc:
            print(
                f"Warning: could not reconstruct {modality} "
                f"(shape={tuple(token_tensor.shape)}, min={int(flat_tokens.min())}, max={int(flat_tokens.max())}). "
                f"Skipping this modality. Error: {exc}"
            )
            continue
        image_array = tensor_to_image_array(decoded, short_name)
        suffix = short_name.replace("tok_", "")
        output_path = output_dir / f"{suffix}.png"
        Image.fromarray(image_array).save(output_path)
        saved_paths[modality] = output_path

    return saved_paths


def score_prediction(reference_text: str, predicted_text: str) -> Dict[str, float]:
    reference_tokens = tokenize(reference_text)
    predicted_tokens = tokenize(predicted_text)
    document_frequency = build_document_frequency([reference_tokens])
    metrics = bleu_scores(reference_tokens, predicted_tokens)
    metrics["rouge_l"] = rouge_l(reference_tokens, predicted_tokens)
    metrics["cider"] = cider_score(reference_tokens, predicted_tokens, document_frequency, 1)
    metrics["exact_match"] = exact_match(reference_text, predicted_text)
    return metrics


def format_score(value: float) -> str:
    return f"{value:.2f}"


def write_report(
    output_path: Path,
    sample_index: int,
    reference_text: str,
    image_paths: Dict[str, Path],
    predictions: List[Dict],
):
    lines = [f"# Sample {sample_index} Comparison", ""]
    if "tok_rgb@256" in image_paths:
        lines.extend(
            [
                "## Reconstructed Inputs",
                f"![RGB]({image_paths['tok_rgb@256'].name})",
                "",
            ]
        )
    if "tok_depth@256" in image_paths or "tok_normal@256" in image_paths:
        lines.append("Additional reconstructed modalities:")
        for modality in ("tok_depth@256", "tok_normal@256"):
            if modality in image_paths:
                label = modality.replace("@256", "")
                lines.append(f"- `{label}`: `{image_paths[modality].name}`")
        lines.append("")

    lines.extend(
        [
            "## Reference Scene Description",
            "",
            reference_text,
            "",
            "## Predictions",
            "",
        ]
    )
    for pred in predictions:
        lines.append(f"### {pred['model_label']}")
        lines.append(pred["predicted_text"])
        lines.append("")

    lines.extend(
        [
            "## Score Table",
            "",
            "| Model | BLEU-4 | CIDEr | ROUGE-L | Exact Match |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for pred in predictions:
        metrics = pred["metrics"]
        lines.append(
            "| "
            + pred["model_label"]
            + " | "
            + " | ".join(
                [
                    format_score(metrics["bleu_4"]),
                    format_score(metrics["cider"]),
                    format_score(metrics["rouge_l"]),
                    format_score(metrics["exact_match"]),
                ]
            )
            + " |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir) / f"sample_{args.index}"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_checkpoint = getattr(args, "baseline_checkpoint")
    _, dataset, _ = load_model_and_dataset(
        checkpoint_path=baseline_checkpoint,
        root_dir=args.root_dir,
        split=args.split,
        device=args.device,
    )
    if not (0 <= args.index < len(dataset)):
        raise IndexError(f"Dataset index {args.index} is out of range for split {args.split}.")

    sample = dataset[args.index]
    tokenizer = dataset.text_tokenizer
    reference_tokens = trim_special_tokens(sample["scene_desc"], tokenizer)
    reference_text = decode_text(reference_tokens, tokenizer)

    image_paths = {}
    if not args.skip_image_reconstruction:
        image_paths = save_modalities_from_tokens(
            sample=sample,
            output_dir=output_dir,
            image_size=args.decode_image_size,
            timesteps=args.decode_timesteps,
        )

    model_specs = [
        ("baseline", "Baseline", getattr(args, "baseline_checkpoint")),
        ("v1", "V1 text-span / image-random", getattr(args, "v1_checkpoint")),
        ("v2", "V2 text-random / image-block", getattr(args, "v2_checkpoint")),
        ("v3", "V3 text-span / image-block", getattr(args, "v3_checkpoint")),
        ("v4", "V4 text-span-random / image-block-random", getattr(args, "v4_checkpoint")),
    ]

    predictions = []
    for model_key, model_label, checkpoint_path in model_specs:
        model, _, _ = load_model_and_dataset(
            checkpoint_path=checkpoint_path,
            root_dir=args.root_dir,
            split=args.split,
            device=args.device,
        )
        predicted_text, predicted_token_count = get_sample_prediction(
            model=model,
            sample=sample,
            tokenizer=tokenizer,
            device=args.device,
            num_steps=args.num_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        metrics = score_prediction(reference_text, predicted_text)
        predictions.append(
            {
                "model_key": model_key,
                "model_label": model_label,
                "checkpoint": checkpoint_path,
                "predicted_text": predicted_text,
                "predicted_token_count": predicted_token_count,
                "metrics": metrics,
            }
        )

    result = {
        "dataset_index": args.index,
        "reference_text": reference_text,
        "reference_token_count": int(reference_tokens.numel()),
        "reconstructed_images": {key: str(path) for key, path in image_paths.items()},
        "predictions": predictions,
    }

    json_path = output_dir / "comparison.json"
    md_path = output_dir / "comparison.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(md_path, args.index, reference_text, image_paths, predictions)

    print(f"Saved JSON comparison to {json_path}")
    print(f"Saved Markdown report to {md_path}")


if __name__ == "__main__":
    main()
