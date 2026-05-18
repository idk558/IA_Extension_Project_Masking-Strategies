# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE_DIR = PROJECT_ROOT / ".cache"
os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_DIR / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_DIR / "matplotlib"))

import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODALITIES = ["tok_rgb@256", "tok_depth@256", "tok_normal@256", "scene_desc"]
DEFAULT_CONFIG = "cfgs/nano4M/multiclevr_d6-6w512.yaml"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate CLEVR caption-to-image fidelity")
    parser.add_argument("--checkpoint", required=True, help="Path to a nano4M safetensors checkpoint")
    parser.add_argument("--baseline-checkpoint", default=None, help="Optional baseline checkpoint to compare")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="nano4M YAML config to read dataset settings from")
    parser.add_argument("--root-dir", default=None, help="CLEVR dataset root. Overrides the YAML config when set")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output-dir", default="outputs/clevr_fidelity", help="Directory for images and JSON report")
    parser.add_argument("--report-name", default="report.json", help="JSON report filename")
    parser.add_argument("--text-tokenizer-path", default="gpt2", help="Tokenizer used for scene descriptions")
    parser.add_argument("--text-max-length", type=int, default=256, help="Maximum scene description length")
    parser.add_argument("--image-tokenizer-dir", default="/tmp/nvidia/Cosmos-0.1-Tokenizer-DI16x16")
    parser.add_argument("--download-image-tokenizer", action="store_true", help="Download Cosmos tokenizer if missing")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detector-device", default=None, help="Device for Grounding DINO. Defaults to --device")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--num-steps", type=int, default=64, help="ROAR decoding steps per generated modality")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=float, default=0.0)
    parser.add_argument(
        "--generation-targets",
        default="tok_rgb@256",
        help="Comma-separated modalities to generate in order, ending with tok_rgb@256",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    from nanofm.data.multimodal.simple_multimodal_dataset import SimpleMultimodalDataset
    from nanofm.eval.clevr_verifier import GroundingDINOVerifier

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root_dir = resolve_root_dir(args)

    dataset = SimpleMultimodalDataset(
        root_dir=root_dir,
        split=args.split,
        modalities=["scene_desc"],
        transforms=None,
        sample_from_k_augmentations=1,
        text_tokenizer_path=args.text_tokenizer_path,
        text_max_length=args.text_max_length,
    )
    image_tokenizer = load_image_tokenizer(args, device)
    verifier = GroundingDINOVerifier(
        device=args.detector_device or device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    jobs = [("candidate", args.checkpoint)]
    if args.baseline_checkpoint is not None:
        jobs.insert(0, ("baseline", args.baseline_checkpoint))

    reports = []
    for label, checkpoint in jobs:
        reports.append(evaluate_checkpoint(
            label=label,
            checkpoint=checkpoint,
            dataset=dataset,
            image_tokenizer=image_tokenizer,
            verifier=verifier,
            args=args,
            device=device,
            output_dir=output_dir / label,
        ))

    report = {
        "num_samples": min(args.num_samples, len(dataset)),
        "split": args.split,
        "root_dir": root_dir,
        "reports": reports,
        "comparison_table": comparison_table(reports),
    }
    report_path = output_dir / args.report_name
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["comparison_table"], indent=2))
    print(f"Wrote report to {report_path}")


def resolve_root_dir(args: argparse.Namespace) -> str:
    if args.root_dir is not None:
        return args.root_dir

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    loader_config = config.get("eval_loader_config", {})
    root_dir = loader_config.get("root_dir")
    if root_dir is None:
        raise ValueError(f"No eval_loader_config.root_dir found in {config_path}")
    return str(root_dir)


def evaluate_checkpoint(
        label: str,
        checkpoint: str,
        dataset: Any,
        image_tokenizer: Any,
        verifier: Any,
        args: argparse.Namespace,
        device: torch.device,
        output_dir: Path,
    ) -> Dict[str, Any]:
    from nanofm.eval.clevr_verifier import compute_fidelity_score
    from nanofm.utils.checkpoint import load_model_from_safetensors

    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_from_safetensors(checkpoint, device=device, to_eval=True)

    sample_count = min(args.num_samples, len(dataset))
    examples = []
    scores = []
    category_values: Dict[str, List[float]] = {}
    target_mods = [mod.strip() for mod in args.generation_targets.split(",") if mod.strip()]

    for idx in range(sample_count):
        sample = dataset[idx]
        caption_tokens = sample["scene_desc"]
        caption = dataset.text_tokenizer.decode(caption_tokens, skip_special_tokens=False)
        image = generate_caption_to_image(
            model=model,
            caption_tokens=caption_tokens,
            dataset=dataset,
            image_tokenizer=image_tokenizer,
            target_mods=target_mods,
            args=args,
            device=device,
        )
        image_path = output_dir / f"{idx:06d}.png"
        image.save(image_path)

        score = compute_fidelity_score(caption, image, verifier=verifier)
        scores.append(score["score"])
        for category, values in score["per_category_breakdown"].items():
            category_values.setdefault(category, []).append(values["match"])

        examples.append({
            "idx": idx,
            "caption": caption,
            "image_path": str(image_path),
            "score": score["score"],
            "category_score": score["category_score"],
            "hallucination_penalty": score["hallucination_penalty"],
            "per_category_breakdown": score["per_category_breakdown"],
            "expected": score["expected"],
            "detected": score["detected"],
        })

        print(f"{label} [{idx + 1}/{sample_count}] score={score['score']:.3f}")

    sorted_examples = sorted(examples, key=lambda item: item["score"])
    return {
        "label": label,
        "checkpoint": checkpoint,
        "mean_score": mean(scores),
        "std_score": std(scores),
        "per_category_breakdown": summarize_categories(category_values),
        "worst_10": sorted_examples[:10],
        "best_10": list(reversed(sorted_examples[-10:])),
    }


def generate_caption_to_image(
        model: torch.nn.Module,
        caption_tokens: torch.Tensor,
        dataset: Any,
        image_tokenizer: Any,
        target_mods: Sequence[str],
        args: argparse.Namespace,
        device: torch.device,
    ) -> Image.Image:
    if not target_mods or target_mods[-1] != "tok_rgb@256":
        raise ValueError("--generation-targets must end with tok_rgb@256")

    pad_token_id = dataset.text_tokenizer.pad_token_id
    valid = caption_tokens != pad_token_id
    if not bool(valid.any()):
        valid = torch.ones_like(caption_tokens, dtype=torch.bool)

    positions = torch.arange(caption_tokens.numel(), device=device)[valid.to(device)]
    x_tokens = caption_tokens[valid.cpu()].unsqueeze(0).to(device)
    x_positions = positions.unsqueeze(0)
    scene_desc_idx = model.modalities.index("scene_desc")
    x_modalities = torch.full_like(x_positions, fill_value=scene_desc_idx)

    pred_tokens = None
    with torch.inference_mode():
        for target_mod in target_mods:
            pred_tokens, x_tokens, x_positions, x_modalities = model.generate_one_modality_roar(
                x_tokens,
                x_positions,
                x_modalities,
                target_mod=target_mod,
                num_steps=args.num_steps,
                temp=args.temp,
                top_p=args.top_p,
                top_k=args.top_k,
            )

    return token_ids_to_image(pred_tokens, image_tokenizer, device)


def token_ids_to_image(token_ids: torch.Tensor, image_tokenizer: Any, device: torch.device) -> Image.Image:
    n_tokens = token_ids.numel()
    side = int(math.sqrt(n_tokens))
    if side * side != n_tokens:
        raise ValueError(f"Expected square image tokens, got {n_tokens}")
    token_ids = token_ids.reshape(1, side, side).to(device)
    with torch.inference_mode():
        reconst = image_tokenizer.decode(token_ids)
    reconst = (reconst[0].clamp(-1, 1).float().cpu() + 1) / 2
    return TF.to_pil_image(reconst)


def load_image_tokenizer(args: argparse.Namespace, device: torch.device) -> Any:
    tokenizer_dir = Path(args.image_tokenizer_dir)
    encoder = tokenizer_dir / "encoder.jit"
    decoder = tokenizer_dir / "decoder.jit"

    if args.download_image_tokenizer and (not encoder.exists() or not decoder.exists()):
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="nvidia/Cosmos-0.1-Tokenizer-DI16x16",
            local_dir=str(tokenizer_dir),
        )

    if not encoder.exists() or not decoder.exists():
        raise FileNotFoundError(
            "Cosmos image tokenizer checkpoints were not found. "
            "Pass --image-tokenizer-dir or use --download-image-tokenizer."
        )

    from cosmos_tokenizer.image_lib import ImageTokenizer
    tokenizer_dtype = "float32" if device.type == "cpu" else "bfloat16"
    return ImageTokenizer(
        checkpoint_enc=str(encoder),
        checkpoint_dec=str(decoder),
        device=str(device),
        dtype=tokenizer_dtype,
    )


def summarize_categories(category_values: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    return {
        category: {
            "mean_match": mean(values),
            "std_match": std(values),
            "num_examples": len(values),
        }
        for category, values in sorted(category_values.items())
    }


def comparison_table(reports: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for report in reports:
        rows.append({
            "label": report["label"],
            "checkpoint": report["checkpoint"],
            "mean_score": report["mean_score"],
            "std_score": report["std_score"],
            "worst_score": report["worst_10"][0]["score"] if report["worst_10"] else None,
            "best_score": report["best_10"][0]["score"] if report["best_10"] else None,
        })
    if len(rows) == 2:
        rows.append({
            "label": "delta(candidate-baseline)",
            "checkpoint": "",
            "mean_score": rows[1]["mean_score"] - rows[0]["mean_score"],
            "std_score": None,
            "worst_score": None,
            "best_score": None,
        })
    return rows


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return float((sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5)


if __name__ == "__main__":
    main(get_args())
