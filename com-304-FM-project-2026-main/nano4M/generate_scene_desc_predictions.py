import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_only import instantiate_from_config, load_checkpoint
from nanofm.data.multimodal.simple_multimodal_dataset import SimpleMultimodalDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate scene_desc predictions from a nano4M checkpoint for later LLM judging."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a training .pth checkpoint.")
    parser.add_argument("--root-dir", required=True, help="Path to the local clevr_com_304 dataset.")
    parser.add_argument("--split", default="val", help="Dataset split to use.")
    parser.add_argument("--start-idx", type=int, default=0, help="First dataset index to evaluate.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to export.")
    parser.add_argument("--device", default="cuda", help="Device to use, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--num-steps", type=int, default=8, help="ROAR decoding steps.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0 for greedy.")
    parser.add_argument("--top-k", type=float, default=0.0, help="Top-k sampling control.")
    parser.add_argument("--top-p", type=float, default=0.0, help="Top-p sampling control.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output-json", default="scene_desc_predictions.json", help="Where to save predictions.")
    return parser.parse_args()


def build_dataset(eval_loader_config: Dict, root_dir: str, split: str) -> SimpleMultimodalDataset:
    return SimpleMultimodalDataset(
        root_dir=root_dir,
        split=split,
        modalities=eval_loader_config["modalities"],
        transforms=None,
        sample_from_k_augmentations=eval_loader_config.get("sample_from_k_augmentations", 10),
        text_tokenizer_path=eval_loader_config.get("text_tokenizer_path", "gpt2"),
        text_max_length=eval_loader_config.get("text_max_length", 256),
    )


def prepare_encoder_inputs(sample: Dict[str, torch.Tensor], model, target_mod: str):
    enc_tokens: List[torch.Tensor] = []
    enc_positions: List[torch.Tensor] = []
    enc_modalities: List[torch.Tensor] = []

    for mod_idx, modality in enumerate(model.modalities):
        if modality == target_mod:
            continue
        tokens = sample[modality].flatten().long()
        positions = torch.arange(tokens.numel(), dtype=torch.long)
        modalities = torch.full((tokens.numel(),), mod_idx, dtype=torch.long)
        enc_tokens.append(tokens)
        enc_positions.append(positions)
        enc_modalities.append(modalities)

    return (
        torch.cat(enc_tokens, dim=0).unsqueeze(0),
        torch.cat(enc_positions, dim=0).unsqueeze(0),
        torch.cat(enc_modalities, dim=0).unsqueeze(0),
    )


def trim_special_tokens(tokens: torch.Tensor, tokenizer) -> torch.Tensor:
    tokens = tokens.detach().cpu()
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id

    values = tokens.tolist()
    if eos_id in values:
        values = values[: values.index(eos_id)]
    values = [tok for tok in values if tok not in {pad_id, bos_id}]
    return torch.tensor(values, dtype=torch.long)


def decode_text(tokens: torch.Tensor, tokenizer) -> str:
    if tokens.numel() == 0:
        return ""
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    state_dict, model_config, eval_loader_config, _ = load_checkpoint(args.checkpoint)
    if model_config is None or eval_loader_config is None:
        raise ValueError("This script expects a .pth training checkpoint with model_config and eval_loader_config.")

    model = instantiate_from_config(model_config)
    model.load_state_dict(state_dict)
    model = model.to(args.device).eval()

    dataset = build_dataset(eval_loader_config, root_dir=args.root_dir, split=args.split)
    tokenizer = dataset.text_tokenizer
    target_mod = "scene_desc"

    outputs = []
    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    for idx in range(args.start_idx, end_idx):
        sample = dataset[idx]
        enc_tokens, enc_positions, enc_modalities = prepare_encoder_inputs(sample, model, target_mod=target_mod)
        enc_tokens = enc_tokens.to(args.device)
        enc_positions = enc_positions.to(args.device)
        enc_modalities = enc_modalities.to(args.device)

        pred_tokens, _, _, _ = model.generate_one_modality_roar(
            enc_input_tokens=enc_tokens,
            enc_input_positions=enc_positions,
            enc_input_modalities=enc_modalities,
            target_mod=target_mod,
            num_steps=args.num_steps,
            temp=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        reference_tokens = trim_special_tokens(sample[target_mod], tokenizer)
        predicted_tokens = trim_special_tokens(pred_tokens[0], tokenizer)

        outputs.append(
            {
                "dataset_index": idx,
                "input_modalities": [m for m in model.modalities if m != target_mod],
                "reference_text": decode_text(reference_tokens, tokenizer),
                "predicted_text": decode_text(predicted_tokens, tokenizer),
                "reference_token_count": int(reference_tokens.numel()),
                "predicted_token_count": int(predicted_tokens.numel()),
            }
        )

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(outputs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
