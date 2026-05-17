import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_only import evaluate, instantiate_from_config, load_checkpoint, normalize_dtype


def checkpoint_sort_key(path: Path):
    name = path.stem
    if name == "checkpoint-final":
        return (1, float("inf"))
    match = re.search(r"checkpoint-(\d+)$", name)
    if match:
        return (0, int(match.group(1)))
    return (2, name)


def infer_step(path: Path):
    match = re.search(r"checkpoint-(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple nano4M checkpoints and plot validation curves.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing checkpoint-*.pth files.")
    parser.add_argument("--root-dir", required=True, help="Path to the local clevr_com_304 dataset.")
    parser.add_argument("--device", default="cuda", help="Device to use, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--dtype", default=None, help="Override autocast dtype: fp32, fp16, or bf16.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap for a quick smoke test.")
    parser.add_argument("--pattern", default="checkpoint-*.pth", help="Glob for checkpoints to evaluate.")
    parser.add_argument("--output-json", default="eval_checkpoints_results.json", help="Path to save metrics as JSON.")
    parser.add_argument("--output-png", default="eval_checkpoints_curve.png", help="Path to save the validation curve PNG.")
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob(args.pattern), key=checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir} with pattern {args.pattern}")

    results = []
    for checkpoint_path in checkpoints:
        print(f"\nEvaluating {checkpoint_path.name}")
        state_dict, model_config, eval_loader_config, ckpt_dtype_name = load_checkpoint(str(checkpoint_path))
        if model_config is None or eval_loader_config is None:
            print(f"Skipping {checkpoint_path.name}: missing model/eval config in checkpoint")
            continue

        eval_loader_config = dict(eval_loader_config)
        eval_loader_config["root_dir"] = args.root_dir
        eval_loader_config["distributed"] = False
        eval_loader_config["shuffle"] = False
        eval_loader_config["drop_last"] = False
        eval_loader_config["infinite"] = False
        if args.batch_size is not None:
            eval_loader_config["batch_size"] = args.batch_size
        if args.num_workers is not None:
            eval_loader_config["num_workers"] = args.num_workers

        model = instantiate_from_config(model_config)
        model.load_state_dict(state_dict)
        model = model.to(args.device)

        dtype = normalize_dtype(args.dtype or ckpt_dtype_name)
        data_loader = instantiate_from_config(eval_loader_config)
        metrics = evaluate(model, data_loader, device=model.device, dtype=dtype, max_batches=args.max_batches)

        results.append(
            {
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "step": infer_step(checkpoint_path),
                "is_final": checkpoint_path.stem == "checkpoint-final",
                "metrics": metrics,
            }
        )

    output_json = Path(args.output_json)
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    curve_points = [r for r in results if r["step"] is not None]
    if curve_points:
        steps = [r["step"] for r in curve_points]
        losses = [r["metrics"]["loss"] for r in curve_points]
        plt.figure(figsize=(8, 5))
        plt.plot(steps, losses, marker="o", linewidth=2)
        plt.xlabel("Checkpoint step")
        plt.ylabel("Validation loss")
        plt.title("nano4M baseline validation curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_png = Path(args.output_png)
        plt.savefig(output_png, dpi=180)
        print(f"Saved curve to {output_png}")

    print(f"Saved results to {output_json}")


if __name__ == "__main__":
    main()
