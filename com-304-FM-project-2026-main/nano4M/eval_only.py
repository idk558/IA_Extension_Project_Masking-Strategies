import argparse
import json
import random
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nanofm.utils.logger import MetricLogger


def import_from_target(target: str):
    module_name, attr_name = target.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def instantiate_from_config(config: Dict[str, Any]):
    config = dict(config)
    target = config.pop("_target_")
    return import_from_target(target)(**config)


def load_checkpoint(checkpoint_path: str):
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path.endswith(".pth"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"]
        args = checkpoint.get("args")
        if args is None:
            raise ValueError("The .pth checkpoint does not contain the original training args.")
        model_config = getattr(args, "model_config", None)
        eval_loader_config = getattr(args, "eval_loader_config", None)
        dtype_name = getattr(args, "dtype", "fp32")
        return state_dict, model_config, eval_loader_config, dtype_name

    if checkpoint_path.endswith(".safetensors"):
        from nanofm.utils.checkpoint import load_safetensors

        state_dict, metadata = load_safetensors(checkpoint_path)
        return state_dict, metadata, None, "fp32"

    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def normalize_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = dtype_name.lower()
    if dtype_name in {"float16", "fp16"}:
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def to_device(data_dict: Dict[str, Any], device: torch.device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in data_dict.items()
    }


@torch.no_grad()
def evaluate(model, data_loader, device: torch.device, dtype: torch.dtype, max_batches: int | None = None):
    model_state = model.training
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    iter_len = len(data_loader) if hasattr(data_loader, "__len__") else -1
    if max_batches is not None and iter_len > 0:
        iter_len = min(iter_len, max_batches)

    for step, data_dict in enumerate(metric_logger.log_every(data_loader, 10, iter_len=iter_len, header="[Eval]")):
        if max_batches is not None and step >= max_batches:
            break

        data_dict = to_device(data_dict, device)
        use_autocast = device.type == "cuda" and dtype != torch.float32
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
            loss, metrics = model(data_dict)

        metric_logger.update(loss=loss.item())
        metric_logger.update(**{name: value.item() for name, value in metrics.items()})

    metric_logger.synchronize_between_processes()
    model.train(model_state)

    return {name: meter.global_avg for name, meter in metric_logger.meters.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a nano4M checkpoint on the validation split.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth checkpoint from training.")
    parser.add_argument("--root-dir", default=None, help="Override dataset root_dir from the checkpoint config.")
    parser.add_argument("--split", default=None, help="Override eval split, e.g. val or test.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count.")
    parser.add_argument("--device", default="cuda", help="Device to use, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--dtype", default=None, help="Override autocast dtype: fp32, fp16, or bf16.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic masking/data order.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap for a quick smoke test.")
    parser.add_argument("--output-json", default=None, help="Optional path to save metrics as JSON.")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    state_dict, model_config, eval_loader_config, ckpt_dtype_name = load_checkpoint(checkpoint_path)

    if model_config is None or eval_loader_config is None:
        raise ValueError(
            "This checkpoint does not include both model_config and eval_loader_config. "
            "Use a training .pth checkpoint for the first evaluation pass."
        )

    eval_loader_config = dict(eval_loader_config)
    eval_loader_config["distributed"] = False
    eval_loader_config["shuffle"] = False
    eval_loader_config["drop_last"] = False
    eval_loader_config["infinite"] = False

    if args.root_dir is not None:
        eval_loader_config["root_dir"] = args.root_dir
    if args.split is not None:
        eval_loader_config["split"] = args.split
    if args.batch_size is not None:
        eval_loader_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        eval_loader_config["num_workers"] = args.num_workers

    dataset_root = Path(eval_loader_config["root_dir"])
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}\n"
            "Pass --root-dir to a local clevr_com_304 copy or run this from the course environment."
        )

    model = instantiate_from_config(model_config)
    model.load_state_dict(state_dict)

    device = torch.device(args.device)
    model = model.to(device)

    dtype_name = args.dtype or ckpt_dtype_name
    dtype = normalize_dtype(dtype_name)

    data_loader = instantiate_from_config(eval_loader_config)
    metrics = evaluate(model, data_loader, device=device, dtype=dtype, max_batches=args.max_batches)

    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset_root": str(dataset_root),
        "split": eval_loader_config["split"],
        "batch_size": eval_loader_config["batch_size"],
        "dtype": dtype_name,
        "metrics": metrics,
    }

    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
