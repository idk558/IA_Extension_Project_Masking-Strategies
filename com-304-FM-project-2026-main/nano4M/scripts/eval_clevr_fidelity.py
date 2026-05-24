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
import html
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
from PIL import Image, ImageDraw, ImageFont

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nanofm.utils.checkpoint import load_model_from_checkpoint


MODALITIES = ["tok_rgb@256", "tok_depth@256", "tok_normal@256", "scene_desc"]
DEFAULT_CONFIG = "cfgs/nano4M/multiclevr_d6-6w512.yaml"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate CLEVR caption-to-image fidelity")
    parser.add_argument("--checkpoint", required=True, help="Path to a nano4M checkpoint (.safetensors or .pth)")
    parser.add_argument("--baseline-checkpoint", default=None, help="Optional baseline checkpoint to compare")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="nano4M YAML config to read dataset settings from")
    parser.add_argument("--root-dir", default=None, help="CLEVR dataset root. Overrides the YAML config when set")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output-dir", default="outputs/clevr_fidelity", help="Directory for images and JSON report")
    parser.add_argument("--report-name", default="report.json", help="JSON report filename")
    parser.add_argument("--visual-report-name", default="visual_report.html", help="HTML report for visual inspection")
    parser.add_argument("--no-visual-report", action="store_true", help="Disable HTML visual report generation")
    parser.add_argument("--text-tokenizer-path", default="gpt2", help="Tokenizer used for scene descriptions")
    parser.add_argument("--text-max-length", type=int, default=256, help="Maximum scene description length")
    parser.add_argument("--image-tokenizer-dir", default="/tmp/nvidia/Cosmos-0.1-Tokenizer-DI16x16")
    parser.add_argument("--download-image-tokenizer", action="store_true", help="Download Cosmos tokenizer if missing")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detector-device", default=None, help="Device for Grounding DINO. Defaults to --device")
    parser.add_argument("--clip-device", default=None, help="Device for CLIP scoring. Defaults to --device")
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32", help="HuggingFace CLIP model ID")
    parser.add_argument("--clip-max-length", type=int, default=77, help="Maximum CLIP text token length")
    parser.add_argument("--no-clip-score", action="store_true", help="Disable CLIP text-image scoring")
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
    from nanofm.eval.clevr_verifier import CLIPScoreComputer, GroundingDINOVerifier

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
    clip_scorer = None
    if not args.no_clip_score:
        clip_scorer = CLIPScoreComputer(
            model_id=args.clip_model_id,
            device=args.clip_device or device,
            max_text_length=args.clip_max_length,
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
            clip_scorer=clip_scorer,
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
    if not args.no_visual_report:
        visual_report_path = output_dir / args.visual_report_name
        write_visual_report(report, visual_report_path)
        print(f"Wrote visual report to {visual_report_path}")
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
        clip_scorer: Any,
        args: argparse.Namespace,
        device: torch.device,
        output_dir: Path,
    ) -> Dict[str, Any]:
    from nanofm.eval.clevr_verifier import compute_clip_score, compute_fidelity_score

    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_from_checkpoint(checkpoint, device=device, to_eval=True)

    sample_count = min(args.num_samples, len(dataset))
    examples = []
    scores = []
    clip_scores = []
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
        clip_score = None
        if clip_scorer is not None:
            clip_score = compute_clip_score(caption, image, scorer=clip_scorer)
            clip_scores.append(clip_score["clip_score"])
        annotated_image_path = output_dir / f"{idx:06d}_detections.png"
        save_detection_overlay(image, score["detected"], annotated_image_path)
        scores.append(score["score"])
        for category, values in score["per_category_breakdown"].items():
            category_values.setdefault(category, []).append(values["match"])

        examples.append({
            "idx": idx,
            "caption": score["caption_clean"],
            "raw_caption": caption,
            "image_path": str(image_path),
            "annotated_image_path": str(annotated_image_path),
            "score": score["score"],
            "clip_score": clip_score["clip_score"] if clip_score is not None else None,
            "clip_cosine": clip_score["clip_cosine"] if clip_score is not None else None,
            "category_score": score["category_score"],
            "hallucination_penalty": score["hallucination_penalty"],
            "per_category_breakdown": score["per_category_breakdown"],
            "expected": score["expected"],
            "detected": score["detected"],
        })

        clip_text = "" if clip_score is None else f" clip={clip_score['clip_score']:.3f}"
        print(f"{label} [{idx + 1}/{sample_count}] score={score['score']:.3f}{clip_text}")

    sorted_examples = sorted(examples, key=lambda item: item["score"])
    return {
        "label": label,
        "checkpoint": checkpoint,
        "mean_score": mean(scores),
        "std_score": std(scores),
        "mean_clip_score": mean(clip_scores),
        "std_clip_score": std(clip_scores),
        "per_category_breakdown": summarize_categories(category_values),
        "examples": examples,
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


def save_detection_overlay(image: Image.Image, detections: Sequence[Dict[str, Any]], output_path: Path) -> None:
    """
    Save a copy of the generated image with detector boxes drawn on top.

    Args:
        image: Generated RGB image.
        detections: Grounding DINO detections from the fidelity scorer.
        output_path: Destination path for the annotated image.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for detection in detections:
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [float(value) for value in bbox]
        label = detection_label(detection)
        color = category_color(detection.get("shape"), detection.get("color"))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        text_bbox = draw.textbbox((x0, y0), label, font=font)
        pad = 2
        background = [
            text_bbox[0] - pad,
            text_bbox[1] - pad,
            text_bbox[2] + pad,
            text_bbox[3] + pad,
        ]
        draw.rectangle(background, fill=color)
        draw.text((x0, y0), label, fill=(255, 255, 255), font=font)

    canvas.save(output_path)


def detection_label(detection: Dict[str, Any]) -> str:
    """Return a short label for one visualized detection."""
    label = str(detection.get("label") or "").strip()
    if not label:
        shape = detection.get("shape")
        color = detection.get("color")
        label = " ".join(str(value) for value in (color, shape) if value)
    confidence = detection.get("confidence")
    if confidence is None:
        return label
    return f"{label} {float(confidence):.2f}"


def category_color(shape: Any, color: Any) -> tuple:
    """Return a stable RGB color for a detected category."""
    palette = [
        (32, 120, 255),
        (0, 170, 120),
        (230, 90, 70),
        (150, 95, 220),
        (230, 170, 40),
        (70, 170, 220),
        (90, 150, 60),
        (180, 90, 120),
    ]
    key = f"{shape}-{color}"
    return palette[sum(ord(char) for char in key) % len(palette)]


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


def write_visual_report(report: Dict[str, Any], report_path: Path) -> None:
    """
    Write an HTML page for manual inspection of captions, images, and scores.

    Args:
        report: JSON-serializable evaluation report.
        report_path: Destination HTML file.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(comparison_row(row) for row in report["comparison_table"])
    sections = "\n".join(checkpoint_section(item, report_path) for item in report["reports"])
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CLEVR Fidelity Visual Report</title>
  <style>
    body {{
      margin: 0;
      background: #f6f7f9;
      color: #20242a;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      background: #111827;
      color: #ffffff;
      padding: 24px 32px;
    }}
    main {{
      padding: 24px 32px 48px;
    }}
    h1, h2, h3 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{
      font-size: 28px;
      margin-bottom: 8px;
    }}
    h2 {{
      font-size: 22px;
      margin: 28px 0 14px;
    }}
    h3 {{
      font-size: 16px;
      margin-bottom: 8px;
    }}
    .meta {{
      color: #cbd5e1;
      font-size: 14px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: #ffffff;
      border: 1px solid #d9dee7;
    }}
    th, td {{
      border-bottom: 1px solid #e6e9ef;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      background: #edf0f5;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d9dee7;
      border-radius: 8px;
      overflow: hidden;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 12px 14px;
      background: #edf0f5;
      border-bottom: 1px solid #d9dee7;
    }}
    .score {{
      font-size: 24px;
      font-weight: 800;
    }}
    .good {{ color: #0f8b57; }}
    .mid {{ color: #b7791f; }}
    .bad {{ color: #c53030; }}
    .images {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      padding: 12px 14px;
    }}
    figure {{
      margin: 0;
    }}
    img {{
      width: 100%;
      height: auto;
      image-rendering: auto;
      border: 1px solid #e1e5ec;
      background: #ffffff;
    }}
    figcaption {{
      margin-top: 4px;
      color: #5f6b7a;
      font-size: 12px;
    }}
    .content {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      padding: 0 14px 14px;
    }}
    .caption {{
      background: #f8fafc;
      border: 1px solid #e1e5ec;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .details {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
      font-size: 13px;
      line-height: 1.5;
    }}
    .small {{
      color: #5f6b7a;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>CLEVR Fidelity Visual Report</h1>
    <div class="meta">Split: {escape(report.get("split"))} | samples: {escape(report.get("num_samples"))} | root: {escape(report.get("root_dir"))}</div>
  </header>
  <main>
    <h2>Summary</h2>
    <table>
      <thead>
        <tr><th>Label</th><th>Checkpoint</th><th>CLEVR mean</th><th>CLEVR std</th><th>CLIP mean</th><th>CLIP std</th><th>Worst</th><th>Best</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    {sections}
  </main>
</body>
</html>
"""
    with open(report_path, "w") as f:
        f.write(document)


def checkpoint_section(report: Dict[str, Any], report_path: Path) -> str:
    """Build the HTML block for one evaluated checkpoint."""
    examples = sorted(report.get("examples", []), key=lambda item: item["idx"])
    cards = "\n".join(example_card(example, report_path) for example in examples)
    return f"""
    <h2>{escape(report.get("label"))}: {format_float(report.get("mean_score"))} CLEVR | {format_float(report.get("mean_clip_score"))} CLIP</h2>
    <div class="small">{escape(report.get("checkpoint"))}</div>
    <div class="grid">{cards}</div>
"""


def example_card(example: Dict[str, Any], report_path: Path) -> str:
    """Build one visual example card."""
    score = float(example.get("score", 0.0))
    score_class = "good" if score >= 0.75 else "mid" if score >= 0.35 else "bad"
    raw_path = path_for_html(example.get("image_path"), report_path)
    annotated_path = path_for_html(example.get("annotated_image_path"), report_path)
    breakdown_rows = "\n".join(
        "<li>{}: expected {}, detected {}, match {}</li>".format(
            escape(category),
            escape(values.get("expected")),
            escape(values.get("detected")),
            format_float(values.get("match")),
        )
        for category, values in sorted(example.get("per_category_breakdown", {}).items())
    )
    expected_rows = "\n".join(
        "<li>{} {} | material: {} | position: {}</li>".format(
            escape(obj.get("color")),
            escape(obj.get("shape")),
            escape(obj.get("material")),
            escape(obj.get("position")),
        )
        for obj in example.get("expected", [])
    )
    detected_rows = "\n".join(
        "<li>{} | conf {} | box {}</li>".format(
            escape(detection.get("label")),
            format_float(detection.get("confidence")),
            escape(detection.get("bbox")),
        )
        for detection in example.get("detected", [])
    )
    if not detected_rows:
        detected_rows = "<li>No detection</li>"

    return f"""
      <section class="card">
        <div class="card-head">
          <h3>Example {escape(example.get("idx"))}</h3>
          <div>
            <div class="score {score_class}">{score:.3f}</div>
            <div class="small">CLIP {format_float(example.get("clip_score"))}</div>
          </div>
        </div>
        <div class="images">
          <figure>
            <img src="{escape(raw_path)}" alt="Generated image for example {escape(example.get("idx"))}">
            <figcaption>Generated image</figcaption>
          </figure>
          <figure>
            <img src="{escape(annotated_path)}" alt="Detected boxes for example {escape(example.get("idx"))}">
            <figcaption>Detector boxes</figcaption>
          </figure>
        </div>
        <div class="content">
          <div>
            <h3>Requested text</h3>
            <div class="caption">{escape(example.get("caption"))}</div>
          </div>
          <div class="details">
            <div>
              <h3>Expected objects</h3>
              <ul>{expected_rows}</ul>
            </div>
            <div>
              <h3>Detected objects</h3>
              <ul>{detected_rows}</ul>
            </div>
            <div>
              <h3>Score details</h3>
              <ul>
                <li>CLEVR score: {format_float(example.get("score"))}</li>
                <li>CLIP score: {format_float(example.get("clip_score"))}</li>
                <li>CLIP cosine: {format_float(example.get("clip_cosine"))}</li>
                <li>category score: {format_float(example.get("category_score"))}</li>
                <li>hallucination penalty: {format_float(example.get("hallucination_penalty"))}</li>
                {breakdown_rows}
              </ul>
            </div>
          </div>
        </div>
      </section>
"""


def comparison_row(row: Dict[str, Any]) -> str:
    """Build one comparison table row."""
    return (
        "<tr>"
        f"<td>{escape(row.get('label'))}</td>"
        f"<td>{escape(row.get('checkpoint'))}</td>"
        f"<td>{format_float(row.get('mean_score'))}</td>"
        f"<td>{format_float(row.get('std_score'))}</td>"
        f"<td>{format_float(row.get('mean_clip_score'))}</td>"
        f"<td>{format_float(row.get('std_clip_score'))}</td>"
        f"<td>{format_float(row.get('worst_score'))}</td>"
        f"<td>{format_float(row.get('best_score'))}</td>"
        "</tr>"
    )


def path_for_html(path: Any, report_path: Path) -> str:
    """Return an image path relative to the HTML report location."""
    if path is None:
        return ""
    image_path = Path(str(path))
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()
    return os.path.relpath(image_path, start=report_path.parent.resolve())


def format_float(value: Any) -> str:
    """Format a float for display in JSON and HTML reports."""
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def escape(value: Any) -> str:
    """HTML-escape a value."""
    return html.escape(str(value if value is not None else ""))


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
            "mean_clip_score": report["mean_clip_score"],
            "std_clip_score": report["std_clip_score"],
            "worst_score": report["worst_10"][0]["score"] if report["worst_10"] else None,
            "best_score": report["best_10"][0]["score"] if report["best_10"] else None,
        })
    if len(rows) == 2:
        rows.append({
            "label": "delta(candidate-baseline)",
            "checkpoint": "",
            "mean_score": rows[1]["mean_score"] - rows[0]["mean_score"],
            "std_score": None,
            "mean_clip_score": rows[1]["mean_clip_score"] - rows[0]["mean_clip_score"],
            "std_clip_score": None,
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
