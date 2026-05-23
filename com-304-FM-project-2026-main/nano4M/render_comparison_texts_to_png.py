import argparse
import json
import re
from pathlib import Path

from render_scene_desc_to_png import render_scene_desc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render reference and predicted scene descriptions from a comparison.json file."
    )
    parser.add_argument("--comparison-json", required=True, help="Path to comparison.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to a sibling folder named rendered_predictions.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def write_index(output_dir: Path, reference_text: str, predictions):
    lines = [
        "# Rendered Comparison",
        "",
        "## Reference",
        "",
        "```text",
        reference_text,
        "```",
        "",
        "![reference](reference.png)",
        "",
        "## Predictions",
        "",
    ]

    for pred in predictions:
        name = pred["model_label"]
        file_name = pred["image_file"]
        lines.extend(
            [
                f"### {name}",
                "",
                "```text",
                pred["predicted_text"],
                "```",
                "",
                f"![{name}]({file_name})",
                "",
            ]
        )

    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    comparison_path = Path(args.comparison_json)
    data = json.loads(comparison_path.read_text(encoding="utf-8"))

    output_dir = Path(args.output_dir) if args.output_dir else comparison_path.parent / "rendered_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_text = data["reference_text"]
    render_scene_desc(reference_text, output_dir / "reference.png")

    rendered_predictions = []
    for pred in data["predictions"]:
        file_name = f"{slugify(pred['model_key'])}.png"
        render_scene_desc(pred["predicted_text"], output_dir / file_name)
        rendered_predictions.append(
            {
                "model_key": pred["model_key"],
                "model_label": pred["model_label"],
                "predicted_text": pred["predicted_text"],
                "image_file": file_name,
            }
        )

    write_index(output_dir, reference_text, rendered_predictions)
    print(f"Saved rendered images to {output_dir}")
    print(f"Saved markdown gallery to {output_dir / 'index.md'}")


if __name__ == "__main__":
    main()
