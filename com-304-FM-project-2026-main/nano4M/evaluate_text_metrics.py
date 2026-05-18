import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated text against reference text using captioning/text metrics "
            "such as BLEU, ROUGE-L, METEOR, CIDEr-D, SPICE, and SPIDEr."
        )
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="JSON file containing objects with predicted_text and reference_text fields.",
    )
    parser.add_argument(
        "--output-json",
        default="text_metric_scores.json",
        help="Path where metric scores will be saved.",
    )
    parser.add_argument(
        "--candidate-key",
        default="predicted_text",
        help="Field name containing the generated text.",
    )
    parser.add_argument(
        "--reference-key",
        default="reference_text",
        help="Field name containing the reference text or list of references.",
    )
    parser.add_argument(
        "--metrics",
        default="default",
        help=(
            "Metric set passed to aac-metrics. Use default for BLEU, ROUGE-L, "
            "METEOR, CIDEr-D, SPICE, and SPIDEr. Example: dcase2024."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for quick tests.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=10,
        help="Number of scored examples to print. Use -1 to print all examples.",
    )
    return parser.parse_args()


def to_float(value: Any):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def to_float_list(value: Any) -> List[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in value]


def normalize_references(reference_value: Any) -> List[str]:
    if isinstance(reference_value, str):
        return [reference_value]
    if isinstance(reference_value, Sequence):
        return [str(reference) for reference in reference_value]
    return [str(reference_value)]


def load_text_pairs(
    input_json: Path,
    candidate_key: str,
    reference_key: str,
    max_samples: Optional[int],
):
    records = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a list of prediction records.")
    if max_samples is not None:
        records = records[:max_samples]

    candidates = []
    mult_references = []
    kept_records = []
    for idx, record in enumerate(records):
        if candidate_key not in record:
            raise KeyError(f"Missing candidate key '{candidate_key}' in record {idx}.")
        if reference_key not in record:
            raise KeyError(f"Missing reference key '{reference_key}' in record {idx}.")

        candidate = str(record[candidate_key]).strip()
        references = [ref.strip() for ref in normalize_references(record[reference_key]) if ref.strip()]
        if not candidate or not references:
            continue

        candidates.append(candidate)
        mult_references.append(references)
        kept_records.append(record)

    if not candidates:
        raise ValueError("No valid candidate/reference text pairs found.")
    return kept_records, candidates, mult_references


def evaluate_text(candidates: List[str], mult_references: List[List[str]], metrics: str):
    try:
        from aac_metrics import evaluate
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'aac-metrics'. Install it with:\n"
            "  pip install aac-metrics\n"
            "  aac-metrics-download\n"
            "SPICE and METEOR also require Java."
        ) from exc

    if metrics == "default":
        return evaluate(candidates, mult_references)
    return evaluate(candidates, mult_references, metrics=metrics)


def build_output(
    kept_records: List[Dict[str, Any]],
    candidates: List[str],
    mult_references: List[List[str]],
    corpus_scores: Dict[str, Any],
    sentence_scores: Dict[str, Any],
):
    sentence_score_lists = {
        metric_name: to_float_list(values)
        for metric_name, values in sentence_scores.items()
    }

    per_sample = []
    for idx, (record, candidate, references) in enumerate(zip(kept_records, candidates, mult_references)):
        per_sample.append(
            {
                "dataset_index": record.get("dataset_index", idx),
                "input_text": references[0],
                "output_text": candidate,
                "predicted_text": candidate,
                "reference_text": references,
                "scores": {
                    metric_name: values[idx]
                    for metric_name, values in sentence_score_lists.items()
                    if idx < len(values)
                },
            }
        )

    return {
        "num_samples": len(candidates),
        "corpus_scores": {
            metric_name: to_float(score)
            for metric_name, score in corpus_scores.items()
        },
        "samples": per_sample,
    }


def print_sample_scores(samples: List[Dict[str, Any]], show_samples: int):
    if show_samples == 0:
        return

    limit = len(samples) if show_samples < 0 else min(show_samples, len(samples))
    print(f"\nShowing {limit} scored text examples")
    for idx, sample in enumerate(samples[:limit], start=1):
        print(f"\nExample {idx} | dataset_index={sample['dataset_index']}")
        print(f"Input/reference : {sample['input_text']}")
        print(f"Output/nano4M   : {sample['output_text']}")
        if sample["scores"]:
            score_text = ", ".join(
                f"{metric_name}={score:.4f}"
                for metric_name, score in sorted(sample["scores"].items())
            )
            print(f"Scores          : {score_text}")


def main():
    args = parse_args()
    input_json = Path(args.input_json)

    kept_records, candidates, mult_references = load_text_pairs(
        input_json=input_json,
        candidate_key=args.candidate_key,
        reference_key=args.reference_key,
        max_samples=args.max_samples,
    )
    corpus_scores, sentence_scores = evaluate_text(candidates, mult_references, args.metrics)
    output = build_output(kept_records, candidates, mult_references, corpus_scores, sentence_scores)

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Evaluated {output['num_samples']} text pairs")
    for metric_name, score in output["corpus_scores"].items():
        print(f"{metric_name}: {score:.4f}")
    print_sample_scores(output["samples"], args.show_samples)
    print(f"Saved scores to {output_path}")


if __name__ == "__main__":
    main()
