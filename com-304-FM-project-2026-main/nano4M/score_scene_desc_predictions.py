import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score scene_desc predictions with automatic metrics and aggregate score averages."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="JSON file containing reference_text and predicted_text, optionally with judge scores.",
    )
    parser.add_argument(
        "--output-json",
        default="scene_desc_scored.json",
        help="Where to save the per-example metrics and summary.",
    )
    return parser.parse_args()


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def make_ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def clipped_precision(reference: Sequence[str], prediction: Sequence[str], n: int) -> float:
    pred_ngrams = Counter(make_ngrams(prediction, n))
    if not pred_ngrams:
        return 0.0
    ref_ngrams = Counter(make_ngrams(reference, n))
    overlap = 0
    for ngram, count in pred_ngrams.items():
        overlap += min(count, ref_ngrams.get(ngram, 0))
    return overlap / max(sum(pred_ngrams.values()), 1)


def bleu_scores(reference: Sequence[str], prediction: Sequence[str]) -> Dict[str, float]:
    precisions = [clipped_precision(reference, prediction, n) for n in range(1, 5)]
    ref_len = len(reference)
    pred_len = len(prediction)
    if pred_len == 0:
        bp = 0.0
    elif pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_len / pred_len) if pred_len > 0 else 0.0

    scores = {}
    for k in range(1, 5):
        clipped = precisions[:k]
        if min(clipped) <= 0:
            scores[f"bleu_{k}"] = 0.0
            continue
        geo_mean = math.exp(sum(math.log(p) for p in clipped) / k)
        scores[f"bleu_{k}"] = 100.0 * bp * geo_mean
    return scores


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, start=1):
            cur = dp[j]
            if token_a == token_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(reference: Sequence[str], prediction: Sequence[str]) -> float:
    if not reference or not prediction:
        return 0.0
    lcs = lcs_length(reference, prediction)
    precision = lcs / len(prediction)
    recall = lcs / len(reference)
    if precision + recall == 0:
        return 0.0
    beta = 1.2
    score = ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)
    return 100.0 * score


def build_document_frequency(references: Iterable[Sequence[str]]) -> Dict[int, Counter]:
    document_frequency = {n: Counter() for n in range(1, 5)}
    for ref_tokens in references:
        for n in range(1, 5):
            unique_ngrams = set(make_ngrams(ref_tokens, n))
            for ngram in unique_ngrams:
                document_frequency[n][ngram] += 1
    return document_frequency


def tfidf_vector(tokens: Sequence[str], n: int, document_frequency: Dict[int, Counter], num_docs: int) -> Dict[Tuple[str, ...], float]:
    counts = Counter(make_ngrams(tokens, n))
    if not counts:
        return {}
    total = sum(counts.values())
    vector = {}
    for ngram, count in counts.items():
        df = document_frequency[n].get(ngram, 0)
        idf = math.log((num_docs + 1.0) / (df + 1.0))
        vector[ngram] = (count / total) * idf
    return vector


def cosine_similarity(vec_a: Dict[Tuple[str, ...], float], vec_b: Dict[Tuple[str, ...], float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(value * vec_b.get(key, 0.0) for key, value in vec_a.items())
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cider_score(reference: Sequence[str], prediction: Sequence[str], document_frequency: Dict[int, Counter], num_docs: int) -> float:
    """
    Lightweight CIDEr-style score.
    This follows the TF-IDF n-gram spirit of CIDEr to support relative model comparison
    without adding the external COCO evaluation dependency.
    """
    sims = []
    for n in range(1, 5):
        ref_vec = tfidf_vector(reference, n, document_frequency, num_docs)
        pred_vec = tfidf_vector(prediction, n, document_frequency, num_docs)
        sims.append(cosine_similarity(ref_vec, pred_vec))
    return 100.0 * (sum(sims) / len(sims))


def exact_match(reference_text: str, predicted_text: str) -> float:
    return 100.0 if reference_text.strip() == predicted_text.strip() else 0.0


def collect_numeric_fields(items: Sequence[Dict]) -> List[str]:
    numeric_fields = set()
    for item in items:
        for key, value in item.items():
            if isinstance(value, (int, float)):
                numeric_fields.add(key)
    return sorted(numeric_fields)


def main():
    args = parse_args()

    input_path = Path(args.input_json)
    items = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Expected a JSON array as input.")

    reference_tokens_all = [tokenize(item["reference_text"]) for item in items]
    document_frequency = build_document_frequency(reference_tokens_all)
    num_docs = len(reference_tokens_all)

    scored_items = []
    for item in items:
        reference_text = item["reference_text"]
        predicted_text = item["predicted_text"]
        reference_tokens = tokenize(reference_text)
        predicted_tokens = tokenize(predicted_text)

        item_scored = dict(item)
        item_scored.update(bleu_scores(reference_tokens, predicted_tokens))
        item_scored["rouge_l"] = rouge_l(reference_tokens, predicted_tokens)
        item_scored["cider"] = cider_score(reference_tokens, predicted_tokens, document_frequency, num_docs)
        item_scored["exact_match"] = exact_match(reference_text, predicted_text)
        scored_items.append(item_scored)

    numeric_fields = collect_numeric_fields(scored_items)
    averages = {}
    for field in numeric_fields:
        values = [float(item[field]) for item in scored_items]
        averages[f"{field}_avg"] = sum(values) / len(values)

    summary = {
        "num_examples": len(scored_items),
        "averages": averages,
    }

    output = {
        "items": scored_items,
        "summary": summary,
    }

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved scored results to {output_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
