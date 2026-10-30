#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline evaluation for MAReX extraction outputs."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

try:
    from rouge import Rouge
except ImportError:
    Rouge = None


FIELDS = [
    "REACTION_PRODUCT",
    "STARTING_MATERIAL",
    "REAGENT_CATALYST",
    "SOLVENT",
    "TEMPERATURE",
    "TIME",
    "YIELD_PERCENT",
]
ACTION_FIELD = "action_sequence"


def normalize_multi_value(value: Optional[str]) -> set:
    if value is None or value == "":
        return set()
    value = str(value).replace("；", ";").replace("，", ",")
    items = []
    for chunk in value.split(";"):
        items.extend(chunk.split(","))
    return {item.strip() for item in items if item.strip()}


def load_json_or_jsonl(path: Path) -> List[Dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def micro_f1(pred_data: List[Dict], ref_data: List[Dict], fields: List[str]):
    true_positive = false_positive = false_negative = 0
    for pred, ref in zip(pred_data, ref_data):
        for field in fields:
            pred_items = normalize_multi_value(pred.get(field))
            ref_items = normalize_multi_value(ref.get(field))
            true_positive += len(pred_items & ref_items)
            false_positive += len(pred_items - ref_items)
            false_negative += len(ref_items - pred_items)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def per_field_f1(pred_data: List[Dict], ref_data: List[Dict], fields: List[str]) -> Dict[str, Dict[str, float]]:
    results = {}
    for field in fields:
        true_positive = false_positive = false_negative = 0
        for pred, ref in zip(pred_data, ref_data):
            pred_items = normalize_multi_value(pred.get(field))
            ref_items = normalize_multi_value(ref.get(field))
            true_positive += len(pred_items & ref_items)
            false_positive += len(pred_items - ref_items)
            false_negative += len(ref_items - pred_items)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results[field] = {"precision": precision, "recall": recall, "f1": f1}
    return results


def normalized_edit_distance(pred: str, ref: str) -> float:
    if not pred and not ref:
        return 0.0
    if Levenshtein is not None:
        return Levenshtein.distance(pred, ref) / max(len(pred), len(ref))

    previous = list(range(len(ref) + 1))
    for i, pred_char in enumerate(pred, 1):
        current = [i]
        for j, ref_char in enumerate(ref, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(pred_char != ref_char),
                )
            )
        previous = current
    return previous[-1] / max(len(pred), len(ref))


def sequence_metrics(pred_data: List[Dict], ref_data: List[Dict], field: str) -> Dict[str, float]:
    bleu_scores, rouge_scores, edit_distances = [], [], []
    rouge = Rouge() if Rouge is not None else None
    smoother = SmoothingFunction().method1

    for pred, ref in zip(pred_data, ref_data):
        pred_tokens = [
            token.strip()
            for token in str(pred.get(field) or "").replace("；", ";").split(";")
            if token.strip()
        ]
        ref_tokens = [
            token.strip()
            for token in str(ref.get(field) or "").replace("；", ";").split(";")
            if token.strip()
        ]
        if not ref_tokens:
            continue

        bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother))
        pred_text = " ".join(pred_tokens)
        ref_text = " ".join(ref_tokens)
        if rouge is not None:
            try:
                rouge_scores.append(rouge.get_scores(pred_text, ref_text)[0]["rouge-l"]["f"])
            except Exception:
                rouge_scores.append(0.0)
        edit_distances.append(normalized_edit_distance(pred_text, ref_text))

    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "bleu": mean(bleu_scores),
        "rouge_l": mean(rouge_scores),
        "edit_distance": mean(edit_distances),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MAReX predictions against references.")
    parser.add_argument("--pred", type=Path, default=Path("predictions.json"), help="Prediction JSON/JSONL file.")
    parser.add_argument("--ref", type=Path, default=Path("references.json"), help="Reference JSON/JSONL file.")
    parser.add_argument("--action-field", default=ACTION_FIELD, help="Field name for action-sequence evaluation.")
    return parser.parse_args()


def main():
    args = parse_args()
    pred_data = load_json_or_jsonl(args.pred)
    ref_data = load_json_or_jsonl(args.ref)

    if len(pred_data) != len(ref_data):
        size = min(len(pred_data), len(ref_data))
        print(f"Warning: length mismatch; truncating to {size} examples.")
        pred_data, ref_data = pred_data[:size], ref_data[:size]

    precision, recall, f1 = micro_f1(pred_data, ref_data, FIELDS)
    print("Micro-average field extraction")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    print("\nPer-field extraction")
    for field, scores in per_field_f1(pred_data, ref_data, FIELDS).items():
        print(
            f"  {field}: P={scores['precision']:.4f}, "
            f"R={scores['recall']:.4f}, F1={scores['f1']:.4f}"
        )

    print("\nAction sequence")
    for name, value in sequence_metrics(pred_data, ref_data, args.action_field).items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
