#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR.parent.parent / "results"
WEIGHT_DASHBOARD_PATH = BASE_DIR / "ner_weight_impact_dashboard.png"
KEY_FIELDS_DASHBOARD_PATH = BASE_DIR / "ner_key_fields_dashboard.png"

MODEL_DIRS = [
    "Qwen3-8B-LoRA-0.6",
    "Qwen3-8B-LoRA-0.7",
    "external_qwen3-lora-08_20260630-121627",
    "Qwen3-8B-LoRA-0.9",
]
PREDICTION_CANDIDATES = ("predictions.jsonl", "prediction.jsonl")

NER_FIELDS = [
    "REACTION_PRODUCT",
    "STARTING_MATERIAL",
    "REAGENT_CATALYST",
    "SOLVENT",
    "TEMPERATURE",
    "TIME",
    "YIELD_PERCENT",
]

FIELD_SHORT = {
    "REACTION_PRODUCT": "Product",
    "STARTING_MATERIAL": "Start.Mat.",
    "REAGENT_CATALYST": "Reag./Cat.",
    "SOLVENT": "Solv.",
    "TEMPERATURE": "Temp",
    "TIME": "Time",
    "YIELD_PERCENT": "Yield",
}

KEY_FIELDS = [
    "STARTING_MATERIAL",
    "REAGENT_CATALYST",
    "SOLVENT",
]
EVAL_FIELDS = KEY_FIELDS

LOGGER = logging.getLogger("parameter_weight")
LOGGER.setLevel(logging.INFO)
LOGGER.handlers.clear()
LOGGER.addHandler(logging.StreamHandler())


def parse_ner_weight_from_dir(model_dir: str) -> float:
    lora_match = re.search(r"[Ll][Oo][Rr][Aa][-_]?(\d+(?:\.\d+)?)", model_dir)
    if lora_match:
        num_str = lora_match.group(1)
        if "." in num_str:
            return float(num_str)
        n = int(num_str)
        return n / 100.0 if n >= 100 else n / 10.0 if n <= 99 else float(n)
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)$", model_dir)
    if not match:
        return np.nan
    try:
        return float(match.group(1))
    except ValueError:
        return np.nan


def normalize_multi_value(text: Optional[str]) -> set:
    if text is None or text == "":
        return set()
    if not isinstance(text, str):
        text = str(text)
    normalized = text.replace("；", ";")
    items: List[str] = []
    for chunk in normalized.split(";"):
        cleaned = chunk.strip()
        if cleaned:
            items.append(cleaned)
    return set(items)


def normalize_parsed_record(parsed: Dict) -> Dict[str, str]:
    result = {field: "" for field in NER_FIELDS}
    if not isinstance(parsed, dict):
        return result
    for field in NER_FIELDS:
        value = parsed.get(field)
        if value is not None and value != "":
            result[field] = str(value)
    return result


FIELD_NAME_MAPPING = {
    "reaction_product": "REACTION_PRODUCT",
    "starting_material": "STARTING_MATERIAL",
    "reagent_catalyst": "REAGENT_CATALYST",
    "solvent": "SOLVENT",
    "temperature": "TEMPERATURE",
    "time": "TIME",
    "yield_percent": "YIELD_PERCENT",
}


def normalize_field_name(field_name: str) -> str:
    normalized = FIELD_NAME_MAPPING.get(field_name.lower())
    return normalized if normalized else field_name.upper()


def normalize_record(record: Dict) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, value in record.items():
        field_name = normalize_field_name(str(key))
        if field_name in NER_FIELDS:
            normalized[field_name] = "" if value is None else str(value)
    for field in NER_FIELDS:
        normalized.setdefault(field, "")
    return normalized


def _try_load_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_fenced_block(text: str, start_markers, end_marker: str):
    low = text.lower()
    start_idx = -1
    for marker in start_markers:
        idx = low.find(marker.lower())
        if idx != -1:
            start_idx = idx + len(marker)
            break
    if start_idx == -1:
        return None
    end_idx = text.find(end_marker, start_idx)
    if end_idx == -1:
        return None
    return text[start_idx:end_idx].strip()


def _extract_fenced_block_allow_eof(text: str, start_markers, end_marker: str):
    block = _extract_fenced_block(text, start_markers, end_marker)
    if block is not None:
        return block
    low = text.lower()
    for marker in start_markers:
        idx = low.find(marker.lower())
        if idx != -1:
            return text[idx + len(marker) :].strip()
    return None


def parse_json_output(output_str: str):
    if not output_str or not isinstance(output_str, str):
        return None
    text = output_str.strip()

    for start_markers, end_marker in (
        (["```json", "```JSON"], "```"),
        (["'''json", "'''JSON"], "'''"),
        (["```markdown", "```MARKDOWN", "```md", "```MD"], "```"),
        (["'''markdown", "'''MARKDOWN", "'''md", "'''MD"], "'''"),
    ):
        block = _extract_fenced_block(text, start_markers, end_marker)
        if block:
            parsed = _try_load_json(block)
            if parsed is not None:
                return parsed

    for start_markers, end_marker in ((["```"], "```"), (["'''"], "'''")):
        block = _extract_fenced_block_allow_eof(text, start_markers, end_marker)
        if block:
            parsed = _try_load_json(block)
            if parsed is not None:
                return parsed

    parsed = _try_load_json(text)
    if parsed is not None:
        return parsed

    first_obj = text.find("{")
    first_arr = text.find("[")
    if first_obj == -1 and first_arr == -1:
        return None

    start = first_obj if (first_obj != -1 and (first_arr == -1 or first_obj < first_arr)) else first_arr
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return _try_load_json(text[start : idx + 1].strip())

    return None


def normalize_parsed_candidate(candidate) -> Dict[str, str]:
    if isinstance(candidate, list):
        if candidate and isinstance(candidate[0], dict):
            candidate = candidate[0]
        else:
            candidate = {}
    return normalize_record(candidate) if isinstance(candidate, dict) else {field: "" for field in NER_FIELDS}


def is_empty_field_value(value) -> bool:
    return len(normalize_multi_value(value)) == 0


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skip invalid json at %s:%s: %s", path, line_no, exc)
    return rows


def find_prediction_path(model_dir: str) -> Path:
    base = RESULTS_DIR / model_dir
    for filename in PREDICTION_CANDIDATES:
        candidate = base / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No prediction jsonl found under {base}")


def is_ner_row(row: Dict) -> bool:
    task_type = row.get("_task_type_eval", row.get("task_type"))
    if task_type is None:
        return True
    return str(task_type).lower() == "ner"


def compute_field_level_ner_metrics(
    predictions: Sequence[Dict[str, str]],
    references: Sequence[Dict[str, str]],
    fields: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    field_metrics: Dict[str, Dict[str, float]] = {}

    for field in fields:
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for pred_dict, ref_dict in zip(predictions, references):
            pred_items = normalize_multi_value(pred_dict.get(field, ""))
            ref_items = normalize_multi_value(ref_dict.get(field, ""))

            true_positive += len(pred_items & ref_items)
            false_positive += len(pred_items - ref_items)
            false_negative += len(ref_items - pred_items)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        field_metrics[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": true_positive,
            "FP": false_positive,
            "FN": false_negative,
        }

    field_metrics["AVERAGE"] = {
        "precision": sum(metric["precision"] for metric in field_metrics.values()) / len(fields),
        "recall": sum(metric["recall"] for metric in field_metrics.values()) / len(fields),
        "f1": sum(metric["f1"] for metric in field_metrics.values()) / len(fields),
    }
    return field_metrics


def load_data() -> List[Dict]:
    rows: List[Dict] = []

    for model_dir in MODEL_DIRS:
        prediction_path = find_prediction_path(model_dir)
        prediction_rows = load_jsonl(prediction_path)
        ner_rows = [row for row in prediction_rows if is_ner_row(row)]

        matched_predictions: List[Dict[str, str]] = []
        matched_references: List[Dict[str, str]] = []

        has_preparsed = any(isinstance(row.get("_ref_parsed"), dict) for row in ner_rows[:5])

        for row in ner_rows:
            if has_preparsed:
                ref_parsed = normalize_parsed_record(row.get("_ref_parsed", {}))
                pred_parsed = normalize_parsed_record(row.get("_pred_parsed", {}))
            else:
                output = row.get("output", "")
                ref_raw = parse_json_output(output)
                ref_parsed = normalize_record(ref_raw) if ref_raw else {field: "" for field in NER_FIELDS}

                pred_raw = parse_json_output(row.get("_prediction", ""))
                pred_parsed = normalize_record(pred_raw) if pred_raw else {field: "" for field in NER_FIELDS}

                if not isinstance(ref_raw, dict):
                    continue

            matched_predictions.append(pred_parsed)
            matched_references.append(ref_parsed)

        if not matched_predictions:
            LOGGER.warning("No valid samples for %s", model_dir)
            continue

        metrics = compute_field_level_ner_metrics(matched_predictions, matched_references, EVAL_FIELDS)
        avg = metrics["AVERAGE"]

        LOGGER.info(
            "%s: matched=%s, avg_f1=%.4f",
            model_dir,
            len(matched_predictions),
            avg["f1"],
        )

        rows.append(
            {
                "model_name": model_dir,
                "ner_weight": parse_ner_weight_from_dir(model_dir),
                "matched_samples": len(matched_predictions),
                "avg_precision": avg["precision"],
                "avg_recall": avg["recall"],
                "avg_f1": avg["f1"],
                "field_f1": {field: metrics[field]["f1"] for field in EVAL_FIELDS},
                "field_prf": {
                    field: {
                        "precision": metrics[field]["precision"],
                        "recall": metrics[field]["recall"],
                        "f1": metrics[field]["f1"],
                    }
                    for field in EVAL_FIELDS
                },
            }
        )

    rows.sort(key=lambda x: x["ner_weight"])
    return rows


def plot_weight_impact_dashboard(data: List[Dict]) -> None:
    x_weight = [d["ner_weight"] for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=False)
    metric_styles = {
        "precision": {"color": "#2ca02c", "marker": "s", "linestyle": "--", "label": "Precision"},
        "recall": {"color": "#ff7f0e", "marker": "^", "linestyle": "-.", "label": "Recall"},
        "f1": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "label": "F1"},
    }

    for idx, field in enumerate(KEY_FIELDS):
        ax = axes[idx]
        all_y = []
        for metric_name, style in metric_styles.items():
            y = [d["field_prf"].get(field, {}).get(metric_name, np.nan) * 100 for d in data]
            valid_y = [v for v in y if not np.isnan(v)]
            all_y.extend(valid_y)
            ax.plot(
                x_weight,
                y,
                marker=style["marker"],
                linewidth=2.0,
                linestyle=style["linestyle"],
                color=style["color"],
                label=style["label"],
            )

        ax.set_title(f"{FIELD_SHORT[field]} P/R/F1")
        ax.set_xlabel("NER Weight")
        ax.set_ylabel("Score (%)")
        ax.set_xticks(x_weight)
        ax.set_xticklabels([f"{w:.1f}" for w in x_weight])

        if all_y:
            y_min = min(all_y)
            y_max = max(all_y)
            margin = max(0.3, (y_max - y_min) * 0.2)
            ax.set_ylim(y_min - margin, y_max + margin)

        ax.grid(True, linestyle=":", alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.08), ncol=3, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    fig.savefig(WEIGHT_DASHBOARD_PATH, dpi=300, bbox_inches="tight")
    LOGGER.info("Saved figure: %s", WEIGHT_DASHBOARD_PATH)
    fig.savefig(KEY_FIELDS_DASHBOARD_PATH, dpi=300, bbox_inches="tight")
    LOGGER.info("Saved figure: %s", KEY_FIELDS_DASHBOARD_PATH)


def print_summary(data: List[Dict]) -> None:
    print("\n" + "=" * 100)
    print("NER WEIGHT SUMMARY (MATCHED BY INPUT)")
    print("=" * 100)
    print(f"{'Model':<28} {'NW':>6} {'Matched':>8} {'P':>10} {'R':>10} {'F1':>10}")
    print("-" * 100)
    for d in data:
        print(
            f"{d['model_name']:<28} {d['ner_weight']:>6.1f} {d['matched_samples']:>8d} "
            f"{d['avg_precision'] * 100:>9.2f}% {d['avg_recall'] * 100:>9.2f}% {d['avg_f1'] * 100:>9.2f}%"
        )
    print("=" * 100)


def print_key_field_metrics(data: List[Dict]) -> None:
    print("\n" + "=" * 100)
    print("KEY FIELD METRICS USED FOR DASHBOARD")
    print("=" * 100)

    for field in KEY_FIELDS:
        print(f"\n[{FIELD_SHORT.get(field, field)}]")
        print(f"{'Model':<28} {'NW':>6} {'Matched':>8} {'P':>10} {'R':>10} {'F1':>10}")
        print("-" * 76)
        for d in data:
            metrics = d["field_prf"].get(field, {})
            print(
                f"{d['model_name']:<28} {d['ner_weight']:>6.1f} {d['matched_samples']:>8d} "
                f"{metrics.get('precision', 0.0) * 100:>9.2f}% "
                f"{metrics.get('recall', 0.0) * 100:>9.2f}% "
                f"{metrics.get('f1', 0.0) * 100:>9.2f}%"
            )

    print("\n" + "=" * 100)


def main() -> None:
    data = load_data()
    if not data:
        raise RuntimeError("No valid model data found.")

    print_summary(data)
    print_key_field_metrics(data)
    plot_weight_impact_dashboard(data)
    plt.show()


if __name__ == "__main__":
    main()
