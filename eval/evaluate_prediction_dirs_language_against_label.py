#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate NER predictions against a specified label file and also split metrics by language.

Usage:
1. Edit CONFIG below.
2. Run: python evaluate_prediction_dirs_language_against_label.py
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PREDICTION_CANDIDATES = ("predictions.jsonl", "prediction.jsonl")

ALL_FIELDS = [
    "REACTION_PRODUCT",
    "STARTING_MATERIAL",
    "REAGENT_CATALYST",
    "SOLVENT",
    "TEMPERATURE",
    "TIME",
    "YIELD_PERCENT",
]

FIELD_ALIASES = {
    "reaction_product": "REACTION_PRODUCT",
    "starting_material": "STARTING_MATERIAL",
    "reagent_catalyst": "REAGENT_CATALYST",
    "solvent": "SOLVENT",
    "temperature": "TEMPERATURE",
    "time": "TIME",
    "yield_percent": "YIELD_PERCENT",
}

_results = Path(__file__).resolve().parent.parent / "results"
_models = Path(__file__).resolve().parent.parent / "models"
CONFIG = {
    "dirs": [
        str(_results / "external_deepseek-v3_20260201-155435"),
        str(_results / "external_qwen-plus_20260201-145935"),
        str(_results / "base_chemdfm-v1.5-8b_20260128-154402"),
        str(_results / "base_ChemLLM-7B-SFT_20260130-155158"),
        str(_results / "external_gpt-4o_20260131-205535"),
    ],
    "label_path": str(_models / "Qwen3-8B-LoRA-0.8" / "predictions_lable_modify.json"),
    "output_name": "eval_metrics_language_from_label.json",
    "fields": ALL_FIELDS,
}

NULL_LIKE = {"", "null", "none", "n/a", "notspecified"}
WS_RE = re.compile(r"\s+")
YIELD_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%?\s*$")


def normalize_field_name(field_name: str) -> str:
    mapped = FIELD_ALIASES.get(field_name.strip().lower())
    return mapped if mapped else field_name.strip().upper()


def resolve_fields(fields: Sequence[str]) -> List[str]:
    resolved: List[str] = []
    for field in fields:
        normalized = normalize_field_name(field)
        if normalized not in ALL_FIELDS:
            raise ValueError(f"Unsupported field: {field}")
        if normalized not in resolved:
            resolved.append(normalized)
    if not resolved:
        raise ValueError("No valid fields selected.")
    return resolved


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skip invalid json at {path}:{line_no}: {exc}")
    return rows


def is_ner_sample(sample: Dict[str, Any]) -> bool:
    task_type = sample.get("task_type")
    if isinstance(task_type, int):
        return task_type == 0
    if isinstance(task_type, str):
        return task_type.strip().lower() == "ner"
    instruction = str(sample.get("instruction", "")).lower()
    return "reaction_product" in instruction and "yield_percent" in instruction


def find_prediction_path(exp_dir: Path) -> Path:
    for filename in PREDICTION_CANDIDATES:
        candidate = exp_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No prediction jsonl found under {exp_dir}")


def strip_all_whitespace(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return WS_RE.sub("", value)


def normalize_yield_value(value: Any) -> str:
    text = strip_all_whitespace(value)
    if not text:
        return ""
    match = YIELD_NUM_RE.match(text)
    if match:
        return match.group(1)
    fallback = re.search(r"(\d+(?:\.\d+)?)", text)
    if fallback:
        return fallback.group(1)
    return text


def extract_value_items(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("{"):
            try:
                return extract_value_items(json.loads(text))
            except Exception:
                pass
        return [value]
    if isinstance(value, dict):
        return [str(value.get("name", ""))]
    if isinstance(value, list):
        items: List[str] = []
        for item in value:
            items.extend(extract_value_items(item))
        return items
    return [str(value)]


def normalize_multi_value(value: Any, field: Optional[str] = None) -> set:
    if field == "YIELD_PERCENT":
        normalized = normalize_yield_value(value)
        return set() if not normalized else {normalized}

    raw_items = extract_value_items(value)
    pieces: List[str] = []
    for item in raw_items:
        text = item.replace("；", ";")
        for chunk in text.split(";"):
            pieces.extend(chunk.split(","))

    normalized = set()
    for piece in pieces:
        cleaned = strip_all_whitespace(piece)
        if not cleaned:
            continue
        if cleaned.lower() in NULL_LIKE:
            continue
        normalized.add(cleaned)
    return normalized


def _try_load_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_fenced_block(text: str, start_markers: Iterable[str], end_marker: str) -> Optional[str]:
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


def _extract_fenced_block_allow_eof(text: str, start_markers: Iterable[str], end_marker: str) -> Optional[str]:
    block = _extract_fenced_block(text, start_markers, end_marker)
    if block is not None:
        return block
    low = text.lower()
    for marker in start_markers:
        idx = low.find(marker.lower())
        if idx != -1:
            return text[idx + len(marker) :].strip()
    return None


def parse_json_output(output_str: str) -> Any:
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


def normalize_parsed_candidate(candidate: Any) -> Dict[str, Any]:
    if isinstance(candidate, list):
        if candidate and isinstance(candidate[0], dict):
            candidate = candidate[0]
        else:
            candidate = {}
    return candidate if isinstance(candidate, dict) else {}


def is_empty_field_value(field: str, value: Any) -> bool:
    return len(normalize_multi_value(value, field=field)) == 0


def parse_prediction_record(row: Dict[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    result = {field: "" for field in fields}
    candidates = [
        normalize_parsed_candidate(row.get("_pred_parsed")),
        normalize_parsed_candidate(row.get("pred_parsed")),
        normalize_parsed_candidate(parse_json_output(row.get("_prediction", ""))),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        for key, value in candidate.items():
            normalized_key = normalize_field_name(str(key))
            if normalized_key not in result:
                continue
            if not is_empty_field_value(normalized_key, result[normalized_key]):
                continue
            if is_empty_field_value(normalized_key, value):
                continue
            result[normalized_key] = value
    return result


def parse_label_record(row: Dict[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    result = {field: "" for field in fields}
    for key, value in row.items():
        normalized_key = normalize_field_name(str(key))
        if normalized_key in result:
            result[normalized_key] = value
    return result


def align_rows_by_input(
    labels: Sequence[Dict[str, Any]],
    preds: Sequence[Dict[str, Any]],
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], Dict[str, int]]:
    label_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pred_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in labels:
        label_groups[str(row.get("input", ""))].append(row)
    for row in preds:
        pred_groups[str(row.get("input", ""))].append(row)

    shared_inputs = sorted(set(label_groups) & set(pred_groups))
    missing_inputs = sorted(set(label_groups) - set(pred_groups))
    extra_inputs = sorted(set(pred_groups) - set(label_groups))

    aligned: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for input_text in shared_inputs:
        label_rows = sorted(label_groups[input_text], key=lambda row: str(row.get("id", "")))
        pred_rows = sorted(pred_groups[input_text], key=lambda row: str(row.get("id", "")))
        if len(label_rows) != len(pred_rows):
            raise ValueError(
                f"Input occurrence count mismatch for input={input_text[:80]!r}: "
                f"labels={len(label_rows)}, preds={len(pred_rows)}"
            )
        aligned.extend(zip(label_rows, pred_rows))

    stats = {
        "matched_inputs": len(shared_inputs),
        "missing_inputs": len(missing_inputs),
        "extra_inputs": len(extra_inputs),
        "matched_samples": len(aligned),
    }
    return aligned, stats


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_field_level_ner_metrics(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    field_metrics: Dict[str, Dict[str, Any]] = {}
    macro_precision: List[float] = []
    macro_recall: List[float] = []
    macro_f1: List[float] = []

    for field in fields:
        tp = 0
        fp = 0
        fn = 0

        for row in rows:
            pred_items = normalize_multi_value(row["pred"].get(field), field=field)
            label_items = normalize_multi_value(row["label"].get(field), field=field)
            overlap = pred_items & label_items
            tp += len(overlap)
            fp += len(pred_items - label_items)
            fn += len(label_items - pred_items)

        precision, recall, f1 = prf(tp, fp, fn)
        field_metrics[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }
        macro_precision.append(precision)
        macro_recall.append(recall)
        macro_f1.append(f1)

    avg_precision = sum(macro_precision) / len(macro_precision) if macro_precision else 0.0
    avg_recall = sum(macro_recall) / len(macro_recall) if macro_recall else 0.0
    avg_f1 = sum(macro_f1) / len(macro_f1) if macro_f1 else 0.0

    field_metrics["AVERAGE"] = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
    }
    return field_metrics


def compute_ner_block(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Any]:
    field_metrics = compute_field_level_ner_metrics(rows, fields)
    return {
        "field_metrics": field_metrics,
        "avg": field_metrics["AVERAGE"],
    }


def evaluate_directory(exp_dir: Path, label_rows: Sequence[Dict[str, Any]], fields: Sequence[str], label_path: Path) -> Path:
    prediction_path = find_prediction_path(exp_dir)
    prediction_rows = [row for row in load_jsonl(prediction_path) if is_ner_sample(row)]

    aligned_raw, align_stats = align_rows_by_input(label_rows, prediction_rows)
    parsed_rows: List[Dict[str, Any]] = []
    for label_row, pred_row in aligned_raw:
        parsed_rows.append(
            {
                "label": parse_label_record(label_row, fields),
                "pred": parse_prediction_record(pred_row, fields),
                "language": str(pred_row.get("language", "")).strip().upper(),
            }
        )

    counts_by_language: Dict[str, int] = defaultdict(int)
    by_language: Dict[str, Dict[str, Any]] = {}
    for lang in sorted({row["language"] for row in parsed_rows if row["language"]}):
        lang_rows = [row for row in parsed_rows if row["language"] == lang]
        counts_by_language[lang] = len(lang_rows)
        by_language[lang] = {
            "ner": compute_ner_block(lang_rows, fields),
        }

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(prediction_path),
        "label_path": str(label_path),
        "total_samples": len(parsed_rows),
        "counts": {
            "ner": len(parsed_rows),
        },
        "ner": compute_ner_block(parsed_rows, fields),
        "action": {},
        "filters": {
            "task_filter": "ner",
            "language_filter": "all",
        },
        "match_key": "input",
        "fields": list(fields),
        "matched_inputs": align_stats["matched_inputs"],
        "missing_inputs": align_stats["missing_inputs"],
        "extra_inputs": align_stats["extra_inputs"],
        "by_language_counts": dict(counts_by_language),
        "by_language": by_language,
    }

    output_path = exp_dir / Path(CONFIG["output_name"]).name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] {exp_dir.name}: matched={len(parsed_rows)}, "
        f"avg_f1={payload['ner']['avg']['f1']:.4f}, saved={output_path}"
    )
    return output_path


def main() -> None:
    fields = resolve_fields(CONFIG["fields"])
    label_path = Path(CONFIG["label_path"]).resolve()
    label_rows = load_json(label_path)
    if not isinstance(label_rows, list):
        raise ValueError(f"Label file must be a JSON list: {label_path}")

    dir_list = CONFIG["dirs"]
    if not dir_list:
        raise ValueError("CONFIG['dirs'] is empty.")

    for dir_str in dir_list:
        evaluate_directory(
            exp_dir=Path(dir_str).resolve(),
            label_rows=label_rows,
            fields=fields,
            label_path=label_path,
        )


if __name__ == "__main__":
    main()
