#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


FIELD_ORDER = [
    "reaction_product",
    "starting_material",
    "reagent_catalyst",
    "solvent",
    "temperature",
    "time",
    "yield_percent",
]

FIELD_ALIASES = {
    "reaction_product": ["reaction_product", "REACTION_PRODUCT"],
    "starting_material": ["starting_material", "STARTING_MATERIAL"],
    "reagent_catalyst": ["reagent_catalyst", "REAGENT_CATALYST"],
    "solvent": ["solvent", "SOLVENT"],
    "temperature": ["temperature", "TEMPERATURE"],
    "time": ["time", "TIME"],
    "yield_percent": ["yield_percent", "YIELD_PERCENT"],
}


def parse_json_from_text(text: Any) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {}
    s = text.strip()
    if not s:
        return {}

    for marker in ("```json", "```JSON"):
        start = s.find(marker)
        if start != -1:
            start += len(marker)
            end = s.find("```", start)
            if end != -1:
                chunk = s[start:end].strip()
                try:
                    obj = json.loads(chunk)
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    pass

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_canonical_fields(parsed: Dict[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for field in FIELD_ORDER:
        value = ""
        for alias in FIELD_ALIASES[field]:
            if alias in parsed:
                value = _to_text(parsed.get(alias))
                break
        result[field] = value
    return result


def _normalize_multi_value(text: str) -> List[str]:
    # 与评测逻辑保持一致：支持 ; / , 拆分并忽略顺序
    if not text:
        return []
    normalized = text.replace("；", ";").replace("，", ",")
    items: List[str] = []
    for chunk in normalized.split(";"):
        for part in chunk.split(","):
            item = part.strip()
            if item:
                items.append(item)
    return sorted(set(items))


def _field_equal(label_value: str, pred_value: str) -> bool:
    return _normalize_multi_value(label_value) == _normalize_multi_value(pred_value)


def extract_record(obj: Dict[str, Any], idx: int) -> Dict[str, Any]:
    pred_parsed = obj.get("_pred_parsed")
    if not isinstance(pred_parsed, dict):
        pred_parsed = parse_json_from_text(obj.get("_prediction", ""))

    ref_parsed = obj.get("_ref_parsed")
    if not isinstance(ref_parsed, dict):
        ref_parsed = parse_json_from_text(obj.get("output", ""))

    return {
        "id": idx,
        "input": obj.get("input", ""),
        "prediction": _extract_canonical_fields(pred_parsed),
        "label": _extract_canonical_fields(ref_parsed),
    }


def build_compare_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": record["id"],
        "input": record["input"],
    }
    for field in FIELD_ORDER:
        label_value = record["label"].get(field, "")
        pred_value = record["prediction"].get(field, "")
        if _field_equal(label_value, pred_value):
            out[field] = label_value
        else:
            out[f"{field}_label"] = label_value
            out[f"{field}_pre"] = pred_value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract label/prediction/compare(only mismatches) from predictions.jsonl"
    )
    parser.add_argument("--input", required=True, help="Path to predictions.jsonl")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory; default: same directory as input file",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Output file prefix; default: input stem",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix if args.prefix else input_path.stem

    label_path = output_dir / f"{prefix}_lable.json"
    prediction_path = output_dir / f"{prefix}_prediction.json"
    compare_path = output_dir / f"{prefix}_compare_mismatch.json"

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            records.append(extract_record(obj, i))

    label_rows = []
    prediction_rows = []
    compare_rows = []

    for r in records:
        label_row = {
            "id": r["id"],
            "input": r["input"],
            **r["label"],
        }
        pred_row = {
            "id": r["id"],
            "input": r["input"],
            **r["prediction"],
        }
        label_rows.append(label_row)
        prediction_rows.append(pred_row)

        mismatched = any(
            not _field_equal(r["label"].get(field, ""), r["prediction"].get(field, ""))
            for field in FIELD_ORDER
        )
        if mismatched:
            compare_rows.append(build_compare_record(r))

    with label_path.open("w", encoding="utf-8") as f:
        json.dump(label_rows, f, ensure_ascii=False, indent=2)

    with prediction_path.open("w", encoding="utf-8") as f:
        json.dump(prediction_rows, f, ensure_ascii=False, indent=2)

    with compare_path.open("w", encoding="utf-8") as f:
        json.dump(compare_rows, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(label_rows)} records -> {label_path}")
    print(f"Saved {len(prediction_rows)} records -> {prediction_path}")
    print(f"Saved {len(compare_rows)} mismatch records -> {compare_path}")


if __name__ == "__main__":
    main()
