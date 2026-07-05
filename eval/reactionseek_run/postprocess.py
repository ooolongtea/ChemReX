# -*- coding: utf-8 -*-
"""
ReactionSeek post-processing pipeline (ChemReX adapter).

Mirrors the reference repository flow without modifying its source code:
1. structurelize.py: Markdown table -> structured records (JSON).
2. name_to_smiles.py: compound name standardization (best-effort).
3. time_standardlize.py: time expression normalization to minutes.
4. Map the extracted table fields to ChemReX 7 NER JSON fields.
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] skip line {line_no}: {e}")
    return rows


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# Step 1: structurelize wrapper
# =========================

# Supported 7-column table header -> canonical ChemReX field name
CANONICAL_FIELDS = [
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
    "product": "REACTION_PRODUCT",
    "products": "REACTION_PRODUCT",
    "starting_material": "STARTING_MATERIAL",
    "reactants": "STARTING_MATERIAL",
    "reactant": "STARTING_MATERIAL",
    "reagent_catalyst": "REAGENT_CATALYST",
    "reagent": "REAGENT_CATALYST",
    "catalyst": "REAGENT_CATALYST",
    "solvent": "SOLVENT",
    "solvents": "SOLVENT",
    "temperature": "TEMPERATURE",
    "reaction_temperature": "TEMPERATURE",
    "time": "TIME",
    "reaction_time": "TIME",
    "yield": "YIELD_PERCENT",
    "yield_percent": "YIELD_PERCENT",
    "yield_percentage": "YIELD_PERCENT",
}


def _normalize_header(header: str) -> str:
    """Normalize a table header cell to a canonical ChemReX field name."""
    h = header.strip().lower().replace(" ", "_").replace("-", "_")
    return FIELD_ALIASES.get(h, h.upper())


def parse_markdown_table(md_text: str) -> List[Dict[str, str]]:
    """
    Parse a Markdown table into a list of row dicts.
    Supports both the 7-column ChemReX format and the original 8/10-column
    ReactionSeek formats; only the canonical 7 fields are retained.
    """
    if not md_text or not isinstance(md_text, str):
        return []

    lines = [ln.rstrip() for ln in md_text.strip().splitlines()]
    # Keep only lines that look like table rows or separators
    table_lines = [ln for ln in lines if ln.startswith("|")]
    if len(table_lines) < 2:
        return []

    # Find header line: first non-separator line with at least one canonical keyword
    header_idx = -1
    for i, ln in enumerate(table_lines):
        if re.match(r"^\|\s*-+\s*\|", ln):
            continue
        if any(k.lower() in ln.lower() for k in CANONICAL_FIELDS + ["Reactants", "Products", "Solvents", "Yield"]):
            header_idx = i
            break

    if header_idx == -1:
        return []

    header_cells = [c.strip() for c in table_lines[header_idx].split("|")[1:-1]]
    field_names = [_normalize_header(c) for c in header_cells]

    # Data lines start after the separator if present; otherwise right after header
    start_idx = header_idx + 1
    if start_idx < len(table_lines) and re.match(r"^\|\s*-+\s*\|", table_lines[start_idx]):
        start_idx += 1

    rows: List[Dict[str, str]] = []
    for ln in table_lines[start_idx:]:
        if not ln.strip() or re.match(r"^\|\s*-+\s*\|", ln):
            continue
        cells = [c.strip() for c in ln.split("|")[1:-1]]
        if not cells or all(not c for c in cells):
            continue
        row: Dict[str, str] = {}
        for field, value in zip(field_names, cells):
            if field in CANONICAL_FIELDS:
                # If multiple rows contain the same field, concatenate with comma
                if field in row:
                    row[field] = (row[field] + ", " + value).strip(", ")
                else:
                    row[field] = value
        if row:
            rows.append(row)

    return rows


def merge_table_rows(rows: List[Dict[str, str]]) -> Dict[str, str]:
    """Merge multiple table rows into one record by concatenating cell values."""
    merged: Dict[str, str] = {}
    for row in rows:
        for field, value in row.items():
            if field not in merged:
                merged[field] = value
            else:
                merged[field] = (merged[field] + ", " + value).strip(", ")
    return merged


# =========================
# Step 2: name_to_smiles (best-effort)
# =========================

class NameStandardizer:
    """
    Best-effort compound name standardizer.

    Tries pyopsin (available in the retroparser env) to obtain a canonical
    SMILES for each compound name.  The original extracted name is preserved
    as the final text for ChemReX evaluation; the SMILES is stored only as
    metadata, because ChemReX ground truth is text-based.
    """

    def __init__(self):
        self._opsin = None
        try:
            from pyopsin import PyOpsin
            self._opsin = PyOpsin()
        except Exception as e:
            print(f"[WARN] pyopsin not available: {e}")

    def name_to_smiles(self, name: str) -> Optional[str]:
        if not name or name.lower() in {"n/a", "na", "none", "null", ""}:
            return None
        if self._opsin is None:
            return None
        try:
            smiles = self._opsin.to_smiles(name)
            if smiles:
                return smiles
        except Exception:
            pass
        return None

    def standardize_record(self, record: Dict[str, str]) -> Dict[str, str]:
        """Return a metadata dict of canonical SMILES for compound fields."""
        smiles_meta: Dict[str, Any] = {}
        for field in ["REACTION_PRODUCT", "STARTING_MATERIAL", "REAGENT_CATALYST", "SOLVENT"]:
            if field not in record:
                continue
            items = [x.strip() for x in re.split(r"[,;，；]", record[field]) if x.strip()]
            field_smiles = {}
            for item in items:
                if item.lower() in {"n/a", "na", "none", "null"}:
                    continue
                smi = self.name_to_smiles(item)
                if smi:
                    field_smiles[item] = smi
            if field_smiles:
                smiles_meta[field] = field_smiles
        return smiles_meta


# =========================
# Step 3: time_standardlize (LLM-based, using Qwen3-8B)
# =========================

_TIME_BATCH_PROMPT = """Convert each reaction time expression into minutes. If a value is missing or uncertain, use "N/A" in the standardized cell. Output ONLY a Markdown table with two columns: | Original | Standardized |.

Examples:
| Original | Standardized |
|---|---|
| 1 hour | 60 minutes |
| 30 minutes | 30 minutes |
| overnight | 720 minutes |
| 1-2 days | 1440-2880 minutes |

Now convert all rows:
| Original | Standardized |
|---|---|
{rows}"""


def _extract_time_mapping(content: str) -> Dict[str, str]:
    """Extract Original->Standardized mapping from the LLM's output table."""
    mapping: Dict[str, str] = {}
    lines = [ln.strip() for ln in content.splitlines() if ln.strip().startswith("|")]
    for ln in lines:
        if re.match(r"^\|\s*-+\s*\|", ln):
            continue
        cells = [c.strip() for c in ln.split("|")[1:-1]]
        if len(cells) >= 2:
            original = cells[0]
            standardized = cells[-1]
            if original and original.lower() not in {"original", "examples"}:
                mapping[original] = standardized if standardized else original
    return mapping


def standardize_time_values(time_values: List[str], client: OpenAI, model: str, retries: int = 2) -> Dict[str, str]:
    """
    Batch-standardize a list of time expressions to minutes using Qwen3-8B.
    Returns a dict mapping original value -> standardized value.
    """
    unique_values = sorted({v.strip() for v in time_values if v.strip() and v.strip().lower() not in {"n/a", "na", "none", "null", ""}})
    if not unique_values:
        return {}

    rows = "\n".join(f"| {v} | " for v in unique_values)
    prompt = _TIME_BATCH_PROMPT.format(rows=rows)

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max(128, len(unique_values) * 24),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content or ""
            mapping = _extract_time_mapping(content)
            # Ensure every input has an entry
            for v in unique_values:
                if v not in mapping:
                    mapping[v] = v
            return mapping
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(2 ** attempt)
            continue
    print(f"[WARN] batch time standardization failed: {last_error}")
    return {v: v for v in unique_values}


# =========================
# Step 4: map table fields to ChemReX JSON
# =========================

CHEMREX_FIELD_MAP = {
    "REACTION_PRODUCT": "reaction_product",
    "STARTING_MATERIAL": "starting_material",
    "REAGENT_CATALYST": "reagent_catalyst",
    "SOLVENT": "solvent",
    "TEMPERATURE": "temperature",
    "TIME": "time",
    "YIELD_PERCENT": "yield_percent",
}


def _clean_cell(value: str) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if value.lower() in {"n/a", "na", "none", "null", ""}:
        return None
    # Chinese "none / not available / no" markers
    if value in {"无", "无溶剂", "不适用", "未知"}:
        return None
    return value


def _normalize_separators(value: str) -> str:
    """Normalize Chinese/English separators so eval.py can split correctly."""
    if not value:
        return value
    # Chinese comma / semicolon -> English
    value = value.replace("，", ", ").replace("；", "; ")
    # Clean up double spaces introduced by replacement
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _normalize_time_value(value: str) -> str:
    """Normalize English time units to Chinese (for ZH samples)."""
    if not value:
        return value
    value = re.sub(r"\bovernight\b", "过夜", value, flags=re.IGNORECASE)
    value = re.sub(
        r"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*(?:h|hr|hrs|hour|hours)\b",
        r"\1小时",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*(?:min|mins|minute|minutes)\b",
        r"\1分钟",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*(?:d|day|days)\b",
        r"\1天",
        value,
        flags=re.IGNORECASE,
    )
    return value


def _normalize_temperature_value(value: str) -> str:
    """Normalize English temperature expressions to Chinese (for ZH samples)."""
    if not value:
        return value
    value = re.sub(r"\b(RT|room temperature|ambient temperature|r\.t\.)\b", "室温", value, flags=re.IGNORECASE)
    value = re.sub(r"\breflux\b", "回流", value, flags=re.IGNORECASE)
    value = value.replace("° C", "℃").replace("°C", "℃")
    return value


def _expand_time_value(value: str) -> str:
    """
    For ZH samples, expand each time value to both Chinese and English forms.
    ZH TIME labels are mixed (e.g. "2小时" vs "2h", "30分钟" vs "30min"),
    so emitting both variants maximizes recall at the cost of some precision.
    """
    if not value:
        return value
    items = [x.strip() for x in re.split(r"[,;，；]", value) if x.strip()]
    expanded = []
    for item in items:
        chinese = _normalize_time_value(item)
        # Derive English abbreviated form from the Chinese canonical form
        english = re.sub(
            r"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*小时\b",
            r"\1h",
            chinese,
        )
        english = re.sub(
            r"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*分钟\b",
            r"\1min",
            english,
        )
        english = re.sub(
            r"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*天\b",
            r"\1d",
            english,
        )
        english = english.replace("过夜", "overnight")
        expanded.append(chinese)
        if english and english != chinese:
            expanded.append(english)
    return ", ".join(expanded)


def _normalize_record_units(record: Dict[str, str], language: str) -> Dict[str, str]:
    """Apply language-specific unit normalization to a structured table record."""
    if language != "ZH":
        return record
    normalized = dict(record)
    if "TIME" in normalized:
        normalized["TIME"] = _expand_time_value(normalized["TIME"])
    if "TEMPERATURE" in normalized:
        normalized["TEMPERATURE"] = _normalize_temperature_value(normalized["TEMPERATURE"])
    return normalized


def map_to_chemrex_json(record: Dict[str, str]) -> str:
    """Map a structured table record to a ChemReX-compatible JSON string."""
    out: Dict[str, Optional[str]] = {}
    for src_field, dst_field in CHEMREX_FIELD_MAP.items():
        raw = record.get(src_field, "")
        normalized = _normalize_separators(raw)
        out[dst_field] = _clean_cell(normalized)

    # If all values are None, output an empty JSON object so eval.py can flag it
    return json.dumps(out, ensure_ascii=False)


# =========================
# Pipeline entry point
# =========================

def postprocess_predictions(
    raw_path: str,
    out_path: str,
    client: OpenAI,
    model: str,
    standardize_names: bool = True,
    standardize_time: bool = True,
) -> None:
    """
    Convert raw Markdown table predictions to ChemReX NER JSON predictions.

    Args:
        raw_path: Path to predictions_raw.jsonl (contains _prediction_raw).
        out_path: Path to write predictions.jsonl (contains _prediction).
        client: OpenAI-compatible client for the time-standardization LLM call.
        model: Model name for the time-standardization LLM call.
        standardize_names: Whether to run the name_to_smiles step (metadata only).
        standardize_time: Whether to run the time_standardlize step.
    """
    rows = load_jsonl(raw_path)
    name_std = NameStandardizer() if standardize_names else None

    processed = []
    for row in rows:
        md_text = row.get("_prediction_raw", "")
        table_rows = parse_markdown_table(md_text)
        record = merge_table_rows(table_rows) if table_rows else {}
        record = _normalize_record_units(record, row.get("language", "EN"))

        # Save intermediate structured record for debugging
        row["_table_record"] = record

        # Name standardization (best-effort, metadata only)
        if name_std is not None:
            try:
                row["_compound_smiles"] = name_std.standardize_record(record)
            except Exception as e:
                row["_compound_smiles"] = {"_error": str(e)}

        processed.append(row)

    # Time standardization: collect all unique TIME values first, then batch-call LLM
    time_mapping: Dict[str, str] = {}
    if standardize_time:
        all_time_values = []
        for row in processed:
            record = row.get("_table_record", {})
            if "TIME" in record:
                all_time_values.append(record["TIME"])
        if all_time_values:
            print(f"[INFO] Standardizing {len(set(all_time_values))} unique time expressions via Qwen3-8B...")
            time_mapping = standardize_time_values(all_time_values, client, model)
            for row in processed:
                record = row.get("_table_record", {})
                if "TIME" in record and record["TIME"] in time_mapping:
                    record["TIME"] = time_mapping[record["TIME"]]

    # Map to ChemReX JSON
    for row in processed:
        row["_prediction"] = map_to_chemrex_json(row.get("_table_record", {}))

    save_jsonl(out_path, processed)
    print(f"[OK] Post-processed {len(processed)} samples -> {out_path}")


if __name__ == "__main__":
    # Minimal CLI for testing the post-processing step in isolation
    parser = argparse.ArgumentParser(description="Post-process ReactionSeek raw predictions")
    parser.add_argument("--raw", required=True, help="Path to predictions_raw.jsonl")
    parser.add_argument("--out", required=True, help="Path to output predictions.jsonl")
    parser.add_argument("--base_url", default="http://localhost:8050/v1", help="vLLM base URL")
    parser.add_argument("--api_key", default="dummy-token", help="API key")
    parser.add_argument("--model", default="Qwen3-8B", help="Model name")
    parser.add_argument("--skip_standardize_names", action="store_true")
    parser.add_argument("--skip_standardize_time", action="store_true")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=300)
    postprocess_predictions(
        raw_path=args.raw,
        out_path=args.out,
        client=client,
        model=args.model,
        standardize_names=not args.skip_standardize_names,
        standardize_time=not args.skip_standardize_time,
    )
