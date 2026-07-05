"""Robust parsing of NER / Action LLM outputs (local impl, does NOT import eval/).

Mirrors eval/eval.py parsing semantics:
- fenced json / markdown blocks, raw json, outermost {...}/[...] extraction
- NER field-name normalization + multi-value split + yield normalization + null handling
- Action verb-sequence parsing (comma / semicolon / json array)
"""

import json
import re
from typing import Any, Dict, List, Optional

NER_FIELDS = [
    "REACTION_PRODUCT",
    "STARTING_MATERIAL",
    "REAGENT_CATALYST",
    "SOLVENT",
    "TEMPERATURE",
    "TIME",
    "YIELD_PERCENT",
]
FIELD_NAME_MAPPING = {field.lower(): field for field in NER_FIELDS}
MULTIVALUE_FIELDS = {"REACTION_PRODUCT", "STARTING_MATERIAL", "REAGENT_CATALYST", "SOLVENT"}
SCALAR_FIELDS = {"TEMPERATURE", "TIME", "YIELD_PERCENT"}

_WS_RE = re.compile(r"\s+")
_YIELD_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%?\s*$")
_NULL_LIKE = {"", "null", "none", "n/a", "notspecified", "not specified"}


# ── JSON extraction ──

def _try_load_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_fenced_block(text: str, start_markers: List[str], end_marker: str) -> Optional[str]:
    low = text.lower()
    start_idx = -1
    for m in start_markers:
        idx = low.find(m.lower())
        if idx != -1:
            start_idx = idx + len(m)
            break
    if start_idx == -1:
        return None
    end_idx = text.find(end_marker, start_idx)
    if end_idx == -1:
        return None
    return text[start_idx:end_idx].strip()


def _extract_fenced_block_allow_eof(text: str, start_markers: List[str], end_marker: str) -> Optional[str]:
    block = _extract_fenced_block(text, start_markers, end_marker)
    if block is not None:
        return block
    low = text.lower()
    for m in start_markers:
        idx = low.find(m.lower())
        if idx != -1:
            return text[idx + len(m):].strip()
    return None


def parse_json_output(output_str: str) -> Any:
    """Extract a JSON object/array from LLM output (fenced or raw)."""
    if not output_str or not isinstance(output_str, str):
        return None
    s = output_str.strip()

    for start, end in (
        ("```json", "```"), ("```JSON", "```"),
        ("'''json", "'''"), ("'''JSON", "'''"),
        ("```markdown", "```"), ("```MARKDOWN", "```"), ("```md", "```"), ("```MD", "```"),
        ("'''markdown", "'''"), ("'''MARKDOWN", "'''"), ("'''md", "'''"), ("'''MD", "'''"),
    ):
        block = _extract_fenced_block(s, [start], end)
        if block:
            parsed = _try_load_json(block)
            if parsed is not None:
                return parsed

    block = _extract_fenced_block_allow_eof(s, ["```"], "```")
    if block:
        parsed = _try_load_json(block)
        if parsed is not None:
            return parsed
    block = _extract_fenced_block_allow_eof(s, ["'''"], "'''")
    if block:
        parsed = _try_load_json(block)
        if parsed is not None:
            return parsed

    parsed = _try_load_json(s)
    if parsed is not None:
        return parsed

    first_obj = s.find("{")
    first_arr = s.find("[")
    if first_obj == -1 and first_arr == -1:
        return None
    start = first_obj if (first_obj != -1 and (first_arr == -1 or first_obj < first_arr)) else first_arr
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return _try_load_json(s[start:i + 1].strip())
    return None


# ── normalization helpers ──

def strip_all_whitespace(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return _WS_RE.sub("", s)


def normalize_yield_value(v: Any) -> str:
    s = strip_all_whitespace(v)
    if not s:
        return ""
    m = _YIELD_NUM_RE.match(s)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d+(?:\.\d+)?)", s)
    if m2:
        return m2.group(1)
    return s


def _is_null_like(s: str) -> bool:
    return s.strip().lower() in _NULL_LIKE


def normalize_field_name(field_name: str) -> str:
    if not field_name:
        return ""
    mapped = FIELD_NAME_MAPPING.get(field_name.strip().lower())
    return mapped if mapped else field_name.strip().upper()


def _split_multi_value(text: str, field: Optional[str] = None) -> List[str]:
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    s0 = text.strip()
    if _is_null_like(s0):
        return []
    if field == "YIELD_PERCENT":
        y = normalize_yield_value(text)
        return [] if not y else [y]
    s = text.replace("；", ";")
    parts: List[str] = []
    for chunk in s.split(";"):
        for item in chunk.split(","):
            cleaned = item.strip()
            if cleaned:
                parts.append(cleaned)
    return parts


# ── NER parsing ──

def parse_ner_output(output_str: str) -> Dict[str, str]:
    """Parse NER LLM output to {FIELD: raw_string} (unsplit, for eval compatibility)."""
    parsed = parse_json_output(output_str)
    if isinstance(parsed, list):
        if len(parsed) == 0:
            return {}
        if isinstance(parsed[0], dict):
            parsed = parsed[0]
        else:
            return {}
    if parsed is None or not isinstance(parsed, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in parsed.items():
        kk = normalize_field_name(str(k))
        out[kk] = "" if v is None else str(v)
    return out


def parse_ner_structured(output_str: str) -> Dict[str, Any]:
    """Parse NER LLM output to structured form: multi-value fields → list, scalar → str|None."""
    raw = parse_ner_output(output_str)
    out: Dict[str, Any] = {}
    for field in NER_FIELDS:
        val = raw.get(field, "")
        if field in MULTIVALUE_FIELDS:
            out[field] = _split_multi_value(val, field=field)
        elif field == "YIELD_PERCENT":
            y = normalize_yield_value(val)
            out[field] = y if y else None
        else:
            s = (val or "").strip()
            out[field] = None if _is_null_like(s) else s
    return out


# ── Action parsing ──

def parse_action_output(output_str: str) -> List[str]:
    """Parse Action LLM output to an ordered verb list."""
    if not output_str or not isinstance(output_str, str):
        return []
    s = output_str.strip()
    parsed = parse_json_output(s)
    if parsed is not None:
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
        if isinstance(parsed, dict):
            for key in ("actions", "action_sequence", "steps", "operations"):
                if key in parsed and isinstance(parsed[key], list):
                    return [str(x).strip() for x in parsed[key] if x is not None and str(x).strip()]
    if "," in s:
        return [a.strip() for a in s.split(",") if a.strip()]
    if ";" in s or "；" in s:
        s = s.replace("；", ";")
        return [a.strip() for a in s.split(";") if a.strip()]
    if "、" in s:
        return [a.strip() for a in s.split("、") if a.strip()]
    return []
