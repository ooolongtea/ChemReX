"""Lightweight field-constraint checker for NER+Action records.

Advisory checks only — does NOT modify fields (per rxn验证模块实现建议.md, only P/S/G + grounding
are modified by the mapping module, never Temp/Time/Y/A). Returns a list of issue dicts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_TEMP_RE = re.compile(r"(℃|°C|°F|K\b|room|reflux|回流|室温|常温|heat|加热|冰|盐浴|液氮)", re.IGNORECASE)
_TIME_RE = re.compile(r"(h\b|hr\b|hour|min\b|minute|day|overnight|过夜|小时|分钟|天|秒|sec\b)", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _extract_name(compound: Any) -> str:
    if isinstance(compound, dict):
        return str(compound.get("name") or compound.get("standard_name") or "").strip()
    return str(compound or "").strip()


def check_constraints(record: Dict[str, Any]) -> Dict[str, Any]:
    """Run lightweight field-constraint checks on a standardized NER+Action record.

    Args:
        record: {"ner": {7 fields}, "action": [...], ...}
    Returns:
        {"passed": bool, "issues": [{"field", "type", "message"}], "summary": str}
    """
    ner = record.get("ner", {}) or {}
    action = record.get("action", []) or []
    issues: List[Dict[str, str]] = []

    products = ner.get("REACTION_PRODUCT", []) or []
    if isinstance(products, str):
        products = [products] if products.strip() else []
    starting = ner.get("STARTING_MATERIAL", []) or []
    if isinstance(starting, str):
        starting = [starting] if starting.strip() else []
    reagents = ner.get("REAGENT_CATALYST", []) or []
    if isinstance(reagents, str):
        reagents = [reagents] if reagents.strip() else []
    solvents = ner.get("SOLVENT", []) or []
    if isinstance(solvents, str):
        solvents = [solvents] if solvents.strip() else []

    # 1. Completeness: product and starting_material should be present
    if not products or not any(_extract_name(p) for p in products):
        issues.append({"field": "REACTION_PRODUCT", "type": "missing", "message": "product is empty"})
    if not starting or not any(_extract_name(s) for s in starting):
        issues.append({"field": "STARTING_MATERIAL", "type": "missing", "message": "starting_material is empty"})

    # 2. Temperature format
    temp = ner.get("TEMPERATURE")
    if temp:
        if not _TEMP_RE.search(str(temp)) and not _NUM_RE.search(str(temp)):
            issues.append({"field": "TEMPERATURE", "type": "format", "message": f"temperature value looks unusual: {temp!r}"})

    # 3. Time format
    time_val = ner.get("TIME")
    if time_val:
        if not _TIME_RE.search(str(time_val)) and not _NUM_RE.search(str(time_val)):
            issues.append({"field": "TIME", "type": "format", "message": f"time value looks unusual: {time_val!r}"})

    # 4. Yield range (0-100)
    yld = ner.get("YIELD_PERCENT")
    if yld is not None:
        try:
            y = float(str(yld).replace("%", "").strip())
            if y < 0 or y > 100:
                issues.append({"field": "YIELD_PERCENT", "type": "range", "message": f"yield out of 0-100 range: {yld!r}"})
        except (ValueError, TypeError):
            issues.append({"field": "YIELD_PERCENT", "type": "format", "message": f"yield not numeric: {yld!r}"})

    # 5. Action sequence non-empty (informational)
    if not action:
        issues.append({"field": "ACTION", "type": "missing", "message": "action sequence is empty"})

    passed = len(issues) == 0
    summary = "all constraints passed" if passed else f"{len(issues)} constraint issue(s)"
    return {"passed": passed, "issues": issues, "summary": summary}
