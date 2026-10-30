"""Bracketed image-label matching shared by reuse guarding and backfill."""

import re


def image_scope_labels(image_evidence):
    """Label keys of the chunk's own image dictionary."""
    return set((image_evidence or {}).get("dict_result") or {})


def _label_pattern(label):
    stripped = str(label or "").strip("()（）[]【】 ")
    if not stripped:
        return None
    return re.compile(r"[(（]\s*" + re.escape(stripped) + r"\s*[)）]")


def carries_image_label(name, labels):
    """True when the name carries one of the labels as a bracketed token.

    A pure-number label must not match by substring ("1" vs "化合物10"):
    the label has to appear inside brackets in the name ("胺(H)",
    "式一化合物(K)", "product of Preparation 8 (J)").
    """
    for label in labels or ():
        pattern = _label_pattern(label)
        if pattern and pattern.search(str(name or "")):
            return True
    return False
