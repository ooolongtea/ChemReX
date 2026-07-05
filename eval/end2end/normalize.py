"""Identifier and SMILES normalization utilities for grounding evaluation."""

import re
from typing import List, Optional, Tuple

_PREFIXES = ("compound", "cmpd", "intermediate", "reactant", "product", "example")
_PUNCT_RE = re.compile(r"[.:_\-\s<>\[\](){}]+")
_PLACEHOLDER_RE = re.compile(r"^\d+[a-z]?$", re.IGNORECASE)


def normalize_identifier(raw: str) -> str:
    s = (raw or "").lower().strip()
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            new = re.sub(rf"^{re.escape(prefix)}[\s.:_-]*", "", s)
            if new != s:
                s = new
                changed = True
    s = _PUNCT_RE.sub("", s)
    return s.strip()


def is_placeholder_identifier(name: str) -> bool:
    return bool(_PLACEHOLDER_RE.match((name or "").strip()))


def canonicalize_smiles(smiles: str, isomeric: bool = True) -> Optional[str]:
    if not smiles:
        return None
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    canon = sorted(
        Chem.MolToSmiles(f, canonical=True, isomericSmiles=isomeric) for f in frags
    )
    return ".".join(canon)


def is_valid_smiles(smiles: str) -> bool:
    if not smiles:
        return False
    from rdkit import Chem
    return Chem.MolFromSmiles(smiles) is not None


def iou(box_a: List[float], box_b: List[float]) -> float:
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cer(pred: str, gold: str) -> float:
    if not gold and not pred:
        return 0.0
    m, n = len(gold), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if gold[i - 1] == pred[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / max(m, n) if max(m, n) > 0 else 0.0


def _center_inside(box_a: List[float], box_b: List[float]) -> bool:
    cx = (box_a[0] + box_a[2]) / 2
    cy = (box_a[1] + box_a[3]) / 2
    return box_b[0] <= cx <= box_b[2] and box_b[1] <= cy <= box_b[3]


def greedy_iou_match(
    pred_boxes: List[List[float]],
    gold_boxes: List[List[float]],
    threshold: float = 0.5,
    center_fallback: bool = False,
) -> List[Tuple[int, int, float]]:
    """Greedy matching by IoU. When center_fallback=True, also match pairs
    where one box's center is inside the other (for small objects like text labels)."""
    pairs = []
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gold_boxes):
            score = iou(pb, gb)
            if score >= threshold:
                pairs.append((score, i, j, True))
            elif center_fallback and (
                _center_inside(pb, gb) or _center_inside(gb, pb)
            ):
                pairs.append((score, i, j, False))
    pairs.sort(key=lambda x: (x[3], x[0]), reverse=True)
    matched_pred, matched_gold = set(), set()
    result = []
    for score, i, j, _ in pairs:
        if i not in matched_pred and j not in matched_gold:
            result.append((i, j, score))
            matched_pred.add(i)
            matched_gold.add(j)
    return result
