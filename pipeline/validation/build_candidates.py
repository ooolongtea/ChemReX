"""Build a small set of reaction expressions for RXNMapper scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .normalize import NormalizedCompound, valid_mapping_compounds


ACTION_WORDS = (
    "react", "reaction", "add", "added", "addition", "dropwise", "treated",
    "coupling", "condensation", "reflux", "heated", "stirred", "charge",
    "加入", "滴加", "反应", "处理", "偶联", "缩合", "回流", "加热", "搅拌",
)
WORKUP_WORDS = (
    "extract", "extracted", "wash", "washed", "dry", "dried", "filter",
    "filtered", "chromatography", "purified", "萃取", "洗涤", "干燥", "过滤",
    "柱层析", "纯化",
)


@dataclass
class ReactionCandidate:
    kind: str
    expression: str
    left_compounds: List[NormalizedCompound]
    products: List[NormalizedCompound]
    added_compound: Optional[NormalizedCompound] = None
    mapped_rxn: str = ""
    confidence: float = 0.0
    error: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        from .atom_sources import check_atom_sources
        return {
            "kind": self.kind,
            "expression": self.expression,
            "confidence": self.confidence,
            "mapped_rxn": self.mapped_rxn,
            "error": self.error,
            "atom_source_check": check_atom_sources(self.expression, self.mapped_rxn),
            "added_compound": self.added_compound.to_public_dict() if self.added_compound else None,
            "left_compounds": [compound.to_public_dict() for compound in self.left_compounds],
            "products": [compound.to_public_dict() for compound in self.products],
        }


def make_reaction_expression(left: List[NormalizedCompound], products: List[NormalizedCompound]) -> str:
    left_smiles = ".".join(compound.canonical_smiles for compound in left)
    product_smiles = ".".join(compound.canonical_smiles for compound in products)
    return f"{left_smiles}>>{product_smiles}"


def _context_text(reaction: Dict[str, Any]) -> str:
    parts = [
        str(reaction.get("reference") or ""),
        str(reaction.get("actions") or reaction.get("operation_sequence") or reaction.get("A") or ""),
    ]
    conditions = reaction.get("conditions")
    if isinstance(conditions, dict):
        parts.append(" ".join(str(value) for value in conditions.values()))
    elif conditions:
        parts.append(str(conditions))
    return " ".join(parts)


def _compound_mentions_text(compound: NormalizedCompound) -> List[str]:
    values = [compound.name]
    std = compound.original.get("standard_name") if isinstance(compound.original, dict) else None
    if std and std != compound.name:
        values.append(str(std))
    return [value for value in values if value]


def _appears_before_workup(compound: NormalizedCompound, context: str) -> bool:
    if not context:
        return False
    lower_context = context.lower()
    mention_positions = []
    for mention in _compound_mentions_text(compound):
        pos = lower_context.find(mention.lower())
        if pos >= 0:
            mention_positions.append(pos)
    if not mention_positions:
        return False
    mention_pos = min(mention_positions)
    workup_positions = [
        lower_context.find(word.lower()) for word in WORKUP_WORDS
        if lower_context.find(word.lower()) >= 0
    ]
    return not workup_positions or mention_pos < min(workup_positions)


def _has_reaction_context(compound: NormalizedCompound, context: str) -> bool:
    if not context:
        return False
    lower_context = context.lower()
    if not any(mention.lower() in lower_context for mention in _compound_mentions_text(compound)):
        return False
    pattern = "|".join(re.escape(word.lower()) for word in ACTION_WORDS)
    for mention in _compound_mentions_text(compound):
        pos = lower_context.find(mention.lower())
        if pos < 0:
            continue
        window = lower_context[max(0, pos - 80): pos + len(mention) + 80]
        if re.search(pattern, window):
            return True
    return False


def select_reagent_candidates(
    reaction: Dict[str, Any],
    reagents: List[NormalizedCompound],
    max_candidates: int = 3,
) -> List[NormalizedCompound]:
    context = _context_text(reaction)
    candidates = []
    for compound in valid_mapping_compounds(reagents):
        if not _appears_before_workup(compound, context):
            continue
        if not _has_reaction_context(compound, context):
            continue
        candidates.append(compound)
    # Do not silently exclude a carbon donor because it follows three auxiliaries.
    return candidates


def build_candidates(
    reaction: Dict[str, Any],
    normalized: Dict[str, List[NormalizedCompound]],
    max_reagent_candidates: int = 3,
) -> List[ReactionCandidate]:
    reactants = valid_mapping_compounds(normalized.get("reactants", []))
    participants = valid_mapping_compounds(normalized.get("participants", []))
    products = valid_mapping_compounds(normalized.get("products", []))
    base_left = reactants + participants

    if not base_left or not products:
        return []

    candidates = [
        ReactionCandidate(
            kind="base",
            expression=make_reaction_expression(base_left, products),
            left_compounds=base_left,
            products=products,
        )
    ]

    existing_left_smiles = {compound.canonical_smiles for compound in base_left}
    for reagent in select_reagent_candidates(
        reaction,
        normalized.get("reagents", []),
        max_candidates=max_reagent_candidates,
    ):
        if reagent.canonical_smiles in existing_left_smiles:
            continue
        left = base_left + [reagent]
        candidates.append(
            ReactionCandidate(
                kind="with_reagent",
                expression=make_reaction_expression(left, products),
                left_compounds=left,
                products=products,
                added_compound=reagent,
            )
        )
    return candidates
