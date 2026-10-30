"""Rule-based diagnosis for mapping confidence patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .build_candidates import ReactionCandidate
from .atom_sources import check_atom_sources, contributes_carbon


@dataclass
class Diagnosis:
    diagnosis: str
    status: str
    needs_human_review: bool
    reason: str
    base_candidate: Optional[ReactionCandidate] = None
    best_candidate: Optional[ReactionCandidate] = None

    def to_public_dict(self) -> Dict:
        return {
            "diagnosis": self.diagnosis,
            "status": self.status,
            "needs_human_review": self.needs_human_review,
            "reason": self.reason,
        }


def diagnose_candidates(
    candidates: List[ReactionCandidate],
    confidence_threshold: float = 0.9,
    confidence_delta: float = 0.2,
    ambiguity_margin: float = 0.05,
) -> Diagnosis:
    if not candidates:
        return Diagnosis(
            diagnosis="structure_unavailable",
            status="skipped",
            needs_human_review=True,
            reason="missing_valid_reactant_or_product_smiles",
        )

    base = next((candidate for candidate in candidates if candidate.kind == "base"), candidates[0])
    ranked = sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)
    best = ranked[0]

    left = {c.canonical_smiles for c in base.left_compounds if c.canonical_smiles}
    products = {c.canonical_smiles for c in base.products if c.canonical_smiles}
    if left and products and products.issubset(left):
        return Diagnosis(
            diagnosis="identity_reaction", status="non_reaction",
            needs_human_review=False, reason="product_structure_is_a_reactant",
            base_candidate=base, best_candidate=best,
        )

    carbon = check_atom_sources(base.expression, base.mapped_rxn)
    if carbon["status"] == "incomplete":
        repairs = [c for c in candidates if c.kind == "with_reagent"
                   and c.confidence >= confidence_threshold
                   and check_atom_sources(c.expression, c.mapped_rxn)["status"] == "complete"
                   and contributes_carbon(c)]
        if len(repairs) == 1:
            return Diagnosis("missing_atom_contributing_reagent", "patch_candidate", False,
                             "source_supported_reagent_closes_carbon_gap", base, repairs[0])
        return Diagnosis("missing_atom_source", "needs_review", True,
                         "carbon_source_incomplete_or_ambiguous", base, best)

    if base.confidence >= confidence_threshold:
        return Diagnosis(
            diagnosis="pass",
            status="passed",
            needs_human_review=False,
            reason="base_expression_confident",
            base_candidate=base,
            best_candidate=best,
        )

    if len(ranked) > 1 and best.confidence - ranked[1].confidence < ambiguity_margin:
        return Diagnosis(
            diagnosis="mapping_uncertain",
            status="needs_review",
            needs_human_review=True,
            reason="top_candidate_scores_are_close",
            base_candidate=base,
            best_candidate=best,
        )

    if (
        best.kind == "with_reagent"
        and best.added_compound is not None
        and best.confidence >= confidence_threshold
        and best.confidence - base.confidence >= confidence_delta
    ):
        return Diagnosis(
            diagnosis="missing_atom_contributing_reagent",
            status="patch_candidate",
            needs_human_review=False,
            reason="adding_reagent_to_reactant_side_rescues_mapping_confidence",
            base_candidate=base,
            best_candidate=best,
        )

    return Diagnosis(
        diagnosis="mapping_uncertain",
        status="needs_review",
        needs_human_review=True,
        reason="no_candidate_reaches_confident_mapping",
        base_candidate=base,
        best_candidate=best,
    )
