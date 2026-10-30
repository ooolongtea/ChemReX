"""Apply validation patches and re-score before accepting changes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from .build_candidates import build_candidates
from .diagnose import Diagnosis
from .normalize import normalize_reaction
from .patch import apply_patch
from .atom_sources import check_atom_sources

if TYPE_CHECKING:
    from .map_score import MappingScorer


def verify_patch(
    reaction: Dict[str, Any],
    patches: List[Dict[str, Any]],
    diagnosis: Diagnosis,
    scorer: MappingScorer,
    confidence_threshold: float,
    confidence_delta: float,
    max_reagent_candidates: int,
) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
    if not patches:
        return reaction, False, {"accepted": False, "reason": "empty_patch"}

    updated = apply_patch(reaction, patches)
    normalized = normalize_reaction(updated)
    candidates = build_candidates(updated, normalized, max_reagent_candidates=max_reagent_candidates)
    scored = scorer.score(candidates)
    base = next((candidate for candidate in scored if candidate.kind == "base"), None)

    before = diagnosis.base_candidate.confidence if diagnosis.base_candidate else 0.0
    after = base.confidence if base else 0.0
    source_repair = diagnosis.reason == "source_supported_reagent_closes_carbon_gap"
    accepted = bool(base and after >= confidence_threshold and
                    check_atom_sources(base.expression, base.mapped_rxn)["status"] == "complete"
                    and (source_repair or after - before >= confidence_delta))
    info = {
        "accepted": accepted,
        "mapping_confidence_before": before,
        "mapping_confidence_after": after,
        "base_expression_after": base.expression if base else "",
        "mapped_rxn_after": base.mapped_rxn if base else "",
        "reason": "reverified_confident_mapping" if accepted else "reverification_failed",
    }
    return (updated if accepted else reaction), accepted, info
