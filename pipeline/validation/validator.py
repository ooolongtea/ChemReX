"""Pipeline-facing reaction validator (Module 5).

Two-layer validation of a standardized NER+Action record:
1. Lightweight field-constraint checks (``constraints.check_constraints``) — advisory,
   never modifies fields.
2. RXNMapper mapping-guided screening + conservative field-local correction
   (``normalize`` → ``build_candidates`` → ``map_score`` → ``diagnose`` → ``patch`` → ``verify``).

Hard rule (per rxn验证模块实现建议.md): only P/S/G + grounding status may be modified
(via ``reactant_side_participants``). Temperature / Time / Yield / Action sequence are
never modified — only reported on.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from loguru import logger

try:
    from ... import config
except ImportError:
    import config

from .build_candidates import build_candidates
from .constraints import check_constraints
from .diagnose import Diagnosis, diagnose_candidates
from .normalize import normalize_reaction
from .patch import make_patch
from .verify import verify_patch

# NER compound fields → reaction-view fields expected by normalize.py
_NER_TO_VIEW = {
    "STARTING_MATERIAL": "reactants",
    "REACTION_PRODUCT": "products",
    "REAGENT_CATALYST": "reagent_catalyst",
    "SOLVENT": "solvents",
}


class ReactionValidator:
    """Mapping-guided screening + conservative local correction for NER+Action records."""

    def __init__(
        self,
        confidence_threshold: float = None,
        confidence_delta: float = None,
        ambiguity_margin: float = None,
        max_reagent_candidates: int = None,
    ):
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else getattr(config, "VALIDATION_CONFIDENCE_THRESHOLD", 0.9)
        )
        self.confidence_delta = (
            confidence_delta
            if confidence_delta is not None
            else getattr(config, "VALIDATION_CONFIDENCE_DELTA", 0.2)
        )
        self.ambiguity_margin = (
            ambiguity_margin
            if ambiguity_margin is not None
            else getattr(config, "VALIDATION_AMBIGUITY_MARGIN", 0.05)
        )
        self.max_reagent_candidates = (
            max_reagent_candidates
            if max_reagent_candidates is not None
            else getattr(config, "VALIDATION_MAX_REAGENT_CANDIDATES", 3)
        )
        self.field_constraints_enabled = getattr(config, "VALIDATION_FIELD_CONSTRAINTS", True)

        # Lazy / optional RXNMapper — module must stay importable without rxnmapper installed.
        self.scorer = None
        try:
            from .map_score import MappingScorer
            self.scorer = MappingScorer()
        except Exception as exc:
            logger.warning(
                f"RXNMapper unavailable — mapping screening disabled, "
                f"only field-constraint checks will run: {exc}"
            )

        logger.info(
            "ReactionValidator ready "
            f"(tau={self.confidence_threshold}, delta={self.confidence_delta}, "
            f"constraints={self.field_constraints_enabled}, "
            f"mapping={'on' if self.scorer else 'off'})"
        )

    # ── public entry: validate a standardized NER+Action record ──
    def validate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single standardized record ``{ner, action, language, task}``.

        Returns the record with an added ``rxn_validation`` block and, when a
        conservative patch is accepted, a ``reactant_side_participants`` list.
        Temp/Time/Y/Action are never modified.
        """
        result = deepcopy(record)

        # 1. Lightweight field-constraint checks (advisory, no field changes)
        if self.field_constraints_enabled:
            field_constraints = check_constraints(result)
        else:
            field_constraints = {"passed": True, "issues": [], "summary": "constraints disabled"}

        # 2. Build reaction-view and run mapping screening
        reaction_view = self._ner_to_reaction_view(result)
        updated_rxn = self.validate_reaction(reaction_view)
        rxn_val = updated_rxn.get("rxn_validation", {})

        # 3. Reflect accepted patch back into the record (grounding annotation only)
        if "reactant_side_participants" in updated_rxn:
            result["reactant_side_participants"] = updated_rxn["reactant_side_participants"]

        # 4. Structured validation output
        verification = rxn_val.get("verification", {}) or {}
        patch_accepted = bool(verification.get("accepted", False))
        conf_before = rxn_val.get("mapping_confidence_before")
        conf_after = (
            verification.get("mapping_confidence_after")
            if patch_accepted
            else conf_before
        )

        result["rxn_validation"] = {
            "status": rxn_val.get("status", "needs_review"),
            "diagnosis": rxn_val.get("diagnosis", ""),
            "reason": rxn_val.get("reason", ""),
            "needs_human_review": rxn_val.get("needs_human_review", True),
            "field_constraints": field_constraints,
            "mapping_confidence_before": conf_before,
            "mapping_confidence_after": conf_after,
            "mapping_confidence_best": rxn_val.get("mapping_confidence_best"),
            "confidence_delta": rxn_val.get("confidence_delta"),
            "base_expression": rxn_val.get("base_expression", ""),
            "best_expression": rxn_val.get("best_expression", ""),
            "patch": rxn_val.get("patch", []),
            "patch_accepted": patch_accepted,
            "candidates": rxn_val.get("candidates", []),
            "normalized_compounds": rxn_val.get("normalized_compounds", {}),
        }
        return result

    # ── internal: NER record → reaction-view expected by normalize.py ──
    @staticmethod
    def _ner_to_reaction_view(record: Dict[str, Any]) -> Dict[str, Any]:
        ner = record.get("ner", {}) or {}
        action = record.get("action", []) or []

        def _as_list(field: str) -> List[Any]:
            value = ner.get(field, [])
            if value is None:
                return []
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                return [value] if value.strip() else []
            return [value]

        reaction: Dict[str, Any] = {
            "reactants": _as_list("STARTING_MATERIAL"),
            "products": _as_list("REACTION_PRODUCT"),
            "reagent_catalyst": _as_list("REAGENT_CATALYST"),
            "solvents": _as_list("SOLVENT"),
            "actions": " ".join(action) if isinstance(action, list) else str(action or ""),
        }
        # Carry original text for context-aware reagent selection when available.
        text = record.get("text") or record.get("reference")
        if text:
            reaction["reference"] = text
        return reaction

    # ── internal: mapping screening on a reaction-view dict ──
    def validate_reaction(self, reaction: Dict[str, Any]) -> Dict[str, Any]:
        normalized = normalize_reaction(reaction)

        if self.scorer is None:
            scored_candidates: List[Any] = []
            diagnosis = Diagnosis(
                diagnosis="mapping_screening_unavailable",
                status="skipped",
                needs_human_review=True,
                reason="rxnmapper_not_available",
                base_candidate=None,
                best_candidate=None,
            )
            patches: List[Dict[str, Any]] = []
        else:
            candidates = build_candidates(
                reaction,
                normalized,
                max_reagent_candidates=self.max_reagent_candidates,
            )
            scored_candidates = self.scorer.score(candidates)
            diagnosis = diagnose_candidates(
                scored_candidates,
                confidence_threshold=self.confidence_threshold,
                confidence_delta=self.confidence_delta,
                ambiguity_margin=self.ambiguity_margin,
            )
            patches = make_patch(diagnosis)

        updated = deepcopy(reaction)
        verification = {
            "accepted": False,
            "reason": "no_patch_generated",
            "mapping_confidence_before": (
                diagnosis.base_candidate.confidence if diagnosis.base_candidate else None
            ),
            "mapping_confidence_after": None,
        }

        if patches and self.scorer is not None:
            updated, accepted, verification = verify_patch(
                reaction,
                patches,
                diagnosis,
                self.scorer,
                confidence_threshold=self.confidence_threshold,
                confidence_delta=self.confidence_delta,
                max_reagent_candidates=self.max_reagent_candidates,
            )
            if accepted:
                status = "corrected"
                needs_review = False
            else:
                status = "needs_review"
                needs_review = True
        else:
            status = diagnosis.status
            needs_review = diagnosis.needs_human_review

        updated["rxn_validation"] = {
            "status": status,
            "diagnosis": diagnosis.diagnosis,
            "needs_human_review": needs_review,
            "reason": diagnosis.reason,
            "mapping_confidence_before": (
                diagnosis.base_candidate.confidence if diagnosis.base_candidate else None
            ),
            "mapping_confidence_best": (
                diagnosis.best_candidate.confidence if diagnosis.best_candidate else None
            ),
            "base_expression": (
                diagnosis.base_candidate.expression if diagnosis.base_candidate else ""
            ),
            "best_expression": (
                diagnosis.best_candidate.expression if diagnosis.best_candidate else ""
            ),
            "confidence_delta": (
                (
                    diagnosis.best_candidate.confidence - diagnosis.base_candidate.confidence
                )
                if diagnosis.best_candidate and diagnosis.base_candidate
                else None
            ),
            "candidates": [candidate.to_public_dict() for candidate in scored_candidates],
            "patch": patches,
            "verification": verification,
            "normalized_compounds": {
                role: [compound.to_public_dict() for compound in compounds]
                for role, compounds in normalized.items()
            },
        }
        return updated
