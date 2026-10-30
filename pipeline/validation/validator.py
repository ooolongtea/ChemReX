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
from .ms_check import evaluate_product_mass
from .atom_sources import check_atom_sources
from .quality import evaluate_quality
from .normalize import normalize_reaction, valid_mapping_compounds
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
        mapping_scorer=None,
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
        self.scorer = mapping_scorer
        try:
            if self.scorer is None:
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

        # 2b. Cross-check the product structure against MS data quoted in the
        # source text. A "passed" mapping built on a structure that contradicts
        # the原文表征数据 is a fabrication (v7 review P-05): downgrade it.
        mass_check = evaluate_product_mass(
            result.get("ms_source_text", result.get("text") or result.get("reference") or ""),
            (rxn_val.get("normalized_compounds") or {}).get("products") or [],
        )
        ms_issue = mass_check["issue"]
        if ms_issue is not None:
            field_constraints = dict(field_constraints)
            field_constraints["issues"] = list(field_constraints.get("issues") or []) + [ms_issue]
            field_constraints["summary"] = f"{len(field_constraints['issues'])} constraint issue(s)"
            field_constraints["passed"] = False
            if rxn_val.get("status") in ("passed", "corrected"):
                rxn_val = dict(rxn_val, status="needs_review", diagnosis="ms_mass_mismatch",
                               reason=ms_issue["message"], needs_human_review=True)

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
            "effective_expression": rxn_val.get("effective_expression", ""),
            "effective_mapped_rxn": rxn_val.get("effective_mapped_rxn", ""),
            "atom_source_check": rxn_val.get("atom_source_check", {}),
            "mass_check": mass_check,
        }
        result["quality"] = evaluate_quality(result)
        return result

    @staticmethod
    def _identity_structures(normalized: Dict[str, Any]) -> bool:
        left = {c.canonical_smiles for role in ("reactants", "participants")
                for c in valid_mapping_compounds(normalized.get(role, []))}
        products = {c.canonical_smiles for c in valid_mapping_compounds(
            normalized.get("products", []))}
        return bool(left and products and products.issubset(left))

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
            "process_kind": record.get("process_kind", "synthesis"),
        }
        # Carry original text for context-aware reagent selection when available.
        text = record.get("text") or record.get("reference")
        if text:
            reaction["reference"] = text
        return reaction

    # ── internal: mapping screening on a reaction-view dict ──
    def validate_reaction(self, reaction: Dict[str, Any]) -> Dict[str, Any]:
        normalized = normalize_reaction(reaction)

        kind = reaction.get("process_kind", "synthesis")
        if kind in ("assay", "analysis", "purification", "formulation"):
            scored_candidates = []
            diagnosis = Diagnosis(
                diagnosis="non_reaction_context", status="non_reaction",
                needs_human_review=kind == "formulation", reason=kind,
            )
            patches = []
        elif self._identity_structures(normalized):
            scored_candidates = []
            diagnosis = Diagnosis(
                diagnosis="identity_reaction", status="non_reaction",
                needs_human_review=False, reason="product_structure_is_a_reactant",
            )
            patches = []
        elif not reaction.get("products"):
            scored_candidates = []
            diagnosis = Diagnosis(
                diagnosis="no_product", status="needs_review",
                needs_human_review=True, reason="no_extracted_product",
            )
            patches = []
        elif not reaction.get("reactants"):
            # Name the actual defect: "structure_unavailable" from the empty
            # candidate set below read as if structures failed to resolve,
            # while the extraction simply never produced a starting material
            # (v7 review P-08, 04档 example_2/3/5).
            scored_candidates = []
            diagnosis = Diagnosis(
                diagnosis="missing_starting_material", status="skipped",
                needs_human_review=True, reason="starting_material_is_empty",
            )
            patches = []
        elif self.scorer is None:
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
            try:
                scored_candidates = self.scorer.score(candidates)
            except ValueError as exc:
                # RXNMapper's transformer caps input at max_position_embeddings
                # (512 tokens): very large molecules (peptides, polymers) can
                # never be atom-mapped. Skip mapping for them instead of
                # failing the whole chunk (deterministic retry loop otherwise).
                if "should be at most" not in str(exc):
                    raise
                scored_candidates = []
                diagnosis = Diagnosis(
                    diagnosis="reaction_too_large_for_mapping",
                    status="skipped",
                    needs_human_review=True,
                    reason=str(exc)[:200],
                    base_candidate=None,
                    best_candidate=None,
                )
                patches = []
            else:
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

        effective_expression = (verification.get("base_expression_after") if verification.get("accepted")
                                else diagnosis.base_candidate.expression if diagnosis.base_candidate else "")
        effective_mapped = (verification.get("mapped_rxn_after") if verification.get("accepted")
                            else diagnosis.base_candidate.mapped_rxn if diagnosis.base_candidate else "")
        if verification.get("accepted"):
            normalized = normalize_reaction(updated)
        updated["rxn_validation"] = {
            "effective_expression": effective_expression,
            "effective_mapped_rxn": effective_mapped,
            "atom_source_check": check_atom_sources(effective_expression, effective_mapped),
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
