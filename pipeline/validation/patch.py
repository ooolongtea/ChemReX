"""Conservative field-local patches for validation diagnoses."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from .diagnose import Diagnosis


ALLOWED_PATCH_FIELDS = {
    "reactants",
    "starting_materials",
    "starting_material",
    "products",
    "product",
    "reagent_catalyst",
    "reagent_catalysts",
    "reagents",
    "catalysts",
    "reactant_side_participants",
    "grounding_status",
}


def make_patch(diagnosis: Diagnosis) -> List[Dict[str, Any]]:
    if diagnosis.diagnosis != "missing_atom_contributing_reagent":
        return []
    best = diagnosis.best_candidate
    base = diagnosis.base_candidate
    if not best or not base or not best.added_compound:
        return []

    compound = deepcopy(best.added_compound.original)
    compound.setdefault("name", best.added_compound.name)
    compound["smiles"] = best.added_compound.canonical_smiles
    compound["validation_role"] = "reactant_side_participant"
    return [
        {
            "op": "add",
            "path": "/reactant_side_participants/-",
            "value": compound,
            "reason": (
                "Including this reagent on the reactant side increases RXNMapper "
                f"confidence from {base.confidence:.3f} to {best.confidence:.3f}."
            ),
        }
    ]


def _check_patch_allowed(patch: Dict[str, Any]) -> bool:
    path = str(patch.get("path") or "").strip("/")
    if not path:
        return False
    field = path.split("/", 1)[0]
    return field in ALLOWED_PATCH_FIELDS


def apply_patch(reaction: Dict[str, Any], patches: List[Dict[str, Any]]) -> Dict[str, Any]:
    updated = deepcopy(reaction)
    for patch in patches:
        if not _check_patch_allowed(patch):
            raise ValueError(f"Patch touches disallowed field: {patch}")
        op = patch.get("op")
        path = patch.get("path")
        if op != "add" or path != "/reactant_side_participants/-":
            raise ValueError(f"Unsupported validation patch: {patch}")
        updated.setdefault("reactant_side_participants", [])
        value = deepcopy(patch.get("value"))
        existing = {
            (item.get("smiles"), item.get("name"))
            for item in updated["reactant_side_participants"]
            if isinstance(item, dict)
        }
        if isinstance(value, dict) and (value.get("smiles"), value.get("name")) not in existing:
            updated["reactant_side_participants"].append(value)
    return updated
