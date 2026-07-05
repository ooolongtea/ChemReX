"""RDKit-backed structure normalization for reaction validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from rdkit import Chem


COMPOUND_FIELD_ALIASES = {
    "reactants": ("reactants", "starting_materials", "starting_material"),
    "products": ("products", "product"),
    "reagents": ("reagent_catalyst", "reagent_catalysts", "reagents", "catalysts"),
    "solvents": ("solvents", "solvent"),
    "participants": ("reactant_side_participants",),
}

IGNORED_CANONICAL_SMILES = {
    "O",                  # water
    "O.O",                # hydrate-like water entries
    "Cl.[Na+]",           # sodium chloride, alternate ionic forms below
    "[Cl-].[Na+]",
    "[Br-].[Na+]",
    "[I-].[Na+]",
    "[Cl-].[K+]",
    "[Br-].[K+]",
    "[I-].[K+]",
    "O=[Na]",             # sodium oxide / inorganic entries from noisy extraction
    "O=[Mg]",
    "O=S(=O)([O-])[O-].[Na+].[Na+]",
    "O=C([O-])[O-].[Na+].[Na+]",
}


@dataclass
class NormalizedCompound:
    """A compound normalized enough to build reaction SMILES."""

    name: str
    smiles: str
    canonical_smiles: str
    source_field: str
    source_index: int
    original: Dict[str, Any]
    status: str = "valid"
    ignored: bool = False
    reason: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "source_field": self.source_field,
            "source_index": self.source_index,
            "status": self.status,
            "ignored": self.ignored,
            "reason": self.reason,
        }


def get_first_field(reaction: Dict[str, Any], aliases: Iterable[str]) -> Optional[str]:
    for field in aliases:
        if field in reaction:
            return field
    return None


def get_compound_list(reaction: Dict[str, Any], aliases: Iterable[str]) -> List[Any]:
    field = get_first_field(reaction, aliases)
    if not field:
        return []
    value = reaction.get(field)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def canonicalize_smiles(smiles: Optional[str]) -> Optional[str]:
    """Return canonical SMILES, raising no exceptions to callers."""

    if not smiles or not str(smiles).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def normalize_compound(compound: Any, source_field: str, source_index: int) -> NormalizedCompound:
    if isinstance(compound, dict):
        name = str(compound.get("name") or compound.get("standard_name") or "").strip()
        smiles = str(compound.get("smiles") or "").strip()
        original = deepcopy(compound)
    else:
        name = str(compound or "").strip()
        smiles = ""
        original = {"name": name}

    if not smiles:
        return NormalizedCompound(
            name=name,
            smiles="",
            canonical_smiles="",
            source_field=source_field,
            source_index=source_index,
            original=original,
            status="structure_unavailable",
            reason="missing_smiles",
        )

    try:
        canonical = canonicalize_smiles(smiles)
    except Exception as exc:
        return NormalizedCompound(
            name=name,
            smiles=smiles,
            canonical_smiles="",
            source_field=source_field,
            source_index=source_index,
            original=original,
            status="structure_unavailable",
            reason=f"invalid_smiles: {exc}",
        )

    if not canonical:
        return NormalizedCompound(
            name=name,
            smiles=smiles,
            canonical_smiles="",
            source_field=source_field,
            source_index=source_index,
            original=original,
            status="structure_unavailable",
            reason="invalid_smiles",
        )

    ignored = canonical in IGNORED_CANONICAL_SMILES
    return NormalizedCompound(
        name=name,
        smiles=smiles,
        canonical_smiles=canonical,
        source_field=source_field,
        source_index=source_index,
        original=original,
        ignored=ignored,
        reason="ignored_auxiliary_structure" if ignored else "",
    )


def normalize_reaction(reaction: Dict[str, Any]) -> Dict[str, List[NormalizedCompound]]:
    """Normalize compounds from known reaction fields."""

    normalized: Dict[str, List[NormalizedCompound]] = {}
    for role, aliases in COMPOUND_FIELD_ALIASES.items():
        field = get_first_field(reaction, aliases)
        raw_items = get_compound_list(reaction, aliases)
        normalized[role] = [
            normalize_compound(item, field or aliases[0], idx)
            for idx, item in enumerate(raw_items)
        ]
    return normalized


def valid_mapping_compounds(compounds: List[NormalizedCompound]) -> List[NormalizedCompound]:
    return [
        compound for compound in compounds
        if compound.status == "valid" and compound.canonical_smiles and not compound.ignored
    ]
