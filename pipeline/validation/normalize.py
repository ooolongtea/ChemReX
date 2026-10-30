"""RDKit-backed structure normalization for reaction validation."""

from __future__ import annotations

from copy import deepcopy
import re
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
    "[C]",               # activated carbon OCR is not elemental carbon reagent
    "[Pd]",              # palladium catalyst
    "*",                 # image/OCR placeholder
    "O",                  # water
    "O.O",                # hydrate-like water entries
    "[N]",                # nitrogen sweep gas from sloppy OCR/name lookup
    "N#N",                # nitrogen atmosphere (inert auxiliary)
    "[Ar]",               # argon atmosphere
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

# Explicit named forms, not a general exemption for disconnected products.
_FORM_COMPONENTS = (
    (r'hydrochloride|\bchloride\b|盐酸盐|氯化', ('Cl', '[Cl-]')),
    (r'hydrobromide|\bbromide\b|氢溴酸盐|溴化', ('Br', '[Br-]')),
    (r'hydroiodide|\biodide\b|氢碘酸盐|碘化', ('I', '[I-]')),
    (r'mesylate|methanesulfonate|甲磺酸盐', ('CS(=O)(=O)O', 'CS(=O)(=O)[O-]')),
    (r'sulfate|sulphate|硫酸盐', ('OS(=O)(=O)O', 'O=S(=O)([O-])[O-]', 'O=S(=O)(O)[O-]')),
    (r'\bacetate\b|乙酸盐|醋酸盐', ('CC(=O)O', 'CC(=O)[O-]')),
    (r'trifluoroacetate|三氟乙酸盐|三氟醋酸盐', ('O=C(O)C(F)(F)F', 'O=C([O-])C(F)(F)F')),
    (r'fumarate|富马酸盐', ('O=C(O)/C=C/C(=O)O', 'O=C([O-])/C=C/C(=O)[O-]', 'O=C(O)/C=C/C(=O)[O-]')),
    (r'maleate|马来酸盐', ('O=C(O)/C=C\\C(=O)O', 'O=C([O-])/C=C\\C(=O)[O-]', 'O=C(O)/C=C\\C(=O)[O-]')),
    (r'\bsodium\b|钠盐', ('[Na+]',)),
    (r'\bpotassium\b|钾盐', ('[K+]',)),
    (r'\bcalcium\b|钙盐', ('[Ca+2]',)),
    (r'hydrate|水合物', ('O',)),
    (r'ethanol\s+solvate|乙醇溶剂合物', ('CCO',)),
    (r'methanol\s+solvate|甲醇溶剂合物', ('CO',)),
)


def _form_formula_matches(mol, evidence):
    from collections import Counter
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    formula = evidence.get('parent_formula') or ''
    if not re.fullmatch(r'(?:[A-Z][a-z]?\d*)+', formula):
        return False
    expected = Counter({e:int(n or '1') for e,n in re.findall(r'([A-Z][a-z]?)(\d*)', formula)})
    components = {'HCl': dict(H=1, Cl=1), 'HBr': dict(H=1, Br=1), 'H2O': dict(H=2, O=1)}
    for item in evidence.get('components') or []:
        for element, count in components[item['component']].items():
            expected[element] += count * item['count']
    actual = Counter({e:int(n or '1') for e,n in re.findall(r'([A-Z][a-z]?)(\d*)', CalcMolFormula(mol))})
    return actual == expected


def named_discrete_form(name, canonical, evidence=None):
    """One parent plus only explicitly named counterions/solvate components."""
    # "Ethyl acetate" and "ethyl chloride" name covalent molecules. Their
    # final word cannot authorize an extra acid/halide as a counterion.
    if re.fullmatch(r'(?:di)?(?:methyl|ethyl|propyl|isopropyl|butyl|tert[- ]butyl|benzyl)\s+'
                    r'(?:acetate|trifluoroacetate|fumarate|maleate|sulfate|sulphate|chloride|bromide|iodide)',
                    str(name).strip(), re.I):
        return False
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return False
    fragments = Chem.GetMolFrags(mol, asMols=True)
    if len(fragments) < 2 or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        return False
    allowed = {Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
               for pattern, smiles in _FORM_COMPONENTS if re.search(pattern, name, re.I)
               for s in smiles}
    if evidence and evidence.get('status') == 'consistent' and evidence.get('parent_formula'):
        equivalents = {'HCl': {'Cl', '[Cl-]'}, 'HBr': {'Br', '[Br-]'}, 'H2O': {'O'}}
        counts = evidence.get('components') or []
        if not all(c.get('component') in equivalents and isinstance(c.get('count'), (int, float))
                   and float(c['count']).is_integer() and c['count'] >= 0 for c in counts):
            return False
        for component in counts:
            allowed.update(equivalents[component['component']])
    if re.search(r'\bsodium\b|\bpotassium\b|\bcalcium\b|钠盐|钾盐|钙盐', name, re.I):
        # In sodium acetate, acetate is the parent, not another counterion.
        allowed = {s for s in allowed if not any(a.GetAtomicNum() == 6
                   for a in Chem.MolFromSmiles(s).GetAtoms())}
    values = [Chem.MolToSmiles(f, canonical=True) for f in fragments]
    parents = [f for f, value in zip(fragments, values) if value not in allowed]
    if evidence and evidence.get('status') == 'consistent' and evidence.get('parent_formula'):
        if len(parents) != 1 or not _form_formula_matches(mol, evidence):
            return False
        if not all(sum(s in equivalents[c['component']] for s in values) == c['count'] for c in counts):
            return False
    return (bool(allowed) and len(parents) == 1 and len(parents) < len(fragments)
            and parents[0].GetNumHeavyAtoms() > 1)


def name_composition_conflict(name, smiles):
    """Reject the demonstrated dictionary homonym, without inventing a structure.

    CN104507943B, p.113: 3.5 g / 9.8 mmol of 三溴化吡啶鎓.
    Pyridinium hydrogen tribromide is C5H6Br3N (PubChem CID 10990802),
    whereas CID 21124087 is three pyridinium bromides, C15H18Br3N3.
    The bare alias returned the latter during the real acceptance run.
    """
    # Hydrohalide names explicitly require a separate halide component.
    # A free base cannot stand in for its named salt; di/tri counts are not
    # inferred from the number of basic nitrogen atoms.
    salt = re.search(r'(?:(di|tri|tetra|mono)?hydro(chloride|bromide|iodide)|'
                     r'(一|二|三|四)?(盐酸盐|氢溴酸盐|氢碘酸盐))', str(name), re.I)
    if salt:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            component = {'chloride':'Cl','bromide':'Br','iodide':'I',
                         '盐酸盐':'Cl','氢溴酸盐':'Br','氢碘酸盐':'I'}[(salt[2] or salt[4]).lower()]
            count = sum(f.GetNumAtoms() == 1 and f.GetAtomWithIdx(0).GetSymbol() == component
                        for f in Chem.GetMolFrags(mol, asMols=True))
            explicit = {'mono':1,'di':2,'tri':3,'tetra':4,'一':1,'二':2,'三':3,'四':4}.get(
                        (salt[1] or salt[3] or '').lower())
            if count == 0:
                return 'named_salt_component_missing'
            if explicit is not None and count != explicit:
                return 'name_structure_composition_conflict'
    if re.fullmatch(r'pyridinium\s+(?:hydrogen\s+)?tribromide|三溴化吡啶鎓', str(name).strip(), re.I):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            from collections import Counter
            counts = Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
            if counts != Counter(C=5, N=1, Br=3):
                return 'name_structure_composition_conflict'
    return None


def elemental_form_evidence(text, smiles):
    """Check explicit analyzed salt/solvate ratios; never round a bulk ratio."""
    from pipeline.document.segmenter import _de_latex
    plain = _de_latex(re.sub(r'\\cdot', '·', text or ''))
    analysis = re.search(r'(?:元素分析|elemental analysis)\s*[:：]?\s*(.{0,240}?)'
                         r'(?:Calc|计算|Found|实测|\n|$)', plain, re.I)
    if not analysis:
        return None
    components = [(name, float(amount or '1')) for amount, name in re.findall(
        r'[·•∙]\s*(\d+(?:\.\d+)?)?\s*(HCl|HBr|H2O)\b', analysis[1], re.I)]
    if not components:
        return None
    formula = re.search(r'(?:[A-Z][a-z]?\d*){2,}', re.sub(r'\s+', '', analysis[1].split('·')[0]))
    evidence = {'source_text': analysis[0], 'parent_formula': formula[0] if formula else None,
                'components': [dict(component=k.upper() if k.lower() != 'h2o' else 'H2O', count=v)
                               for k,v in components]}
    for item in evidence['components']:
        item['component'] = {'HCL':'HCl','HBR':'HBr'}.get(item['component'], item['component'])
    if any(not count.is_integer() for _, count in components):
        evidence['status'] = 'nonstoichiometric_form'
        return evidence
    mol = Chem.MolFromSmiles(smiles or '')
    fragments = [Chem.MolToSmiles(m, canonical=True) for m in Chem.GetMolFrags(mol, asMols=True)] if mol else []
    equivalents = {'hcl': {'Cl', '[Cl-]'}, 'hbr': {'Br', '[Br-]'}, 'h2o': {'O'}}
    evidence['status'] = 'consistent' if all(sum(s in equivalents[name.lower()] for s in fragments) == count
                                          for name, count in components) else 'salt_solvate_composition_conflict'
    if evidence['status'] == 'consistent' and evidence['parent_formula'] and not _form_formula_matches(mol, evidence):
        evidence['status'] = 'parent_formula_conflict'
    return evidence


def repair_elemental_form(compound, text):
    """Complete explicitly counted neutral salt/solvate components only.

    The parent must match the printed molecular formula exactly. Charged
    representations and enhanced stereo are retained without reconstruction;
    protonation sites and noninteger bulk compositions are never inferred.
    """
    if not isinstance(compound, dict):
        return None
    original = compound.get('smiles')
    evidence = elemental_form_evidence(text, original)
    if not evidence:
        return None
    compound['discrete_form_evidence'] = evidence
    if (evidence['status'] != 'salt_solvate_composition_conflict' or not evidence.get('parent_formula')
            or compound.get('cxsmiles') or compound.get('compound_class') in ('mixture', 'polymer', 'excipient')):
        return evidence
    mol = Chem.MolFromSmiles(original or '')
    if mol is None or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return evidence
    equivalents = {'Cl', 'Br', 'O'}
    fragments = list(Chem.GetMolFrags(mol, asMols=True))
    parents = [f for f in fragments if Chem.MolToSmiles(f) not in equivalents]
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    if len(parents) != 1 or CalcMolFormula(parents[0]) != evidence['parent_formula']:
        return evidence
    requested = {c['component'] for c in evidence['components']}
    if any(Chem.MolToSmiles(f) in equivalents and {'Cl':'HCl','Br':'HBr','O':'H2O'}[Chem.MolToSmiles(f)] not in requested
           for f in fragments):
        return evidence
    additions = {'HCl':'Cl', 'HBr':'Br', 'H2O':'O'}
    repaired = '.'.join([Chem.MolToSmiles(parents[0]), *[additions[c['component']]
                        for c in evidence['components'] for _ in range(int(c['count']))]])
    repaired_evidence = elemental_form_evidence(text, repaired)
    if not named_discrete_form(compound.get('name', ''), repaired, repaired_evidence):
        return evidence
    compound['smiles'] = repaired
    compound['discrete_form_evidence'] = repaired_evidence
    compound['structure_form_repair'] = dict(original_smiles=original, source='elemental_analysis',
                                            source_text=evidence['source_text'])
    compound.setdefault('resolution_trace', []).append(dict(service='source_elemental_form',
        query=evidence['parent_formula'], status='explicit_components_completed'))
    return repaired_evidence


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
    fragment_count: int = 1

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
            "fragment_count": self.fragment_count,
            "structure_source": self.original.get("structure_source", "unknown"),
            "cxsmiles": self.original.get("cxsmiles"),
            "stereochemistry": self.original.get("stereochemistry"),
            "resolution_trace": self.original.get("resolution_trace"),
            "resolution_status": self.original.get("resolution_status"),
            "discrete_form_evidence": self.original.get("discrete_form_evidence"),
            "structure_form_repair": self.original.get("structure_form_repair"),
            "structure_rejection": self.original.get("structure_rejection"),
            "rejected_structure_smiles": self.original.get("rejected_structure_smiles"),
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

    from pipeline.chemical_quality import unspecified_structure_name
    if (unspecified_structure_name(name) and not original.get('image_source')
            and (not smiles or original.get('structure_source') == 'name_resolution')):
        # Reject old name-service guesses too, while retaining the candidate
        # as evidence. A separately grounded image can specify the structure.
        original['resolution_status'] = 'unspecified_structure'
        original['structure_rejection'] = 'unspecified_structure'
        if smiles:
            original['rejected_structure_smiles'] = smiles
        return NormalizedCompound(name=name, smiles=smiles, canonical_smiles='',
            source_field=source_field, source_index=source_index, original=original,
            status='structure_unavailable', reason='unspecified_structure')

    if not smiles:
        return NormalizedCompound(
            name=name,
            smiles="",
            canonical_smiles="",
            source_field=source_field,
            source_index=source_index,
            original=original,
            status="structure_unavailable",
            reason=(original.get('structure_rejection') or
                    ("known_non_discrete_material" if original.get("compound_class")
                    in ("polymer", "mixture", "excipient") else "unresolved_structure")),
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

    if any(atom.GetAtomicNum() == 0 for atom in Chem.MolFromSmiles(canonical).GetAtoms()):
        return NormalizedCompound(name=name, smiles=smiles, canonical_smiles=canonical,
            source_field=source_field, source_index=source_index, original=original,
            status='structure_unavailable', reason='unknown_atom_placeholder')

    from pipeline.chemical_quality import racemic_structure_name
    if racemic_structure_name(name) and '@' in canonical:
        enhanced = Chem.MolFromSmiles(original.get('cxsmiles') or '')
        groups = [g for g in enhanced.GetStereoGroups() if g.GetGroupType() == Chem.StereoGroupType.STEREO_AND] if enhanced is not None else []
        centers = {a.GetIdx() for a in enhanced.GetAtoms()
                   if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED} if enhanced is not None else set()
        represented = (len(groups) == 1 and bool(centers)
                       and {a.GetIdx() for a in groups[0].GetAtoms()} == centers
                       and Chem.MolToSmiles(enhanced) == canonical)
        if not represented:
            original['rejected_structure_smiles'] = smiles
            return NormalizedCompound(name=name, smiles=smiles, canonical_smiles=canonical,
                source_field=source_field, source_index=source_index, original=original,
                status='structure_unavailable', reason='racemic_stereochemistry_unrepresented')

    conflict = name_composition_conflict(name, canonical) or name_composition_conflict(
        original.get('standard_name', ''), canonical)
    form_status = (original.get('discrete_form_evidence') or {}).get('status')
    if form_status in {'nonstoichiometric_form', 'salt_solvate_composition_conflict', 'parent_formula_conflict'}:
        conflict = form_status
    if conflict:
        return NormalizedCompound(name=name, smiles=smiles, canonical_smiles=canonical,
                                  source_field=source_field, source_index=source_index,
                                  original=original, status='structure_unavailable', reason=conflict)

    # RDKit accepts open-shell SMILES ("[N]" radical nitrogen from a bad
    # name lookup, metal fragments like "CC(=O)O.[Na]" with a lone electron);
    # such structures are never chemically valid reaction participants
    # (v7 review P-06).
    try:
        radical_electrons = sum(atom.GetNumRadicalElectrons()
                                for atom in Chem.MolFromSmiles(canonical).GetAtoms())
    except Exception:
        radical_electrons = 0
    if radical_electrons:
        return NormalizedCompound(
            name=name,
            smiles=smiles,
            canonical_smiles=canonical,
            source_field=source_field,
            source_index=source_index,
            original=original,
            status="structure_unavailable",
            reason=f"open_shell_radical_smiles({radical_electrons})",
        )

    category = str(original.get("compound_class") or "").lower()
    ambiguous = bool(re.search(r"(?:\b(?:or|mixture|composition|tablets?|solution)\b|或|混合物|组合物|片剂|溶液)",
                               name, re.I))
    multicomponent_product = (source_field == "products" and "." in canonical
                             and not named_discrete_form(name, canonical, original.get('discrete_form_evidence')))
    ignored = canonical in IGNORED_CANONICAL_SMILES or bool(original.get("auxiliary"))
    fragment_count = len(Chem.GetMolFrags(Chem.MolFromSmiles(canonical)))
    if category in ("mixture", "polymer", "excipient") or ambiguous or multicomponent_product:
        return NormalizedCompound(
            name=name, smiles=smiles, canonical_smiles=canonical,
            source_field=source_field, source_index=source_index, original=original,
            status="structure_unavailable", reason="mixture_or_non_discrete_compound",
            fragment_count=fragment_count,
        )
    return NormalizedCompound(
        name=name,
        smiles=smiles,
        canonical_smiles=canonical,
        source_field=source_field,
        source_index=source_index,
        original=original,
        ignored=ignored,
        reason="ignored_auxiliary_structure" if ignored else "",
        fragment_count=fragment_count,
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
