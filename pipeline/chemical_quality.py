"""Conservative OCR quality gates for image-derived structures."""

import re

try:
    from rdkit import Chem
except ImportError:
    Chem = None

_FAKE_ATOMS = re.compile(r"\[(?:Rf|Re|Fr|Cf|Rh|Rg|As)\]", re.I)
_GARBLED_CONDITIONS = re.compile(r"\b(?:Me2SO|EOHH|MCe\.OH|COOAC|1\.78c22h)\b", re.I)
_ISOMER_LABEL = re.compile(r'\s*[（(]\s*(?:异构体|isomer)\s*(\d+)\s*[）)]\s*$', re.I)


def split_isomer_label(value):
    """Separate a trailing sample label, preserving chemical stereo descriptors."""
    name = str(value or '').strip()
    match = _ISOMER_LABEL.search(name)
    if not match or not name[:match.start()].strip():
        return name, None
    return name[:match.start()].rstrip(), match[0].strip()


def unspecified_structure_name(value):
    """An explicit class/placeholder is not a specified molecular structure."""
    name = re.sub(r'\s+', ' ', str(value or '')).strip()
    return bool(re.fullmatch(
        r'(?:(?:所需|相应|目标)(?:的)?|粗(?:制)?)*(?:酮|醛|胺|酰胺|醇|酯|烯烃|酸|腈|二腈|腙|肼盐|'
        r'盐酸盐|氢溴酸盐|硫酸盐|频哪醇酯|卤化物衍生物|卤代衍生物|化合物|标题化合物|产物)|'
        r'(?:the\s+)?(?:(?:desired|corresponding|target|crude)\s+)*(?:ketone|aldehyde|amine|amide|alcohol|ester|alkene|acid|'
        r'nitrile|dinitrile|hydrazone|hydrazine salt|hydrochloride salt|hydrobromide salt|sulfate salt|'
        r'pinacol ester|boronate(?: ester)?|boronic acid|halide derivative|title compound|compound|product)',
        name, re.I))


def racemic_structure_name(value):
    return bool(re.search(r'外消旋|\bracemic\b|\bracemate\b|\brac-', str(value or ''), re.I))


def valid_image_smiles(value: str, allow_markush: bool = False) -> bool:
    if not value or value.strip() in ("*", "[C]", "[Pd]") or _FAKE_ATOMS.search(value):
        return False
    if Chem is None:
        return False
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return False
    # A bare single atom from OCR ("[Cd]", "[Co]" — table row numbers and
    # margin marks mis-detected as molecules, v7 review P-09) is never a
    # usable structure, and open-shell species are never valid participants.
    if mol.GetNumAtoms() == 1:
        return False
    # A wildcard atom ([1*], *) marks a Markush/general-scheme structure: a
    # legitimate image-side identifier for scheme figures, but never a
    # concrete molecule. Admit it only when the caller explicitly allows a
    # Markush dictionary (identifier→structure mapping); reaction validation
    # and product assignment keep the strict gate. Even then the entry must
    # be a real scaffold with wildcard positions ("[1*]c1cnc(Cl)nc1Cl"),
    # never a bare placeholder like "*=*" ("R alkene").
    if allow_markush:
        if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            if sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1) < 3:
                return False
        if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
            return False
        return True
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return False
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        return False
    return True


def _geometry_label_candidates(item):
    boxes = item.get("molecule_boxes") or []
    labels = item.get("figure_labels") or []
    for link in item.get("label_mol_links") or []:
        label_idx, mol_idx = link.get("label_idx"), link.get("mol_idx")
        if type(label_idx) is not int or type(mol_idx) is not int:
            continue
        if not (0 <= label_idx < len(labels) and 0 <= mol_idx < len(boxes)):
            continue
        value = str(boxes[mol_idx].get("smiles") or "")
        if value:
            yield labels[label_idx].get("text"), value


def _resolve_label_candidates(candidates):
    values, invalid = {}, set()
    for key, value in candidates:
        key = normalize_image_label(key)
        if not key:
            continue
        value = str(value or '')
        if not valid_image_smiles(value, allow_markush=True):
            invalid.add(key)
            continue
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(value), isomericSmiles=True)
        values.setdefault(key, {}).setdefault(canonical, value)
    conflicting = {key for key, choices in values.items() if len(choices) > 1}
    mapping = {key: next(iter(choices.values())) for key, choices in values.items()
               if key not in invalid | conflicting}
    return {'dict_result': mapping,
            'markush_entries': {key: value for key, value in mapping.items()
                                if not valid_image_smiles(value)},
            'conflicting_identifiers': sorted(conflicting),
            'unusable_identifiers': sorted(invalid | conflicting)}


def image_label_evidence(items):
    """Resolve all dictionary/geometry candidates before losing duplicate labels."""
    def candidates():
        for item in items:
            yield from (item.get('dict_result') or {}).items()
            yield from _geometry_label_candidates(item)
    return _resolve_label_candidates(candidates())


def markush_label_smiles(item: dict) -> dict:
    """Unambiguous geometry mappings, including real Markush scaffolds."""
    return _resolve_label_candidates(_geometry_label_candidates(item))['dict_result']


def normalize_image_label(value: str) -> str:
    label = " ".join(str(value or "").split())
    if not label or not any(ch.isalnum() for ch in label):
        return ""
    # NOTE: pure numbers stay valid labels — scheme images legitimately label
    # compounds "1", "2" (tests: "1"→CCO). Garbage numeric entries are cut by
    # the SMILES gate (single atoms like "[Cd]" from table row numbers).
    # Roman numeral OCR commonly confuses uppercase I and lowercase l, at any
    # position: "(XVIll)"→"(XVIII)" and "(XIlc)"→"(XIIc)" are both observed.
    match = re.fullmatch(r"(\(?)([XVIl]+)([a-z]?)(\)?)", label)
    if match and "l" in match.group(2):
        label = "".join((match.group(1), match.group(2).replace("l", "I"),
                         match.group(3), match.group(4)))
    return label


def condition_quality(value: str) -> str:
    return "low_confidence" if _GARBLED_CONDITIONS.search(value or "") else "ok"


def valid_reaction_sides(reactants: list, products: list) -> bool:
    """Require parseable, non-placeholder structures and a changed product side."""
    if not reactants or not products:
        return False
    if not all(valid_image_smiles(str(item)) for item in reactants + products):
        return False
    left = {Chem.MolToSmiles(Chem.MolFromSmiles(item), canonical=True)
            for item in reactants}
    right = {Chem.MolToSmiles(Chem.MolFromSmiles(item), canonical=True)
             for item in products}
    return bool(right - left)
