"""Cross-check product structures against MS data quoted in the source text.

The v7 review (P-05) found a product SMILES whose exact mass contradicted the
MS value printed in the very chunk it came from (C32H31N3O6, M+1 = 554.23 vs
"MS(m/e):479.5(M+1)"): the name-resolution cascade happily returns a structure
for an LLM-guessed name, and nothing ever compared it to the原文表征数据.
This module closes that gap conservatively: it only flags when a single
discrete product matches NONE of the quoted MS values, and it never mutates
structures.

Legacy patent chunks routinely merge several example blocks, so the text usually
contains peaks for compounds other than the product; the check therefore
passes when ANY quoted peak is consistent with the product and flags only
when every interpretable peak contradicts it. Peaks whose adduct cannot be
interpreted ("[M-K]+", "[M-tBu]+", "[M+2H]2+") are skipped entirely —
guessing an adduct for them caused mass false positives in the v8 run.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
except ImportError:  # pragma: no cover - rdkit is a hard pipeline dependency
    Chem = None
    rdMolDescriptors = None

_PROTON_MASS = 1.007276
_SODIUM_ADDUCT = 22.989218

# "MS(m/e):479.5(M+1)." / "MS (m/z) 554.2 [M+H]+" / "m/z 479.5 (M+1)".
# The bracketed adduct annotation is captured whole so unrecognized adducts
# can be skipped instead of being silently read as the molecular ion.
_MS_VALUE = re.compile(
    r"m\s*/\s*(?:e|z)\)?\s*[:=]?\s*(\d{2,4}(?:\.\d+)?)"
    r"(?:\s*[\[(（]([^\])）]*)[\])）])?",
    re.I)

# Salt/solvate products legitimately mismatch the quoted ion mass.
_NON_DISCRETE_NAME = re.compile(
    r"(?:salt|hydrochloride|hcl\b|hemifumarate|fumarate|tosylate|maleate|"
    r"mesylate|methanesulfonate|sulfate|hydrate|solvate|monohydrate|"
    r"盐|水合物|溶剂化|钠盐|钾盐|盐酸盐|富马酸盐|甲磺酸盐|对甲苯磺酸盐)", re.I)

# Recognized adduct annotations → mass shift from the neutral exact mass.
_ION_MASS_SHIFT = {
    "": 0.0,
    "m": 0.0,
    "m+": _PROTON_MASS,
    "m+1": _PROTON_MASS,
    "m+h": _PROTON_MASS,
    "m+h+": _PROTON_MASS,
    "m-1": -_PROTON_MASS,
    "m-h": -_PROTON_MASS,
    "m-h+": -_PROTON_MASS,
    "m+na": _SODIUM_ADDUCT,
    "m+na+": _SODIUM_ADDUCT,
}
# A peak with no adduct annotation is ambiguous: in patents it is typically
# the quasi-molecular ion (M, [M+H]+ or [M+Na]+, depending on mode).
_UNTAGGED_SHIFTS = (0.0, _PROTON_MASS, -_PROTON_MASS, _SODIUM_ADDUCT)
_TOLERANCE_DA = 0.5


def _plain_ms(text):
    # Track original offsets through OCR/LaTeX normalization.
    chars = list(text or "")
    offsets = list(range(len(chars)))
    pattern = re.compile(r"\\(?:mathrm|text|mathbf|operatorname)\s*\{|\\(?:left|right)|\\[,;! ]|[$^_{}]")
    for match in reversed(list(pattern.finditer(''.join(chars)))):
        del chars[match.start():match.end()]
        del offsets[match.start():match.end()]
    translation = str.maketrans({'⁺': '+', '⁻': '-', '−': '-', '（': '(', '）': ')', '２': '2'})
    return ''.join(chars).translate(translation), offsets


def extract_ms_references(text: str) -> List[Dict[str, Any]]:
    """Read both MS orders and preserve each original source span."""
    plain, offsets = _plain_ms(text)
    pattern = re.compile(
        r"(?:(?:m\s*/\s*[ez]\)?\s*[:=]?\s*|(?:\b(?:ESI|HR|LC|APCI|EI)[- ]?MS|\bMS)(?:\s*\([^)]{0,24}\))?\s*[:=]\s*)(?P<forward>\d{2,4}(?:\.\d+)?)|"
        r"(?P<reverse>\d{2,4}(?:\.\d+)?)\s*m\s*/\s*[ez])"
        r"(?:\s*[\[(](?P<ion>[^\])]*?)[\])](?P<charge>\s*\d*[+-])?)?", re.I)
    references = []
    for match in pattern.finditer(plain):
        observed = float(match.group('forward') or match.group('reverse'))
        raw_ion = match.group('ion')
        charge = (match.group('charge') or '').strip()
        if charge and any(c.isdigit() and c != '1' for c in charge):
            continue
        ion = re.sub(r"\s", "", raw_ion or '').lower()
        isotope_label = ion in ('m+2', 'm+3', 'm+4')
        if raw_ion is None:
            shifts = list(_UNTAGGED_SHIFTS)
        elif ion == 'm':
            shifts = [0.0, _PROTON_MASS]
        elif isotope_label:
            shifts = [_PROTON_MASS]
        elif ion in _ION_MASS_SHIFT:
            shifts = [_ION_MASS_SHIFT[ion]]
        else:
            continue
        a, b = offsets[match.start()], offsets[match.end() - 1] + 1
        references.append({'observed': observed, 'ion': ion or '(untagged)',
                           'tagged': raw_ion is not None, 'shifts': shifts,
                           'isotope_label': isotope_label,
                           'span': [a, b], 'source_text': (text or '')[a:b]})
    # Peptide patents also report "MS (M+H+): expected ..., observed ...".
    # Use the explicitly observed value, never the calculated/expected mass.
    labeled = re.compile(r'\b(?:MS|(?:ESI|HR|LC|APCI|EI)[- ]?MS)\s*[\[(](?P<ion>[^\])]{1,24})[\])]'
                         r'\s*[:=]\s*(?:expected[^;\n。]{0,100}?,\s*)?'
                         r'(?:observed|found)\s*[:=]?\s*(?P<mass>\d{2,4}(?:\.\d+)?)', re.I)
    for match in labeled.finditer(plain):
        ion = re.sub(r'\s', '', match['ion']).lower()
        if ion not in _ION_MASS_SHIFT:
            continue
        a, b = offsets[match.start()], offsets[match.end() - 1] + 1
        references.append({'observed': float(match['mass']), 'ion': ion, 'tagged': True,
                           'shifts': [_ION_MASS_SHIFT[ion]], 'isotope_label': False,
                           'span': [a,b], 'source_text': (text or '')[a:b]})
    return references


def _isotope_offsets(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # Exact isotope substitutions, bounded by the number of available atoms.
    counts = {symbol: sum(a.GetSymbol() == symbol and not a.GetIsotope()
                          for a in mol.GetAtoms()) for symbol in ('Br', 'Cl')}
    return [i * 1.9979521 + j * 1.9970499
            for i in range(counts['Br'] + 1) for j in range(counts['Cl'] + 1)]


def _product_exact_mass(smiles: str) -> Optional[float]:
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        return None
    return rdMolDescriptors.CalcExactMolWt(mol)


def evaluate_product_mass(text, products):
    """Caller must supply only the selected step's product characterization."""
    candidates = [p for p in products or [] if p.get('status') == 'valid' and p.get('canonical_smiles')]
    if len(candidates) != 1:
        return {'status': 'unknown', 'references': [], 'issue': None}
    product = candidates[0]
    if _NON_DISCRETE_NAME.search(str(product.get('name') or '')):
        return {'status': 'unknown', 'references': [], 'issue': None}
    exact = _product_exact_mass(product['canonical_smiles'])
    if exact is None:
        return {'status': 'unknown', 'references': [], 'issue': None}
    references = extract_ms_references(text)
    if not references:
        return {'status': 'not_reported_or_uninterpretable', 'references': [], 'issue': None}
    isotopes = _isotope_offsets(product['canonical_smiles'])
    closest = None
    for reference in references:
        # [M+2] needs a physically possible isotope rather than a generic +2 Da.
        offsets = [x for x in isotopes if x > 0] if reference['isotope_label'] else isotopes
        for isotope in offsets:
            for shift in reference['shifts']:
                delta = exact + isotope + shift - reference['observed']
                if abs(delta) <= _TOLERANCE_DA:
                    return {'status': 'matched', 'references': references, 'exact_mass': exact,
                            'matched_reference': reference, 'isotope_offset': isotope, 'issue': None}
                if closest is None or abs(delta) < abs(closest[0]):
                    closest = (delta, reference, shift + isotope)
    if closest is None:
        return {'status': 'unknown', 'references': references, 'issue': None}
    delta, reference, shift = closest
    issue = {'field': 'REACTION_PRODUCT', 'type': 'ms_mass_mismatch',
             'message': f"product exact mass {exact:.2f} vs text MS m/z {reference['observed']} ({reference['ion']}): expected {exact + shift:.2f}, off by {delta:.2f} Da"}
    return {'status': 'mismatch', 'references': references, 'exact_mass': exact, 'issue': issue}


def check_product_mass(text, products):
    """Compatibility entrypoint; None means no contradiction, not positive proof."""
    return evaluate_product_mass(text, products)['issue']
