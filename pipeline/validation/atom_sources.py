"""Necessary carbon-source checks, independent of mapping confidence."""
from collections import Counter
import re
from rdkit import Chem

_HALOGEN_DONORS = {
    Chem.MolToSmiles(Chem.MolFromSmiles(smiles)): (element, count)
    for smiles, element, count in (
        ('[I-].[Na+]', 'I', 1), ('[I-].[K+]', 'I', 1),
        ('O=C1CCC(=O)N1Cl', 'Cl', 1), ('O=C1CCC(=O)N1Br', 'Br', 1),
        ('O=C1CCC(=O)N1I', 'I', 1), ('ClCl', 'Cl', 2), ('BrBr', 'Br', 2), ('II', 'I', 2),
    )
}

# Maximum transferred groups, never the full reagent/catalyst inventory.
_GROUP_DONORS = {
    Chem.MolToSmiles(Chem.MolFromSmiles(smiles)): contribution
    for smiles, contribution in (
        ('CC(C)(C)OC(=O)OC(=O)OC(C)(C)C', {'C': 5, 'O': 2}),
        ('CS(=O)(=O)[O-].[Na+]', {'C': 1, 'S': 1, 'O': 2}),
        ('CS(=O)[O-].[Na+]', {'C': 1, 'S': 1, 'O': 2}),
        ('FC(F)CI', {'C': 2, 'F': 2}),
        ('NN.O', {'N': 2}), ('NN', {'N': 2}),
    )
}


def _fragments_without_maps(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Counter(Chem.MolToSmiles(f) for f in Chem.GetMolFrags(mol, asMols=True))


def check_heavy_atom_sources(expression, record=None):
    """Necessary elemental check; stoichiometry and chemistry remain unproven.

    Credit explicit source-supported group donors, without treating arbitrary
    solvents or catalyst ligands as carbon sources. This does not prove mapping.
    """
    carbon = check_atom_sources(expression)
    try:
        left, right = expression.split('>>')
        if not left or not right or not all(Chem.MolFromSmiles(s) is not None for s in (left, right)):
            raise ValueError('invalid expression')
    except (ValueError, AttributeError, TypeError):
        return {'status': 'unknown', 'element_deficit': {}, 'explicit_water_source': False, 'necessary_check_only': True}
    deficit = dict(carbon['element_deficit'])
    record = record or {}
    evidence = record.get('field_evidence') or {}
    ner = record.get('ner') or {}
    water = False
    for value in ner.get('SOLVENT') or []:
        name = value.get('name', '') if isinstance(value, dict) else str(value)
        water_name = name.lower().strip().translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
        if water_name in ('water', 'h2o', '水') and any(
                e.get('status') in ('local', 'inherited') and e.get('extracted_value', e.get('value')) == name
                for e in evidence.get('SOLVENT', [])):
            water = True
    if water and deficit.get('O'):
        deficit['O'] = max(0, deficit['O'] - 1)
    reagent_sources = []
    credited_donors = set()
    left_components = _fragments_without_maps(left)
    from .normalize import _FORM_COMPONENTS, named_discrete_form, normalize_compound
    for reagent in ner.get('REAGENT_CATALYST') or []:
        name = reagent.get('name', '') if isinstance(reagent, dict) else str(reagent)
        supported = any(e.get('status') in ('local', 'inherited')
                        and e.get('extracted_value', e.get('value')) == name
                        for e in evidence.get('REAGENT_CATALYST') or [])
        if not supported:
            continue
        donor = normalize_compound(reagent, 'reagents', 0)
        if donor.status != 'valid' or donor.canonical_smiles in credited_donors:
            continue
        credited_donors.add(donor.canonical_smiles)
        # Effective expressions may already include a mapped reagent repair.
        if not (_fragments_without_maps(donor.canonical_smiles) - left_components):
            continue
        identity = _HALOGEN_DONORS.get(donor.canonical_smiles) if donor.status == 'valid' else None
        if identity:
            element, count = identity
            credited = min(count, max(0, deficit.get(element, 0)))
            if credited:
                deficit[element] -= credited
                reagent_sources.append(dict(element=element, count=credited, reagent=name))
        for element, count in _GROUP_DONORS.get(donor.canonical_smiles, {}).items():
            credited = min(count, max(0, deficit.get(element, 0)))
            if credited:
                deficit[element] -= credited
                reagent_sources.append(dict(element=element, count=credited, reagent=name))
    # Named, validated salts contain disconnected counterions whose donor
    # often appears under REAGENT_CATALYST, not on the mapped substrate side.
    # Credit only new counterion components and source-attributed donors.
    # Carbon still requires the existing mapped carbon-source check.
    counterion_sources = []
    acid_names = {
        'Cl': r'HCl|氯化氢|盐酸|hydrogen chloride|hydrochloric acid',
        'Br': r'HBr|溴化氢|氢溴酸|hydrogen bromide|hydrobromic acid',
        'I': r'HI|碘化氢|氢碘酸|hydrogen iodide|hydroiodic acid',
    }
    def fragments(smiles):
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Counter(Chem.MolToSmiles(f) for f in Chem.GetMolFrags(mol, asMols=True))
    left_fragments, right_fragments = fragments(left), fragments(right)
    credited_families = Counter()
    for product in ((record.get('rxn_validation') or {}).get('normalized_compounds') or {}).get('products') or []:
        smiles = product.get('canonical_smiles') or product.get('smiles')
        if product.get('status') != 'valid' or not smiles or not named_discrete_form(
                product.get('name', ''), smiles, product.get('discrete_form_evidence')):
            continue
        product_fragments = fragments(smiles)
        if any(count > right_fragments[fragment] for fragment, count in product_fragments.items()):
            continue
        alkali_form = re.search(r'\bsodium\b|\bpotassium\b|\bcalcium\b|钠盐|钾盐|钙盐', product.get('name', ''), re.I)
        for form_pattern, equivalents in _FORM_COMPONENTS:
            if not re.search(form_pattern, product.get('name', ''), re.I):
                continue
            forms = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in equivalents}
            if alkali_form and any(a.GetAtomicNum() == 6 for s in forms for a in Chem.MolFromSmiles(s).GetAtoms()):
                continue
            introduced = max(0, sum(right_fragments[s] - left_fragments[s] for s in forms)
                             - credited_families[form_pattern])
            components = [(s, min(count, introduced)) for s, count in product_fragments.items()
                          if s in forms and introduced]
            if not components:
                continue
            for reagent in ner.get('REAGENT_CATALYST') or []:
                name = reagent.get('name', '') if isinstance(reagent, dict) else str(reagent)
                supported = any(e.get('status') in ('local', 'inherited')
                                and e.get('extracted_value', e.get('value')) == name
                                for e in evidence.get('REAGENT_CATALYST') or [])
                if not supported:
                    continue
                donor = normalize_compound(reagent, 'reagents', 0)
                matches = donor.status == 'valid' and donor.canonical_smiles in forms
                # A reported aqueous hydrohalic acid is a real solution, not
                # a single molecule. Its halide identity is still explicit.
                for element, acid in acid_names.items():
                    pattern = (r'\s*(?:\d+(?:\.\d+)?\s*(?:%|N|M)\s*)?(?:' + acid
                               + r')(?:\s*(?:水溶液|(?:aqueous\s+)?solution|aq\.?))?\s*')
                    matches = matches or (forms == {element, '[' + element + '-]'}
                                          and bool(re.fullmatch(pattern, name, re.I)))
                if not matches:
                    continue
                for component, count in components:
                    count = min(count, introduced)
                    atoms = Counter(a.GetSymbol() for a in Chem.MolFromSmiles(component).GetAtoms())
                    for element, amount in atoms.items():
                        credited = min(count * amount, max(0, deficit.get(element, 0))) if element not in ('C', 'H') else 0
                        if credited:
                            deficit[element] -= credited
                            counterion_sources.append(dict(element=element, count=credited, reagent=name))
                    introduced -= count
                    credited_families[form_pattern] += count
                break
    missing = {e: n for e, n in deficit.items() if e != 'H' and n > 0}
    return {'status': 'incomplete' if missing else 'unknown' if carbon['status'] == 'unknown' and not expression else 'complete',
            'element_deficit': missing, 'explicit_water_source': water,
            'counterion_sources': counterion_sources, 'reagent_sources': reagent_sources,
            'necessary_check_only': True}


def check_atom_sources(expression, mapped_rxn=""):
    try:
        left, right = expression.split(">>")
        mols = [Chem.MolFromSmiles(side) for side in (left, right)]
        if not all(mols):
            raise ValueError("invalid reaction sides")
        counts = [Counter(a.GetSymbol() for a in mol.GetAtoms()) for mol in mols]
        deficit = {e: n - counts[0][e] for e, n in counts[1].items()
                   if n > counts[0][e]}
        missing = []
        mapping_known = False
        if mapped_rxn:
            ml, mr = [Chem.MolFromSmiles(s) for s in mapped_rxn.split(">>")]
            if ml is not None and mr is not None:
                left_maps = {a.GetAtomMapNum(): a.GetAtomicNum() for a in ml.GetAtoms()
                             if a.GetAtomMapNum()}
                carbons = [a for a in mr.GetAtoms() if a.GetAtomicNum() == 6]
                mapping_known = bool(carbons) and all(a.GetAtomMapNum() for a in carbons)
                missing = [a.GetAtomMapNum() for a in carbons
                           if left_maps.get(a.GetAtomMapNum()) != 6]
        incomplete = bool(deficit.get("C") or (mapping_known and missing))
        return {"status": "incomplete" if incomplete else
                "complete" if mapping_known else "unknown",
                "element_deficit": deficit, "missing_carbon_maps": missing,
                "mapping_known": mapping_known, "necessary_check_only": True}
    except (ValueError, TypeError, AttributeError):
        return {"status": "unknown", "element_deficit": {},
                "missing_carbon_maps": [], "mapping_known": False,
                "necessary_check_only": True}


def contributes_carbon(candidate):
    """The added component must actually supply mapped product carbon."""
    if not candidate.added_compound or not candidate.mapped_rxn:
        return False
    try:
        left, right = candidate.mapped_rxn.split(">>")
        product = Chem.MolFromSmiles(right)
        product_maps = {a.GetAtomMapNum() for a in product.GetAtoms()
                        if a.GetAtomicNum() == 6 and a.GetAtomMapNum()}
        target = candidate.added_compound.canonical_smiles
        for fragment in left.split("."):
            mol = Chem.MolFromSmiles(fragment)
            if mol is None:
                continue
            maps = {a.GetAtomMapNum() for a in mol.GetAtoms()
                    if a.GetAtomicNum() == 6 and a.GetAtomMapNum()}
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            # Ionic partners can contain counterions: compare the carbon fragment.
            target_mol = Chem.MolFromSmiles(target)
            fragments = Chem.GetMolFrags(target_mol, asMols=True)
            if any(Chem.MolToSmiles(f) == Chem.MolToSmiles(mol) for f in fragments):
                if maps & product_maps:
                    return True
    except (ValueError, AttributeError):
        pass
    return False
