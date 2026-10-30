"""Document-scoped source evidence for confirmed name/structure conflicts."""
from collections import Counter
import json
import os
from pathlib import Path
import re


def compound_evidence_for_source(source):
    path = os.getenv('CHEMREX_SOURCE_DEFECTS_FILE')
    if not path or not source:
        return []
    parts = Path(str(source)).parts
    return [d for d in json.loads(Path(path).read_text()).get('compound_defects', []) if d['uuid'] in parts]


def compound_conflict(compound, evidence):
    if not isinstance(compound, dict) or not compound.get('smiles'):
        return None
    name = re.sub(r'\s+', '', compound.get('name', '')).lower()
    for fact in evidence:
        if name != re.sub(r'\s+', '', fact['name']).lower():
            continue
        from rdkit import Chem
        mol = Chem.MolFromSmiles(compound['smiles'])
        actual = Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum()!=1) if mol else Counter()
        expected = {element:int(count or '1') for element,count in
                    re.findall(r'([A-Z][a-z]?)(\d*)', fact['formula']) if element!='H'}
        if actual != expected:
            return dict(name=compound.get('name'), reason='source_name_formula_conflict',
                        source_evidence=fact, expected_heavy_atoms=expected,
                        candidate_heavy_atoms=dict(actual), rejected_smiles=compound['smiles'])
    return None


def reject_source_conflicts(coref, evidence):
    conflicts = []
    for field in ('STARTING_MATERIAL','REACTION_PRODUCT','REAGENT_CATALYST','SOLVENT'):
        for compound in (coref.get('ner') or {}).get(field) or []:
            conflict = compound_conflict(compound, evidence)
            if conflict:
                conflicts.append(dict(conflict, field=field))
                compound['rejected_structure_smiles'] = compound['smiles']
                compound['smiles'] = None
                compound['structure_rejection'] = conflict['reason']
                compound['resolution_status'] = 'source_conflict'
                compound.setdefault('resolution_trace', []).append(dict(service='source_evidence',
                    query=compound.get('name'), status='source_conflict', reason=conflict['reason']))
    if conflicts:
        coref['source_conflicts'] = conflicts
    return conflicts
