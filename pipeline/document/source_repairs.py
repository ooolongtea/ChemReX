"""Field repairs supported by the current step's literal experimental text."""
import re
from copy import deepcopy

from pipeline.validation.quality import reaction_condition_text, _condition_norm


def repair_local_fields(ner, text):
    ner = deepcopy(ner)
    repairs = []
    body = reaction_condition_text(text, workup=True)
    patterns = {
        'TEMPERATURE': r'(?:冷却|降温|加热|升温)(?:到|至|到达)?\s*'
                       r'(-?\d+(?:\.\d+)?\s*(?:℃|°\s*C))',
        'TIME': r'(?:预搅拌|搅拌|保持|保温|反应)\s*'
                r'(\d+(?:\.\d+)?\s*(?:分钟|小时|秒钟))',
    }
    for field, pattern in patterns.items():
        for match in re.finditer(pattern, body, re.I):
            value = match[1]
            existing = re.split(r'[;；]', str(ner.get(field) or ''))
            if any(_condition_norm(value, field) == _condition_norm(v, field) for v in existing):
                continue
            ner[field] = ';'.join(v for v in [ner.get(field), value] if v)
            repairs.append(dict(field=field, value=value, source_text=match[0],
                                reason='explicit_local_condition'))
    # DMF can be a reagent (e.g. formylation). Remove only a duplicated solvent
    # with an explicit volume-only source amount and no reactive-role cue.
    aliases = {'dmf', 'n,n-二甲基甲酰胺', '二甲基甲酰胺', 'n,n-dimethylformamide', 'dimethylformamide'}
    def name(value):
        return value.get('name', '') if isinstance(value, dict) else value
    def is_dmf(value):
        return re.sub(r'\s+', '', str(name(value))).lower() in aliases
    if (any(is_dmf(v) for v in ner.get('SOLVENT') or [])
            and not re.search(r'催化|试剂|甲酰化|维尔斯迈尔|Vilsmeier|formylat|reagent', body, re.I)):
        kept = []
        for value in ner.get('REAGENT_CATALYST') or []:
            mentions = list(re.finditer(re.escape(str(name(value))), body, re.I)) if is_dmf(value) else []
            if mentions and all(re.match(r'\s*[（(]\s*\d+(?:\.\d+)?\s*(?:mL|L|毫升|升)\s*[）)]',
                                         body[m.end():], re.I) for m in mentions):
                repairs.append(dict(field='REAGENT_CATALYST', value=name(value),
                                    reason='volume_only_dmf_already_in_solvent'))
            else:
                kept.append(value)
        ner['REAGENT_CATALYST'] = kept
    return ner, repairs


def repair_nitrile_structures(ner, text):
    """Resolve 苄腈 ambiguity only when paired g/mmol evidence selects a structure.

    Benzyl cyanide is a legitimate meaning of 苄腈. Never replace that suffix
    globally. A unique aryl-CH2-CN candidate may become aryl-CN only if the
    source molecular weight supports the latter and contradicts the former.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from pipeline.validation.quality import _norm
    body = _norm(text)
    for field in ('STARTING_MATERIAL', 'REACTION_PRODUCT', 'REAGENT_CATALYST'):
        for compound in ner.get(field) or []:
            if not isinstance(compound, dict) or not str(compound.get('name', '')).endswith('苄腈'):
                continue
            smiles = compound.get('smiles')
            if not isinstance(smiles, str):
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            pattern = re.escape(_norm(compound['name'])) + r'\((\d+(?:\.\d+)?)(mg|g),'
            pattern += r'(\d+(?:\.\d+)?)(mmol|mol)(?:[,，)]|当量)'
            amounts = list(re.finditer(pattern, body))
            if len(amounts) != 1 or float(amounts[0][3]) <= 0:
                continue
            amount = amounts[0]
            reported = float(amount[1]) * (0.001 if amount[2] == 'mg' else 1)
            reported /= float(amount[3]) * (0.001 if amount[4] == 'mmol' else 1)
            if reported <= 0:
                continue
            mass = Descriptors.MolWt(mol)
            if abs(mass - reported) / reported <= .02:
                continue
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts('[N]#[C]-[CH2]-[c]'))
            corrected = None
            if len(matches) == 1:
                _, nitrile, methylene, aryl = matches[0]
                edit = Chem.RWMol(mol)
                edit.AddBond(nitrile, aryl, Chem.BondType.SINGLE)
                edit.RemoveAtom(methylene)
                candidate = edit.GetMol()
                Chem.SanitizeMol(candidate)
                if abs(Descriptors.MolWt(candidate) - reported) / reported <= .02:
                    corrected = Chem.MolToSmiles(candidate)
            evidence = dict(reported_mw=reported, original_mw=mass, source_text=amount[0],
                            original_smiles=smiles)
            compound['source_amount_check'] = evidence
            compound['rejected_structure_smiles'] = smiles
            compound.pop('cxsmiles', None)
            compound.pop('canonical_smiles', None)
            if corrected:
                compound['smiles'] = corrected
                original_name = compound.get('standard_name')
                compound['original_standard_name'] = original_name
                match = re.fullmatch(r'2-\((.+)phenyl\)acetonitrile', original_name or '')
                compound['standard_name'] = match[1] + 'benzonitrile' if match else None
                compound['structure_source'] = 'source_amount_supported_nitrile'
                compound['resolution_status'] = 'resolved'
                evidence.update(status='corrected', corrected_smiles=corrected)
            else:
                compound['smiles'] = None
                compound['structure_rejection'] = 'source_amount_mass_mismatch'
                compound['resolution_status'] = 'conflicting_structure'
                evidence['status'] = 'mismatch'
