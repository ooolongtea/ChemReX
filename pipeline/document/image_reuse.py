"""Retain valid image evidence while repairing missing standalone molecules."""
from copy import deepcopy
from pathlib import Path
import json
import os
import tempfile

from pipeline.chemical_quality import valid_image_smiles
from pipeline.versions import IMAGE_PIPELINE_VERSION

LEGACY_IMAGE_VERSION = '2026-09-28-v3'


def known_image_defects(uuid):
    path = os.getenv('CHEMREX_SOURCE_DEFECTS_FILE')
    if not path:
        return []
    return [d for d in json.loads(Path(path).read_text()).get('image_defects', []) if d['uuid'] == uuid]


def reject_confirmed_candidates(payload, defects):
    """Retain source-proven model errors as evidence, never as usable structures."""
    updated = deepcopy(payload)
    rejected = []
    for defect in defects:
        bad = set(defect['rejected_smiles'])
        name = defect['image_file']
        for key in defect['rejected_dict_keys']:
            if updated.get('dict_result', {}).get(key) in bad:
                updated['dict_result'].pop(key)
        for entry in updated.get('per_image_detections', []):
            if entry['image_file_name'] != name:
                continue
            for molecule in entry.get('molecule_boxes', []):
                if molecule.get('smiles') in bad:
                    rejected.append(dict(image_file=name, smiles=molecule['smiles'], reason=defect['reason']))
                    molecule['rejected_smiles'] = molecule['smiles']
                    molecule['smiles'] = None
                    molecule['structure_rejection'] = 'confirmed_source_conflict'
            entry['dict_result'] = {k:v for k,v in entry.get('dict_result', {}).items() if v not in bad}
        for entry in updated.get('per_image_detections', []):
            if entry['image_file_name'] != name:
                continue
            for key in defect['rejected_dict_keys']:
                candidate = entry.get('dict_result', {}).get(key)
                if candidate and (not bad or candidate in bad):
                    rejected.append(dict(image_file=name, dict_key=key, smiles=candidate,
                                         reason=defect['reason'], rejection_kind='identifier_assignment'))
                    entry['dict_result'].pop(key)
                    if updated.get('dict_result', {}).get(key) == candidate:
                        updated['dict_result'].pop(key)
        for entry in updated.get('image_result', []):
            if entry['image_file_name'] == name:
                entry['reactions'] = [r for r in entry.get('reactions', []) if not any(
                    s in str(r.get('reactants', '')).split(';') + str(r.get('products', '')).split(';') for s in bad)]
    if rejected:
        updated.setdefault('confirmed_structure_rejections', []).extend(rejected)
    return updated, rejected


def repair_missing_molecules(payload, image_base, process, image_names=None, known_defects=None):
    previous_version = payload.get('image_pipeline_version')
    payload, rejected = reject_confirmed_candidates(payload, known_defects or [])
    if payload.get('image_pipeline_version') != LEGACY_IMAGE_VERSION and not rejected:
        return None
    base = Path(image_base)
    if not base.is_dir():
        raise FileNotFoundError(f'Image cache directory is missing: {base}')
    paths = {p.name: p for p in base.rglob('*') if p.suffix.lower() in
             {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}}
    entries = {r['image_file_name']: r for r in payload.get('per_image_detections', [])}
    missing = {name for name in paths if not any(valid_image_smiles(m.get('smiles'))
               for m in entries.get(name, {}).get('molecule_boxes', []))}
    missing.update(r['image_file'] for r in rejected if r['image_file'] in paths)
    selected = missing if image_names is None else missing.intersection(image_names)
    updated = deepcopy(payload)
    if selected:
        with tempfile.TemporaryDirectory(prefix='chemrex_missing_molecules_') as folder:
            for name in sorted(selected):
                (Path(folder) / name).symlink_to(paths[name].resolve())
            fresh = process(folder, return_per_image=True)
            fresh, _ = reject_confirmed_candidates(fresh, known_defects or [])
        if fresh.get('failed_images') or fresh.get('image_failures'):
            raise RuntimeError('Image repair failed: ' + str(fresh.get('failed_images') or fresh.get('image_failures')))
        returned = {r['image_file_name'] for r in fresh.get('per_image_detections', [])}
        if not selected.issubset(returned):
            raise RuntimeError('Image repair did not return all requested images')
        updated['per_image_detections'] = [r for r in updated.get('per_image_detections', [])
                                         if r['image_file_name'] not in selected] + fresh['per_image_detections']
        updated['image_result'] = [r for r in updated.get('image_result', [])
                                  if r['image_file_name'] not in selected] + fresh.get('image_result', [])
        conflicts = set()
        for key, smiles in fresh.get('dict_result', {}).items():
            previous = updated.setdefault('dict_result', {}).get(key)
            if previous and previous != smiles:
                conflicts.add(key)
            else:
                updated['dict_result'][key] = smiles
        for key in conflicts:
            updated['dict_result'].pop(key, None)
    updated['image_reuse'] = {'previous_version': previous_version,
        'reprocessed_images': sorted(selected), 'retained_images': len(paths) - len(selected),
        'partial': selected != missing}
    if selected == missing:
        updated['image_pipeline_version'] = IMAGE_PIPELINE_VERSION
    return updated
