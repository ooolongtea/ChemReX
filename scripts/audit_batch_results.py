"""Step-aligned, latest-result audit. Read-only unless --output-dir is given."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.document.results import latest_by_id
from pipeline.extraction.parser import parse_json_output
from rdkit import Chem


def audit_document(task):
    folder = Path(task['output_dir'])
    chunks = [json.loads(l) for l in (folder / 'chunks.jsonl').read_text().splitlines() if l.strip()]
    latest = latest_by_id(folder / 'results.jsonl', {c['id']: c['chunk_fingerprint'] for c in chunks})
    delivery = [json.loads(l) for l in (folder / 'delivery/reactions.jsonl').read_text().splitlines() if l.strip()]
    by_key = {(d['source']['chunk_id'], d['source']['reaction_index']): d for d in delivery}
    counts, statuses, issues, comparisons = Counter(), Counter(), Counter(), []
    counts.update(chunks=len(chunks), current_results=len(latest), delivery_records=len(delivery))
    for record in latest.values():
        statuses[record['status']] += 1
        for rx in record.get('reactions') or []:
            counts['step_records'] += 1
            ex = rx.get('extraction') or {}
            raw = ex.get('raw') or {}
            actions = parse_json_output(raw.get('action', ''))
            if isinstance(actions, list) and actions:
                counts['action_raw_nonempty'] += 1
                if not ex.get('action'):
                    counts['action_nonempty_rejected'] += 1
            dv = by_key.get((record['id'], rx['index']))
            comparisons.append({'chunk_id': record['id'], 'reaction_index': rx['index'],
                                'raw_ner': raw.get('ner'), 'parsed_ner': ex.get('ner'),
                                'delivered_conditions': dv.get('conditions') if dv else None})
    for dv in delivery:
        issues.update(dv['quality']['blocking_issues'])
        if dv['quality']['admission_eligible']:
            counts['admission_eligible'] += 1
        expression = dv.get('reaction_smiles')
        if expression:
            counts['reaction_smiles_present'] += 1
            counts['delivery_with_smiles'] += 1
            sides = expression.split('>')
            valid = len(sides) == 3 and bool(sides[0] and sides[2]) and all(
                (m := Chem.MolFromSmiles(s)) is not None and m.GetNumAtoms() > 0 for s in (sides[0], sides[2]))
            counts['reaction_smiles_valid' if valid else 'reaction_smiles_invalid'] += 1
    from scripts.verify_delivery import verify_document
    verification_error = None
    try:
        control = verify_document(folder)
    except Exception as exc:
        verification_error = str(exc)
        control = {}
    return {'uuid': task.get('uuid', folder.name), 'output_dir': str(folder), 'counts': dict(counts),
            'chunk_status': dict(statuses), 'blocking_issues': dict(issues),
            'processing_status': control.get('processing_status'), 'verification_error': verification_error,
            'comparisons': comparisons}


def main(argv=None, default_docs=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docs', default=default_docs, required=default_docs is None)
    parser.add_argument('--output-dir', help='Write NEW audit files here; otherwise print only')
    parser.add_argument('--no-write', action='store_true', help='Print only, even if an output directory is supplied')
    args = parser.parse_args(argv)
    documents = json.loads(Path(args.docs).read_text())
    results, failures, total = [], [], Counter()
    for task in documents:
        try:
            result = audit_document(task)
            total.update(result['counts'])
            results.append(result)
            if result['verification_error']:
                failures.append({'uuid': result['uuid'], 'error': result['verification_error']})
        except Exception as exc:
            failures.append({'uuid': task.get('uuid'), 'error': str(exc)})
    summary = {'documents': len(documents), 'verified': sum(not r['verification_error'] for r in results), 'counts': dict(total), 'failures': failures}
    if args.output_dir and not args.no_write:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name, value in [('summary.json', summary), ('documents.json', results)]:
            (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
