"""Find genuinely affected DONE documents and verify their repaired deliveries."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.document.results import latest_by_id
from pipeline.document.segmenter import load_blocks, segment_blocks, select_procedure_chunks
from pipeline.document.step_context import split_steps
from pipeline.extraction.parser import parse_action_output, parse_ner_records
from pipeline.validation.quality import source_metadata, evaluate_quality
from pipeline.validation.ms_check import evaluate_product_mass
from rdkit import RDLogger


def rows(path):
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as con:
        con.row_factory = sqlite3.Row
        return [dict(r, db=str(Path(path).resolve())) for r in con.execute("SELECT * FROM pdf_task WHERE status='DONE' ORDER BY id")]


def read_lines(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []


def shape(chunks):
    return [(c['id'], c.get('title'), c.get('text'), c.get('process_kind'), c.get('block_indices'),
             [(s['title'], s['text'], [(r['id'], r['status']) for r in s.get('procedure_references', [])])
              for s in c.get('steps') or split_steps(c)]) for c in chunks]


def affected(task):
    folder = Path(task['output_dir'])
    old = read_lines(folder / 'chunks.jsonl')
    latest = latest_by_id(folder / 'results.jsonl', {c['id']: c['chunk_fingerprint'] for c in old})
    marker = folder / 'pdf_prepared/parse_result.json'
    payload = json.loads(marker.read_text()) if marker.is_file() else {}
    source = payload.get('content_list_file') or (old[0].get('source') if old else None)
    if not source:
        paths = list((folder / 'pdf_prepared').rglob('*content_list.json'))
        if len(paths) != 1:
            raise ValueError('no unambiguous cached PDF content list')
        source = str(paths[0])
    blocks = load_blocks(source)
    new = select_procedure_chunks(blocks, source, segment_blocks(blocks, source))
    reasons = set()
    if shape(old) != shape(new):
        reasons.add('segmentation_title_classification_or_reference_changed')
    for record in latest.values():
        if record.get('reactions') and (record.get('source') or {}).get('process_kind') == 'formulation':
            reasons.add('formulation_previously_extracted')
        for rx in record.get('reactions') or []:
            ex = rx.get('extraction') or {}
            raw = ex.get('raw') or {}
            if parse_action_output(raw.get('action', '')) != (ex.get('action') or []):
                reasons.add('action_parser_changed')
            parsed = parse_ner_records(raw.get('ner', '')) if raw.get('ner') else []
            index = rx.get('record_index', 1) - 1
            if index < len(parsed) and parsed[index].get('YIELD_PERCENT') != (ex.get('ner') or {}).get('YIELD_PERCENT'):
                reasons.add('yield_parser_changed')
            st = rx.get('step_source') or {}
            info = source_metadata(deepcopy(ex.get('ner') or {}), st.get('local_text', ''), st, ex.get('action'))['yield_info']
            validation = rx.get('validation') or {}
            old_yield = validation.get('yield_info') or (rx.get('coref') or {}).get('yield_info') or {}
            if info.get('kind') != old_yield.get('kind') or info.get('reported_values') != old_yield.get('reported_values'):
                reasons.add('yield_evidence_changed')
            products = (validation.get('rxn_validation') or {}).get('normalized_compounds', {}).get('products') or []
            mass = evaluate_product_mass(st.get('local_text', ''), products)
            if mass['status'] != (validation.get('rxn_validation') or {}).get('mass_check', {}).get('status') and mass['references']:
                reasons.add('mass_evidence_changed')
            if validation:
                q = evaluate_quality(validation)
                before = set((rx.get('quality') or {}).get('blocking_issues') or [])
                relevant = {'local_time_not_preserved', 'local_temperature_not_preserved', 'heavy_atom_source_incomplete', 'salt_form_unverified', 'ms_mass_mismatch'}
                if (set(q['blocking_issues']) & relevant) != (before & relevant):
                    reasons.add('condition_or_structure_evidence_changed')
            for layer in ('coref', 'validation'):
                for values in ((rx.get(layer) or {}).get('ner') or {}).values():
                    if isinstance(values, list) and any(isinstance(c, dict) and '\n' in str(c.get('naming_raw') or '') for c in values):
                        reasons.add('naming_evidence_contaminated')
            if any(i in (rx.get('quality') or {}).get('blocking_issues', []) for i in ('source_temperature_not_extracted', 'source_time_not_extracted', 'source_yield_not_extracted')):
                reasons.add('source_fields_require_reextraction')
    return dict(task, reasons=sorted(reasons), old_chunks=len(old), new_chunks=len(new)) if reasons else None


def scan(dbs, output):
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')
    tasks, errors, counts = [], [], Counter()
    for db in dbs:
        for n, task in enumerate(rows(db), 1):
            counts['scanned_done'] += 1
            try:
                result = affected(task)
                if result:
                    tasks.append(result)
                    counts.update(result['reasons'])
            except Exception as exc:
                errors.append({'uuid': task['uuid'], 'db': task['db'], 'error': str(exc)})
            if n % 25 == 0:
                print(json.dumps({'db': Path(db).name, 'scanned': n, 'affected': len(tasks), 'errors': len(errors)}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'tasks': tasks, 'errors': errors, 'counts': dict(counts)}
    (output / 'affected.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for db in dbs:
        ids = [t['uuid'] for t in tasks if t['db'] == str(Path(db).resolve())]
        (output / (Path(db).stem + '-ids.json')).write_text(json.dumps(ids, indent=2) + '\n')
    print(json.dumps({'affected': len(tasks), 'errors': errors, 'counts': dict(counts)}, ensure_ascii=False), flush=True)
    return 1 if errors else 0


def backup(manifest, output):
    for task in manifest['tasks']:
        destination = output / Path(task['db']).stem / task['uuid']
        if (destination / 'task.json').is_file():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        folder = Path(task['output_dir'])
        for name in ('chunks.jsonl', 'results.jsonl', 'manifest.json', 'summary.json', 'links.json',
                     'document_result.json', 'final_product.json', 'source_identity.json', 'delivery'):
            src, dst = folder / name, destination / name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif src.is_file():
                shutil.copy2(src, dst)
        (destination / 'task.json').write_text(json.dumps(task, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'backed_up': len(manifest['tasks']), 'backup_root': str(output)}))
    return 0


def verify(manifest, output):
    from scripts.audit_batch_results import audit_document
    results, errors = [], []
    for task in manifest['tasks']:
        try:
            with sqlite3.connect(Path(task['db']).resolve().as_uri() + '?mode=ro', uri=True) as con:
                row = con.execute('SELECT status, error FROM pdf_task WHERE uuid=?', (task['uuid'],)).fetchone()
            if not row or row[0] != 'DONE':
                raise ValueError(f'rerun task not DONE: {row}')
            result = audit_document(task)
            if result['verification_error']:
                raise ValueError(result['verification_error'])
            folder = Path(task['output_dir'])
            summary = json.loads((folder / 'summary.json').read_text())
            if summary.get('segmentation_version') != '2026-10-07-auditfix':
                raise ValueError('document was not re-extracted with the fix')
            if result['chunk_status'].get('error') or result['counts']['current_results'] != result['counts']['chunks']:
                raise ValueError('current chunks have errors or missing results')
            result.pop('comparisons', None)
            results.append(result)
        except Exception as exc:
            errors.append({'uuid': task['uuid'], 'db': task['db'], 'error': str(exc)})
    report = {'verified': len(results), 'expected': len(manifest['tasks']), 'errors': errors, 'documents': results}
    before_counts, after_counts, before_issues, after_issues = Counter(), Counter(), Counter(), Counter()
    comparison_errors = []
    for result in results:
        after_counts.update(result['counts'])
        after_issues.update(result['blocking_issues'])
    for task in manifest['tasks']:
        saved = output.parent / 'before' / Path(task['db']).stem / task['uuid']
        if not saved.is_dir():
            continue
        try:
            old = audit_document({'uuid': task['uuid'], 'output_dir': str(saved)})
            before_counts.update(old['counts'])
            before_issues.update(old['blocking_issues'])
        except Exception as exc:
            comparison_errors.append({'uuid': task['uuid'], 'error': str(exc)})
    report['comparison'] = {'before_counts': dict(before_counts), 'after_counts': dict(after_counts),
                            'before_blocking_issues': dict(before_issues), 'after_blocking_issues': dict(after_issues),
                            'comparison_errors': comparison_errors}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 受影响文档重跑验收记录', '',
             f"验证 {len(results)} / {len(manifest['tasks'])} 篇；失败 {len(errors)} 篇。", '',
             '状态和文件验证不等于化学正确；下面统计包含保留但不能自动入库的候选。', '',
             '| 统计项 | 原结果 | 最新结果 |', '| --- | ---: | ---: |']
    for key in ('chunks', 'step_records', 'delivery_records', 'action_nonempty_rejected',
                'reaction_smiles_present', 'reaction_smiles_valid', 'admission_eligible'):
        lines.append(f'| {key} | {before_counts[key]} | {after_counts[key]} |')
    lines.extend(['', '分段和步骤范围已改变，数量变化不能直接解释为新增/修复的化学反应数。', '',
                  f'原结果比较异常 {len(comparison_errors)} 篇；逐篇验收异常见 verification.json.errors。', '',
                  '最新保留的质量问题：', '', '| 原因 | 候选记录数 |', '| --- | ---: |'])
    for key, count in after_issues.most_common():
        lines.append(f'| {key} | {count} |')
    lines.extend(['', '逐篇结果、原/新统计和异常保存在 verification.json；代表样本原始输出和交付在 pilot/。',
                  'OCR 名称、盐型脱盐和模型温度/时间遗漏仍按代码修复记录中的证据限制处理。'])
    output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'verified': len(results), 'expected': len(manifest['tasks']), 'errors': errors}, ensure_ascii=False))
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('scan'); s.add_argument('--db', action='append', required=True); s.add_argument('--output-dir', type=Path, required=True)
    b = sub.add_parser('backup'); b.add_argument('--manifest', type=Path, required=True); b.add_argument('--output-dir', type=Path, required=True)
    v = sub.add_parser('verify'); v.add_argument('--manifest', type=Path, required=True); v.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'scan':
        return scan(args.db, args.output_dir)
    manifest = json.loads(args.manifest.read_text())
    return backup(manifest, args.output_dir) if args.command == 'backup' else verify(manifest, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
