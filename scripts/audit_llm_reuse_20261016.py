"""Read-only LLM reuse accounting for the fixed production repair scope.

Segmentation and reuse checks run locally; no model clients are constructed.
Planning-dependent steps and retries are reported separately from known calls.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import repair_quality_20261013 as driver


def configuration(language):
    from pipeline.prompts.prompt_loader import PromptLoader
    loader = PromptLoader(prompts_dir=str(ROOT / 'pipeline/extraction/prompts'))
    return dict(model='deepseek-v4.1-flash', preset='huawei-maas', prompts={
        task: dict(version=loader.get_version(f'{task}_{language}'),
                   content=loader.get_content(f'{task}_{language}')) for task in ('ner', 'action')})


def audit(work, output, limit=None, shard_index=0, shards=1, policy='source-checked'):
    from pipeline.document.extraction_reuse import ExtractionReuse
    from pipeline.document.segmenter import load_blocks, segment_blocks, select_procedure_chunks
    from pipeline.document.step_context import extraction_text, split_steps
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    os.environ['CHEMREX_LEGACY_CODE_REVISION'] = '66523a5'
    tasks = json.loads((work / 'affected.json').read_text())['tasks']
    states = {}
    for db in {t['db'] for t in tasks}:
        with sqlite3.connect(Path(db).as_uri()+'?mode=ro', uri=True) as con:
            states[db] = dict(con.execute('select uuid,status from pdf_task'))
    configs = {lang: configuration(lang) for lang in ('en', 'zh')}
    totals, remaining, reasons, errors, documents = Counter(), Counter(), Counter(), [], []
    started = time.time()
    selected = (tasks[:limit] if limit else tasks)[shard_index::shards]
    for n, task in enumerate(selected, 1):
        counts, why = Counter(documents=1), Counter()
        try:
            original = work/'before'/Path(task['db']).stem/task['uuid']
            old_chunks, old_records = driver.records(original)
            reuse = ExtractionReuse(list(old_records.values()), str(original/'results.jsonl'))
            reuse.allow_unversioned = False
            src = driver.source(task, old_chunks)
            blocks = load_blocks(src)
            chunks = select_procedure_chunks(blocks, src, segment_blocks(blocks, src))
            for chunk in chunks:
                if chunk.get('process_kind') in ('assay', 'analysis', 'purification', 'formulation'):
                    counts['non_reaction_chunks'] += 1
                    continue
                for step in chunk.get('steps') or split_steps(chunk):
                    if step.get('process_kind') in ('assay', 'analysis', 'purification', 'formulation'):
                        counts['non_reaction_steps'] += 1
                        continue
                    counts['steps'] += 1
                    if step.get('needs_step_planning'):
                        counts['planning_steps'] += 1
                    strict = reuse.assess(step, task['language'], configs[task['language']])
                    counts['strict_initial_extraction_calls'] += int(strict['needs_ner']) + int(strict['needs_action'])
                    decision = strict
                    if policy == 'source-checked' and strict['unit'] is None:
                        reuse.allow_unversioned = True
                        decision = reuse.assess(step, task['language'], configs[task['language']])
                        reuse.allow_unversioned = False
                    mode = ('both' if decision['needs_ner'] and decision['needs_action'] else
                            'ner' if decision['needs_ner'] else 'action' if decision['needs_action'] else 'none')
                    counts['step_calls_'+mode] += 1
                    counts['initial_extraction_calls'] += int(decision['needs_ner']) + int(decision['needs_action'])
                    why.update(decision['reasons'])
                    candidates = reuse.inputs.get((extraction_text(step), task['language']), [])
                    counts['exact_input_steps'] += bool(candidates)
                    counts['unknown_configuration_steps'] += bool(candidates and not any(c.get('configuration') for c in candidates))
            counts['documents_needing_extraction'] += bool(counts['initial_extraction_calls'])
            counts['strict_documents_needing_extraction'] += bool(counts['strict_initial_extraction_calls'])
            counts['saved_initial_extraction_calls'] = counts['strict_initial_extraction_calls']-counts['initial_extraction_calls']
            counts['documents_fully_reusing_extraction'] += not bool(counts['initial_extraction_calls'])
            counts['documents_needing_planning'] += bool(counts['planning_steps'])
            state = states[task['db']].get(task['uuid'])
            counts['status_'+str(state)] += 1
            totals.update(counts)
            reasons.update(why)
            if state != 'DONE':
                remaining.update(counts)
            documents.append(dict(uuid=task['uuid'], db=task['db'], status=state, counts=dict(counts), reasons=dict(why)))
        except Exception as exc:
            errors.append(dict(uuid=task['uuid'],error=f'{type(exc).__name__}: {exc}'))
        if n % 25 == 0:
            progress = dict(scanned=n,total=len(tasks),elapsed_seconds=round(time.time()-started,1),
                            counts=dict(totals),remaining=dict(remaining),reasons=dict(reasons),errors=len(errors))
            driver.save(output/'progress.json', progress)
            print(json.dumps(progress,ensure_ascii=False),flush=True)
    report = dict(scope=len(tasks),scanned=len(documents)+len(errors),policy=policy,counts=dict(totals),
                  remaining=dict(remaining),reasons=dict(reasons),errors=errors,documents=documents,
                  note='Read-only projection under current reuse rules. Planning may change step counts; retries, naming, coreference and final-product calls are additional. DONE documents need no further execution.')
    driver.save(output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='documents'},ensure_ascii=False),flush=True)
    return bool(errors)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--policy',choices=['strict','source-checked'],default='source-checked')
    args=parser.parse_args()
    raise SystemExit(audit(args.work_dir,args.output,args.limit,args.shard_index,args.shards,args.policy))
