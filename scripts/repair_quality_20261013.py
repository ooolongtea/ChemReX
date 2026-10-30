"""Scoped inventory, source-grounded samples and shadow reruns; no queue resume.

The scope is the saved October 13 document list, never a fresh DONE query.
Inventory and scan read production results. Pilot writes only --output.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_rows(path):
    path = Path(path)
    if path.is_file():
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def scope(path):
    tasks = json.loads(Path(path).read_text())['tasks']
    if len(tasks) != 5864 or len({t['output_dir'] for t in tasks}) != 5864:
        raise ValueError('Expected the exact 5864-document audit scope')
    return tasks


def records(folder):
    from pipeline.document.results import latest_by_id
    chunks = list(read_rows(Path(folder) / 'chunks.jsonl'))
    latest = latest_by_id(Path(folder) / 'results.jsonl', {c['id']: c.get('chunk_fingerprint') for c in chunks})
    return chunks, latest


def source(task, chunks):
    folder = Path(task['output_dir'])
    marker = folder / 'pdf_prepared/parse_result.json'
    value = json.loads(marker.read_text()).get('content_list_file') if marker.is_file() else None
    if not value and chunks:
        value = chunks[0]['source']
    if not value:
        choices = list((folder / 'pdf_prepared').rglob('*content_list.json'))
        if len(choices) != 1:
            raise ValueError('No unambiguous PDF parsing cache')
        value = str(choices[0])
    from pipeline.document.source_paths import resolve_source
    return str(resolve_source(value))


def step_source_blocks(reaction):
    """Include title evidence when repairing an old carried-over heading."""
    step = (reaction or {}).get('step_source') or {}
    parts = list(step.get('source_parts') or [])
    heading = step.get('title_source') or {}
    parts.extend(heading.get('blocks') or [heading])
    return {p['block_idx'] for p in parts if p.get('block_idx') is not None}


def pilot_image_names(chunks, blocks, wanted_blocks):
    """Repair sampled-step images; other steps retain their cached evidence.

    A newly merged chunk can contain dozens of unrelated methods. Keep its
    complete text and method context, but do not treat all of its images as
    part of the sampled step. Single-step and whole-document samples retain
    their full image scope. No production processing uses this selection.
    """
    from batch.document_processor import SHORTCIRCUIT_KINDS
    names = set()
    for chunk in chunks:
        if chunk.get('process_kind') in SHORTCIRCUIT_KINDS:
            continue
        steps = chunk.get('steps') or []
        if not wanted_blocks or len(steps) <= 1:
            names.update(Path(p).name for p in chunk.get('images', []))
            continue
        selected_steps = [s for s in steps if wanted_blocks.intersection(
            step_source_blocks(dict(step_source=s)))]
        pages = {p.get('page_idx') for s in selected_steps
                 for p in [*s.get('source_parts', []), s.get('title_source') or {}]
                 if p.get('page_idx') is not None}
        chunk_images = {Path(p).name for p in chunk.get('images', [])}
        names.update(Path(b['img_path']).name for b in blocks
                     if b.get('img_path') and b.get('page_idx') in pages
                     and Path(b['img_path']).name in chunk_images)
    return names


def inventory(tasks, output):
    import sqlite3
    from pipeline.extraction.parser import parse_action_output
    from pipeline.document.step_context import _GENERIC, is_template_definition
    from pipeline.validation.normalize import named_discrete_form
    totals, groups, unavailable = Counter(), defaultdict(Counter), Counter()
    populations = defaultdict(list)
    errors, documents = [], []
    task_states = {}
    for path in {t['db'] for t in tasks}:
        with sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True) as connection:
            task_states[path] = dict(connection.execute('select uuid,status from pdf_task'))
    for n, task in enumerate(tasks, 1):
        try:
            folder = Path(task['output_dir'])
            chunks, latest = records(folder)
            rows = list(read_rows(folder / 'delivery/reactions.jsonl'))
            counts = Counter(documents=1, delivery_rows=len(rows), chunks=len(chunks))
            state = task_states[task['db']].get(task['uuid'])
            counts['task_status_' + str(state)] += 1
            for row in rows:
                missing = not bool(row.get('reaction_smiles'))
                kind = row.get('record_kind')
                if kind is None:
                    counts['legacy_record_kind_missing_rows'] += 1
                    ref = row.get('source') or {}
                    record = latest.get(ref.get('chunk_id'), {})
                    reaction = next((r for r in record.get('reactions') or [] if r['index'] == ref.get('reaction_index')), {})
                    step = reaction.get('step_source') or {}
                    src = record.get('source') or {}
                    kind = 'procedure_template' if is_template_definition(src.get('kind'), step.get('title'),
                              step.get('source_parts') or src.get('source_parts') or []) else 'reaction_candidate'
                    counts['retrospectively_classified_templates'] += kind == 'procedure_template'
                counts['no_smiles'] += missing
                counts['templates'] += kind == 'procedure_template'
                if kind != 'procedure_template' and row.get('process_kind') == 'synthesis':
                    counts['synthesis_candidates'] += 1
                    counts['synthesis_no_smiles'] += missing
                counts['admitted'] += bool(row['quality']['admission_eligible'])
                counts['admitted_no_smiles'] += bool(row['quality']['admission_eligible']) and missing
                if missing:
                    products = [c for c in row['compounds'] if 'product' in c['roles']]
                    counts['missing_no_product' if not products else 'missing_product_no_structure' if
                           not any(c.get('canonical_smiles') for c in products) else 'missing_expression_only'] += 1
            failures = [r for r in latest.values() if r.get('status') not in ('ok', 'needs_review', 'non_reaction')]
            counts['processing_failed_documents'] += state != 'DONE' or bool(failures) or len(latest) < len(chunks)
            raw_candidates = [r for record in latest.values() for r in record.get('reactions') or []
                if (r.get('step_source') or {}).get('process_kind') == 'synthesis'
                and (r.get('step_source') or {}).get('record_kind') != 'procedure_template']
            counts['extracted_synthesis_candidates'] = len(raw_candidates)
            if not counts['synthesis_candidates'] and not counts['processing_failed_documents']:
                counts['no_synthesis_documents'] += 1
                counts['zero_chunks_documents' if not chunks else
                       'synthesis_candidate_empty_delivery_documents' if raw_candidates and not rows else
                       'zero_delivery_documents' if not rows
                       else 'template_or_non_synthesis_only_documents'] += 1
                populations[('no_reaction_document', task['language'])].append(dict(task=task, chunk_id=None))
            for record in latest.values():
                for rx in record.get('reactions') or []:
                    ex, co = rx.get('extraction') or {}, rx.get('coref') or {}
                    rv = (rx.get('validation') or {}).get('rxn_validation') or {}
                    products = (rv.get('normalized_compounds') or {}).get('products') or []
                    issues = set((rx.get('quality') or {}).get('blocking_issues') or [])
                    candidate = dict(task=task, chunk_id=record['id'], reaction_index=rx['index'],
                                     blocks=(record.get('source') or {}).get('block_indices', []))
                    categories = []
                    if any(r.get('status') != 'resolved' for r in (rx.get('step_source') or {}).get('procedure_references', [])):
                        categories.append('reference')
                    if any(p.get('reason') == 'mixture_or_non_discrete_compound' and p.get('canonical_smiles')
                           and named_discrete_form(p.get('name', ''), p['canonical_smiles']) for p in products):
                        categories.append('salt')
                    if parse_action_output((ex.get('raw') or {}).get('action', '')) != ex.get('action', []):
                        categories.append('action')
                    src = record.get('source') or {}
                    if src.get('kind') == 'fallback' and not (rx.get('step_source') or {}).get('title') and any(
                            _GENERIC.fullmatch(p.get('text', '').strip()) for p in src.get('source_parts', [])):
                        categories.append('template')
                    if 'chunk_requires_review' in issues or 'terminal_product_coverage_unknown' in issues:
                        categories.append('propagation')
                    if (rx.get('quality') or {}).get('admission_eligible'):
                        categories.append('normal_control')
                    if products and not any(p.get('status') == 'valid' for p in products):
                        categories.append('still_unavailable')
                    for p in products:
                        if p.get('status') != 'valid':
                            unavailable[p.get('resolution_status') or p.get('reason') or 'undetermined'] += 1
                    for category in categories:
                        populations[(category, task['language'])].append(candidate)
            totals.update(counts)
            groups[task['language']].update(counts)
            documents.append(dict(uuid=task['uuid'], db=task['db'], counts=dict(counts)))
        except Exception as exc:
            errors.append(dict(uuid=task['uuid'], error=f'{type(exc).__name__}: {exc}'))
            counts = dict(documents=1, processing_failed_documents=1, inventory_failed_documents=1)
            totals.update(counts)
            groups[task['language']].update(counts)
            documents.append(dict(uuid=task['uuid'], db=task['db'], counts=counts,
                                  error=errors[-1]['error']))
        if n % 250 == 0:
            print(json.dumps(dict(inventoried=n, errors=len(errors))), flush=True)
    report = dict(scope_documents=len(tasks), counts=dict(totals),
                  by_language={k: dict(v) for k, v in groups.items()},
                  unavailable_product_status=dict(unavailable), errors=errors, documents=documents)
    for denominator, numerator, key in [('delivery_rows', 'no_smiles', 'no_smiles_pct'),
                                       ('synthesis_candidates', 'synthesis_no_smiles', 'synthesis_no_smiles_pct')]:
        report[key] = round(100 * totals[numerator] / totals[denominator], 2) if totals[denominator] else None
    report['no_synthesis_documents_pct'] = round(100 * totals['no_synthesis_documents'] / len(tasks), 2)
    save(output / 'statistics.json', report)
    rng, selected, used = random.Random(20261013), [], set()
    for category, desired in [('reference',10), ('salt',10), ('action',10), ('template',10), ('propagation',10),
                              ('normal_control',20), ('still_unavailable',20), ('no_reaction_document',10)]:
        pools = [list(populations[(category, lang)]) for lang in ('zh', 'en')]
        for pool in pools:
            rng.shuffle(pool)
        chosen = []
        def take(pool, count):
            for item in pool:
                key = (item['task']['output_dir'], item['chunk_id'], item.get('reaction_index'))
                if key not in used:
                    used.add(key)
                    chosen.append(dict(item, category=category))
                    if len(chosen) >= count:
                        break
        take(pools[0], desired // 2)
        take(pools[1], desired)
        if len(chosen) < desired:
            take(pools[0], desired)
        selected.extend(chosen)
    save(output / 'sample.json', dict(seed=20261013, requested=100, selected=len(selected),
                                      populations={f'{k[0]}:{k[1]}':len(v) for k,v in populations.items()}, samples=selected))
    print(json.dumps({k:v for k,v in report.items() if k not in ('documents',)}, ensure_ascii=False), flush=True)
    return bool(errors)


def scan(tasks, output):
    from pipeline.document.segmenter import load_blocks, segment_blocks, select_procedure_chunks
    from scripts.reprocess_audit_20261007 import shape
    selected, errors = [], []
    for n, task in enumerate(tasks, 1):
        try:
            chunks, latest = records(task['output_dir'])
            reasons = set()
            for record in latest.values():
                if record.get('reactions'):
                    reasons.add('reaction_quality_recalculation')
                for rx in record.get('reactions') or []:
                    rv = (rx.get('validation') or {}).get('rxn_validation') or {}
                    issues = (rx.get('quality') or {}).get('blocking_issues', [])
                    if any(i in issues for i in ('chunk_requires_review','terminal_product_coverage_unknown')):
                        reasons.add('step_quality_propagation')
                    for p in (rv.get('normalized_compounds') or {}).get('products') or []:
                        if p.get('status') != 'valid':
                            reasons.add('product_structure_requires_current_resolution')
                    for values in ((rx.get('coref') or {}).get('ner') or {}).values():
                        if isinstance(values, list) and any(isinstance(c,dict) and not c.get('smiles') for c in values):
                            reasons.add('compound_resolution_missing')
            if not reasons:
                blocks = load_blocks(source(task, chunks))
                new = select_procedure_chunks(blocks, source(task, chunks), segment_blocks(blocks, source(task, chunks)))
                if shape(chunks) != shape(new):
                    reasons.add('segmentation_title_classification_or_reference_changed')
            if reasons:
                selected.append(dict(task, reasons=sorted(reasons)))
        except Exception as exc:
            errors.append(dict(uuid=task['uuid'], error=f'{type(exc).__name__}: {exc}'))
        if n % 100 == 0:
            print(json.dumps(dict(scanned=n, affected=len(selected), errors=len(errors))), flush=True)
    save(output / 'affected.json', dict(tasks=selected, errors=errors))
    print(json.dumps(dict(scanned=len(tasks), affected=len(selected), errors=errors)), flush=True)
    return bool(errors)


def pilot(tasks, sample_path, output, refresh=False, document_uuids=None):
    from loguru import logger
    logger.remove(); logger.add(sys.stderr, level='WARNING')
    os.environ.setdefault('CHEMREX_LEGACY_CODE_REVISION','66523a5')
    defects = ROOT/'doc/reaction-quality-repair-20261013/confirmed-source-defects.json'
    if defects.is_file():
        os.environ.setdefault('CHEMREX_SOURCE_DEFECTS_FILE', str(defects))
    saved_env = json.loads((ROOT/'doc/pause-20261013/state.json').read_text())['schedulers'][0]['resume_environment']
    for key in ('MAAS_RATE_LIMIT_FILE', 'STANDARDIZE_NETWORK_TIMEOUT', 'SSL_CERT_FILE',
                'HF_HOME', 'TORCH_HOME', 'CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES',
                'IMGPARSER_PYTHON', 'PADDLE_PYTHON', 'BATCH_IMAGE_BATCH_SIZE', 'BATCH_IMAGE_MIN_FREE_MB'):
        if key in saved_env:
            os.environ.setdefault(key, saved_env[key])
    from batch.cli import apply_llm_env
    apply_llm_env(argparse.Namespace(preset='huawei-maas', llm_provider='maas',
                                    llm_model='deepseek-v4.1-flash', offline_standardize=False))
    from batch.document_processor import DocumentProcessor
    from pipeline.document.extraction_reuse import ExtractionReuse
    from pipeline.document.delivery import export_document
    from scripts.verify_delivery import verify_document
    from batch.image_client import ImageWorkerClient
    from batch import config as batch_config
    from pipeline.document.image_reuse import repair_missing_molecules, known_image_defects
    image_python = os.getenv('IMGPARSER_PYTHON') or saved_env.get('IMGPARSER_PYTHON')
    image_command = ([image_python, '-u', str(ROOT/'batch/image_worker.py')] if image_python
                     else batch_config.image_worker_command(ROOT/'batch/image_worker.py'))
    image_client = ImageWorkerClient(command=image_command,
                                    timeout=batch_config.IMAGE_TIMEOUT, log_path=output/'image_worker.log')
    samples = json.loads(Path(sample_path).read_text())['samples']
    wanted = defaultdict(set)
    for item in samples:
        # Populate document keys here; precise step evidence is read below.
        wanted[item['task']['output_dir']]
    # Explicit original audit repro documents, beyond the random sample.
    known = {'US8669281': {'example_1', 'fallback_597_601_1'},
             'US9090558': {'fallback_584_588_1'}, 'US9359399': {'synthesis_0_14'}}
    processors, reports = {}, []
    selected = [t for t in tasks if t['output_dir'] in wanted or t['uuid'].split('_')[0] in known]
    if document_uuids is not None:
        chosen = set(document_uuids)
        if chosen - {t['uuid'] for t in selected}:
            raise ValueError('Pilot document selection must belong to the sampled audit scope')
        selected = [t for t in selected if t['uuid'] in chosen]
    for language in sorted({t['language'] for t in selected}):
        processors[language] = DocumentProcessor(language, chunk_workers=16, no_delivery=True)
    def one(task):
        folder = Path(task['output_dir'])
        old, latest = records(folder)
        blocks_wanted = set(wanted[task['output_dir']])
        for item in samples:
            if item['task']['output_dir'] == task['output_dir']:
                reaction = next((r for r in latest.get(item['chunk_id'], {}).get('reactions') or []
                                 if r['index'] == item.get('reaction_index')), None)
                blocks_wanted.update(step_source_blocks(reaction) or set(item.get('blocks') or []))
        for c in old:
            if c['id'] in known.get(task['uuid'].split('_')[0], set()):
                blocks_wanted.update(c['block_indices'])
        target = output / 'pilot' / Path(task['db']).stem / task['uuid']
        _, staged_latest = records(target)
        processor = processors.setdefault(task['language'], None)
        if processor is None:
            processor = processors[task['language']] = DocumentProcessor(task['language'], chunk_workers=16, no_delivery=True)
        prepared = processor.prepare(source(task, old), target)
        all_chunks = len(prepared.chunks)
        if blocks_wanted:
            prepared.chunks = [c for c in prepared.chunks if blocks_wanted.intersection(c['block_indices'])]
        prepared.manifest['document_chunks'] = all_chunks
        prepared.manifest['selected_chunks'] = [c['id'] for c in prepared.chunks]
        prepared.manifest['total_chunks'] = len(prepared.chunks)
        save(target / 'manifest.json', prepared.manifest)
        (target / 'chunks.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in prepared.chunks))
        prepared.extraction_reuse = ExtractionReuse([*latest.values(), *staged_latest.values()],
                                                    folder / 'results.jsonl')
        if refresh and (target / 'results.jsonl').is_file():
            with (target / 'previous-results.jsonl').open('a') as archived:
                archived.write((target / 'results.jsonl').read_text())
            (target / 'results.jsonl').unlink()
        # The final-product evidence is unchanged and can be copied as evidence.
        import shutil
        if (folder / 'final_product.json').is_file() and not (target / 'final_product.json').exists():
            shutil.copy2(folder / 'final_product.json', target / 'final_product.json')
        image_path = target / 'image_result.json'
        cached_image = image_path if image_path.is_file() else folder / 'image_result.json'
        image = json.loads(cached_image.read_text()) if cached_image.is_file() else {}
        selected_images = pilot_image_names(prepared.chunks, prepared.matcher.blocks, blocks_wanted)
        prepared.manifest['selected_image_repairs'] = sorted(selected_images)
        prepared.manifest['image_repair_scope'] = 'sampled_step_pages'
        save(target / 'manifest.json', prepared.manifest)
        if selected_images:
            image_base = Path(source(task, old)).parent / 'images'
            defects = known_image_defects(task['uuid'])
            def process_images(folder, return_per_image=True):
                names = {p.name for p in Path(folder).iterdir()}
                from pipeline.versions import IMAGE_PIPELINE_VERSION
                import filecmp
                for defect in defects:
                    replay = Path((defect.get('model_replay') or {}).get('report', ''))
                    original_image = Path((defect.get('model_replay') or {}).get('source_image_copy', ''))
                    if (names == {defect['image_file']} and replay.is_file() and original_image.is_file()
                            and filecmp.cmp(Path(folder)/defect['image_file'], original_image, shallow=False)):
                        fresh = json.loads(replay.read_text())
                        if (fresh.get('image_pipeline_version') == IMAGE_PIPELINE_VERSION
                                and {r['image_file_name'] for r in fresh.get('per_image_detections', [])} == names):
                            return fresh
                return image_client.process(folder, return_per_image=return_per_image)
            repaired = repair_missing_molecules(image, image_base, process_images, selected_images, defects)
            if repaired is not None:
                image = repaired
                save(image_path, image)
        summary = processor.extract(prepared, target, image, retry_errors=True)
        failures = summary.get('result_status', {}).get('error', 0)
        if (summary.get('final_product') or {}).get('error'):
            failures += 1
        if not failures:
            export_document(target, pdf=task['pdf_path'])
            verify_document(target)
        return dict(uuid=task['uuid'], source_dir=str(folder), output_dir=str(target),
                            selected_chunks=len(prepared.chunks), document_chunks=all_chunks,
                            status=summary.get('result_status'), failures=failures)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(one, task): task for task in selected}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                task = futures[future]
                result = dict(uuid=task['uuid'], failures=1, error=f'{type(exc).__name__}: {exc}')
            reports.append(result)
            save(output / 'pilot-report.json', reports)
            print(json.dumps(result), flush=True)
    image_client.stop()
    return any(r['failures'] for r in reports)


def packets(sample_path, output):
    samples = json.loads(Path(sample_path).read_text())['samples']
    cache, after_cache, packets = {}, {}, []
    for number, item in enumerate(samples, 1):
        task = item['task']
        if task['output_dir'] not in cache:
            old, latest = records(task['output_dir'])
            src = source(task, old)
            cache[task['output_dir']] = (old, latest, src)
        old, latest, src = cache[task['output_dir']]
        rx = next((r for r in latest.get(item['chunk_id'], {}).get('reactions', [])
                   if r['index'] == item.get('reaction_index')), None)
        stage = output / 'pilot' / Path(task['db']).stem / task['uuid']
        if task['output_dir'] not in after_cache:
            after_cache[task['output_dir']] = records(stage)[1]
        after_records = after_cache[task['output_dir']]
        wanted_parts = step_source_blocks(rx)
        source_blocks = set(item.get('blocks') or []) | wanted_parts
        aligned = [r for r in after_records.values() if source_blocks.intersection(
                    (r.get('source') or {}).get('block_indices', []))]
        if not wanted_parts:
            wanted_parts = set(item.get('blocks') or [])
        aligned_reactions = [reaction for record in aligned for reaction in record.get('reactions') or []
            if wanted_parts.intersection(step_source_blocks(reaction))]
        packet = dict(number=number, category=item['category'], uuid=task['uuid'], language=task['language'],
                      pdf=task['pdf_path'], parsed_pdf=src, before=rx, after=aligned,
                      aligned_reactions=aligned_reactions,
                      stage=str(stage), verdict=None, reason=None)
        if rx is None:
            blocks = json.loads(Path(src).read_text())
            packet['source_outline'] = [dict(index=i, page_idx=b.get('page_idx'), text=b.get('text'))
                                        for i,b in enumerate(blocks) if b.get('text')]
        packets.append(packet)
    save(output / 'review-packets.json', packets)
    print(json.dumps(dict(packets=len(packets), after_present=sum(bool(p['after']) for p in packets))))


def repair(tasks, affected_path, acceptance_path, output):
    """Back up first, then use existing transactional, UUID-scoped queue tools."""
    import shutil
    import sqlite3
    import subprocess
    import time
    acceptance = json.loads(Path(acceptance_path).read_text())
    if (acceptance.get('phase') != 'pre_batch_acceptance' or acceptance.get('passed') is not True
            or acceptance.get('reviewed_units') != 100 or acceptance.get('failures')):
        raise ValueError('Full source-grounded pre-batch acceptance has not passed')
    manifest = json.loads(Path(affected_path).read_text())
    if manifest.get('errors'):
        raise ValueError('Resolve scope scan errors before updating production results')
    selected = manifest['tasks']
    allowed = {(t['db'], t['uuid'], t['output_dir']) for t in tasks}
    if not all((t['db'],t['uuid'],t['output_dir']) in allowed for t in selected):
        raise ValueError('Affected list contains a document outside the audit scope')
    save(output / 'scope.json', dict(tasks=tasks))
    save(output / 'affected.json', manifest)
    save(output / 'acceptance.json', acceptance)
    state_path = output / 'repair-status.json'
    existing = json.loads(state_path.read_text()) if state_path.is_file() else {}
    state = dict(existing, phase=existing.get('phase', 'backing_up'),
                 scope=len(tasks), affected=len(selected), backup_in_progress=True)
    save(state_path, state)
    grouped = defaultdict(list)
    for task in selected:
        grouped[task['db']].append(task)
    backed_up = 0
    for db, batch in grouped.items():
        dst = output / (Path(db).stem + '-before.db')
        if not dst.exists():
            with sqlite3.connect(Path(db).as_uri()+'?mode=ro',uri=True) as src:
                with sqlite3.connect(dst) as target:
                    src.backup(target)
        for task in batch:
            destination = output / 'before' / Path(db).stem / task['uuid']
            if (destination / 'task.json').is_file():
                backed_up += 1
                continue
            destination.mkdir(parents=True, exist_ok=True)
            for name in ('chunks.jsonl','results.jsonl','manifest.json','summary.json','links.json',
                         'document_result.json','final_product.json','source_identity.json',
                         'image_result.json','pdf_prepared','delivery'):
                origin = Path(task['output_dir']) / name
                if origin.is_dir():
                    shutil.copytree(origin,destination/name,dirs_exist_ok=True)
                elif origin.is_file():
                    (destination/name).parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(origin,destination/name)
            save(destination / 'task.json',task)
            backed_up += 1
            if backed_up % 25 == 0 or backed_up == len(selected):
                state.update(backed_up=backed_up, updated_at=time.time())
                save(state_path, state)
                print(json.dumps(dict(phase='backing_up', backed_up=backed_up,
                                      affected=len(selected))), flush=True)
        save(output / (Path(db).stem+'-ids.json'), [t['uuid'] for t in batch])
    settings = json.loads((ROOT/'doc/pause-20261013/state.json').read_text())['schedulers']
    if set(grouped) - {s['db'] for s in settings}:
        raise ValueError('Selected database has no saved scheduler configuration')
    state['backup_in_progress'] = False
    children = []
    try:
        for spec in settings:
            db = spec['db']
            if db not in grouped:
                continue
            env = dict(os.environ)
            env.update(spec['resume_environment'])
            defects = ROOT/'doc/reaction-quality-repair-20261013/confirmed-source-defects.json'
            if defects.is_file():
                env['CHEMREX_SOURCE_DEFECTS_FILE'] = str(defects)
            env.update(CHEMREX_REUSE_RESULTS_ROOT=str(output/'before'),
                       CHEMREX_LEGACY_CODE_REVISION='66523a5',
                       CHEMREX_CALL_METRICS=str(output/'model-calls.jsonl'),
                       BATCH_LOG_DIR=str(output/'logs'/Path(db).stem), PYTHONDONTWRITEBYTECODE='1')
            ids = output / (Path(db).stem+'-ids.json')
            base = [spec['resume_argv'][0], str(ROOT/'run_batch.py'), '--db', db]
            if existing.get('phase') not in ('running','processed'):
                subprocess.run(base+['retry','--ids-file',str(ids),'--reset-attempts'],
                               cwd=ROOT,env=env,check=True)
            log = (output/(Path(db).stem+'-repair.log')).open('a')
            process = subprocess.Popen(base+['run','--ids-file',str(ids),'--preset','huawei-maas',
                                       '--llm-provider','maas','--llm-model','deepseek-v4.1-flash'],
                                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append((db,process,log))
        while any(p.poll() is None for _,p,_ in children):
            state.update(phase='running', children=[dict(db=db,pid=p.pid,returncode=p.poll()) for db,p,_ in children],
                         updated_at=time.time())
            save(state_path,state)
            print(json.dumps(state),flush=True)
            time.sleep(30)
        state.update(phase='processed', returncodes={db:p.returncode for db,p,_ in children})
        save(state_path,state)
        return any(p.returncode for _,p,_ in children)
    except BaseException:
        for _,p,_ in children:
            if p.poll() is None:
                p.terminate()
        raise
    finally:
        for _,_,log in children:
            log.close()


def verify(tasks, output):
    import sqlite3
    from scripts.verify_delivery import verify_document
    from pipeline.validation.quality import QUALITY_VERSION
    errors, checked = [], 0
    for task in tasks:
        try:
            with sqlite3.connect(Path(task['db']).as_uri()+'?mode=ro',uri=True) as db:
                row = db.execute('select status,error from pdf_task where uuid=?',(task['uuid'],)).fetchone()
            if not row or row[0] != 'DONE':
                raise ValueError(f'Task not successfully completed: {row}')
            verify_document(task['output_dir'])
            _, latest = records(task['output_dir'])
            for record in latest.values():
                for rx in record.get('reactions') or []:
                    if (rx.get('quality') or {}).get('version') != QUALITY_VERSION:
                        raise ValueError('Reaction quality was not recalculated')
            checked += 1
        except Exception as exc:
            errors.append(dict(uuid=task['uuid'], error=f'{type(exc).__name__}: {exc}'))
    save(output/'full-verification.json', dict(expected=len(tasks),verified=checked,errors=errors))
    print(json.dumps(dict(expected=len(tasks),verified=checked,errors=len(errors))))
    return bool(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'scan', 'pilot', 'packets', 'repair', 'verify'])
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample', type=Path)
    parser.add_argument('--affected', type=Path)
    parser.add_argument('--acceptance', type=Path)
    parser.add_argument('--refresh', action='store_true', help='Revalidate shadow results, preserving previous raw outputs')
    parser.add_argument('--documents', nargs='+', help='Rerun only these sampled document UUIDs in the isolated pilot')
    args = parser.parse_args()
    tasks = scope(args.scope)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'inventory':
        return inventory(tasks, args.output)
    if args.command == 'scan':
        return scan(tasks, args.output)
    if args.command == 'verify':
        return verify(tasks,args.output)
    if args.command == 'repair':
        if not args.affected or not args.acceptance:
            parser.error('repair requires --affected and --acceptance')
        return repair(tasks,args.affected,args.acceptance,args.output.resolve())
    if not args.sample:
        parser.error('pilot requires --sample')
    if args.command == 'packets':
        return packets(args.sample, args.output)
    return pilot(tasks, args.sample, args.output, args.refresh, args.documents)


if __name__ == '__main__':
    from run_batch import exit_completed_cli
    exit_completed_cli(int(bool(main())))
