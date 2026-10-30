import json
from copy import deepcopy
from unittest.mock import Mock

from pipeline.document.extraction_reuse import ExtractionReuse


def sample(time='2 h'):
    step = {'step_id': 'one', 'title': 'formaldehyde',
            'text': 'Methanol was stirred for 2 h to afford formaldehyde.',
            'local_text': 'Methanol was stirred for 2 h to afford formaldehyde.',
            'source_parts': [{'block_idx': 1, 'page_idx': 0,
                              'text': 'Methanol was stirred for 2 h to afford formaldehyde.'}],
            'procedure_references': [], 'context_issues': [], 'boundary_status': 'source_grounded'}
    ner = {'STARTING_MATERIAL': ['Methanol'], 'REACTION_PRODUCT': ['formaldehyde'],
           'TIME': time}
    rx = {'segment_index': 1, 'record_index': 1, 'text': step['text'], 'extraction': {
        'language': 'en', 'raw': {'ner': json.dumps(ner), 'action': '["stirred"]'}},
        'coref': {'ner': {'STARTING_MATERIAL': [{'name': 'Methanol', 'standard_name': 'methanol',
            'smiles': 'CO', 'structure_source': 'name_resolution',
            'naming_raw': 'Methanol | methanol\nformaldehyde | methanal'}]}}}
    return step, ner, {'id': 'old_one', 'chunk_fingerprint': 'old', 'reactions': [rx]}


def test_original_actions_are_reparsed_without_model_and_evidence_is_scoped():
    step, ner, record = sample()
    reuse = ExtractionReuse([record], '/saved/results.jsonl')
    extractor = Mock()
    result = reuse.extract(extractor, step, 'en')
    extractor.extract.assert_not_called()
    assert result['action'] and result['ner']['TIME'] == '2 h'
    assert result['reuse_provenance']['chunk_id'] == 'old_one'
    enriched = reuse.enrich_known_names(ner, step, 'en', 1)
    assert enriched['STARTING_MATERIAL'][0]['naming_raw'] == 'Methanol | methanol'
    assert enriched['REACTION_PRODUCT'] == ['formaldehyde']
    assert '\n' in record['reactions'][0]['coref']['ner']['STARTING_MATERIAL'][0]['naming_raw']


def test_empty_deferred_record_does_not_make_saved_response_ambiguous():
    step, _, saved = sample()
    deferred = deepcopy(saved)
    deferred['reactions'][0]['extraction']['raw'] = {}
    deferred['reactions'][0]['extraction']['deferred_llm'] = {'tasks': ['ner', 'action']}
    extractor = Mock()
    result = ExtractionReuse([saved, deferred]).extract(extractor, step, 'en')
    extractor.extract.assert_not_called()
    assert result['ner']['REACTION_PRODUCT'] == ['formaldehyde']


def test_missing_condition_reextracts_ner_but_reuses_actions():
    step, _, record = sample(time=None)
    extractor = Mock()
    extractor.extract.return_value = {'ner': {'TIME': '2h'}, 'ner_records': [{'TIME': '2h'}],
        'raw': {'ner': '{"TIME":"2h"}'}, 'parse_status': {'ner': 'ok'}}
    result = ExtractionReuse([record]).extract(extractor, step, 'en')
    extractor.extract.assert_called_once_with(step['text'], 'en', task='ner')
    assert result['reuse_provenance']['action_reused']
    assert not result['reuse_provenance']['ner_reused']


def test_isotope_time_wrong_yield_and_wrong_actions_are_not_reused():
    step, ner, record = sample(time='1 h')
    step['text'] = step['local_text'] = 'Methanol was stirred to afford formaldehyde (75-85% yield). 1H NMR (400 MHz).'
    step['source_parts'][0]['text'] = step['text']
    record['reactions'][0]['text'] = step['text']
    ner['YIELD_PERCENT'] = '75%'
    record['reactions'][0]['extraction']['raw']['ner'] = json.dumps(ner)
    record['reactions'][0]['extraction']['raw']['action'] = '["heated"]'
    decision = ExtractionReuse([record]).assess(step, 'en')
    assert decision['needs_ner'] and decision['needs_action']
    assert {'time_unattributed', 'yield_unattributed', 'action_unattributed'} <= set(decision['reasons'])


def test_latex_nmr_prefix_cannot_support_a_false_reaction_time():
    from pipeline.validation.quality import source_metadata, reaction_condition_text
    step, ner, record = sample(time='1 h')
    text = r'Methanol was stirred to afford formaldehyde. \({}^{1}\mathrm{H}\) NMR (400 MHz).'
    step['text'] = step['local_text'] = step['source_parts'][0]['text'] = text
    record['reactions'][0]['text'] = text
    metadata = source_metadata(ner, text, step, ['stirred'])
    assert metadata['field_evidence']['TIME'][0]['status'] == 'unattributed'
    assert '1' not in reaction_condition_text(text)
    assert 'time_unattributed' in ExtractionReuse([record]).assess(step, 'en')['reasons']


def test_invalid_ner_does_not_hide_unsupported_actions():
    step, _, record = sample()
    record['reactions'][0]['extraction']['raw'] = {'ner': '{}', 'action': '["heated"]'}
    decision = ExtractionReuse([record]).assess(step, 'en')
    assert decision['needs_ner'] and decision['needs_action']
    assert 'action_unattributed' in decision['reasons']


def test_changed_reference_context_reuses_same_body_response_after_regating():
    step, _, record = sample()
    changed = deepcopy(step)
    changed['procedure_references'] = [{'id': 'example_2', 'status': 'resolved',
                                       'blocks': [{'block_idx': 5, 'text': 'Stir for 3 h.'}], 'text': 'Stir for 3 h.'}]
    reuse = ExtractionReuse([record])
    decision = reuse.assess(changed, 'en')
    assert not decision['needs_ner'] and not decision['needs_action']
    assert 'same_body_fallback_after_reference_context_change' in decision['reasons']
    assert decision['unit']['body_fallback'] and decision['unit']['reactions'] == []
    assert decision['unit']['raw']['ner'] == record['reactions'][0]['extraction']['raw']['ner']
    # A different language has no same-body pool entry and still defers.
    assert reuse.assess(step, 'zh')['needs_ner']
    # Conflicting same-body responses stay deferred as ambiguous.
    conflicting = deepcopy(record)
    conflicting['reactions'][0]['extraction']['raw']['ner'] = '{"TIME":"3h"}'
    decision = ExtractionReuse([record, conflicting]).assess(changed, 'en')
    assert decision['unit'] is None and 'ambiguous_prior_responses' in decision['reasons']


def test_same_body_fallback_reports_honest_provenance():
    step, _, record = sample()
    changed = deepcopy(step)
    changed['procedure_references'] = [{'id': 'example_2', 'status': 'resolved',
                                       'blocks': [{'block_idx': 5, 'text': 'Stir for 3 h.'}], 'text': 'Stir for 3 h.'}]
    extractor = Mock()
    result = ExtractionReuse([record]).extract(extractor, changed, 'en')
    extractor.extract.assert_not_called()
    provenance = result['reuse_provenance']
    assert provenance['same_step_input'] is False
    assert provenance['compatibility_basis'] == 'same_procedure_body_and_current_source_checks'
    assert result['ner']['TIME'] == '2 h'


def test_same_body_fallback_response_must_still_pass_source_checks():
    step, _, record = sample(time='3 h')
    changed = deepcopy(step)
    changed['procedure_references'] = [{'id': 'example_2', 'status': 'resolved',
                                       'blocks': [{'block_idx': 5, 'text': 'Stir for 3 h.'}], 'text': 'Stir for 3 h.'}]
    decision = ExtractionReuse([record]).assess(changed, 'en')
    assert decision['needs_ner']
    assert 'local_time_not_preserved' in decision['reasons']


def test_exact_input_match_is_preferred_over_same_body_fallback():
    step, _, record = sample()
    referenced = deepcopy(record)
    referenced['reactions'][0]['text'] = step['text'] + (
        '\n\n引用程序（仅用于本步骤，局部明确修改优先；通用量不替代本步实际量）：\n'
        '程序 example_2 [blocks 5]:\nStir for 3 h.')
    referenced['reactions'][0]['extraction']['raw'] = {
        'ner': json.dumps({'STARTING_MATERIAL': ['Methanol'], 'REACTION_PRODUCT': ['formaldehyde'],
                           'TIME': '9 h'}), 'action': '["heated"]'}
    decision = ExtractionReuse([record, referenced]).assess(step, 'en')
    assert 'same_body_fallback_after_reference_context_change' not in decision['reasons']
    assert decision['unit']['raw'] == record['reactions'][0]['extraction']['raw']


def test_other_names_evidence_and_image_smiles_are_not_reused():
    step, ner, record = sample()
    compound = record['reactions'][0]['coref']['ner']['STARTING_MATERIAL'][0]
    compound['naming_raw'] = 'another_name | methanol'
    reuse = ExtractionReuse([record])
    assert reuse.enrich_known_names(ner, step, 'en', 1) == ner
    compound['naming_raw'] = 'Methanol | methanol'
    compound['structure_source'] = 'image_identifier'
    assert reuse.enrich_known_names(ner, step, 'en', 1) == ner


def test_replayed_candidate_is_revalidated_and_conflict_blocks_admission():
    from pipeline.document.chunk_processing import process_chunk
    step, ner, record = sample()
    chunk = {'id': 'new_one', 'kind': 'preparation', 'text': step['text'], 'steps': [step],
             'position': {'page_idx': 0, 'bbox': None, 'match_method': 'block_index', 'pages': [0]}}
    pipe = Mock()
    pipe._get_coref.return_value.resolve.side_effect = lambda payload: dict(payload)
    pipe._get_validation.return_value.validate_record.side_effect = lambda payload: dict(payload,
        rxn_validation={'status': 'passed', 'needs_human_review': False, 'mass_check': {'status': 'mismatch'},
            'effective_expression': 'CO>>C=O', 'normalized_compounds': {'products': [
                {'name': 'formaldehyde', 'status': 'valid', 'canonical_smiles': 'C=O'}]}})
    result = process_chunk(pipe, chunk, 'en', {}, reuse=ExtractionReuse([record]))
    pipe._get_extraction.return_value.extract.assert_not_called()
    pipe._get_validation.return_value.validate_record.assert_called_once()
    q = result['reactions'][0]['quality']
    assert 'ms_mass_mismatch' in q['blocking_issues'] and not q['admission_eligible']


def test_partial_standardization_reuses_own_names_and_calls_only_missing_names():
    from pipeline.standardize.standardizer import Standardizer
    std = object.__new__(Standardizer)
    std.compound_std = Mock()
    std.compound_std.standardize.return_value = {'formaldehyde': {'standard_name': 'methanal', 'smiles': 'C=O'}}
    step, ner, record = sample()
    enriched = ExtractionReuse([record]).enrich_known_names(ner, step, 'en', 1)
    result = std.standardize({'ner': enriched, 'action': ['stir']})
    std.compound_std.standardize.assert_called_once_with(['formaldehyde'])
    assert result['ner']['STARTING_MATERIAL'][0]['smiles'] == 'CO'
    assert result['ner']['REACTION_PRODUCT'][0]['smiles'] == 'C=O'


def test_batch_reuses_saved_raw_across_changed_fingerprint_and_appends_current_result(tmp_path, monkeypatch):
    from batch.document_processor import DocumentProcessor
    processor = object.__new__(DocumentProcessor)
    processor.language, processor.chunk_workers = 'en', 1
    processor.skip_final_product, processor.no_delivery = True, True
    processor.global_dict_enabled = False
    processor.pipe = Mock()
    processor.pipe._get_coref.return_value.resolve.side_effect = lambda payload: dict(payload)
    processor.pipe._get_validation.return_value.validate_record.side_effect = lambda payload: dict(payload,
        rxn_validation={'status': 'passed', 'needs_human_review': False,
            'effective_expression': 'CO>>C=O', 'normalized_compounds': {'products': [
                {'name': 'formaldehyde', 'status': 'valid', 'canonical_smiles': 'C=O'}]}})
    content = tmp_path / 'content_list.json'
    text = 'Methanol was stirred for 2 h to afford formaldehyde.'
    content.write_text(json.dumps([{'type': 'text', 'text': 'Example 1: formaldehyde', 'page_idx': 0},
                                  {'type': 'text', 'text': text, 'page_idx': 0}]))
    output = tmp_path / 'document'
    first = processor.prepare(str(content), output)
    chunk = first.chunks[0]
    step = chunk['steps'][0]
    ner = {'STARTING_MATERIAL': ['Methanol'], 'REACTION_PRODUCT': ['formaldehyde'], 'TIME': '2 h'}
    old = {'id': chunk['id'], 'chunk_fingerprint': 'historical-version', 'status': 'ok',
        'reactions': [{'segment_index': 1, 'record_index': 1, 'text': step['text'], 'extraction': {
            'language': 'en', 'raw': {'ner': json.dumps(ner), 'action': '["stirred"]'}}}]}
    saved = tmp_path / 'before/tasks/document'
    saved.mkdir(parents=True)
    (saved / 'chunks.jsonl').write_text(json.dumps(dict(chunk, chunk_fingerprint='historical-version')) + '\n')
    (saved / 'results.jsonl').write_text(json.dumps(old) + '\n')
    (output / 'results.jsonl').write_text(json.dumps(old) + '\n')
    monkeypatch.setenv('CHEMREX_REUSE_RESULTS_ROOT', str(tmp_path / 'before'))
    monkeypatch.setenv('BATCH_DB_PATH', str(tmp_path / 'tasks.db'))
    current = processor.prepare(str(content), output)
    processor.extract(current, output, {})
    rows = [json.loads(l) for l in (output / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and rows[-1]['chunk_fingerprint'] == current.chunks[0]['chunk_fingerprint']
    assert rows[-1]['chunk_fingerprint'] != 'historical-version'
    assert rows[-1]['reactions'][0]['extraction']['reuse_provenance']['ner_reused']
    processor.pipe._get_extraction.return_value.extract.assert_not_called()
    processor.pipe._get_validation.return_value.validate_record.assert_called_once()


def test_non_synthetic_document_does_not_call_abstract_model(tmp_path, monkeypatch):
    from batch.document_processor import DocumentProcessor
    processor = object.__new__(DocumentProcessor)
    processor.language, processor.chunk_workers = 'en', 1
    processor.skip_final_product, processor.no_delivery = False, True
    processor.global_dict_enabled = False
    processor.pipe = Mock()
    content = tmp_path / 'content_list.json'
    content.write_text(json.dumps([{'type': 'text', 'text': 'Example 1: Tablet formulation', 'page_idx': 0},
        {'type': 'text', 'text': 'Tablets were prepared with lactose and magnesium stearate.', 'page_idx': 0}]))
    output = tmp_path / 'document'
    prepared = processor.prepare(str(content), output)
    assert prepared.chunks and all(c['process_kind'] == 'formulation' for c in prepared.chunks)
    target = Mock(side_effect=AssertionError('non-synthetic document should not need an abstract call'))
    monkeypatch.setattr('pipeline.document.target_extractor.TargetExtractor', target)
    summary = processor.extract(prepared, output, {})
    target.assert_not_called()
    assert summary['final_product'] is None
    processor.pipe._get_extraction.assert_not_called()


def test_titleless_step_with_missing_compounds_reextracts_ner():
    step, _, record = sample()
    step['title'] = ''
    record['reactions'][0]['extraction']['raw']['ner'] = json.dumps({'TIME': '2 h'})
    decision = ExtractionReuse([record]).assess(step, 'en')
    assert decision['needs_ner']
    assert {'starting_material_evidence_missing', 'reaction_product_evidence_missing'} <= set(decision['reasons'])


def test_same_body_fallback_rejects_stale_reference_conditions():
    step, _, record = sample()
    step['text'] = step['local_text'] = 'Methanol was stirred to afford formaldehyde.'
    step['source_parts'][0]['text'] = step['local_text']
    record['reactions'][0]['text'] = step['text']
    changed = deepcopy(step)
    changed['procedure_references'] = [{'id': 'example_2', 'status': 'resolved',
        'blocks': [{'block_idx': 5, 'page_idx': 0, 'text': 'Stir for 2 h, then stir for 4 h.'}],
        'text': 'Stir for 2 h, then stir for 4 h.'}]
    decision = ExtractionReuse([record]).assess(changed, 'en')
    assert decision['needs_ner']
    assert 'reference_time_not_preserved' in decision['reasons']
    assert 'same_body_fallback_after_reference_context_change' in decision['reasons']


def test_reused_actions_in_wrong_order_trigger_action_reextraction():
    step, _, record = sample(time=None)
    step['text'] = step['local_text'] = 'Methanol was stirred, then filtered to afford formaldehyde.'
    step['source_parts'][0]['text'] = step['local_text']
    record['reactions'][0]['text'] = step['text']
    record['reactions'][0]['extraction']['raw']['action'] = '["filtered","stirred"]'
    decision = ExtractionReuse([record]).assess(step, 'en')
    assert decision['needs_action'] and 'action_order_unverified' in decision['reasons']
