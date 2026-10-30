"""Regression coverage for real-data acceptance failures, without API calls."""
import json
from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from batch.document_processor import DocumentProcessor, PreparedDocument
from pipeline.chemical_quality import image_label_evidence, markush_label_smiles
from pipeline.document.analogy_instantiate import apply_analogy_repair
from pipeline.document.business import business_reaction
from pipeline.document.chunk_processing import process_chunk
from pipeline.document.run_document import _global_dict, _image_for_chunk, merged_image_dict
from pipeline.extraction.extractor import ChemReXExtractor
from pipeline.extraction.parser import parse_ner_records
from pipeline.extraction.yield_values import reported_yields

from test_analogy_instantiation_20261027 import full_reaction, analogy_chunk
from test_image_markush_dict_20261027 import detection
from test_route_repair import compound, record
from test_scheme_template_20261027 import (template_chunk, scheme_extraction,
                                         resolved_coref, MARKUSH)


def test_scheme_actions_survive_real_parser_and_chunk_processing():
    raw_records = scheme_extraction()['ner_records']
    extractor = ChemReXExtractor()
    resolved = iter(resolved_coref(r) for r in raw_records)
    validation = {'rxn_validation': {'status': 'deferred', 'needs_human_review': True,
                  'normalized_compounds': {'reactants': [], 'products': []}}}
    pipe = SimpleNamespace(_get_extraction=lambda: extractor,
        _get_coref=lambda: SimpleNamespace(resolve=lambda _: deepcopy(next(resolved))),
        _get_validation=lambda: SimpleNamespace(validate_record=lambda _: deepcopy(validation)))
    with patch.object(extractor.scheme_client, 'chat', return_value=json.dumps(raw_records)), \
         patch.object(extractor.client, 'chat', side_effect=AssertionError('unexpected normal call')):
        result = process_chunk(pipe, template_chunk(), 'zh', MARKUSH)
    assert [r['extraction']['action'] for r in result['reactions']] == [['偶合'], ['取代'], ['酰化']]
    assert all('action_missing' not in r['quality']['blocking_issues'] for r in result['reactions'])
    assert [r['extraction']['step_order'] for r in result['reactions']] == [1, 1, 1]


def test_scheme_extra_fields_do_not_change_normal_ner():
    raw = json.dumps({'reaction_product': ['P'], 'action': '偶合', 'step_order': 2})
    assert 'action' not in parse_ner_records(raw)[0]
    assert parse_ner_records(raw, prompt_variant='scheme_template')[0]['step_order'] == 2


@pytest.mark.parametrize('expression', ['69.54%', '$69.54\\%$', '69.54％'])
def test_local_yield_and_export_replace_all_template_evidence(expression):
    template_rx = full_reaction(1, [compound('实施例16B', 'CC')],
                               [compound('16C', 'CCO')], '产率19.14%')
    template_rx['coref']['ner']['YIELD_PERCENT'] = '19.14'
    template_rx['validation']['yield_info'] = {
        'kind': 'explicit', 'value': '19.14', 'source_text': ['产率19.14%'],
        'reported_values': ['19.14']}
    text = f'依据实施例16C的方法，将实施例16B替换为实施例43B。得标题化合物，产率 {expression}。'
    chunk = analogy_chunk('example_43c', '化合物43C', text)
    target = record('example_43c', [])
    records = [record('example_16c', [template_rx]), target]
    apply_analogy_repair(records, [chunk])
    rx = target['reactions'][0]
    assert target['status'] == 'needs_review'
    assert target['coref'] is rx['coref']
    for layer in ('coref', 'extraction', 'validation'):
        assert rx[layer]['ner']['YIELD_PERCENT'] == '69.54'
        assert rx[layer]['yield_info']['reported_values'] == ['69.54']
        assert '19.14' not in str(rx[layer]['yield_info'])
    assert business_reaction('doc.pdf', target, rx)[0]['conditions']['yield']['value'] == 69.54
    rx['validation']['yield_info']['value'] = '19.14'  # simulate an already saved bad instance
    original_index = rx['index']
    repaired = apply_analogy_repair(records, [chunk])
    assert repaired['refreshed_chunk_ids'] == ['example_43c']
    assert len(target['reactions']) == 1 and target['reactions'][0]['index'] == original_index
    assert apply_analogy_repair(records, [chunk]) is None


def test_missing_local_yield_is_not_inherited():
    rx = full_reaction(1, [compound('实施例16B')], [compound('16C', 'CCO')], '产率85%')
    rx['coref']['ner']['YIELD_PERCENT'] = '85'
    target = record('example_43c', [])
    chunk = analogy_chunk('example_43c', '化合物43C',
        '依据实施例16C的方法，将实施例16B替换为实施例43B。得标题化合物。')
    apply_analogy_repair([record('example_16c', [rx]), target], [chunk])
    for layer in ('coref', 'extraction', 'validation'):
        assert target['reactions'][0][layer]['ner']['YIELD_PERCENT'] is None
        assert target['reactions'][0][layer]['yield_info']['kind'] == 'not_reported'


@pytest.mark.parametrize('kind', ['analysis', 'assay', 'formulation', 'purification'])
def test_analogy_does_not_promote_nonreaction_classifications(kind):
    rx = full_reaction(1, [compound('实施例16B')], [compound('16C', 'CCO')], '原文')
    target = record('example_43c', [], process_kind=kind)
    target.update(status='non_reaction', reason=kind)
    chunk = analogy_chunk('example_43c', '化合物43C', '依据实施例16C的方法，将实施例16B替换为实施例43B。')
    chunk['process_kind'] = kind
    assert apply_analogy_repair([record('example_16c', [rx]), target], [chunk]) is None
    assert target['status'] == 'non_reaction' and not target['reactions']


@pytest.mark.parametrize('coref', [None, {'ner': {}}])
def test_failed_template_does_not_create_another_failed_reaction(coref):
    rx = full_reaction(1, [], [], '原文')
    rx.update(coref=coref, reason='invalid_or_empty_ner')
    target = record('example_43c', [])
    chunk = analogy_chunk('example_43c', '化合物43C', '依据实施例16C的方法，将实施例16B替换为实施例43B。')
    assert apply_analogy_repair([record('example_16c', [rx]), target], [chunk]) is None
    assert target['reactions'] == []


def test_legacy_failed_instance_is_not_duplicated_on_resume():
    rx = full_reaction(1, [compound('实施例16B')], [compound('16C', 'CCO')], '原文')
    failed = {'index': 7, 'reason': 'invalid_or_empty_ner', 'coref': None,
              'analogical_instantiation': {'template_chunk_id': 'example_16c'}}
    target = record('example_43c', [failed])
    chunk = analogy_chunk('example_43c', '化合物43C', '依据实施例16C的方法，将实施例16B替换为实施例43B。')
    assert apply_analogy_repair([record('example_16c', [rx]), target], [chunk]) is None
    assert target['reactions'] == [failed]


def test_reclassification_updates_saved_validation_and_is_idempotent():
    text = '合成化合物19的实验步骤与实施例8的方法基本相同。LRMS m/z 538。'
    rx = full_reaction(1, [compound('019-1')], [compound('化合物19', 'CCO')],
                       text, record_kind='procedure_template')
    rx['quality']['blocking_issues'] = ['procedure_template_without_specific_instance']
    rx['validation'].update(record_kind='procedure_template', step_source=deepcopy(rx['step_source']))
    rx['extraction']['record_kind'] = 'procedure_template'
    target = record('example_19', [rx])
    assert apply_analogy_repair([target], [])['reclassified_chunk_ids'] == ['example_19']
    for layer in ('extraction', 'coref', 'validation'):
        assert rx[layer]['record_kind'] == 'reaction_candidate'
    assert 'procedure_template_without_specific_instance' not in rx['quality']['blocking_issues']
    assert apply_analogy_repair([target], []) is None


def test_latex_yield_keeps_original_evidence_offsets_and_rejects_purity():
    text = '得产物，产率 $69.54\\%$，纯度99%。'
    refs = reported_yields(text)
    assert len(refs) == 1 and refs[0]['value'] == 69.54
    assert refs[0]['source_text'] == text[slice(*refs[0]['span'])]
    assert reported_yields('纯度 $99\\%$，浓度10%。') == []


def test_same_image_conflict_blocks_global_fallback_and_geometry_wrapper():
    item = detection(['(VI)', '(VI)'], ['*c1ccccc1', '*c1ccncc1'], [(0, 0), (1, 1)], {})
    data = {'dict_result': {'(VI)': 'CCO'}, 'per_image_detections': [item]}
    associated = _image_for_chunk({'images': ['fig.jpg']}, data)
    assert associated['conflicting_identifiers'] == ['(VI)']
    assert '(VI)' not in markush_label_smiles(item)
    assert '(VI)' not in _global_dict(data)
    assert '(VI)' not in merged_image_dict({'(VI)': 'CCO'}, associated)


def test_equivalent_smiles_are_not_a_conflict():
    item = detection(['(A)', '(A)'], ['CCO', 'OCC'], [(0, 0), (1, 1)], {})
    evidence = image_label_evidence([item])
    assert evidence['dict_result'] == {'(A)': 'CCO'}
    assert evidence['unusable_identifiers'] == []


@pytest.mark.parametrize('module', ['pipeline.coref.run_coref', 'pipeline.coref.run_coref_fast'])
def test_cli_drops_same_record_conflicts(module, tmp_path):
    a = detection(['(VI)'], ['*c1ccccc1'], [(0, 0)], {})
    b = detection(['(VI)'], ['*c1ccncc1'], [(0, 0)], {})
    a['image_file_name'] = 'openexp_1_a.jpg'
    b['image_file_name'] = 'openexp_1_b.jpg'
    path = tmp_path / 'images.json'
    path.write_text(json.dumps({'per_image_detections': [a, b]}))
    assert import_module(module)._load_dict_per_record(str(path)) == {1: {}}


@pytest.mark.parametrize('mixed,changed,expected_calls,single', [
    (False, False, 0, False), (False, True, 1, False),
    (True, False, 0, False), (True, True, 1, False), (True, False, 0, True)])
def test_resume_uses_current_step_variant(mixed, changed, expected_calls, single, tmp_path):
    class Extractor:
        def configuration(self, language, variant=None):
            return {'model': 'scheme' if variant else 'normal', 'variant': variant, 'language': language}
    extractor = Extractor()
    steps = [{'step_id': 's1', 'record_kind': 'procedure_template'}]
    if mixed:
        steps.append({'step_id': 's2', 'record_kind': 'reaction_candidate'})
    reactions = []
    for step in steps:
        if single and step['step_id'] == 's2':
            continue
        config = extractor.configuration('en', 'scheme_template' if step['step_id'] == 's1' else None)
        if changed and step['step_id'] == 's1':
            config['model'] = 'previous-model'
        reactions.append({'step_id': step['step_id'], 'status': 'needs_review',
                          'extraction': {'extraction_configuration': config}})
    chunk = {'id': 'scheme', 'has_procedure': True, 'kind': 'scheme',
             'chunk_fingerprint': 'source', 'steps': steps}
    saved = {'id': 'scheme', 'chunk_fingerprint': 'source', 'status': 'needs_review', 'reactions': reactions}
    if single:
        saved['extraction'] = deepcopy(reactions[0]['extraction'])
    (tmp_path / 'results.jsonl').write_text(json.dumps(saved) + '\n')
    processor = object.__new__(DocumentProcessor)
    processor.language, processor.chunk_workers = 'en', 1
    processor.pipe = SimpleNamespace(_extraction=extractor)
    processor.global_dict_enabled = False
    processor.skip_final_product = processor.no_delivery = True
    processor._final_product = lambda *args: None
    processor._source_models = lambda: {}
    processor._run_one = Mock(return_value=saved)
    processor.extract(PreparedDocument([chunk], None, {}), tmp_path, {})
    assert processor._run_one.call_count == expected_calls
