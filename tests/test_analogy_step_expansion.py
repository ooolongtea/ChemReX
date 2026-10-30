"""Mixed explicit/referenced syntheses, using the CN105 source sentence forms."""
from copy import deepcopy

import pytest

from pipeline.document.analogy_instantiate import (
    _apply_substitution, _substitutions, apply_analogy_repair)
from pipeline.document.step_context import (
    _product_heading, procedure_citations, referenced_step_numbers, resolve_programs, split_steps)
from pipeline.document.routes import assemble_routes_v2, compound_lists
from test_analogy_instantiation_20261027 import full_reaction, analogy_chunk
from test_route_repair import compound, record, scope_doc


def source_chunk(cid, parts):
    return {'id': cid, 'reference_id': cid, 'kind': 'example', 'title': '',
            'text': '\n'.join(parts), 'procedure_index': {},
            'source_parts': [dict(text=t, block_idx=i, page_idx=0)
                             for i, t in enumerate(parts)]}


def template(cid='example_101', count=5):
    chunk = source_chunk(cid, [p for i in range(1, count + 1) for p in (
        f'{i}. 中间体{101 if cid == "example_101" else 201}-{i}的合成',
        f'加入原料，在{i * 10}℃反应{i}小时，得到产物。')])
    chunk['steps'] = split_steps(chunk)
    reactions = []
    for i, step in enumerate(chunk['steps'], 1):
        rx = full_reaction(i, [compound(f'101-{i-1}', 'CC')],
                           [compound(f'101-{i}', 'CCC')], step['local_text'])
        rx.update(step_id=step['step_id'], step_source=deepcopy(step))
        for layer in ('coref', 'extraction'):
            rx[layer]['ner'].update(TEMPERATURE=f'{i * 10}℃', TIME=f'{i}小时')
        reactions.append(rx)
    return record(cid, reactions), chunk


@pytest.mark.parametrize('title', ['1. 中间体104-1的合成', '[1125] 2. 化合物104的合成',
                                  '5.化合物101的合成', '2、化合物19的制备',
                                  '3．中间体A-12的合成'])
def test_numbered_identity_headings(title):
    assert _product_heading(title)


@pytest.mark.parametrize('title', ['1. 化合物的合成', '1. 产率 95%', '2. IC50 104 nM',
                                  '1. NMR (400 MHz)', '1. 化合物104的合成如下。'])
def test_numbered_lists_are_not_step_headings(title):
    assert not _product_heading(title)


def test_split_explicit_then_analogy_and_analogy_then_explicit():
    c = source_chunk('example_104', [
        '[1122] 1. 中间体104-1的合成', '加入048-2，搅拌得到104-2。',
        '[1125] 2. 化合物104的合成', '[1126]',
        '化合物104的实验步骤及反应条件与上述实施例101中第二步至第五步的化学反应相同。'])
    assert len(split_steps(c)) == 2
    c = source_chunk('example_102', [
        '1. 中间体102-1的合成', '实验步骤与实施例101中第一步相同。',
        '2. 中间体102-2的合成', '加入102-1，搅拌得到102-2。'])
    steps = split_steps(c)
    assert len(steps) == 2
    assert '第一步' not in steps[1]['local_text']


@pytest.mark.parametrize('text,expected', [
    ('第一步', (1,)), ('第二步至第五步', (2, 3, 4, 5)),
    ('第十一至第十三步', (11, 12, 13)), ('第一、二步', (1, 2)),
    ('第二步和第三步', (2, 3)), ('第三步及第五步', (3, 5)),
    ('Steps 2-5', (2, 3, 4, 5)), ('Steps 2 and 4', (2, 4)),
    ('第一至第三步及第五步', (1, 2, 3, 5)), ('Steps 1-3 and 5', (1, 2, 3, 5)),
    ('第九步至第二步', ()), ('同样方法', None), ('第末步', ())])
def test_step_selectors(text, expected):
    assert referenced_step_numbers(text) == expected


def test_two_template_citations_keep_their_own_selectors():
    text = '两个实验步骤分别与实施例8中第二步以及实施例22中第四步的化学反应相同。'
    assert [(c['id'], c['steps']) for c in procedure_citations(text)] == [
        ('example_8', (2,)), ('example_22', (4,))]
    for text in ['实验步骤与实施例8中第一步及实施例8中第三步相同。',
                 '实验步骤与实施例8中第一步和第三步相同。']:
        assert procedure_citations(text)[0]['steps'] == (1, 3)
    assert procedure_citations('步骤与实施例101中第二\n步至第五步的化学反应相同。')[0]['steps'] == (2, 3, 4, 5)
    assert procedure_citations('The reactant was prepared as in Example 2.') == []


def test_replacement_mentions_do_not_erase_method_step_ranges():
    text = ('与实施例55中第三步至第五步的化学反应相同,不同的是这里用中间体039-5'
            '取代了实施例55中的中间体055-3,用中间体057-3取代了实施例55中的中间体001-10。')
    assert procedure_citations(text)[0]['steps'] == (3, 4, 5)
    assert _substitutions(text) == [('中间体055-3', '中间体039-5'), ('中间体001-10', '中间体057-3')]
    assert _substitutions("用N,N'-二甲基乙二胺取代了实施例1中的乙二胺,用THF取代了实施例1中的DMF。") == [
        ('乙二胺', "N,N'-二甲基乙二胺"), ('DMF', 'THF')]


def test_replace_material_alias_and_named_reagent_without_substring_matches():
    items = [compound('101-1', 'CC'), compound('101-10', 'CCC')]
    items[0]['image_source'] = {'dict_key': '101-1'}
    _apply_substitution(items, '中间体101-1', '中间体104-2')
    assert items[0]['name'] == '中间体104-2'
    assert not items[0].get('smiles') and not items[0].get('image_source')
    assert items[1]['smiles'] == 'CCC'
    assert _substitutions('不同的是这里用异丙基磺酰氯取代了实施例101中的乙基磺酰氯。') == [
        ('乙基磺酰氯', '异丙基磺酰氯')]
    assert _substitutions("将N,N',N'-三甲基-1,2-乙二胺替换为N-甲基哌嗪。") == [
        ("N,N',N'-三甲基-1,2-乙二胺", 'N-甲基哌嗪')]
    assert _substitutions('合成化合物10的步骤与实施例8第一步至第三步完全一样,唯一不同的是这里第一步中的原料010-1取代了实施例8第一步中的原料008-1。') == [
        ('原料008-1', '原料010-1')]
    assert _substitutions('唯一不同处是这里的反应起始物036-2取代了实施例6中的006-2。') == [
        ('006-2', '036-2')]


def mixed_target():
    chunk = source_chunk('example_104', [
        '[1122] 1. 中间体104-1的合成', '加入048-2，搅拌得到104-2。',
        '[1125] 2. 化合物104的合成',
        '化合物104及其甲磺酸盐合成的实验步骤及反应条件与上述实施例101中第二步至第五步的化学反应相同。'
        '不同的是这里用中间体104-2取代了实施例101中的中间体101-1。得化合物104，产率31%。LCMS:87.1。'])
    chunk['steps'] = split_steps(chunk)
    first = full_reaction(1, [compound('048-2', 'C')], [compound('104-2', 'CCCC')], 'explicit')
    first.update(step_id=chunk['steps'][0]['step_id'], step_source=chunk['steps'][0])
    thin = full_reaction(2, [compound('104-2')], [compound('化合物104', 'CCCCCC')], 'analogy')
    thin.update(step_id=chunk['steps'][1]['step_id'], step_source=chunk['steps'][1])
    return record('example_104', [first, thin]), chunk


def test_104_expands_only_second_step_with_local_terminal_evidence_and_unique_intermediates():
    base, base_chunk = template()
    target, chunk = mixed_target()
    first = deepcopy(target['reactions'][0])
    result = apply_analogy_repair([base, target], [base_chunk, chunk])
    assert result['instantiated']
    assert len(target['reactions']) == 5
    assert target['reactions'][0] == first
    clones = target['reactions'][1:]
    assert [r['analogical_instantiation']['template_reaction_index'] for r in clones] == [2, 3, 4, 5]
    assert compound_lists(clones[0])['reactants'][0]['name'] == '104-2'
    for before, after in zip(clones, clones[1:]):
        product = compound_lists(before)['products'][0]
        assert product['name'] == compound_lists(after)['reactants'][0]['name']
        assert not product.get('smiles') and not product.get('canonical_smiles')
        assert before['coref']['ner']['YIELD_PERCENT'] is None
        assert before['validation']['rxn_validation']['mass_check']['status'] != 'mismatch'
    final = compound_lists(clones[-1])['products'][0]
    assert final['name'] == '化合物104' and final['smiles'] == 'CCCCCC'
    assert clones[-1]['coref']['ner']['YIELD_PERCENT'] == '31'
    saved = deepcopy(target)
    assert apply_analogy_repair([base, target], [base_chunk, chunk]) is None
    assert target == saved
    routes, fragments = assemble_routes_v2([scope_doc('doc.pdf', [base, target])])
    route = next(r for r in routes + fragments if r['terminal'].ref['id'] == target['id'])
    assert [u.ref['index'] for u in route['selected'] if u.ref['id'] == target['id']] == [1, 2, 3, 4, 5]
    assert all(c['from_ref']['id'] == 'example_104' for c in route['connections'])


@pytest.mark.parametrize('name', ['化合物104及其甲磺酸盐', '化合物104的磺酸盐104.(MSOH)₂',
                                  '化合物104及其甲磺酸盐104.(M)SOH)₂',
                                  r'化合物104及其甲磺酸盐104. $(\mathrm{MSOH})_{2}$'])
def test_expansion_preserves_resolved_local_parent_and_salt_products(name):
    base, base_chunk = template()
    target, chunk = mixed_target()
    local = compound(name, 'CCCCCC', structure_source='image_identifier')
    rx = target['reactions'][1]
    rx['validation']['rxn_validation']['normalized_compounds']['products'] = [deepcopy(local)]
    rx['coref']['ner']['REACTION_PRODUCT'] = [deepcopy(local)]
    apply_analogy_repair([base, target], [base_chunk, chunk])
    assert compound_lists(target['reactions'][-1])['products'] == [local]
    assert target['terminal_product_coverage'] == 'supported'


def test_empty_title_uses_explicit_unique_body_target_without_guessing_example_id():
    base, base_chunk = template()
    chunk = analogy_chunk('example_68', '',
        '化合物68及其甲磺酸盐合成的步骤与实施例101中第三步至第五步相同。化合物68的分析数据：LCMS 123。')
    target = record('example_68', [])
    apply_analogy_repair([base, target], [base_chunk, chunk])
    assert len(target['reactions']) == 3
    assert compound_lists(target['reactions'][-1])['products'][0]['name'] == '化合物68'
    for text in ['与实施例101中第三步至第五步相同。',
                 '化合物68的分析数据与化合物69的分析数据。与实施例101中第三步至第五步相同。']:
        unknown = analogy_chunk('example_70', '', text)
        current = record('example_70', [])
        assert apply_analogy_repair([base, current], [base_chunk, unknown]) is None
        assert current['reactions'] == []


def test_template_image_dictionary_resolves_a_numbered_substitution_to_named_reagent():
    base, base_chunk = template(count=3)
    base['image'] = {'dict_result': {'001-10': 'NCCN'}}
    compounds = base['reactions'][1]['validation']['rxn_validation']['normalized_compounds']
    compounds['reagents'] = [compound('乙二胺', 'NCCN')]
    text = ('化合物61的合成与实施例101中第二步至第三步相同，'
            '用中间体057-3取代了实施例101中的中间体001-10。')
    chunk = analogy_chunk('example_61', '化合物61', text)
    target = record('example_61', [])
    apply_analogy_repair([base, target], [base_chunk, chunk])
    assert len(target['reactions']) == 2
    replaced = compound_lists(target['reactions'][0])['reagents'][0]
    assert replaced['name'] == '中间体057-3' and not replaced.get('smiles')
    provenance = target['reactions'][0]['analogical_instantiation']['substitution_aliases'][0]
    assert provenance['identifier'] == '中间体001-10'
    assert provenance['basis'] == 'template_image_dictionary_structure'
    base['image']['conflicting_identifiers'] = ['001-10']
    unresolved = record('example_61', [])
    assert apply_analogy_repair([base, unresolved], [base_chunk, chunk]) is None
    assert not unresolved['reactions']


def test_missing_printed_step_does_not_select_by_array_position():
    base, base_chunk = template()
    base['reactions'].pop(2)
    target, chunk = mixed_target()
    original = deepcopy(target)
    assert apply_analogy_repair([base, target], [base_chunk, chunk]) is None
    assert target == original


def test_reversed_dependency_chain_longer_than_four_rounds():
    base, base_chunk = template(count=1)
    records, chunks = [base], [base_chunk]
    previous = 'example_101'
    for n in range(102, 109):
        cid = f'example_{n}'
        chunk = analogy_chunk(cid, f'化合物{n}', f'化合物{n}的实验步骤与实施例{previous[8:]}相同。')
        records.append(record(cid, []))
        chunks.append(chunk)
        previous = cid
    result = apply_analogy_repair(list(reversed(records)), chunks)
    assert len(result['instantiated']) == 7
    assert all(r['reactions'] for r in records)
    assert apply_analogy_repair(records, chunks) is None


def test_cycles_and_following_compound_introductions_leave_current_extraction():
    chunks = [analogy_chunk('example_1', '化合物1', '与实施例2的方法相同。'),
              analogy_chunk('example_2', '化合物2', '与实施例1的方法相同。')]
    records = [record(c['id'], [full_reaction(1, [compound('A')], [compound(c['title'])], c['text'])])
               for c in chunks]
    saved = deepcopy(records)
    assert apply_analogy_repair(records, chunks) is None
    assert records == saved
    chunk = analogy_chunk('example_f_1', 'Target F-1',
        'Compound 49 was converted to Target F-1. According to the same manner as that of Example F-1, '
        'the following Example compounds F-2 to F-63 were synthesized.')
    current = record('example_f_1', [full_reaction(1, [compound('49')], [compound('Target F-1')], chunk['text'])])
    saved = deepcopy(current)
    assert apply_analogy_repair([current], [chunk]) is None
    assert current == saved


def test_explicit_local_substrates_and_unknown_conditions_keep_existing_ner():
    base, base_chunk = template(count=1)
    for tail in ('使用本地原料作为起始原料。', '唯一不同的是这里的温度改为50℃。'):
        chunk = analogy_chunk('example_104', '化合物104', '与实施例101的方法相同。' + tail)
        target = record('example_104', [full_reaction(1, [compound('本地原料')],
                                                   [compound('化合物104', 'CCCC')], chunk['text'])])
        saved = deepcopy(target)
        assert apply_analogy_repair([base, target], [base_chunk, chunk]) is None
        assert target == saved


def test_existing_single_step_keeps_resolved_structures_and_material_names():
    base, base_chunk = template(count=1)
    chunk = analogy_chunk('example_104', '化合物104', '与实施例101相同。')
    target = record('example_104', [full_reaction(1, [compound('本例底物', 'CCCC')],
                                               [compound('化合物104', 'CCCCCC')], chunk['text'])])
    original = deepcopy(target)
    assert apply_analogy_repair([base, target], [base_chunk, chunk]) is None
    assert target == original


def test_single_step_solvent_override_does_not_expand_entire_template():
    base, base_chunk = template(count=3)
    chunk = analogy_chunk('example_29', '中间体029-2',
        '合成实验步骤与实施例101中第一步相同。唯一不同处是这里用的是无水四氢呋喃作溶剂，而实施例101中用的是DMF作溶剂。')
    target = record('example_29', [])
    apply_analogy_repair([base, target], [base_chunk, chunk])
    assert len(target['reactions']) == 1
    assert [c['name'] for c in compound_lists(target['reactions'][0])['solvents']] == ['无水四氢呋喃']


def test_inherited_evidence_is_scoped_to_the_requested_steps():
    _base, chunk = template()
    entry = {'id': chunk['id'], 'text': chunk['text'], 'blocks': chunk['source_parts']}
    refs, issues = resolve_programs('实验步骤与实施例101中第二步至第四步相同。', {'example_101': [entry]})
    assert not issues
    assert '10℃' not in refs[0]['text'] and '50℃' not in refs[0]['text']
    assert '20℃' in refs[0]['text'] and '40℃' in refs[0]['text']


def test_a_preexisting_instance_does_not_block_another_step():
    base, base_chunk = template(count=1)
    chunk = source_chunk('example_104', ['1. 化合物104-1的合成', '与实施例101相同。',
                                         '2. 化合物104-2的合成', '与实施例101相同。'])
    chunk['steps'] = split_steps(chunk)
    saved = full_reaction(1, [compound('A')], [compound('104-1')], 'old instance')
    saved.update(step_id=chunk['steps'][0]['step_id'], step_source=chunk['steps'][0],
                 analogical_instantiation={'template_chunk_id': 'example_101'})
    target = record('example_104', [saved])
    apply_analogy_repair([base, target], [base_chunk, chunk])
    assert len(target['reactions']) == 2
    assert target['reactions'][0]['index'] == 1
    assert compound_lists(target['reactions'][1])['products'][0]['name'] == '化合物104-2'


def test_sequential_different_templates_connect_through_the_unique_coded_substrate():
    first, first_chunk = template('example_8', count=2)
    second, second_chunk = template('example_22', count=4)
    for r, old, new in [(first['reactions'][1], '008-2', '008-3'),
                         (second['reactions'][3], '022-6', '化合物22')]:
        for layer in ('coref', 'validation'):
            r[layer]['ner'] = dict(STARTING_MATERIAL=[compound(old)], REACTION_PRODUCT=[compound(new)])
        normalized = r['validation']['rxn_validation']['normalized_compounds']
        normalized.update(reactants=[compound(old)], products=[compound(new)])
    chunk = source_chunk('example_28', ['1. 中间体028-2的合成', '加入028-1，得到028-2。',
        '2. 化合物28的合成', '两个步骤分别与实施例8中第二步以及实施例22中第四步的化学反应相同。'])
    chunk['steps'] = split_steps(chunk)
    explicit = full_reaction(1, [compound('028-1')], [compound('028-2')], 'explicit')
    explicit.update(step_id=chunk['steps'][0]['step_id'], step_source=chunk['steps'][0])
    target = record('example_28', [explicit])
    apply_analogy_repair([first, second, target], [first_chunk, second_chunk, chunk])
    assert len(target['reactions']) == 3
    a, b = target['reactions'][1:]
    assert compound_lists(a)['reactants'][0]['name'] == '028-2'
    assert compound_lists(b)['reactants'][0]['name'] == compound_lists(a)['products'][0]['name']
    assert compound_lists(b)['products'][0]['name'] == '化合物28'


def test_next_source_step_uses_only_the_previous_expansion_terminal():
    first, first_chunk = template('example_8', count=2)
    second, second_chunk = template('example_22', count=1)
    chunk = source_chunk('example_23', ['1. 中间体023-3的合成', '与实施例8中第一步至第二步相同。',
                                         '2. 化合物23的合成', '与实施例22中第一步相同。'])
    chunk['steps'] = split_steps(chunk)
    target = record('example_23', [])
    apply_analogy_repair([first, second, target], [first_chunk, second_chunk, chunk])
    assert len(target['reactions']) == 3
    previous, following = target['reactions'][-2:]
    assert compound_lists(previous)['products'][0]['name'] == '中间体023-3'
    assert compound_lists(following)['reactants'][0]['name'] == '中间体023-3'
