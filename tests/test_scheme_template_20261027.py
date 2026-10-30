"""Scheme-template extraction regressions (2026-10-27): a general-scheme
paragraph yields one record per transformation, and the steps assemble into
a reference route chained by letter-code continuity."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import json
import pytest

from pipeline.document.chunk_processing import process_chunk, bind_scheme_antecedents, ground_scheme_compounds
from pipeline.document.route_repair import convert_template_route
from pipeline.document.routes import assemble_routes_v2, assemble_template_routes
from pipeline.document.segmenter import (is_scheme_description, scheme_chunks,
                                         select_procedure_chunks, scheme_codes, load_blocks)
from pipeline.document.step_context import is_template_definition
from pipeline.extraction.parser import parse_action_details
from pipeline.validation.quality import evaluate_quality, QUALITY_VERSION

from test_route_repair import compound, reaction, record, scope_doc

SCHEME_TEXT = ("[0097] 在上述方案A中，二氯嘧啶衍生物(A)在路易斯酸如FeCl3的存在下与富电的三环或二环"
               "化合物(B)发生类富克反应，或者与硼酸酯(C)经由Suzuki偶合反应生成化合物(D)。"
               "化合物(D)与芳胺(E)经由SNAr取代得到(F)。胺(H)可以与丙烯酰氯(I)在碱如DIEA存在下，"
               "在低温直接酰化得式一化合物(K)。")

MARKUSH = {"(A)": "[1*]c1cnc(Cl)nc1Cl", "(D)": "[1*]c1cnc(Cl)nc1[W]",
           "(F)": "[1*]c1cnc(Nc2cc([N+](=O)[O-])c(F)cc2[2*])nc1[W]",
           "(H)": "[1*]c1cnc(Nc2cc(N)c([3*])cc2[2*])nc1[W]",
           "(K)": "*c1cnc(Nc2cc(NC(=O)C=C)c([3*])cc2[2*])nc1[W]"}


def test_scheme_action_verbs_are_supported_operations():
    details = parse_action_details('["偶合","取代","酰化","消除","环化","缩合"]')
    assert details["actions"] == ["偶合", "取代", "酰化", "消除", "环化", "缩合"]
    assert details["status"] == "ok"


def test_scheme_template_prompt_variant_configuration():
    import os
    from pipeline.extraction.extractor import ChemReXExtractor
    from pipeline.prompts.prompt_loader import PromptLoader
    extractor = ChemReXExtractor.__new__(ChemReXExtractor)
    extractor.client = SimpleNamespace(model="test-model", preset=None)
    extractor.loader = PromptLoader(prompts_dir=os.path.join(
        os.path.dirname(os.path.abspath(
            __import__("pipeline.extraction.extractor", fromlist=["__file__"]).__file__)),
        "prompts"))
    config = extractor.configuration("zh", "scheme_template")
    assert config["prompt_variant"] == "scheme_template"
    assert "EVERY chemical transformation" in config["prompts"]["ner"]["content"]
    prompt = extractor._build_prompt("ner", "zh", "方案描述文本", "scheme_template")
    assert "方案描述文本" in prompt and "EVERY chemical transformation" in prompt
    default = extractor.configuration("zh")
    assert "prompt_variant" not in default


def test_scheme_template_variant_runs_on_its_own_model():
    import config
    from pipeline.extraction.extractor import ChemReXExtractor
    from unittest.mock import patch
    # The scheme-template variant defaults to the MaaS deepseek preset while
    # regular NER keeps the configured extraction preset.
    assert config.SCHEME_EXTRACTION_PRESET == "huawei-maas"
    extractor = ChemReXExtractor(variant_preset="qwen3-lora-08")
    assert extractor.variant_preset == "qwen3-lora-08"
    scheme = extractor.scheme_client
    assert scheme is not extractor.client
    assert scheme.preset == "qwen3-lora-08"
    # A scheme narrative yields one JSON record per transformation: the
    # variant client needs room for the whole array.
    assert scheme.max_tokens == 4096
    variant_config = extractor.configuration("zh", "scheme_template")
    assert variant_config["model"] == scheme.model
    assert variant_config["prompt_variant"] == "scheme_template"
    default_config = extractor.configuration("zh")
    assert default_config["model"] == extractor.client.model
    assert "prompt_variant" not in default_config
    # extract_ner routes the variant call to the scheme client.
    with patch.object(extractor.client, "chat") as default_chat, \
            patch.object(extractor.scheme_client, "chat", create=True) as scheme_chat:
        scheme_chat.return_value = '[{"reaction_product": ["化合物(D)"], "starting_material": []}]'
        default_chat.return_value = '{"reaction_product": ["P"], "starting_material": []}'
        extractor.extract_ner("方案描述", "zh", "scheme_template")
        scheme_chat.assert_called_once()
        default_chat.assert_not_called()
        extractor.extract_ner("常规步骤", "zh")
        default_chat.assert_called_once()


def test_quality_gate_marks_scheme_template_records_not_blocked():
    record_kind = {"record_kind": "procedure_template",
                   "scheme_template": True,
                   "step_source": {"record_kind": "procedure_template",
                                   "boundary_status": "source_grounded",
                                   "local_text": SCHEME_TEXT, "context_issues": []},
                   "rxn_validation": {"status": "deferred", "needs_human_review": True},
                   "ner": {}, "action": ["偶合"], "parse_status": {"ner": "ok"}}
    quality = evaluate_quality(record_kind, "needs_review")
    assert "procedure_template_without_specific_instance" not in quality["blocking_issues"]
    assert quality["template_record"] is True
    # Without the scheme-template flag the old blocking behaviour remains.
    legacy = dict(record_kind, scheme_template=False)
    legacy_quality = evaluate_quality(legacy, "needs_review")
    assert "procedure_template_without_specific_instance" in legacy_quality["blocking_issues"]


def template_chunk():
    step = {"step_id": "scheme_248:step:1", "record_kind": "procedure_template",
            "title": "", "local_text": SCHEME_TEXT, "text": SCHEME_TEXT,
            "source_parts": [{"text": SCHEME_TEXT, "block_idx": 248, "page_idx": 20}],
            "procedure_references": [], "context_issues": [],
            "boundary_status": "source_grounded", "unit_kind": "single_step",
            "process_kind": "synthesis"}
    return {"id": "scheme_248", "reference_id": "scheme_248",
            "kind": "scheme", "process_kind": "synthesis", "title": "",
            "position": {"page_idx": 20, "bbox": [0, 0, 1, 1],
                         "match_method": "test", "pages": [21]},
            "text": SCHEME_TEXT, "text_parts": [SCHEME_TEXT],
            "steps": [step], "block_indices": [248], "images": []}


def scheme_extraction():
    def step(product, reactants, action):
        return {"reaction_product": [product], "starting_material": reactants,
                "reagent_catalyst": ["DIEA"], "solvent": [], "temperature": None,
                "time": None, "yield_percent": None, "action": action, "step_order": 1}
    return {"language": "zh", "task": "ner", "ner": {},
            "ner_records": [
                step("化合物(D)", ["二氯嘧啶衍生物(A)", "三环化合物(B)"], "偶合"),
                step("化合物(F)", ["化合物(D)", "芳胺(E)"], "取代"),
                step("式一化合物(K)", ["胺(H)", "丙烯酰氯(I)"], "酰化")],
            "action": [], "raw": {"ner": "[]"}, "parse_status": {"ner": "ok"},
            "extraction_configuration": {"prompt_variant": "scheme_template"}}


def resolved_coref(record):
    """Coref output as if the image dictionary resolved every code."""
    def resolve(name):
        for code, smiles in MARKUSH.items():
            if code.strip("()") in name:
                return {"name": name, "smiles": smiles, "dict_key": code,
                        "structure_source": "image_identifier"}
        return {"name": name}
    return {"ner": {"STARTING_MATERIAL": [resolve(n) for n in record["starting_material"]],
                    "REACTION_PRODUCT": [resolve(n) for n in record["reaction_product"]]},
            "standardize_status": "applied",
            "id_smiles_mapping": {}, "unresolved_identifiers": []}


def test_process_chunk_routes_scheme_template_and_keeps_per_record_actions():
    extraction = scheme_extraction()
    extract = Mock(return_value=extraction)
    resolves = iter([resolved_coref(record) for record in extraction["ner_records"]])
    validation = {"rxn_validation": {"status": "deferred", "needs_human_review": True,
                                     "normalized_compounds": {"reactants": [], "products": [],
                                                              "reagents": [], "solvents": [],
                                                              "participants": []}}}
    pipe = SimpleNamespace(
        _get_extraction=lambda: SimpleNamespace(extract=extract),
        _get_coref=lambda: SimpleNamespace(resolve=Mock(side_effect=lambda payload: deepcopy(next(resolves)))),
        _get_validation=lambda: SimpleNamespace(validate_record=Mock(return_value=deepcopy(validation))))
    result = process_chunk(pipe, template_chunk(), "zh", dict(MARKUSH))
    # The scheme-template prompt variant was routed with NER-only extraction.
    assert extract.call_args.kwargs.get("prompt_variant") == "scheme_template"
    assert extract.call_args.kwargs.get("task") == "ner"
    assert len(result["reactions"]) == 3
    actions = [r["extraction"]["action"] for r in result["reactions"]]
    assert actions == [["偶合"], ["取代"], ["酰化"]]
    for reaction in result["reactions"]:
        issues = reaction["quality"]["blocking_issues"]
        assert "multiple_records_without_step_alignment" not in issues
        assert "procedure_template_without_specific_instance" not in issues
        assert reaction["quality"]["template_record"] is True


def test_assemble_template_routes_chains_letter_codes():
    def template_reaction(index, reactants, products, action):
        rx = reaction(index, reactants, products, text=SCHEME_TEXT,
                      record_kind="procedure_template")
        rx["coref"] = {"ner": {"STARTING_MATERIAL": deepcopy(reactants),
                               "REACTION_PRODUCT": deepcopy(products)},
                       "action": [action], "record_kind": "procedure_template",
                       "step_source": rx["step_source"]}
        return rx

    a = compound("二氯嘧啶衍生物(A)", MARKUSH["(A)"], dict_key="(A)")
    b = compound("三环化合物(B)", "CC1(C)OB([W])OC1(C)C", dict_key="(B)")
    d = compound("化合物(D)", MARKUSH["(D)"], dict_key="(D)")
    e = compound("芳胺(E)", "[2*]c1cc(F)c([N+](=O)[O-])cc1N", dict_key="(E)")
    f = compound("化合物(F)", MARKUSH["(F)"], dict_key="(F)")
    diamine = compound("N1,N1二甲基乙烷-1,2-二胺", "CNCCN(C)C")
    g = compound("化合物(G)", "*c1cnc(Nc2cc([N+](=O)[O-])c(NCCN(C)C)cc2[2*])nc1[W]", dict_key="(G)")
    h = compound("胺(H)", MARKUSH["(H)"], dict_key="(H)")
    i = compound("丙烯酰氯(I)", "C=CC(=O)Cl", dict_key="(I)")
    k = compound("式一化合物(K)", MARKUSH["(K)"], dict_key="(K)")
    record_row = record("fallback_248_250_1", [
        template_reaction(1, [a, b], [d], "偶合"),
        template_reaction(2, [d, e], [f], "取代"),
        template_reaction(3, [f, diamine], [g], "反应"),
        template_reaction(4, [g], [h], "还原"),
        template_reaction(5, [h, i], [k], "酰化"),
    ])
    documents = [scope_doc("doc.pdf", [record_row])]
    routes = assemble_template_routes(documents)
    assert len(routes) == 1
    route = routes[0]
    assert route["kind"] == "reference_route"
    assert route["completeness"] == "complete"
    codes = [edge["code"] for edge in route["connections"]]
    assert codes == ["D", "F", "G", "H"]
    # Template steps never leak into the concrete route graph.
    concrete_routes, concrete_fragments = assemble_routes_v2(documents)
    assert not concrete_routes and not concrete_fragments
    # The converter emits a route-v2 reference-route row.
    row = convert_template_route({"patent_key": "p", "scope_id": "s"}, route)
    assert row["kind"] == "reference_route" and row["schema_version"] == "route-v2"
    assert len(row["steps"]) == 5 and len(row["connections"]) == 4
    assert "[1*]" in row["target"]["identity"] or "K" in str(row["target"]["display"])


def test_is_scheme_description_detects_conceptual_narratives():
    assert is_scheme_description(SCHEME_TEXT)
    us7365205 = ("A mixed acid anhydride, acid halide, activated ester or the like, which is "
                 "derived from carboxylic acid (3), may react with diamine (2), giving compound (4). "
                 "The resultant compound (4) may react with carboxylic acid (5) under the same "
                 "conditions, giving compound (1) according to the present invention.")
    assert is_scheme_description(us7365205)
    us7943788 = ("The compound of the formula II may be prepared by reacting the compound of the "
                 "formula III with the compound of the formula IV, and the compound of the formula I "
                 "may be prepared by deprotecting the compound of the formula II according to the "
                 "following Processes.")
    assert is_scheme_description(us7943788)
    # A worked example with amounts is never a scheme narrative.
    worked = ("将2-甲氧基-4-氟硝基苯(100.00克，584.35毫摩尔)溶于甲醇(400mL)中，加入Pd/C(10克)，"
              "得到标题化合物(83.00克，产率93.28%)。")
    assert not is_scheme_description(worked)


def test_scheme_chunks_collect_uncovered_narratives():
    blocks = [
        {"type": "text", "text": "Field of the invention.", "page_idx": 0},
        {"type": "text", "text": "The compound of the formula II may be prepared by reacting the "
                                 "compound of the formula III with the compound of the formula IV, "
                                 "and the compound of the formula I may be prepared by deprotecting "
                                 "the compound of the formula II.", "page_idx": 0},
        {"type": "image", "img_path": "images/scheme.jpg", "page_idx": 0},
        {"type": "text", "text": "Example 1", "page_idx": 1},
        {"type": "text", "text": "Compound A (2 g) was stirred in ethanol (5 mL) for 2 h.",
         "page_idx": 1},
    ]
    chunks = select_procedure_chunks(blocks, "sample.json", [])
    scheme = [c for c in chunks if c["kind"] == "scheme"]
    assert len(scheme) == 1
    assert scheme[0]["images"] == ["images/scheme.jpg"]
    assert is_template_definition("scheme", "", [{"text": scheme[0]["text"]}])
    # A narrative already covered by a fallback procedure chunk is not
    # collected twice.
    titled = [{"type": "text", "text": "Example 1", "page_idx": 1},
              {"type": "text", "text": "Compound A (2 g) was stirred in ethanol (5 mL) for 2 h.",
               "page_idx": 1}]
    segmented = select_procedure_chunks(blocks, "sample.json", [])
    fallback_ids = [c["id"] for c in segmented if c["kind"] == "fallback"]
    scheme_ids = [c["id"] for c in segmented if c["kind"] == "scheme"]
    assert not (set(fallback_ids) & set(scheme_ids))
    assert all(c["kind"] != "scheme" for c in select_procedure_chunks(titled, "s.json", []))


def _template_reaction(index, reactants, products, action):
    rx = reaction(index, reactants, products, text=SCHEME_TEXT,
                  record_kind="procedure_template")
    rx["coref"] = {"ner": {"STARTING_MATERIAL": deepcopy(reactants),
                           "REACTION_PRODUCT": deepcopy(products)},
                   "action": [action], "record_kind": "procedure_template",
                   "step_source": rx["step_source"]}
    return rx


def test_template_routes_chain_numbered_formula_codes():
    """通式(9)→通式(10): numbered formula codes chain like letter codes."""
    nine = compound("通式(9)的化合物", "[1*]c1ccccc1O", dict_key="(9)")
    ten = compound("通式(10)的化合物", "[1*]c1ccccc1Br", dict_key="(10)")
    eleven = compound("通式(11)的化合物", "[1*]c1ccccc1N", dict_key="(11)")
    record_row = record("scheme_380", [
        _template_reaction(1, [nine], [ten], "取代"),
        _template_reaction(2, [ten], [eleven], "取代"),
    ])
    routes = assemble_template_routes([scope_doc("doc.pdf", [record_row])])
    assert len(routes) == 1
    assert [e["code"] for e in routes[0]["connections"]] == ["10"]
    assert routes[0]["completeness"] == "complete"


def test_template_code_accepts_formula_word_codes():
    """US style "compound of formula V" carries its code in the word formula;
    the -a suffix stays part of the identity (II-a is not II)."""
    from pipeline.document.routes import _template_code
    assert _template_code({"name": "compound of the formula V"}) == "V"
    assert _template_code({"name": "compounds of the formula II-a"}) == "II-A"
    assert _template_code({"name": "compound of the formula II"}) == "II"
    assert _template_code({"name": "式一化合物(K)"}) == "K"
    vii = compound("compound of the formula VII", "O=C(O)c1ccccc1")
    vi = compound("compound of the formula VI", "Nc1ccccc1")
    five = compound("compound of formula V", "O=C(Nc1ccccc1)c1ccccc1")
    two = compound("compound of the formula II", "CC(=O)Nc1ccccc1")
    record_row = record("scheme_430", [
        _template_reaction(1, [vii, vi], [five], "condensation"),
        _template_reaction(2, [five], [two], "condensation"),
    ])
    routes = assemble_template_routes([scope_doc("doc.pdf", [record_row])])
    assert len(routes) == 1
    assert [e["code"] for e in routes[0]["connections"]] == ["V"]


def test_template_route_target_is_a_coded_chain_end():
    """A scheme often closes with alternative-activation steps whose products
    are uncoded auxiliaries; the route target is the coded chain end, not the
    narrative's last sentence."""
    diamine = compound("diamine (2)", "NCCN", dict_key="(2)")
    four = compound("compound (4)", "O=C(NCCN)c1ccccc1", dict_key="(4)")
    acid5 = compound("carboxylic acid (5)", "O=C(O)c1ccccc1", dict_key="(5)")
    one = compound("compound (1)", "CC(=O)NCCNc1ccccc1", dict_key="(1)")
    acid3 = compound("carboxylic acid (3)", "O=C(O)CCc1ccccc1", dict_key="(3)")
    anhydride = compound("mixed acid anhydride", "")
    ester = compound("activated ester", "")
    record_row = record("scheme_405", [
        _template_reaction(1, [anhydride, diamine], [four], "reaction"),
        _template_reaction(2, [compound("acid halide", ""), diamine], [four], "reaction"),
        _template_reaction(3, [ester, diamine], [four], "reaction"),
        _template_reaction(4, [four, acid5], [one], "condensation"),
        _template_reaction(5, [acid3, compound("ethyl chloroformate", "")],
                           [anhydride], "reaction"),
        _template_reaction(6, [acid3, compound("p-nitrophenol", "")],
                           [ester], "reaction"),
    ])
    routes = assemble_template_routes([scope_doc("doc.pdf", [record_row])])
    assert len(routes) == 1
    route = routes[0]
    # The main chain (2)→(4)→(1) connects; the uncoded activation steps 5-6
    # are carried as gaps instead of discarding the whole route.
    assert [e["code"] for e in route["connections"]] == ["4", "4", "4"]
    assert route["terminal"].ref["index"] == 4
    assert "compound (1)" in str(route["target"]["display"])
    # Parallel preparations are explicit alternatives, not disconnected roots;
    # uncoded activation auxiliaries remain visible as evidence gaps.
    assert {g["reaction_ids"][0]["index"] for g in route["gaps"]} == {5, 6}
    assert route['alternatives'][0]['code'] == '4'
    assert [ref['index'] for ref in route['alternatives'][0]['producers']] == [1, 2, 3]


@pytest.mark.parametrize('label,code', [
    ('化合物A1', 'A1'), ('产品B3', 'B3'), ('苯胺A7', 'A7'), ('(VIII)', 'VIII'),
    ('(XVIII)', 'XVIII'), ('(XIV-16)', 'XIV-16'), ('(I-20a)', 'I-20A'),
    ('(2a)', '2A'), ('formula VI', 'VI'), ('formula XXXV', 'XXXV'),
    ('compound XVII', 'XVII'), ('Compound XVII', 'XVII'), ('通式4', '4')])
def test_scheme_detection_and_route_linking_share_complete_identifiers(label, code):
    from pipeline.document.routes import _template_code
    assert scheme_codes(label) == [code]
    assert _template_code({'name': label}) == code


def test_scheme_words_and_code_context_do_not_match_structural_descriptions():
    text = ('(1) The main structure, condensed heterocycle, is substituted with oxo (=O), hydroxyl (OH).\n'
            '(2) A substituted carbamoyl group is attached on the condensed heterocycle.')
    assert not is_scheme_description(text)
    assert scheme_codes('溶剂 (THF) (DMF)，氯化钯(II)，R1和R2') == []
    assert scheme_codes('由卤素取代的C1-C3烷基或C3-C6环烷基') == []
    assert not is_scheme_description('通式(I)的化合物可以(R)或(S)构型存在，得到外消旋混合物。')
    assert not is_scheme_description('树胶，(2)天然存在的磷脂，(3)所述偏酯与环氧乙烷的缩合产物。')
    assert not is_scheme_description('The optionally condensed heterocycle includes compound (I) and compound (II).')
    assert scheme_codes('化合物9和10转化为通式4') == ['9', '10', '4']
    assert not is_scheme_description('1 → 2 → 3')


def test_single_transformation_scheme_and_list_formulation_boundary():
    text = '从中间体E1也可以通过与丙烯酸酐E2反应生成化合物E3。'
    assert is_scheme_description(text)
    assert is_scheme_description('Reacting a compound (XIV) with compound (XV) provides compound (XVI).')
    assert is_scheme_description('化合物9转化为化合物10。')
    blocks = [{'type': 'list', 'page_idx': 0, 'list_items': [
        '[0103] 其中X如上所述。', '[0104] ' + text,
        '[0105] 本申请的第三方面，提供一种药物组合物。']}]
    chunks = select_procedure_chunks(blocks, 'sample.json', [])
    schemes = [c for c in chunks if c['kind'] == 'scheme']
    assert len(schemes) == 1
    assert '[0104]' in schemes[0]['text'] and '[0105]' not in schemes[0]['text']
    assert all('[0105]' not in s['local_text'] for s in schemes[0]['steps'])


def test_missing_markdown_scheme_keeps_original_indices_and_source_provenance(tmp_path):
    blocks = [{'type': 'image', 'page_idx': 2, 'img_path': 'scheme.jpg', 'image_caption': ['[0093]']},
              {'type': 'text', 'page_idx': 3, 'text': '[0096] 下一通用反应序列。'}]
    source = tmp_path / 'patent_content_list.json'
    source.write_text(json.dumps(blocks))
    markdown = tmp_path / 'patent.md'
    markdown.write_text('[0093] 反应式\n\n[0095] 化合物A1与化合物A2反应生成产品B1。\n\n[0096] 下一反应式\n')
    recovered = load_blocks(str(source))
    assert len(recovered) == len(blocks)
    assert all({k: b[k] for k in original} == original for b, original in zip(recovered, blocks))
    chunks = select_procedure_chunks(recovered, str(source), [])
    scheme = next(c for c in chunks if c['kind'] == 'scheme')
    assert scheme['block_indices'] == [0]
    assert scheme['images'] == ['scheme.jpg']
    evidence = scheme['steps'][0]['source_parts'][0]
    assert evidence['markdown_source'] == str(markdown)
    assert evidence['paragraph_number'] == '0095' and evidence['block_idx'] == 0
    # An unanchored paragraph cannot be assigned to a source page.
    markdown.write_text('[0095] 化合物A1与化合物A2反应生成产品B1。\n')
    assert load_blocks(str(source)) == blocks


def test_single_step_and_independent_salt_terminal_routes():
    from pipeline.document.routes import _template_code
    def material(code):
        return compound(('甲磺酸盐' if code == 'B4' else '盐酸盐' if code == 'B5' else '化合物') + code, '')
    source = record('scheme_e', [_template_reaction(1, [material('E1'), material('E2')], [material('E3')], '酰化')])
    routes = assemble_template_routes([scope_doc('doc', [source])])
    assert len(routes) == 1 and len(routes[0]['selected']) == 1
    assert routes[0]['completeness'] == 'complete'
    source = record('scheme_b', [
        _template_reaction(1, [material('B1')], [material('B2')], '还原'),
        _template_reaction(2, [material('B2')], [material('B3')], '酰化'),
        _template_reaction(3, [material('B3')], [material('B4')], '成盐'),
        _template_reaction(4, [material('B3')], [material('B5')], '成盐')])
    routes = assemble_template_routes([scope_doc('doc', [source])])
    assert len(routes) == 2
    assert [[u.ref['index'] for u in r['selected']] for r in routes] == [[1, 2, 3], [1, 2, 4]]
    assert {_template_code(r['terminal'].compounds['products'][0]) for r in routes} == {'B4', 'B5'}
    exported = [convert_template_route({'patent_key': 'p', 'scope_id': 's'}, r) for r in routes]
    assert len({r['route_id'] for r in exported}) == 2
    assert all(r['target']['basis'] == 'template_code' for r in exported)
    assert {r['target']['display']['name'] for r in exported} == {'甲磺酸盐B4', '盐酸盐B5'}


def test_scheme_functional_group_antecedents_use_explicit_previous_products():
    records = [
        {'STARTING_MATERIAL': ['A5'], 'REACTION_PRODUCT': ['A6']},
        {'STARTING_MATERIAL': ['硝基苯'], 'REACTION_PRODUCT': ['苯胺A7']},
        {'STARTING_MATERIAL': ['硝基苯'], 'REACTION_PRODUCT': ['苯胺A7']},
        {'STARTING_MATERIAL': ['苯胺', 'A8'], 'REACTION_PRODUCT': ['A9']}]
    text = ('二级胺取代A5中的氟原子而得到产品A6。催化氢化或铁粉还原将硝基苯转化成苯胺A7。'
            '在和丙烯酰氯A8反应后，苯胺生成最终产物A9。')
    bound, repairs = bind_scheme_antecedents(records, text)
    assert bound[1]['STARTING_MATERIAL'] == bound[2]['STARTING_MATERIAL'] == ['A6']
    assert bound[3]['STARTING_MATERIAL'] == ['A7', 'A8']
    assert records[1]['STARTING_MATERIAL'] == ['硝基苯']
    assert repairs[1][0]['basis'] == 'unique_preceding_scheme_product'
    ambiguous = text.replace('得到产品A6', '得到产品A5和A6')
    assert bind_scheme_antecedents(records, ambiguous)[0][1]['STARTING_MATERIAL'] == ['硝基苯']
    assert bind_scheme_antecedents(records, '硝基苯转化成苯胺A7。')[0][1]['STARTING_MATERIAL'] == ['硝基苯']


def test_coded_scheme_compounds_cannot_be_resolved_as_literal_deuterium_or_aniline():
    coref = {'ner': {'STARTING_MATERIAL': [
        compound('D2', '[2H][2H]'), compound('苯胺A7', 'Nc1ccccc1')],
        'REACTION_PRODUCT': [compound('化合物D3', 'CC')]}}
    ground_scheme_compounds(coref, {'dict_result': {'D3': '[1*]c1ccccc1'}})
    for molecule in coref['ner']['STARTING_MATERIAL']:
        assert not molecule.get('smiles') and not molecule.get('canonical_smiles')
        assert molecule['structure_source'] == 'scheme_identifier'
    product = coref['ner']['REACTION_PRODUCT'][0]
    assert product['smiles'] == '[1*]c1ccccc1' and not product.get('canonical_smiles')
    assert product['structure_source'] == 'image_identifier'


def test_literature_pointer_paragraphs_are_not_schemes():
    """US8853156-style boilerplate defers every preparation to a citation;
    it carries scheme-shaped codes and verbs but no transformation."""
    pointer = ("With respect to embodiment A, the methods of synthesis for the DPP-4 "
               "inhibitors according to embodiment A of this invention are known to the "
               "skilled person. Advantageously, the DPP-4 inhibitors according to "
               "embodiment A of this invention can be prepared using synthetic methods "
               "as described in the literature. Thus, for example, purine derivatives "
               "of formula (I) can be obtained as described in WO 2002/068420, "
               "WO 2004/018468 or WO 2005/085246. Purine derivatives of formula (II) "
               "can be obtained as described, for example, in WO 2004/050658. Purine "
               "derivatives of formula (III) and (IV) can be obtained as described in "
               "WO 2006/068163.")
    assert not is_scheme_description(pointer)
    # A real scheme may close with "known to the skilled person" as long as it
    # also describes an actual transformation (ZL202080031189.4 style).
    real = ("通式(9)的化合物可以通过还原方法转化为通式(10)的化合物,这些方法是本领域"
            "技术人员已知的。这些还原可以使用下列物质进行:铁粉、氯化铵。")
    assert is_scheme_description(real)
