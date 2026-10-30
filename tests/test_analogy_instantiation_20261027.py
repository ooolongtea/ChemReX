"""Analogical-example repair regressions (2026-10-27): "依据实施例16C的方法，
将实施例16B替换为实施例43B" steps must contribute reactions to the graph."""

from copy import deepcopy

from pipeline.document.analogy_instantiate import apply_analogy_repair
from pipeline.document.routes import assemble_routes_v2
from pipeline.document.step_context import (_bare_general_heading,
                                            is_template_definition)
from pipeline.validation.quality import QUALITY_VERSION

from test_route_repair import compound, reaction, record, scope_doc

ANALOGY_TEXT = ("[1306] 本实施例依据实施例16C的方法，将实施例16B替换为实施例43B。"
                "得标题化合物(230.00毫克，产率 69.54%)。"
                " 1H NMR (400MHz, CD3OD): δ8.56 (br.s., 1H)。")


def full_reaction(index, reactants, products, text, record_kind="reaction_candidate",
                  eligible=True):
    rx = reaction(index, reactants, products, text=text, record_kind=record_kind,
                  eligible=eligible)
    rx["coref"] = {"ner": {"STARTING_MATERIAL": deepcopy(reactants),
                           "REACTION_PRODUCT": deepcopy(products)},
                   "record_kind": record_kind, "action": ["反应"],
                   "yield_info": {"kind": "not_reported", "value": None},
                   "stereochemistry": {"kind": "unreported"},
                   "parse_status": {"ner": "ok", "action": "ok"},
                   "step_source": rx["step_source"]}
    rx["extraction"] = {"ner": {"STARTING_MATERIAL": [c["name"] for c in reactants],
                                "REACTION_PRODUCT": [c["name"] for c in products]}}
    return rx


def analogy_chunk(chunk_id, title, text):
    step = {"step_id": f"{chunk_id}:step:1", "record_kind": "reaction_candidate",
            "title": title, "local_text": text, "procedure_references": [],
            "boundary_status": "source_grounded", "unit_kind": "single_step"}
    return {"id": chunk_id, "reference_id": chunk_id, "kind": "example",
            "title": title, "text": text, "steps": [step]}


def test_instantiates_reaction_for_empty_analogical_example():
    template = record("example_16c", [full_reaction(
        1, [compound("实施例16B", ""), compound("3-氯丙酰氯", "O=C(Cl)CCl")],
        [compound("Compound 16C", "CCO")], "将实施例16B与3-氯丙酰氯反应。")])
    target = record("example_43c", [])
    producer = record("example_43b", [full_reaction(
        1, [], [compound("Compound 43B", "CCC")], "还原得到Compound 43B。")])
    chunk = analogy_chunk("example_43c", "[1304] N-(螺环吲哚基)丙烯酰胺", ANALOGY_TEXT)
    summary = apply_analogy_repair([template, target, producer], [chunk])
    assert summary and summary["instantiated"] == [
        {"chunk_id": "example_43c", "template": "example_16c",
         "substitutions": [["实施例16B", "实施例43B"]]}]
    instantiated = target["reactions"][0]
    assert instantiated["analogical_instantiation"]["template_chunk_id"] == "example_16c"
    # The substituted reactant keeps no stale structure and carries the new name.
    reactant = instantiated["validation"]["rxn_validation"]["normalized_compounds"]["reactants"][0]
    assert reactant["name"] == "实施例43B"
    assert reactant["structure_source"] == "analogical_substitution"
    assert not reactant.get("smiles")
    # The product is this example's own title compound, paragraph tag stripped.
    product = instantiated["validation"]["rxn_validation"]["normalized_compounds"]["products"][0]
    assert product["name"] == "N-(螺环吲哚基)丙烯酰胺"
    # The local yield is parsed into the record.
    assert instantiated["coref"]["ner"]["YIELD_PERCENT"] == "69.54"
    assert instantiated["status"] == "needs_review"


def test_instantiated_reaction_joins_the_route_graph():
    template = record("example_16c", [full_reaction(
        1, [compound("实施例16B", ""), compound("3-氯丙酰氯", "O=C(Cl)CCl")],
        [compound("Compound 16C", "CCO")], "将实施例16B与3-氯丙酰氯反应。")])
    target = record("example_43c", [])
    producer = record("example_43b", [full_reaction(
        1, [], [compound("Compound 43B", "CCC")], "还原得到Compound 43B。")])
    chunk = analogy_chunk("example_43c", "[1304] N-(螺环吲哚基)丙烯酰胺", ANALOGY_TEXT)
    apply_analogy_repair([template, target, producer], [chunk])
    routes, fragments = assemble_routes_v2(
        [scope_doc("doc.pdf", [template, target, producer])])
    target_rows = [row for row in routes + fragments
                   if row["terminal"].ref["id"] == "example_43c"]
    assert target_rows
    edge = next(conn for row in target_rows for conn in row["connections"]
                if conn["to_ref"]["id"] == "example_43c")
    assert edge["from_ref"]["id"] == "example_43b"
    assert edge["basis"] == "material_reference"


def test_reclassifies_misclassified_analogical_instance():
    text = ("[0375] 合成化合物19及其甲磺酸盐19.MsOH的实验步骤及反应条件与实施例8中第一步至第三步"
            "的化学反应基本相同。不同的是这里第一步中的原料019-1取代了实施例8第一步中的原料008-1。"
            " [0376] LRMS (母分子) C31H35N7O2 (ES, m/z): [M+H]+=538。")
    rx = full_reaction(1, [compound("019-1", "C1CCNCC1")],
                       [compound("化合物19", "CCO")], text,
                       record_kind="procedure_template", eligible=False)
    rx["quality"]["blocking_issues"] = ["procedure_template_without_specific_instance"]
    rx["status"] = "needs_review"
    rx["reason"] = "procedure_template_without_specific_instance"
    example = record("example_19", [rx])
    summary = apply_analogy_repair([example], [])
    assert summary and summary["reclassified_chunk_ids"] == ["example_19"]
    assert rx["step_source"]["record_kind"] == "reaction_candidate"
    assert rx["analogical_reclassified"] is True
    assert "procedure_template_without_specific_instance" not in rx["quality"]["blocking_issues"]


def test_general_procedure_is_not_reclassified():
    text = "[0100] 通用程序A：将中间体与胺在DIEA存在下反应，得到目标产物。"
    rx = full_reaction(1, [compound("中间体", "")], [compound("目标产物", "")], text,
                       record_kind="procedure_template", eligible=False)
    rx["quality"]["blocking_issues"] = ["procedure_template_without_specific_instance"]
    general = record("procedure_a", [rx])
    assert apply_analogy_repair([general], []) is None


def test_bare_general_heading_rejects_specific_compound_headings():
    assert _bare_general_heading("1. 化合物19的合成") is None
    assert _bare_general_heading("A. General Procedure for Amide Coupling") is not None
    # The CN105-style chunk is an instance, not a template definition.
    parts = [{"text": "1. 化合物19的合成"},
             {"text": "合成化合物19的步骤与实施例8基本相同。LRMS m/z 538。"}]
    assert not is_template_definition("example", "", parts)
