"""Reference-as-edge regressions (2026-10-27): a reactant named after a
numbered preparation/example resolves to that record's terminal product."""

from pipeline.document.routes import assemble_routes_v2
from pipeline.validation.quality import QUALITY_VERSION

from test_route_repair import compound, reaction, record, scope_doc


def reference_record(cid, reactants, products, text, block=0, reference_id=None):
    rec = record(cid, [reaction(1, reactants, products, text=text)], block=block)
    if reference_id:
        rec["source"]["reference_id"] = reference_id
    return rec


def test_bare_reference_name_builds_edge():
    producer = reference_record("example_b", [], [compound("Compound B", "CC")],
                                "将实施例A与苯胺反应得到Compound B。")
    consumer = reference_record(
        "example_1a", [compound("实施例B", "")], [compound("Target C", "CCC")],
        "将实施例B(2.80克,6.68毫摩尔)与N,N'-二甲基乙二胺溶于DMA，升温至90℃搅拌2小时。",
        block=1)
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [producer, consumer])])
    assert not fragments and len(routes) == 1
    route = routes[0]
    assert [unit.ref["id"] for unit in route["selected"]] == ["example_b", "example_1a"]
    assert len(route["connections"]) == 1
    edge = route["connections"][0]
    assert edge["from_ref"]["id"] == "example_b" and edge["to_ref"]["id"] == "example_1a"
    assert edge["basis"] == "material_reference"
    assert "实施例B" in edge["evidence_text"]
    assert route["completeness"] == "complete"
    assert all(boundary["reason"] != "method_not_disclosed_in_scope"
               for boundary in route["boundaries"])


def test_placeholder_reference_name_builds_edge():
    producer = reference_record("example_8", [], [compound("Compound H", "CCO")],
                                "还原得到Compound H。")
    consumer = reference_record(
        "example_9", [compound("制备例8的产物", "")], [compound("Target J", "CCCO")],
        "将制备例8的产物与丙烯酰氯在DIEA存在下反应。", block=1)
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [producer, consumer])])
    assert not fragments and len(routes) == 1
    edge = routes[0]["connections"][0]
    assert edge["from_ref"]["id"] == "example_8"
    assert edge["basis"] == "material_reference"


def test_missing_producer_is_internal_reference_gap():
    consumer = reference_record(
        "example_1a", [compound("实施例B", "")], [compound("Target C", "CCC")],
        "将实施例B(2.80克)与胺溶于DMA搅拌。")
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [consumer])])
    assert not routes and len(fragments) == 1
    fragment = fragments[0]
    assert fragment["completeness"] == "incomplete"
    assert any(gap["code"] == "missing_internal_material" for gap in fragment["gaps"])
    assert any(boundary["reason"] == "internal_reference_missing"
               for boundary in fragment["boundaries"])


def test_method_citation_does_not_become_material_edge():
    producer = reference_record("example_8", [], [compound("Compound H", "CCO")],
                                "制备Compound H。")
    consumer = reference_record(
        "example_9", [compound("实施例8", "")], [compound("Target J", "CCCO")],
        "与实施例8的方法基本相同，不同的是使用其他原料。", block=1)
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [producer, consumer])])
    for row in routes + fragments:
        assert all(edge["from_ref"]["id"] != "example_8" for edge in row["connections"])


def test_provenance_sentence_attaches_reference_to_named_material():
    producer = reference_record("example_7", [], [compound("Compound X", "CC")],
                                "制备得到Compound X。")
    consumer = reference_record(
        "example_10", [compound("化合物X", "")], [compound("Target Y", "CCC")],
        "由实施例7得到的化合物X直接用于下一步反应。", block=1)
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [producer, consumer])])
    assert not fragments and len(routes) == 1
    edge = routes[0]["connections"][0]
    assert edge["from_ref"]["id"] == "example_7"
    assert edge["basis"] == "material_reference"


def test_duplicate_reference_ids_become_ambiguity():
    first = reference_record("example_b", [], [compound("Compound B one", "CC")],
                             "制备Compound B one。")
    second = reference_record("example_b_2", [], [compound("Compound B two", "CCC")],
                              "制备Compound B two。", block=1, reference_id="example_b")
    consumer = reference_record(
        "example_1a", [compound("实施例B", "")], [compound("Target C", "CCCC")],
        "将实施例B(2.80克)与胺溶于DMA搅拌。", block=2)
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [first, second, consumer])])
    target_rows = [row for row in routes + fragments
                   if row["terminal"].ref["id"] == "example_1a"]
    assert target_rows and all(row["completeness"] == "incomplete" for row in target_rows)
    assert any(gap["code"] == "ambiguous_upstream" for row in target_rows for gap in row["gaps"])


def test_reference_resolves_to_terminal_step_product():
    step_one = reaction(1, [compound("吲哚-2-甲醛", "")], [compound("Intermediate P1", "CC")],
                        text="吲哚-2-甲醛经Wittig反应关环得到P1。")
    step_two = reaction(2, [compound("Intermediate P1", "CC")],
                        [compound("Compound B", "CCC")], text="P1与胺反应得到Compound B。")
    producer = record("example_b", [step_one, step_two])
    consumer = reference_record(
        "example_1a", [compound("实施例B", "")], [compound("Target C", "CCCC")],
        "将实施例B(2.80克)与二胺溶于DMA搅拌。", block=1)
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [producer, consumer])])
    assert not fragments and len(routes) == 1
    edge = next(row for row in routes[0]["connections"] if row["to_ref"]["id"] == "example_1a")
    assert edge["from_ref"]["id"] == "example_b"
    assert edge["from_ref"]["index"] == 2
