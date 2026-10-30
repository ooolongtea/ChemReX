"""Route-repair scope assembly regressions (order-free dependencies)."""

import pytest

from pipeline.document.route_repair import (group_scopes, normalize_patent_key,
                                            volume_from_name, _referenced_reaction_ids)
from pipeline.document.routes import _split_sentences, assemble_routes_v2
from pipeline.validation.quality import QUALITY_VERSION


def compound(name, smiles="", **extra):
    return {"name": name, "smiles": smiles, "canonical_smiles": smiles,
            "status": "valid" if smiles else "structure_unavailable", "ignored": False,
            "reason": "" if smiles else "unresolved_structure", "source_field": "products",
            "source_index": 0, "structure_source": "name_resolution", "cxsmiles": None, **extra}


def reaction(index, reactants, products, text="原文", title=None, record_kind="reaction_candidate",
             eligible=True):
    step_source = {"step_id": f"step:{index}",
                   "title": title if title is not None else (products[0]["name"] if products else ""),
                   "local_text": text,
                   "source_parts": [{"text": text, "block_idx": index, "page_idx": 0}],
                   "procedure_references": [], "unit_kind": "single_step",
                   "record_kind": record_kind}
    return {"index": index, "step_id": f"step:{index}", "status": "ok", "reason": None,
            "step_source": step_source,
            "quality": {"version": QUALITY_VERSION, "admission_eligible": eligible,
                        "blocking_issues": [] if eligible else ["mapping_or_structure_requires_review"],
                        "structure_status": "supported", "record_completeness": "complete"},
            "validation": {"ner": {"REACTION_PRODUCT": products, "TEMPERATURE": None,
                                   "TIME": "", "YIELD_PERCENT": []},
                           "action": ["add"], "yield_info": {"kind": "not_reported", "value": None},
                           "stereochemistry": {"kind": "unreported"},
                           "rxn_validation": {"status": "passed", "needs_human_review": False,
                                              "patch_accepted": False, "base_expression": "",
                                              "effective_expression": "",
                                              "normalized_compounds": {
                                                  "reactants": reactants, "products": products,
                                                  "reagents": [], "solvents": [],
                                                  "participants": []}}}}


def record(cid, reactions, block=0, terminal="supported", process_kind="synthesis"):
    return {"id": cid, "chunk_fingerprint": "fp", "status": "ok", "reason": None,
            "terminal_product_coverage": terminal, "reactions": reactions,
            "source": {"id": cid, "reference_id": cid, "kind": "example", "number": block + 1,
                       "block_start": block, "page_start": 1, "title": cid,
                       "process_kind": process_kind, "references": []}}


def scope_doc(document_id, records, volume_order=0):
    return {"document_id": document_id, "volume_order": volume_order, "records": records}


def test_order_independent_post_ordered_preparation():
    a, b = compound("Intermediate A", "CC"), compound("Target B", "CCC")
    final = record("example_2", [reaction(1, [a], [b], text="Intermediate A was converted to Target B.")])
    preparation = record("example_1", [reaction(1, [], [a], text="Intermediate A was obtained.")])
    documents = [scope_doc("doc.pdf", [final, preparation])]
    routes, fragments = assemble_routes_v2(documents)
    assert len(routes) == 1 and not fragments
    route = routes[0]
    assert [unit.ref["id"] for unit in route["selected"]] == ["example_1", "example_2"]
    assert len(route["connections"]) == 1
    assert route["completeness"] == "complete"
    # Reversing the source order must not change the assembled route.
    reversed_routes, reversed_fragments = assemble_routes_v2(
        [scope_doc("doc.pdf", [preparation, final])])
    assert not reversed_fragments
    assert [unit.ref["id"] for unit in reversed_routes[0]["selected"]] == ["example_1", "example_2"]
    assert reversed_routes[0]["target"]["identity"] == route["target"]["identity"]


def test_cross_volume_dependency_spans_documents():
    a, b = compound("Intermediate A", "CC"), compound("Target B", "CCC")
    first = scope_doc("patent(1).pdf", [record("example_1", [reaction(1, [], [a])])], 0)
    second = scope_doc("patent(2).pdf", [record("example_2", [reaction(1, [a], [b])])], 1)
    routes, fragments = assemble_routes_v2([first, second])
    assert not fragments and len(routes) == 1
    route = routes[0]
    assert route["volumes"] == ["patent(1).pdf", "patent(2).pdf"]
    assert len(route["selected"]) == 2


def test_alternative_preparations_are_not_serialised():
    a, b = compound("Intermediate A", "CC"), compound("Target B", "CCC")
    method_one = record("example_1", [reaction(1, [], [compound("Intermediate A", "CC")])])
    method_two = record("example_2", [reaction(1, [], [compound("Intermediate A", "CC")])], block=1)
    consumer = record("example_3", [reaction(1, [a], [b])], block=2)
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [method_one, method_two, consumer])])
    # The unused alternative preparation stays an independent route; the target
    # route uses one preparation and records the other as an alternative.
    assert not fragments and len(routes) == 2
    target_route = next(row for row in routes if "CCC" in row["target"]["identity"])
    assert [unit.ref["id"] for unit in target_route["selected"]] == ["example_1", "example_3"]
    alternative = target_route["alternatives"][0]
    assert alternative["chosen"]["id"] == "example_1"
    assert alternative["alternatives"][0]["id"] == "example_2"


def test_ambiguous_upstream_becomes_incomplete_fragment():
    first = compound("Intermediate X", "CC")
    second = compound("Intermediate X", "CCC")
    producer_one = record("example_1", [reaction(1, [], [first])])
    producer_two = record("example_2", [reaction(1, [], [second])], block=1)
    consumer = record("example_3", [reaction(1, [compound("Intermediate X")],
                                             [compound("Target Y", "CCCC")])], block=2)
    routes, fragments = assemble_routes_v2(
        [scope_doc("doc.pdf", [producer_one, producer_two, consumer])])
    assert len(routes) == 2
    assert len(fragments) == 1
    gap_codes = [gap["code"] for gap in fragments[0]["gaps"]]
    assert "ambiguous_upstream" in gap_codes
    assert fragments[0]["completeness"] == "incomplete"


def test_procedure_template_never_provides_material():
    template = record("general_procedure", [reaction(
        1, [], [compound("Compound of formula I", "CC")], record_kind="procedure_template")])
    final = record("example_1", [reaction(1, [compound("Compound of formula I")],
                                          [compound("Target Z", "CCC")])])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [template, final])])
    assert len(routes) == 1 and not fragments
    route = routes[0]
    assert [unit.ref["id"] for unit in route["selected"]] == ["example_1"]
    assert not route["connections"]


def test_missing_internal_material_is_a_gap_but_method_reference_is_not():
    missing = record("example_1", [reaction(
        1, [compound("Intermediate Q")], [compound("Target R", "CCC")],
        text="Intermediate Q was obtained from Example 7 and converted to Target R.")])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [missing])])
    assert not routes and fragments
    assert "missing_internal_material" in [gap["code"] for gap in fragments[0]["gaps"]]

    method = record("example_2", [reaction(
        1, [compound("Prepared intermediate")], [compound("Target S", "CCCC")],
        text="Prepared intermediate was prepared in the same manner as Preparative Example 7.")])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [method])])
    assert routes and not fragments
    assert all("missing_internal_material" not in [item["code"] for item in row["gaps"]]
               for row in fragments)


def test_direct_preparation_reference_is_a_missing_internal_material_gap():
    # D1: "在制备例 24 记载的<NAME>中加入" has no obtain verb, but the numbered
    # reference directly modifies the consumed material name.
    name = "7-甲氧基-4-氧代-1,4-二氢喹啉-6-甲腈"
    final = record("example_488", [reaction(
        1, [compound(name)], [compound("Compound 488", "CCC")],
        text=f"在制备例 24 记载的 {name}(2g)中加入 N,N-二甲基甲酰胺…")])
    routes, fragments = assemble_routes_v2([scope_doc("CN1308310C(3).pdf", [final])])
    assert not routes and len(fragments) == 1
    fragment = fragments[0]
    assert fragment["completeness"] == "incomplete"
    assert "missing_internal_material" in [gap["code"] for gap in fragment["gaps"]]
    assert fragment["boundaries"][0]["reason"] == "internal_reference_missing"


def test_english_direct_reference_survives_method_word_in_other_sentence():
    # D2: English sentences must split on "." so a method word from a separate
    # sentence ("similar reaction") cannot veto the real material reference.
    final = record("step_1_9", [reaction(
        1, [compound("Intermediate X")], [compound("Target Y", "CCC")],
        text=("A solution of Intermediate X from Step 3 of Preparation 9 (2.0 g) "
              "in THF was added dropwise. The resulting mixture was stirred for 2 h "
              "in a similar reaction to a previous example."))])
    routes, fragments = assemble_routes_v2([scope_doc("US10000480.pdf", [final])])
    assert not routes and len(fragments) == 1
    fragment = fragments[0]
    assert fragment["completeness"] == "incomplete"
    assert "missing_internal_material" in [gap["code"] for gap in fragment["gaps"]]
    assert fragment["boundaries"][0]["reason"] == "internal_reference_missing"


def test_placeholder_product_name_is_missing_but_method_use_is_not():
    # D2b: a placeholder material name that is itself a numbered reference is a
    # missing internal preparation, while a pure method reference stays a
    # complete single-step route.
    placeholder = record("example_7", [reaction(
        1, [compound("product of Preparation 8")], [compound("Target Z", "CCC")],
        text="To a solution of product of Preparation 8 (2.0 g) in THF was added "
             "sodium hydride.")])
    routes, fragments = assemble_routes_v2([scope_doc("US7288657.pdf", [placeholder])])
    assert not routes and len(fragments) == 1
    fragment = fragments[0]
    assert fragment["completeness"] == "incomplete"
    assert "missing_internal_material" in [gap["code"] for gap in fragment["gaps"]]
    assert fragment["boundaries"][0]["reason"] == "internal_reference_missing"

    method = record("example_2", [reaction(
        1, [compound("Prepared intermediate")], [compound("Target S", "CCCC")],
        text="Prepared intermediate was prepared in the same manner as "
             "Preparative Example 7.")])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [method])])
    assert len(routes) == 1 and not fragments
    assert routes[0]["completeness"] == "complete"
    assert all("missing_internal_material" not in [item["code"] for item in row["gaps"]]
               for row in fragments)


def test_cross_sentence_material_anaphor_is_a_missing_reference():
    # R1: the provenance sentence need not repeat the material name; an explicit
    # material anaphor close to the obtain verb refers to the previous sentence.
    en = record("example_1", [reaction(
        1, [compound("Intermediate X")], [compound("Target Y", "CCC")],
        text="Intermediate X was dissolved in THF. "
             "The material was obtained from Example 7.")])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [en])])
    assert not routes and len(fragments) == 1
    fragment = fragments[0]
    assert fragment["completeness"] == "incomplete"
    assert "missing_internal_material" in [gap["code"] for gap in fragment["gaps"]]
    boundary = fragment["boundaries"][0]
    assert boundary["reason"] == "internal_reference_missing"
    assert "Intermediate X" in boundary["evidence_text"]
    assert "Example 7" in boundary["evidence_text"]

    zh = record("example_2", [reaction(
        1, [compound("中间体 X")], [compound("目标产物 Z", "CCC")],
        text="将中间体 X 溶于四氢呋喃。该中间体由实施例 7 制备得到。")])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [zh])])
    assert not routes and len(fragments) == 1
    assert "missing_internal_material" in [gap["code"] for gap in fragments[0]["gaps"]]
    assert fragments[0]["boundaries"][0]["reason"] == "internal_reference_missing"


def test_formulation_method_sentence_is_not_a_cross_sentence_material_reference():
    # R1 guard (US10548846 example_6): the next sentence has no material anaphor,
    # so the method reference must not become a missing internal material.
    formulation = record("example_6", [reaction(
        1, [compound("emtricitabine")], [compound("Tablet formulation F7")],
        text=("A bilayer formulation of the compound of Formula II, emtricitabine, "
              "was prepared. Tablet formulation F7 was prepared using the method "
              "described in Example 8."))])
    routes, fragments = assemble_routes_v2([scope_doc("US10548846.pdf", [formulation])])
    assert len(routes) == 1 and not fragments
    assert routes[0]["completeness"] == "complete"
    assert [boundary["reason"] for boundary in routes[0]["boundaries"]] == [
        "method_not_disclosed_in_scope"]
    assert not [gap for gap in routes[0]["gaps"]
                if gap["code"] == "missing_internal_material"]


def test_sentence_splitter_handles_decimals_abbreviations_and_lowercase():
    assert _split_sentences(
        "Added 2.5 g of X. The mixture was stirred, e.g. at 60 °C. "
        "Example 7. the mixture was concentrated...") == (
            "Added 2.5 g of X.",
            "The mixture was stirred, e.g. at 60 °C.",
            "Example 7.",
            "the mixture was concentrated...")
    assert _split_sentences("2.5 g") == ("2.5 g",)
    assert _split_sentences("stirred, e.g. at 60 °C") == ("stirred, e.g. at 60 °C",)
    assert _split_sentences("Compound No. 5 was isolated") == ("Compound No. 5 was isolated",)
    assert _split_sentences("Example 7.The mixture was stirred.") == (
        "Example 7.", "The mixture was stirred.")


def test_lowercase_sentence_after_period_keeps_the_method_word_separate():
    # R4: "...similar reaction. the material was obtained from Example 7." must
    # split so the method word cannot veto the reference in the lowercase sentence.
    final = record("example_1", [reaction(
        1, [compound("Intermediate X")], [compound("Target L", "CCC")],
        text="Intermediate X was converted in a similar reaction. "
             "the material was obtained from Example 7.")])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [final])])
    assert not routes and len(fragments) == 1
    assert fragments[0]["completeness"] == "incomplete"
    assert "missing_internal_material" in [gap["code"] for gap in fragments[0]["gaps"]]


def test_external_document_cue_wins_over_internal_reference():
    # R7: an explicit external document is a start boundary, not a missing gap.
    final = record("example_1", [reaction(
        1, [compound("Intermediate X")], [compound("Target E", "CCC")],
        text="Intermediate X from Example 3 of US1234567 was added to the mixture.")])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [final])])
    assert len(routes) == 1 and not fragments
    assert routes[0]["completeness"] == "complete"
    assert [boundary["reason"] for boundary in routes[0]["boundaries"]] == [
        "external_document_reference"]


def test_commercial_cue_wins_over_placeholder_reference():
    # R7: an explicit purchase cue is a start boundary even for a placeholder name.
    final = record("example_1", [reaction(
        1, [compound("product of Preparation 8")], [compound("Target C", "CCC")],
        text="The product of Preparation 8 was purchased from Sigma-Aldrich "
             "and used directly.")])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [final])])
    assert len(routes) == 1 and not fragments
    assert routes[0]["completeness"] == "complete"
    assert [boundary["reason"] for boundary in routes[0]["boundaries"]] == [
        "commercial_material"]


def test_unrelated_commercial_cue_does_not_suppress_the_reference_gap():
    # R7 refinement: a purchase cue belonging to another material in the same
    # step must not close the material's own internal-reference gap.
    final = record("example_1", [reaction(
        1, [compound("Intermediate X")], [compound("Target G", "CCC")],
        text="Intermediate X from Example 7 was added. "
             "Commercially available sodium chloride was used as eluent.")])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [final])])
    assert not routes and len(fragments) == 1
    fragment = fragments[0]
    assert fragment["completeness"] == "incomplete"
    assert "missing_internal_material" in [gap["code"] for gap in fragment["gaps"]]
    assert fragment["boundaries"][0]["reason"] == "internal_reference_missing"


def test_placeholder_gap_survives_commercial_cue_in_another_sentence():
    # R7 refinement: only the sentence naming the placeholder may supply a start
    # cue; a purchase cue in another sentence must not turn the gap into a start.
    final = record("example_1", [reaction(
        1, [compound("product of Preparation 8")], [compound("Target P", "CCC")],
        text="The product of Preparation 8 was used directly. "
             "The starting material was purchased from Sigma-Aldrich.")])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [final])])
    assert not routes and len(fragments) == 1
    fragment = fragments[0]
    assert fragment["completeness"] == "incomplete"
    assert "missing_internal_material" in [gap["code"] for gap in fragment["gaps"]]
    assert fragment["boundaries"][0]["reason"] == "internal_reference_missing"


def test_completeness_is_independent_of_quality_admission():
    a, b = compound("Intermediate A", "CC"), compound("Target B", "CCC")
    producer = record("example_1", [reaction(1, [], [a], eligible=False)])
    consumer = record("example_2", [reaction(1, [a], [b], eligible=False)], block=1)
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [producer, consumer])])
    assert not fragments and len(routes) == 1
    assert routes[0]["completeness"] == "complete"


def test_group_scopes_merges_volumes_and_identical_pdfs():
    rows = [
        {"uuid": "cn1", "source": "cn", "document_id": "CN1308310C(1).pdf",
         "patent_key": "CN1308310C", "volume_index": 1, "pdf_sha256": "sha1",
         "source_label": "cn1", "filename": "CN1308310C(1).pdf"},
        {"uuid": "cn2", "source": "cn", "document_id": "CN1308310C(2).pdf",
         "patent_key": "CN1308310C", "volume_index": 2, "pdf_sha256": "sha2",
         "source_label": "cn2", "filename": "CN1308310C(2).pdf"},
        {"uuid": "cn3", "source": "cn", "document_id": "CN1308310C(3).pdf",
         "patent_key": "CN1308310C", "volume_index": 3, "pdf_sha256": "sha3",
         "source_label": "cn3", "filename": "CN1308310C(3).pdf"},
        {"uuid": "dup", "source": "cn", "document_id": "CN1308310C(3).pdf",
         "patent_key": "CN1308310C", "volume_index": 3, "pdf_sha256": "sha3",
         "source_label": "dup", "filename": "CN1308310C(3).pdf"},
    ]
    scopes = group_scopes(rows)
    assert len(scopes) == 1
    scope = scopes[0]
    assert scope["scope_id"] == "cn::CN1308310C"
    assert [document["uuid"] for document in scope["documents"]] == ["cn1", "cn2", "cn3"]
    assert scope["aliases"] == {"dup": "cn3"}


def test_patent_key_and_volume_normalisation():
    assert normalize_patent_key("US8598181") == "US8598181"
    assert normalize_patent_key("USRE49444") == "USRE49444"
    assert normalize_patent_key("ZL200780004302.4") == "CN2007800043024"
    assert volume_from_name("CN1308310C(2).pdf") == 2
    assert volume_from_name("US8598181.pdf") is None


def test_referenced_reaction_ids_group_by_document():
    rows = [{"route_id": "r1", "steps": [{"document_id": "a.pdf", "reaction_id": "a.pdf::x::1"}],
             "target": {"reaction_id": "a.pdf::x::1"},
             "connections": [{"from": "b.pdf::y::1", "to": "a.pdf::x::1"}],
             "gaps": [{"code": "missing_internal_material",
                       "reaction_ids": ["b.pdf::y::1"]}]}]
    referenced = _referenced_reaction_ids(rows)
    assert referenced["a.pdf"] == {"a.pdf::x::1"}
    assert referenced["b.pdf"] == {"b.pdf::y::1"}


def test_same_record_ids_in_distinct_volumes_do_not_share_terminal_or_siblings():
    first = scope_doc("vol1.pdf", [record("example_1", [
        reaction(1, [compound("Feed A", "C")], [compound("Intermediate A", "CC")]),
        reaction(2, [compound("Intermediate A", "CC")], [compound("Target B", "CCC")]),
    ])])
    second = scope_doc("vol2.pdf", [record("example_1", [reaction(
        1, [compound("Feed C", "CCCC")], [compound("Target D", "CCCCC")])])], 1)
    routes, fragments = assemble_routes_v2([first, second])
    assert not fragments
    assert {(r["terminal"].document_id, len(r["selected"])) for r in routes} == {
        ("vol1.pdf", 2), ("vol2.pdf", 1)}


def test_missing_product_remains_an_unconfirmed_source_grounded_fragment():
    preparation = record("example_24", [reaction(
        1, [compound("Precursor", "CC")], [],
        title="7-甲氧基-4-氧代-1,4-二氢喹啉-6-甲腈",
        text="将原料加热闭环，得到标题化合物。")], terminal="unknown")
    routes, fragments = assemble_routes_v2([scope_doc("patent.pdf", [preparation])])
    assert not routes and len(fragments) == 1
    assert fragments[0]["target"]["product_index"] is None
    assert fragments[0]["completeness"] == "unconfirmed"
    assert "extraction_product_missing" in {g["code"] for g in fragments[0]["gaps"]}
    assert fragments[0]["target"]["display"]["name"].startswith("7-甲氧基")
    preparation["source"]["title"] = "7-甲氧基-4-氧代-1,4-二氢喹啉-6-甲腈"
    preparation["reactions"][0]["step_source"]["title"] = "与制备例457-3同样进行，将原料加热"
    _, repeated = assemble_routes_v2([scope_doc("patent.pdf", [preparation])])
    assert repeated[0]["target"]["display"]["name"] == preparation["source"]["title"]


def test_complete_preparation_is_used_regardless_of_alternative_source_order():
    intermediate = compound("Intermediate A", "CC")
    bad = record("example_1", [reaction(1, [compound("Missing X")], [intermediate],
                                       text="Missing X was obtained from Example 7.")])
    good = record("example_2", [reaction(1, [compound("Feed", "C")], [intermediate])], 1)
    other_good = record("example_4", [reaction(1, [compound("Other feed", "CO")],
                                              [intermediate])], 3)
    target = record("example_3", [reaction(1, [intermediate], [compound("Target B", "CCC")])], 2)
    for preparations in ([bad, good, other_good], [other_good, good, bad]):
        routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [*preparations, target])])
        route = next(r for r in routes if r["terminal"].record["id"] == "example_3")
        assert {u.record["id"] for u in route["selected"]} == {"example_2", "example_3"}
        assert route["alternatives"][0]["alternatives"][0]["id"] == "example_4"
        assert all(a["id"] != "example_1" for choice in route["alternatives"]
                   for a in choice["alternatives"])
        assert any(f["terminal"].record["id"] == "example_1" for f in fragments)


@pytest.mark.parametrize("coverage,disconnected", [("unknown", False), ("supported", True)])
def test_consumer_inherits_coverage_gaps_of_its_upstream_method(coverage, disconnected):
    steps = [reaction(1, [compound("Feed", "C")], [compound("Unused", "CO")])] if disconnected else []
    steps.append(reaction(len(steps) + 1, [compound("Other feed", "CN")],
                          [compound("Intermediate", "CC")]))
    preparation = record("example_1", steps, terminal=coverage)
    consumer = record("example_2", [reaction(1, [compound("Intermediate", "CC")],
                                              [compound("Target", "CCC")])], 1)
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [preparation, consumer])])
    assert not any(r["terminal"].record["id"] == "example_2" for r in routes)
    target = next(f for f in fragments if f["terminal"].record["id"] == "example_2")
    assert ("disconnected_procedure_steps" if disconnected else "terminal_product_coverage_unknown") in {
        g["code"] for g in target["gaps"]}


def test_crystal_forms_with_identical_smiles_do_not_supply_each_other():
    preparation = record("example_1", [reaction(1, [compound("Feed", "C")],
                                                 [compound("Drug crystal Form A", "CC")])])
    final = record("example_2", [reaction(1, [compound("Drug crystal Form B", "CC")],
                                           [compound("Target", "CCC")])], 1)
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [preparation, final])])
    target = next(r for r in routes + fragments if r["terminal"].record["id"] == "example_2")
    assert len(target["selected"]) == 1 and not target["connections"]
    assert target["boundaries"][0]["reason"] == "material_form_mismatch"
    assert target["completeness"] != "complete"


def test_equivalent_salt_labels_do_not_block_a_confirmed_structure_match():
    preparation = record("example_1", [reaction(1, [compound("Feed", "C")],
                                                 [compound("甲胺盐酸盐", "C[NH3+].[Cl-]")])])
    final = record("example_2", [reaction(1, [compound("Methylamine hydrochloride", "C[NH3+].[Cl-]")],
                                           [compound("Target", "CCC")])], 1)
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [preparation, final])])
    assert not fragments and len(routes) == 1
    assert len(routes[0]["connections"]) == 1


def test_scan_reads_zero_delivery_reactions_and_keeps_unprocessed_disclosures(tmp_path):
    import json
    from pipeline.document.route_repair import scan_document
    delivery = tmp_path / "delivery"
    artifacts = delivery / "artifacts"
    artifacts.mkdir(parents=True)
    chunks = [{"id": "example_1", "chunk_fingerprint": "fp", "title": "Compound B",
               "candidate_evidence": True, "has_procedure": True, "process_kind": "synthesis",
               "block_start": 10, "page_start": 2,
               "source_parts": [{"text": "A was converted to B.", "block_idx": 11, "page_idx": 1}],
               "procedure": "A was converted to B."}]
    (artifacts / "chunks.jsonl").write_text(json.dumps(chunks[0]) + "\n")
    (artifacts / "results.jsonl").write_text("")
    row = scan_document({"uuid": "doc", "document_id": "US1234567.pdf", "reaction_rows": 0,
                         "delivery_dir": str(delivery), "output_dir": str(tmp_path)})
    assert row["scan_error"] is None
    assert row["disclosed_steps"] == 1
    routes, fragments = assemble_routes_v2([scope_doc("US1234567.pdf", row["records"])])
    assert not routes and len(fragments) == 1
    assert fragments[0]["selected"][0].context["source_parts"][0]["block_idx"] == 11
    from pipeline.document.route_repair import run_scan, run_assemble, run_export, write_jsonl, read_jsonl
    entry = {"uuid": "doc", "source": "fda", "document_id": "US1234567.pdf", "reaction_rows": 0,
             "delivery_dir": str(delivery), "output_dir": str(tmp_path)}
    write_jsonl(tmp_path / "input/documents.jsonl", [entry])
    out = tmp_path / "out"
    assert run_scan(tmp_path / "input", out)["errors"] == 0
    charter = json.loads((out / "scan/supplement_charter.json").read_text())
    assert charter["items"] == []
    diagnostics = read_jsonl(out / "scan/repair_actions.jsonl")
    assert diagnostics[0]["repair_action"] == "missing_extraction_record"
    run_assemble(out)
    summary = run_export(out, drug_sources=False)
    assert summary["fragments_unconfirmed"] == 1
    assert summary["documents_with_disclosed_steps"] == 1
    assert summary["reaction_reference_gaps"] == 0
    assert len(read_jsonl(out / "reactions/fda/doc.jsonl")) == 1


def test_resegmentation_does_not_reuse_changed_reaction_ids_and_maps_absorbed_routes(tmp_path):
    from pipeline.document.route_supplement import merge_records
    from pipeline.document.route_repair import (write_gzip_jsonl, write_jsonl, convert_scope,
                                               _generate_reaction_file)
    old = record("example_18", [reaction(1, [compound("Feed", "C")],
                                         [compound("Target", "CCC")], text="Original combined method.")])
    cache = tmp_path / "scan/records/doc.json.gz"
    write_gzip_jsonl(cache, [old])
    delivery = tmp_path / "delivery"
    write_jsonl(delivery / "routes.jsonl", [{"route_id": "old-route", "steps": [{}],
                "target": {"reaction_id": "doc.pdf::example_18::1", "compound_id": "c2"}}])
    old_business, _ = _generate_reaction_file("doc.pdf", [old], set())
    write_jsonl(delivery / "reactions.jsonl", old_business)
    fresh = record("example_18", [
        reaction(1, [compound("Feed", "C")], [compound("Intermediate", "CC")], text="First conversion."),
        reaction(2, [compound("Intermediate", "CC")], [compound("Target", "CCC")], text="Second conversion."),
    ])
    row = {"uuid": "doc", "document_id": "doc.pdf", "delivery_dir": str(delivery)}
    merged = merge_records(tmp_path, row, [fresh])
    assert all(r["index"] != 1 for r in merged[0]["reactions"])
    assert len({r["index"] for r in merged[0]["reactions"]}) == 2
    again = merge_records(tmp_path, row, [fresh])
    assert [r["index"] for r in again[0]["reactions"]] == [r["index"] for r in merged[0]["reactions"]]
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", merged)])
    scope = {"scope_id": "patent", "documents": [row], "aliases": {}}
    converted, _, mapping = convert_scope(scope, routes, fragments, {"doc": str(delivery)})
    assert mapping[0]["new_ids"] == [converted[0]["route_id"]]
    assert mapping[0]["action"] == "remapped"
    rows, _ = _generate_reaction_file("doc.pdf", merged, set())
    assert all("doc.pdf::example_18::1" in r["provenance"]["source_reaction_ids"] for r in rows)


def test_unchanged_reextracted_step_keeps_id_and_absorbed_route_maps_to_parent(tmp_path):
    from pipeline.document.route_supplement import merge_records
    from pipeline.document.route_repair import write_gzip_jsonl, write_jsonl, convert_scope
    old = record("example_1", [reaction(1, [compound("Feed", "C")],
                                        [compound("Intermediate", "CC")])])
    write_gzip_jsonl(tmp_path / "scan/records/doc.json.gz", [old])
    row = {"uuid": "doc", "document_id": "doc.pdf"}
    merged = merge_records(tmp_path, row, [old])
    assert merged[0]["reactions"][0]["index"] == 1
    final = record("example_2", [reaction(1, [compound("Intermediate", "CC")],
                                           [compound("Target", "CCC")])], 1)
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [*merged, final])])
    delivery = tmp_path / "delivery"
    write_jsonl(delivery / "routes.jsonl", [{"route_id": "old-route", "steps": [{}],
                "target": {"reaction_id": "doc.pdf::example_1::1"}}])
    scope = {"scope_id": "patent", "documents": [row], "aliases": {}}
    converted, _, mapping = convert_scope(scope, routes, fragments, {"doc": str(delivery)})
    assert mapping[0]["action"] == "merged"
    assert mapping[0]["new_ids"] == [converted[0]["route_id"]]


def test_reextraction_never_reuses_an_index_of_a_retired_step(tmp_path):
    from pipeline.document.route_supplement import merge_records
    from pipeline.document.route_repair import write_gzip_jsonl
    old = record("example_1", [reaction(1, [], [compound("A", "C")], text="A.")])
    write_gzip_jsonl(tmp_path / "scan/records/doc.json.gz", [old])
    row = {"uuid": "doc", "document_id": "doc.pdf"}
    extended = record("example_1", [*old["reactions"], reaction(2, [], [compound("B", "CC")], text="B.")])
    merged = merge_records(tmp_path, row, [extended])
    retired = merged[0]["reactions"][-1]["index"]
    merge_records(tmp_path, row, [old])
    other = record("example_1", [*old["reactions"], reaction(2, [], [compound("C", "CCC")], text="C.")])
    result = merge_records(tmp_path, row, [other])
    assert result[0]["reactions"][-1]["index"] > retired


def test_failed_supplement_keeps_the_disclosed_step_and_other_index_entries(tmp_path, monkeypatch):
    from pipeline.document import route_supplement
    from pipeline.document.route_repair import write_jsonl, write_json, read_jsonl, read_gzip_jsonl
    chunk = {"id": "example_1", "candidate_evidence": True, "process_kind": "synthesis",
             "title": "Compound B", "procedure": "A was converted to B.",
             "source_parts": [{"block_idx": 10, "page_idx": 1}]}
    row = {"uuid": "doc", "document_id": "doc.pdf", "disclosures": [
        {"chunk_id": "other", "extraction_available": True}], "source": "fda"}
    write_jsonl(tmp_path / "scan/documents.jsonl", [row])
    charter = tmp_path / "charter.json"
    write_json(charter, {"items": [{"uuid": "doc", "chunks": ["example_1"]}]})
    def failed(*_args):
        return [{"id": "example_1", "status": "error", "source": chunk}], {"completed_at": "now"}
    monkeypatch.setattr(route_supplement, "supplement_document", failed)
    route_supplement.run_charter(tmp_path, charter)
    cached = read_gzip_jsonl(tmp_path / "scan/records/doc.json.gz")
    assert cached[0]["reactions"][0]["reason"] == "extraction_missing"
    updated = read_jsonl(tmp_path / "scan/documents.jsonl")[0]
    assert {d["chunk_id"] for d in updated["disclosures"]} == {"other", "example_1"}
    assert updated["reaction_rows"] == 1
    assert assemble_routes_v2([scope_doc("doc.pdf", cached)])[1]


def test_nonreaction_formulation_is_retained_without_fake_missing_reactions(tmp_path):
    from pipeline.document.route_repair import _disclosed_records, write_supplement_charter, read_json
    chunk = {"id": "tablet", "process_kind": "formulation", "has_procedure": True,
             "title": "Tablet", "procedure": "Mix and compress into tablets."}
    records = [{"id": "tablet", "source": chunk, "status": "non_reaction", "reason": "formulation",
                "reactions": [{"index": 1, "reason": "extraction_missing"}]}]
    disclosures = _disclosed_records(records, [chunk])
    assert records[0]["reactions"] == []
    assert disclosures[0]["repair_action"] == "retain_formulation"
    write_supplement_charter(tmp_path, [{"uuid": "doc", "document_id": "doc.pdf", "disclosures": disclosures}])
    assert read_json(tmp_path / "scan/supplement_charter.json")["items"] == []


def test_step_alignment_uses_unique_source_body_not_numeric_position():
    from pipeline.document.route_repair import _disclosed_records
    chunk = {"id": "example_1", "process_kind": "synthesis", "has_procedure": True,
             "steps": [{"step_id": "corrected", "local_text": "A affords B.", "source_parts": []}]}
    old = reaction(1, [compound("A", "C")], [compound("B", "CC")], text="A affords B.")
    records = [{"id": chunk["id"], "source": chunk, "status": "needs_review", "reactions": [old]}]
    assert _disclosed_records(records, [chunk])[0]["extraction_available"]
    duplicate = dict(old, step_id="another")
    records[0]["reactions"].append(duplicate)
    assert not _disclosed_records(records, [chunk])[0]["extraction_available"]


def test_model_charter_requires_changed_source_and_does_not_repeat_failed_attempt(tmp_path):
    from pipeline.document.route_repair import _disclosed_records, write_supplement_charter, read_json
    chunk = {"id": "example_1", "process_kind": "synthesis", "has_procedure": True,
             "steps": [{"step_id": "step:1", "local_text": "A affords B.", "source_parts": []}]}
    missing = reaction(1, [], [], text="A affords B.")
    missing["extraction"] = {"ner": {}, "deferred_llm": {"tasks": ["ner"]}}
    record = {"id": chunk["id"], "source": chunk, "status": "needs_review", "reactions": [missing]}
    def charter():
        ds = _disclosed_records([record], [chunk])
        write_supplement_charter(tmp_path, [{"uuid": "doc", "document_id": "doc.pdf", "disclosures": ds}])
        return read_json(tmp_path / "scan/supplement_charter.json")
    assert not charter()["items"]
    record["repair_plan"] = {"step:1": {"effective_changes": [{"kind": "source_product_title_corrected"}], "tasks": ["ner"]}}
    item = charter()["items"][0]
    assert item["step_ids"] == ["step:1"] and item["preset"] == "huawei-maas"
    record["model_attempted_step_ids"] = ["step:1"]
    assert not charter()["items"]


def test_effective_changes_ignore_cache_versions_and_identify_corrected_title():
    from pipeline.document.route_supplement import _effective_changes
    old = {"step_id": "one", "title": "Wrong heading", "local_text": "A affords B.",
           "text": "Wrong heading\nA affords B.", "procedure_references": [],
           "source_parts": [{"block_idx": 1}], "context_version": "old"}
    current = dict(old, context_version="new")
    previous = {"reactions": [{"step_source": old}]}
    assert not _effective_changes({"steps": [current]}, previous)
    current.update(title="Product B", text="Product B\nA affords B.")
    changes = _effective_changes({"steps": [current]}, previous)
    assert changes["one"][0]["kind"] == "source_product_title_corrected"
    assert changes["one"][0]["previous_input"] != changes["one"][0]["current_input"]
    resumed = {"reactions": [{"step_source": current}], "repair_plan": {
        "one": {"effective_changes": changes["one"]}}}
    assert _effective_changes({"steps": [current]}, resumed)["one"] == changes["one"]
    resumed["model_attempted_step_ids"] = ["one"]
    assert not _effective_changes({"steps": [current]}, resumed)


def test_targeted_supplement_preserves_unselected_steps():
    from pipeline.document.route_supplement import _combine_selected_record
    unchanged = reaction(1, [compound("A", "C")], [compound("B", "CC")])
    failed = reaction(2, [], [])
    replacement = reaction(1, [compound("B", "CC")], [compound("C", "CCC")])
    replacement["step_id"] = "step:2"
    before = {"reactions": [unchanged, failed], "terminal_product_coverage": "supported"}
    fresh = {"reactions": [replacement], "status": "ok"}
    merged = _combine_selected_record(before, fresh, {"step:2"}, {"steps": [{"step_id": "step:1"}, {"step_id": "step:2"}]})
    assert merged["reactions"][0] == unchanged
    assert merged["reactions"][1]["validation"] == replacement["validation"]


def test_offline_replay_preserves_product_when_unchanged_sibling_becomes_empty():
    from pipeline.document.route_supplement import _preserve_unchanged_products
    step = {"step_id": "step:1", "local_text": "A affords B.", "text": "A affords B.",
            "procedure_references": []}
    other = dict(step, step_id="step:2", local_text="B affords C.", text="B affords C.")
    source = {"id": "example_1", "block_start": 10, "steps": [step, other]}
    saved = reaction(1, [compound("A", "C")], [compound("B", "CC")])
    saved.update(step_source=step, status="needs_review", quality={"admission_eligible": False})
    empty = reaction(1, [], [])
    empty["step_source"] = step
    recovered = reaction(2, [compound("B", "CC")], [compound("C", "CCC")])
    recovered["step_source"] = other
    previous = {"source": source, "reactions": [saved, reaction(2, [], [])]}
    fresh = {"source": source, "reactions": [empty, recovered], "status": "ok",
             "offline_repair": {"new_llm_calls": 0}}
    merged = _preserve_unchanged_products(previous, fresh, source)
    assert merged["reactions"][0]["validation"] == saved["validation"]
    assert merged["reactions"][0]["quality"] == saved["quality"]
    assert merged["reactions"][1] == recovered
    assert merged["status"] == "needs_review"
    assert merged["offline_repair"]["retained_step_ids"] == ["step:1"]
    assert "retained_previous_extraction" not in saved
    for changed in (dict(step, text="A affords D."), dict(step, title="New product heading")):
        changed_source = dict(source, steps=[changed, other])
        assert _preserve_unchanged_products(previous, fresh, changed_source) == fresh
    shifted_source = dict(source, block_start=99)
    assert _preserve_unchanged_products(previous, fresh, shifted_source) == fresh
    named_only = dict(fresh, reactions=[dict(reaction(1, [], [compound("B", None)]), step_source=step), recovered])
    retained = _preserve_unchanged_products(previous, named_only, source)
    assert retained["reactions"][0]["validation"] == saved["validation"]
    valid_replacement = dict(fresh, reactions=[dict(reaction(1, [], [compound("D", "CCCC")]), step_source=step), recovered])
    assert _preserve_unchanged_products(previous, valid_replacement, source) == valid_replacement
    attempted = dict(named_only, model_attempted_step_ids=["step:1"])
    assert _preserve_unchanged_products(previous, attempted, source) == attempted
    formed = reaction(1, [], [compound("B, Form C", "CC")])
    formed["step_source"] = step
    previous_form = dict(previous, reactions=[formed, previous["reactions"][1]])
    split_form = dict(fresh, reactions=[dict(reaction(1, [], [compound("B", "CC"), compound("Form C", None)]),
                                           step_source=step), recovered])
    retained_form = _preserve_unchanged_products(previous_form, split_form, source)
    assert retained_form["reactions"][0]["validation"] == formed["validation"]
    unresolved_form = dict(reaction(1, [], [compound("B, Form C", None)]), step_source=step)
    previous_unresolved_form = dict(previous_form, reactions=[unresolved_form, previous["reactions"][1]])
    retained_form = _preserve_unchanged_products(previous_unresolved_form, split_form, source)
    assert retained_form["reactions"][0]["validation"] == unresolved_form["validation"]


def test_saved_supplement_for_one_step_does_not_skip_another_step():
    from pipeline.document.route_supplement import _cached_supplement_covers
    chunk = {"chunk_fingerprint": "same", "steps": [{"step_id": "one"}, {"step_id": "two"}]}
    saved = {"chunk_fingerprint": "same", "model_attempted_step_ids": ["one"]}
    assert _cached_supplement_covers(saved, chunk, ["one"])
    assert not _cached_supplement_covers(saved, chunk, ["two"])


def test_supplement_resume_compares_actual_input_and_image_dictionary():
    from pipeline.document.route_supplement import _cached_supplement_covers
    step = {"step_id": "one", "text": "A affords B.", "procedure_references": []}
    chunk = {"chunk_fingerprint": "new-code-version", "steps": [step]}
    saved = {"chunk_fingerprint": "old-code-version", "model_attempted_step_ids": ["one"],
             "model_inputs": {"one": {"text": step["text"], "material_dictionary": {"A": "C"}}}}
    assert _cached_supplement_covers(saved, chunk, ["one"], {"A": "C"})
    assert not _cached_supplement_covers(saved, chunk, ["one"], {"A": "CC"})
    step["text"] = "A affords C."
    assert not _cached_supplement_covers(saved, chunk, ["one"], {"A": "C"})


def test_failed_targeted_model_call_preserves_other_steps(tmp_path, monkeypatch):
    from pipeline.document import route_supplement, chunk_processing
    from pipeline.document.route_repair import write_gzip_jsonl
    first = reaction(1, [compound("A", "C")], [compound("B", "CC")])
    second = reaction(2, [], [])
    before = record("example_1", [first, second])
    write_gzip_jsonl(tmp_path / "scan/records/doc.json.gz", [before])
    chunk = {"id": "example_1", "images": [], "chunk_fingerprint": "fp", "steps": [
        {"step_id": "step:1", "text": "A affords B.", "procedure_references": []},
        {"step_id": "step:2", "text": "B affords C.", "procedure_references": []}]}
    monkeypatch.setattr(route_supplement, "_set_model_env", lambda _: {})
    monkeypatch.setattr(route_supplement, "_select_chunks", lambda *_: ([chunk], [], {"content_list": "source", "segmentation_version": "current"}))
    monkeypatch.setattr(route_supplement, "_image_result", lambda _: {})
    def fail(*args, **kwargs):
        raise RuntimeError("service unavailable")
    monkeypatch.setattr(chunk_processing, "process_chunk", fail)
    rows, _ = route_supplement.supplement_document(tmp_path,
        {"uuid": "doc", "source": "fda", "document_id": "doc.pdf", "delivery_dir": str(tmp_path / "delivery")},
        {"chunks": ["example_1"], "step_ids": ["step:2"]})
    assert rows[0]["reactions"][0] == first
    assert rows[0]["reactions"][1] == second
    assert "service unavailable" in rows[0]["error"]


def test_analogy_sentence_does_not_replace_original_product_title():
    from pipeline.document.step_context import split_steps
    title = "7-甲氧基-4-氧-1,4-二氢-喹啉-6-甲腈"
    text = "与制备例457-3同样进行，将4-氨基-2-甲氧基-苯甲腈"
    chunk = {"id": "example_24", "title": title, "kind": "example", "process_kind": "synthesis",
             "text": text + "\n加热闭环，得到标题化合物。", "procedure_index": {},
             "source_parts": [{"text": text, "block_idx": 1},
                              {"text": "加热闭环，得到标题化合物。", "block_idx": 2}]}
    steps = split_steps(chunk)
    assert len(steps) == 1 and steps[0]["title"] == title


def test_offline_naming_forwards_language_without_calling_llm(monkeypatch):
    from scripts.reprocess_without_llm_20261016 import OfflineNames
    from pipeline.standardize.compound_standardizer import CompoundStandardizer
    seen = []
    def saved(self, names, prior_naming=None, language=None):
        seen.append(language)
        return {"ethanol": {"smiles": "CCO"}}
    monkeypatch.setattr(CompoundStandardizer, "standardize", saved)
    names = object.__new__(OfflineNames)
    names.deferred = []
    assert names.standardize(["ethanol"], language="en")["ethanol"]["smiles"] == "CCO"
    assert seen == ["en"]


def test_parallel_supplement_reuses_saved_responses_and_merges_each_document(tmp_path, monkeypatch):
    import json
    from pipeline.document.route_supplement import _select_chunks, run_charter
    from pipeline.document.step_context import extraction_text
    from pipeline.document.route_repair import write_json, write_jsonl, read_jsonl
    # Cached responses must suffice even when a model cannot be constructed.
    monkeypatch.setenv("MAAS_API_KEY", "")
    out = tmp_path / "out"
    index, items = [], []
    for number in (1, 2):
        document = tmp_path / ("document" + str(number))
        prepared = document / "pdf_prepared"
        prepared.mkdir(parents=True)
        blocks = [{"type": "text", "text": "Example 1", "page_idx": 0},
                  {"type": "text", "text": "Methanol was stirred for 2 h to afford formaldehyde (80% yield).", "page_idx": 0}]
        (prepared / "patent_content_list.json").write_text(json.dumps(blocks))
        row = {"uuid": "doc" + str(number), "document_id": "doc" + str(number) + ".pdf",
               "source": "fda", "delivery_dir": str(document / "delivery"), "patent_key": "US123456" + str(number)}
        chunks, missing, _ = _select_chunks(row, ["example_1"])
        assert len(chunks) == 1 and not missing
        chunk = chunks[0]
        step_ids = [s["step_id"] for s in chunk["steps"]]
        cached = {"id": chunk["id"], "source": chunk, "chunk_fingerprint": chunk["chunk_fingerprint"],
                  "status": "needs_review", "response_reused_marker": number,
                  "model_attempted_step_ids": step_ids,
                  "model_inputs": {s["step_id"]: {"text": extraction_text(s), "material_dictionary": {}}
                                   for s in chunk["steps"]}}
        write_jsonl(out / "supplement" / row["uuid"] / "results.jsonl", [cached])
        index.append(row)
        items.append({"uuid": row["uuid"], "chunks": [chunk["id"]], "step_ids": step_ids})
    write_jsonl(out / "scan/documents.jsonl", index)
    charter = tmp_path / "charter.json"
    write_json(charter, {"items": items})
    assert run_charter(out, charter, workers=2)["documents"] == 2
    updated = read_jsonl(out / "scan/documents.jsonl")
    assert {r["uuid"] for r in updated} == {"doc1", "doc2"}
    assert all(r["supplemented_chunks"] == ["example_1"] for r in updated)
    for number in (1, 2):
        saved = read_jsonl(out / "supplement" / ("doc" + str(number)) / "results.jsonl")
        assert saved[0]["response_reused_marker"] == number


def test_chunk_selection_follows_source_blocks_when_counters_shift(monkeypatch):
    from pathlib import Path
    from pipeline.document import route_supplement, segmenter
    from pipeline.document.position_matcher import PositionMatcher
    current = [
        {"id": "step_a_1", "block_start": 20, "block_indices": [20, 21], "candidate_evidence": True,
         "steps": [{"step_id": "step_a_1:step:1"}]},
        {"id": "step_a_2", "block_start": 10, "block_indices": [10, 11], "candidate_evidence": False,
         "has_procedure": True, "process_kind": "unverified",
         "steps": [{"step_id": "step_a_2:step:1"}]},
    ]
    monkeypatch.setattr(route_supplement, "_locate_content_list", lambda _: Path("source.json"))
    monkeypatch.setattr(segmenter, "load_blocks", lambda _: [])
    monkeypatch.setattr(segmenter, "segment_blocks", lambda *_: [])
    monkeypatch.setattr(segmenter, "select_procedure_chunks", lambda *_: current)
    monkeypatch.setattr(PositionMatcher, "match_chunk", lambda *_: {})
    row = {"archived_sources": {"step_a_1": {"block_start": 10, "block_indices": [10, 11], "candidate_evidence": True}}}
    selected, missing, _ = route_supplement._select_chunks(row, ["step_a_1"])
    assert not missing and len(selected) == 1
    assert selected[0]["block_start"] == 10
    assert selected[0]["id"] == "step_a_1"
    assert selected[0]["current_chunk_id"] == "step_a_2"
    assert selected[0]["steps"][0]["step_id"] == "step_a_1:step:1"
    assert selected[0]["candidate_evidence"]


def test_chunk_selection_does_not_accept_same_counter_at_unrelated_source(monkeypatch):
    from pathlib import Path
    from pipeline.document import route_supplement, segmenter
    from pipeline.document.position_matcher import PositionMatcher
    monkeypatch.setattr(route_supplement, "_locate_content_list", lambda _: Path("source.json"))
    monkeypatch.setattr(segmenter, "load_blocks", lambda _: [])
    monkeypatch.setattr(segmenter, "segment_blocks", lambda *_: [])
    monkeypatch.setattr(segmenter, "select_procedure_chunks", lambda *_: [
        {"id": "step_a_1", "block_start": 99, "block_indices": [99], "steps": []}])
    monkeypatch.setattr(PositionMatcher, "match_chunk", lambda *_: {})
    row = {"archived_sources": {"step_a_1": {"block_start": 10, "block_indices": [10]}}}
    selected, missing, _ = route_supplement._select_chunks(row, ["step_a_1"])
    assert not selected and missing == ["step_a_1"]


def test_prepare_restores_shifted_source_before_classifying_model_work(tmp_path, monkeypatch):
    from pipeline.document import route_supplement
    from pipeline.document.route_repair import write_json, write_jsonl, write_gzip_jsonl, read_gzip_jsonl, read_json
    source = {"id": "example_1", "chunk_fingerprint": "archive", "candidate_evidence": True,
              "process_kind": "synthesis", "has_procedure": True, "block_start": 10,
              "block_indices": [10], "procedure": "A affords B.",
              "steps": [{"step_id": "step:1", "local_text": "A affords B.", "source_parts": []}]}
    original = {"id": "example_1", "chunk_fingerprint": "archive", "source": source,
                "status": "needs_review", "reason": "invalid_or_empty_ner", "reactions": [reaction(1, [], [])]}
    delivery = tmp_path / "document/delivery"
    write_jsonl(delivery / "artifacts/chunks.jsonl", [source])
    write_jsonl(delivery / "artifacts/results.jsonl", [original])
    out = tmp_path / "out"
    row = {"uuid": "doc", "document_id": "doc.pdf", "source": "fda", "patent_key": "US1234567",
           "delivery_dir": str(delivery)}
    write_jsonl(out / "scan/documents.jsonl", [row])
    write_json(out / "scan/index/doc.json", row)
    shifted = dict(original, source=dict(source, block_start=99, block_indices=[99]),
                   offline_repair={"new_llm_calls": 0},
                   reactions=[reaction(1, [], [compound("Wrong method product", "CCC")])])
    write_gzip_jsonl(out / "scan/records/doc.json.gz", [shifted])
    monkeypatch.setattr(route_supplement, "_select_chunks", lambda *_: ([], ["example_1"], {}))
    summary = route_supplement.run_prepare(out)
    restored = read_gzip_jsonl(out / "scan/records/doc.json.gz")[0]
    assert restored["source"]["block_start"] == 10
    assert not restored["reactions"][0]["validation"]["rxn_validation"]["normalized_compounds"]["products"]
    assert summary["counts"]["restored_source_identity_records"] == 1
    assert read_json(out / "scan/supplement_charter.json")["items"] == []
