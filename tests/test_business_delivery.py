"""Business-facing regression cases: identity, source semantics and references."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline.document.business import business_records, business_reaction, validate_snapshot
from pipeline.document.delivery import export_document, export_run
from pipeline.document.routes import assemble_routes
from test_delivery import compound, reaction, record, write_document


def load(folder, name):
    return [json.loads(line) for line in (folder / name).read_text().splitlines()]


def test_filename_ids_are_readable_and_structure_repairs_keep_source_identity(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    r = record("one", [reaction(1, [], [compound("A", "CC")])])
    write_document(source, [r])
    pdf = source / "original.pdf"
    export_document(source, output, pdf)
    before = load(output, "routes.jsonl")[0]
    before_reaction = load(output, "reactions.jsonl")[0]
    assert before["document_id"] == "original.pdf"
    assert before_reaction["reaction_id"] == "original.pdf::one::1"
    assert before["route_id"] == "original.pdf::one::1::route::1"
    assert before["target_group_id"] == "original.pdf::one::1::target::1"
    r["reactions"][0]["validation"]["rxn_validation"]["normalized_compounds"]["products"][0]["canonical_smiles"] = "CCC"
    (source / "results.jsonl").write_text(json.dumps(r) + "\n")
    export_document(source, output, pdf)
    after = load(output, "routes.jsonl")[0]
    assert after["route_id"] == before["route_id"]
    assert load(output, "reactions.jsonl")[0]["reaction_id"] == before_reaction["reaction_id"]
    pdf.rename(source / "renamed.pdf")
    export_document(source, output, source / "renamed.pdf")
    assert load(output, "reactions.jsonl")[0]["document_id"] == "renamed.pdf"


def test_each_product_has_specific_target_even_when_structures_are_equal():
    r = record("multi", [reaction(1, [], [compound("batch A", "CC"), compound("batch B", "CC")])])
    rows, routes, _ = business_records("doc_test", [r], assemble_routes([r])[0])
    assert len(routes) == 2 and len({x["route_id"] for x in routes}) == 2
    assert {x["target"]["compound_id"] for x in routes} == {"c1", "c2"}
    assert len({x["target_group_id"] for x in routes}) == 1
    validate_snapshot("doc_test", rows, routes)


def test_connections_point_to_actual_materials_and_preserve_branches():
    a, b, c, d = [compound(name, smiles) for name, smiles in zip("ABCD", ["C", "CC", "CO", "CCO"])]
    r = record("branch", [reaction(1, [], [a]), reaction(2, [a], [b]),
                          reaction(3, [a], [c]), reaction(4, [b, c], [d])])
    rows, routes, _ = business_records("doc_test", [r], assemble_routes([r])[0])
    assert len(routes) == 1 and len(routes[0]["connections"]) == 4
    by_id = {x["reaction_id"]: x for x in rows}
    for edge in routes[0]["connections"]:
        product = next(c for c in by_id[edge["from"]]["compounds"] if c["compound_id"] == edge["product_id"])
        ingredient = next(c for c in by_id[edge["to"]]["compounds"] if c["compound_id"] == edge["input_id"])
        assert product["canonical_smiles"] == ingredient["canonical_smiles"]
        assert "product" in product["roles"] and "reactant" in ingredient["roles"]


def test_normalized_empty_is_authoritative_and_ignored_solvent_is_retained():
    r = reaction(1, [], [])
    r["coref"] = {"ner": {"REACTION_PRODUCT": [compound("fallback", "CC")]}}
    normal = r["validation"]["rxn_validation"]["normalized_compounds"]
    normal["solvents"] = [compound("water", "O", ignored=True, source_field="solvents")]
    row, _ = business_reaction("doc_test", record("one", [r]), r)
    assert row["compounds"] == [{"compound_id": "c1", "roles": ["solvent"], "name": "water", "canonical_smiles": "O"}]


def test_fallback_structure_checked_and_conflicts_never_promoted():
    r = reaction(1, [], [])
    r["validation"] = {"rxn_validation": {"status": "skipped"}}
    r["coref"] = {"ner": {"REACTION_PRODUCT": [{"name": "valid", "smiles": "C(C)O"},
        {"name": "invalid", "smiles": "fake-smiles"},
        {"name": "conflict", "smiles": "CC", "structure_source": "adjacent_image_conflict"}]},
        "action": ["stir"]}
    row, _ = business_reaction("doc_test", record("one", [r]), r)
    assert [c["canonical_smiles"] for c in row["compounds"]] == ["CCO", None, None]
    assert not row["quality"]["admission_eligible"]
    assert "normalized_compounds_missing" in row["quality"]["blocking_issues"]


def test_merge_only_same_source_reagent_participant_and_keep_repeated_additions():
    r = reaction(1, [], [compound("product", "CC")])
    material = compound("reagent", "C", source_field="reagents", source_index=0)
    normal = r["validation"]["rxn_validation"]["normalized_compounds"]
    normal["reagents"] = [material, dict(material, source_index=1)]
    normal["participants"] = [deepcopy(material)]
    row, slots = business_reaction("doc_test", record("one", [r]), r)
    assert len(row["compounds"]) == 3
    assert slots[("reagents", 0)] == slots[("participants", 0)]
    assert slots[("reagents", 1)] != slots[("reagents", 0)]
    assert row["compounds"][1]["roles"] == ["reagent", "participant"]


@pytest.mark.parametrize("info,kind,value", [
    ({"kind": "explicit", "value": "85", "reported_values": ["85"], "source_text": ["yield 85%"]}, "explicit", 85),
    ({"kind": "derived", "value": "50", "basis": {"substrate_mmol": "2", "product_mmol": "1", "formula": "100 * product_mmol / substrate_mmol", "decimals": 2}}, "derived", 50),
    ({"kind": "not_reported", "value": None}, "not_reported", None),
    ({"kind": "not_extracted", "value": "wrong", "reported_values": ["45"]}, "not_extracted", None),
    ({"kind": "unattributed", "value": "85"}, "unattributed", None),
])
def test_yield_semantics_numeric_values_and_no_false_certainty(info, kind, value):
    r = reaction(1, [], [compound("A", "CC")])
    r["validation"]["yield_info"] = info
    row, _ = business_reaction("doc_test", record("one", [r]), r)
    assert row["conditions"]["yield"]["kind"] == kind
    assert row["conditions"]["yield"]["value"] == value
    if kind == "not_extracted":
        assert row["conditions"]["yield"]["reported_values"] == [45]


def test_invalid_yield_never_zero_and_step_pages_never_chunk_range():
    r = reaction(1, [], [compound("A", "CC")])
    r["validation"]["yield_info"] = {"kind": "explicit", "value": "unknown"}
    r["step_source"]["source_parts"] = [{"page_idx": 9}, {"page_idx": 9}, {"page_idx": 10}]
    row, _ = business_reaction("doc_test", record("one", [r], page_start=2, page_end=15), r)
    assert row["conditions"]["yield"] is None
    assert row["source"]["pages"] == [10, 11]
    assert "yield_value_uninterpretable" in row["quality"]["blocking_issues"]
    r["step_source"]["source_parts"] = []
    assert business_reaction("doc_test", record("one", [r], page_start=2), r)[0]["source"]["pages"] == []


def test_stereo_cxsmiles_and_unresolved_procedure_reference_preserved():
    product = compound("cis A", "CCO", cxsmiles="CCO |o1:1|", stereochemistry={"kind": "relative"})
    r = reaction(1, [], [product])
    stereo = {"kind": "mixture", "descriptors": ["cis/trans mixture"], "structure_coarsened": True}
    r["validation"]["stereochemistry"] = stereo
    r["step_source"]["procedure_references"] = [{"id": "procedure_1", "status": "missing", "blocks": [{"page_idx": 2}]}]
    row, _ = business_reaction("doc_test", record("one", [r]), r)
    assert row["stereochemistry"] == stereo
    assert row["compounds"][0]["cxsmiles"] == product["cxsmiles"]
    assert row["source"]["references"] == [{"kind": "procedure", "reference_id": "procedure_1", "status": "missing", "pages": [3]}]
    assert not row["quality"]["admission_eligible"]


def test_old_expression_requires_accepted_patch_and_effective_null_is_authoritative():
    r = reaction(1, [], [compound("A", "CC")])
    rv = r["validation"]["rxn_validation"]
    rv.pop("effective_expression")
    rv.update(base_expression="C>>CC", best_expression="C>>CCC", patch_accepted=False)
    assert business_reaction("doc_test", record("one", [r]), r)[0]["reaction_smiles"] == "C>>CC"
    rv["patch_accepted"] = True
    assert business_reaction("doc_test", record("one", [r]), r)[0]["reaction_smiles"] == "C>>CCC"
    rv["effective_expression"] = None
    assert business_reaction("doc_test", record("one", [r]), r)[0]["reaction_smiles"] is None


def test_missing_pdf_fails_instead_of_hashing_the_content_list(tmp_path):
    source = tmp_path / "source"
    write_document(source, [])
    (source / "original.pdf").unlink()
    with pytest.raises(FileNotFoundError, match="Original PDF required"):
        export_document(source)
    assert not (source / "delivery").exists()


def test_missing_chunks_and_failed_processing_have_explicit_control(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    write_document(source, [record("one", [reaction(1, [], [compound("A", "CC")])])])
    with (source / "chunks.jsonl").open("a") as stream:
        stream.write(json.dumps({"id": "missing", "chunk_fingerprint": "fp"}) + "\n")
    export_document(source, output)
    manifest = json.loads((output / "artifacts/manifest.json").read_text())
    assert manifest["processing_status"] == "partial" and manifest["missing_chunks"] == ["missing"]
    (source / "results.jsonl").write_text("")
    export_document(source, output)
    manifest = json.loads((output / "artifacts/manifest.json").read_text())
    assert manifest["processing_status"] == "failed"
    assert manifest["business_files"]["reactions.jsonl"]["lines"] == 0


def test_invalid_roles_broken_candidates_and_cycles_are_rejected():
    a, b = compound("A", "C"), compound("B", "CC")
    r = record("one", [reaction(1, [], [a]), reaction(2, [a], [b])])
    rows, routes, _ = business_records("doc_test", [r], assemble_routes([r])[0])
    bad = deepcopy(routes)
    bad[0]["quality"]["blocking_issues"].append({"code": "ambiguous_upstream", "candidate_reaction_ids": ["missing"]})
    with pytest.raises(ValueError, match="ambiguity candidate"):
        validate_snapshot("doc_test", rows, bad)
    bad = deepcopy(routes)
    edge = deepcopy(bad[0]["connections"][0])
    edge["input_id"] = "c2"
    with pytest.raises(ValueError, match="material roles"):
        validate_snapshot("doc_test", rows, [{**bad[0], "connections": [edge]}])
    # A legal material pair with a self-loop must still be rejected.
    bad = deepcopy(routes)
    second = rows[1]
    bad[0]["connections"].append({"from": second["reaction_id"], "to": second["reaction_id"],
        "product_id": "c2", "input_id": "c1", "basis": "exact_name"})
    with pytest.raises(ValueError, match="Cyclic"):
        validate_snapshot("doc_test", rows, bad)


def test_title_only_target_is_not_a_fabricated_compound():
    r = record("one", [reaction(1, [], [])], title="Tablet preparation", process_kind="formulation")
    rows, routes, _ = business_records("doc_test", [r], assemble_routes([r])[0])
    assert rows[0]["compounds"] == []
    assert routes[0]["target"]["compound_id"] is None
    assert routes[0]["target"]["basis"] == "source_title"
    assert routes[0]["target_group_basis"] == "source_title"
    assert not routes[0]["quality"]["admission_eligible"]


def test_no_method_reference_becomes_a_material_edge():
    a = compound("A", "CC")
    first = record("example_1", [reaction(1, [], [a])], number=1)
    second = record("example_2", [reaction(1, [a], [a])], block=2, references=["example_1"])
    second["reactions"][0]["step_source"]["local_text"] = "Prepared by a procedure similar to Example 1."
    _, routes, _ = business_records("doc_test", [first, second], assemble_routes([first, second])[0])
    assert len(routes) == 2 and not any(r["connections"] for r in routes)


def test_unknown_policy_and_string_boolean_cannot_be_admitted():
    r = reaction(1, [], [compound("A", "CC")])
    for changes in [{"version": "future"}, {"admission_eligible": "true"}]:
        bad = deepcopy(r)
        bad["quality"].update(changes)
        row, _ = business_reaction("doc_test", record("one", [bad]), bad)
        assert row["quality"]["admission_eligible"] is False
        assert row["quality"]["blocking_issues"]


def test_structural_target_can_have_unknown_name():
    r = record("one", [reaction(1, [], [compound(None, "CCO")])])
    rows, routes, _ = business_records("doc_test", [r], assemble_routes([r])[0])
    assert len(routes) == 1 and rows[0]["compounds"][0]["name"] is None
    assert routes[0]["target_group_basis"] == "canonical_smiles"


def test_missing_result_file_is_distinct_from_a_complete_empty_file(tmp_path):
    source = tmp_path / "source"
    write_document(source, [])
    (source / "results.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="results.jsonl missing"):
        export_document(source)


def test_portable_verifier_detects_business_and_archive_tampering(tmp_path):
    from scripts.verify_delivery import verify_document
    source, output = tmp_path / "source", tmp_path / "output"
    write_document(source, [record("one", [reaction(1, [], [compound("A", "CC")])])])
    export_document(source, output)
    assert verify_document(output)["processing_status"] == "complete"
    for name in ["reactions.jsonl", "artifacts/results.jsonl"]:
        path = output / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="integrity mismatch"):
            verify_document(output)
        path.write_bytes(original)


def test_duplicate_filenames_use_result_folder_and_export_stays_local(tmp_path):
    from scripts.verify_delivery import verify_run
    root = tmp_path / "parsed"
    for label in ["first", "second"]:
        write_document(root / label, [record("one", [reaction(1, [], [compound("A", "CC")])])])
    original_root_files = {p.name for p in root.iterdir()}
    export_run(root)
    assert {p.name for p in root.iterdir()} == original_root_files
    for label in ["first", "second"]:
        source = root / label
        assert load(source / "delivery", "reactions.jsonl")[0]["document_id"] == label + "/original.pdf"
        before = (source / "delivery/reactions.jsonl").read_bytes()
        export_document(source)
        assert (source / "delivery/reactions.jsonl").read_bytes() == before
        export_run(source)
        assert (source / "delivery/reactions.jsonl").read_bytes() == before
    assert verify_run(root)["verified_documents"] == 2


def test_local_delivery_verifier_rejects_tampered_business_file(tmp_path):
    from scripts.verify_delivery import verify_document
    source = tmp_path / "source"
    write_document(source, [record("one", [reaction(1, [], [compound("A", "CC")])])])
    export_document(source)
    with (source / "delivery/routes.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_document(source)


def test_source_models_are_exported_per_reaction_and_manifest(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    configuration = {"model": "qwen3-lora-08", "preset": "qwen3-lora-08",
                     "prompts": {"ner": {"version": "v1"},
                                 "action": {"version": "v1"}}}
    r = reaction(1, [], [compound("A", "CC")],
                 extraction={"extraction_configuration": configuration})
    write_document(source, [record("one", [r])])
    (source / "document_result.json").write_text(json.dumps({
        "status": "procedure_processed", "final_product": None,
        "source_models": {
            "m1": {"model": "qwen3-lora-08", "preset": "qwen3-lora-08"},
            "m3": {"model": "deepseek-v4.1-flash", "provider": "maas"},
            "m4": {"model": "deepseek-v4.1-flash", "provider": "maas"},
        },
    }))
    export_document(source, output, source / "original.pdf")

    reaction_row = load(output, "reactions.jsonl")[0]
    assert reaction_row["source_models"]["m1"]["model"] == "qwen3-lora-08"
    manifest = json.loads((output / "artifacts/manifest.json").read_text())
    assert manifest["source_models"]["m3"]["model"] == "deepseek-v4.1-flash"
