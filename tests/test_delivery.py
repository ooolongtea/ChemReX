"""Route assembly and field-preserving delivery contract regressions."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.document.delivery import compact_record, current_records, export_document, export_run, try_export_document
from pipeline.document.routes import assemble_routes
from pipeline.validation.quality import QUALITY_VERSION


def compound(name, smiles="", **extra):
    return {"name": name, "smiles": smiles, "canonical_smiles": smiles,
            "status": "valid" if smiles else "structure_unavailable", "ignored": False,
            "reason": "" if smiles else "unresolved_structure", "source_field": "products",
            "source_index": 0, "structure_source": "name_resolution", "cxsmiles": None, **extra}


def reaction(index, reactants, products, **extra):
    return {"index": index, "step_id": f"step:{index}", "status": "ok", "reason": None,
            "step_source": {"step_id": f"step:{index}", "title": products[0]["name"] if products else "",
                            "local_text": "原文", "source_parts": [{"text": "原文", "block_idx": index, "page_idx": 0}],
                            "procedure_references": [], "unit_kind": "single_step"},
            "quality": {"version": QUALITY_VERSION, "admission_eligible": True, "blocking_issues": [],
                        "structure_status": "supported", "record_completeness": "complete"},
            "validation": {"ner": {"REACTION_PRODUCT": products, "TEMPERATURE": None, "TIME": "", "YIELD_PERCENT": []},
                           "action": ["add"], "raw": {"ner": "large raw"},
                           "yield_info": {"kind": "not_reported", "value": None},
                           "stereochemistry": {"kind": "unreported"},
                           "rxn_validation": {"status": "passed", "needs_human_review": False,
                             "patch_accepted": False, "base_expression": "", "effective_expression": "",
                             "candidates": [{"expression": "debug"}], "normalized_compounds": {
                              "reactants": reactants, "products": products, "reagents": [], "solvents": [], "participants": []}}}, **extra}


def record(cid, reactions, block=0, **source_extra):
    return {"id": cid, "chunk_fingerprint": "fp", "status": "ok", "reason": None,
            "position": {"page_idx": 0, "bbox": None}, "reactions": reactions,
            "source": {"id": cid, "reference_id": cid, "kind": "example", "number": block + 1,
                       "block_start": block, "page_start": 1, "title": cid, "process_kind": "synthesis",
                       "references": [], **source_extra},
            "coref": {"ner": "duplicate"}, "validation": {"ner": "duplicate"}}


def write_document(folder, records):
    folder.mkdir(parents=True)
    (folder / "original.pdf").write_bytes(b"%PDF-1.4\n" + folder.name.encode())
    (folder / "chunks.jsonl").write_text("".join(json.dumps(r["source"] | {"chunk_fingerprint": r["chunk_fingerprint"]}) + "\n" for r in records))
    (folder / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (folder / "document_result.json").write_text(json.dumps({"status": "procedure_processed", "final_product": None}))


def test_branch_merge_shared_intermediate_and_multiple_targets():
    a, b, c, d, e = [compound(n, s) for n, s in zip("ABCDE", ["C", "CC", "CO", "CCO", "CCN"])]
    r = record("example_1", [reaction(1, [], [a]), reaction(2, [a], [b]), reaction(3, [a], [c]),
                              reaction(4, [b, c], [d]), reaction(5, [b], [e])])
    routes, unassigned = assemble_routes([r])
    assert len(routes) == 2 and not unassigned
    methods = [route["methods"][0] for route in routes]
    assert sorted(len(m["steps"]) for m in methods) == [3, 4]
    assert sorted(len(m["connections"]) for m in methods) == [2, 4]
    assert all(m["steps"][0]["index"] == 1 for m in methods)


def test_alternative_methods_and_reference_is_not_dependency():
    product = compound("ethanol", "CCO")
    first = record("example_1", [reaction(1, [], [product])], number=1)
    second = record("example_2", [reaction(1, [product], [product])], block=2,
                    number=2, references=["example_1"])
    second["reactions"][0]["step_source"]["local_text"] = "Prepared by a procedure similar to Example 1."
    routes, _ = assemble_routes([first, second])
    assert len(routes) == 1 and len(routes[0]["methods"]) == 2
    assert not routes[0]["methods"][1]["connections"]
    assert routes[0]["methods"][1]["method_references"] == ["example_1"]


def test_unique_cross_chunk_dependency():
    a, b = compound("A", "CC"), compound("B", "CCC")
    routes, unassigned = assemble_routes([record("example_1", [reaction(1, [], [a])]),
                                         record("example_2", [reaction(1, [a], [b])], block=1)])
    assert len(routes) == 1 and not unassigned
    assert len(routes[0]["methods"][0]["connections"]) == 1


def test_ambiguity_and_explicit_material_reference():
    a, b = compound("A", "CC"), compound("B", "CCC")
    sources = [record("example_1", [reaction(1, [], [a])], number=1),
               record("example_2", [reaction(1, [], [a])], block=1, number=2)]
    destination = record("example_3", [reaction(1, [a], [b])], block=2, references=["example_1"])
    routes, _ = assemble_routes(sources + [destination])
    method = next(r["methods"][0] for r in routes if r["product"]["name"] == "B")
    assert method["issues"][0]["reason"] == "ambiguous_upstream" and not method["admission_eligible"]
    destination["reactions"][0]["step_source"]["local_text"] = "Using A obtained from Example 1."
    routes, _ = assemble_routes(sources + [destination])
    method = next(r["methods"][0] for r in routes if r["product"]["name"] == "B")
    assert method["connections"][0]["from"]["id"] == "example_1" and not method["issues"]


def test_crystal_salt_solvate_aliases_group_with_specific_names_preserved():
    names = ["式1化合物结晶形式I", "式I化合物盐酸盐晶型II", "乙酸乙酯溶剂合物结晶形式V"]
    records = [record(f"example_{i}", [reaction(1, [], [compound(name)])], block=i,
                      title=(name if i < 2 else "式1化合物乙酸乙酯溶剂合物结晶形式V的制备"))
               for i, name in enumerate(names)]
    routes, unassigned = assemble_routes(records, {"product": "Parent", "reference": "Parent（下称式1化合物）"})
    assert len(routes) == 1 and len(routes[0]["methods"]) == 3 and not unassigned
    assert routes[0]["product"]["name"] == "Parent"
    assert [r["reactions"][0]["validation"]["rxn_validation"]["normalized_compounds"]["products"][0]["name"] for r in records] == names


def test_named_parent_and_hydrochloride_group():
    records = [record("parent", [reaction(1, [], [compound("ethylamine", "CCN")])]),
               record("salt", [reaction(1, [], [compound("ethylamine hydrochloride")])], block=1)]
    routes, _ = assemble_routes(records)
    assert len(routes) == 1 and len(routes[0]["methods"]) == 2


def test_distinct_structures_stereo_and_formulations_stay_distinct():
    products = [compound("alcohol", "CCO"), compound("alcohol", "CO"),
                compound("isomer", "N[C@H](C)C(=O)O"), compound("isomer", "N[C@@H](C)C(=O)O")]
    records = [record(str(i), [reaction(1, [], [p])], block=i) for i, p in enumerate(products)]
    routes, _ = assemble_routes(records)
    assert len(routes) == 4
    formulations = [record(str(i), [reaction(1, [], [compound(n, "CCO")])], block=i, process_kind="formulation")
                    for i, n in enumerate(["tablets", "suppositories"])]
    assert len(assemble_routes(formulations)[0]) == 2


def test_unknown_and_conflict_are_retained_and_not_connected():
    unknown = record("unknown", [reaction(1, [], [compound("目标产物")])])
    conflict = record("conflict", [reaction(1, [], [compound("A", "CC", structure_source="adjacent_image_conflict")])], block=1)
    routes, unassigned = assemble_routes([unknown, conflict])
    assert routes == [] and len(unassigned) == 2


def test_same_step_multiple_reactions_have_distinct_references():
    reactions = [reaction(1, [], [compound("A", "C")]), reaction(2, [], [compound("B", "CC")])]
    for item in reactions:
        item["step_id"] = "same"
    routes, _ = assemble_routes([record("example_1", reactions)])
    refs = [r["methods"][0]["steps"][0] for r in routes]
    assert {r["index"] for r in refs} == {1, 2} and {r["step_id"] for r in refs} == {"same"}


def test_disconnected_example_intermediate_is_not_an_extra_target():
    r = record("example_1", [reaction(1, [], [compound("intermediate", "CC")]),
                              reaction(2, [compound("other intermediate", "CO")], [compound("target", "CCO")])])
    r["terminal_product_coverage"] = "unknown"
    routes, unassigned = assemble_routes([r])
    assert len(routes) == 1 and routes[0]["product"]["name"] == "target"
    method = routes[0]["methods"][0]
    assert len(method["steps"]) == 2 and not method["connections"]
    assert method["issues"][0]["reason"] == "disconnected_procedure_steps" and not unassigned
    assert not method["admission_eligible"]


def test_relative_stereochemistry_is_not_merged():
    records = [record(str(i), [reaction(1, [], [compound("isomer", "CCO", stereochemistry={"kind": "relative", "descriptors": [d]})])], block=i)
               for i, d in enumerate(["cis", "trans"])]
    assert len(assemble_routes(records)[0]) == 2


def test_projection_keeps_types_values_quality_and_fallback_paths():
    original = record("one", [reaction(1, [], [compound("A", "C", naming_raw="raw")])])
    saved = deepcopy(original)
    compact = compact_record(original)
    assert original == saved and "coref" not in compact
    item = compact["reactions"][0]
    assert item["quality"] == original["reactions"][0]["quality"]
    assert item["validation"]["ner"] == {"TEMPERATURE": None, "TIME": "", "YIELD_PERCENT": []}
    assert "raw" not in item["validation"] and "candidates" not in item["validation"]["rxn_validation"]
    assert "smiles" not in item["validation"]["rxn_validation"]["normalized_compounds"]["products"][0]
    assert item["step_source"]["local_text"] == "原文" and "text" not in item["step_source"]["source_parts"][0]
    original["reactions"][0]["validation"] = None
    original["reactions"][0]["coref"] = {"ner": {"REACTION_PRODUCT": [{"name": "A", "smiles": None}], "TIME": []}}
    item = compact_record(original)["reactions"][0]
    assert item["coref"]["ner"]["REACTION_PRODUCT"][0]["smiles"] is None and "validation" not in item
    original["reactions"][0]["coref"] = None
    original["reactions"][0]["extraction"] = {"ner": {}, "action": [], "raw": {"ner": "failed"}}
    assert compact_record(original)["reactions"][0]["extraction"] == {"ner": {}, "action": []}


def test_mapping_only_validator_keeps_business_fallback():
    r = reaction(1, [], [compound("A", "C")])
    r["validation"] = {"rxn_validation": {"status": "skipped"}}
    r["coref"] = {"ner": {"REACTION_PRODUCT": [compound("A", "C")], "TIME": "2h"}, "action": ["stir"]}
    compact = compact_record(record("one", [r]))["reactions"][0]
    assert compact["validation"]["rxn_validation"]["status"] == "skipped"
    assert compact["coref"]["ner"]["TIME"] == "2h"
    assert compact["coref"]["ner"]["REACTION_PRODUCT"][0]["smiles"] == "C"


def test_single_document_runner_defaults_and_disable(tmp_path):
    from pipeline.document.run_document import main
    content = tmp_path / "content.json"
    content.write_text(json.dumps([{"type": "text", "text": "Example 1: Ethanol", "page_idx": 0},
                                    {"type": "text", "text": "A mixture was stirred at 25 C for 2 h to afford ethanol.", "page_idx": 0}]))
    pdf = tmp_path / "original.pdf"
    pdf.write_bytes(b"%PDF-1.4\nrunner")
    def process(pipe, chunk, *args):
        return dict(record(chunk["id"], [reaction(1, [], [compound("ethanol", "CCO")])]),
                    source=chunk, position=chunk["position"])
    for disabled in [False, True]:
        output = tmp_path / ("disabled" if disabled else "enabled")
        argv = ["document", "--content-list", str(content), "--source-pdf", str(pdf), "--output-dir", str(output), "--skip-final-product"]
        if disabled:
            argv.append("--no-delivery")
        with patch("sys.argv", argv), patch("pipeline.pipeline.ChemReXPipeline"), \
                patch("pipeline.document.run_document.process_chunk", side_effect=process) as processing:
            main()
            assert processing.call_count == 1
        summary = json.loads((output / "summary.json").read_text())
        assert (output / "delivery").exists() is not disabled
        if not disabled:
            assert summary["delivery"]["status"] == "ready"


def test_versioned_final_product_cache_overrides_deferred_stub(tmp_path):
    from pipeline.document.target_extractor import FINAL_PRODUCT_VERSION
    names = ["式1化合物结晶形式I", "式I化合物盐酸盐晶型II", "乙酸乙酯溶剂合物结晶形式V"]
    records = [record(f"example_{i}", [reaction(1, [], [compound(n)])], block=i,
                      title=(n if i < 2 else "式1化合物乙酸乙酯溶剂合物结晶形式V的制备"))
               for i, n in enumerate(names)]

    def archived_target_name(source):
        output = tmp_path / f"delivery_{source.name}"
        export_document(source, output)
        document = json.loads((output / "artifacts/document_result.json").read_text().splitlines()[0])
        return sorted({group["product"]["name"] for group in document["routes"]})

    # Without a final product the grouped target keeps its formula label.
    bare = tmp_path / "bare"
    write_document(bare, records)
    assert archived_target_name(bare) == ["式1化合物"]

    # A stale cache version must not override the document result.
    stale = tmp_path / "stale"
    write_document(stale, records)
    (stale / "final_product.json").write_text(json.dumps(
        {"_version": "outdated", "product": "Parent", "reference": "Parent（下称式1化合物）"}))
    assert archived_target_name(stale) == ["式1化合物"]

    # A reprocessed document keeps its recomputed final product in the
    # versioned cache while document_result.json holds a deferred stub;
    # the archived route target must carry the cached product name.
    cached = tmp_path / "cached"
    write_document(cached, records)
    (cached / "final_product.json").write_text(json.dumps(
        {"_version": FINAL_PRODUCT_VERSION, "product": "Parent", "reference": "Parent（下称式1化合物）"}))
    assert archived_target_name(cached) == ["Parent"]


def test_atomic_publication_failure_retains_previous_delivery(tmp_path):
    import os
    source, output = tmp_path / "source", tmp_path / "delivery"
    write_document(source, [record("a", [reaction(1, [], [compound("A", "C")])])])
    export_document(source, output)
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    replace = os.replace
    def fail_publication(src, dst):
        if Path(src).name == "new":
            raise OSError("publication failed")
        replace(src, dst)
    with patch("pipeline.document.delivery.os.replace", side_effect=fail_publication), pytest.raises(OSError):
        export_document(source, output)
    assert before == {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}


def test_per_document_verification_rejects_stale_source(tmp_path):
    from scripts.verify_delivery import verify_document
    source = tmp_path / "source"
    write_document(source, [record("a", [reaction(1, [], [compound("A", "C")])])])
    export_document(source)
    verify_document(source)
    with (source / "results.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Source changed"):
        verify_document(source)


def test_untrusted_step_never_promotes_method_or_other_method():
    p = compound("A", "C")
    good, bad = record("one", [reaction(1, [], [p])]), record("two", [reaction(1, [], [p])], block=1)
    bad["status"] = "needs_review"
    routes, _ = assemble_routes([good, bad])
    assert [m["admission_eligible"] for m in routes[0]["methods"]] == [True, False]


def test_fingerprint_last_wins_stable_export_and_source_immutable(tmp_path):
    source, output = tmp_path / "source", tmp_path / "delivery"
    a = record("a", [reaction(1, [], [compound("A", "C")])], block=2)
    b = record("b", [reaction(1, [], [compound("B", "CC")])], block=1)
    write_document(source, [a, b])
    with (source / "results.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(a, status="needs_review")) + "\n")
        stream.write(json.dumps(dict(a, chunk_fingerprint="stale", status="error")) + "\n")
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    export_document(source, output)
    exported = [json.loads(l) for l in (output / "reactions.jsonl").read_text().splitlines()]
    assert [r["source"]["chunk_id"] for r in exported] == ["b", "a"] and not exported[1]["quality"]["admission_eligible"]
    first = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    export_document(source, output)
    assert first == {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert len(exported) == 2
    routes = [json.loads(l) for l in (output / "routes.jsonl").read_text().splitlines()]
    assert {s for r in routes for s in r["steps"]} == {r["reaction_id"] for r in exported}


def test_empty_document_errors_and_overlap_protection(tmp_path):
    root = tmp_path / "run"
    write_document(root / "empty", [])
    manifest = export_run(root, tmp_path / "delivery")
    assert manifest["statistics"]["reactions"] == 0
    document_id = next(iter(manifest["documents"]))
    folder = tmp_path / "delivery" / "empty"
    assert (folder / "reactions.jsonl").read_bytes() == b""
    assert (folder / "routes.jsonl").read_bytes() == b""
    assert manifest["documents"][document_id]["processing_status"] == "complete"
    with pytest.raises(ValueError):
        export_run(root, root)
    with patch("pipeline.document.delivery.export_document", side_effect=OSError("disk full")):
        assert try_export_document(root)["status"] == "error"
    existing = tmp_path / "other"
    existing.mkdir()
    (existing / "keep").write_text("user data")
    with pytest.raises(ValueError):
        export_run(root, existing)
    assert (existing / "keep").read_text() == "user data"


SAMPLE = Path(__file__).resolve().parents[1] / "results/acceptance_sample10/20261004_v9_auditfix_qualified"


@pytest.mark.skipif(not SAMPLE.is_dir(), reason="local acceptance corpus unavailable")
def test_sample10_complete_disposition_and_compact_contract(tmp_path):
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SAMPLE.rglob("*") if p.is_file()}
    output = tmp_path / "delivery"
    manifest = export_run(SAMPLE, output)
    assert manifest["statistics"]["source_reactions"] == 779
    assert manifest["statistics"]["reactions"] == 778
    assert manifest["statistics"]["filtered_reactions"] == 1
    # Old admission evidence is not accepted by the revised quality policy.
    assert manifest["statistics"]["trusted_steps"] == 0
    assert manifest["statistics"]["business_bytes"] < manifest["statistics"]["source_result_bytes"] / 4
    assert manifest["statistics"]["routes"] == 283
    assert manifest["statistics"]["trusted_routes"] == 0
    assert not (output / "artifacts").exists()
    derived, fallbacks = 0, 0
    from pipeline.document.business import validate_snapshot
    for did, control in manifest["documents"].items():
        folder = output / control["source_label"]
        assert {p.name for p in folder.iterdir()} == {"reactions.jsonl", "routes.jsonl", "artifacts"}
        records = [json.loads(l) for l in (folder / "reactions.jsonl").read_text().splitlines()]
        routes = [json.loads(l) for l in (folder / "routes.jsonl").read_text().splitlines()]
        validate_snapshot(did, records, routes)
        assert control["processing_status"] == "complete"
        source = SAMPLE / control["source_label"]
        originals = {r["id"]: r for r in current_records(source)[0]}
        assert (folder / "artifacts/results.jsonl").read_bytes() == (source / "results.jsonl").read_bytes()
        for name, info in control["business_files"].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == info["sha256"]
            assert len((folder / name).read_text().splitlines()) == info["lines"]
        for row in records:
            assert not ({"validation", "coref", "extraction", "route_ids", "method_ids"} & row.keys())
            ref = row["source"]
            original = next(x for x in originals[ref["chunk_id"]]["reactions"] if x["index"] == ref["reaction_index"])
            assert ref["text"] == original["step_source"]["local_text"]
            assert ref["pages"] == sorted({p["page_idx"] + 1 for p in original["step_source"]["source_parts"] if type(p.get("page_idx")) is int})
            value = row["conditions"]["yield"]
            if value and value["kind"] == "derived":
                derived += 1
                basis = value["basis"]
                assert value["value"] == round(100 * basis["product_mmol"] / basis["substrate_mmol"], int(basis["decimals"]))
            if not isinstance((original.get("validation") or {}).get("rxn_validation", {}).get("normalized_compounds"), dict):
                fallbacks += 1
        if "202110246110" in control["source_label"]:
            assert len(routes) == 55 and len({r["target_group_id"] for r in routes}) == 1
        if "Jaypirca" in control["source_label"]:
            by_id = {r["reaction_id"]: r for r in records}
            chosen = [r for r in routes if by_id[r["target"]["reaction_id"]]["source"]["chunk_id"] in {"example_1", "example_1_2"}]
            assert len(chosen) == 2 and len({r["target_group_id"] for r in chosen}) == 1
            assert all(len(r["steps"]) == 3 and len(r["connections"]) == 2 for r in chosen)
    assert derived == 149 and fallbacks == 5
    assert source_hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SAMPLE.rglob("*") if p.is_file()}
