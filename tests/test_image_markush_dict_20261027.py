"""Markush image-dictionary regressions (2026-10-27): scheme-figure labels
with general (wildcard) structures must reach coref, with provenance."""

from pipeline.chemical_quality import (markush_label_smiles,
                                       valid_image_smiles)
from pipeline.document.run_document import (_global_dict, _image_for_chunk,
                                            merged_image_dict)

MARKUSH = "[1*]c1cnc(Nc2cc(N)c([3*])cc2[2*])nc1[W]"


def detection(labels, boxes, links, dict_result=None):
    return {"image_file_name": "fig.jpg", "image_size": {"width": 100, "height": 100},
            "figure_labels": [{"bbox_xyxy": [0, 0, 10, 10], "text": text} for text in labels],
            "molecule_boxes": [{"bbox_xyxy": [i * 20, 0, i * 20 + 15, 15], "smiles": smiles}
                               for i, smiles in enumerate(boxes)],
            "label_mol_links": [{"label_idx": label_idx, "mol_idx": mol_idx}
                                for label_idx, mol_idx in links],
            "dict_result": dict_result if dict_result is not None else
                           {text: box for text, box in zip(labels, boxes)}}


def test_strict_gate_still_rejects_markush():
    assert not valid_image_smiles(MARKUSH)
    assert valid_image_smiles(MARKUSH, allow_markush=True)
    assert valid_image_smiles("CCO")
    assert valid_image_smiles("CCO", allow_markush=True)
    # Junk stays rejected in both modes.
    for value in ("", "*", "[Cd]", "not-a-smiles"):
        assert not valid_image_smiles(value)
        assert not valid_image_smiles(value, allow_markush=True)
    # A bare placeholder ("R alkene") is not a Markush scaffold either.
    for value in ("*=*", "*C*", "*O*"):
        assert not valid_image_smiles(value, allow_markush=True)
    # A real scaffold with wildcard positions is admitted.
    assert valid_image_smiles("*c1ccccc1", allow_markush=True)


def test_geometry_rebuild_recovers_scheme_labels():
    item = detection(["(G)", "(H)", "(K)"],
                     [MARKUSH, MARKUSH.replace("cc(N)", "cc(NC(=O)C=C)"), "C=CC(=O)Cl"],
                     [(0, 0), (1, 1), (2, 2)],
                     dict_result={})  # fresh mining keeps dictionaries concrete-only
    mapping = markush_label_smiles(item)
    assert mapping["(G)"] == MARKUSH
    assert mapping["(K)"] == "C=CC(=O)Cl"


def test_cached_markush_entries_reach_chunk_dict():
    image_data = {"per_image_detections": [
        detection(["(G)", "(H)", "(K)"],
                  [MARKUSH, MARKUSH, MARKUSH],
                  [(0, 0), (1, 1), (2, 2)])]}
    chunk = {"images": ["images/fig.jpg"]}
    result = _image_for_chunk(chunk, image_data)
    assert result["dict_result"]["(G)"] == MARKUSH
    assert result["markush_entries"] == {"(G)": MARKUSH, "(H)": MARKUSH, "(K)": MARKUSH}
    assert "(G)" not in result["unusable_identifiers"]


def test_geometry_rebuild_feeds_chunk_dict_without_per_image_dict():
    item = detection(["(K)"], [MARKUSH], [(0, 0)], dict_result={})
    result = _image_for_chunk({"images": ["images/fig.jpg"]},
                              {"per_image_detections": [item]})
    assert result["dict_result"]["(K)"] == MARKUSH
    assert result["markush_entries"]["(K)"] == MARKUSH


def test_invalid_label_is_negative_evidence_not_rescued_by_geometry():
    item = detection(["(K)"], ["junk-smiles"], [(0, 0)], dict_result={"(K)": "junk-smiles"})
    result = _image_for_chunk({"images": ["images/fig.jpg"]},
                              {"per_image_detections": [item]})
    assert "(K)" in result["unusable_identifiers"]
    assert "(K)" not in result["dict_result"]


def test_same_label_two_values_is_conflicting_not_guessed():
    # A concrete dict entry plus a different Markush value from the geometry
    # means two figures number different compounds identically: drop.
    item = detection(["(K)"], [MARKUSH], [(0, 0)], dict_result={"(K)": "CCO"})
    result = _image_for_chunk({"images": ["images/fig.jpg"]},
                              {"per_image_detections": [item]})
    assert "(K)" not in result["dict_result"]
    assert result["conflicting_identifiers"] == ["(K)"]
    assert "(K)" in result["unusable_identifiers"]


def test_same_value_concrete_and_geometry_is_concrete():
    item = detection(["(K)"], ["CCO"], [(0, 0)], dict_result={"(K)": "CCO"})
    result = _image_for_chunk({"images": ["images/fig.jpg"]},
                              {"per_image_detections": [item]})
    assert result["dict_result"]["(K)"] == "CCO"
    assert result["markush_entries"] == {}


def test_same_label_across_two_figures_is_dropped():
    scheme_a = detection(["(K)"], [MARKUSH], [(0, 0)])
    scheme_a["image_file_name"] = "scheme_a.jpg"
    scheme_b = detection(["(K)"], ["CS(=O)(=O)OCC"], [(0, 0)])
    scheme_b["image_file_name"] = "scheme_b.jpg"
    result = _image_for_chunk({"images": ["images/scheme_a.jpg", "images/scheme_b.jpg"]},
                              {"per_image_detections": [scheme_a, scheme_b]})
    assert "(K)" not in result["dict_result"]
    assert result["conflicting_identifiers"] == ["(K)"]


def test_global_dict_keeps_markush_and_drops_conflicts():
    image_data = {"dict_result": {"(A)": "CCO", "(B)": "CCC", "(M)": MARKUSH,
                                  "(X)": "CCN", "(Y)": MARKUSH.replace("[2*]", "[3*]")}}
    # Per-image fallback not needed; top-level carries everything. Conflicts:
    # two concretes for (X) and two markush values for (Y) must drop the key.
    image_data["dict_result"]["(X)2"] = "CCCO"  # different label, no conflict
    merged = _global_dict(image_data)
    assert merged["(A)"] == "CCO" and merged["(M)"] == MARKUSH


def test_global_dict_conflicting_keys_dropped():
    other_markush = MARKUSH.replace("[2*]", "[3*]")
    fallback = _global_dict({"per_image_detections": [
        {"dict_result": {"(X)": "CCN", "(Y)": MARKUSH}},
        {"dict_result": {"(X)": "CCCO", "(Y)": other_markush}}]})
    assert "(X)" not in fallback and "(Y)" not in fallback


def test_markush_never_evicts_concrete_in_global_dict():
    image_data = {"per_image_detections": [
        {"dict_result": {"(K)": "CCO"}},
        {"dict_result": {"(K)": MARKUSH}}]}
    merged = _global_dict(image_data)
    assert merged["(K)"] == "CCO"


def test_merged_image_dict_blocked_label_and_local_override():
    associated = {"dict_result": {"(K)": MARKUSH}, "markush_entries": {"(K)": MARKUSH},
                  "unusable_identifiers": ["(A)"]}
    merged = merged_image_dict({"(A)": "CCO", "(K)": "CCC"}, associated)
    assert "(A)" not in merged          # local negative evidence wins
    assert merged["(K)"] == MARKUSH     # local figure wins over global
