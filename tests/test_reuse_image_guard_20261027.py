"""Reuse-guard and image-label backfill regressions (2026-10-27): a name
carrying an image-side label belongs to the current scope's image dictionary,
never to a stale prior result."""

from pipeline.document.chunk_processing import backfill_image_label_compounds
from pipeline.document.extraction_reuse import ExtractionReuse
from pipeline.document.image_labels import carries_image_label

from test_extraction_reuse import sample

MARKUSH = "[1*]c1cnc(Nc2cc(N)c([3*])cc2[2*])nc1[W]"


def test_carries_image_label_requires_brackets():
    assert carries_image_label("胺(H)", {"(H)"})
    assert carries_image_label("式一化合物(K)", {"(K)"})
    assert carries_image_label("product of Preparation 8 (J)", {"(J)"})
    # A pure-number label must not match by substring.
    assert not carries_image_label("化合物10", {"1", "(1)"})
    assert carries_image_label("化合物(1)", {"(1)"})
    assert not carries_image_label("KOH", {"(K)"})


def test_reuse_skips_image_labelled_names():
    step, ner, record = sample()
    # The prior result resolved "Methanol" — but the current scope's image
    # dictionary owns the "(H)" label this name carries.
    step = dict(step, local_text="将胺(H)与甲醛反应。", text="将胺(H)与甲醛反应。")
    ner = {"STARTING_MATERIAL": ["胺(H)"], "REACTION_PRODUCT": ["formaldehyde"]}
    reuse = ExtractionReuse([record])
    enriched = reuse.enrich_known_names(
        ner, step, "en", 1, image_evidence={"dict_result": {"(H)": "CCN"}})
    # No stale structure may be attached to the image-owned name.
    assert enriched["STARTING_MATERIAL"] == ["胺(H)"]
    # Without image evidence the historical behaviour is unchanged.
    default = reuse.enrich_known_names(ner, step, "en", 1)
    assert default["STARTING_MATERIAL"] == ["胺(H)"]


def test_backfill_fills_identifier_only_product_from_local_dict():
    coref = {"ner": {"REACTION_PRODUCT": [
        {"name": "式一化合物(K)", "identifier_only": True, "smiles": None,
         "resolution_trace": []}]}}
    backfill_image_label_compounds(coref, {"dict_result": {"(K)": MARKUSH}})
    product = coref["ner"]["REACTION_PRODUCT"][0]
    assert product["smiles"] == MARKUSH
    assert product["structure_source"] == "image_identifier"
    assert product["dict_key"] == "(K)"
    assert product["resolution_trace"][-1]["service"] == "image_label_backfill"


def test_backfill_fills_structureless_reactant():
    coref = {"ner": {"STARTING_MATERIAL": [
        {"name": "胺(H)", "smiles": None}]}}
    backfill_image_label_compounds(coref, {"dict_result": {"(H)": MARKUSH}})
    assert coref["ner"]["STARTING_MATERIAL"][0]["smiles"] == MARKUSH


def test_backfill_never_overwrites_existing_structure():
    coref = {"ner": {"REACTION_PRODUCT": [
        {"name": "式一化合物(K)", "smiles": "CCO", "structure_source": "name_resolution"}]}}
    backfill_image_label_compounds(coref, {"dict_result": {"(K)": MARKUSH}})
    assert coref["ner"]["REACTION_PRODUCT"][0]["smiles"] == "CCO"
    assert coref["ner"]["REACTION_PRODUCT"][0]["structure_source"] == "name_resolution"


def test_backfill_ignores_names_without_bracketed_label():
    coref = {"ner": {"REACTION_PRODUCT": [
        {"name": "化合物10", "smiles": None}]}}
    backfill_image_label_compounds(coref, {"dict_result": {"(1)": "CCO", "1": "CCO"}})
    assert coref["ner"]["REACTION_PRODUCT"][0]["smiles"] is None


def test_backfill_skips_invalid_dict_values():
    coref = {"ner": {"REACTION_PRODUCT": [
        {"name": "式一化合物(K)", "smiles": None}]}}
    backfill_image_label_compounds(coref, {"dict_result": {"(K)": "junk-smiles"}})
    assert coref["ner"]["REACTION_PRODUCT"][0]["smiles"] is None
