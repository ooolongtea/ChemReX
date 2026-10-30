"""Offline unit tests for the drug provenance link module.

All inputs are small synthetic metadata files written under ``tmp_path``; the
tests never read the server data directories and never touch the network.  The
route-repair relation regressions at the bottom use the same synthetic style.
"""
import csv
import json

import pytest

from pipeline.document import drug_links
from pipeline.document.route_repair import _relation_for_target, _route_drug_rows


FDA_METADATA = {
    "products": [
        {
            "ingredient": "MINOCYCLINE HYDROCHLORIDE",
            "trade_name": "AMZEEQ",
            "application_number": "212379",
            "product_number": "001",
            "dosage_form": "AEROSOL, FOAM",
            "route": "TOPICAL",
            "strength": "4%",
            "marketing_status": "RX",
            "patents": [{
                "patent_number": "10137200",
                "expiry_date": "Oct 1, 2030",
                "pdf_status": "downloaded",
                "pdf_local_path": "/meta/pdf/MINOCYCLINE HYDROCHLORIDE/US10137200.pdf",
            }],
        },
        {
            "ingredient": "MINOCYCLINE HYDROCHLORIDE",
            "trade_name": "AMZEEQ",
            "application_number": "212379",
            "product_number": "002",
            "dosage_form": "AEROSOL, FOAM",
            "route": "TOPICAL",
            "strength": "2%",
            "marketing_status": "RX",
            "patents": [{
                "patent_number": "10137200",
                "expiry_date": "Oct 1, 2030",
                "pdf_status": "downloaded",
                "pdf_local_path": "/meta/pdf/MINOCYCLINE HYDROCHLORIDE/US10137200.pdf",
            }],
        },
        {
            "ingredient": "BUPROPION HYDROBROMIDE",
            "application_number": "987654",
            "product_number": "001",
            "dosage_form": "TABLET",
            "route": "ORAL",
            "strength": "100MG",
            "marketing_status": "RX",
            "patents": [{"patent_number": "RE43879", "pdf_status": "downloaded"}],
        },
        {
            "ingredient": "ESTRADIOL; PROGESTERONE",
            "trade_name": "COMBI",
            "application_number": "111111",
            "product_number": "001",
            "dosage_form": "CAPSULE",
            "route": "ORAL",
            "strength": "1MG/100MG",
            "marketing_status": "RX",
            "patents": [{
                "patent_number": "7654321",
                "pdf_status": "downloaded",
                "pdf_local_path": "/meta/pdf/ESTRADIOL; PROGESTERONE/US7654321.pdf",
            }],
        },
        {
            "ingredient": "ARIPIPRAZOLE",
            "application_number": "222222",
            "product_number": "001",
            "dosage_form": "TABLET",
            "route": "ORAL",
            "strength": "5MG",
            "marketing_status": "RX",
            "patents": [{
                "patent_number": "8888888",
                "pdf_status": "downloaded",
                "pdf_local_path": "/meta/pdf/ARIPIPRAZOLE/US8888888.pdf",
            }],
        },
        {
            "ingredient": "ETHANOL",
            "application_number": "333333",
            "product_number": "001",
            "dosage_form": "SOLUTION",
            "route": "TOPICAL",
            "strength": "70%",
            "marketing_status": "OTC",
            "patents": [{
                "patent_number": "9999999",
                "pdf_status": "downloaded",
                "pdf_local_path": "/meta/pdf/ETHANOL/US9999999.pdf",
            }],
        },
    ]
}

CN_REGISTRATIONS = [
    {"registration_id": "reg-a", "approval_number": "H20171146", "drug_name": "依维莫司片",
     "dosage_form": "片剂", "strength": "10mg", "registration_status": "已公示"},
    {"registration_id": "reg-b", "approval_number": "H20190001", "drug_name": "利斯的明透皮贴剂（2次/W）",
     "dosage_form": "贴剂", "strength": "9.5mg/24h", "registration_status": "已公示"},
    {"registration_id": "reg-c", "approval_number": "H20210002", "drug_name": "盐酸凯普拉生片",
     "dosage_form": "片剂", "strength": "20mg", "registration_status": "已公示"},
    {"registration_id": "reg-d", "approval_number": "H20200003", "drug_name": "丁苯酞氯化钠注射液",
     "dosage_form": "注射剂", "strength": "25mg/100ml", "registration_status": "已公示"},
    {"registration_id": "reg-e1", "approval_number": "H20200004", "drug_name": "依维莫司片",
     "dosage_form": "片剂", "strength": "5mg", "registration_status": "已公示"},
    {"registration_id": "reg-e2", "approval_number": "H20200005", "drug_name": "依维莫司片",
     "dosage_form": "片剂", "strength": "2.5mg", "registration_status": "已公示"},
]

CN_METADATA = [
    {"registration_id": "reg-a", "approval_number": "H20171146", "drug_name": "依维莫司片",
     "dosage_form": "片剂", "strength": "10mg", "registration_status": "已公示",
     "patent_key": "ZL200780004302.4", "number_type": "application",
     "patent_number_raw": "ZL200780004302.4", "patent_title": "依维莫司制剂",
     "patentee": "诺华股份有限公司", "grant_date": "2016-06-01",
     "publication_number": "", "mapping_status": "unresolved",
     "claims": [{"zlid": 346, "patentStatus": "有效"}],
     "pdf_status": "downloaded",
     "attachments": [{"fileName": "ZL200780004302.4.pdf", "pdf_status": "downloaded",
                      "pdf_local_path": "/cn/pdf/依维莫司片/ZL200780004302.4.pdf"}]},
    {"registration_id": "reg-b", "approval_number": "H20190001", "drug_name": "利斯的明透皮贴剂（2次/W）",
     "dosage_form": "贴剂", "strength": "9.5mg/24h", "registration_status": "已公示",
     "patent_key": "ZL02123000.5", "number_type": "application",
     "patent_number_raw": "ZL 02123000.5", "patent_title": "透皮治疗系统",
     "patentee": "LTS", "grant_date": "2006-03-08",
     "publication_number": "", "mapping_status": "unresolved", "claims": [],
     "pdf_status": "downloaded",
     "attachments": [{"fileName": "ZL 02123000.5授权证明文件.pdf", "pdf_status": "downloaded",
                      "pdf_local_path": "/cn/pdf/利斯的明透皮贴剂（2次_W）/ZL 02123000.5授权证明文件.pdf"}]},
    {"registration_id": "reg-c", "approval_number": "H20210002", "drug_name": "盐酸凯普拉生片",
     "dosage_form": "片剂", "strength": "20mg", "registration_status": "已公示",
     "patent_key": "CN107207432B", "number_type": "publication",
     "patent_number_raw": "CN107207432B", "patent_title": "钾离子竞争性酸阻滞剂",
     "patentee": "Cinclus", "grant_date": "2020-01-01",
     "publication_number": "CN107207432B", "mapping_status": "source_publication",
     "claims": [], "pdf_status": "downloaded",
     "attachments": [{"fileName": "CN107207432B.pdf", "pdf_status": "downloaded",
                      "pdf_local_path": "/cn/pdf/盐酸凯普拉生片/CN107207432B.pdf"}]},
    {"registration_id": "reg-d", "approval_number": "H20200003", "drug_name": "丁苯酞氯化钠注射液",
     "dosage_form": "注射剂", "strength": "25mg/100ml", "registration_status": "已公示",
     "patent_key": "ZL201210184391.8", "number_type": "application",
     "patent_number_raw": "ZL201210184391.8", "patent_title": "丁苯酞注射液",
     "patentee": "石药", "grant_date": "2015-01-01",
     "publication_number": "", "mapping_status": "unresolved", "claims": [],
     "pdf_status": "downloaded",
     "attachments": [{"fileName": "other.pdf", "pdf_status": "downloaded",
                      "pdf_local_path": "/cn/pdf/丁苯酞氯化钠注射液/other.pdf"}]},
    {"registration_id": "reg-e1", "approval_number": "H20200004", "drug_name": "依维莫司片",
     "dosage_form": "片剂", "strength": "5mg", "registration_status": "已公示",
     "patent_key": "ZL201310425110.8", "number_type": "application",
     "patent_number_raw": "ZL201310425110.8", "patent_title": "依维莫司制剂",
     "patentee": "诺华股份有限公司", "grant_date": "2017-11-21",
     "publication_number": "", "mapping_status": "unresolved", "claims": [],
     "pdf_status": "downloaded",
     "attachments": [{"fileName": "shared.pdf", "pdf_status": "downloaded",
                      "pdf_local_path": "/cn/pdf/依维莫司片/shared.pdf"}]},
    {"registration_id": "reg-e2", "approval_number": "H20200005", "drug_name": "依维莫司片",
     "dosage_form": "片剂", "strength": "2.5mg", "registration_status": "已公示",
     "patent_key": "ZL201310425110.8", "number_type": "application",
     "patent_number_raw": "ZL201310425110.8", "patent_title": "依维莫司制剂",
     "patentee": "诺华股份有限公司", "grant_date": "2017-11-21",
     "publication_number": "", "mapping_status": "unresolved", "claims": [],
     "pdf_status": "downloaded",
     "attachments": [{"fileName": "shared.pdf", "pdf_status": "downloaded",
                      "pdf_local_path": "/cn/pdf/依维莫司片/shared.pdf"}]},
]

CHEMBL_ROWS = [
    {"source_file": "synthetic.csv", "source_row": "1", "record_id": "chem-ethanol",
     "cas": "64-17-5", "drug_name": "Ethanol(乙醇)", "active_ingredient_en": "Ethanol",
     "drug_name_cn": "乙醇", "formula": "C2H6O", "smiles": "CCO",
     "source_patent_count": "0", "source_patent_ids": "", "route_flag": "无路线",
     "patent_number": "", "patent_status": "pending_candidate_matching"},
    {"source_file": "synthetic.csv", "source_row": "2", "record_id": "chem-everolimus",
     "cas": "159351-69-6", "drug_name": "Everolimus(依维莫司)",
     "active_ingredient_en": "Everolimus", "drug_name_cn": "依维莫司",
     "formula": "C53H83NO14", "smiles": "CCO", "source_patent_count": "0",
     "source_patent_ids": "", "route_flag": "无路线", "patent_number": "",
     "patent_status": "pending_candidate_matching"},
]

DOCUMENTS = [
    {"source": "fda", "uuid": "u1", "document_id": "US10137200.pdf"},
    {"source": "fda", "uuid": "u2", "document_id": "USRE43879.pdf"},
    {"source": "cn", "uuid": "u3", "document_id": "别称文件.pdf"},
    {"source": "cn", "uuid": "u4", "document_id": "other.pdf"},
    {"source": "fda", "uuid": "u5", "document_id": "US8888888.pdf"},
    {"source": "cn", "uuid": "u6", "document_id": "CN107207432B.pdf"},
    {"source": "cn", "uuid": "u7", "document_id": "mystery.pdf"},
    {"source": "cn", "uuid": "u8", "document_id": "无编号文件.pdf"},
    {"source": "cn", "uuid": "u9", "document_id": "ZL200780004302.4.pdf"},
    {"source": "cn", "uuid": "u10", "document_id": "shared.pdf"},
]

DELIVERABLE_FILES = {
    ("fda", "u1"): {
        "identity": {"pdf_path": "/meta/pdf/MINOCYCLINE HYDROCHLORIDE/US10137200.pdf",
                     "document_id": "US10137200.pdf"},
        "manifest": {"pdf_sha256": "sha-u1", "patent_numbers": ["US10137200"]},
    },
    ("cn", "u3"): {
        "identity": {"pdf_path": "/cn/pdf/利斯的明透皮贴剂（2次_W）/别称文件.pdf",
                     "document_id": "别称文件.pdf"},
    },
    ("cn", "u4"): {
        "identity": {"pdf_path": "/moved/unknown_folder/other.pdf", "document_id": "other.pdf"},
    },
    ("fda", "u5"): {
        "identity": {"pdf_path": "/meta/pdf/ESTRADIOL; PROGESTERONE/US7654321.pdf",
                     "document_id": "US8888888.pdf"},
    },
    ("cn", "u7"): {
        "identity": {"pdf_path": "/nowhere/mystery.pdf", "document_id": "mystery.pdf"},
    },
    ("cn", "u9"): {
        "identity": {"pdf_path": "/cn/pdf/依维莫司片/ZL200780004302.4.pdf",
                     "document_id": "ZL200780004302.4.pdf"},
        "manifest": {"pdf_sha256": "sha-u9"},
    },
    ("cn", "u10"): {
        "identity": {"pdf_path": "/cn/pdf/依维莫司片/shared.pdf",
                     "document_id": "shared.pdf"},
    },
}

T1_FIELDS = {
    "drug_id", "name_raw", "ingredient_en", "base_name", "salt_form", "dosage_form",
    "strength", "route_admin", "trade_name", "approval_number", "application_number",
    "registration_id", "marketing_status", "combo_flag", "combo_parts",
    "chembl_record_id", "smiles", "inchikey", "inchikey_skeleton", "structure_source",
    "provenance",
}

T2_FIELDS = {
    "drug_id", "source", "patent_key", "patent_key_type", "patent_number_raw",
    "publication_number", "patent_title", "patentee", "grant_date", "expiry_date",
    "pdf_status", "pdf_path", "claims", "mapping_status",
}

T3_FIELDS = {
    "document_id", "uuid", "source", "pdf_path", "patent_keys", "drug_ids",
    "primary_drug_id", "match_channels", "ambiguity", "match_basis",
    "unresolved_reason",
}


@pytest.fixture()
def fixture(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    fda_path = metadata_dir / "fda.json"
    fda_path.write_text(json.dumps(FDA_METADATA, ensure_ascii=False), encoding="utf-8")
    cn_path = metadata_dir / "cn_metadata.jsonl"
    cn_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in CN_METADATA),
                       encoding="utf-8")
    registrations_path = metadata_dir / "cn_registrations.jsonl"
    registrations_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in CN_REGISTRATIONS),
        encoding="utf-8")
    chembl_path = metadata_dir / "chembl.csv"
    fields = list(CHEMBL_ROWS[0])
    with chembl_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(CHEMBL_ROWS)

    root = tmp_path / "deliverable"
    for (source, uuid), payload in DELIVERABLE_FILES.items():
        base = root / source / uuid
        base.mkdir(parents=True, exist_ok=True)
        if "identity" in payload:
            (base / "source_identity.json").write_text(
                json.dumps(payload["identity"], ensure_ascii=False), encoding="utf-8")
        if "manifest" in payload:
            artifacts = base / "delivery" / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / "manifest.json").write_text(
                json.dumps(payload["manifest"], ensure_ascii=False), encoding="utf-8")

    sources = drug_links.load_drug_sources(fda_path, cn_path, registrations_path, chembl_path)
    index = drug_links.build_document_index(sources, root, DOCUMENTS)
    return {"sources": sources, "index": index, "root": root, "documents": DOCUMENTS}


# ---------------------------------------------------------------------------
# document index: channel priority, expansion, conflicts, unresolved rows
# ---------------------------------------------------------------------------

def test_pdf_path_channel_wins_and_multiple_products_expand(fixture):
    row = fixture["index"]["US10137200.pdf"]
    # Two product rows of the same ingredient: both ids expand, but this is one
    # drug identity, so it must not be reported as ambiguous.
    assert row["drug_ids"] == ["FDA:212379:001", "FDA:212379:002"]
    assert row["primary_drug_id"] == "FDA:212379:001"
    assert row["ambiguity"] is False
    assert row["match_basis"]["distinct_identities"] == 1
    assert row["match_channels"] == ["pdf_path", "patent_key", "pdf_dir", "name"]
    assert row["source_basis"] == "pdf_path"
    assert row["unresolved_reason"] is None
    assert row["pdf_sha256"] == "sha-u1"
    assert {"value": "10137200", "type": "patent", "source": "filename"} in row["patent_keys"]
    assert {"value": "10137200", "type": "patent", "source": "manifest"} in row["patent_keys"]


def test_reissue_patent_number_falls_back_to_filename(fixture):
    row = fixture["index"]["USRE43879.pdf"]
    assert row["drug_ids"] == ["FDA:987654:001"]
    assert row["match_channels"] == ["patent_key"]
    assert row["pdf_path"] is None
    assert row["unresolved_reason"] is None
    assert {"value": "RE43879", "type": "patent", "source": "filename"} in row["patent_keys"]


def test_cn_pdf_folder_underscore_is_normalised_to_slash(fixture):
    row = fixture["index"]["别称文件.pdf"]
    assert row["drug_ids"] == ["CN:reg-b"]
    assert row["match_channels"][0] == "pdf_dir"


def test_cn_attachment_filename_channel(fixture):
    row = fixture["index"]["other.pdf"]
    assert row["drug_ids"] == ["CN:reg-d"]
    assert row["match_channels"] == ["attachment_filename"]


def test_conflicting_channels_are_all_kept(fixture):
    row = fixture["index"]["US8888888.pdf"]
    # PDF provenance points at ESTRADIOL; PROGESTERONE while the filename is a
    # different patent (ARIPIPRAZOLE).  Both candidates must survive.
    assert row["drug_ids"] == ["FDA:111111:001", "FDA:222222:001"]
    assert row["primary_drug_id"] == "FDA:111111:001"
    assert row["match_channels"][0] == "pdf_path"
    assert "patent_key" in row["match_channels"]
    assert row["ambiguity"] is True


def test_cn_publication_number_channel(fixture):
    row = fixture["index"]["CN107207432B.pdf"]
    assert row["drug_ids"] == ["CN:reg-c"]
    assert row["match_channels"] == ["patent_key"]
    assert {"value": "CN107207432B", "type": "publication", "source": "filename"} in row["patent_keys"]


def test_same_drug_name_registrations_expand_without_ambiguity(fixture):
    # Two registrations of the same drug name share one PDF path; both ids expand
    # but they represent one drug identity, so ``ambiguity`` stays false.
    row = fixture["index"]["shared.pdf"]
    assert row["drug_ids"][:2] == ["CN:reg-e1", "CN:reg-e2"]
    assert row["ambiguity"] is False
    assert row["match_basis"]["distinct_identities"] == 1
    assert row["match_channels"][0] == "pdf_path"


def test_unmatched_documents_keep_reason_codes(fixture):
    unmatched = fixture["index"]["mystery.pdf"]
    assert unmatched["drug_ids"] == []
    assert unmatched["primary_drug_id"] is None
    assert unmatched["match_channels"] == []
    assert unmatched["unresolved_reason"] == "no_metadata_match"
    no_identity = fixture["index"]["无编号文件.pdf"]
    assert no_identity["drug_ids"] == []
    assert no_identity["pdf_path"] is None
    assert no_identity["unresolved_reason"] == "no_source_identity"


def test_document_index_is_stable_and_keyed_by_document_id(fixture):
    reversed_documents = list(reversed(fixture["documents"]))
    second = drug_links.build_document_index(fixture["sources"], fixture["root"], reversed_documents)
    assert set(second) == set(fixture["index"])
    for document_id, row in fixture["index"].items():
        assert second[document_id] == row


def test_document_drugs_rows_have_contract_fields(fixture):
    rows = list(drug_links.document_drugs(fixture["index"]))
    assert [row["document_id"] for row in rows] == sorted(row["document_id"] for row in rows)
    for row in rows:
        assert T3_FIELDS <= set(row)
        assert isinstance(row["patent_keys"], list)
        assert isinstance(row["drug_ids"], list)
        assert isinstance(row["match_channels"], list)
        assert isinstance(row["match_basis"], dict)
    assert rows == list(drug_links.document_drugs(fixture["index"]))


# ---------------------------------------------------------------------------
# T1 catalog
# ---------------------------------------------------------------------------

def test_drug_catalog_contract_and_fda_fields(fixture):
    rows = list(drug_links.drug_catalog(fixture["sources"]))
    assert rows == list(drug_links.drug_catalog(fixture["sources"]))
    assert len(rows) == len(FDA_METADATA["products"]) + len(CN_REGISTRATIONS)
    by_id = {row["drug_id"]: row for row in rows}
    assert len(by_id) == len(rows)
    for row in rows:
        assert T1_FIELDS <= set(row)
        assert isinstance(row["provenance"], dict)

    minocycline = by_id["FDA:212379:001"]
    assert minocycline["ingredient_en"] == "MINOCYCLINE HYDROCHLORIDE"
    assert minocycline["base_name"] == "MINOCYCLINE"
    assert minocycline["salt_form"] == "HYDROCHLORIDE"
    assert minocycline["dosage_form"] == "AEROSOL, FOAM"
    assert minocycline["trade_name"] == "AMZEEQ"
    assert minocycline["application_number"] == "212379"
    assert minocycline["registration_id"] is None
    assert minocycline["combo_flag"] is False

    combo = by_id["FDA:111111:001"]
    assert combo["combo_flag"] is True
    assert combo["combo_parts"] == ["ESTRADIOL", "PROGESTERONE"]
    assert combo["salt_form"] is None


def test_drug_catalog_cn_fields_and_chembl_structure(fixture):
    by_id = {row["drug_id"]: row for row in drug_links.drug_catalog(fixture["sources"])}
    everolimus = by_id["CN:reg-a"]
    assert everolimus["name_raw"] == "依维莫司片"
    assert everolimus["ingredient_en"] is None
    assert everolimus["base_name"] == "依维莫司"
    assert everolimus["salt_form"] is None
    assert everolimus["approval_number"] == "H20171146"
    assert everolimus["registration_id"] == "reg-a"
    assert everolimus["marketing_status"] == "已公示"
    assert everolimus["chembl_record_id"] == "chem-everolimus"
    assert everolimus["smiles"] == "CCO"
    assert everolimus["structure_source"] == "chembl"

    kaipulasheng = by_id["CN:reg-c"]
    assert kaipulasheng["base_name"] == "凯普拉生"
    assert kaipulasheng["salt_form"] == "盐酸"
    assert kaipulasheng["chembl_record_id"] is None


def test_inchikey_and_skeleton_use_rdkit(fixture):
    pytest.importorskip("rdkit")
    by_id = {row["drug_id"]: row for row in drug_links.drug_catalog(fixture["sources"])}
    ethanol = by_id["FDA:333333:001"]
    assert ethanol["inchikey"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert ethanol["inchikey_skeleton"] == "LFQSCWFLJHTTHZ"
    assert ethanol["smiles"] == "CCO"


# ---------------------------------------------------------------------------
# T2 drug-patent rows
# ---------------------------------------------------------------------------

def test_drug_patents_rows_and_deduplication(fixture):
    rows = list(drug_links.drug_patents(fixture["sources"], fixture["index"]))
    assert rows == list(drug_links.drug_patents(fixture["sources"], fixture["index"]))
    for row in rows:
        assert T2_FIELDS <= set(row)
    keys = [(row["drug_id"], row["patent_key"], row["patent_key_type"]) for row in rows]
    assert len(keys) == len(set(keys))

    fda_minocycline = [row for row in rows
                       if row["drug_id"] == "FDA:212379:001" and row["patent_key"] == "10137200"]
    assert len(fda_minocycline) == 1
    assert fda_minocycline[0]["expiry_date"] == "Oct 1, 2030"
    assert fda_minocycline[0]["pdf_path"].endswith("/US10137200.pdf")
    assert fda_minocycline[0]["patent_key_type"] == "patent"

    cn_everolimus = [row for row in rows
                     if row["drug_id"] == "CN:reg-a" and row["patent_key"] == "2007800043024"]
    assert len(cn_everolimus) == 1
    cn_row = cn_everolimus[0]
    assert cn_row["patent_key_type"] == "application"
    assert cn_row["patent_title"] == "依维莫司制剂"
    assert cn_row["patentee"] == "诺华股份有限公司"
    assert cn_row["grant_date"] == "2016-06-01"
    assert cn_row["publication_number"] is None
    assert cn_row["mapping_status"] == "unresolved"
    assert cn_row["claims"] == [{"zlid": 346, "patentStatus": "有效"}]
    assert cn_row["pdf_status"] == "downloaded"

    cn_publication = [row for row in rows
                      if row["drug_id"] == "CN:reg-c" and row["patent_key"] == "CN107207432B"]
    assert len(cn_publication) == 1
    assert cn_publication[0]["patent_key_type"] == "publication"
    assert cn_publication[0]["publication_number"] == "CN107207432B"


def test_drug_patents_only_covers_referenced_drugs(fixture):
    subset = {key: fixture["index"][key] for key in ("USRE43879.pdf",)}
    rows = list(drug_links.drug_patents(fixture["sources"], subset))
    assert {row["drug_id"] for row in rows} == {"FDA:987654:001"}
    assert [row["patent_key"] for row in rows] == ["RE43879"]


# ---------------------------------------------------------------------------
# route-repair drug relations: D3 name matching, D4 precursor roles
# ---------------------------------------------------------------------------

def _relation_drug(name_raw=None, base_name=None, ingredient_en=None,
                   inchikey=None, salt_form=None):
    return {"drug_id": "D:test", "name_raw": name_raw, "base_name": base_name,
            "ingredient_en": ingredient_en, "inchikey": inchikey, "salt_form": salt_form}


def _relation_target(name=None, smiles=None, inchikey=None):
    return {"name": name, "canonical_smiles": smiles, "inchikey": inchikey}


def _route(route_id, name, smiles=None, reaction_ids=()):
    return {"route_id": route_id, "document_id": "doc.pdf",
            "scope_id": "doc.pdf::scope", "patent_id": "US1",
            "reaction_ids": list(reaction_ids),
            "target": {"reaction_id": "rx-" + route_id, "compound_id": "c-" + route_id,
                       "name": name, "canonical_smiles": smiles}}


def _doc_index(drug_id="D1"):
    return {"doc.pdf": {"drug_ids": [drug_id], "primary_drug_id": drug_id,
                        "match_channels": ["pdf_path"], "unresolved_reason": None}}


def test_short_formula_labels_never_match_by_containment():
    # D3: "I"/"A"/"C" are formula labels, not names; they must not be found as
    # substrings of ingredient names such as CRISABOROLE OINTMENT.
    drug = _relation_drug(name_raw="CRISABOROLE OINTMENT", base_name="CRISABOROLE")
    for label in ("I", "A", "C", "1", "II"):
        relation, form, confidence, evidence = _relation_for_target(
            _relation_target(label), drug, ["pdf_path"])
        assert relation == "其他", (label, relation)
        assert (form, confidence, evidence["name_match"]) == ("none", "none", None)
    # A target without name or structure is unresolved, not a name match.
    assert _relation_for_target({}, drug, ["pdf_path"])[0] == "未确认"


def test_token_containment_keeps_salt_and_exact_forms():
    # A longer target that carries the base name as a whole ASCII token is the
    # legitimate salt-form relation.
    relation, form, confidence, evidence = _relation_for_target(
        _relation_target("MINOCYCLINE HYDROCHLORIDE"),
        _relation_drug(name_raw="MINOCYCLINE"), ["pdf_path"])
    assert (relation, form, confidence) == ("相关形式", "salt", "low")
    assert evidence["name_match"] == "contains"
    # CJK adjacency is allowed: the salt prefix does not have to be tokenised.
    relation, form, _, _ = _relation_for_target(
        _relation_target("盐酸凯普拉生"),
        _relation_drug(name_raw="凯普拉生片", base_name="凯普拉生"), ["pdf_path"])
    assert (relation, form) == ("相关形式", "salt")
    # Exact matches are unchanged.
    assert _relation_for_target(
        _relation_target("MINOCYCLINE HYDROCHLORIDE"),
        _relation_drug(name_raw="MINOCYCLINE HYDROCHLORIDE"), ["pdf_path"])[:2] == ("药物", "salt")


def test_short_cjk_base_names_keep_salt_containment():
    # R5: the containment minimum is script-dependent; three-character CJK
    # drug bases must keep matching the names derived from them.
    for target_name, drug_name in (("布洛芬钠", "布洛芬"), ("泼尼松磷酸钠", "泼尼松")):
        relation, form, confidence, evidence = _relation_for_target(
            _relation_target(target_name), _relation_drug(name_raw=drug_name), ["pdf_path"])
        assert (relation, form, confidence) == ("相关形式", "salt", "low"), target_name
        assert evidence["name_match"] == "contains"
    # Two-character CJK keys are still too short for containment.
    assert _relation_for_target(
        _relation_target("甲苯"), _relation_drug(name_raw="二甲苯"), ["pdf_path"])[0] == "其他"
    # Pure-ASCII short labels keep the four-character rule.
    drug = _relation_drug(name_raw="CRISABOROLE OINTMENT", base_name="CRISABOROLE")
    for label in ("I", "A", "C", "1", "II"):
        assert _relation_for_target(_relation_target(label), drug, ["pdf_path"])[0] == "其他"


def test_containment_guard_is_ascii_token_bounded_and_keeps_dosage_path():
    # Names embedded inside an ASCII word are not token matches.
    assert _relation_for_target(
        _relation_target("BOROLE"), _relation_drug(name_raw="CRISABOROLE"),
        ["pdf_path"])[0] == "其他"
    assert _relation_for_target(
        _relation_target("CYCLINE"), _relation_drug(name_raw="MINOCYCLINE"),
        ["pdf_path"])[0] == "其他"
    # A base name inside a dosage label keeps its previous containment result
    # (the existing 剂型 fallback still applies when no drug name matches).
    assert _relation_for_target(
        _relation_target("CRISABOROLE OINTMENT"), _relation_drug(name_raw="CRISABOROLE"),
        ["pdf_path"])[0] == "相关形式"
    assert _relation_for_target(
        _relation_target("CRISABOROLE OINTMENT"), _relation_drug(name_raw="ARIPIPRAZOLE"),
        ["pdf_path"])[:2] == ("剂型", "none")


def test_degradation_products_are_other_not_related_form():
    drug = _relation_drug(name_raw="凯普拉生片", base_name="凯普拉生")
    for name in ("凯普拉生水解产物", "凯普拉生降解杂质", "MINOCYCLINE DEGRADATION PRODUCT"):
        relation, form, confidence, evidence = _relation_for_target(
            _relation_target(name), drug, ["pdf_path"])
        assert (relation, form, confidence) == ("其他", "none", "none"), name
        assert evidence.get("relation_guard") == "degradation_product"
    # Skeleton similarity is kept as evidence but is not a 相关形式.
    relation, form, _, evidence = _relation_for_target(
        _relation_target("凯普拉生代谢物", inchikey="ABCDEFGHIJKLMN-AAAA-BB"),
        _relation_drug(name_raw="凯普拉生", inchikey="ABCDEFGHIJKLMN-CCCC-DD"), ["pdf_path"])
    assert (relation, form) == ("其他", "none")
    assert evidence["structure_match"] == "skeleton"
    # A full identical InChIKey still wins as 药物.
    relation, _, confidence, _ = _relation_for_target(
        _relation_target("MINOCYCLINE 杂质", inchikey="ABCDEFGHIJKLMN-AAAA-BB"),
        _relation_drug(name_raw="MINOCYCLINE", inchikey="ABCDEFGHIJKLMN-AAAA-BB"), ["pdf_path"])
    assert (relation, confidence) == ("药物", "high")


def test_oxidation_and_related_impurity_names_are_other():
    # R6: explicit oxidation/impurity phrases are degradation products too.
    drug = _relation_drug(name_raw="凯普拉生")
    for name in ("凯普拉生氧化产物", "凯普拉生氧化杂质", "凯普拉生有关物质",
                 "MINOCYCLINE OXIDATION PRODUCT", "MINOCYCLINE OXIDIZED DERIVATIVE",
                 "MINOCYCLINE RELATED SUBSTANCES", "MINOCYCLINE PROCESS IMPURITY"):
        relation, form, confidence, evidence = _relation_for_target(
            _relation_target(name), drug, ["pdf_path"])
        assert (relation, form, confidence) == ("其他", "none", "none"), name
        assert evidence.get("relation_guard") == "degradation_product"
    # Bare 氧化/oxid is not a degradation marker: magnesium-oxide names keep
    # their normal containment/exact relation instead of being forced to 其他.
    relation, form, _, evidence = _relation_for_target(
        _relation_target("氧化镁"), _relation_drug(name_raw="氢氧化镁"), ["pdf_path"])
    assert (relation, form) == ("相关形式", "salt")
    assert "relation_guard" not in evidence
    assert _relation_for_target(
        _relation_target("氧化镁"), _relation_drug(name_raw="氧化镁"), ["pdf_path"])[0] == "药物"


def test_precursor_pass_only_uses_route_materials(tmp_path):
    catalog = {"D1": _relation_drug(name_raw="MINOCYCLINE")}
    catalog["D1"]["drug_id"] = "D1"
    routes = [
        _route("drug", "MINOCYCLINE", reaction_ids=["rx-drug"]),
        _route("precursor", "PRECURSOR-1"),
        _route("both-roles", "BOTH-ROLES"),
        _route("solvent", "DMSO"),
        _route("reagent", "DMSO"),
        _route("merged", "TRIETHYLAMINE"),
        _route("product", "SIDE-PRODUCT"),
    ]
    reactions = {"rx-drug": [
        {"compound_id": "c1", "roles": ["reactant"], "name": "PRECURSOR-1",
         "canonical_smiles": None},
        {"compound_id": "c2", "roles": ["reactant", "participant"], "name": "BOTH-ROLES",
         "canonical_smiles": None},
        {"compound_id": "c3", "roles": ["product"], "name": "MINOCYCLINE",
         "canonical_smiles": None},
        {"compound_id": "c4", "roles": ["solvent"], "name": "DMSO", "canonical_smiles": None},
        {"compound_id": "c5", "roles": ["reagent"], "name": "DMSO", "canonical_smiles": "CS(C)=O"},
        {"compound_id": "c6", "roles": ["reagent", "participant"], "name": "TRIETHYLAMINE",
         "canonical_smiles": None},
        {"compound_id": "c7", "roles": ["product"], "name": "SIDE-PRODUCT",
         "canonical_smiles": None},
    ]}
    rows = _route_drug_rows(tmp_path / "out-not-read", routes, [], _doc_index(), catalog, reactions)
    by_route = {row["route_id"]: row for row in rows}
    assert by_route["drug"]["relation_type"] == "药物"
    # Only true route material becomes a precursor (name or structure key).
    assert by_route["precursor"]["relation_type"] == "前体"
    assert by_route["precursor"]["evidence"]["precursor_path"] == ["drug"]
    assert by_route["both-roles"]["relation_type"] == "前体"
    # Solvents, reagents (incl. merged reagent/participant) and products never do.
    for route_id in ("solvent", "reagent", "merged", "product"):
        row = by_route[route_id]
        assert row["relation_type"] == "其他", (route_id, row["relation_type"])
        assert "precursor_path" not in row["evidence"]


def test_precursor_pass_matches_reactant_structure(tmp_path):
    pytest.importorskip("rdkit")
    catalog = {"D1": _relation_drug(name_raw="MINOCYCLINE")}
    catalog["D1"]["drug_id"] = "D1"
    routes = [
        _route("drug", "MINOCYCLINE", reaction_ids=["rx-drug"]),
        _route("intermediate", "ADVANCED INTERMEDIATE", smiles="CCO"),
    ]
    reactions = {"rx-drug": [
        {"compound_id": "c1", "roles": ["reactant"], "name": "PRECURSOR-1",
         "canonical_smiles": "CCO"},
        {"compound_id": "c2", "roles": ["product"], "name": "MINOCYCLINE",
         "canonical_smiles": None},
    ]}
    rows = _route_drug_rows(tmp_path / "out-not-read", routes, [], _doc_index(), catalog, reactions)
    by_route = {row["route_id"]: row for row in rows}
    # The route target is named differently from the reactant but shares the
    # structure, so the structure key must still make it a precursor.
    assert by_route["intermediate"]["relation_type"] == "前体"


def test_supplement_source_models_are_deduplicated_and_json_safe():
    # D5(b): supplement manifests record the M1 model/preset provenance from the
    # re-extracted records (single-reaction records carry it at record level,
    # multi-reaction records per reaction) without non-JSON values.
    from pipeline.document.route_supplement import source_models_from_records

    records = [
        {"extraction": {"extraction_configuration": {"model": "m1-a", "preset": "p-a",
                                                     "prompts": {"ner": {"version": "v1"}}}},
         "reactions": []},
        {"extraction": None,
         "reactions": [
             {"extraction": {"extraction_configuration": {"model": "m1-a", "preset": "p-a"}}},
             {"extraction": {"extraction_configuration": {"model": "m1-b", "preset": "p-b"}}},
         ]},
        {"status": "error"},
    ]
    models = source_models_from_records(records)
    assert models == [{"model": "m1-a", "preset": "p-a"},
                      {"model": "m1-b", "preset": "p-b"}]
    assert json.loads(json.dumps({"m1": models})) == {"m1": models}


def test_precursor_drug_is_determined_by_path_and_keeps_multiple_drugs(tmp_path):
    catalog = {"X": _relation_drug(name_raw="Drug X"),
               "Y": _relation_drug(name_raw="Drug Y"),
               "Z": _relation_drug(name_raw="Drug Z")}
    document = {"doc.pdf": {"drug_ids": ["X", "Y", "Z"], "match_channels": ["patent_key"]}}
    routes = [_route("precursor", "Intermediate P"),
              _route("drug-y", "Drug Y", reaction_ids=["ry"]),
              _route("drug-z", "Drug Z", reaction_ids=["rz"])]
    material = {"roles": ["reactant"], "name": "Intermediate P"}
    rows = _route_drug_rows(tmp_path, routes, [], document, catalog,
                           {"ry": [material], "rz": [material]})
    precursors = [r for r in rows if r["route_id"] == "precursor"]
    assert {(r["drug_id"], r["relation_type"]) for r in precursors} == {
        ("Y", "前体"), ("Z", "前体")}
    assert {r["drug_id"]: r["evidence"]["precursor_path"] for r in precursors} == {
        "Y": ["drug-y"], "Z": ["drug-z"]}


def test_stereo_difference_never_becomes_a_salt_or_exact_name_override():
    from pipeline.document.route_repair import _inchikey
    left, right = "C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"
    drug = _relation_drug(name_raw="Same ambiguous name", inchikey=_inchikey(right))
    for name in ("opposite enantiomer", "Same ambiguous name", "Same ambiguous name hydrochloride"):
        relation, form, confidence, evidence = _relation_for_target(
            _relation_target(name, smiles=left), drug, [])
        assert relation == "未确认" and form == "none" and confidence == "none"
        assert evidence["relation_guard"] == "stereochemistry_mismatch_or_unspecified"


def test_salt_relation_requires_parent_structure_with_matching_stereochemistry():
    from pipeline.document.route_repair import _inchikey
    parent = "C[C@H](N)C(=O)O"
    target = "C[C@H]([NH3+])C(=O)O.[Cl-]"
    drug = dict(_relation_drug(name_raw="Alanine", inchikey=_inchikey(parent)), smiles=parent)
    relation, form, _, evidence = _relation_for_target(
        _relation_target("Alanine hydrochloride", smiles=target), drug, [])
    assert (relation, form) == ("相关形式", "salt")
    assert evidence["structure_match"] == "stereo_preserving_parent"


def test_all_equally_confirmed_drug_candidates_are_exported(tmp_path):
    catalog = {name: _relation_drug(name_raw="Drug") for name in ("D1", "D2")}
    rows = _route_drug_rows(tmp_path, [_route("drug", "Drug")], [],
                           {"doc.pdf": {"drug_ids": ["D1", "D2"]}}, catalog, {})
    assert {r["drug_id"] for r in rows} == {"D1", "D2"}


def test_precursor_name_does_not_override_a_stereo_or_crystal_mismatch(tmp_path):
    catalog = {"D1": _relation_drug(name_raw="MINOCYCLINE")}
    routes = [_route("drug", "MINOCYCLINE", reaction_ids=["rx"]),
              _route("opposite", "Intermediate", "C[C@H](O)C(=O)O"),
              _route("actual", "Intermediate", "C[C@@H](O)C(=O)O"),
              _route("wrong-form", "Material crystal Form B", "CC")]
    reactions = {"rx": [
        {"roles": ["reactant"], "name": "Intermediate", "canonical_smiles": "C[C@@H](O)C(=O)O"},
        {"roles": ["reactant"], "name": "Material crystal Form A", "canonical_smiles": "CC"}]}
    rows = _route_drug_rows(tmp_path, routes, [], _doc_index(), catalog, reactions)
    by_id = {r["route_id"]: r for r in rows}
    assert by_id["actual"]["relation_type"] == "前体"
    assert by_id["opposite"]["relation_type"] != "前体"
    assert by_id["wrong-form"]["relation_type"] != "前体"
