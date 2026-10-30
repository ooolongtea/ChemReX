"""Drug provenance links: FDA/CN drug metadata -> patent -> source document (T1-T3).

Pure offline module: no model calls, no network and no writes.  Importing this
module reads nothing; callers load explicit paths with :func:`load_drug_sources`.

Each drug entity is a regulatory product/registration:

* FDA: ``FDA:{application_number}:{product_number}`` (Orange Book product row).
* CN: ``CN:{registration_id}`` (CDE registration).

Documents are attributed to drugs with a fixed evidence priority
(:data:`CHANNELS`): PDF absolute path > normalised patent key > PDF folder /
attachment file name > drug name (salt and dosage-form stripped).  A document may
keep several drug ids (combination products, repeated authorisations); the
module never silently keeps only the first row.

Route-target relations (T4/T5: "药物/相关形式/剂型/前体/其他/未确认") are
implemented on top of this module: ``route_repair._route_drug_rows`` joins
route targets to these document drug candidates and writes
``drugs/route_drug.jsonl``.  Relation labels and their evidence are produced
there; this module stays a pure document-index layer.
"""
from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Evidence priority for document -> drug attribution.  Lower index wins.
CHANNELS = ("pdf_path", "patent_key", "pdf_dir", "attachment_filename", "name")

# English salt/hydrate suffixes that are stripped for base-name matching only.
EN_SALT_SUFFIXES = (
    "HYDROCHLORIDE", "HYDROBROMIDE", "HYDROIODIDE", "MESYLATE", "BESYLATE",
    "TOSYLATE", "SULFATE", "HEMISULFATE", "CITRATE", "MALATE", "MALEATE",
    "TARTRATE", "FUMARATE", "PHOSPHATE", "DIPHOSPHATE", "SUCCINATE", "ACETATE",
    "LACTATE", "OXALATE", "NITRATE", "BROMIDE", "CHLORIDE", "SODIUM",
    "POTASSIUM", "CALCIUM", "MAGNESIUM", "ZINC", "DIHYDRATE", "MONOHYDRATE",
    "TRIHYDRATE", "HEMIHYDRATE", "ANHYDROUS", "HYDRATE",
)

# Chinese salt prefixes that are stripped for base-name matching only.
CN_SALT_PREFIXES = (
    "二盐酸", "双羟萘酸", "苯磺酸", "对甲苯磺酸", "盐酸", "氢溴酸", "硫酸",
    "甲磺酸", "马来酸", "枸橼酸", "柠檬酸", "富马酸", "酒石酸", "琥珀酸",
    "磷酸", "醋酸", "乳酸", "硝酸", "溴化", "氯化",
)

# Chinese dosage-form suffixes; longest first so "缓释片" wins over "片".
CN_DOSAGE_SUFFIXES = tuple(sorted((
    "注射用", "缓释片", "控释片", "咀嚼片", "分散片", "口腔崩解片", "口崩片",
    "薄膜衣片", "肠溶片", "软胶囊", "吸入粉雾剂", "吸入制剂", "口服混悬剂",
    "口服溶液", "混悬液", "注射液", "注射剂", "滴眼液", "软膏", "乳膏",
    "凝胶贴膏", "贴膏", "贴剂", "栓剂", "气雾剂", "喷雾剂", "凝胶", "糖浆",
    "滴剂", "乳剂", "洗剂", "酊剂", "颗粒", "散剂", "滴丸", "丸", "片",
    "胶囊", "组合包装", "冻干粉", "干混悬剂",
), key=len, reverse=True))

_CN_ZL = re.compile(r"ZL\s*([0-9][0-9.\s]*)", re.I)
_CN_PUBLICATION = re.compile(r"CN\s*([0-9]{6,12})\s*([A-Z]\d?)", re.I)
_CN_APPLICATION = re.compile(r"CN\s*([0-9]{8,12}(?:\s*\.\s*[0-9])?)", re.I)
_BARE_APPLICATION = re.compile(r"(?<![A-Za-z0-9])([0-9]{8,12}\s*\.\s*[0-9])(?![0-9])")
_PAREN_TAIL = re.compile(r"\s*[（(][^（()）]{1,12}[)）]\s*$")


# ---------------------------------------------------------------------------
# small text helpers
# ---------------------------------------------------------------------------

def _text(value):
    """Return stripped text or None; never invent values."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _key(value):
    """Matching key for names/paths fragments: NFKC, upper, no whitespace."""
    text = unicodedata.normalize("NFKC", _text(value) or "")
    text = re.sub(r"\s+", "", text)
    return text.replace("_", "/").upper()


def _pdf_key(value):
    """Normalise a PDF path for exact equality (POSIX separators, no padding)."""
    text = _text(value)
    if not text:
        return None
    return text.replace("\\", "/").strip()


def _digits(value):
    return re.sub(r"\D", "", str(value or ""))


def _unique(values):
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _folder_names(path):
    """Return the PDF's parent folder name, or None."""
    path = _pdf_key(path)
    if not path:
        return None
    parent = os.path.dirname(path)
    return os.path.basename(parent) or None


def _strip_salt_en(name):
    """Split an English ingredient name into ``(base, salt)``.

    Only suffix stripping for name matching; the raw name is always preserved by
    callers.  ``MINOCYCLINE HYDROCHLORIDE`` -> ``("MINOCYCLINE", "HYDROCHLORIDE")``.
    """
    text = _text(name)
    if not text:
        return None, None
    text = unicodedata.normalize("NFKC", text).upper()
    text = re.sub(r"\s+", " ", text).strip()
    salts = []
    while True:
        match = None
        for suffix in EN_SALT_SUFFIXES:
            if text == suffix:
                break
            if text.endswith(" " + suffix):
                match = suffix
                break
        if not match:
            break
        salts.insert(0, match)
        text = text[: -(len(match) + 1)].strip()
    return (text or None), (" ".join(salts) or None)


def _strip_cn_parentheses(key):
    previous = None
    while key and key != previous:
        previous = key
        key = _PAREN_TAIL.sub("", key)
    return key


def _cn_base_key(name):
    """Dosage-form and salt stripped Chinese base name key (matching only)."""
    key = _strip_cn_parentheses(_key(name))
    stripped = True
    while stripped and key:
        stripped = False
        for suffix in CN_DOSAGE_SUFFIXES:
            if key.endswith(suffix) and len(key) > len(suffix):
                key = key[: -len(suffix)]
                stripped = True
                break
    for prefix in CN_SALT_PREFIXES:
        if key.startswith(prefix) and len(key) > len(prefix):
            key = key[len(prefix):]
            break
    return _strip_cn_parentheses(key) or None


def _cn_salt_form(name):
    key = _strip_cn_parentheses(_key(name))
    for suffix in CN_DOSAGE_SUFFIXES:
        if key.endswith(suffix) and len(key) > len(suffix):
            key = key[: -len(suffix)]
            break
    for prefix in CN_SALT_PREFIXES:
        if key.startswith(prefix) and len(key) > len(prefix):
            return prefix
    return None


def _split_components(name):
    return [part.strip() for part in re.split(r"[;]", _text(name) or "") if part.strip()]


# ---------------------------------------------------------------------------
# patent-number normalisation
# ---------------------------------------------------------------------------

def _fda_patent_key(value):
    """FDA patent key: ``US7855211.pdf``/``7,855,211`` -> ``7855211``.

    Reissue patents keep their ``RE`` prefix (``USRE49444`` -> ``RE49444``), which
    is how the Orange Book metadata stores them.
    """
    text = _text(value)
    if not text:
        return None
    text = text.upper().replace(",", "").replace(" ", "")
    if text.endswith(".PDF"):
        text = text[:-4]
    if text.startswith("US"):
        text = text[2:]
    return text or None


def _cn_number_candidates(value):
    """Return ordered ``(type, key)`` candidates parsed from a CN number text.

    ``type`` is ``"application"`` (ZL application number) or ``"publication"``
    (CN publication number).  Keys are digits for applications and
    ``CN{number}{letter}`` for publications.
    """
    text = _text(value) or ""
    result = []

    def push(kind, key):
        if key and (kind, key) not in result:
            result.append((kind, key))

    for match in _CN_ZL.finditer(text):
        push("application", _digits(match.group(1)))
    for match in _CN_PUBLICATION.finditer(text):
        push("publication", "CN" + _digits(match.group(1)) + match.group(2).upper())
    for match in _CN_APPLICATION.finditer(text):
        tail = text[match.end():match.end() + 1]
        if tail and tail.isascii() and tail.isalpha():
            continue  # the CN prefix belongs to a publication number
        push("application", _digits(match.group(1)))
    for match in _BARE_APPLICATION.finditer(text):
        push("application", _digits(match.group(1)))
    return result


def _cn_row_key(row):
    """Best ``(type, key)`` for one CN metadata row."""
    key_text = _text(row.get("patent_key")) or _text(row.get("normalized_number"))
    candidates = _cn_number_candidates(key_text)
    declared = (_text(row.get("number_type")) or "").lower()
    for kind, key in candidates:
        if declared and kind == declared:
            return kind, key
    if candidates:
        return candidates[0]
    return None, None


def _document_patent_keys(source, value):
    """Patent keys encoded in a document name/identifier."""
    if source == "fda":
        key = _fda_patent_key(value)
        return [("patent", key)] if key else []
    return list(_cn_number_candidates(value))


# ---------------------------------------------------------------------------
# structure lookup (RDKit imported lazily, like routes._canonical)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16384)
def _inchikey(smiles):
    """Isomeric InChIKey for a SMILES string, or None when RDKit cannot parse."""
    smiles = _text(smiles)
    if not smiles:
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    try:
        molecule = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(molecule) if molecule is not None else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# data sources and indices
# ---------------------------------------------------------------------------

@dataclass
class DrugSources:
    """Loaded FDA/CN/ChemBL metadata plus in-memory lookup indices."""

    fda: dict = field(default_factory=dict)
    cn_rows: list = field(default_factory=list)
    cn_registrations: list = field(default_factory=list)
    chembl_rows: list = field(default_factory=list)
    fda_path: str = None
    cn_metadata_path: str = None
    cn_registrations_path: str = None
    chembl_path: str = None

    fda_products: list = field(default_factory=list, init=False, repr=False)
    fda_drug_rows: dict = field(default_factory=dict, init=False, repr=False)
    fda_by_pdf: dict = field(default_factory=dict, init=False, repr=False)
    fda_by_patent: dict = field(default_factory=dict, init=False, repr=False)
    fda_by_ingredient: dict = field(default_factory=dict, init=False, repr=False)
    fda_by_tokens: dict = field(default_factory=dict, init=False, repr=False)
    fda_by_base: dict = field(default_factory=dict, init=False, repr=False)

    cn_drug_rows: dict = field(default_factory=dict, init=False, repr=False)
    cn_rows_by_registration: dict = field(default_factory=dict, init=False, repr=False)
    cn_by_pdf: dict = field(default_factory=dict, init=False, repr=False)
    cn_by_file_name: dict = field(default_factory=dict, init=False, repr=False)
    cn_by_application: dict = field(default_factory=dict, init=False, repr=False)
    cn_by_publication: dict = field(default_factory=dict, init=False, repr=False)
    cn_by_drug: dict = field(default_factory=dict, init=False, repr=False)
    cn_by_drug_base: dict = field(default_factory=dict, init=False, repr=False)

    chembl_by_name: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._build_fda_index()
        self._build_cn_index()
        self._build_chembl_index()

    # -- FDA ---------------------------------------------------------------
    def _build_fda_index(self):
        products = self.fda.get("products") if isinstance(self.fda, dict) else self.fda
        self.fda_products = list(products or [])
        for product in self.fda_products:
            if not isinstance(product, dict):
                continue
            drug_id = _fda_drug_id(product)
            self.fda_drug_rows.setdefault(drug_id, product)
            ingredient = _text(product.get("ingredient"))
            if ingredient:
                self.fda_by_ingredient.setdefault(_key(ingredient), []).append(drug_id)
                tokens = tuple(_key(part) for part in _split_components(ingredient))
                if tokens:
                    self.fda_by_tokens.setdefault(frozenset(tokens), []).append(drug_id)
                base, _salt = _strip_salt_en(ingredient)
                if base and len(_split_components(ingredient)) == 1:
                    self.fda_by_base.setdefault(_key(base), []).append(drug_id)
            for patent in product.get("patents") or []:
                if not isinstance(patent, dict):
                    continue
                pdf_path = _pdf_key(patent.get("pdf_local_path"))
                if pdf_path:
                    self.fda_by_pdf.setdefault(pdf_path, []).append((drug_id, patent))
                patent_key = _fda_patent_key(patent.get("patent_number"))
                if patent_key:
                    self.fda_by_patent.setdefault(patent_key, []).append((drug_id, patent))

    # -- CN ----------------------------------------------------------------
    def _build_cn_index(self):
        for registration in self.cn_registrations:
            if not isinstance(registration, dict):
                continue
            registration_id = _text(registration.get("registration_id"))
            if not registration_id:
                continue
            self.cn_drug_rows.setdefault(registration_id, registration)
            self._index_cn_drug_name(registration_id, registration.get("drug_name"))
        for row in self.cn_rows:
            if not isinstance(row, dict):
                continue
            registration_id = _text(row.get("registration_id"))
            if not registration_id:
                continue
            self.cn_rows_by_registration.setdefault(registration_id, []).append(row)
            if registration_id not in self.cn_drug_rows:
                # Keep rows whose registration row is absent from registrations.jsonl.
                self.cn_drug_rows[registration_id] = {
                    "registration_id": registration_id,
                    "drug_name": row.get("drug_name"),
                    "dosage_form": row.get("dosage_form"),
                    "strength": row.get("strength"),
                    "approval_number": row.get("approval_number"),
                    "registration_status": row.get("registration_status"),
                }
                self._index_cn_drug_name(registration_id, row.get("drug_name"))
            for attachment in row.get("attachments") or []:
                if not isinstance(attachment, dict):
                    continue
                pdf_path = _pdf_key(attachment.get("pdf_local_path"))
                if pdf_path:
                    self.cn_by_pdf.setdefault(pdf_path, []).append(registration_id)
                file_name = _text(attachment.get("fileName"))
                if file_name:
                    self.cn_by_file_name.setdefault(file_name, []).append(registration_id)
            kind, key = _cn_row_key(row)
            if kind == "application":
                self.cn_by_application.setdefault(key, []).append(registration_id)
            elif kind == "publication":
                self.cn_by_publication.setdefault(key, []).append(registration_id)

    def _index_cn_drug_name(self, registration_id, name):
        key = _key(name)
        if key:
            self.cn_by_drug.setdefault(key, []).append(registration_id)
        base = _cn_base_key(name)
        if base and base != key:
            self.cn_by_drug_base.setdefault(base, []).append(registration_id)

    # -- ChemBL ------------------------------------------------------------
    def _build_chembl_index(self):
        for row in self.chembl_rows:
            if not isinstance(row, dict):
                continue
            keys = set()
            for value in (row.get("active_ingredient_en"), row.get("drug_name_cn"),
                          _PAREN_TAIL.sub("", _text(row.get("drug_name")) or "")):
                key = _key(value)
                if key:
                    keys.add(key)
            base, _salt = _strip_salt_en(row.get("active_ingredient_en"))
            if base:
                keys.add(_key(base))
            base_cn = _cn_base_key(row.get("drug_name_cn"))
            if base_cn:
                keys.add(base_cn)
            for key in keys:
                self.chembl_by_name.setdefault(key, []).append(row)


def _fda_drug_id(product):
    application = _text(product.get("application_number")) or "NA"
    product_number = _text(product.get("product_number")) or "NA"
    return "FDA:%s:%s" % (application, product_number)


def _cn_drug_id(registration_id):
    return "CN:%s" % registration_id


def _cn_ids(registration_ids):
    return [_cn_drug_id(registration_id) for registration_id in registration_ids]


def _identity_key(sources, drug_id):
    """Molecule/name identity shared by repeated products or registrations.

    Two product rows of the same FDA ingredient (different strength/product
    number) or two CN registrations of the same drug name are the same drug for
    attribution purposes; they expand in ``drug_ids`` but do not create ambiguity.
    """
    if drug_id.startswith("FDA:"):
        product = sources.fda_drug_rows.get(drug_id) or {}
        return "FDA:ING:" + (_key(product.get("ingredient")) or drug_id)
    registration = sources.cn_drug_rows.get(drug_id[3:]) or {}
    return "CN:DRUG:" + (_key(registration.get("drug_name")) or drug_id)


# ---------------------------------------------------------------------------
# loaders (explicit paths; never called at import time)
# ---------------------------------------------------------------------------

def load_drug_sources(fda_path, cn_metadata_path, cn_registrations_path, chembl_path):
    """Load the three read-only metadata sources and build lookup indices.

    ``fda_path`` is the Orange Book JSON (dict with ``products``); CN metadata and
    registrations are JSONL; the ChemBL file is the ``result7_drug_manifest.csv``
    CSV.  The CN jsonl is streamed line by line; the ChemBL CSV is streamed too.
    """
    fda = _read_json(fda_path)
    if fda is None:
        raise FileNotFoundError("FDA drug metadata not readable: %s" % (fda_path,))
    if not isinstance(fda, dict):
        raise ValueError("FDA drug metadata must be a JSON object: %s" % (fda_path,))

    cn_rows = []
    with open(cn_metadata_path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                cn_rows.append(json.loads(line))

    cn_registrations = []
    with open(cn_registrations_path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                cn_registrations.append(json.loads(line))

    chembl_rows = []
    with open(chembl_path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            chembl_rows.append(dict(row))

    return DrugSources(
        fda=fda,
        cn_rows=cn_rows,
        cn_registrations=cn_registrations,
        chembl_rows=chembl_rows,
        fda_path=str(fda_path),
        cn_metadata_path=str(cn_metadata_path),
        cn_registrations_path=str(cn_registrations_path),
        chembl_path=str(chembl_path),
    )


# ---------------------------------------------------------------------------
# ChemBL structure lookup
# ---------------------------------------------------------------------------

def _chembl_entry(sources, candidates):
    for candidate in candidates:
        key = _key(candidate)
        if not key:
            continue
        entries = sources.chembl_by_name.get(key)
        if entries:
            return sorted(entries, key=lambda row: _text(row.get("record_id")) or "")[0]
    return None


def _structure_fields(sources, candidates):
    entry = _chembl_entry(sources, candidates)
    if not entry:
        return {"chembl_record_id": None, "smiles": None, "inchikey": None,
                "inchikey_skeleton": None, "structure_source": None}
    smiles = _text(entry.get("smiles"))
    inchikey = _inchikey(smiles) if smiles else None
    return {
        "chembl_record_id": _text(entry.get("record_id")),
        "smiles": smiles,
        "inchikey": inchikey,
        "inchikey_skeleton": inchikey[:14] if inchikey else None,
        "structure_source": "chembl" if smiles else None,
    }


def _fda_structure_candidates(product):
    ingredient = _text(product.get("ingredient"))
    components = _split_components(ingredient)
    candidates = [ingredient]
    if len(components) == 1:
        base, _salt = _strip_salt_en(ingredient)
        candidates.append(base)
    return candidates


def _cn_structure_candidates(name):
    key_base = _cn_base_key(name)
    candidates = [name]
    if key_base:
        candidates.append(key_base)
    # The ChemBL index is keyed by _key; pass the stripped variants through it.
    return candidates


# ---------------------------------------------------------------------------
# T1 drug catalog
# ---------------------------------------------------------------------------

def drug_catalog(sources):
    """Yield T1 ``drug_entity`` rows (FDA product rows, then CN registrations)."""
    for drug_id in sorted(sources.fda_drug_rows):
        product = sources.fda_drug_rows[drug_id]
        yield _fda_catalog_row(sources, drug_id, product)
    for drug_id in sorted(sources.cn_drug_rows):
        registration = sources.cn_drug_rows[drug_id]
        yield _cn_catalog_row(sources, _cn_drug_id(drug_id), registration)


def _fda_catalog_row(sources, drug_id, product):
    ingredient = _text(product.get("ingredient"))
    components = _split_components(ingredient)
    combo = len(components) > 1
    base, salt = _strip_salt_en(ingredient) if not combo else (None, None)
    return {
        "drug_id": drug_id,
        "name_raw": ingredient,
        "ingredient_en": ingredient,
        "base_name": "; ".join(_strip_salt_en(part)[0] or part for part in components) if combo else base,
        "salt_form": salt,
        "dosage_form": _text(product.get("dosage_form")),
        "strength": _text(product.get("strength")),
        "route_admin": _text(product.get("route")),
        "trade_name": _text(product.get("trade_name")),
        "approval_number": None,
        "application_number": _text(product.get("application_number")),
        "registration_id": None,
        "marketing_status": _text(product.get("marketing_status")),
        "combo_flag": combo,
        "combo_parts": components if combo else None,
        **_structure_fields(sources, _fda_structure_candidates(product)),
        "provenance": {
            "dataset": "fda",
            "path": sources.fda_path,
            "application_number": _text(product.get("application_number")),
            "product_number": _text(product.get("product_number")),
            "trade_name": _text(product.get("trade_name")),
        },
    }


def _cn_catalog_row(sources, drug_id, registration):
    name = _text(registration.get("drug_name"))
    combo = bool(name and ("/" in name or name.startswith("复方") or "组合包装" in name))
    parts = [part.strip() for part in name.split("/") if part.strip()] if name and "/" in name else None
    return {
        "drug_id": drug_id,
        "name_raw": name,
        "ingredient_en": None,
        "base_name": _cn_base_key(name),
        "salt_form": _cn_salt_form(name),
        "dosage_form": _text(registration.get("dosage_form")),
        "strength": _text(registration.get("strength")),
        "route_admin": None,
        "trade_name": None,
        "approval_number": _text(registration.get("approval_number")),
        "application_number": None,
        "registration_id": _text(registration.get("registration_id")),
        "marketing_status": _text(registration.get("registration_status")),
        "combo_flag": combo,
        "combo_parts": parts,
        **_structure_fields(sources, _cn_structure_candidates(name)),
        "provenance": {
            "dataset": "cn",
            "path": sources.cn_registrations_path or sources.cn_metadata_path,
            "registration_id": _text(registration.get("registration_id")),
        },
    }


# ---------------------------------------------------------------------------
# document -> drug index (T3 core)
# ---------------------------------------------------------------------------

def _iter_documents(documents):
    if isinstance(documents, (str, os.PathLike)):
        with open(documents, encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if hasattr(documents, "read"):
        for line in documents:
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    for document in documents or ():
        yield document


def _candidate_paths(deliverable_root, document):
    source = _text(document.get("source")) or ""
    uuid = _text(document.get("uuid")) or ""
    identity_paths, manifest_paths = [], []
    if deliverable_root is not None and source and uuid:
        base = Path(str(deliverable_root)) / source / uuid
        identity_paths.append(base / "source_identity.json")
        manifest_paths.append(base / "delivery" / "artifacts" / "manifest.json")
    output_dir = _text(document.get("output_dir"))
    if output_dir:
        output_dir = Path(output_dir)
        identity_paths.append(output_dir / "source_identity.json")
        manifest_paths.append(output_dir / "delivery" / "artifacts" / "manifest.json")
    delivery_dir = _text(document.get("delivery_dir"))
    if delivery_dir:
        manifest_paths.append(Path(delivery_dir) / "artifacts" / "manifest.json")
    return identity_paths, manifest_paths


def _first_json(paths):
    for path in paths:
        if os.path.isfile(path):
            return _read_json(path) or {}
    return {}


def build_document_index(sources, deliverable_root, documents):
    """Build ``{document_id: doc_row}`` with drug candidates and match channels.

    ``documents`` may be a ``documents.jsonl`` path, an open stream or an iterable
    of dicts (the deliverable ``documents.jsonl`` rows).  PDF provenance comes from
    each document's ``source_identity.json``; the delivery artifact manifest only
    adds the PDF checksum and filename-derived patent numbers.
    """
    result = {}
    ordered = sorted(_iter_documents(documents),
                     key=lambda row: (_text(row.get("source")) or "",
                                      _text(row.get("document_id")) or "",
                                      _text(row.get("uuid")) or ""))
    for document in ordered:
        document_id = _text(document.get("document_id"))
        if not document_id:
            raise ValueError("document row without document_id: %r" % (document,))
        if document_id in result:
            raise ValueError("duplicate document_id in deliverable: %s" % document_id)
        result[document_id] = _document_row(sources, deliverable_root, document)
    return result


def _document_row(sources, deliverable_root, document):
    source = _text(document.get("source")) or ""
    identity_paths, manifest_paths = _candidate_paths(deliverable_root, document)
    identity = _first_json(identity_paths)
    manifest = _first_json(manifest_paths)
    pdf_path = _pdf_key(identity.get("pdf_path"))
    pdf_sha256 = _text(manifest.get("pdf_sha256"))

    document_id = _text(document.get("document_id"))
    locators = [document_id]
    if pdf_path:
        pdf_name = os.path.basename(pdf_path)
        if pdf_name not in locators:
            locators.append(pdf_name)
    patent_keys = []
    for locator in locators:
        for kind, key in _document_patent_keys(source, locator):
            patent_keys.append({"value": key, "type": kind, "source": "filename"})
    for raw in manifest.get("patent_numbers") or []:
        for kind, key in _document_patent_keys(source, raw):
            patent_keys.append({"value": key, "type": kind, "source": "manifest"})
    patent_keys = sorted(
        {(item["value"], item["type"], item["source"]): item for item in patent_keys}.values(),
        key=lambda item: (item["source"], item["type"], item["value"]))

    channels = {}
    folder = _folder_names(pdf_path)

    def add(channel, drug_ids):
        for drug_id in sorted(set(drug_ids or ())):
            channels.setdefault(channel, [])
            if drug_id not in channels[channel]:
                channels[channel].append(drug_id)

    if source == "fda":
        if sources.fda_products:
            if pdf_path:
                add("pdf_path", [drug_id for drug_id, _patent in sources.fda_by_pdf.get(pdf_path, ())])
            patent_ids = []
            for item in patent_keys:
                if item["type"] == "patent":
                    patent_ids.extend(drug_id for drug_id, _patent
                                      in sources.fda_by_patent.get(item["value"], ()))
            add("patent_key", patent_ids)
            if folder:
                add("pdf_dir", _fda_folder_drug_ids(sources, folder))
                add("name", _fda_name_drug_ids(sources, folder))
    elif source == "cn":
        if sources.cn_drug_rows:
            if pdf_path:
                add("pdf_path", _cn_ids(sources.cn_by_pdf.get(pdf_path, ())))
                add("attachment_filename",
                    _cn_ids(sources.cn_by_file_name.get(os.path.basename(pdf_path), ())))
            patent_ids = []
            for item in patent_keys:
                if item["type"] == "application":
                    patent_ids.extend(_cn_ids(sources.cn_by_application.get(item["value"], ())))
                elif item["type"] == "publication":
                    patent_ids.extend(_cn_ids(sources.cn_by_publication.get(item["value"], ())))
            add("patent_key", patent_ids)
            if folder:
                add("pdf_dir", _cn_ids(sources.cn_by_drug.get(_key(folder), ())))
                base = _cn_base_key(folder)
                if base:
                    add("name", _cn_ids(sources.cn_by_drug_base.get(base, ())))

    drug_ids = []
    for channel in CHANNELS:
        for drug_id in channels.get(channel, ()):
            if drug_id not in drug_ids:
                drug_ids.append(drug_id)
    identity_keys = []
    for drug_id in drug_ids:
        identity = _identity_key(sources, drug_id)
        if identity not in identity_keys:
            identity_keys.append(identity)
    match_channels = [channel for channel in CHANNELS if channels.get(channel)]
    if drug_ids:
        source_basis = match_channels[0]
        unresolved_reason = None
    else:
        source_basis = "unresolved"
        unresolved_reason = _unresolved_reason(sources, source, identity_paths, pdf_path, patent_keys)

    return {
        "document_id": document_id,
        "uuid": _text(document.get("uuid")),
        "source": source,
        "pdf_path": pdf_path,
        "pdf_sha256": pdf_sha256,
        "patent_keys": patent_keys,
        "drug_ids": drug_ids,
        "primary_drug_id": drug_ids[0] if drug_ids else None,
        "match_channels": match_channels,
        "ambiguity": len(identity_keys) > 1,
        "source_basis": source_basis,
        "match_basis": {
            "primary": source_basis,
            "channels": match_channels,
            "candidate_counts": {channel: len(channels[channel]) for channel in match_channels},
            "distinct_identities": len(identity_keys),
        },
        "unresolved_reason": unresolved_reason,
    }


def _unresolved_reason(sources, source, identity_paths, pdf_path, patent_keys):
    if source not in {"fda", "cn"}:
        return "unknown_source"
    if source == "fda" and not sources.fda_products:
        return "source_not_loaded"
    if source == "cn" and not sources.cn_drug_rows:
        return "source_not_loaded"
    identity_found = any(os.path.isfile(path) for path in identity_paths)
    if not identity_found and not pdf_path and not patent_keys:
        return "no_source_identity"
    if not pdf_path and not patent_keys:
        return "no_locator"
    return "no_metadata_match"


def _fda_folder_drug_ids(sources, folder):
    drug_ids = sources.fda_by_ingredient.get(_key(folder))
    if drug_ids:
        return drug_ids
    tokens = frozenset(_key(part) for part in _folder_tokens(folder) if _key(part))
    if len(tokens) > 1:
        return sources.fda_by_tokens.get(tokens, ())
    return ()


def _folder_tokens(folder):
    text = _text(folder) or ""
    return [part for part in text.replace("_", ";").split(";") if part.strip()]


def _fda_name_drug_ids(sources, folder):
    base, _salt = _strip_salt_en(folder)
    if base:
        return sources.fda_by_base.get(_key(base), ())
    return ()


def document_drugs(doc_index):
    """Yield T3 ``doc_drug_candidate`` rows in deterministic order."""
    for document_id in sorted(doc_index):
        row = doc_index[document_id]
        yield {
            "document_id": row["document_id"],
            "uuid": row["uuid"],
            "source": row["source"],
            "pdf_path": row["pdf_path"],
            "patent_keys": row["patent_keys"],
            "drug_ids": row["drug_ids"],
            "primary_drug_id": row["primary_drug_id"],
            "match_channels": row["match_channels"],
            "ambiguity": row["ambiguity"],
            "match_basis": row["match_basis"],
            "unresolved_reason": row["unresolved_reason"],
        }


# ---------------------------------------------------------------------------
# T2 drug -> patent rows
# ---------------------------------------------------------------------------

def drug_patents(sources, doc_index):
    """Yield T2 ``drug_patent`` rows for the drugs referenced by ``doc_index``."""
    drug_ids = sorted({drug_id for row in doc_index.values() for drug_id in row["drug_ids"]})
    seen = set()
    for drug_id in drug_ids:
        if drug_id.startswith("FDA:"):
            product = sources.fda_drug_rows.get(drug_id)
            if not product:
                continue
            patents = [patent for patent in product.get("patents") or [] if isinstance(patent, dict)]
            patents.sort(key=lambda patent: (_fda_patent_key(patent.get("patent_number")) or "",
                                             _text(patent.get("pdf_local_path")) or ""))
            for patent in patents:
                patent_key = _fda_patent_key(patent.get("patent_number"))
                if not patent_key:
                    continue
                dedupe = (drug_id, patent_key, "patent")
                if dedupe in seen:
                    continue
                seen.add(dedupe)
                yield {
                    "drug_id": drug_id,
                    "source": "fda",
                    "patent_key": patent_key,
                    "patent_key_type": "patent",
                    "patent_number_raw": _text(patent.get("patent_number")),
                    "publication_number": None,
                    "patent_title": None,
                    "patentee": None,
                    "grant_date": None,
                    "expiry_date": _text(patent.get("expiry_date")),
                    "pdf_status": _text(patent.get("pdf_status")),
                    "pdf_path": _pdf_key(patent.get("pdf_local_path")),
                    "claims": None,
                    "mapping_status": None,
                }
        elif drug_id.startswith("CN:"):
            registration_id = drug_id[3:]
            rows = sorted(sources.cn_rows_by_registration.get(registration_id, ()),
                          key=_cn_row_sort_key)
            for row in rows:
                kind, patent_key = _cn_row_key(row)
                if not patent_key:
                    continue
                dedupe = (drug_id, patent_key, kind)
                if dedupe in seen:
                    continue
                seen.add(dedupe)
                yield {
                    "drug_id": drug_id,
                    "source": "cn",
                    "patent_key": patent_key,
                    "patent_key_type": kind,
                    "patent_number_raw": _text(row.get("patent_number_raw")),
                    "publication_number": _text(row.get("publication_number")),
                    "patent_title": _text(row.get("patent_title")),
                    "patentee": _text(row.get("patentee")),
                    "grant_date": _text(row.get("grant_date")),
                    "expiry_date": None,
                    "pdf_status": _text(row.get("pdf_status")),
                    "pdf_path": _cn_row_pdf_path(row),
                    "claims": row.get("claims") or None,
                    "mapping_status": _text(row.get("mapping_status")),
                }


def _cn_row_sort_key(row):
    kind, key = _cn_row_key(row)
    return (key or "", kind or "", _text(row.get("publication_number")) or "",
            _text(row.get("patent_record_id")) or "")


def _cn_row_pdf_path(row):
    attachment_paths = []
    for attachment in row.get("attachments") or []:
        if isinstance(attachment, dict):
            path = _pdf_key(attachment.get("pdf_local_path"))
            if path:
                attachment_paths.append(path)
    if attachment_paths:
        return sorted(attachment_paths)[0]
    path = _pdf_path_only(row.get("pdf_local_path"))
    return path


def _pdf_path_only(value):
    path = _pdf_key(value)
    if path and path.lower().endswith(".pdf"):
        return path
    return None


# ---------------------------------------------------------------------------
# Route-target relation tables (T4/T5) live in
# ``route_repair._route_drug_rows``: it consumes this module's document index
# (drug candidates plus evidence channels), joins each route target by InChIKey
# (isomeric; skeleton for parent forms) and normalised name (whole-token
# containment with salt/dosage forms handled), then writes
# ``drugs/route_drug.jsonl`` with relation type 药物 / 相关形式 / 剂型 / 前体 /
# 其他 / 未确认, evidence and unresolved codes.  Keep this module write-free.
# ---------------------------------------------------------------------------
