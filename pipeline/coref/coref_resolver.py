"""Coreference resolution: standardize compounds, then map identifiers to image-side SMILES.

Scope (per user decision): **ID→SMILES mapping only**. No language-layer coreference
("该化合物 / 所得固体" mentions are NOT resolved here).

Input contract:
    {"ner": {REACTION_PRODUCT, STARTING_MATERIAL, REAGENT_CATALYST, SOLVENT, ...}, "dict_result": {id_key: smiles}}

Compound fields may be either:
  - standardized: ``list[{name, standard_name, smiles}]`` — identifiers have ``standard_name == "ID"``;
  - raw extraction: ``list[str]`` — standardized internally before mapping.

Reuses ``prompts/coref_resolution/v1.md`` (LLM semantic core-ID matching) + ``pipeline/gpt_api.py``.
"""

import json
import re
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from loguru import logger

try:
    import config
except ImportError:
    from .. import config

from pipeline.gpt_api import GPTAPI
from pipeline.prompts import PromptLoader
from pipeline.standardize.standardizer import Standardizer

# NER compound fields that may contain compound identifiers.
_COMPOUND_FIELDS = ("STARTING_MATERIAL", "REACTION_PRODUCT", "REAGENT_CATALYST", "SOLVENT")

# Heuristic identifier detection (used only for raw list[str] compounds or dicts
# without a standard_name label). Conservative: relies on the LLM prompt's strict
# "no hallucination / core-ID match" rule to reject false positives.
_ID_KEYWORD_RE = re.compile(
    r"(化合物|式|target|compound|产物|原料|中间体|intermediate|product|starting\s+material)",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(r"^[\(\（]?(\d+|[IVXLCDM]+|[①-⑳]|[A-Za-z])[\)\）]?$")


def _looks_like_identifier(name: str) -> bool:
    s = (name or "").strip()
    if not s:
        return False
    if _ID_KEYWORD_RE.search(s):
        return True
    if _LABEL_RE.match(s):
        return True
    return False


def _is_identifier_compound(compound: Any) -> bool:
    """A compound is an identifier if labelled 'ID' (standardized) or matches the heuristic."""
    if isinstance(compound, dict):
        if compound.get("identifier_only"):
            return True
        std = (compound.get("standard_name") or "").strip()
        if std:
            return std.upper() == "ID"
        return _looks_like_identifier(str(compound.get("name") or ""))
    return _looks_like_identifier(str(compound))


def _compound_name(compound: Any) -> str:
    if isinstance(compound, dict):
        return str(compound.get("name") or "").strip()
    return str(compound or "").strip()


class CorefResolver:
    """Resolve compound identifiers in NER fields to SMILES from the image dictionary."""

    def __init__(self, model: str = None, temperature: float = None, auto_standardize: bool = None,
                 standardizer=None):
        loader = PromptLoader()
        prompt_data = loader.load_prompt("coref_resolution")
        self.loader = loader
        self.prompt_name = "coref_resolution"
        meta = prompt_data["metadata"]
        model = model or getattr(config, "COREF_MODEL", None) or meta.get("model", "gpt-4o")
        temperature = (
            temperature
            if temperature is not None
            else getattr(config, "COREF_TEMPERATURE", meta.get("temperature", 0.0))
        )
        self.auto_standardize = (
            auto_standardize
            if auto_standardize is not None
            else getattr(config, "COREF_AUTO_STANDARDIZE", True)
        )
        self._standardizer = standardizer
        self.api = GPTAPI(model=model, temperature=temperature,
                          provider=os.getenv("COREF_PROVIDER") or None)
        logger.debug(f"CorefResolver ready: model={model}, auto_standardize={self.auto_standardize}")

    def resolve(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize a NER record if needed, then resolve ID→SMILES.

        Returns:
            {
              "id_smiles_mapping": {text_id: smiles},
              "unresolved_identifiers": [text_id, ...],
              "ner": <ner with dict-compound smiles filled>,
              "summary": {total_identifiers, resolved, unresolved, dict_entries},
            }
        """
        record, standardize_status = self._ensure_standardized_record(payload)
        ner = deepcopy(record.get("ner", {}) or {})
        dict_result = payload.get("dict_result", {}) or {}

        if not dict_result:
            logger.warning("coref skipped: empty dict_result")
            return self._empty_result(record, dict_result, standardize_status)

        identifiers = self._collect_identifiers(ner)
        if not identifiers:
            logger.info("coref skipped: no compound identifiers found in NER")
            return self._empty_result(record, dict_result, standardize_status)

        logger.info(
            f"Coref resolution: {len(identifiers)} identifiers, {len(dict_result)} dict entries"
        )

        mapping = self._build_llm_mapping(identifiers, dict_result)

        resolved_count, unresolved = self._apply_mapping(ner, mapping, dict_result)
        id_smiles_mapping = {
            tid: dict_result[dk]
            for tid, dk in mapping.items()
            if dk in dict_result and dict_result[dk]
        }

        logger.info(
            f"Coref done: resolved={resolved_count}, unresolved={len(unresolved)}"
        )
        result = self._base_result(record, dict_result, standardize_status)
        result.update({
            "id_smiles_mapping": id_smiles_mapping,
            "unresolved_identifiers": unresolved,
            "ner": ner,
            "summary": {
                "total_identifiers": len(identifiers),
                "resolved": resolved_count,
                "unresolved": len(unresolved),
                "dict_entries": len(dict_result),
            },
        })
        return result

    # ── helpers ──
    @property
    def standardizer(self) -> Standardizer:
        if self._standardizer is None:
            self._standardizer = Standardizer()
        return self._standardizer

    def _ensure_standardized_record(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        record = {
            key: deepcopy(value)
            for key, value in (payload or {}).items()
            if key != "dict_result"
        }
        ner = record.get("ner", {}) or {}
        if self._is_standardized_ner(ner):
            return record, "already_standardized"
        if not self.auto_standardize:
            return record, "disabled"
        try:
            standardized = self.standardizer.standardize(record)
            for key in record:
                if key in record and key not in standardized:
                    standardized[key] = record[key]
            return standardized, "applied"
        except Exception as exc:
            logger.error(f"coref standardize failed; falling back to raw NER: {exc}")
            record["standardize_error"] = f"{type(exc).__name__}: {exc}"
            return record, "failed"

    @staticmethod
    def _is_standardized_ner(ner: Dict[str, Any]) -> bool:
        for field in _COMPOUND_FIELDS:
            raw = ner.get(field, []) or []
            if isinstance(raw, str):
                return False
            if not isinstance(raw, list):
                raw = [raw]
            for comp in raw:
                if not comp:
                    continue
                if not isinstance(comp, dict):
                    return False
                if "standard_name" not in comp or "smiles" not in comp:
                    return False
        return True

    def _collect_identifiers(self, ner: Dict[str, Any]) -> List[str]:
        ids: List[str] = []
        seen = set()
        for field in _COMPOUND_FIELDS:
            raw = ner.get(field, []) or []
            if isinstance(raw, str):
                raw = [raw] if raw.strip() else []
            for comp in (raw if isinstance(raw, list) else [raw]):
                if _is_identifier_compound(comp):
                    name = _compound_name(comp)
                    if name and name not in seen:
                        seen.add(name)
                        ids.append(name)
        return ids

    def _build_llm_mapping(
        self, text_identifiers: List[str], dict_result: Dict[str, str]
    ) -> Dict[str, str]:
        text_ids = sorted(text_identifiers)
        dict_keys = sorted(str(k) for k in dict_result.keys())
        prompt = self.loader.format_prompt(
            self.prompt_name,
            text_identifiers=json.dumps(text_ids, ensure_ascii=False),
            dict_keys=json.dumps(dict_keys, ensure_ascii=False),
        )
        # A transport-level failure (timeout / 429 / network) must propagate:
        # swallowing it here would return an empty mapping, the chunk would
        # be recorded ok and the whole ID→SMILES mapping silently lost
        # (doc/batch-code-review-round2.md H4).
        response = self.api.chat(prompt)
        start = response.find("{")
        end = response.rfind("}") + 1
        if 0 <= start < end:
            try:
                data = json.loads(response[start:end])
            except json.JSONDecodeError:
                logger.warning("coref LLM JSON decode failed; empty mapping")
                return {}
            mapping = data.get("mapping", {}) or {}
            if not isinstance(mapping, dict):
                logger.warning('coref LLM mapping is not an object; leaving identifiers unresolved')
                return {}
            logger.debug(f"LLM mapping: {len(mapping)} pairs")
            return {
                str(k): str(v)
                for k, v in mapping.items()
                # A model sometimes returns a list of alternatives. Never
                # use it as a hash key or choose one of those alternatives.
                if k in text_identifiers and isinstance(v, (str, int))
                and not isinstance(v, bool) and str(v) in dict_result
            }
        logger.warning("coref LLM returned no parseable JSON; falling back to empty mapping")
        return {}

    def _apply_mapping(
        self, ner: Dict[str, Any], mapping: Dict[str, str], dict_result: Dict[str, str]
    ) -> Tuple[int, List[str]]:
        resolved = 0
        unresolved: List[str] = []
        for field in _COMPOUND_FIELDS:
            raw = ner.get(field, []) or []
            if not isinstance(raw, list):
                raw = [raw] if raw else []
            for comp in raw:
                if not _is_identifier_compound(comp):
                    continue
                name = _compound_name(comp)
                if not name:
                    continue
                dict_key = mapping.get(name)
                smiles = dict_result.get(dict_key) if dict_key else None
                if smiles:
                    if isinstance(comp, dict):
                        comp["smiles"] = smiles
                        comp["dict_key"] = dict_key
                        comp["structure_source"] = "image_identifier"
                    resolved += 1
                else:
                    unresolved.append(name)
        return resolved, unresolved

    @staticmethod
    def _base_result(
        record: Dict[str, Any],
        dict_result: Dict[str, Any],
        standardize_status: str,
    ) -> Dict[str, Any]:
        result = {
            "ner": record.get("ner", {}) or {},
            "standardize_status": standardize_status,
        }
        for key, value in record.items():
            if key not in result and key != "dict_result":
                result[key] = deepcopy(value)
        return result

    @classmethod
    def _empty_result(
        cls,
        record: Dict[str, Any],
        dict_result: Dict[str, Any],
        standardize_status: str,
    ) -> Dict[str, Any]:
        result = cls._base_result(record, dict_result, standardize_status)
        result.update({
            "id_smiles_mapping": {},
            "unresolved_identifiers": [],
            "summary": {
                "total_identifiers": 0,
                "resolved": 0,
                "unresolved": 0,
                "dict_entries": len(dict_result or {}),
            },
        })
        return result
