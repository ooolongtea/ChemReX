"""Standardizer — unify extraction NER+Action into structured JSON.

- compound fields (REACTION_PRODUCT/STARTING_MATERIAL/REAGENT_CATALYST/SOLVENT): list[str] → list[{name, standard_name, smiles}]
- scalar fields (TEMPERATURE/TIME/YIELD_PERCENT): str | null
- action: ordered list (unchanged order)
- null/empty/n-a normalized to null for scalars, [] for lists
"""

from typing import Any, Dict, List
import re

from loguru import logger

from .compound_standardizer import CompoundStandardizer
from pipeline.chemical_quality import split_isomer_label
from .names import string_names
from copy import deepcopy

COMPOUND_FIELDS = ("REACTION_PRODUCT", "STARTING_MATERIAL", "REAGENT_CATALYST", "SOLVENT")
SCALAR_FIELDS = ("TEMPERATURE", "TIME", "YIELD_PERCENT")


class Standardizer:
    """Produce unified structured JSON from extraction output."""

    def __init__(self, use_opsin: bool = True):
        try:
            import config
            use_opsin = getattr(config, "STANDARDIZE_USE_OPSIN", use_opsin)
        except ImportError:
            pass
        self.compound_std = CompoundStandardizer(use_opsin=use_opsin)
        logger.debug("Standardizer ready")

    def standardize(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        ner = deepcopy(extraction_result.get("ner", {}) or {})
        for field in COMPOUND_FIELDS:
            ner[field] = self._compound_values(ner.get(field))
        action = list(extraction_result.get("action", []) or [])
        language = extraction_result.get("language")
        task = extraction_result.get("task", "both")

        compound_names = self._collect_compound_names(ner)
        prior_naming = {c['name']: c for field in COMPOUND_FIELDS for c in (ner.get(field) or [])
                        if isinstance(c,dict) and c.get('requires_structure_resolution')}
        if compound_names:
            extra = {"language": language} if language else {}
            std_map = (self.compound_std.standardize(compound_names, prior_naming=prior_naming, **extra)
                       if prior_naming else
                       self.compound_std.standardize(compound_names, **extra))
        else:
            std_map = {}

        out_ner: Dict[str, Any] = {}
        for field in COMPOUND_FIELDS:
            raw = ner.get(field, [])
            if isinstance(raw, str):
                raw = [raw] if raw.strip() else []
            out_ner[field] = []
            for n in raw or []:
                if isinstance(n,dict) and n.get('requires_structure_resolution'):
                    enriched = self._enrich_compound(n['name'],std_map)
                    enriched['normalization_reuse'] = deepcopy(n.get('normalization_reuse'))
                else:
                    enriched = deepcopy(n) if isinstance(n,dict) else self._enrich_compound(n,std_map)
                out_ner[field].append(enriched)
        for field in SCALAR_FIELDS:
            val = ner.get(field)
            out_ner[field] = self._norm_scalar(val)

        result = {
            "ner": out_ner,
            "action": action,
            "language": language,
            "task": task,
        }
        for key, value in extraction_result.items():
            if key not in result and key not in ("ner_records", "dict_result"):
                result[key] = deepcopy(value)
        logger.info(
            f"Standardize done | compounds enriched={sum(1 for v in std_map.values() if v.get('smiles'))} "
            f"actions={len(action)}"
        )
        return result

    @staticmethod
    def _compound_values(value):
        if isinstance(value, (list, tuple)):
            return [item for part in value for item in Standardizer._compound_values(part)]
        if isinstance(value, str):
            return string_names(value)
        if not isinstance(value, dict):
            return []
        names = string_names(value.get('name'))
        result = []
        for name in names:
            item = deepcopy(value)
            item['name'] = name
            if len(names) > 1:
                # One structure cannot be assigned to several model names.
                for key in ('smiles', 'canonical_smiles', 'cxsmiles', 'standard_name'):
                    item.pop(key, None)
                item['requires_structure_resolution'] = True
            for key in ('smiles', 'canonical_smiles', 'cxsmiles', 'standard_name'):
                if item.get(key) is not None and not isinstance(item[key], str):
                    item['rejected_' + key] = item.pop(key)
                    item['structure_rejection'] = 'invalid_structure_value_type'
            result.append(item)
        return result

    @staticmethod
    def _collect_compound_names(ner: Dict[str, Any]) -> List[str]:
        names: List[str] = []
        for field in COMPOUND_FIELDS:
            raw = Standardizer._compound_values(ner.get(field))
            for n in (raw or []):
                if isinstance(n, str) and n.strip():
                    names.append(n.strip())
                elif isinstance(n,dict) and n.get('requires_structure_resolution') and n.get('name'):
                    names.append(n['name'])
        return list(dict.fromkeys(names))

    @staticmethod
    def _enrich_compound(name: str, std_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        info = std_map.get(name, {})
        standard_name = info.get("standard_name")
        label = split_isomer_label(name)[1] or split_isomer_label(standard_name)[1]
        stereo = info.get('stereochemistry')
        if label:
            # Also annotate legacy normalization candidates reused by name.
            stereo = dict(kind='unassigned_isomer', descriptors=[label], structure_coarsened=True)
        if str(standard_name or "").lower() in ("5-aminosalicylic acid", "mesalazine"):
            standard_name = "mesalamine"
        identifier_only = str(standard_name or "").upper() == "ID"
        lowered = name.lower()
        if re.search(r"polymer|聚合物|carbopol|witepsol|polysorbate|cellulose|"
                     r"CMC[-\s]*\(?MV\)?", lowered, re.I):
            compound_class = "polymer"
        elif re.search(r"mixture|composition|tablets?|solution|混合物|组合物|片剂|溶液|或", name, re.I):
            compound_class = "mixture"
        elif re.search(r"activated carbon|活性炭|钯碳|palladium on carbon", name, re.I):
            compound_class = "excipient"
        else:
            compound_class = "discrete"
        return {
            "name": name,
            "standard_name": None if identifier_only else standard_name,
            "identifier_only": identifier_only,
            "compound_class": compound_class,
            "auxiliary": compound_class == "excipient",
            "structure_source": info.get("structure_source", "name_resolution"),
            "cxsmiles": info.get("cxsmiles"),
            "naming_raw": info.get("naming_raw"),
            "resolution_trace": info.get("resolution_trace"),
            "resolution_status": info.get("resolution_status"),
            "stereochemistry": stereo,
            # Mixtures and polymers must not carry a single-molecule SMILES:
            # resolving "乙腈或乙腈-水" to plain acetonitrile (v7 P-07) reads
            # as a structure claim for what is actually an option list.
            "smiles": None if compound_class in ("excipient", "mixture", "polymer")
                      else info.get("smiles"),
        }

    @staticmethod
    def _norm_scalar(val: Any) -> Any:
        if val is None:
            return None
        s = str(val).strip()
        if not s or s.lower() in ("null", "none", "n/a", "not specified", "notspecified",
                                      "未指定", "未知", "—", "-"):
            return None
        return s
