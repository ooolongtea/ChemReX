"""Compound name → standard_name → SMILES cascade (migrated from old SMILESEnricher)."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import json
import os
import re
from typing import Callable, Dict, List, Optional, TypeVar

from loguru import logger

try:
    import cirpy as cir
except Exception:
    cir = None

try:
    import pubchempy as pcp
except Exception:
    pcp = None

try:
    from pyopsin import PyOpsin
except Exception:
    PyOpsin = None

try:
    from rdkit import Chem
except Exception:
    Chem = None

from pipeline.gpt_api import GPTAPI
from pipeline.prompts import PromptLoader


T = TypeVar("T")

_CAS_PATTERN = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
_CAS_REMOVE_PATTERN = re.compile(r"\(?\b\d{2,7}-\d{2}-\d\b\)?")
_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
_INVALID_NAMES = {"", "ID", "UNKNOWN", "none", "None", "NONE", "Not Found"}


def _normalize_smiles(smiles: Optional[str]) -> Optional[str]:
    if not smiles or not smiles.strip():
        return smiles
    if Chem is None:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smiles


class CompoundStandardizer:
    """Standardize compound names and convert to SMILES."""

    def __init__(self, use_opsin: bool = True):
        self.use_opsin = use_opsin and PyOpsin is not None
        offline = os.getenv("STANDARDIZE_OFFLINE", "").strip().lower() in ("1", "true", "yes")
        self.use_cir = cir is not None and not offline
        self.use_pubchem = pcp is not None and not offline
        self.network_timeout = self._load_network_timeout()
        loader = PromptLoader()
        prompt_data = loader.load_prompt("compound_standardization")
        self.standardize_prompt = prompt_data["content"]
        meta = prompt_data["metadata"]
        model = meta.get("model")
        if model:
            try:
                import config
                model = getattr(config, "STANDARDIZE_MODEL", model)
            except ImportError:
                pass
        self.api = GPTAPI(model=model, temperature=meta.get("temperature", 0.0))
        self.opsin = PyOpsin() if self.use_opsin else None
        logger.debug(
            "CompoundStandardizer ready "
            f"(use_cir={self.use_cir}, use_pubchem={self.use_pubchem}, "
            f"use_opsin={self.use_opsin}, network_timeout={self.network_timeout}s, "
            f"offline={not self.use_cir and not self.use_pubchem})"
        )

    def standardize(self, names: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
        """Return {original_name: {standard_name, smiles}}."""
        if not names:
            return {}
        unique = list(dict.fromkeys(names))
        std_info = self._standardize_names(unique)
        name_to_smiles = self._name_to_smiles_batch(unique, std_info)
        return {
            n: {
                "standard_name": std_info.get(n, {}).get("standard_name"),
                "smiles": name_to_smiles.get(n),
            }
            for n in unique
        }

    def _standardize_names(self, names: List[str]) -> Dict[str, Dict[str, str]]:
        if not names:
            return {}
        std_info: Dict[str, Dict[str, str]] = {}
        for i in range(0, len(names), 50):
            batch = names[i: i + 50]
            try:
                prompt = self.standardize_prompt + "\n" + json.dumps(batch, ensure_ascii=False)
                response = self.api.chat(prompt)
                parsed = self._parse_text_response(response)
                std_info.update(parsed)
            except Exception as e:
                logger.error(f"Standardize batch {i} failed: {e}")
            for n in batch:
                if n not in std_info:
                    std_info[n] = {"standard_name": "UNKNOWN"}
        return std_info

    @staticmethod
    def _parse_text_response(response: str) -> Dict[str, Dict[str, str]]:
        mapping: Dict[str, Dict[str, str]] = {}
        if not response:
            return mapping
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or len(line) > 300 or line.startswith("#") or line.startswith("//"):
                continue
            if "|" in line:
                parts = line.split("|", 1)
                if len(parts) >= 2:
                    mapping[parts[0].strip()] = {"standard_name": parts[1].strip()}
        return mapping

    def _name_to_smiles_batch(
        self,
        names: List[str],
        std_info: Dict[str, Dict[str, str]],
    ) -> Dict[str, Optional[str]]:
        name_to_smiles: Dict[str, Optional[str]] = {}
        for orig_name in names:
            std_name = std_info.get(orig_name, {}).get("standard_name") or "UNKNOWN"
            smiles = self._resolve_smiles(orig_name, std_name)
            if smiles:
                name_to_smiles[orig_name] = _normalize_smiles(smiles) or smiles
        return name_to_smiles

    def _resolve_smiles(self, orig_name: str, std_name: str) -> Optional[str]:
        tried_cir: set[str] = set()
        tried_pubchem: set[str] = set()
        tried_opsin: set[str] = set()

        def cir_lookup(query: str) -> Optional[str]:
            if query in tried_cir:
                return None
            tried_cir.add(query)
            return self._cir_convert(query)

        def pubchem_lookup(query: str) -> Optional[str]:
            if query in tried_pubchem:
                return None
            tried_pubchem.add(query)
            return self._pubchem_convert(query)

        def opsin_lookup(query: str) -> Optional[str]:
            if query in tried_opsin:
                return None
            tried_opsin.add(query)
            return self._opsin_convert(query)

        cas = self._extract_cas(orig_name)
        if cas:
            smiles = cir_lookup(cas) or pubchem_lookup(cas)
            if smiles:
                return smiles

        if (std_name or "").strip().upper() in _INVALID_NAMES:
            logger.debug(f"skip network cascade for [{orig_name}]: standard_name={std_name!r}")
            return None

        raw_candidates = self._raw_search_candidates(orig_name)
        for candidate in raw_candidates:
            smiles = cir_lookup(candidate) or pubchem_lookup(candidate)
            if smiles:
                return smiles

        for candidate in self._dedupe([std_name, *raw_candidates, orig_name]):
            smiles = opsin_lookup(candidate)
            if smiles:
                return smiles

        for candidate in self._dedupe([std_name]):
            smiles = pubchem_lookup(candidate) or cir_lookup(candidate)
            if smiles:
                return smiles

        return None

    def _cir_convert(self, name: str) -> Optional[str]:
        if not self.use_cir or not self._is_queryable_name(name):
            return None
        try:
            result = self._call_with_timeout(
                lambda: cir.resolve(name, "smiles"),
                self.network_timeout,
            )
            if result:
                return str(result)
        except Exception as e:
            logger.debug(f"CIR failed [{name}]: {e}")
        return None

    def _pubchem_convert(self, name: str) -> Optional[str]:
        if not self.use_pubchem or not self._is_queryable_name(name):
            return None
        try:
            compounds = self._call_with_timeout(
                lambda: pcp.get_compounds(name, "name"),
                self.network_timeout,
            )
            if compounds:
                smiles = getattr(compounds[0], "isomeric_smiles", None)
                if smiles:
                    return smiles
        except Exception as e:
            logger.debug(f"PubChem failed [{name}]: {e}")
        return None

    def _opsin_convert(self, name: str) -> Optional[str]:
        if not self.use_opsin or not self.opsin or not self._is_queryable_name(name):
            return None
        try:
            result = self.opsin.to_smiles([name], num_workers=1)
            if result and result[0] is not None:
                return result[0]
        except Exception as e:
            logger.debug(f"OPSIN failed [{name}]: {e}")
        return None

    @staticmethod
    def _call_with_timeout(func: Callable[[], T], timeout: int) -> Optional[T]:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            future.cancel()
            return None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _load_network_timeout() -> int:
        try:
            import config
            return int(
                getattr(
                    config,
                    "STANDARDIZE_NETWORK_TIMEOUT",
                    os.getenv("STANDARDIZE_NETWORK_TIMEOUT", "8"),
                )
            )
        except Exception:
            return int(os.getenv("STANDARDIZE_NETWORK_TIMEOUT", "8"))

    @classmethod
    def _raw_search_candidates(cls, name: str) -> List[str]:
        if not name:
            return []
        candidates = []
        cleaned = cls._remove_cas(name).replace("\n", "").replace("\r", "").strip()
        if ":" in cleaned:
            candidates.append(cleaned.split(":", 1)[1].split(";", 1)[0].strip())
        elif ",\n" in name:
            candidates.append(cls._remove_cas(name.split(",\n", 1)[1]).split(" (", 1)[0].strip())
        else:
            candidates.append(cleaned.split(" (", 1)[0].strip())
        candidates.append(cleaned)
        return [c for c in cls._dedupe(candidates) if cls._is_queryable_name(c)]

    @staticmethod
    def _remove_cas(name: str) -> str:
        return _CAS_REMOVE_PATTERN.sub("", name).strip(" ;,()")

    @classmethod
    def _extract_cas(cls, name: str) -> Optional[str]:
        match = _CAS_PATTERN.search(name or "")
        if not match:
            return None
        cas = match.group(0)
        return cas if cls._is_valid_cas(cas) else None

    @staticmethod
    def _is_valid_cas(cas: str) -> bool:
        digits = cas.replace("-", "")
        check = int(digits[-1])
        total = sum(int(digit) * (idx + 1) for idx, digit in enumerate(reversed(digits[:-1])))
        return total % 10 == check

    @staticmethod
    def _is_queryable_name(name: Optional[str]) -> bool:
        if not name:
            return False
        stripped = name.strip()
        if stripped in _INVALID_NAMES or _CJK_PATTERN.search(stripped):
            return False
        return any(ch.isalnum() for ch in stripped)

    @staticmethod
    def _dedupe(values: List[Optional[str]]) -> List[str]:
        result = []
        seen = set()
        for value in values:
            if not value:
                continue
            normalized = value.strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                result.append(normalized)
        return result
