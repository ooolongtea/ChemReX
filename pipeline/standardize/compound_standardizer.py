"""Compound name → standard_name → SMILES cascade (migrated from old SMILESEnricher)."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import hashlib
import json
import os
import re
import ssl
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

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
from pipeline.chemical_quality import split_isomer_label
from .naming_cache import NamingCache
from .names import string_names


T = TypeVar("T")
_resolution_state = threading.local()
# Transient lookup failures must be retried, not frozen into the negative cache.
_TRANSIENT_RESOLUTION_STATUSES = {"rate_limited", "timeout", "service_failure", "response_mismatch"}


def _retryable_cached_result(value):
    if value.get('smiles'):
        return False
    # Before response_mismatch existed, an absent response line became UNKNOWN
    # without naming_raw. Explicit model negatives have a response line and stay
    # cacheable; only the old synthetic negatives need another naming attempt.
    return (value.get('resolution_status') in _TRANSIENT_RESOLUTION_STATUSES or
            (value.get('resolution_status') == 'unusable_standard_name' and
             value.get('standard_name') == 'UNKNOWN' and not value.get('naming_raw')))


def _lookup_error(exc):
    code = getattr(exc, 'code', None) or getattr(exc, 'status_code', None)
    if code is None:
        match = re.search(r'\b(408|429|503|404|500|502|504)\b', str(exc))
        code = int(match[1]) if match else None
    status = ('rate_limited' if code == 429 or 'Too many requests' in str(exc) else
              'no_result' if code == 404 else 'timeout' if isinstance(exc, TimeoutError) or
              isinstance(getattr(exc, 'reason', None), TimeoutError) or code in (408,504) else
              'service_failure')
    return {'status': status, 'error_type': type(exc).__name__, 'http_status': code}


def _transient_lookup(error):
    return bool(error and (error.get('status') in ('timeout', 'rate_limited') or
                error.get('http_status') in (500,502,503,504) or error.get('error_type') in
                ('ConnectionError','ConnectionResetError','RemoteDisconnected',
                 'URLError','SSLError','SSLEOFError')))

_CAS_PATTERN = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
_CAS_REMOVE_PATTERN = re.compile(
    r"\(\s*(?:CAS(?:\s*(?:RN|No\.?|number))?\s*[:#]?\s*)?\d{2,7}-\d{2}-\d\s*\)|"
    r"\b(?:CAS(?:\s*(?:RN|No\.?|number))?\s*[:#]?\s*)?\d{2,7}-\d{2}-\d\b", re.I)
_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
_INVALID_NAMES = {"", "ID", "UNKNOWN", "NONE", "NOT FOUND"}

# Offline, high-confidence resolutions for names that network dictionaries
# get WRONG. CIR resolves the bare acronym "DCM" to deoxycytidine
# monophosphate and "CMC" to a carbodiimide tosylate (v7 review P-04), so
# ambiguous acronyms are deliberately absent here — those must resolve via
# their LLM-expanded standard_name or not at all.
_AUXILIARY_SMILES = {
    # solvents
    "dcm": "ClCCl", "ch2cl2": "ClCCl", "dichloromethane": "ClCCl",
    "dmf": "CN(C)C=O", "dimethylformamide": "CN(C)C=O",
    "dmso": "CS(C)=O", "dimethyl sulfoxide": "CS(C)=O",
    "dma": "CN(C)C(C)=O", "dmac": "CN(C)C(C)=O",
    "nmp": "CN1CCCCC1=O",
    "thf": "C1CCOC1", "tetrahydrofuran": "C1CCOC1",
    "2-methf": "CC1CCOC1",
    "mecn": "CC#N", "acn": "CC#N", "acetonitrile": "CC#N",
    "etoac": "CCOC(C)=O", "ethyl acetate": "CCOC(C)=O",
    "meoh": "CO", "methanol": "CO",
    "etoh": "CCO", "ethanol": "CCO",
    "ipa": "CC(O)C", "isopropanol": "CC(O)C",
    "nbuoh": "CCCCO", "n-buoh": "CCCCO",
    "tbuoh": "CC(C)(C)O",
    "mtbe": "COC(C)(C)C",
    "water": "O", "h2o": "O",
    # gases and auxiliary atmospheres
    "nitrogen": "N#N", "n2": "N#N", "氮气": "N#N",
    "oxygen": "O=O", "o2": "O=O", "氧气": "O=O",
    "hydrogen": "[H][H]", "h2": "[H][H]", "氢气": "[H][H]",
    "argon": "[Ar]", "氩气": "[Ar]",
    "carbon dioxide": "O=C=O", "co2": "O=C=O",
    # a few unambiguous common reagents
    "tea": "CCN(CC)CC", "triethylamine": "CCN(CC)CC",
    "dipea": "CCN(C(C)C)C(C)C",
    "tfa": "OC(=O)C(F)(F)F",
}
# A bare short uppercase token ("DCM", "MTBE") whose LLM standard_name
# disagrees must not be sent to CIR: CIR's synonym dictionary resolves
# acronyms arbitrarily. Querying the full standard_name instead is strictly
# safer (v7: "DCM"→dCMP while "dichloromethane"→ClCCl).
_BARE_ACRONYM = re.compile(r"[A-Z][A-Z0-9]{1,5}\Z")
# Relative-configuration descriptors: OPSIN resolves a wildcard stereocentre
# to one arbitrary enantiomer, so the SMILES must lose its stereo annotation
# to stay faithful to "(1S*,3R*)"-style names (v7 review P-10).
_RELATIVE_STEREO = re.compile(r"\d\s*[SR]\s*\\?\*|\brel-", re.I)


def _strip_stereo(smiles: Optional[str]) -> Optional[str]:
    if not smiles or Chem is None:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        return smiles


def _ensure_ca_bundle() -> None:
    """Fix urllib HTTPS verification on conda builds whose OpenSSL default CA
    file is a build-time placeholder path that does not exist — without a
    bundle every CIR call fails with "self-signed certificate in certificate
    chain". Point OpenSSL at certifi's bundle unless the env already sets one.
    """
    cafile = ssl.get_default_verify_paths().openssl_cafile
    if cafile and Path(cafile).is_file():
        return
    if os.getenv("SSL_CERT_FILE") or os.getenv("SSL_CERT_DIR"):
        return
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        logger.info("default OpenSSL CA file unusable ({}); SSL_CERT_FILE -> {}",
                    cafile, os.environ["SSL_CERT_FILE"])
    except ImportError:
        pass


_ensure_ca_bundle()

# These SDK versions call urllib.urlopen without a socket timeout. A future
# timeout alone leaves the worker alive and Python waits for it at shutdown.
# Restrict the transport override to the two SDKs and the current worker call.
_network_context = threading.local()


def _timed_urlopen(original):
    def request(*args, **kwargs):
        timeout = getattr(_network_context, 'timeout', None)
        if timeout is not None:
            kwargs.setdefault('timeout', timeout)
        return original(*args, **kwargs)
    return request


for _sdk in (cir, pcp):
    if _sdk is not None and hasattr(_sdk, 'urlopen'):
        _sdk.urlopen = _timed_urlopen(_sdk.urlopen)


def _network_call(func, timeout):
    previous = getattr(_network_context, 'timeout', None)
    _network_context.timeout = timeout
    try:
        return func()
    finally:
        _network_context.timeout = previous

# Bounded, shared pool for timeouts around CIR/PubChem calls. Timed-out calls
# keep running to completion (a thread cannot be interrupted), so a per-call
# executor would leak unboundedly; this caps the damage at max_workers. When
# every worker is stuck on a hung call, the pool is rebuilt and the old
# workers abandoned (they cannot be interrupted; see review round-2 M5).
_TIMEOUT_POOL_SIZE = 4
_TIMEOUT_EXECUTOR = ThreadPoolExecutor(max_workers=_TIMEOUT_POOL_SIZE,
                                       thread_name_prefix="standardize")
_executor_lock = threading.Lock()
_consecutive_timeouts = 0


def _rebuild_timeout_executor() -> None:
    """Replace the shared timeout pool once its workers are all stuck.

    ``future.cancel()`` cannot interrupt a call that is already running, so a
    few hung CIR/PubChem requests would otherwise occupy every worker and
    fail all later queries until their (possibly infinite) sockets return.
    The old executor is shut down without waiting; its stuck threads are
    abandoned."""
    global _TIMEOUT_EXECUTOR, _consecutive_timeouts
    with _executor_lock:
        old = _TIMEOUT_EXECUTOR
        _TIMEOUT_EXECUTOR = ThreadPoolExecutor(max_workers=_TIMEOUT_POOL_SIZE,
                                               thread_name_prefix="standardize")
        _consecutive_timeouts = 0
    old.shutdown(wait=False)
    logger.warning("standardize timeout pool saturated with hung network calls;"
                   " rebuilt ({} old worker thread(s) abandoned)", _TIMEOUT_POOL_SIZE)


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

    def __init__(self, use_opsin: bool = True, naming_client=None,
                 cache_path=None, use_cache: bool = None):
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
        self.api = naming_client if naming_client is not None else GPTAPI(
            model=model, temperature=meta.get("temperature", 0.0),
            provider=os.getenv("STANDARDIZE_PROVIDER") or None)
        self.opsin = None
        self._resolved_cache = {}
        self._cache_lock = threading.Lock()
        # Cache writes are shared across scheduler processes; the in-memory
        # cache remains the fast path for repeated names in one process.
        self._naming_cache = (
            NamingCache(path=cache_path, enabled=bool(use_cache))
            if cache_path is not None or use_cache is not None
            else NamingCache.from_env())
        self._cache_enabled = self._naming_cache.enabled
        self._cache_config_version = self._config_version(model, prompt_data)
        if self.use_opsin:
            try:
                import jpype
                if not jpype.isJVMStarted():
                    # -Xrs: an embedded JVM must not install its own
                    # SIGINT/SIGTERM handlers — they silently replace the
                    # scheduler's graceful-shutdown handler, so every stop
                    # request kills the process mid-task instead.
                    from pkg_resources import resource_filename
                    jpype.startJVM("-Xrs", classpath=[
                        resource_filename("pyopsin", "opsin_cli.jar")])
            except Exception as exc:
                logger.warning("could not pre-start JVM with -Xrs ({}); the JVM"
                               " may take over SIGINT handling", exc)
            self.opsin = PyOpsin()
        logger.debug(
            "CompoundStandardizer ready "
            f"(use_cir={self.use_cir}, use_pubchem={self.use_pubchem}, "
            f"use_opsin={self.use_opsin}, network_timeout={self.network_timeout}s, "
            f"offline={not self.use_cir and not self.use_pubchem})"
        )

    def _config_version(self, model, prompt_data) -> str:
        """Stable identity of everything that can change a cached naming result."""
        from pipeline.validation.quality import QUALITY_VERSION
        payload = {
            "model": getattr(self.api, "model", None) or model or "",
            "provider": getattr(self.api, "provider", "") or "",
            "prompt_sha256": hashlib.sha256(
                (self.standardize_prompt or "").encode("utf-8")).hexdigest(),
            "prompt_version": (prompt_data.get("metadata") or {}).get("version"),
            "quality_version": QUALITY_VERSION,
            "use_cir": bool(self.use_cir),
            "use_pubchem": bool(self.use_pubchem),
            "use_opsin": bool(self.use_opsin),
            "network_timeout": self.network_timeout,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:24]

    def standardize(self, names: List[str], prior_naming=None,
                    language: str = None) -> Dict[str, Dict[str, Optional[str]]]:
        """Return {original_name: {standard_name, smiles}}."""
        if not names:
            return {}
        unique = string_names(names)
        lang = language or ""
        from copy import deepcopy
        with self._cache_lock:
            cached = {n: deepcopy(self._resolved_cache[n]) for n in unique
                      if n in self._resolved_cache and not _retryable_cached_result(self._resolved_cache[n])}
        pending = [n for n in unique if n not in cached]
        naming_cache = getattr(self, "_naming_cache", None)
        cache_version = getattr(self, "_cache_config_version", None)
        cache_enabled = bool(getattr(self, "_cache_enabled", False) and naming_cache and cache_version)
        disk_hits = (naming_cache.get_many(pending, lang, cache_version)
                     if pending and cache_enabled else {})
        disk_hits = {n: v for n, v in disk_hits.items() if not _retryable_cached_result(v)}
        positive = {n: v for n, v in disk_hits.items() if v.get("smiles")}
        negative = {n: v for n, v in disk_hits.items() if not v.get("smiles")}
        pending = [n for n in pending if n not in positive]
        reusable = {}
        for name in pending:
            prior = (prior_naming or {}).get(name) or {}
            parsed = self._parse_text_response(prior.get('naming_raw') or '').get(name)
            if parsed and parsed['standard_name'] == prior.get('standard_name'):
                reusable[name] = parsed
        # A negative hit is final unless source-side prior naming can still
        # resolve the same name; that path is data, not a repeated guess.
        negative_final = {n: v for n, v in negative.items() if n not in reusable}
        pending = [n for n in pending if n not in negative_final]
        if pending:
            logger.debug("naming cache miss: {} name(s) (language={})", len(pending), lang or "-")
        missing_names = [n for n in pending if n not in reusable]
        std_info = self._standardize_names(missing_names) if missing_names else {}
        std_info.update(reusable)
        name_to_smiles = self._name_to_smiles_batch(pending, std_info)
        resolved = {
            n: {
                "standard_name": std_info.get(n, {}).get("standard_name"),
                "smiles": name_to_smiles.get(n),
                "cxsmiles": std_info.get(n, {}).get("cxsmiles"),
                "naming_raw": std_info.get(n, {}).get("naming_raw"),
                "resolution_trace": std_info.get(n, {}).get("resolution_trace", []),
                "resolution_status": std_info.get(n, {}).get("resolution_status", "unresolved"),
                "stereochemistry": std_info.get(n, {}).get("stereochemistry"),
            }
            for n in pending
        }
        if cache_enabled:
            cacheable = {
                n: v for n, v in resolved.items()
                if v.get("smiles")
                or v.get("resolution_status") not in _TRANSIENT_RESOLUTION_STATUSES}
            naming_cache.put_many(cacheable, lang, cache_version)
            with self._cache_lock:
                self._resolved_cache.update({
                    n: deepcopy(v) for n, v in {**positive, **negative_final, **cacheable}.items()})
        else:
            # CHEMREX_NAMING_CACHE=0 preserves the old positive-only behavior.
            with self._cache_lock:
                self._resolved_cache.update({
                    n: deepcopy(v) for n, v in resolved.items() if v.get("smiles")})
        if disk_hits:
            logger.debug("naming cache hit: {}/{} names (language={})",
                         len(disk_hits), len(unique), lang or "-")
        return {**cached, **positive, **negative_final, **resolved}

    def _standardize_names(self, names: List[str]) -> Dict[str, Dict[str, Any]]:
        if not names:
            return {}
        std_info: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(names), 50):
            batch = names[i: i + 50]
            pending = batch
            responses = []
            for attempt in range(2):
                try:
                    logger.debug("standardize LLM batch: {} name(s), attempt {}", len(pending), attempt + 1)
                    prompt = self.standardize_prompt + "\n" + json.dumps(pending, ensure_ascii=False)
                    response = self.api.chat(prompt)
                    responses.append(response)
                    parsed = self._parse_text_response(response)
                    # Exact original names only: fuzzy matching can merge salts
                    # or stereoisomers. Retry only missing/empty response lines.
                    std_info.update({n: parsed[n] for n in pending
                                     if n in parsed and parsed[n]['standard_name']})
                except Exception as e:
                    raise RuntimeError(f"standardize LLM batch {i} failed: {e}") from e
                pending = [n for n in pending if n not in std_info]
                if not pending:
                    break
            for n in pending:
                std_info[n] = {"standard_name": None, "resolution_status": "response_mismatch",
                               "resolution_trace": [dict(service='naming', query=n,
                                   status='response_mismatch', responses=responses)]}
        return std_info

    @staticmethod
    def _parse_text_response(response: str) -> Dict[str, Dict[str, str]]:
        mapping: Dict[str, Dict[str, str]] = {}
        if not response:
            return mapping
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if "|" in line:
                parts = line.split("|", 1)
                if len(parts) >= 2:
                    mapping[parts[0].strip()] = {"standard_name": parts[1].strip(), "naming_raw": line}
        return mapping

    def _name_to_smiles_batch(
        self,
        names: List[str],
        std_info: Dict[str, Dict[str, str]],
    ) -> Dict[str, Optional[str]]:
        name_to_smiles: Dict[str, Optional[str]] = {}
        for orig_name in names:
            if std_info.get(orig_name, {}).get('resolution_status') == 'response_mismatch':
                continue
            std_name = std_info.get(orig_name, {}).get("standard_name") or "UNKNOWN"
            smiles = self._resolve_smiles(orig_name, std_name)
            std_info.setdefault(orig_name, {}).update(
                resolution_trace=getattr(_resolution_state, 'trace', []),
                resolution_status=getattr(_resolution_state, 'status', 'unresolved'))
            label = split_isomer_label(orig_name)[1] or split_isomer_label(std_name)[1]
            if label:
                std_info[orig_name]['stereochemistry'] = dict(kind='unassigned_isomer',
                    descriptors=[label], structure_coarsened=True)
            if smiles:
                mol = Chem.MolFromSmiles(smiles) if Chem is not None else None
                if mol is not None and mol.GetStereoGroups():
                    std_info.setdefault(orig_name, {})["cxsmiles"] = Chem.MolToCXSmiles(mol)
                elif _RELATIVE_STEREO.search(f"{orig_name} {std_name}"):
                    smiles = _strip_stereo(smiles) or smiles
                name_to_smiles[orig_name] = _normalize_smiles(smiles) or smiles
        return name_to_smiles

    def _resolve_smiles(self, orig_name: str, std_name: str) -> Optional[str]:
        _resolution_state.trace = []
        _resolution_state.status = 'unresolved'
        from pipeline.chemical_quality import unspecified_structure_name
        if unspecified_structure_name(split_isomer_label(orig_name)[0]):
            _resolution_state.status = 'unspecified_structure'
            _resolution_state.trace = [dict(service='source_name', query=orig_name,
                status='unspecified_structure', reason='compound_class_or_placeholder')]
            return None
        tried_cir: set[str] = set()
        tried_pubchem: set[str] = set()
        tried_opsin: set[str] = set()

        def lookup(service, query, convert):
            _resolution_state.error = None
            _resolution_state.attempts = []
            value = convert(query)
            error = _resolution_state.error
            status = (error or {}).get('status', 'success' if value else 'no_result')
            if value and Chem is not None:
                try:
                    if Chem.MolFromSmiles(value) is None:
                        value, status = None, 'invalid_smiles'
                    elif value:
                        from pipeline.validation.normalize import name_composition_conflict
                        if name_composition_conflict(orig_name, value) or name_composition_conflict(std_name, value):
                            value, status = None, 'conflicting_structure'
                except Exception:
                    value, status = None, 'invalid_smiles'
            _resolution_state.trace.append({'service': service, 'query': query, 'status': status,
                                           **(error or {}), 'attempts': deepcopy_attempts()})
            for original in (std_name, orig_name):
                cleaned, label = split_isomer_label(original)
                if label and query == cleaned:
                    _resolution_state.trace[-1].update(original_query=original,
                        normalization='isomer_label_removed', isomer_label=label,
                        structure_scope='constitution')
                    break
            if value:
                _resolution_state.status = 'resolved'
            return value

        def deepcopy_attempts():
            return [dict(a) for a in getattr(_resolution_state, 'attempts', [])]

        def cir_lookup(query: str) -> Optional[str]:
            if query in tried_cir:
                return None
            tried_cir.add(query)
            return lookup('cir', query, self._cir_convert)

        def pubchem_lookup(query: str) -> Optional[str]:
            if query in tried_pubchem:
                return None
            tried_pubchem.add(query)
            return lookup('pubchem', query, self._pubchem_convert)

        def opsin_lookup(query: str) -> Optional[str]:
            if query in tried_opsin:
                return None
            tried_opsin.add(query)
            return lookup('opsin', query, self._opsin_convert)

        cas = self._extract_cas(orig_name)
        if cas:
            smiles = cir_lookup(cas) or pubchem_lookup(cas)
            if smiles:
                return smiles

        if (std_name or "").strip().upper() in _INVALID_NAMES:
            logger.debug(f"skip network cascade for [{orig_name}]: standard_name={std_name!r}")
            # The offline dictionary stays available even without a usable
            # standard_name, but no network/OPSIN resolution happens for a
            # name the LLM could not standardize (unchanged v7 semantics).
            key = orig_name.strip().lower()
            value = _AUXILIARY_SMILES.get(key)
            _resolution_state.status = ('resolved' if value else 'identifier_only' if std_name.strip().upper() == 'ID'
                                        else 'unusable_standard_name')
            _resolution_state.trace.append({'service': 'dictionary', 'query': orig_name,
                                            'status': _resolution_state.status})
            return value

        # 1. Offline dictionary: unambiguous solvents/gases/reagents. Beats
        #    every network source because those mis-resolve short acronyms
        #    (CIR: "DCM"→dCMP, "CMC"→carbodiimide tosylate) and elemental
        #    gases (OPSIN: "nitrogen"→[N] radical instead of N#N).
        for candidate in self._dedupe([orig_name, std_name]):
            key = candidate.strip().lower()
            if key in _AUXILIARY_SMILES:
                _resolution_state.status = 'resolved'
                _resolution_state.trace.append({'service': 'dictionary', 'query': candidate, 'status': 'success'})
                return _AUXILIARY_SMILES[key]

        # Sample numbers do not specify R/S or cis/trans. Keep them on the
        # output record, but resolve the chemical name without that suffix.
        std_query = split_isomer_label(std_name)[0]
        raw_query = split_isomer_label(orig_name)[0]
        # 2. OPSIN on the LLM-expanded chemical name first, then the raw name.
        for candidate in self._dedupe([std_query, raw_query]):
            smiles = opsin_lookup(candidate)
            if smiles:
                return smiles

        # 3. Network on the full standard_name.
        for candidate in self._dedupe([std_query]):
            smiles = pubchem_lookup(candidate) or cir_lookup(candidate)
            if smiles:
                return smiles

        # 4. Raw-alias network lookups last. A bare acronym with a usable
        #    full name is skipped: CIR/PubChem answer such acronyms
        #    arbitrarily (v7 P-04) and a wrong structure is worse than none.
        raw_candidates = self._raw_search_candidates(raw_query)
        for candidate in raw_candidates:
            if (_BARE_ACRONYM.fullmatch(candidate)
                    and self._is_queryable_name(std_name)
                    and std_name.strip().lower() != candidate.strip().lower()):
                logger.debug(f"skip acronym network lookup [{candidate}]; "
                             f"standard_name={std_name!r} is more specific")
                continue
            smiles = cir_lookup(candidate) or pubchem_lookup(candidate)
            if smiles:
                return smiles

        for candidate in self._dedupe([*raw_candidates, raw_query]):
            smiles = opsin_lookup(candidate)
            if smiles:
                return smiles

        statuses = {v['status'] for v in _resolution_state.trace}
        _resolution_state.status = next((s for s in ('conflicting_structure', 'rate_limited', 'timeout', 'service_failure', 'invalid_smiles')
                                         if s in statuses), 'no_result')
        if statuses and statuses <= {'disabled', 'unqueryable_name'}:
            _resolution_state.status = 'resolution_unavailable'
        return None

    def _cir_convert(self, name: str) -> Optional[str]:
        if not self.use_cir or not self._is_queryable_name(name):
            _resolution_state.error = {'status': 'disabled' if not self.use_cir else 'unqueryable_name'}
            return None
        _resolution_state.attempts = []
        for attempt, delay in enumerate((0.0,1.5,3.0),start=1):
            if delay:
                time.sleep(delay)
            _resolution_state.error = None
            try:
                result = self._call_with_timeout(
                    lambda: _network_call(lambda: cir.resolve(name, "smiles"), self.network_timeout),
                    self.network_timeout)
                if result:
                    _resolution_state.attempts.append(dict(attempt=attempt,status='success'))
                    return str(result)
            except Exception as e:
                _resolution_state.error = _lookup_error(e)
                logger.debug(f"CIR failed [{name}]: {e}")
            _resolution_state.attempts.append(dict(attempt=attempt, **(_resolution_state.error or {'status':'no_result'})))
            if not _transient_lookup(_resolution_state.error):
                break
        return None

    def _pubchem_convert(self, name: str) -> Optional[str]:
        if not self.use_pubchem or not self._is_queryable_name(name):
            _resolution_state.error = {'status': 'disabled' if not self.use_pubchem else 'unqueryable_name'}
            return None
        # PUG REST answers 503 ServerBusy under load; a short backoff recovers
        # most of them. The sleep runs in the calling chunk worker, not in the
        # shared timeout pool, so hung-call capacity is unaffected.
        _resolution_state.attempts = []
        for attempt, delay in enumerate((0.0, 1.5, 3.0), start=1):
            if delay:
                time.sleep(delay)
            try:
                _resolution_state.error = None
                compounds = self._call_with_timeout(
                    lambda: _network_call(lambda: pcp.get_compounds(name, "name"), self.network_timeout),
                    self.network_timeout,
                )
                if compounds:
                    smiles = getattr(compounds[0], "isomeric_smiles", None)
                    if smiles:
                        _resolution_state.attempts.append(dict(attempt=attempt,status='success'))
                        return smiles
                _resolution_state.attempts.append(dict(attempt=attempt, **(_resolution_state.error or {'status':'no_result'})))
                if attempt < 3 and _transient_lookup(_resolution_state.error):
                    continue
                return None
            except Exception as e:
                _resolution_state.error = _lookup_error(e)
                message = str(e)
                _resolution_state.attempts.append(dict(attempt=attempt, **_resolution_state.error))
                if (attempt < 3 and (_transient_lookup(_resolution_state.error)
                                     or "ServerBusy" in message or "Too many requests" in message)):
                    continue
                logger.debug(f"PubChem failed [{name}]: {e}")
                return None
        logger.debug(f"PubChem failed [{name}]: rate limited (503) after retries")
        return None

    def _opsin_convert(self, name: str) -> Optional[str]:
        if not self.use_opsin or not self.opsin or not self._is_queryable_name(name):
            _resolution_state.error = {'status': 'disabled' if not self.use_opsin or not self.opsin else 'unqueryable_name'}
            return None
        try:
            result = self.opsin.to_smiles([name], num_workers=1)
            if result and result[0] is not None:
                return result[0]
        except Exception as e:
            _resolution_state.error = _lookup_error(e)
            logger.debug(f"OPSIN failed [{name}]: {e}")
        return None

    @staticmethod
    def _call_with_timeout(func: Callable[[], T], timeout: int) -> Optional[T]:
        """Run *func* with a timeout on a shared bounded executor.

        The future is cancelled on timeout; a call already running in a worker
        thread keeps going but cannot spawn further threads, so concurrent
        batch documents cannot accumulate unbounded non-daemon threads. Once
        enough consecutive timeouts have saturated the pool, it is rebuilt so
        later queries are not doomed to time out as well (review round-2 M5).
        """
        global _consecutive_timeouts
        with _executor_lock:
            future = _TIMEOUT_EXECUTOR.submit(func)
        try:
            result = future.result(timeout=timeout)
        except FuturesTimeoutError:
            _resolution_state.error = {'status': 'timeout'}
            future.cancel()
            with _executor_lock:
                _consecutive_timeouts += 1
                saturated = _consecutive_timeouts >= _TIMEOUT_POOL_SIZE
            if saturated:
                _rebuild_timeout_executor()
            return None
        with _executor_lock:
            _consecutive_timeouts = 0
        return result

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
        labelled = re.match(r'^(?:name|compound|chemical(?: name)?|product|reagent|solvent|'
                            r'starting material|reaction product)\s*:\s*(.+)$', cleaned, re.I)
        if labelled:
            candidates.append(labelled[1].strip())
        # Parentheses, commas and colons are part of stereochemical and fused
        # ring names. Only remove explicit metadata, never chemical fragments.
        candidates.append(cleaned)
        return [c for c in cls._dedupe(candidates) if cls._is_queryable_name(c)]

    @staticmethod
    def _remove_cas(name: str) -> str:
        return _CAS_REMOVE_PATTERN.sub("", name).strip(" ;,")

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
        if stripped.upper() in _INVALID_NAMES or _CJK_PATTERN.search(stripped):
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
