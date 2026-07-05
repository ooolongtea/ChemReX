"""ChemReX pipeline orchestrator — thin coordinator for 4 primary modules.

Modules (each independently runnable via `python -m pipeline.<m>.run_*`):
    1. extraction   — NER(7 fields) + Action(verb sequence) from clean text
    2. image        — compound dictionary(ID→SMILES) + reaction structures
    3. coref        — standardize compounds, then map identifiers to image-side SMILES
    4. validation   — field-constraint check + RXNMapper mapping screening

Hard dependencies:
    - extraction   depends on nothing
    - coref        needs extraction NER; image dict_result enables ID mapping
    - validation   needs extraction NER+Action (else warn & skip)
"""

import time
from typing import Dict, List, Optional, Any

from loguru import logger

try:
    import config
except ImportError:
    from .. import config


class ChemReXPipeline:
    """Thin coordinator. By default does NOT chain; use ``chain`` param to chain."""

    MODULES = ("extraction", "image", "coref", "validation")

    def __init__(
        self,
        enable_image: bool = True,
        enable_coref: bool = True,
        enable_standardize: bool = True,
        enable_validation: bool = True,
    ):
        self.enable_image = enable_image
        self.enable_coref = enable_coref
        self.enable_standardize = enable_standardize
        self.enable_validation = enable_validation
        self._extraction = None
        self._image = None
        self._standardize = None
        self._coref = None
        self._validation = None
        logger.info(
            "ChemReXPipeline ready "
            f"(image={enable_image}, standardize={enable_standardize}, "
            f"coref={enable_coref}, validation={enable_validation})"
        )

    # ── lazy module loaders ──
    def _get_extraction(self):
        if self._extraction is None:
            from .extraction.extractor import ChemReXExtractor
            self._extraction = ChemReXExtractor()
        return self._extraction

    def _get_standardize(self):
        if self._standardize is None:
            from .standardize.standardizer import Standardizer
            self._standardize = Standardizer()
        return self._standardize

    def _get_coref(self):
        if self._coref is None:
            from .coref.coref_resolver import CorefResolver
            self._coref = CorefResolver()
        return self._coref

    def _get_validation(self):
        if self._validation is None:
            from .validation.validator import ReactionValidator
            self._validation = ReactionValidator()
        return self._validation

    def _get_image(self):
        if self._image is None:
            from .image.image_pipeline import ImagePipeline
            self._image = ImagePipeline()
        return self._image

    # ── main entry: run on a single text segment ──
    def run(
        self,
        text: str,
        language: str,
        task: str = "both",
        image_base: Optional[str] = None,
        chain: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run modules on a single text segment.

        Args:
            text: clean experimental paragraph.
            language: "zh" or "en" (provided by caller, NOT detected).
            task: "ner" / "action" / "both".
            image_base: path to image directory (for image module). None = skip image.
            chain: explicit ordered list of modules to run; None = run all enabled.
        Returns:
            dict with keys per module run: extraction / image / coref / validation.
        """
        start = time.time()
        language = self._norm_language(language)
        if not text or not text.strip():
            raise ValueError("text must be a non-empty experimental paragraph")
        if language not in ("zh", "en"):
            raise ValueError(f"language must be 'zh' or 'en', got: {language!r}")

        order = chain or [m for m in ("extraction", "image", "coref", "validation")
                          if getattr(self, f"enable_{m}", True)]
        result: Dict[str, Any] = {}

        # 1. extraction
        if "extraction" in order:
            ext = self._get_extraction()
            result["extraction"] = ext.extract(text=text, language=language, task=task)

        # Compatibility: standalone standardize step when explicitly requested.
        if "standardize" in order:
            if "extraction" not in result:
                logger.warning("standardize skipped: extraction result missing")
                result["standardize"] = {"status": "skipped", "reason": "extraction result missing"}
            else:
                try:
                    std = self._get_standardize()
                    result["standardize"] = std.standardize(result["extraction"])
                except Exception as e:
                    logger.error(f"standardize module failed: {e}")
                    result["standardize"] = {"status": "error", "reason": str(e)}

        # 2. image (independent of extraction; needs image_base)
        if "image" in order:
            if not image_base:
                logger.warning("image skipped: no image_base provided")
                result["image"] = {"status": "skipped", "reason": "no image_base provided"}
            else:
                try:
                    img = self._get_image()
                    result["image"] = img.run(image_base)
                except Exception as e:
                    logger.error(f"image module failed: {e}")
                    result["image"] = {"status": "error", "reason": str(e), "dict_result": {}, "image_result": []}

        # 3. coref (needs NER; internally standardizes raw extraction records.
        #    image dict_result enables ID→SMILES mapping).
        if "coref" in order:
            std_res = result.get("standardize")
            ext_res = result.get("extraction")
            # Only use a standardized record that actually carries NER; an
            # error/skipped standardize dict has no "ner" key, so fall back.
            record_source = std_res if isinstance(std_res, dict) and "ner" in (std_res or {}) else ext_res
            dict_result = (result.get("image") or {}).get("dict_result") or {}
            if not record_source or "ner" not in record_source:
                logger.warning("coref skipped: need NER result")
                result["coref"] = {"status": "skipped", "reason": "need NER result"}
            else:
                try:
                    coref = self._get_coref()
                    payload = dict(record_source)
                    payload["dict_result"] = dict_result
                    result["coref"] = coref.resolve(payload)
                except Exception as e:
                    logger.error(f"coref module failed: {e}")
                    result["coref"] = {"status": "error", "reason": str(e)}

        # 4. validation (needs NER+Action). Prefer coref output because it carries
        #    standardized compounds and any ID→SMILES mapping; then fall back.
        if "validation" in order:
            coref_res = result.get("coref")
            std_res = result.get("standardize")
            ext_res = result.get("extraction")
            if isinstance(coref_res, dict) and "ner" in (coref_res or {}):
                val_res = coref_res
            elif isinstance(std_res, dict) and "ner" in (std_res or {}):
                val_res = std_res
            else:
                val_res = ext_res
            if not val_res or "ner" not in val_res:
                logger.warning("validation skipped: need NER+Action record")
                result["validation"] = {"status": "skipped", "reason": "need NER+Action record"}
            else:
                try:
                    val = self._get_validation()
                    result["validation"] = val.validate_record(val_res)
                except Exception as e:
                    logger.error(f"validation module failed: {e}")
                    result["validation"] = {"status": "error", "reason": str(e)}

        elapsed = time.time() - start
        logger.info(f"Pipeline done | {elapsed:.1f}s | modules: {list(result.keys())}")
        return result

    @staticmethod
    def _norm_language(lang: str) -> str:
        if not lang:
            raise ValueError("language must be provided ('zh' or 'en'); detection is not supported")
        s = str(lang).strip().lower()
        if s in ("zh", "zh-cn", "chinese", "cn"):
            return "zh"
        if s in ("en", "english"):
            return "en"
        return s
