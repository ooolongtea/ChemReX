"""Final-product extraction from the document head (abstract region).

Runs once per document in the document runner: the abstract usually names the
target compound of the whole synthesis route, which gives the route direction
(complementing the Example/Intermediate candidate links in ``links.json``).

Reuses ``prompts/final_product_extraction/v1.md`` + ``pipeline/gpt_api.py``.
"""

import json
import os
import re
from typing import Any, Dict, Optional

from loguru import logger

try:
    import config
except ImportError:
    from ... import config

from pipeline.gpt_api import GPTAPI
from pipeline.prompts import PromptLoader


FINAL_PRODUCT_VERSION = "2026-09-25-v3"


class TargetExtractor:
    """Extract the document-level final product named in the abstract."""

    @staticmethod
    def plausible(product: str, reference: str) -> bool:
        name = str(product or "").strip()
        evidence = str(reference or "").strip()
        if not name or not evidence or name not in evidence:
            return False
        if re.search(r"et al\.|\bvol\.|Annual Meeting|\bdoi\b", evidence, re.I):
            return False
        if re.search(r"^(?:可用作|一种如结构式|compound of formula|compounds? useful for)",
                     name, re.I):
            return False
        if len(name) > 100 and not re.search(r"\d|\(|\)", name):
            return False
        return True

    def __init__(self, model: str = None, temperature: float = None):
        loader = PromptLoader()
        prompt_data = loader.load_prompt("final_product_extraction")
        self.loader = loader
        self.prompt_name = "final_product_extraction"
        meta = prompt_data["metadata"]
        model = (
            model
            or getattr(config, "FINAL_PRODUCT_MODEL", None)
            or meta.get("model", "gpt-4o")
        )
        temperature = (
            temperature
            if temperature is not None
            else getattr(config, "FINAL_PRODUCT_TEMPERATURE", meta.get("temperature", 0.0))
        )
        self.api = GPTAPI(
            model=model,
            temperature=temperature,
            provider=os.getenv("FINAL_PRODUCT_PROVIDER") or None,
        )
        logger.debug(f"TargetExtractor ready: model={model}, temperature={temperature}")

    def extract(self, head_text: str, language: str = "en") -> Optional[Dict[str, Any]]:
        """Extract ``{"product", "reference"}`` from the document head text.

        Returns None when the head is empty or no final product is found.
        """
        if not head_text or not head_text.strip():
            logger.warning("final product skipped: empty document head text")
            return None
        prompt = self.loader.format_prompt(
            self.prompt_name, content=head_text, language=language
        )
        try:
            response = self.api.chat(prompt)
        except Exception as exc:
            logger.error(f"final product LLM call failed: {exc}")
            raise
        start, end = response.find("{"), response.rfind("}") + 1
        if start < 0 or end <= start:
            logger.warning("final product: no JSON object in LLM response")
            return None
        try:
            data = json.loads(response[start:end])
        except json.JSONDecodeError:
            logger.warning("final product: JSON decode failed")
            return None
        product = data.get("product")
        reference = data.get("reference")
        if not self.plausible(product, reference) or reference not in head_text:
            logger.warning("final product rejected: no specific located compound")
            return None
        result: Dict[str, Any] = {"product": product, "reference": reference,
                                  "_version": FINAL_PRODUCT_VERSION}
        logger.info(f"Final product extracted: {product}")
        return result
