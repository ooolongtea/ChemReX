"""ChemReXExtractor — NER + Action extraction from clean experimental text.

Prompt construction mirrors eval/evaluate_ner.py:build_user_content:
  instruction (from versioned .md template, with {{TEXT}} filled) → single user message.
"""

import os
from typing import Dict, List, Optional, Any

from loguru import logger

from .inference import InferenceClient
from .parser import parse_ner_structured, parse_action_output


class ChemReXExtractor:
    """Extract NER (7 fields) + Action (verb sequence) from a clean text segment."""

    def __init__(self, preset: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, api_key: Optional[str] = None):
        from pipeline.prompts.prompt_loader import PromptLoader
        prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
        self.loader = PromptLoader(prompts_dir=prompts_dir)
        self.client = InferenceClient(preset=preset, model=model, base_url=base_url, api_key=api_key)
        self.preset = preset
        logger.debug(f"ChemReXExtractor ready (preset={preset or 'default'}, model={self.client.model})")

    def _build_prompt(self, task: str, language: str, text: str) -> str:
        name = f"{task}_{language}"
        content = self.loader.get_content(name)
        return content.replace("{{TEXT}}", text)

    def extract_ner(self, text: str, language: str) -> Dict[str, Any]:
        prompt = self._build_prompt("ner", language, text)
        raw = self.client.chat(prompt)
        structured = parse_ner_structured(raw)
        return {"ner": structured, "ner_raw": raw}

    def extract_action(self, text: str, language: str) -> Dict[str, Any]:
        prompt = self._build_prompt("action", language, text)
        raw = self.client.chat(prompt)
        actions = parse_action_output(raw)
        return {"action": actions, "action_raw": raw}

    def extract(
        self,
        text: str,
        language: str,
        task: str = "both",
    ) -> Dict[str, Any]:
        """Run extraction.

        Args:
            text: clean experimental paragraph.
            language: "zh" or "en".
            task: "ner" / "action" / "both".
        Returns:
            {"ner": {...7 fields...}, "action": [...verbs...], "language": ..., "task": ..., "raw": {...}}
        """
        if task not in ("ner", "action", "both"):
            raise ValueError(f"task must be 'ner'/'action'/'both', got {task!r}")
        result: Dict[str, Any] = {"language": language, "task": task, "ner": {}, "action": [], "raw": {}}
        if task in ("ner", "both"):
            ner_res = self.extract_ner(text, language)
            result["ner"] = ner_res["ner"]
            result["raw"]["ner"] = ner_res["ner_raw"]
        if task in ("action", "both"):
            act_res = self.extract_action(text, language)
            result["action"] = act_res["action"]
            result["raw"]["action"] = act_res["action_raw"]
        logger.info(
            f"Extraction done | task={task} lang={language} | "
            f"ner fields filled={sum(1 for v in result['ner'].values() if v)} | "
            f"actions={len(result['action'])}"
        )
        return result
