"""ChemReXExtractor — NER + Action extraction from clean experimental text.

Prompt construction mirrors eval/evaluate_ner.py:build_user_content:
  instruction (from versioned .md template, with {{TEXT}} filled) → single user message.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any

from loguru import logger

from .inference import InferenceClient
from .parser import parse_ner_records, parse_action_output, parse_action_details, parse_json_output


class ChemReXExtractor:
    """Extract NER (7 fields) + Action (verb sequence) from a clean text segment."""

    def __init__(self, preset: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, api_key: Optional[str] = None,
                 variant_preset: Optional[str] = None):
        from pipeline.prompts.prompt_loader import PromptLoader
        prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
        self.loader = PromptLoader(prompts_dir=prompts_dir)
        self.client = InferenceClient(preset=preset, model=model, base_url=base_url, api_key=api_key)
        self.preset = preset
        # Prompt variants (scheme_template) run on their own model: a
        # multi-step scheme narrative is outside the single-step LoRA's
        # fine-tuning distribution. Empty string disables the separate model.
        if variant_preset is None:
            try:
                import config
                variant_preset = getattr(config, "SCHEME_EXTRACTION_PRESET", "")
            except ImportError:
                variant_preset = ""
        self.variant_preset = variant_preset or ""
        self._variant_client = None
        logger.debug(f"ChemReXExtractor ready (preset={preset or 'default'}, model={self.client.model}, "
                     f"scheme_preset={self.variant_preset or 'same'})")

    @property
    def scheme_client(self) -> InferenceClient:
        """Client for prompt variants; the default client when unset."""
        if not getattr(self, "variant_preset", ""):
            return self.client
        if self._variant_client is None:
            # A scheme narrative yields one JSON record per transformation:
            # give the variant model room for the whole array.
            self._variant_client = InferenceClient(preset=self.variant_preset, max_tokens=4096)
        return self._variant_client

    def _build_prompt(self, task: str, language: str, text: str, variant: str = None) -> str:
        name = f"{task}_{language}" + (f"_{variant}" if variant else "")
        content = self.loader.get_content(name)
        return content.replace("{{TEXT}}", text)

    def configuration(self, language, variant: str = None):
        prompts = {}
        for task in ('ner', 'action'):
            # A prompt variant (scheme_template) replaces the NER prompt only.
            name = f"{task}_{language}" + (f"_{variant}" if variant and task == 'ner' else "")
            prompts[task] = {'version': self.loader.get_version(name),
                             'content': self.loader.get_content(name)}
        if variant:
            scheme = self.scheme_client
            config = {'model': scheme.model, 'preset': scheme.preset, 'prompts': prompts,
                      'prompt_variant': variant}
        else:
            config = {'model': self.client.model, 'preset': self.client.preset, 'prompts': prompts}
        return config

    def plan_steps(self, text: str, title: str) -> Dict[str, Any]:
        import json
        prompt = self.loader.get_content("step_plan")
        prompt = prompt.replace("{{TITLE}}", title).replace("{{TEXT}}", json.dumps(text, ensure_ascii=False))
        raw = self.client.chat(prompt, max_tokens=2048)
        parsed = parse_json_output(raw)
        return {"raw": raw, "ranges": parsed if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed) else []}

    def extract_ner(self, text: str, language: str, variant: str = None) -> Dict[str, Any]:
        client = self.scheme_client if variant else self.client
        prompt = self._build_prompt("ner", language, text, variant)
        token_limit = min(4096, max(client.max_tokens, 512 + len(text) // 2))
        raw = ""
        for attempt in range(2):
            raw = client.chat(prompt, max_tokens=token_limit)
            diagnostics = []
            records = parse_ner_records(raw, diagnostics, prompt_variant=variant)
            if records:
                if attempt == 0 and any(d.get('reason') == 'unbalanced_brackets' for d in diagnostics):
                    continue
                return {"ner": records[0] if len(records) == 1 else {},
                        "ner_records": records, "ner_raw": raw, "parse_status": "ok",
                        "ner_parse_diagnostics": diagnostics}
            token_limit = min(4096, token_limit * 2)
        return {"ner": {}, "ner_records": [], "ner_raw": raw,
                "parse_status": "invalid_ner_json"}

    def extract_action(self, text: str, language: str) -> Dict[str, Any]:
        prompt = self._build_prompt("action", language, text)
        raw = ""
        for _ in range(2):
            raw = self.client.chat(prompt)
            details = parse_action_details(raw)
            actions = details['actions']
            if actions:
                return {"action": actions, "action_raw": raw, "parse_status": details['status'],
                        "action_parse_details": details}
        return {"action": [], "action_raw": raw, "parse_status": details['status'],
                "action_parse_details": details}

    def extract(
        self,
        text: str,
        language: str,
        task: str = "both",
        prompt_variant: str = None,
    ) -> Dict[str, Any]:
        """Run extraction.

        Args:
            text: clean experimental paragraph.
            language: "zh" or "en".
            task: "ner" / "action" / "both".
            prompt_variant: optional NER prompt variant name (e.g.
                "scheme_template" for general-scheme paragraphs that describe
                several transformations in one narrative).
        Returns:
            {"ner": {...7 fields...}, "action": [...verbs...], "language": ..., "task": ..., "raw": {...}}
        """
        if task not in ("ner", "action", "both"):
            raise ValueError(f"task must be 'ner'/'action'/'both', got {task!r}")
        result: Dict[str, Any] = {"language": language, "task": task, "ner": {},
                                  "ner_records": [], "action": [], "raw": {}, "parse_status": {}}
        result['extraction_configuration'] = self.configuration(language, prompt_variant)
        if task == 'both':
            # Independent calls share the existing provider rate limiter.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix='ner-action') as pool:
                ner_future = pool.submit(self.extract_ner, text, language, prompt_variant)
                action_future = pool.submit(self.extract_action, text, language)
                ner_res, act_res = ner_future.result(), action_future.result()
        if task in ("ner", "both"):
            if task == 'ner':
                ner_res = self.extract_ner(text, language, prompt_variant)
            result["ner"] = ner_res["ner"]
            result["ner_records"] = ner_res["ner_records"]
            result["raw"]["ner"] = ner_res["ner_raw"]
            result["parse_status"]["ner"] = ner_res["parse_status"]
            result['ner_parse_diagnostics'] = ner_res.get('ner_parse_diagnostics', [])
        if task in ("action", "both"):
            if task == 'action':
                act_res = self.extract_action(text, language)
            result["action"] = act_res["action"]
            result["raw"]["action"] = act_res["action_raw"]
            result["parse_status"]["action"] = act_res["parse_status"]
            result["action_parse_details"] = act_res["action_parse_details"]
        logger.info(
            f"Extraction done | task={task} lang={language} | "
            f"ner fields filled={sum(1 for v in result['ner'].values() if v)} | "
            f"actions={len(result['action'])}"
        )
        return result
