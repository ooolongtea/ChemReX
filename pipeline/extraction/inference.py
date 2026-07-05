"""OpenAI-compatible inference client — single user message (aligned with eval contract).

Mirrors eval/evaluate.py:generate_with_external_api:
- single user message, no system message
- temperature / max_tokens / extra_body(chat_template_kwargs.enable_thinking)
- response: msg.content or getattr(msg, "reasoning_content", "") or ""
  (vLLM may return content=None when reasoning is enabled, aligned with eval)

Supports named presets (config.MODEL_PRESETS) to target a vLLM-served
Qwen3-8B + LoRA adapter, aligned with eval/vllm_lora_reproduce.md.

Precedence: explicit ctor args > preset > env vars > config defaults.
"""

import os
from typing import Optional

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _resolve_preset(preset: Optional[str]) -> dict:
    """Resolve a preset name to its config dict (empty dict if no preset)."""
    if not preset:
        return {}
    try:
        import config
    except ImportError:
        config = None
    presets = getattr(config, "MODEL_PRESETS", {}) if config is not None else {}
    if preset not in presets:
        raise ValueError(f"unknown preset: {preset!r}. available: {list(presets)}")
    return presets[preset]


class InferenceClient:
    """Single-user-message OpenAI-compatible client."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        preset: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        try:
            import config
        except ImportError:
            config = None

        def _cfg(name, default):
            if config is not None:
                return getattr(config, name, default)
            return default

        # preset name: explicit arg > config default
        preset_name = preset or _cfg("EXTRACTION_PRESET", "")
        p = _resolve_preset(preset_name)

        # precedence: explicit arg > preset > env var > config default
        self.api_key = api_key or p.get("api_key") or os.getenv("API_KEY", "") or ""
        self.base_url = (
            base_url
            or p.get("base_url")
            or os.getenv("BASE_URL")
            or _cfg("BASE_URL", "https://api.openai.com/v1")
        )
        self.model = model or p.get("model") or _cfg("EXTRACTION_MODEL", "deepseek-v3.2")

        if temperature is not None:
            self.temperature = temperature
        elif "temperature" in p:
            self.temperature = p["temperature"]
        else:
            self.temperature = 0.0

        if max_tokens is not None:
            self.max_tokens = max_tokens
        elif "max_tokens" in p:
            self.max_tokens = p["max_tokens"]
        else:
            self.max_tokens = 512

        if enable_thinking is not None:
            self.enable_thinking = enable_thinking
        elif "enable_thinking" in p:
            self.enable_thinking = p["enable_thinking"]
        else:
            self.enable_thinking = False

        self.timeout = timeout or _cfg("EXTRACTION_TIMEOUT", 300)
        self.preset = preset_name

        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)

    def chat(self, user_content: str) -> str:
        """Send a single user message and return the text response."""
        messages = [{"role": "user", "content": user_content}]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.enable_thinking is not None:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        try:
            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            return msg.content or getattr(msg, "reasoning_content", "") or ""
        except Exception:
            try:
                kwargs.pop("extra_body", None)
                response = self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                return msg.content or getattr(msg, "reasoning_content", "") or ""
            except Exception as e:
                from loguru import logger
                logger.error(f"Inference failed: {e}")
                return ""
