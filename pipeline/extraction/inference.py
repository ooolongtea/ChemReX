"""OpenAI-compatible inference client — single user message (aligned with eval contract).

Mirrors eval/evaluate.py:generate_with_external_api:
- single user message, no system message
- temperature / max_tokens / extra_body(chat_template_kwargs.enable_thinking)
- response: final message content; reasoning-only and empty messages are
  retried and then raised so the scheduler can retry the failed chunk.

Supports named presets (config.MODEL_PRESETS) to target a vLLM-served
Qwen3-8B + LoRA adapter, aligned with eval/vllm_lora_reproduce.md.

Precedence: explicit ctor args > preset > env vars > config defaults.
"""

import os
import time
from typing import Optional

from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError
from pipeline.maas_rate_limit import wait_for_slot
from pipeline.model_provider import sdk_base_url, request_timeout


class ContextLengthError(ValueError):
    """Deterministic input overflow; retrying the document cannot fix it."""

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
        if preset_name == "huawei-maas":
            self.api_key = api_key if api_key is not None else p.get("api_key", "")
            if not self.api_key:
                raise ValueError("MAAS_API_KEY is required for the huawei-maas preset")
        else:
            self.api_key = api_key or p.get("api_key") or os.getenv("API_KEY", "") or ""
        self.base_url = sdk_base_url(
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

        self.timeout = timeout if timeout is not None else _cfg("EXTRACTION_TIMEOUT", 300)
        self.preset = preset_name
        self._maas = (preset_name == "huawei-maas"
                      or "modelarts-maas" in self.base_url)

        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                 timeout=request_timeout(self.timeout), max_retries=0)
        else:
            self.client = OpenAI(api_key=self.api_key, timeout=request_timeout(self.timeout),
                                 max_retries=0)

    def chat(self, user_content: str, max_tokens: Optional[int] = None) -> str:
        """Send a single user message and return the text response."""
        messages = [{"role": "user", "content": user_content}]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if self.enable_thinking is not None:
            thinking_key = "thinking" if self.preset == "huawei-maas" else "enable_thinking"
            kwargs["extra_body"] = {"chat_template_kwargs": {thinking_key: self.enable_thinking}}
        from loguru import logger

        def request(options):
            if self._maas:
                wait_for_slot()
            started = time.monotonic()
            response = self.client.chat.completions.create(**options)
            from pipeline.call_metrics import record_call
            record_call("extraction", self.model, started, response, preset=self.preset)
            return response

        for attempt in range(4):
            try:
                response = request(kwargs)
            except (RateLimitError, APIConnectionError, InternalServerError):
                if attempt == 3:
                    raise
                time.sleep(min(10 * 2 ** attempt, 40))
                continue
            except Exception as exc:
                # Only an explicit unsupported-option response warrants
                # dropping the thinking setting. Authentication, disk and
                # other failures must retain their original diagnosis.
                detail = str(exc).lower()
                if (getattr(exc, 'status_code', None) == 400 and
                        any(term in detail for term in ('maximum context length',
                            'context_length_exceeded', 'max model len', 'max_model_len',
                            'context window', 'too many tokens'))):
                    raise ContextLengthError(str(exc)) from exc
                if (attempt < 3 and "extra_body" in kwargs
                        and getattr(exc, "status_code", None) in (400, 422)
                        and any(key in detail for key in ("thinking", "chat_template_kwargs", "extra_body"))
                        and any(word in detail for word in ("unsupported", "unknown", "unexpected", "not permitted"))):
                    kwargs.pop("extra_body")
                    continue
                raise

            choice = response.choices[0] if response.choices else None
            msg = choice.message if choice else None
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                return content
            finish_reason = getattr(choice, "finish_reason", None)
            usage = getattr(response, "usage", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            diagnosis = (f"M1 returned an empty final response (model={self.model}, "
                         f"finish_reason={finish_reason}, completion_tokens={completion_tokens}, "
                         f"reasoning_only={bool(getattr(msg, 'reasoning_content', None))})")
            if attempt == 3:
                raise RuntimeError(diagnosis)
            logger.warning("{}; retrying ({}/4)", diagnosis, attempt + 1)
            if finish_reason == "length":
                kwargs["max_tokens"] = min(4096, kwargs["max_tokens"] * 2)
            time.sleep(min(2 ** attempt, 4))
