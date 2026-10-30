"""OpenAI-compatible LLM client."""

from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError
from dotenv import load_dotenv
import os
import time
from pipeline.model_provider import sdk_base_url, request_timeout
from pipeline.maas_rate_limit import wait_for_slot

load_dotenv()


class GPTAPI:
    def __init__(self, model="deepseek-v3.2", temperature=0.0, provider=None):
        self.model = model
        self.temperature = temperature
        maas_model = self._env("MAAS_MODEL", "deepseek-v4.1-flash")
        self.provider = (provider or ("maas" if (model or "").lower() == maas_model.lower() else "")).lower()
        self.api_key, self.base_url = self._resolve_provider_config(model, provider)
        if not self.api_key:
            raise RuntimeError(
                "GPTAPI 初始化失败：未找到 API key。"
                "请设置 API_KEY，或设置 MAAS_API_KEY / DEEPSEEK_API_KEY / CHATGPT_API_KEY "
                "（ChatGPT 也兼容 OPENAI_API_KEY）。"
            )
        # A dead MaaS/TLS socket must fail the chunk and let the task retry;
        # without an explicit timeout the SDK connection pool can hang for
        # hours (observed as 26 CLOSE-WAIT sockets and a stalled batch).
        timeout = request_timeout(self._env("LLM_TIMEOUT", self._env("EXTRACTION_TIMEOUT", "300")))
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                 max_retries=0,
                                 timeout=timeout)
        else:
            self.client = OpenAI(api_key=self.api_key,
                                 max_retries=0,
                                 timeout=timeout)

    @staticmethod
    def _env(name: str, default: str = "") -> str:
        return os.getenv(name) or default

    @classmethod
    def _resolve_provider_config(cls, model: str, provider=None):
        model_name = (model or "").lower()
        provider_name = (provider or "").lower()
        if provider_name not in ("", "maas", "deepseek", "chatgpt", "openai", "generic"):
            raise ValueError(f"unknown LLM provider: {provider!r}")
        generic_key = cls._env("API_KEY")
        generic_base = cls._env("BASE_URL")
        maas_model = cls._env("MAAS_MODEL", "deepseek-v4.1-flash").lower()
        if provider_name == "maas" or (not provider_name and model_name == maas_model):
            return (
                cls._env("MAAS_API_KEY"),
                sdk_base_url(cls._env("MAAS_BASE_URL", "https://api.modelarts-maas.com/v2/chat/completions")),
            )
        if provider_name in ("chatgpt", "openai") or (not provider_name and ("gpt" in model_name or "chatgpt" in model_name)):
            return (
                cls._env("CHATGPT_API_KEY", cls._env("OPENAI_API_KEY", generic_key)),
                cls._env(
                    "CHATGPT_BASE_URL",
                    cls._env("OPENAI_BASE_URL", generic_base or "https://api.openai.com/v1"),
                ),
            )
        if provider_name == "deepseek" or (not provider_name and "deepseek" in model_name):
            return (
                cls._env("DEEPSEEK_API_KEY", generic_key),
                cls._env("DEEPSEEK_BASE_URL", generic_base or "https://api.deepseek.com"),
            )
        return (
            generic_key or cls._env("OPENAI_API_KEY"),
            generic_base or cls._env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    def chat(self, system_prompt, user_content=None):
        # MaaS can return a successful response with content=null for a
        # system-only request. The existing M3 prompts include their input, so
        # send that single prompt as a user message on this provider.
        if self.provider == "maas" and user_content is None:
            messages = [{"role": "user", "content": system_prompt}]
        else:
            messages = [{"role": "system", "content": system_prompt}]
            if user_content is not None:
                messages.append({"role": "user", "content": "content:\n" + user_content})
        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self.provider == "maas":
            kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": False}}
        for attempt in range(4):
            if self.provider == "maas":
                wait_for_slot()
            try:
                started = time.monotonic()
                response = self.client.chat.completions.create(**kwargs)
                from pipeline.call_metrics import record_call
                record_call("standardize_coref_final", self.model, started, response,
                            provider=self.provider or None)
                break
            except (RateLimitError, APIConnectionError, InternalServerError):
                if attempt == 3:
                    raise
                time.sleep(min(10 * 2 ** attempt, 40))
        message = response.choices[0].message
        content = message.content or getattr(message, "reasoning_content", None)
        if not content:
            raise RuntimeError("LLM returned an empty message")
        return content
