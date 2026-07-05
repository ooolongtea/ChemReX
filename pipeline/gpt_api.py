"""OpenAI-compatible LLM client."""

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class GPTAPI:
    def __init__(self, model="deepseek-v3.2", temperature=0.0):
        self.model = model
        self.temperature = temperature
        self.api_key, self.base_url = self._resolve_provider_config(model)
        if not self.api_key:
            raise RuntimeError(
                "GPTAPI 初始化失败：未找到 API key。"
                "请设置 API_KEY，或设置 DEEPSEEK_API_KEY / CHATGPT_API_KEY "
                "（ChatGPT 也兼容 OPENAI_API_KEY）。"
            )
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)

    @staticmethod
    def _env(name: str, default: str = "") -> str:
        return os.getenv(name) or default

    @classmethod
    def _resolve_provider_config(cls, model: str):
        model_name = (model or "").lower()
        generic_key = cls._env("API_KEY")
        generic_base = cls._env("BASE_URL")
        if "gpt" in model_name or "chatgpt" in model_name:
            return (
                cls._env("CHATGPT_API_KEY", cls._env("OPENAI_API_KEY", generic_key)),
                cls._env(
                    "CHATGPT_BASE_URL",
                    cls._env("OPENAI_BASE_URL", generic_base or "https://api.openai.com/v1"),
                ),
            )
        if "deepseek" in model_name:
            return (
                cls._env("DEEPSEEK_API_KEY", generic_key),
                cls._env("DEEPSEEK_BASE_URL", generic_base or "https://api.deepseek.com"),
            )
        return (
            generic_key or cls._env("OPENAI_API_KEY"),
            generic_base or cls._env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    def chat(self, system_prompt, user_content=None):
        messages = [{"role": "system", "content": system_prompt}]
        if user_content is not None:
            messages.append({"role": "user", "content": "content:\n" + user_content})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
