"""ChemReX Module 1 — Text extraction (NER + Action).

Aligned with eval main experiment (eval/evaluate.py build_user_content contract):
- single user message, no system message
- temperature=0.0, max_tokens=512, enable_thinking=False
- 4 prompts (ner_zh/ner_en/action_zh/action_en) verbatim from data/seed-67/test.json
- response fallback msg.content or reasoning_content (vLLM-safe, aligned with eval)

Model selection:
- default: config.EXTRACTION_MODEL (e.g. deepseek-v3.2) via OpenAI-compatible API
- --preset qwen3-lora-08: Qwen3-8B + LoRA adapter 0.8 served by vLLM, aligned with
  eval/vllm_lora_reproduce.md (base_url=http://localhost:8050/v1, api_key=token-abc123)
"""
from .extractor import ChemReXExtractor

__all__ = ["ChemReXExtractor"]
