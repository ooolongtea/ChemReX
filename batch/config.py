"""Batch configuration — everything overridable via environment variables.

Loads the root ``.env`` directly (same file as the root ``config`` module)
but deliberately does NOT import that module: the root config freezes
``EXTRACTION_PRESET`` / ``STANDARDIZE_*`` / ``COREF_*`` at import time, and
the scheduler must be able to apply the batch LLM defaults (below) before
the pipeline modules are first imported.

LLM defaults: batch runs on the Huawei MaaS preset with ``MAAS_MODEL``
(``deepseek-v4.1-flash`` by default) for M1 and M3, unless overridden by
``--preset`` / ``--llm-provider`` / ``--llm-model`` or the ``BATCH_*``
variables below. The core pipeline keeps its own ``.env`` defaults.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")  # existing environment variables win
except Exception:
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


# ── LLM defaults (MaaS) ──
BATCH_EXTRACTION_PRESET = _env("BATCH_EXTRACTION_PRESET", "huawei-maas")
BATCH_LLM_PROVIDER = _env("BATCH_LLM_PROVIDER", "maas")
BATCH_LLM_MODEL = _env("BATCH_LLM_MODEL", _env("MAAS_MODEL", "deepseek-v4.1-flash-1017"))

# M1 routing by document language. Chinese uses the local Qwen3-8B LoRA
# adapter with the training-aligned v1 prompts; English keeps MaaS.
BATCH_EXTRACTION_PRESET_ZH = _env("BATCH_EXTRACTION_PRESET_ZH", "qwen3-lora-08")
BATCH_EXTRACTION_PRESET_EN = _env("BATCH_EXTRACTION_PRESET_EN", BATCH_EXTRACTION_PRESET)
BATCH_EXTRACTION_PRESETS = {
    "zh": BATCH_EXTRACTION_PRESET_ZH,
    "en": BATCH_EXTRACTION_PRESET_EN,
}

# ── Storage ──
BATCH_DB_PATH = Path(_env("BATCH_DB_PATH", str(BASE_DIR / "data" / "batch" / "tasks.db")))
BATCH_OUTPUT_ROOT = Path(_env("BATCH_OUTPUT_ROOT", str(BASE_DIR / "results" / "batch")))
BATCH_LOG_DIR = Path(_env("BATCH_LOG_DIR", str(BASE_DIR / "results" / "batch" / "logs")))

# ── Resident MinerU parsing service ──
MINERU_SERVICE_URL = _env("BATCH_MINERU_URL", "http://127.0.0.1:9246")
MINERU_BACKEND = _env("MINERU_BACKEND", "vlm-auto-engine")
MINERU_TIMEOUT = int(_env("BATCH_MINERU_TIMEOUT", "7200"))  # per-PDF HTTP timeout

# ── Resident image worker (imgparser conda env) ──
IMGPARSER_PYTHON = _env("IMGPARSER_PYTHON", "")
IMGPARSER_CONDA_ENV = _env("IMGPARSER_CONDA_ENV", "imgparser")
IMAGE_TIMEOUT = int(_env("BATCH_IMAGE_TIMEOUT", "1800"))  # per-document timeout

# ── Scheduler concurrency ──
TASK_WORKERS = int(_env("BATCH_TASK_WORKERS", "2"))      # concurrent PDFs
CHUNK_WORKERS = int(_env("BATCH_CHUNK_WORKERS", "16"))   # concurrent LLM calls per PDF
POLL_INTERVAL = float(_env("BATCH_POLL_INTERVAL", "10"))

# ── Task lifecycle ──
MAX_ATTEMPTS = int(_env("BATCH_MAX_ATTEMPTS", "3"))
LOCK_EXPIRE_SECONDS = int(_env("BATCH_LOCK_EXPIRE", "7200"))
RETRY_BACKOFF = float(_env("BATCH_RETRY_BACKOFF", "5"))  # base seconds, doubles per attempt


def image_worker_command(script_path: Path) -> list:
    """Argv prefix that runs the image worker in the imgparser environment."""
    if IMGPARSER_PYTHON:
        return [IMGPARSER_PYTHON, "-u", str(script_path)]
    return ["conda", "run", "--no-capture-output", "-n", IMGPARSER_CONDA_ENV,
            "python", "-u", str(script_path)]
