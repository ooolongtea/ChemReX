"""
ChemReX 配置文件 — 收缩版（5 模块：extraction/standardize/image/coref/validation）

所有模型路径、conda 环境均可通过环境变量覆盖。
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass


def _env(name: str, default: str = "") -> str:
    return os.getenv(name) or default


# ── LLM API（extraction / standardize / coref 共用）──
DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_V4_PRO_MODEL = _env("DEEPSEEK_V4_PRO_MODEL", "deepseek-v4-pro")
CHATGPT_API_KEY = _env("CHATGPT_API_KEY", _env("OPENAI_API_KEY"))
CHATGPT_BASE_URL = _env(
    "CHATGPT_BASE_URL",
    _env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)
CHATGPT_MODEL = _env("CHATGPT_MODEL", "gpt-4o")
API_KEY = _env("API_KEY", DEEPSEEK_API_KEY or CHATGPT_API_KEY)
BASE_URL = _env("BASE_URL", DEEPSEEK_BASE_URL if DEEPSEEK_API_KEY else CHATGPT_BASE_URL)

# ── 模块1 文本提取 (NER+Action) ──
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "deepseek-v3.2")
EXTRACTION_TEMPERATURE = 0.0
EXTRACTION_MAX_TOKENS = 512
EXTRACTION_ENABLE_THINKING = False
EXTRACTION_TIMEOUT = int(os.getenv("EXTRACTION_TIMEOUT", "300"))

# ── 模型预设（命名配置，对齐 eval/vllm_lora_reproduce.md）──
# 用 --preset <name> 选择；预设覆盖 model/base_url/api_key/temperature/max_tokens/enable_thinking。
# 优先级：显式 CLI 参数 > 预设 > 环境变量 > 上方默认值。
MODEL_PRESETS = {
    "deepseek-v4-pro": {
        "model": DEEPSEEK_V4_PRO_MODEL,
        "base_url": DEEPSEEK_BASE_URL,
        "api_key": DEEPSEEK_API_KEY or API_KEY,
        "temperature": 0.0,
        "max_tokens": 512,
        "enable_thinking": False,
        "description": "DeepSeek V4 Pro via OpenAI-compatible API. Configure DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL in .env.",
    },
    "chatgpt": {
        "model": CHATGPT_MODEL,
        "base_url": CHATGPT_BASE_URL,
        "api_key": CHATGPT_API_KEY or API_KEY,
        "temperature": 0.0,
        "max_tokens": 512,
        "enable_thinking": False,
        "description": "ChatGPT/OpenAI via OpenAI-compatible API. Configure CHATGPT_API_KEY/CHATGPT_BASE_URL in .env.",
    },
    "qwen3-lora-08": {
        "model": "qwen3-lora-08",
        "base_url": "http://localhost:8050/v1",
        "api_key": "token-abc123",
        "temperature": 0.0,
        "max_tokens": 512,
        "enable_thinking": False,
        "description": "Qwen3-8B + LoRA adapter 0.8 (vLLM OpenAI-compatible)。需先部署 vLLM 服务并挂载 qwen3-lora-08 adapter，详见 README_CN.md 模型准备与 vLLM 启动。",
    },
}
# M1 默认预设：Qwen3-8B + LoRA 0.8（需本地启动 vLLM 并挂载该 LoRA adapter）
EXTRACTION_PRESET = os.getenv("EXTRACTION_PRESET", "qwen3-lora-08")

# ── 模块3 数据标准化（name→SMILES 级联）──
STANDARDIZE_MODEL = os.getenv("STANDARDIZE_MODEL", CHATGPT_MODEL)
STANDARDIZE_TEMPERATURE = 0.0
STANDARDIZE_USE_OPSIN = True

# ── 模块4 指代消解（ID→SMILES 映射）──
COREF_MODEL = os.getenv("COREF_MODEL", CHATGPT_MODEL)
COREF_TEMPERATURE = 0.0

# ── 路径 ──
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
PROMPTS_DIR = BASE_DIR / "pipeline" / "prompts"

# ── 模块2 图像挖掘：conda 环境 ──
IMGPARSER_CONDA_ENV = os.getenv("IMGPARSER_CONDA_ENV", "imgparser")
PADDLE_CONDA_ENV = os.getenv("PADDLE_CONDA_ENV", "paddle")

# ── 模块2 图像挖掘：模型权重路径（均可通过环境变量覆盖）──
MODELS_DIR = os.getenv("CHEMREX_MODELS_DIR", os.path.expanduser("~/.retro_parser_models"))
RXNSCRIBE_CKPT = os.getenv("RXNSCRIBE_CKPT", os.path.join(MODELS_DIR, "pix2seq_reaction_full.ckpt"))
MOLDETECT_CKPT = os.getenv("MOLDETECT_CKPT", os.path.join(MODELS_DIR, "coref_best_hf.ckpt"))
MOLSCRIBE_CKPT = os.getenv("MOLSCRIBE_CKPT", os.path.join(MODELS_DIR, "swin_base_char_aux_1m.pth"))

# ── 模块5 反应一致性验证 ──
VALIDATION_ENABLED = True
VALIDATION_CONFIDENCE_THRESHOLD = 0.9
VALIDATION_CONFIDENCE_DELTA = 0.2
VALIDATION_AMBIGUITY_MARGIN = 0.05
VALIDATION_MAX_REAGENT_CANDIDATES = 3
# 轻量字段约束检查（字段完整性 + 温度/时间/收率/溶剂/试剂范围检查）
VALIDATION_FIELD_CONSTRAINTS = True

# ── 依赖延迟导入提示 ──
