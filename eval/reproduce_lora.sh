#!/usr/bin/env bash
# reproduce_lora.sh — ChemReX LoRA 复现实验脚本
#
# 用法:
#   bash reproduce_lora.sh                 # 跑当前服务挂载的全部模型（自动探测）
#   bash reproduce_lora.sh Qwen3-8B        # 只跑指定模型
#   bash reproduce_lora.sh qwen3-lora-08 Qwen3-8B   # 跑多个指定模型
#   SMOKETEST=1 bash reproduce_lora.sh     # 只跑 --limit 20 冒烟测试
#   BASE_URL=http://localhost:8050/v1 API_KEY=dummy-token bash reproduce_lora.sh
#
# 环境变量:
#   BASE_URL    vLLM 服务地址        (默认 http://localhost:8050/v1)
#   API_KEY     API key              (默认 dummy-token)
#   DATA        测试集路径           (默认 ../data/seed-67/test.json)
#   MAX_TOKENS  最大生成 token       (默认 512)
#   WORKERS     并发数               (默认 32)
#   BASE_BATCH  基座模型 batch       (默认 16)
#   LORA_BATCH  LoRA 模型 batch      (默认 12)
#   SMOKETEST   =1 只跑 20 条冒烟    (默认 0)

set -euo pipefail

# ========== 路径与默认配置 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/evaluate.py"

BASE_URL="${BASE_URL:-http://localhost:8050/v1}"
API_KEY="${API_KEY:-dummy-token}"
DATA="${DATA:-${SCRIPT_DIR}/../data/seed-67/test.json}"
MAX_TOKENS="${MAX_TOKENS:-512}"
WORKERS="${WORKERS:-32}"
BASE_BATCH="${BASE_BATCH:-16}"
LORA_BATCH="${LORA_BATCH:-12}"
SMOKETEST="${SMOKETEST:-0}"

# ========== 颜色输出 ==========
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${GREEN}[reproduce]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[error]${NC} $*" >&2; }
step() { echo -e "\n${BLUE}========== $* ==========${NC}"; }

# ========== 前置检查 ==========
step "前置检查"

if [[ ! -f "$EVAL_PY" ]]; then
  err "找不到评估脚本: $EVAL_PY"
  exit 1
fi

if [[ ! -f "$DATA" ]]; then
  err "找不到测试集: $DATA"
  exit 1
fi
log "测试集: $DATA ($(du -h "$DATA" | cut -f1))"

# 激活 conda 环境（非交互式，兼容 tmux/nohup）
CONDA_ENV="${CONDA_ENV:-$HOME/conda-envs/retroparser}"
if [[ -f "$CONDA_ENV/bin/activate" ]]; then
  source "$CONDA_ENV/bin/activate" retroparser 2>/dev/null
  # 兜底：直接把环境 bin 加入 PATH，确保 python 指向环境内
  export PATH="$CONDA_ENV/bin:$PATH"
  log "已激活 conda 环境: retroparser  (python=$(which python))"
else
  warn "未找到 conda 环境 $CONDA_ENV，使用系统 python"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 确认关键依赖
if ! python -c "import torch" 2>/dev/null; then
  err "python 无法导入 torch: $(which python)"
  err "请检查 conda 环境: $CONDA_ENV"
  exit 1
fi

# 检查 vLLM 服务
log "检查 vLLM 服务: $BASE_URL"
if ! curl -sf "${BASE_URL}/models" -H "Authorization: Bearer ${API_KEY}" >/dev/null 2>&1; then
  err "vLLM 服务无响应: $BASE_URL"
  err "请先启动服务，参考 serve_qwen3_vllm_lora.sh"
  exit 1
fi

# 探测服务挂载的模型
SERVED_MODELS=$(curl -sf "${BASE_URL}/models" -H "Authorization: Bearer ${API_KEY}" \
  | python3 -c "import sys,json; print('\n'.join(m['id'] for m in json.load(sys.stdin)['data']))")
log "服务可用，已挂载模型:\n${SERVED_MODELS}"

# ========== 确定要跑的模型 ==========
if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=($SERVED_MODELS)
fi

log "本次将评测模型: ${MODELS[*]}"

# 冒烟测试参数
LIMIT_ARG=""
if [[ "$SMOKETEST" == "1" ]]; then
  LIMIT_ARG="--limit 20"
  log "冒烟模式: 仅跑 20 条样本"
fi

# ========== 单模型评测函数 ==========
run_eval() {
  local model="$1"
  local batch_size
  # 基座模型用大 batch，LoRA 用小 batch
  if [[ "$model" == "Qwen3-8B" || "$model" == "qwen3-8b" ]]; then
    batch_size="$BASE_BATCH"
  else
    batch_size="$LORA_BATCH"
  fi

  step "评测模型: $model  (batch=$batch_size, workers=$WORKERS)"
  python3 "$EVAL_PY" \
    --model_type external \
    --external_model "$model" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --data "$DATA" \
    --max_new_tokens "$MAX_TOKENS" \
    --batch_size "$batch_size" \
    --max_workers "$WORKERS" \
    --temperature 0 \
    $LIMIT_ARG \
    --save_predictions
}

# ========== 冒烟测试 ==========
if [[ "$SMOKETEST" == "1" ]]; then
  step "冒烟测试 (limit=20)"
  for model in "${MODELS[@]}"; do
    run_eval "$model"
  done

  step "冒烟测试结果检查"
  LATEST=$(ls -td "${SCRIPT_DIR}/../results/external_"* 2>/dev/null | head -1)
  if [[ -n "$LATEST" && -f "${LATEST}/predictions.jsonl" ]]; then
    log "最新结果目录: $LATEST"
    python3 - "$LATEST/predictions.jsonl" <<'PY'
import json, sys
path = sys.argv[1]
empty = total = 0
with open(path) as f:
    for line in f:
        total += 1
        if not (json.loads(line).get('_prediction') or '').strip():
            empty += 1
print(f"  total={total}  empty_predictions={empty}")
PY
  fi
  log "冒烟测试完成。确认 _prediction 非空后，去掉 SMOKETEST=1 跑全量。"
  exit 0
fi

# ========== 全量评测 ==========
step "全量评测开始 (${#MODELS[@]} 个模型)"
declare -A RESULTS_DIR
for model in "${MODELS[@]}"; do
  run_eval "$model"
  # 记录最新结果目录
  RESULTS_DIR["$model"]=$(ls -td "${SCRIPT_DIR}/../results/external_${model}_"* 2>/dev/null | head -1)
done

# ========== 汇总结果 ==========
step "评测结果汇总"
python3 - <<PY
import json
from pathlib import Path

results_dir = Path("${SCRIPT_DIR}/../results")
models = """${MODELS[*]}""".split()
print(f"{'模型':<22} {'样本数':>6} {'NER-F1':>8} {'BLEU-4':>8} {'EM':>8} {'BERTScore':>10}")
print("-" * 70)
for model in models:
    dirs = sorted(results_dir.glob(f"external_{model}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        print(f"{model:<22}  (无结果)")
        continue
    mpath = dirs[0] / "eval_metrics.json"
    if not mpath.exists():
        print(f"{model:<22}  (无 eval_metrics.json)  dir={dirs[0].name}")
        continue
    m = json.loads(mpath.read_text())
    ner = m.get('ner', {}).get('avg', {})
    action = m.get('action', {})
    print(f"{model:<22} {m.get('total_samples','?'):>6} "
          f"{ner.get('f1',0)*100 if ner.get('f1') else 0:>7.2f}% "
          f"{action.get('bleu_4',0)*100 if action.get('bleu_4') else 0:>7.2f}% "
          f"{action.get('em',0)*100 if action.get('em') else 0:>7.2f}% "
          f"{action.get('bertscore_f1',0)*100 if action.get('bertscore_f1') else 0:>9.2f}%")
    print(f"  └─ {dirs[0].name}")
PY

log "全部完成。结果目录: ${SCRIPT_DIR}/../results/"
