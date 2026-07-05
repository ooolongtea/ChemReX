#!/bin/bash
# 启动 Qwen3-8B + qwen3-lora-08 的 vLLM OpenAI-compatible 服务
# 默认配置对齐 ChemReX M1 抽取预设（config.py::MODEL_PRESETS["qwen3-lora-08"]）
#
# 用法:
#   bash scripts/serve_vllm.sh              # 默认参数启动
#   GPU_ID=0 PORT=8050 bash scripts/serve_vllm.sh
#
# 环境变量:
#   GPU_ID                     指定 GPU ID（默认 0）
#   PORT                       服务端口（默认 8050）
#   GPU_MEMORY_UTILIZATION     显存占用比例（默认 0.75）
#   MAX_MODEL_LEN              最大模型长度（默认 32768）
#   API_KEY                    API 鉴权密钥（默认 dummy-token）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/model/Qwen3-8B}"
LORA_08_PATH="${LORA_08_PATH:-${PROJECT_DIR}/model/qwen3-lora-08}"

PORT="${PORT:-8050}"
GPU_ID="${GPU_ID:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
API_KEY="${API_KEY:-dummy-token}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "错误: 基座模型不存在: ${MODEL_PATH}" >&2
  echo "请先将 Qwen3-8B 下载到 ${PROJECT_DIR}/model/Qwen3-8B，或设置 MODEL_PATH" >&2
  exit 1
fi

if [[ ! -d "${LORA_08_PATH}" ]]; then
  echo "错误: LoRA adapter 不存在: ${LORA_08_PATH}" >&2
  echo "请先将 qwen3-lora-08 下载到 ${PROJECT_DIR}/model/qwen3-lora-08，或设置 LORA_08_PATH" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "启动 vLLM 服务:"
echo "  基座模型: ${MODEL_PATH}"
echo "  LoRA adapter: ${LORA_08_PATH}"
echo "  GPU: ${GPU_ID}, 端口: ${PORT}, API_KEY: ${API_KEY}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "Qwen3-8B" \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enforce-eager \
  --trust-remote-code \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank 16 \
  --lora-modules "qwen3-lora-08=${LORA_08_PATH}" \
  --api-key "${API_KEY}" \
  --uvicorn-log-level info
