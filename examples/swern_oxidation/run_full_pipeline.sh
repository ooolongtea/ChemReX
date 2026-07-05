#!/bin/bash
# ChemReX 全流程驱动脚本（双环境）
#
#   主流程（extraction / standardize / coref / validation）→ retroparser 环境
#   图像模块（image）                          → imgparser 环境
#
# 用法:
#   bash run_full_pipeline.sh <entry.json> [output_dir]
#
# 示例:
#   bash run_full_pipeline.sh input_direct.json
#   bash run_full_pipeline.sh input_coref.json  out_coref
#
# 环境变量（可选）:
#   RETROPARSER_PY   retroparser 环境的 python 路径
#   IMGPARSER_PY     imgparser 环境的 python 路径
#   API_KEY / BASE_URL / EXTRACTION_PRESET / STANDARDIZE_MODEL / COREF_MODEL
#   IMAGE_GPU        指定图像模块使用的 GPU ID（默认自动选择空闲显存最大的卡）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── conda 环境 python 路径 ──
RETROPARSER_PY="${RETROPARSER_PY:-$HOME/conda-envs/retroparser/bin/python}"
IMGPARSER_PY="${IMGPARSER_PY:-$HOME/conda-envs/imgparser/bin/python}"

# ── 模型配置：M1 使用 qwen3-lora-08；M3 默认使用 GPT-4o ──
export EXTRACTION_PRESET="${EXTRACTION_PRESET:-qwen3-lora-08}"
export STANDARDIZE_MODEL="${STANDARDIZE_MODEL:-gpt-4o}"
export COREF_MODEL="${COREF_MODEL:-gpt-4o}"
export API_KEY="${API_KEY:-${CHATGPT_API_KEY:-${OPENAI_API_KEY:-}}}"
export BASE_URL="${BASE_URL:-${CHATGPT_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}}"
VLLM_API_KEY="${VLLM_API_KEY:-token-abc123}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8050/v1}"

# ── SSL 证书（conda Python 编译时 CA 路径是占位符，需显式指向系统 CA bundle）──
# 否则 CIR/PubChem 网络查询会报 CERTIFICATE_VERIFY_FAILED
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"

# ── 参数 ──
ENTRY="${1:?用法: bash run_full_pipeline.sh <entry.json> [output_dir]}"
ENTRY="$(cd "$(dirname "$ENTRY")" && pwd)/$(basename "$ENTRY")"  # 转绝对路径
ENTRY_DIR="$(dirname "$ENTRY")"

INPUT_NAME="$(basename "$ENTRY" .json)"
OUTPUT_DIR="${2:-$SCRIPT_DIR/run_output_${INPUT_NAME}}"
mkdir -p "$OUTPUT_DIR"

# image_base: entry.json 所在目录（其下应有 images/ 子目录）
IMAGE_BASE="$ENTRY_DIR"

echo "=== ChemReX 全流程 ==="
echo "  Entry:        $ENTRY"
echo "  Image base:   $IMAGE_BASE"
echo "  Output dir:   $OUTPUT_DIR"
echo "  retroparser:  $RETROPARSER_PY"
echo "  imgparser:    $IMGPARSER_PY"
echo "  M1 模型预设:  $EXTRACTION_PRESET"
echo "  M3 标准化:    $STANDARDIZE_MODEL"
echo "  M3 Coref:     $COREF_MODEL"
echo ""

# ── 预检: vLLM 服务 ──
if ! curl -sf -m 5 "$VLLM_BASE_URL/models" -H "Authorization: Bearer $VLLM_API_KEY" >/dev/null 2>&1; then
    echo "错误: vLLM 服务不可达 ($VLLM_BASE_URL)，请先启动:" >&2
    echo "  conda activate retroparser && bash scripts/serve_vllm.sh" >&2
    exit 1
fi
echo "  vLLM 服务:    OK"
echo ""

# ============================================================
# Step 1: 图像模块（imgparser 环境）
# ============================================================
echo ">>> [1/2] 图像模块 (imgparser 环境) ..."

if [[ -n "${IMAGE_GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$IMAGE_GPU"
fi

cd "$REPO"
"$IMGPARSER_PY" -m pipeline.image.run_image \
    --image-base "$IMAGE_BASE" \
    --output "$OUTPUT_DIR/image_result.json"

# 从 image_result.json 中提取 dict_result ({id: smiles}) 供 coref 使用
"$RETROPARSER_PY" -c "
import json
d = json.load(open('$OUTPUT_DIR/image_result.json', encoding='utf-8'))
json.dump(d.get('dict_result', {}), open('$OUTPUT_DIR/dict_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('  dict_result:', json.dumps(d.get('dict_result', {}), ensure_ascii=False))
"

echo "    -> $OUTPUT_DIR/image_result.json"
echo "    -> $OUTPUT_DIR/dict_result.json"
echo ""

# ============================================================
# Step 2: 主流程（retroparser 环境）
#         extraction → standardize → coref → validation
# ============================================================
echo ">>> [2/2] 主流程 (retroparser 环境) ..."

# 主流程不占用 GPU（LLM 走 vLLM HTTP），清除 CUDA_VISIBLE_DEVICES 避免误占
unset CUDA_VISIBLE_DEVICES

cd "$REPO"
"$RETROPARSER_PY" "$SCRIPT_DIR/run_main_pipeline.py" \
    --entry "$ENTRY" \
    --dict-result "$OUTPUT_DIR/dict_result.json" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== 全流程完成 ==="
echo "结果目录: $OUTPUT_DIR"
echo "  image_result.json     — M2 图像模块输出 (dict_result + image_result)"
echo "  dict_result.json      — {id: smiles} 提供给 coref"
echo "  1_extraction.json     — M1 NER(7字段) + Action"
echo "  2_standardized.json   — M3a 化合物标准化 (name→SMILES 级联)"
echo "  3_coref.json          — M3b 指代消解 (ID→SMILES 映射)"
echo "  4_validated.json      — M4 反应一致性验证"
