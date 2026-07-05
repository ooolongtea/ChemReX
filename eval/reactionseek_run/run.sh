#!/usr/bin/env bash
# Run the faithful ReactionSeek pipeline on ChemReX NER data.
#
# Pipeline: extract_gpt-style table prompt -> structurelize -> name_to_smiles
# -> time_standardlize -> ChemReX 7-field JSON.
#
# Usage:
#   bash run.sh                    # full NER set (4166 samples)
#   bash run.sh --limit 20         # smoke test
#   BASE_URL=... API_KEY=... bash run.sh
#   bash run.sh --skip_standardize_time   # disable time normalization

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PY="${SCRIPT_DIR}/run_reactionseek.py"

BASE_URL="${BASE_URL:-http://localhost:8050/v1}"
API_KEY="${API_KEY:-dummy-token}"
MODEL="${MODEL:-Qwen3-8B}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_WORKERS="${MAX_WORKERS:-32}"

# Activate conda environment if available
CONDA_ENV="${CONDA_ENV:-$HOME/conda-envs/retroparser}"
if [[ -f "$CONDA_ENV/bin/activate" ]]; then
  source "$CONDA_ENV/bin/activate" retroparser 2>/dev/null || true
  export PATH="$CONDA_ENV/bin:$PATH"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd "$SCRIPT_DIR"
python3 "$RUN_PY" \
  --base_url "$BASE_URL" \
  --api_key "$API_KEY" \
  --model "$MODEL" \
  --max_tokens "$MAX_TOKENS" \
  --max_workers "$MAX_WORKERS" \
  --temperature 0 \
  "$@"
