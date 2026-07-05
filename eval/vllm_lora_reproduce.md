# ChemReX vLLM + LoRA 复现实验命令

> 本文档为 `xyq/experiments/vllm_lora_reproduce.md` 的 ChemReX 集成版。
> 实验代码已从 `experiments/code/` 迁移至 `ChemReX/eval/`，数据已复制到 `ChemReX/data/seed-67/`。

目标：用 Qwen3-8B ModelScope 基座和已有 ChemReX/MAReX LoRA adapter，复现 7 个实验。

本流程统一使用 vLLM OpenAI-compatible API，且目标实验统一为 `enable_thinking=False`。LoRA 启动脚本不加 `--enable-reasoning --reasoning-parser deepseek_r1`，避免 reasoning parser 干扰正文解析。

## 0. 路径和模型名

```bash
REPO=/path/to/ChemReX
VLLM_DIR=/path/to/vllm
BASE_MODEL=$HOME/models/Qwen3-8B
DATA=$REPO/data/seed-67/test.json
EVAL_DIR=$REPO/eval
RESULTS_DIR=$REPO/results
VLLM_SCRIPT=$VLLM_DIR/serve_qwen3_vllm_lora.sh
BASE_URL=http://localhost:8050/v1
API_KEY=token-abc123
```

vLLM 服务里的模型名：

| 实验 | external_model |
|---|---|
| base Qwen3-8B | `Qwen3-8B` |
| LoRA 0.6 | `qwen3-lora-06` |
| LoRA 0.7 | `qwen3-lora-07` |
| LoRA 0.8 | `qwen3-lora-08` |
| LoRA 0.9 | `qwen3-lora-09` |
| LoRA single | `qwen3-lora-single` |
| LoRA 0.7 额外运行 | `qwen3-lora-07` |

## 1. 评估代码已支持 vLLM thinking=false + tqdm 进度条

已修改并迁移到：

```text
$REPO/eval/evaluate.py
```

external 模式传：

```python
max_tokens=args.max_new_tokens
extra_body={"chat_template_kwargs": {"enable_thinking": args.enable_thinking}}
```

并用：

```python
msg.content or getattr(msg, "reasoning_content", "") or ""
```

避免 vLLM 返回 `content=None` 时得到空预测。

推理过程带 tqdm 进度条（本地 batch 推理 + 外部 API 并发调用两处）。

默认 `data` 路径已修正为 `../data/seed-67/test.json`（相对 `ChemReX/eval/`），指向本地复制的数据，无需每次传 `--data`。

## 2. 启动 vLLM LoRA 服务

先停止旧服务：

```bash
pgrep -af "vllm.entrypoints.openai.api_server"
pkill -f "vllm.entrypoints.openai.api_server"
```

### 方案 A：显存够，一次挂全部 LoRA

```bash
cd $VLLM_DIR

setsid env \
  GPU_ID=5 \
  PORT=8050 \
  GPU_MEMORY_UTILIZATION=0.85 \
  LORA_SET=all \
  bash serve_qwen3_vllm_lora.sh \
  > logs/vllm_lora_all.log 2>&1 < /dev/null &

tail -f logs/vllm_lora_all.log
```

看到 `Application startup complete` 后继续。

### 方案 B：显存不够，分两轮挂载

第一轮挂 `0.6/0.7/0.8`：

```bash
cd $VLLM_DIR

setsid env \
  GPU_ID=5 \
  PORT=8050 \
  GPU_MEMORY_UTILIZATION=0.75 \
  LORA_SET=core \
  bash serve_qwen3_vllm_lora.sh \
  > logs/vllm_lora_core.log 2>&1 < /dev/null &

tail -f logs/vllm_lora_core.log
```

第二轮挂 `0.9/single`。先停掉第一轮服务：

```bash
pkill -f "vllm.entrypoints.openai.api_server"
sleep 5

cd $VLLM_DIR

setsid env \
  GPU_ID=5 \
  PORT=8050 \
  GPU_MEMORY_UTILIZATION=0.75 \
  LORA_SET=tail \
  bash serve_qwen3_vllm_lora.sh \
  > logs/vllm_lora_tail.log 2>&1 < /dev/null &

tail -f logs/vllm_lora_tail.log
```

也可以只挂单个 LoRA（例如只跑 0.8）：

```bash
LORA_SET=08 bash $VLLM_DIR/serve_qwen3_vllm_lora.sh
```

## 3. 检查服务模型列表

```bash
curl http://localhost:8050/v1/models \
  -H "Authorization: Bearer token-abc123"
```

如果是 `LORA_SET=all`，应能看到：

```text
Qwen3-8B
qwen3-lora-06
qwen3-lora-07
qwen3-lora-08
qwen3-lora-09
qwen3-lora-single
```

## 4. 一键复现（推荐）

`ChemReX/eval/reproduce_lora.sh` 封装了冒烟测试、全量评测、结果汇总，自带服务探测和进度条。

### 4.1 冒烟测试（必做，确认非空预测）

```bash
cd $REPO/eval
SMOKETEST=1 bash reproduce_lora.sh
```

脚本会自动探测服务挂载的模型并各跑 20 条，检查 `predictions.jsonl` 里 `_prediction` 是否非空。

### 4.2 全量复现

```bash
# 跑当前服务挂载的全部模型
bash reproduce_lora.sh

# 或只跑指定模型
bash reproduce_lora.sh qwen3-lora-08 Qwen3-8B

# 自定义服务地址
BASE_URL=http://localhost:8050/v1 API_KEY=token-abc123 bash reproduce_lora.sh
```

结束后会自动打印汇总表（NER-F1 / BLEU-4 / EM / BERTScore）。

## 5. 手动单条命令（等价于脚本内部）

### 5.1 冒烟测试

```bash
source $HOME/conda-envs/retroparser/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $REPO/eval

python evaluate.py \
  --model_type external \
  --external_model Qwen3-8B \
  --base_url http://localhost:8050/v1 \
  --api_key token-abc123 \
  --max_new_tokens 512 \
  --batch_size 16 \
  --max_workers 32 \
  --temperature 0 \
  --limit 20 \
  --save_predictions

python evaluate.py \
  --model_type external \
  --external_model qwen3-lora-08 \
  --base_url http://localhost:8050/v1 \
  --api_key token-abc123 \
  --max_new_tokens 512 \
  --batch_size 12 \
  --max_workers 32 \
  --temperature 0 \
  --limit 20 \
  --save_predictions
```

查看最新结果：

```bash
ls -td $REPO/results/external_* | head
tail -n 3 "$(ls -td $REPO/results/external_* | head -n 1)/predictions.jsonl"
```

### 5.2 全量跑：显存够时一次跑完

```bash
source $HOME/conda-envs/retroparser/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $REPO/eval

DATA=$REPO/data/seed-67/test.json
BASE_URL=http://localhost:8050/v1
API_KEY=token-abc123

run_eval() {
  local model="$1"
  local batch_size="$2"
  python evaluate.py \
    --model_type external \
    --external_model "$model" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --data "$DATA" \
    --max_new_tokens 512 \
    --batch_size "$batch_size" \
    --max_workers 32 \
    --temperature 0 \
    --save_predictions
}

run_eval Qwen3-8B 16
run_eval qwen3-lora-06 12
run_eval qwen3-lora-07 12
run_eval qwen3-lora-08 12
run_eval qwen3-lora-09 12
run_eval qwen3-lora-single 12
run_eval qwen3-lora-07 12
```

最后一次 `qwen3-lora-07` 是实验 13。

### 5.3 全量跑：分两轮挂载时

第一轮服务使用 `LORA_SET=core`，跑 base、0.6、0.7、0.8，以及实验 13 的 0.7 重复：

```bash
source $HOME/conda-envs/retroparser/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $REPO/eval

DATA=$REPO/data/seed-67/test.json
BASE_URL=http://localhost:8050/v1
API_KEY=token-abc123

run_eval() {
  local model="$1"
  local batch_size="$2"
  python evaluate.py \
    --model_type external \
    --external_model "$model" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --data "$DATA" \
    --max_new_tokens 512 \
    --batch_size "$batch_size" \
    --max_workers 32 \
    --temperature 0 \
    --save_predictions
}

run_eval Qwen3-8B 16
run_eval qwen3-lora-06 12
run_eval qwen3-lora-07 12
run_eval qwen3-lora-08 12
run_eval qwen3-lora-07 12
```

第二轮服务使用 `LORA_SET=tail`，跑 0.9 和 single：

```bash
run_eval qwen3-lora-09 12
run_eval qwen3-lora-single 12
```

## 6. 结果位置和检查

每次运行会生成新目录：

```text
$REPO/results/external_<model>_<timestamp>/
```

核心文件：

```text
config.json
evaluate.log
eval_metrics.json
predictions.jsonl
```

列出最新结果：

```bash
ls -td $REPO/results/external_* | head -n 20
```

快速看指标：

```bash
python - <<'PY'
import json
from pathlib import Path

for d in sorted(Path('$REPO/results').glob('external_*'), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
    mpath = d / 'eval_metrics.json'
    if not mpath.exists():
        continue
    m = json.loads(mpath.read_text())
    ner = m.get('ner', {}).get('avg', {})
    action = m.get('action', {})
    print(d.name)
    print('  total:', m.get('total_samples'), 'counts:', m.get('counts'))
    print('  NER F1:', ner.get('f1'))
    print('  Action BLEU-4:', action.get('bleu_4'), 'EM:', action.get('em'), 'BERTScore-F1:', action.get('bertscore_f1'))
PY
```

检查是否有空预测：

```bash
python - <<'PY'
import json
from pathlib import Path

latest = sorted(Path('$REPO/results').glob('external_*'), key=lambda p: p.stat().st_mtime, reverse=True)[0]
pred_path = latest / 'predictions.jsonl'
empty = 0
total = 0
with pred_path.open() as f:
    for line in f:
        total += 1
        row = json.loads(line)
        if not (row.get('_prediction') or '').strip():
            empty += 1
print(latest)
print('empty_predictions:', empty, 'total:', total)
PY
```

## 7. 常见处理

如果启动时报显存不足，按顺序尝试：

```bash
GPU_MEMORY_UTILIZATION=0.75 LORA_SET=core bash $VLLM_DIR/serve_qwen3_vllm_lora.sh
GPU_MEMORY_UTILIZATION=0.65 LORA_SET=core bash $VLLM_DIR/serve_qwen3_vllm_lora.sh
GPU_MEMORY_UTILIZATION=0.65 LORA_SET=08 bash $VLLM_DIR/serve_qwen3_vllm_lora.sh
```

如果评估脚本报 `ModuleNotFoundError: torch`，说明没有进环境：

```bash
source $HOME/conda-envs/retroparser/bin/activate
```

如果指标接近 0，先查：

```bash
tail -n 3 "$(ls -td $REPO/results/external_* | head -n 1)/predictions.jsonl"
```

若 `_prediction` 为空，优先确认使用的是 `ChemReX/eval/evaluate.py`，且命令里没有传 `--enable_thinking`。
