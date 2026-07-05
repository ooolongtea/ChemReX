# ChemReX: Validation-Aware Reaction-Record Construction from Chemical Patent Literature

ChemReX is a modular system for extracting structured chemical reaction records from clean experimental paragraphs and associated reaction images. It combines text extraction, optional image mining, compound standardization/coreference, and conservative reaction validation.

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Usage](#3-usage)
4. [Evaluation](#4-evaluation)
5. [Citation](#5-citation)

## 1. Overview

ChemReX takes a clean experimental paragraph with a provided language tag (`zh` or `en`) as input. It does not perform PDF parsing, layout analysis, language detection, or automatic document chunking.

The pipeline contains four main modules:

- **M1 Text extraction**: extracts seven NER fields and an action sequence from reaction text.
- **M2 Image mining**: extracts an image-side `ID -> SMILES` dictionary and reaction structures from reference images.
- **M3 Coreference and standardization**: standardizes compound names and maps text identifiers to image-side SMILES.
- **M4 Reaction validation**: applies field constraints and RXNMapper-based consistency checks with conservative repairs.

The seven NER fields are:

`REACTION_PRODUCT`, `STARTING_MATERIAL`, `REAGENT_CATALYST`, `SOLVENT`, `TEMPERATURE`, `TIME`, and `YIELD_PERCENT`.

## 2. Installation

ChemReX uses a lightweight text environment for M1/M3/M4 and a separate image environment for M2. The image module is optional if only text extraction or offline evaluation is needed.

### 2.1 Text Environment

Python 3.10 is recommended.

```bash
conda create -n retroparser python=3.10 -y
conda activate retroparser

# Install PyTorch according to your CUDA version. Example for CUDA 12.4:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

Copy the environment template and configure API keys if using external OpenAI-compatible models:

```bash
cp .env.example .env
```

Note: copying `.env.example` may set `EXTRACTION_PRESET=deepseek-v4-pro`. For the local Qwen3-8B + LoRA setup, set `EXTRACTION_PRESET=qwen3-lora-08` or omit this variable.

### 2.2 Default Local Model for M1

The default M1 preset is `qwen3-lora-08`, served by a local vLLM OpenAI-compatible server. The LoRA weights will be released upon acceptance. Place the base model and LoRA adapter under:

```text
model/
├── Qwen3-8B/
└── qwen3-lora-08/
```

Install vLLM in the text environment or a separate deployment environment:

```bash
pip install vllm==0.8.5 transformers
```

Start the service:

```bash
bash scripts/serve_vllm.sh
```

The default service endpoint is `http://localhost:8050/v1`, model id `qwen3-lora-08`, API key `dummy-token`.

Alternatively, use an external OpenAI-compatible model:

```bash
python run_pipeline.py extract \
  --text "..." \
  --task both \
  --language en \
  --preset deepseek-v4-pro
```

### 2.3 Image Environment

M2 requires a separate Python 3.9 environment because the OCSR and detection stacks are version-sensitive.

```bash
conda create -n imgparser python=3.9 -y
conda activate imgparser

pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-imgparser.txt
```

PaddleOCR is invoked through a separate conda environment named `paddle`:

```bash
conda create -n paddle python=3.9 -y
conda activate paddle

python -m pip install paddlepaddle-gpu==3.3.1.post118 \
  -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install paddleocr==3.5.0 paddlex==3.5.0
pip install opencv-contrib-python shapely pyclipper
```

Image model weights are configured by environment variables:


| Variable         | Default file name            | Purpose                     |
| ---------------- | ---------------------------- | --------------------------- |
| `MOLSCRIBE_CKPT` | `swin_base_char_aux_1m.pth`  | molecule image to SMILES    |
| `MOLDETECT_CKPT` | `coref_best_hf.ckpt`         | molecule/label detection    |
| `RXNSCRIBE_CKPT` | `pix2seq_reaction_full.ckpt` | reaction scheme recognition |

You may also set `CHEMREX_MODELS_DIR` to a directory containing these three files.

## 3. Usage

The main entry point is `run_pipeline.py`. Each module can be run independently; `chain` runs selected modules on a single text segment.

### 3.1 Text Extraction

Single example:

```bash
python run_pipeline.py extract \
  --text "To a solution of 1a in CH2Cl2 was added DMSO at -78 °C..." \
  --task both \
  --language en
```

Batch JSONL input:

```bash
python run_pipeline.py extract \
  --input data/seed-67-grouped/test.jsonl \
  --output results/m1_predictions.jsonl \
  --max-workers 32
```

Each JSONL row should contain `text` or `input`, plus `language`. If `task_type` is present, it is used to infer `ner` or `action`; otherwise both tasks are run.

### 3.2 Image Mining

```bash
python run_pipeline.py image --image-base path/to/image_directory
```

The output contains `dict_result` for identifier-to-SMILES mappings and `image_result` for reaction structures.

### 3.3 Coreference and Standardization

```bash
python run_pipeline.py coref \
  --ner-result extraction.json \
  --dict-result dict_result.json \
  --output coref.json
```

For standalone compound standardization:

```bash
python run_pipeline.py standardize \
  --input extraction.json \
  --output standardized.json
```

### 3.4 Reaction Validation

```bash
python run_pipeline.py validate \
  --ner-result coref.json \
  --output validated.json
```

### 3.5 Chained Run

```bash
python run_pipeline.py chain \
  --text "..." \
  --language en \
  --chain extract+image+coref+validate \
  --image-base path/to/image_directory \
  --output result.json
```

If an optional input is missing, the corresponding module is skipped with a warning.

## 4. Evaluation

ChemReX includes two evaluation settings: NER+Action extraction and end-to-end grounding.

### 4.1 NER + Action Evaluation

Run model inference and compute metrics:

```bash
python eval/evaluate.py \
  --model_type lora \
  --base_model_path model/Qwen3-8B \
  --adapter_path model/qwen3-lora-08 \
  --data data/seed-67-grouped/test.json \
  --save_predictions
```

Evaluate an external OpenAI-compatible model:

```bash
python eval/evaluate.py \
  --model_type external \
  --external_model deepseek-v4-pro \
  --data data/seed-67-grouped/test.json \
  --save_predictions
```

Recompute metrics from an existing prediction file:

```bash
python eval/verify_metrics.py \
  results/<experiment>/predictions.jsonl \
  --data data/seed-67-grouped/test.json
```

The main metrics are:

- **NER**: micro F1 and per-field F1 over the seven reaction fields.
- **Action**: BLEU-4, ROUGE-L, exact match, and BERTScore.

### 4.2 End-to-End Grounding Evaluation

The end-to-end evaluation measures the full grounding chain:

```text
text identifier -> figure label -> molecule box -> OCSR SMILES
```

Run the full grounding evaluation:

```bash
python eval/end2end/run_grounding_eval.py \
  --sidecar data/raw/openexp_500_coref_sidecar.json \
  --eval data/raw/openexp_500_eval.json \
  --image-dir data/raw/reference_sheets \
  --m2 data/outputs/m2/per_image_detections.jsonl \
  --m1 data/outputs/m1/m1_predictions.jsonl \
  --m3 data/outputs/m3/m3_coref.jsonl \
  --text-only
```

Outputs are written to `eval/end2end/outputs/`:

- `grounding_eval_summary.csv`
- `grounding_eval_per_sample.jsonl`
- `grounding_eval_errors.jsonl`

The primary end-to-end metrics are identifier-to-SMILES grounding F1 and strict end-to-end grounding accuracy. Additional diagnostics include text identifier recognition, figure label OCR, molecule box detection, OCSR exact match, link F1, and error-type counts.

## 5. Citation

```bibtex
@inproceedings{chemrex2025,
  title={ChemReX: Chemical Reaction Information Extraction from Literature},
  author={...},
  booktitle={...},
  year={2025}
}
```
