# MAReX Training Code

This directory contains the training and offline evaluation code for the
MAReX chemical reaction information extraction model used in ChemReX.
The official training entry is `train.py`.

## Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Data Format](#data-format)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Default Settings](#default-settings)

## Overview

MAReX is trained with a weighted multi-task instruction-tuning objective:

- **NER / reaction-role extraction**: extract reaction product, starting
  material, reagent/catalyst, solvent, temperature, time, and yield.
- **Action-sequence generation**: generate the operation sequence described
  by the reaction procedure.

The training script loads Qwen3-8B with 4-bit NF4 quantization and trains LoRA
adapters on the attention and MLP projection modules.

## Directory Structure

```text
training/
├── README.md
├── train.py                         # official LoRA/QLoRA training entry
├── evaluate.py                      # offline NER + action-sequence evaluation
├── data/
│   ├── meta.json                    # raw-data configuration
│   ├── instructions.md              # instruction templates
│   ├── generate_dataset.py          # train/valid/test split generation
│   └── extract_action_sequence.py   # action-sequence annotation utility
```

Model weights and generated datasets are not included. Prepare `Qwen3-8B` and
`data/seed-67/{train,valid,test}.json` before training.

## Data Format

Each example is a JSON object with the following fields:

```json
{
  "instruction": "Task instruction with the reaction procedure.",
  "input": "Original reaction procedure text.",
  "output": "Target JSON fields or action sequence.",
  "task_type": "NER",
  "language": "EN",
  "source": "dataset_name"
}
```

`task_type` may be `NER`/`ner`/`0` or `Action`/`action`/`1`.

## Installation

Install dependencies in a Python environment with CUDA support:

```bash
pip install torch transformers peft bitsandbytes
pip install nltk bert_score rouge python-Levenshtein
pip install matplotlib numpy
```

## Training

Generate the dataset if needed:

```bash
cd xyq/ChemReX/training
python data/generate_dataset.py \
  --meta data/meta.json \
  --instructions data/instructions.md \
  --output data \
  --seed 67 \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1
```

Run training:

```bash
cd xyq/ChemReX/training
python train.py
```

Edit the `CONFIG` block at the top of `train.py` to change local paths,
GPU ids, or hyperparameters. By default, the LoRA adapter is saved to
`{model_name}-LoRA-{timestamp}/`, and the exact configuration is saved as
`config.json` in the output directory.

## Evaluation

During training, validation metrics are computed by generation:

- NER: per-field precision, recall, and F1; best checkpoint selected by
  `eval_ner_avg_f1`.
- Action sequence: BLEU-4, exact match, and BERTScore F1.

For offline evaluation of exported predictions:

```bash
python evaluate.py --pred predictions.json --ref references.json
```

The evaluator accepts JSON or JSONL files and reports micro-average field F1,
per-field F1, and action-sequence BLEU/ROUGE-L/edit distance.

## Default Settings

| Parameter | Value |
|---|---:|
| Base model | `./Qwen3-8B` |
| Data directory | `./data/seed-67` |
| Epochs | `2` |
| Per-device batch size | `8` |
| Gradient accumulation | `4` |
| Learning rate | `2e-5` |
| Warmup steps | `400` |
| Max sequence length | `1024` |
| Max generated tokens | `512` |
| NER weight | `0.9` |
| Action weight | `0.1` |
| LoRA rank | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| Quantization | 4-bit NF4 |
| Optimizer | `paged_adamw_8bit` |
| Seed | `67` |
