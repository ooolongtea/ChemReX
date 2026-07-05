# ChemReX Evaluation Results Summary

Generated: 2026-07-04 12:33

Evaluator: `eval/eval.py` (canonical, unified metric logic)

Environment: `retroparser` (nltk + torch + bert-score 0.3.13 available)

BERTScore model: `FacebookAI/xlm-roberta-large`, num_layers=17 (multilingual, replaces `bert-base-chinese` which produced invalid scores for English)

## Overall NER & Action Metrics

| Model | NER Avg F1 (%) | EN Avg F1 (%) | ZH Avg F1 (%) | Action BLEU-4 (%) | Action EM (%) | BERTScore F1 (%) |
|-------|---------------:|--------------:|--------------:|------------------:|--------------:|-----------------:|
| ChemLLM-7B-SFT | 66.47 | 61.19 | 71.41 | 24.80 | 5.06 | 90.23 |
| ChemDFM-v1.5-8B | 67.72 | 70.72 | 64.02 | 17.35 | 0.29 | 86.73 |
| Qwen3-8B Base | 76.76 | 75.78 | 77.34 | 40.67 | 13.42 | 93.36 |
| DeepSeek-V3 | 73.88 | 71.60 | 75.87 | 45.37 | 26.24 | 92.95 |
| DeepSeek-V4-Pro | 72.77 | 73.10 | 71.93 | 35.58 | 12.65 | 94.00 |
| GPT-4o | 74.52 | 74.13 | 74.37 | 43.18 | 21.23 | 93.17 |
| Qwen-Plus | 65.16 | 61.23 | 68.89 | 36.69 | 18.61 | 91.56 |
| Qwen3-8B LoRA | 92.86 | 93.85 | 91.53 | 83.24 | 55.20 | 98.86 |
| Qwen3-8B + ReactionSeek | 82.61 | 84.23 | 80.32 | - | - | - |
| Qwen3-8B LoRA + ReactionSeek | 91.78 | 93.04 | 90.17 | - | - | - |

## Per-Field NER F1 (Overall)

| Model | Product | Starting Material | Reagent/Catalyst | Solvent | Temperature | Time | Yield |
|---|---|---|---|---|---|---|---|
| ChemLLM-7B-SFT | 73.70 | 67.20 | 50.26 | 71.96 | 51.09 | 73.74 | 77.30 |
| ChemDFM-v1.5-8B | 77.04 | 56.00 | 51.74 | 71.41 | 62.30 | 75.43 | 80.14 |
| Qwen3-8B Base | 84.92 | 78.97 | 68.52 | 74.38 | 71.67 | 77.63 | 81.24 |
| DeepSeek-V3 | 87.25 | 79.28 | 46.28 | 69.00 | 68.86 | 71.86 | 94.62 |
| DeepSeek-V4-Pro | 89.14 | 65.16 | 56.69 | 74.49 | 59.13 | 66.82 | 97.95 |
| GPT-4o | 87.33 | 78.47 | 60.88 | 74.46 | 67.66 | 67.04 | 85.81 |
| Qwen-Plus | 88.94 | 71.41 | 57.93 | 74.45 | 57.38 | 63.24 | 42.79 |
| Qwen3-8B LoRA | 94.72 | 94.40 | 87.57 | 93.03 | 91.04 | 90.41 | 98.87 |
| Qwen3-8B + ReactionSeek | 85.44 | 85.65 | 71.37 | 83.98 | 74.77 | 83.99 | 93.04 |
| Qwen3-8B LoRA + ReactionSeek | 93.68 | 93.37 | 86.15 | 92.64 | 88.12 | 89.66 | 98.85 |

## Per-Field NER F1 by Language (EN / ZH)

### ChemLLM-7B-SFT

| Field | EN F1 (%) | ZH F1 (%) |
|-------|----------:|----------:|
| Product | 75.62 | 71.26 |
| Starting Material | 67.96 | 66.21 |
| Reagent/Catalyst | 49.86 | 50.67 |
| Solvent | 68.90 | 75.12 |
| Temperature | 29.99 | 72.13 |
| Time | 66.49 | 80.91 |
| Yield | 69.51 | 83.55 |

### ChemDFM-v1.5-8B

| Field | EN F1 (%) | ZH F1 (%) |
|-------|----------:|----------:|
| Product | 84.22 | 66.82 |
| Starting Material | 72.53 | 37.40 |
| Reagent/Catalyst | 58.96 | 44.39 |
| Solvent | 72.61 | 70.08 |
| Temperature | 54.01 | 71.13 |
| Time | 74.52 | 76.40 |
| Yield | 78.16 | 81.93 |

### Qwen3-8B Base

| Field | EN F1 (%) | ZH F1 (%) |
|-------|----------:|----------:|
| Product | 92.33 | 74.47 |
| Starting Material | 80.59 | 76.73 |
| Reagent/Catalyst | 66.56 | 70.56 |
| Solvent | 71.33 | 77.98 |
| Temperature | 66.59 | 76.99 |
| Time | 75.18 | 80.28 |
| Yield | 77.86 | 84.38 |

### DeepSeek-V3

| Field | EN F1 (%) | ZH F1 (%) |
|-------|----------:|----------:|
| Product | 94.02 | 77.80 |
| Starting Material | 82.98 | 74.15 |
| Reagent/Catalyst | 37.46 | 55.58 |
| Solvent | 64.72 | 73.86 |
| Temperature | 64.39 | 73.69 |
| Time | 64.57 | 80.00 |
| Yield | 93.06 | 96.00 |

### DeepSeek-V4-Pro

| Field | EN F1 (%) | ZH F1 (%) |
|-------|----------:|----------:|
| Product | 95.74 | 79.94 |
| Starting Material | 70.06 | 58.52 |
| Reagent/Catalyst | 59.06 | 54.02 |
| Solvent | 75.72 | 73.14 |
| Temperature | 52.00 | 66.84 |
| Time | 61.05 | 73.23 |
| Yield | 98.10 | 97.82 |

### GPT-4o

| Field | EN F1 (%) | ZH F1 (%) |
|-------|----------:|----------:|
| Product | 94.14 | 77.73 |
| Starting Material | 82.72 | 72.61 |
| Reagent/Catalyst | 59.91 | 61.89 |
| Solvent | 71.42 | 77.89 |
| Temperature | 66.21 | 69.18 |
| Time | 62.84 | 71.60 |
| Yield | 81.69 | 89.66 |

### Qwen-Plus

| Field | EN F1 (%) | ZH F1 (%) |
|-------|----------:|----------:|
| Product | 94.29 | 81.50 |
| Starting Material | 74.34 | 67.51 |
| Reagent/Catalyst | 52.44 | 63.95 |
| Solvent | 72.89 | 76.14 |
| Temperature | 49.60 | 66.05 |
| Time | 54.47 | 72.93 |
| Yield | 30.57 | 54.17 |

### Qwen3-8B LoRA

| Field | EN F1 (%) | ZH F1 (%) |
|-------|----------:|----------:|
| Product | 98.31 | 89.70 |
| Starting Material | 96.46 | 91.52 |
| Reagent/Catalyst | 87.85 | 87.28 |
| Solvent | 93.71 | 92.27 |
| Temperature | 91.87 | 90.04 |
| Time | 89.77 | 91.17 |
| Yield | 99.01 | 98.75 |

### Qwen3-8B + ReactionSeek

| Field | EN F1 (%) | ZH F1 (%) |
|-------|----------:|----------:|
| Product | 92.47 | 75.47 |
| Starting Material | 88.97 | 81.06 |
| Reagent/Catalyst | 71.07 | 71.67 |
| Solvent | 83.16 | 84.94 |
| Temperature | 79.28 | 69.85 |
| Time | 83.43 | 84.60 |
| Yield | 91.24 | 94.64 |

### Qwen3-8B LoRA + ReactionSeek

| Field | EN F1 (%) | ZH F1 (%) |
|-------|----------:|----------:|
| Product | 97.40 | 88.46 |
| Starting Material | 95.79 | 90.08 |
| Reagent/Catalyst | 87.31 | 84.93 |
| Solvent | 93.81 | 91.33 |
| Temperature | 89.43 | 86.64 |
| Time | 88.29 | 91.25 |
| Yield | 99.23 | 98.52 |

## Radar Chart

- PNG: `eval/radar_en_zh_comparison_all.png`
- SVG: `eval/radar_en_zh_comparison_all.svg`
