# ChemReX Grounding Evaluation — Experiment Report

## 1. Experiment Setup

### 1.1 Goal

Evaluate the full grounding chain of ChemReX:

```
text identifier → figure label → molecule box → OCSR SMILES
```

即：给定反应文本中的标识符（如 "1a"、"compound 3a"、"<2a>"），系统能否正确定位到参考图上对应的 figure label、molecule box，并输出正确的 SMILES。

### 1.2 Dataset

- **来源**: OpenExp_500 — 500 条化学反应记录，每条包含反应文本、提取的分子列表（ID→SMILES）、反应条件等。
- **参考图**: 为每条记录生成一张 reference sheet（PNG），将所有参与反应的分子结构绘制在图上，并标注 label（如 "1a"、"2a"）。
- **引用风格**: 3 种 placeholder 风格，各 ~167 条：
  - **dollar**: 文本中标识符为裸形式，如 `1a (4.0 g, 28.1 mmol)`
  - **compound**: 文本中带前缀，如 `compound 3a (25.3 g, 100 mmol)`
  - **angle**: 文本中带尖括号，如 `<1a> (19.8 g, 0.10 mol)`
- **Gold 标识符**: 共 1640 个（来自 dict_result，即 ID→SMILES 映射），平均每条 ~3.28 个。
- **文本中出现的标识符**: 1085 个（66.2%）。其余 555 个（主要为产物，文本中写为 "title compound" / "the product"）不以代码形式出现，无法通过文本链路 grounding。

### 1.3 Pipeline 架构


| 模块          | 功能                                                                               | 模型/方法                                                            | 环境             |
| ------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ---------------- |
| M1 Extraction | 从反应文本中抽取 NER（反应物/产物/试剂/溶剂/温度/时间/收率）+ Action               | Qwen3-8B + LoRA (adapter 0.8), vLLM 推理                             | localhost:8050   |
| M2 Image      | 检测参考图上的 molecule box + figure label，OCSR 输出 SMILES，建立 label→mol 链接 | RxnScribe (Pix2Seq 检测) + MolScribe (OCSR) + PaddleOCR (label 文本) | imgparser, GPU 6 |
| M3 Coref      | 将 M1 文本标识符与 M2 图像 dict_result 匹配，输出 ID→SMILES 映射                  | 直接归一化匹配（无需 LLM）                                           | retroparser      |
| M4 Validation | 字段约束检查 + RXNMapper 反应映射筛查                                              | RDKit + RXNMapper                                                    | retroparser      |

### 1.4 评测指标

共 11 类指标，覆盖 grounding 链路的每个环节：


| #  | 指标                              | 说明                                                                                    |
| -- | --------------------------------- | --------------------------------------------------------------------------------------- |
| 1  | 标识符归一化                      | lowercase, 去前缀 (compound/cmpd/product/...), 去标点 (.:-_<>[]), 后缀小写              |
| 2  | SMILES 归一化                     | RDKit canonical (strict isomeric + relaxed non-isomeric), 多片段排序                    |
| 3  | 文本标识符识别                    | identifier-set P/R/F1 (归一化后集合交并)                                                |
| 4  | Figure label OCR                  | IoU≥0.5 匹配 + center-point 回退; normalized label exact match + CER                   |
| 5  | 分子框检测                        | IoU≥0.5 贪心匹配; P/R/F1                                                               |
| 6  | OCSR                              | 仅对匹配的分子框评价: RDKit valid rate, strict/relaxed canonical exact match            |
| 7  | Label→mol 链接                   | pred link 正确当且仅当 label 和 mol 均正确匹配且 gold 中存在该链接                      |
| 8  | ID→label 链接                    | pred link 正确当且仅当 normalized text ID == normalized figure label 且 gold 中存在     |
| 9  | **ID→SMILES grounding (主指标)** | pred pair 正确当且仅当 normalized ID 匹配 + SMILES valid + strict canonical 匹配        |
| 10 | 端到端 accuracy                   | 全链路通过 (text ID + label + link + molbox + valid SMILES + canonical 匹配) / gold IDs |
| 11 | 错误分析                          | 每个 failed gold ID 归因一个主错误类型                                                  |

### 1.5 评测配置

- **IoU 阈值**: 分子框 0.5; label 框 0.5 + center-point 回退（因 OCR 检测框比渲染文本框略大，IoU 普遍 ~0.39）
- **Gold bbox 处理**: 原始 gold bbox 为完整 tile (210×172px)，包含分子结构 + RDKit 白边 + label 文字区域。评测时裁掉 label 区域并 trim 白边，使 gold bbox 紧贴分子结构，与检测器输出语义一致。
- **评测范围**: 仅评估文本中以代码形式出现的标识符（1085 个，占 gold 标识符的 66.2%）。其余标识符主要为产物，在文本中写为 "title compound" / "the product"，不以代码出现，不属于 grounding 评价范围。

---

## 2. Results

### 2.1 Summary


| 指标                 | Text-only (1085 IDs) |
| -------------------- | :------------------: |
| n_samples            |         500         |
| n_gold_identifiers   |         1085         |
| text_id_p            |        0.8940        |
| text_id_r            |        0.9641        |
| text_id_f1           |        0.9277        |
| label_det_p          |        1.0000        |
| label_det_r          |        0.9853        |
| label_det_f1         |        0.9926        |
| label_ocr_em         |        1.0000        |
| label_cer            |        0.0000        |
| molbox_p             |        0.9972        |
| molbox_r             |        0.9825        |
| molbox_f1            |        0.9898        |
| ocsr_valid_rate      |        0.9764        |
| ocsr_strict_exact    |        0.7988        |
| ocsr_relaxed_exact   |        0.9136        |
| link_p               |        0.9972        |
| link_r               |        0.9825        |
| link_f1              |        0.9898        |
| grounding_p          |        0.8136        |
| grounding_r          |        0.7724        |
| **grounding_f1**     |      **0.7924**      |
| grounding_relaxed_f1 |        0.8128        |
| **e2e_acc**          |      **0.7714**      |

### 2.2 E2E 漏斗分析 (Text-only, 1085 IDs)

```
1085 gold 标识符
  │
  ├─ TEXT_ID_MISS:        39 (3.6%)     ← M1 遗漏少量标识符
  └─ 1046 通过文本识别 (96.4%)
       │
       ├─ LABEL_MISS:     16 (1.5%)     ← M2 label 检测遗漏
       └─ 1030 通过 label 检测 (98.5%)
            │
            ├─ MOLBOX_MISS:  3 (0.3%)   ← M2 分子框检测遗漏
            └─ 1027 通过分子框检测 (99.7%)
                 │
                 ├─ OCSR_INVALID:  18 (1.8%)  ← SMILES 无法解析
                 ├─ OCSR_MISMATCH: 172 (16.7%) ← 立体化学/结构差异
                 └─ 837 全链路通过 (81.5%)
                      │
                      └─ E2E accuracy = 837/1085 = 77.1%
```

### 2.3 Per-Style 错误分布


| 错误类型         | dollar | compound |  angle  |  合计  |
| ---------------- | :-----: | :------: | :-----: | :-----: |
| OCSR_MISMATCH    |   56   |    53    |   63   |   172   |
| TEXT_ID_SPURIOUS |   35   |    44    |   45   |   124   |
| TEXT_ID_MISS     |   11   |    7    |   21   |   39   |
| OCSR_INVALID     |    4    |    8    |    6    |   18   |
| LABEL_MISS       |    8    |    6    |    2    |   16   |
| MOLBOX_MISS      |    0    |    1    |    2    |    3    |
| **合计**         | **114** | **119** | **139** | **372** |

---

## 3. Analysis

### 3.1 各环节表现

**文本标识符识别 (M1, F1=0.93)**

- M1 (Qwen3-8B + LoRA) 对三种引用风格的标识符识别表现优秀，recall 96.4%。
- 主要遗漏来自 angle 风格（21 miss vs dollar 11 vs compound 7），可能因为 `<>` 标记增加了 LLM 的识别难度。
- SPURIOUS（124 个）主要是 M1 将非占位符化合物名（如 "E1"、"D2"）误识别为标识符。

**Figure label 检测与 OCR (M2, F1=0.99, EM=1.00)**

- M2 的 label 检测近乎完美：500 张图共 1619 个 gold label，检测到 1605 个 (R=98.7%)，无误检 (P=100%)。
- 所有匹配的 label OCR 文本 100% 精确匹配，CER=0。

**分子框检测 (M2, F1=0.99)**

- Gold bbox 修正后（裁掉 label 区域 + trim 白边），分子框检测 F1 从 0.68 提升至 0.99。
- 仅有 3 个分子框未检测到，2 个误检。

**OCSR (M2, valid=97.6%, strict=79.9%, relaxed=91.4%)**

- SMILES 有效率 97.6%，仅 18 个无效。
- 严格同构匹配 79.9%，非同构匹配 91.4%，差距 ~11pp 主要来自立体化学标注差异。
- 典型 OCSR_MISMATCH 案例：
  - **立体化学过度标注**: gold `COP(=O)(CC(C)=O)OC` → pred `CO[P@SP3](=O)(CC(C)=O)OC`（MolScribe 额外标注了磷立体化学）
  - **立体化学遗漏**: gold `CS(=O)(=O)Nc1ccc(B(O)O)cc1` → pred `C[S@SP3](=O)(=O)Nc1ccc(B(O)O)cc1`
  - **原子电荷差异**: gold `CC(=O)O[BH-](OC(C)=O)OC(C)=O` → pred `CC(=O)OB(OC(C)=O)OC(C)=O`（硼氢化物的电荷表示不同）
  - **连接点差异**: gold `ClCc1cc(Cl)nc(Cl)n1` → pred `*.ClCc1cc(Cl)nc(Cl)n1`（多了一个断点标记 `*`）

**Label→mol 链接 (F1=0.99)**

- 链接准确率与分子框检测一致，因为 reference sheet 上每个 label 自然链接到其所属分子。

**ID→SMILES grounding (主指标, F1=0.79)**

- Precision ~0.81（resolved 的标识符中 81% SMILES 正确）。
- Recall 0.77，受限於 M3 resolve rate (77.4%) 和 OCSR 准确率。

### 3.2 Gold bbox 修正的影响

原始 gold 分子框为参考图 tile 的完整区域 (210×172px)，包含分子结构 + RDKit 内边距 + label 文字区域。而 M2 检测器（RxnScribe）输出紧贴分子结构的框 (~193×108px)。两者语义不一致导致 IoU 均值仅 0.56，31.3% 的正确检测因 IoU<0.5 被判 miss。

修正方法：(1) 裁掉 label 区域（使用 label_bbox 上边界作为分子框下界）；(2) trim 白边（取非白像素的最小外接矩形）。

修正前后对比：


| 指标                | 修正前 | 修正后 |
| ------------------- | :----: | :----: |
| molbox_f1           |  0.68  |  0.99  |
| e2e_acc (text-only) |  0.56  |  0.77  |

### 3.3 主要瓶颈

1. **OCSR 立体化学差异 (172/1085 = 15.9%)** — 最大瓶颈。strict vs relaxed 差距 11pp，说明大部分 mismatch 是立体化学标注差异而非结构错误。MolScribe 倾向于额外标注磷/硫等原子的立体化学，或遗漏手性中心的立体标注。
2. **M3 resolve rate (77.4%)** — 部分标识符未被 M3 解析，主要因为 M1 NER 未将其识别为化合物（如长化学名中嵌入的标识符）。

### 3.4 Per-Style 分析

三种引用风格的错误分布基本一致，差异不大：

- **dollar**: 最简洁，TEXT_ID_MISS 最少 (11)，OCSR_MISMATCH 最多 (56)
- **compound**: 中等，TEXT_ID_SPURIOUS 最多 (44)，可能因为 "compound" 前缀导致 LLM 误提取
- **angle**: TEXT_ID_MISS 最多 (21)，`<>` 标记可能增加 LLM 识别难度；但 OCSR 表现与其他风格一致

---

## 4. Output Files


| 文件                                                   | 说明                                |
| ------------------------------------------------------ | ----------------------------------- |
| `eval/end2end/outputs/grounding_eval_summary.csv`      | 汇总指标 (25 列, 单行)              |
| `eval/end2end/outputs/grounding_eval_per_sample.jsonl` | 逐样本指标 (500 行)                 |
| `eval/end2end/outputs/grounding_eval_errors.jsonl`     | 错误归因 (每行一个 failed ID)       |
| `data/gold/gold_annotations.jsonl`                        | Gold 标注 (500 行, 含 trimmed bbox) |
| `eval/end2end/pred/predictions.jsonl`                  | 预测 (500 行, M1+M2+M3 合并)        |

---

## 5. Pipeline 配置

### 模型权重


| 模型                | 路径                                                                |
| ------------------- | ------------------------------------------------------------------- |
| Qwen3-8B + LoRA 0.8 | vLLM localhost:8050, model id: qwen3-lora-08                        |
| RxnScribe (Pix2Seq) | ~/.retro_parser_models/pix2seq_reaction_full.ckpt |
| MolScribe           | ~/.retro_parser_models/swin_base_char_aux_1m.pth  |
| PaddleOCR           | paddle 环境, 自动下载                                               |
| Coref (未使用)      | ~/.retro_parser_models/coref_best_hf.ckpt         |

### 环境


| 环境        | Python | 用途                                    |
| ----------- | ------ | --------------------------------------- |
| retroparser | 3.10   | M1/M3/M4, 评测 (含 RDKit)               |
| imgparser   | 3.9    | M2 图像管线 (含 torch/opencv/molscribe) |
| paddle      | —     | PaddleOCR 子进程                        |

### GPU 分配

- M1: vLLM API (GPU 0, 已部署)
- M2: CUDA 6 (81GB)
- M3/M4: CPU (RXNMapper 可选 GPU)

### 数据流

```
OpenExp_500.json
    ↓ build_openexp_eval.py (--image-dir)
openexp_500_eval.json (1000 rows: 500 NER + 500 Action)
openexp_500_coref_sidecar.json (500 records, 含 dict_result + image_annotation)
reference_sheets/openexp_{index}.png (500 images)
    ↓
M1: m1_input.jsonl → m1_predictions.jsonl (500 rows, _prediction.ner + _prediction.action)
M2: per_image_detections.jsonl → m2_image_result.json (500 rows, molecule_boxes + figure_labels + dict_result)
M3: m1 + m2 → m3_coref.jsonl (500 rows, id_smiles_mapping)
M4: m3 → m4_validated.jsonl (500 rows, rxn_validation)
    ↓
build_gold.py → gold_annotations.jsonl (sidecar + eval.json, 位于 dataset/)
parse_predictions.py (M1 + M2 + M3 → predictions.jsonl)
grounding_eval.py (gold vs pred → 11 metrics + error analysis)
```

---

## 6. Key Findings

1. **图像管线 (M2) 表现优秀**: label 检测 F1=0.99, OCR EM=1.00, 分子框检测 F1=0.99, OCSR 有效率 97.6%。RxnScribe + MolScribe 在合成参考图上的检测和识别能力很强。
2. **OCSR 立体化学是主要瓶颈**: strict exact 79.9% vs relaxed exact 91.4%，11pp 差距几乎全部来自立体化学标注差异。MolScribe 对 P/S 等原子的立体化学标注与 gold SMILES 不一致。
3. **文本 grounding 链路基本畅通**: text_id F1=0.93, grounding P=0.81, M3 resolve rate=77.4%。M1 (Qwen3-8B + LoRA) 能有效识别三种引用风格的标识符。
