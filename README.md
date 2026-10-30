# ChemReX: Validation-Aware Reaction-Record Construction from Chemical Patent Literature

ChemReX is a modular system for extracting structured chemical reaction records from clean experimental paragraphs and associated reaction images. A PDF preprocessing command extracts page text and images for the existing text and image modules. It combines text extraction, optional image mining, compound standardization/coreference, and conservative reaction validation. A separate batch package (`run_batch.py`) scales whole-document extraction to large PDF collections with resident models and a SQLite task queue.

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Usage](#3-usage)
4. [Training](#4-training)
5. [Evaluation](#5-evaluation)
6. [Citation](#6-citation)

## 1. Overview

ChemReX's core is text-only: it takes a clean experimental paragraph with a provided language tag (`zh` or `en`) as input and runs M1/M3/M4 on it. Whole-document PDF processing is an extension layer added on top of this core: a PDF preprocessing command extracts page text and images, and the `document` command turns ordered PDF extraction blocks into reaction title and procedure units, then runs the core M1/M3/M4 for each unit with a resumable result file. M2 runs once for the image set when requested.

Large-batch PDF processing lives in a separate `batch/` package (entry point `run_batch.py`), modeled on the same stages but engineered for thousands of PDFs: a resident MinerU parse service and a resident OCR/OCSR image worker load their models once and serve many documents, while a scheduler drives a SQLite task queue with per-document concurrency, auto-retry, and stage-level resume. The core `pipeline/` modules stay single-text/single-image.

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

The default M1 preset is `qwen3-lora-08`, served by a local vLLM OpenAI-compatible server. Place the base model and LoRA adapter under:

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

The default service endpoint is `http://localhost:8050/v1`, model id `qwen3-lora-08`, API key `token-abc123`.

Alternatively, use an external OpenAI-compatible model:

```bash
python run_pipeline.py extract \
  --text "..." \
  --task both \
  --language en \
  --preset deepseek-v4-pro
```

### 2.3 Huawei ModelArts MaaS

Set `MAAS_API_KEY` in `.env`, then select the MaaS preset for M1:

```bash
python run_pipeline.py extract \
  --text "To a solution of 1a in CH2Cl2 was added DMSO..." \
  --language en --preset huawei-maas
```

The preset uses model `deepseek-v4.1-flash` and the MaaS endpoint `https://api.modelarts-maas.com/v2/chat/completions`. It disables MaaS thinking mode so short structured outputs are not consumed by reasoning tokens. Set `MAAS_MODEL` to change the model or `MAAS_BASE_URL` to change the endpoint. `MAAS_BASE_URL` accepts either the full chat-completions URL or an OpenAI SDK base URL. Set `EXTRACTION_PRESET=huawei-maas` in `.env` to make it the M1 default; `--preset` overrides that setting for one run.

For M3 compound standardization and coreference, set `STANDARDIZE_MODEL=deepseek-v4.1-flash`, `COREF_MODEL=deepseek-v4.1-flash`, `STANDARDIZE_PROVIDER=maas`, and `COREF_PROVIDER=maas` in `.env`. The provider settings also let you use a different model name on MaaS. When omitted, these modules infer MaaS if the model matches `MAAS_MODEL`. Keep the API key only in `.env`, never in command examples or committed files.

### 2.4 Image Environment

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

If `conda run -n paddle` resolves the wrong environment, set `PADDLE_PYTHON` to the absolute path of the `paddle` environment's Python executable. To disable OPSIN when the available Java runtime is incompatible, set `STANDARDIZE_USE_OPSIN=0`; otherwise use Java 11 or newer.

Image model weights are configured by environment variables:


| Variable         | Default file name            | Purpose                     |
| ---------------- | ---------------------------- | --------------------------- |
| `MOLSCRIBE_CKPT` | `swin_base_char_aux_1m.pth`  | molecule image to SMILES    |
| `MOLDETECT_CKPT` | `coref_best_hf.ckpt`         | molecule/label detection    |
| `RXNSCRIBE_CKPT` | `pix2seq_reaction_full.ckpt` | reaction scheme recognition |

You may also set `CHEMREX_MODELS_DIR` to a directory containing these three files.

### 2.5 Batch Service Environment

The large-batch resident parse service (section 3.4) runs inside the `mineru` conda environment (MinerU 3.x, see section 3.2) and additionally needs a web framework:

```bash
conda activate mineru
pip install "fastapi>=0.115" uvicorn
```

The batch scheduler runs in the text environment (section 2.1) and reuses the `imgparser` and `paddle` environments from section 2.4 for the resident image worker.

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
  --input data/seed-67/test.jsonl \
  --output results/m1_predictions.jsonl \
  --max-workers 32
```

Each JSONL row should contain `text` or `input`, plus `language`. If `task_type` is present, it is used to infer `ner` or `action`; otherwise both tasks are run.

### 3.2 PDF Preprocessing

The default backend is MinerU with a VLM parse backend, which restores the reading order of two-column patent scans and crops reaction images. It requires the `mineru` CLI (MinerU 3.x); point `MINERU_BIN` at it if it is not on `PATH` (for example `MINERU_BIN=<conda env>/bin/mineru`). The parse backend defaults to `vlm-auto-engine` and can be changed with `--mineru-backend` or `MINERU_BACKEND` (use `vlm-http-client` with `--mineru-url` against a remote vLLM server).

```bash
python run_pipeline.py pdf \
  --pdf data/pdf/US9732058B2.pdf \
  --output-dir results/us9732058 \
  --language en
```

This writes an ordered `*_content_list.json` plus cropped `images/` under `<output-dir>/<pdf name>/vlm/`, and `segments.jsonl` (M1-compatible `text` and `language` per ordered block, with page numbers) at the top of the output directory; `mineru.log` records the MinerU run. Pass `--backend poppler` to fall back to the Poppler tools (`pdfinfo`, `pdftotext`, `pdfimages`; `pdftoppm` is needed for `--render-pages`); plain page text can interleave columns on two-column scans and scanned PDFs without a text layer are not OCR'd.

After reviewing or filtering the segments, run the existing modules:

```bash
python run_pipeline.py extract \
  --input results/us9732058/segments.jsonl \
  --output results/us9732058/extraction.jsonl
python run_pipeline.py image --image-base results/us9732058/images
```

In the `imgparser` environment, `pdf --run-image` also runs M2 immediately and writes `image_result.json` to the output directory. This can be costly for PDFs containing full-page scans.

### 3.3 Whole-document reaction extraction

The document runner follows the title-plus-procedure input used by ReactionSeek. For patents with two columns or scanned pages, use an ordered OCR result such as MinerU's `*_content_list.json`; Poppler's plain page text may interleave columns. The runner recognizes `Example N`, `Intermediate N`, letter suffixes and repeated alternative syntheses. It joins procedure continuations across pages, removes standalone page and margin numbers, and records source page/block indices. It does not infer a fully connected synthesis route from these records.

```bash
python run_pipeline.py document \
  --content-list data/md/US9732058B2/vlm/US9732058B2_content_list.json \
  --output-dir results/us9732058_document \
  --prepare-only

python run_pipeline.py document \
  --content-list data/md/US9732058B2/vlm/US9732058B2_content_list.json \
  --output-dir results/us9732058_document \
  --preset huawei-maas --llm-provider maas --llm-model deepseek-v4.1-flash \
  --offline-standardize \
  --image-result results/us9732058_compare/mineru_all_images.json
```

The first command writes `chunks.jsonl`, `manifest.json`, and `links.json` without model calls. `links.json` records textual references to other Examples or Intermediates as candidate links, not verified synthesis connections. The second command writes one record per procedure to `results.jsonl`, flushes each result, and writes `summary.json`. Repeating it skips successful records; `--retry-errors` retries failures. Use `--max-records N` for a short trial. `--kind example` or `--kind intermediate` limits the run. For direct PDF input, pass `--pdf document.pdf` instead of `--content-list`; it preprocesses with MinerU (VLM backend, same options as the `pdf` subcommand) into `<output-dir>/pdf_prepared` and segments the resulting content list. Pass `--pdf-backend poppler` for the old Poppler path, which is less reliable on two-column pages.

The document runner adds three whole-document features on top of the per-procedure core:

- **Position matching**: every unit gets `page_idx` + `bbox` by mapping its recorded block range back to the MinerU `*_content_list.json` blocks (text-match fallback). Each `results.jsonl` record carries a top-level `position` with `page_idx`, `bbox`, `match_method`, and `pages`.
- **Document-level image dictionary**: the M2 `dict_result` is applied globally for coreference; per-procedure image detections take precedence on key conflicts. Pass `--no-global-dict` to use only local detections. Each record's `image.dict_scope` reports local/global/merged entry counts.
- **Final product (route direction)**: the document head (abstract region, before the first Example) is extracted once into `final_product.json` with `product`, `reference`, and position info, complementing the candidate links in `links.json`. Pass `--skip-final-product` to disable. The result is cached and reused on resume; a failed extraction is retried on the next run.

### Reaction and route business delivery

Document and batch extraction create a separate `delivery/` snapshot by default. Each PDF has only two business files: `reactions.jsonl` (one candidate step per line, `reaction-v1`) and `routes.jsonl` (one preparation method per line, `route-v1`). A route references specific reaction and compound IDs; it does not duplicate reaction content or nest methods. Evidence, original PDF, ID mappings, processing status, file hashes and snapshot revision live in `artifacts/`. Candidate storage and automatic adoption are separate: use `quality.admission_eligible` with the supported quality policy version.

Document IDs use the original PDF filename, including its extension. Duplicate filenames in one batch use `result-folder/filename`; reaction, route and target-group IDs use readable source positions. Existing hashes remain solely as archive/integrity checks, not business primary keys. The batch scheduler records the original PDF automatically. When running from parsed input, provide `--source-pdf original.pdf`; offline export accepts `--pdf original.pdf` for one document or `--pdf-map mapping.json` for a batch. Missing PDF identity is an explicit delivery error, never replaced with a hash of parsed text. Export errors can be repaired offline without repeating extraction. Pass `--no-delivery` to disable. No batch table, batch manifest or shared delivery index is generated.

Export an existing run without model calls:

```bash
python -m scripts.export_delivery \
  --run-root results/acceptance_sample10/20261004_v9_auditfix_qualified

python -m scripts.verify_delivery \
  results/acceptance_sample10/20261004_v9_auditfix_qualified
```

Offline batches write `RESULT_FOLDER/delivery/{reactions.jsonl,routes.jsonl,artifacts/}` for each parsed document, one PDF at a time. An optional `--output-root` mirrors the existing result-folder names elsewhere. Counts printed by the command are not written as a total table. Each document snapshot is staged, checked and published together. Receivers use the per-document manifest's processing status, revision and file checksums; partial or failed snapshots must not automatically replace a complete database snapshot. Field semantics and ingestion rules are in [the JSONL design](doc/反应与路线的%20JSONL%20设计.md), with [sample delivery notes](doc/business-delivery-20261007.md).

Each document groups one target product per route, with alternative methods underneath. Crystal forms, salts and solvates of an explicitly identified parent are methods of that target. Different covalent structures and explicit stereoisomers stay separate. Method references do not establish material dependencies; ambiguous connections stay pending review. Every step remains available, including failed extraction placeholders and steps that cannot yet be assigned to a route.

Retained JSON fields keep their names, paths, values and types; repeated chemical data, model logs and mapping candidates are omitted. The generated field guide documents fallback reads and original-record lookup. Original run files and existing CSV exporters remain available. Automatic admission still requires the current V9 evidence contract plus normal step/chunk states; an assembled route is a candidate for review, not proof of chemical correctness or full-text coverage.

To run M2 once inside the document command, pass `--image-base <MinerU output directory>` and use the image environment's Python with `--image-python`. M2 results are cached in `image_result.json`. When per-image detections are available, detections for images attached to a procedure take precedence for that procedure's ID mapping, and the document-wide dictionary fills the remaining identifiers. Existing M2 JSON without `per_image_detections` still supplies image reactions and the document-level dictionary, but not local ID mappings.

### 3.4 Large-Batch PDF Extraction

The `batch/` package scales whole-document extraction to large PDF collections while keeping the core pipeline single-text/single-image. It adds three long-running components:

- **Resident MinerU service** (`batch/mineru_service.py`, runs in the `mineru` conda env): loads the parse models once on the first request and parses any number of PDFs afterwards over HTTP. Measured on one GPU: the first PDF pays the model load (~3 min); subsequent PDFs parse in seconds (~10 s for two pages).
- **Resident image worker** (`batch/image_worker.py`, runs in the `imgparser` env): keeps MolDetect/MolScribe/RxnScribe loaded across documents, supervised over a stdin/stdout JSON line protocol with crash restart and per-request timeout.
- **Scheduler** (`batch/scheduler.py`, text env): a SQLite task queue (`PENDING → RUNNING(parsing/mining/extracting) → DONE/FAILED`) with atomic claiming, auto-retry (up to `BATCH_MAX_ATTEMPTS`), per-chunk error detection (error records force a retry and never end as a silent `DONE`), stuck-task recovery on restart and while idle in daemon mode, and stage-level resume. Each document runs its M1/M3/M4 chunk calls concurrently.

Start the resident parse service (it re-execs into the `mineru` conda env automatically; `--gpu` pins a device, otherwise the freest GPU is auto-selected):

```bash
python run_batch.py serve-mineru --gpu 3 --port 9246
```

On shared GPUs, cap the vLLM pre-allocation with `MINERU_VLM_MEM_UTIL` — a fraction of *total* GPU memory (for example `MINERU_VLM_MEM_UTIL=0.25` for a 79 GB card with ~20 GB free).

Register PDFs and run the scheduler:

```bash
python run_batch.py add /path/to/pdfs                 # auto-detect zh/en; recursive; idempotent
python run_batch.py run --task-workers 2 --chunk-workers 16
python run_batch.py status
python run_batch.py list --status FAILED
python run_batch.py retry --failed
python run_batch.py retry --errors   # DONE tasks whose results.jsonl still ends in error records
python run_batch.py retry --empty --reset-attempts  # re-segment legacy DONE tasks with zero procedures
python run_batch.py retry --outdated --limit 100 --reset-attempts  # reprocess DONE PDFs after segmentation rule changes
```

`run` exits when the queue drains; `--daemon` keeps it polling for new `add`s from other terminals; `--max-tasks N` bounds a trial run. A second `run` against the same database refuses to start (process lock `<db>.lock`), so the single-scheduler assumption is enforced. `--no-image` skips M2, `--skip-final-product` and `--no-global-dict` mirror the document runner flags, and `--offline-standardize` disables the network standardization cascade.

For a targeted repair, `retry --ids-file affected-ids.json --reset-attempts` requeues exactly the UUIDs in a JSON array; unknown or RUNNING tasks reject the transaction. `run --ids-file affected-ids.json` restricts claiming and startup recovery to that list. Use the same `--db` for both commands. PDFs and valid image caches are reused; changed extraction inputs are re-extracted.

The October 7 audit fixes preserve multiline product titles, alphabetic/Roman/compound Example headings and image captions. Uncovered experimental blocks are reported in `manifest.json.coverage_gaps`. Pure assays, characterization, formulation and purification units retain source text without synthesis extraction. Step-level `operation_conditions` record reported temperature/time, source spans, extraction presence and reaction/workup phase.

Single yields retain `conditions.yield.value`. An explicit yield interval uses `value: null`, `range: {lower, upper}`, `raw` and `approximate`; an interval endpoint is never exported as a single yield. Candidate reaction SMILES remain available when MS contradicts the product or referenced procedures/atom sources are incomplete, with `admission_eligible: false` and explicit quality reasons. Mapping confidence alone cannot admit them.

`python scripts/audit_batch_results.py --docs documents.json --no-write` audits latest current records with step-aligned raw/parsed/delivered fields and parses reaction sides separately. `scripts/reprocess_audit_20261007.py` provides `scan`, `backup` and `verify` commands for evidence-based affected-document selection and post-rerun verification. Scanning reads SQLite in read-only mode and never selects documents solely because their version changed.

**Language detection**: PDF API uploads and `run_batch.py add` default to `language=auto`. The worker reads embedded PDF text locally, or detects from parsed text for scans; explicit `en`/`zh` overrides detection. Task status records the original `requested_language`, effective `language`, and `language_source`; unrecognizable text falls back to English (`fallback_en`).

**LLM defaults**: batch routes M1 by document language. Chinese (`language=zh`) uses the local `qwen3-lora-08` preset (Qwen3-8B + LoRA 0.8 over vLLM) with the training-aligned v1 prompts; English (`language=en`) keeps the Huawei MaaS preset with `MAAS_MODEL` (e.g. `deepseek-v4.1-flash`). Override the routing with `BATCH_EXTRACTION_PRESET_ZH` / `BATCH_EXTRACTION_PRESET_EN`; `BATCH_EXTRACTION_PRESET` remains the fallback for other languages. M3 standardization, M4 coreference and final-product extraction always stay on `BATCH_LLM_PROVIDER` / `BATCH_LLM_MODEL` (MaaS). The core pipeline's own `.env` defaults (`EXTRACTION_PRESET`, `STANDARDIZE_MODEL`, `COREF_MODEL`) do not apply to batch; `--preset` / `--llm-provider` / `--llm-model` override the batch defaults. Each reaction's `source_models.m1` and each document's `source_models` record the effective M1/M3/M4/final-product models; `delivery/artifacts/manifest.json` carries the document-level copy. A task whose procedure chunks ended in error is never marked `DONE`: it is auto-retried and, once attempts are exhausted, stored as `FAILED` (retry it with `retry --failed` or `retry --errors`). `reset-stuck` force-requeues tasks left `RUNNING` by a crashed scheduler; the scheduler also recovers them automatically — on start, tasks claimed by a previous `run_id` are requeued immediately, and in `--daemon` mode stuck tasks are recovered whenever the queue is idle.

Each task writes the same file set as the document runner under `results/batch/<uuid>/`:

```text
pdf_prepared/          MinerU output + parse_result.json (parsing resume marker)
image_result.json      M2 result (mining resume marker)
chunks.jsonl / links.json / manifest.json
results.jsonl          append-only, one fsync'd record per procedure
document_result.json   per-PDF outcome: procedure_processed / no_reaction_found / extraction_failed
final_product.json / summary.json
```

Restarting the scheduler reuses completed stages and skips successful records automatically; failed records are retried when the task is retried. Unit tests live in `tests/test_batch_*.py`.

Configuration (all overridable via environment variables):

| Variable                 | Default                    | Purpose                              |
| ------------------------ | -------------------------- | ------------------------------------ |
| `BATCH_DB_PATH`          | `data/batch/tasks.db`      | SQLite task database                 |
| `BATCH_OUTPUT_ROOT`      | `results/batch`            | Per-task output root                 |
| `BATCH_MINERU_URL`       | `http://127.0.0.1:9246`    | Resident MinerU service endpoint     |
| `BATCH_TASK_WORKERS`     | `2`                        | Concurrent documents                 |
| `BATCH_CHUNK_WORKERS`    | `16`                       | Concurrent LLM calls per document    |
| `BATCH_MAX_ATTEMPTS`     | `3`                        | Auto-retry limit per task            |
| `BATCH_RETRY_BACKOFF`    | `5`                        | Base seconds before an auto-requeue (doubles per attempt, capped at 60; `0` disables) |
| `BATCH_EXTRACTION_PRESET`| `huawei-maas`              | Fallback M1 preset for batch runs    |
| `BATCH_EXTRACTION_PRESET_ZH` | `qwen3-lora-08`        | M1 preset for Chinese documents      |
| `BATCH_EXTRACTION_PRESET_EN` | `BATCH_EXTRACTION_PRESET` | M1 preset for English documents   |
| `QWEN3_LORA_BASE_URL`    | `http://localhost:8050/v1` | Local vLLM endpoint for zh M1        |
| `QWEN3_LORA_API_KEY`     | `token-abc123`             | Local vLLM API key for zh M1         |
| `BATCH_LLM_PROVIDER`     | `maas`                     | M3 provider for batch runs           |
| `BATCH_LLM_MODEL`        | `MAAS_MODEL` (`deepseek-v4.1-flash`) | M3 model for batch runs     |
| `BATCH_MINERU_TIMEOUT`   | `7200`                     | Per-PDF parse timeout (s)            |
| `BATCH_IMAGE_TIMEOUT`    | `3600`                     | Per-document M2 timeout (s)          |
| `MINERU_VLM_MEM_UTIL`    | unset                      | vLLM memory fraction for shared GPUs |
| `IMGPARSER_PYTHON`       | `conda run -n imgparser`   | Image worker interpreter             |

### 3.5 Image Mining

```bash
python run_pipeline.py image --image-base path/to/image_directory
```

The output contains `dict_result` for identifier-to-SMILES mappings and `image_result` for reaction structures.

### 3.4 Coreference and Standardization

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

### 3.5 Reaction Validation

```bash
python run_pipeline.py validate \
  --ner-result coref.json \
  --output validated.json
```

### 3.6 Chained Run

```bash
python run_pipeline.py chain \
  --text "..." \
  --language en \
  --chain extract+image+coref+validate \
  --image-base path/to/image_directory \
  --output result.json
```

If an optional input is missing, the corresponding module is skipped with a warning.

## 4. Training

The reviewer-facing MAReX training code is in `training/`. The legacy
`MAReX_training_code/` directory is kept only as a local archive and is ignored
by git.

Generate training data if needed:

```bash
cd training
python data/generate_dataset.py \
  --meta data/meta.json \
  --instructions data/instructions.md \
  --output data \
  --seed 67 \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1
```

Run Qwen3-8B LoRA/QLoRA training:

```bash
cd training
python train.py
```

Edit the `CONFIG` block in `training/train.py` to set local model/data paths,
GPU ids, and hyperparameters. See `training/README.md` for the complete
training-code description.

## 5. Evaluation

ChemReX includes two evaluation settings: NER+Action extraction and end-to-end grounding.

### 5.1 NER + Action Evaluation

Run model inference and compute metrics:

```bash
python eval/evaluate.py \
  --model_type lora \
  --base_model_path model/Qwen3-8B \
  --adapter_path model/qwen3-lora-08 \
  --data data/seed-67/test.json \
  --save_predictions
```

Evaluate an external OpenAI-compatible model:

```bash
python eval/evaluate.py \
  --model_type external \
  --external_model deepseek-v4-pro \
  --data data/seed-67/test.json \
  --save_predictions
```

Recompute metrics from an existing prediction file:

```bash
python eval/verify_metrics.py \
  results/<experiment>/predictions.jsonl \
  --data data/seed-67/test.json
```

The main metrics are:

- **NER**: micro F1 and per-field F1 over the seven reaction fields.
- **Action**: BLEU-4, ROUGE-L, exact match, and BERTScore.

### 5.2 End-to-End Grounding Evaluation

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

### 5.3 Route Repair (route-v2, patent scope)

`run_route_repair.py` rebuilds complete patent routes from an existing delivery
snapshot without re-running extraction for unchanged text. It is the entry for
patent-scope assembly, completeness/gap reporting and drug attribution:

```bash
# full batch: scan archived deliveries -> reassemble -> export route-v2
python run_route_repair.py all \
  --deliverable <deliverable_root> --out <output_dir> --workers 8

# resumable stages (the assembly cache lives under <output_dir>/scan)
python run_route_repair.py scan --deliverable <deliverable_root> --out <output_dir>
python run_route_repair.py assemble --out <output_dir>
python run_route_repair.py export --out <output_dir>

# classify gaps, reuse saved responses, and repair source/structure parsing
# in an independent output (this stage never calls an LLM)
python run_route_repair.py prepare --from-out <previous_output> --out <output_dir>

# targeted re-extraction for confirmed gaps (the only stage calling a model)
python run_route_repair.py supplement --out <output_dir> --charter <charter.json> --workers 4
```

`scan` reads every document's own `delivery/artifacts/{chunks,results}.jsonl`
snapshot, including documents with no previously delivered reactions, and never calls a model; text-identical steps whose earlier round lost
its cached NER are reused from the same document's previous snapshot and marked
as reused. `assemble` groups documents into confirmed patent scopes (volumes of
one patent are assembled together, identical PDFs are collapsed with aliases),
resolves material dependencies without a source-order restriction, keeps
alternative preparations as alternatives, and writes complete routes plus
fragments. A charter names exact document/chunk ids and their original-text
evidence, so only confirmed gaps are re-extracted.

The source-step index is included in `scan/documents.jsonl`; disclosed
preparations without extracted products remain unconfirmed fragments.
`scan/repair_actions.jsonl` classifies missing products as formulation/nonreaction,
purification review, saved-response reuse, structure/mapping repair, step alignment,
or unresolved extraction. These diagnostics are not a model rerun queue.
Chunk counters are aligned against archived source blocks before processing.
Renumbered chunks retain their archived ids and record the current parse id;
ambiguous source matches remain review items.
`prepare` reconstructs current PDF context, reuses saved raw responses with the
current parsers, and retries local structure/mapping resolution without LLM calls.
An empty offline replay, or one that loses a saved product structure, preserves
the unchanged step's saved extraction and existing quality flags. A fresh valid
structure with its material identity intact remains authoritative.
Only remaining gaps with concrete changes to source titles, step text/boundaries,
or resolved references enter `scan/supplement_charter.json`. Cache versions and
fingerprints alone do not qualify. The charter records before/after inputs and
the selected step ids; `supplement` preserves all unselected steps and defaults
to Huawei MaaS for both languages. Failed attempts remain review items rather
than being automatically queued again. Resume compares saved step inputs and
material dictionaries, so code-version changes alone do not repeat model calls.
Document workers share the existing MaaS rate limiter; scan/index updates are
merged sequentially. Re-extraction updates the index
and the remaining gap charter. Complete routes use complete preparations only;
unfinished upstream methods remain fragments linked to the relevant routes.

After upgrading an existing repair output from the October 20 implementation,
rebuild its scan cache with `scan --force`, replay any previously used supplement
charters, then run `assemble --force` and `export`. Keep the existing supplement
results: responses with unchanged source fingerprints are reused. A normal
`assemble` resume does not invalidate earlier assembly shards after code edits.

Reaction ids of unchanged transformations are preserved. Changed or split steps
receive unused reaction indices; their `provenance.source_reaction_ids` and
source parts identify the original records. Numeric reaction indices may differ
from execution order: use the route's ordered `steps` array. Old-to-new mappings
distinguish absorbed routes (`merged`), re-extracted targets (`remapped`), retained
targets and fragments, and include the reason and new record ids.

Outputs under `<output_dir>`: `routes.jsonl` (route-v2), `route_fragments.jsonl`
(route-fragment-v1), `reactions/<source>/<uuid>.jsonl` (updated per-document
reaction snapshots), `drugs/{drug_catalog,drug_patent,document_drug,route_drug}.jsonl`,
`mapping/old_to_new.jsonl`, `scan/` (patent/document index and record cache),
`supplement/` (charter, per-document model output, manifests) and
`stats/summary.json`. Completeness (`complete` / `incomplete` / `unconfirmed`)
is computed independently of `quality.admission_eligible`; the existing queue
database and original delivery directories are never modified.

Regression tests for this entry: `tests/test_route_repair.py`,
`tests/test_drug_links.py`, plus the step/segmenter and delivery suites listed
above.

## 6. Citation

```bibtex
@inproceedings{chemrex2025,
  title={ChemReX: Chemical Reaction Information Extraction from Literature},
  author={...},
  booktitle={...},
  year={2025}
}
```
