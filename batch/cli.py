"""Batch CLI — task management and scheduler control.

    python run_batch.py add <pdf | dir | glob>... [--language en]
    python run_batch.py list [--status PENDING] [--limit 50] [--json]
    python run_batch.py status [--json]
    python run_batch.py retry <uuid> | --failed | --errors [--reset-attempts]
    python run_batch.py reset-stuck          # force-requeue RUNNING tasks
    python run_batch.py run [--task-workers 2] [--chunk-workers 16] [--daemon]
    python run_batch.py serve-mineru [--port 9246] [--gpu 3] [--backend vlm-auto-engine]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from loguru import logger

from . import config as batch_config
from .store import TaskStore, build_uuid, dump_task, dump_tasks_json


# ── add ──
def discover_pdfs(paths: List[str], limit: Optional[int] = None) -> List[Path]:
    """Expand files / directories / glob patterns into a sorted PDF list."""
    pdfs = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            pdfs.extend(sorted(path.rglob("*.pdf")))
            pdfs.extend(sorted(path.rglob("*.PDF")))
        elif path.is_file():
            pdfs.append(path)
        else:
            pdfs.extend(sorted(Path().glob(raw)))
    unique = []
    seen = set()
    for pdf in pdfs:
        resolved = pdf.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(pdf)
    if limit is not None:
        unique = unique[:limit]
    return unique


def cmd_add(args) -> int:
    pdfs = discover_pdfs(args.paths, limit=args.limit)
    if not pdfs:
        logger.error("no PDF files found in: {}", args.paths)
        return 1
    store = TaskStore(Path(args.db))
    added = skipped = 0
    for pdf in pdfs:
        resolved = str(pdf.resolve())
        uuid = build_uuid(pdf)
        output_dir = str(Path(args.output_root).resolve() / uuid)
        _, created = store.add_task(uuid, resolved, output_dir,
                                    language=args.language)
        if not created and store.get(uuid)["pdf_path"] != resolved:
            # 8-hex digest collision between two different PDFs sharing a
            # stem — fall back to a longer digest (review round-2 minor).
            uuid = build_uuid(pdf, digest_len=16)
            output_dir = str(Path(args.output_root).resolve() / uuid)
            _, created = store.add_task(uuid, resolved, output_dir,
                                        language=args.language)
        if created:
            added += 1
            print(f"added   {uuid}  {pdf}")
            continue
        skipped += 1
        existing = store.get(uuid)
        if existing and (existing.get('requested_language', existing['language']) != args.language
                         or existing["output_dir"] != output_dir):
            # Review round-2 M1: a re-add silently keeps the original
            # language / output dir; say so instead of processing with the
            # wrong prompts or writing to the old location.
            print(f"warning: {uuid} already present with language="
                  f"{existing['language']}, output_dir={existing['output_dir']}; "
                  f"ignoring requested language={args.language}, "
                  f"output_dir={output_dir}")
    store.close()
    print(f"{added} added, {skipped} already present (total scanned: {len(pdfs)})")
    return 0


# ── list / status ──
def cmd_list(args) -> int:
    store = TaskStore(Path(args.db))
    tasks = store.list_tasks(status=args.status, limit=args.limit)
    store.close()
    if args.json:
        print(dump_tasks_json(tasks))
    else:
        for task in tasks:
            print(dump_task(task))
        if not tasks:
            print("no tasks" + (f" with status {args.status}" if args.status else ""))
    return 0


def cmd_status(args) -> int:
    store = TaskStore(Path(args.db))
    counts = store.counts()
    store.close()
    if args.json:
        print(json.dumps(counts, indent=2))
    else:
        print("status\tcount")
        for status, count in counts.items():
            print(f"{status}\t{count}")
    return 0


# ── retry / recovery ──
def selected_uuids(path):
    if path is None:
        return None
    import json
    values = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError('--ids-file must contain a JSON array of nonempty task UUIDs')
    return list(dict.fromkeys(values))


def cmd_retry(args) -> int:
    store = TaskStore(Path(args.db))
    try:
        if getattr(args, 'ids_file', None):
            if any(getattr(args, k, False) for k in ('outdated', 'empty', 'errors', 'failed', 'uuid')):
                raise ValueError('--ids-file cannot be combined with another retry selector')
            moved = store.retry_selected(selected_uuids(args.ids_file), reset_attempts=args.reset_attempts)
        elif args.outdated:
            from pipeline.document.segmenter import SEGMENTATION_VERSION
            moved = store.retry_done_outdated(
                SEGMENTATION_VERSION, limit=args.limit,
                reset_attempts=args.reset_attempts)
        elif args.empty:
            moved = store.retry_done_without_procedures(
                reset_attempts=args.reset_attempts)
        elif args.errors:
            moved = store.retry_done_with_errors(reset_attempts=args.reset_attempts)
        elif args.failed:
            moved = store.retry(all_failed=True, reset_attempts=args.reset_attempts)
        elif args.uuid:
            moved = store.retry(uuid=args.uuid, reset_attempts=args.reset_attempts)
        else:
            raise ValueError("retry requires a uuid, --failed, --errors, --empty or --outdated")
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
    print(f"{moved} task(s) requeued")
    return 0


def cmd_reset_stuck(args) -> int:
    db_path = Path(args.db)
    lock_file = None
    try:
        import fcntl

        lock_file = open(db_path.with_suffix(".lock"), "a+")
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        lock_file = None  # non-POSIX platform: skip single-instance protection
    except (BlockingIOError, OSError) as exc:
        # Review round-2 M3: resetting while a scheduler runs makes both
        # write the same output dirs concurrently (stale writer). The
        # scheduler must be stopped first.
        print(f"error: a scheduler is currently running for {db_path} ({exc});"
              " stop it before reset-stuck", file=sys.stderr)
        if lock_file is not None:
            lock_file.close()
        return 1
    store = TaskStore(db_path)
    try:
        moved = store.reset_stuck()
    finally:
        store.close()
        if lock_file is not None:
            lock_file.close()
    print(f"{moved} RUNNING task(s) requeued")
    return 0


# ── run (scheduler) ──
def apply_llm_env(args) -> None:
    """Pin the pipeline LLM env vars before the pipeline modules are imported.

    Priority: CLI flags > BATCH_* env vars > MaaS defaults (huawei-maas preset
    with MAAS_MODEL for M1 and M3). The root config freezes these variables at
    import time, so this must run before anything imports ``pipeline.*``.
    """
    os.environ["EXTRACTION_PRESET"] = args.preset or batch_config.BATCH_EXTRACTION_PRESET
    # Per-language presets are read by BatchScheduler._processor. CLI flags
    # must update the in-memory mapping before the scheduler starts.
    if getattr(args, "preset_zh", None):
        batch_config.BATCH_EXTRACTION_PRESETS["zh"] = args.preset_zh
    if getattr(args, "preset_en", None):
        batch_config.BATCH_EXTRACTION_PRESETS["en"] = args.preset_en
    provider = args.llm_provider or batch_config.BATCH_LLM_PROVIDER
    model = args.llm_model or batch_config.BATCH_LLM_MODEL
    os.environ["STANDARDIZE_PROVIDER"] = provider
    os.environ["COREF_PROVIDER"] = provider
    os.environ["STANDARDIZE_MODEL"] = model
    os.environ["COREF_MODEL"] = model
    # Final-product extraction is part of M3/M4 and must stay on MaaS even
    # when M1 is routed to a local LoRA preset.
    os.environ["FINAL_PRODUCT_PROVIDER"] = provider
    os.environ["FINAL_PRODUCT_MODEL"] = model
    if args.offline_standardize:
        os.environ["STANDARDIZE_OFFLINE"] = "1"


def cmd_run(args) -> int:
    apply_llm_env(args)
    from .mineru_client import MineruClient
    from .scheduler import BatchScheduler

    db_path = Path(args.db)
    lock_file = None
    try:
        import fcntl

        lock_file = open(db_path.with_suffix(".lock"), "a+")
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        lock_file = None  # non-POSIX platform: skip single-instance protection
    except BlockingIOError:
        logger.error("another scheduler is already running for {}", db_path)
        if lock_file is not None:
            lock_file.close()
        return 1
    except OSError as exc:
        logger.error("cannot acquire scheduler lock for {}: {}", db_path, exc)
        if lock_file is not None:
            lock_file.close()
        return 1

    store = TaskStore(db_path)
    mineru = MineruClient(args.mineru_url, timeout=batch_config.MINERU_TIMEOUT)
    image_client = None
    if not args.no_image:
        from .image_client import ImageWorkerClient

        script = Path(__file__).resolve().parent / "image_worker.py"
        image_client = ImageWorkerClient(
            command=batch_config.image_worker_command(script),
            timeout=batch_config.IMAGE_TIMEOUT,
            log_path=Path(batch_config.BATCH_LOG_DIR) / "image_worker.log",
        )
    scheduler = BatchScheduler(
        store=store,
        mineru=mineru,
        image_client=image_client,
        task_workers=args.task_workers,
        chunk_workers=args.chunk_workers,
        poll_interval=args.poll_interval,
        daemon=args.daemon,
        max_tasks=args.max_tasks,
        mineru_backend=args.mineru_backend,
        max_attempts=batch_config.MAX_ATTEMPTS,
        lock_expire_seconds=batch_config.LOCK_EXPIRE_SECONDS,
        auto_retry=not args.no_auto_retry,
        no_image=args.no_image,
        skip_final_product=args.skip_final_product,
        no_global_dict=args.no_global_dict,
        retry_backoff=batch_config.RETRY_BACKOFF,
        no_delivery=args.no_delivery,
        task_uuids=selected_uuids(getattr(args, 'ids_file', None)),
    )
    try:
        claimed = scheduler.run()
    finally:
        store.close()
        if lock_file is not None:
            lock_file.close()
    return 0 if claimed >= 0 else 1


# ── serve-mineru (re-exec into the mineru conda env when needed) ──
def cmd_serve_mineru(rest: List[str]) -> int:
    try:
        from .mineru_service import main as service_main
    except ImportError:
        # mineru missing or too old in this env — re-exec inside the mineru env
        env = os.getenv("MINERU_CONDA_ENV", "mineru")
        script = Path(__file__).resolve().parent / "mineru_service.py"
        command = ["conda", "run", "--no-capture-output", "-n", env,
                   "python", str(script)] + rest
        print(f"mineru not importable here; re-executing in conda env '{env}':")
        print("  " + " ".join(command))
        os.execvp(command[0], command)
        return 1  # unreachable
    sys.argv = ["mineru_service.py"] + rest
    service_main()
    return 0


# ── parser ──
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_batch.py",
        description="ChemReX batch — large-scale PDF extraction with resident models",
    )
    parser.add_argument("--db", default=str(batch_config.BATCH_DB_PATH),
                        help=f"SQLite task database (default: {batch_config.BATCH_DB_PATH})")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Register PDFs (files, dirs, or glob patterns)")
    p_add.add_argument("paths", nargs="+", help="PDF files / directories / globs")
    p_add.add_argument("--language", default="auto", choices=["auto", "en", "zh"],
                       help="Detect PDF language by default; en/zh explicitly override detection")
    p_add.add_argument("--limit", type=int, help="Register at most N PDFs")
    p_add.add_argument("--output-root", default=str(batch_config.BATCH_OUTPUT_ROOT),
                       help=f"Task output root (default: {batch_config.BATCH_OUTPUT_ROOT})")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="List tasks")
    p_list.add_argument("--status", choices=["PENDING", "RUNNING", "DONE", "FAILED"])
    p_list.add_argument("--limit", type=int, default=100)
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_status = sub.add_parser("status", help="Task counts by status")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_retry = sub.add_parser("retry", help="Requeue FAILED tasks")
    p_retry.add_argument("uuid", nargs="?")
    p_retry.add_argument("--failed", action="store_true", help="Requeue all FAILED tasks")
    p_retry.add_argument("--errors", action="store_true",
                         help="Requeue DONE tasks whose results.jsonl ends in error records")
    p_retry.add_argument("--empty", action="store_true",
                         help="Requeue legacy DONE tasks with zero procedure chunks")
    p_retry.add_argument("--outdated", action="store_true",
                         help="Requeue DONE tasks produced by an older segmentation version")
    p_retry.add_argument("--limit", type=int, help="Maximum outdated tasks to requeue")
    p_retry.add_argument("--reset-attempts", action="store_true")
    p_retry.add_argument('--ids-file', help='Requeue only UUIDs in a JSON array (DONE/FAILED; RUNNING refused)')
    p_retry.set_defaults(func=cmd_retry)

    p_reset = sub.add_parser("reset-stuck", help="Force-requeue RUNNING tasks")
    p_reset.set_defaults(func=cmd_reset_stuck)

    p_run = sub.add_parser("run", help="Start the scheduler")
    p_run.add_argument('--ids-file', help='Claim and recover only UUIDs in a JSON array')
    p_run.add_argument("--task-workers", type=int, default=batch_config.TASK_WORKERS,
                       help=f"Concurrent documents (default: {batch_config.TASK_WORKERS})")
    p_run.add_argument("--chunk-workers", type=int, default=batch_config.CHUNK_WORKERS,
                       help=f"Concurrent LLM calls per document (default: {batch_config.CHUNK_WORKERS})")
    p_run.add_argument("--poll-interval", type=float, default=batch_config.POLL_INTERVAL)
    p_run.add_argument("--daemon", action="store_true",
                       help="Keep running when the queue is empty (accept new `add`s)")
    p_run.add_argument("--max-tasks", type=int, help="Process at most N tasks this run")
    p_run.add_argument("--mineru-url", default=batch_config.MINERU_SERVICE_URL)
    p_run.add_argument("--mineru-backend", default=batch_config.MINERU_BACKEND)
    p_run.add_argument("--no-image", action="store_true", help="Skip M2 image mining")
    p_run.add_argument("--no-auto-retry", action="store_true",
                       help="Failed tasks stay FAILED instead of auto-requeue")
    p_run.add_argument("--skip-final-product", action="store_true")
    p_run.add_argument("--no-delivery", action="store_true", help="Disable compact route deliveries")
    p_run.add_argument("--no-global-dict", action="store_true",
                       help="Disable the document-level image dictionary for coref")
    p_run.add_argument("--preset",
                       help="Fallback M1 preset for languages without a specific preset")
    p_run.add_argument("--preset-zh", help="M1 preset for Chinese documents "
                       "(default: BATCH_EXTRACTION_PRESET_ZH)")
    p_run.add_argument("--preset-en", help="M1 preset for English documents "
                       "(default: BATCH_EXTRACTION_PRESET_EN)")
    p_run.add_argument("--llm-provider", help="M3 provider override")
    p_run.add_argument("--llm-model", help="M3 model override")
    p_run.add_argument("--offline-standardize", action="store_true")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve-mineru":
        return cmd_serve_mineru(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
