"""HTTP frontend and separate worker for the existing full-document pipeline."""

import argparse
import os

from batch import config as batch_config  # load .env before choosing defaults


def main(argv=None):
    parser = argparse.ArgumentParser(description="ChemReX PDF upload API")
    parser.add_argument("mode", nargs="?", choices=("serve", "worker"), default="serve")
    parser.add_argument("--host", default=os.getenv("CHEMREX_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CHEMREX_API_PORT", "9250")))
    args = parser.parse_args(argv)
    from batch.api import ApiConfig
    config = ApiConfig.from_env()
    if args.mode == "worker":
        from batch.cli import main as batch_main
        config.db_path.parent.mkdir(parents=True, exist_ok=True)
        batch_config.BATCH_LOG_DIR = config.data_root / "logs"
        # Also needed by the existing extraction-reuse machinery.
        os.environ["BATCH_DB_PATH"] = str(config.db_path)
        # Historical batch outputs plus the live API database. New DONE tasks
        # are available on the next upload; no cache rebuild/import is needed.
        os.environ.setdefault('CHEMREX_REUSE_TASK_DATABASES', os.pathsep.join([
            str(config.data_root.parent / 'tasks.db'), str(config.db_path)]))
        batch_config.BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('CHEMREX_CALL_METRICS', str(batch_config.BATCH_LOG_DIR / 'model-calls.jsonl'))
        return batch_main([
            "--db", str(config.db_path), "run", "--daemon",
            "--task-workers", os.getenv("CHEMREX_API_TASK_WORKERS", "1"),
            "--chunk-workers", str(batch_config.CHUNK_WORKERS),
            "--poll-interval", "0.5",
        ])
    import uvicorn
    from batch.api import create_app
    uvicorn.run(create_app(config), host=args.host, port=args.port,
                timeout_graceful_shutdown=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
