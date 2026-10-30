"""ChemReX batch CLI — large-scale PDF extraction with resident models.

    python run_batch.py add <pdf | dir>...        # register tasks (idempotent)
    python run_batch.py list / status             # inspect the queue
    python run_batch.py serve-mineru --port 9246  # resident MinerU service (mineru env)
    python run_batch.py run --task-workers 2      # scheduler (runs until queue empty)
    python run_batch.py retry --failed            # requeue failures

Typical large-batch session:

    # terminal 1 — resident parsing service (models load once)
    python run_batch.py serve-mineru --gpu 3 --port 9246

    # terminal 2 — register and process thousands of PDFs
    python run_batch.py add /path/to/pdfs --language en
    python run_batch.py run --task-workers 2 --chunk-workers 16
    python run_batch.py status
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch.cli import main  # noqa: E402


def exit_completed_cli(code, shutdown_timeout=30):
    """Bound interpreter shutdown after the CLI has returned and released locks.

    Third-party executor threads can outlive their timed-out network calls and
    block Python's shutdown joins. Never arm this while the scheduler runs.
    """
    import os
    import threading

    def finish_shutdown():
        os.write(2, b"CLI completed; stopping leftover interpreter threads\n")
        os._exit(code)

    timer = threading.Timer(shutdown_timeout, finish_shutdown)
    timer.daemon = True
    timer.start()
    sys.exit(code)


if __name__ == "__main__":
    # faulthandler handles this in C, so a GIL/native-lock stall can still be
    # diagnosed without ptrace or executing a diagnostic agent in the process.
    import faulthandler
    import signal
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    exit_completed_cli(main())
