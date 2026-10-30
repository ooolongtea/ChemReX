"""Cross-process request pacing for the shared ModelArts MaaS account."""

import fcntl
import os
import time
from pathlib import Path


def wait_for_slot() -> None:
    rpm = int(os.getenv("MAAS_REQUESTS_PER_MINUTE", "70"))
    if rpm <= 0:
        return
    path = Path(os.getenv("MAAS_RATE_LIMIT_FILE",
                          f"/tmp/chemrex_maas_rate_{os.getuid()}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            stream.seek(0)
            try:
                next_at = float(stream.read().strip())
            except ValueError:
                next_at = 0.0
            now = time.time()
            slot = max(now, next_at)
            stream.seek(0)
            stream.truncate()
            stream.write(str(slot + 60.0 / rpm))
            stream.flush()
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
    delay = slot - time.time()
    if delay > 0:
        time.sleep(delay)
