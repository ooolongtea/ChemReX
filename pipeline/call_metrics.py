"""Optional append-only model usage accounting; never logs prompts or credentials."""
import fcntl
import json
import os
import time
from pathlib import Path


def record_call(module, model, started, response, preset=None, provider=None):
    path = os.getenv('CHEMREX_CALL_METRICS')
    if not path:
        return
    usage = getattr(response, 'usage', None)
    if hasattr(usage, 'model_dump'):
        usage = usage.model_dump()
    elif not isinstance(usage, dict):
        usage = None
    row = {'module': module, 'model': model,
           'elapsed_seconds': round(time.monotonic()-started, 3), 'usage': usage}
    if preset:
        row['preset'] = preset
    if provider:
        row['provider'] = provider
    with Path(path).open('a', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)
