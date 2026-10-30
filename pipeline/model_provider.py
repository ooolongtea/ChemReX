"""Shared URL handling for OpenAI-compatible model providers."""

import httpx


def request_timeout(seconds):
    """Bound every blocking transport phase, including connection-pool waits."""
    seconds = float(seconds)
    if not 0 < seconds < float('inf'):
        raise ValueError('LLM timeout must be finite and positive')
    return httpx.Timeout(seconds, connect=min(10, seconds),
                         write=min(30, seconds), pool=min(30, seconds))


def sdk_base_url(url: str) -> str:
    """Accept either an SDK base URL or a full chat-completions endpoint."""
    value = (url or "").rstrip("/")
    suffix = "/chat/completions"
    return value[:-len(suffix)] if value.endswith(suffix) else value
