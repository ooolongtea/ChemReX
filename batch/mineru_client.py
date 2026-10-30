"""HTTP client for the resident MinerU parsing service (batch/mineru_service.py)."""

import time
from typing import Any, Dict, Optional

import requests


class MineruServiceError(RuntimeError):
    pass


class MineruClient:
    def __init__(self, base_url: str, timeout: int = 7200):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self, timeout: int = 10) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=timeout)
        response.raise_for_status()
        return response.json()

    def wait_healthy(self, attempts: int = 1, interval: float = 5.0) -> Dict[str, Any]:
        """Probe /health; returns the payload or raises MineruServiceError."""
        last_error: Optional[Exception] = None
        for _ in range(attempts):
            try:
                return self.health()
            except Exception as exc:
                last_error = exc
                time.sleep(interval)
        raise MineruServiceError(
            f"MinerU service at {self.base_url} is not reachable: {last_error}")

    def parse(self, pdf_path: str, output_dir: str, backend: str,
              language: str = "en", first_page: int = 1,
              last_page: Optional[int] = None,
              server_url: Optional[str] = None) -> Dict[str, Any]:
        """Parse one PDF; returns the service payload or raises MineruServiceError."""
        payload = {
            "pdf_path": pdf_path,
            "output_dir": output_dir,
            "backend": backend,
            "language": language,
            "first_page": first_page,
            "last_page": last_page,
        }
        if server_url:
            payload["server_url"] = server_url
        try:
            response = requests.post(f"{self.base_url}/parse", json=payload,
                                     timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise MineruServiceError(f"MinerU service request failed: {exc}") from exc
        if data.get("status") != "ok":
            raise MineruServiceError(
                f"MinerU parse failed for {pdf_path}: {data.get('error')}")
        return data
