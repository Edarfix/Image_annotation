"""Fields shared by every ``configs/models/<name>.yaml``, and helpers shared by wrappers."""

from __future__ import annotations

import os
import time

import httpx
from pydantic import BaseModel, ConfigDict

from ovannot.core.errors import ModelError


class ModelConfig(BaseModel):
    """Provenance and serving fields; each wrapper extends it with its own settings.

    ``endpoint_env`` names an environment variable that overrides ``endpoint``: the launch
    scripts export it, so the same config works whatever node the server runs on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    checkpoint: str  # Hugging Face id
    revision: str  # commit of the checkpoint
    licence: str
    licence_source: str
    source: str  # where the format and settings below were checked
    endpoint: str
    endpoint_env: str | None = None
    timeout_s: float = 600.0
    max_retries: int = 3
    cache: bool = True
    notes: str = ""

    def resolved_endpoint(self) -> str:
        if self.endpoint_env and os.environ.get(self.endpoint_env):
            return os.environ[self.endpoint_env]
        return self.endpoint


def post_json(client: httpx.Client, path: str, body: dict, max_retries: int) -> dict:
    """POST ``body``, retrying connection errors and 5xx answers with exponential backoff.

    Raises ``TimeoutError`` on timeouts and :class:`ModelError` otherwise, so that
    :func:`ovannot.core.errors.classify_exception` records the right failure kind.
    """
    for attempt in range(max_retries + 1):
        last = attempt == max_retries
        try:
            response = client.post(path, json=body)
        except httpx.TimeoutException as exc:
            if last:
                raise TimeoutError(f"{client.base_url}{path}: {exc}") from exc
        except httpx.TransportError as exc:
            if last:
                raise ModelError(f"{client.base_url}{path}: {type(exc).__name__}: {exc}") from exc
        else:
            if response.status_code < 400:
                return response.json()
            if response.status_code < 500 or last:
                raise ModelError(f"HTTP {response.status_code} from {path}: {response.text[:500]}")
        time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def http_client(cfg: ModelConfig) -> httpx.Client:
    # trust_env=False: model traffic never goes through the cluster's HTTP proxy, so images
    # (proprietary ones included) stay on the local network.
    return httpx.Client(
        base_url=cfg.resolved_endpoint(),
        timeout=cfg.timeout_s,
        trust_env=False,
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
    )
