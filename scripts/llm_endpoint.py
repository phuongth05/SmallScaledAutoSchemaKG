"""Resolve and validate local or remote OpenAI-compatible LLM endpoints.

Endpoint URLs and credentials are operational settings, not experiment
semantics.  Callers must not persist :class:`LLMConnection` in provenance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import requests

from colab_v2_utils import normalize_base_url


LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
BACKENDS = {"local", "remote"}
TRANSIENT_STATUS_CODES = {429, 502, 503, 504}


@dataclass(frozen=True)
class LLMConnection:
    backend: str
    base_url: str
    api_key: str = field(repr=False)


def resolve_llm_connection(
    backend: str | None = None,
    base_url: str | None = None,
    explicit_local_api_key: str | None = None,
) -> LLMConnection:
    """Resolve an endpoint without writing or logging secrets."""
    backend = (backend or os.environ.get("LLM_BACKEND", "local")).strip().lower()
    if backend not in BACKENDS:
        raise ValueError(f"LLM backend must be one of {sorted(BACKENDS)}, got {backend!r}")

    if backend == "remote":
        remote_url = os.environ.get("REMOTE_LLM_BASE_URL", "").strip()
        if not remote_url:
            # An explicitly non-local CLI URL remains supported, while environment
            # configuration is preferred so pod IDs do not enter saved commands.
            candidate = (base_url or "").strip()
            if candidate and normalize_base_url(candidate) != LOCAL_BASE_URL:
                remote_url = candidate
        if not remote_url:
            raise ValueError("REMOTE_LLM_BASE_URL is required when LLM_BACKEND=remote")
        api_key = os.environ.get("REMOTE_LLM_API_KEY", "")
        if not api_key:
            raise ValueError("REMOTE_LLM_API_KEY is required when LLM_BACKEND=remote")
        return LLMConnection("remote", normalize_base_url(remote_url), api_key)

    local_url = normalize_base_url(base_url or LOCAL_BASE_URL)
    local_key = explicit_local_api_key or os.environ.get("LOCAL_LLM_API_KEY", "EMPTY")
    return LLMConnection("local", local_url, local_key)


def check_llm_health(
    connection: LLMConnection,
    expected_model: str,
    timeout: float = 15,
    session=requests,
) -> list[str]:
    """Check /v1/models with authentication; this function has no writes."""
    headers = {"Authorization": f"Bearer {connection.api_key}"}
    url = f"{connection.base_url.rstrip('/')}/models"
    try:
        response = session.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"{connection.backend} LLM health check could not reach {url}: {exc}"
        ) from exc
    if response.status_code in {401, 403}:
        raise RuntimeError(
            f"{connection.backend} LLM health check authentication failed "
            f"with HTTP {response.status_code}"
        )
    try:
        response.raise_for_status()
        payload = response.json()
        available = [str(item["id"]) for item in payload.get("data", [])]
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{connection.backend} LLM health check returned an invalid response from {url}: {exc}"
        ) from exc
    if expected_model not in available:
        raise RuntimeError(
            f"Expected model {expected_model!r} is not served by {url}; available={available}"
        )
    return available


def is_transient_llm_error(exc: BaseException) -> bool:
    """Return whether an API failure is safe to retry with the same request."""
    status = getattr(exc, "status_code", None)
    if status in TRANSIENT_STATUS_CODES:
        return True
    if isinstance(exc, (requests.ConnectionError, requests.Timeout, TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }
