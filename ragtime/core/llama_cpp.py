"""
llama.cpp server discovery and embedding probes.

The llama.cpp server exposes OpenAI-compatible inference endpoints, but its
metadata shape varies by build and flags. This module keeps that tolerance in
one place so the rest of Ragtime can treat llama.cpp like a first-class local
provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CHAT_BASE_URL = "http://host.docker.internal:8080"
DEFAULT_EMBEDDING_BASE_URL = "http://host.docker.internal:8081"
OPENAI_COMPATIBLE_CHAT_ENDPOINTS = ["/chat/completions"]


@dataclass
class LlamaCppModelInfo:
    """Model metadata discovered from a llama.cpp server."""

    id: str
    name: str
    created: Optional[int] = None
    context_limit: Optional[int] = None
    dimensions: Optional[int] = None
    capabilities: list[str] | None = None
    supported_endpoints: list[str] | None = None


def normalize_base_url(
    base_url: str | None, default: str = DEFAULT_CHAT_BASE_URL
) -> str:
    """Normalize a llama.cpp base URL for endpoint joins."""
    value = str(base_url or "").strip() or default
    return value.rstrip("/")


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data", payload.get("models", payload))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            return [data]
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        parsed = int(value)
        return parsed if parsed > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed > 0 else None
    return None


def extract_props_context(payload: dict[str, Any]) -> int | None:
    """Extract active runtime context from /props."""
    default_generation = payload.get("default_generation_settings")
    if isinstance(default_generation, dict):
        parsed = _coerce_positive_int(default_generation.get("n_ctx"))
        if parsed:
            return parsed

    for key in ("n_ctx", "context_length", "context_size"):
        parsed = _coerce_positive_int(payload.get(key))
        if parsed:
            return parsed

    return None


def extract_model_context(row: dict[str, Any]) -> int | None:
    """Extract model training or runtime context from a model metadata row."""
    for container_key in ("meta", "metadata", "model_info", "details"):
        container = row.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in ("n_ctx_train", "n_ctx", "context_length", "max_context_length"):
            parsed = _coerce_positive_int(container.get(key))
            if parsed:
                return parsed

    for key in ("n_ctx_train", "n_ctx", "context_length", "max_context_length"):
        parsed = _coerce_positive_int(row.get(key))
        if parsed:
            return parsed

    return None


def extract_embedding_dimension(payload: dict[str, Any]) -> int | None:
    """Extract embedding vector length from an OpenAI-compatible response."""
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    embedding = first.get("embedding")
    if isinstance(embedding, list):
        return len(embedding)
    return None


async def is_reachable(
    base_url: str, timeout: float = 5.0
) -> tuple[bool, Optional[str]]:
    """Check llama.cpp reachability using tolerant endpoint fallbacks."""
    normalized_base = normalize_base_url(base_url)
    endpoints = ["/health", "/v1/health", "/v1/models", "/models"]
    last_status: int | None = None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for endpoint in endpoints:
                try:
                    response = await client.get(f"{normalized_base}{endpoint}")
                except httpx.HTTPStatusError:
                    raise
                except Exception:
                    raise

                if response.status_code < 500 and response.status_code != 404:
                    return True, None
                last_status = response.status_code
    except httpx.ConnectError:
        return (
            False,
            f"Failed to connect to llama.cpp at {normalized_base}. Make sure llama-server is running and reachable.",
        )
    except httpx.TimeoutException:
        return False, f"Connection to llama.cpp at {normalized_base} timed out."
    except Exception as exc:
        return False, f"Error connecting to llama.cpp at {normalized_base}: {exc}"

    return (
        False,
        f"llama.cpp at {normalized_base} did not expose a usable health or models endpoint (last status {last_status}).",
    )


async def fetch_props(
    base_url: str, client: httpx.AsyncClient | None = None
) -> dict[str, Any]:
    """Fetch /props when supported, returning an empty dict on failure."""
    normalized_base = normalize_base_url(base_url)
    try:
        if client is not None:
            response = await client.get(f"{normalized_base}/props")
        else:
            async with httpx.AsyncClient(timeout=5.0) as new_client:
                response = await new_client.get(f"{normalized_base}/props")
        if response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.debug(
            "Failed to fetch llama.cpp props from %s: %s", normalized_base, exc
        )
    return {}


async def list_chat_models(base_url: str) -> list[LlamaCppModelInfo]:
    """List chat-capable model records from a llama.cpp server."""
    normalized_base = normalize_base_url(base_url)
    reachable, error = await is_reachable(normalized_base)
    if not reachable:
        raise ConnectionError(
            error or f"Cannot connect to llama.cpp at {normalized_base}"
        )

    last_error: str | None = None
    async with httpx.AsyncClient(timeout=10.0) as client:
        props = await fetch_props(normalized_base, client)
        active_context = extract_props_context(props)

        rows: list[dict[str, Any]] = []
        for endpoint in ("/v1/models", "/models"):
            try:
                response = await client.get(f"{normalized_base}{endpoint}")
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                rows = _extract_rows(response.json())
                if rows:
                    break
            except Exception as exc:
                last_error = str(exc)

    if not rows:
        raise RuntimeError(last_error or "llama.cpp returned no models")

    models: list[LlamaCppModelInfo] = []
    for row in rows:
        model_id = str(
            row.get("id") or row.get("name") or row.get("model") or ""
        ).strip()
        if not model_id:
            continue
        context_limit = active_context or extract_model_context(row)
        models.append(
            LlamaCppModelInfo(
                id=model_id,
                name=str(row.get("name") or model_id),
                created=_coerce_positive_int(row.get("created")),
                context_limit=context_limit,
                capabilities=["chat"],
                supported_endpoints=list(OPENAI_COMPATIBLE_CHAT_ENDPOINTS),
            )
        )

    return models


async def probe_embedding_dimension(base_url: str, model: str) -> int | None:
    """Probe /v1/embeddings and return the output vector length."""
    normalized_base = normalize_base_url(base_url, DEFAULT_EMBEDDING_BASE_URL)
    model_id = str(model or "").strip()
    if not model_id:
        return None

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{normalized_base}/v1/embeddings",
            json={"model": model_id, "input": "test"},
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return extract_embedding_dimension(payload)
    return None


async def list_embedding_models(
    base_url: str, selected_model: str | None = None
) -> list[LlamaCppModelInfo]:
    """List/probe embedding model records from a llama.cpp embedding server."""
    normalized_base = normalize_base_url(base_url, DEFAULT_EMBEDDING_BASE_URL)
    reachable, error = await is_reachable(normalized_base)
    if not reachable:
        raise ConnectionError(
            error or f"Cannot connect to llama.cpp at {normalized_base}"
        )

    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for endpoint in ("/v1/models", "/models"):
            try:
                response = await client.get(f"{normalized_base}{endpoint}")
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                rows = _extract_rows(response.json())
                if rows:
                    break
            except Exception as exc:
                logger.debug("Failed to list llama.cpp embedding models: %s", exc)

    candidates: list[str] = []
    if selected_model:
        candidates.append(selected_model.strip())
    for row in rows:
        model_id = str(
            row.get("id") or row.get("name") or row.get("model") or ""
        ).strip()
        if model_id and model_id not in candidates:
            candidates.append(model_id)

    if not candidates:
        candidates.append("default")

    models: list[LlamaCppModelInfo] = []
    last_error: str | None = None
    for model_id in candidates:
        try:
            dimensions = await probe_embedding_dimension(normalized_base, model_id)
        except httpx.HTTPStatusError as exc:
            last_error = exc.response.text[:300] if exc.response.text else str(exc)
            continue
        except Exception as exc:
            last_error = str(exc)
            continue
        models.append(
            LlamaCppModelInfo(
                id=model_id,
                name=model_id,
                dimensions=dimensions,
                capabilities=["embedding"],
            )
        )

    if not models and last_error:
        raise RuntimeError(f"llama.cpp embedding probe failed: {last_error}")

    return models


async def get_model_context_length(model: str, base_url: str) -> int | None:
    """Return the active/context metadata for a llama.cpp model if discoverable."""
    target = str(model or "").strip()
    try:
        models = await list_chat_models(base_url)
    except Exception:
        return None
    if not target and models:
        return models[0].context_limit
    for info in models:
        if info.id == target:
            return info.context_limit
    return None
