"""
LM Studio native model discovery and model-management helpers.

LM Studio serves OpenAI-compatible inference at /v1, but its native API exposes
the metadata Ragtime needs for first-class provider support. Keep that native
surface isolated here so routes and runtime code can use a stable provider
shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = "http://host.docker.internal:1234"
OPENAI_COMPATIBLE_CHAT_ENDPOINTS = ["/chat/completions"]
CHAT_MODEL_TYPES = {"llm", "vlm"}
EMBEDDING_MODEL_TYPES = {"embedding", "embeddings"}


@dataclass
class LmStudioModelInfo:
    """Model metadata discovered from LM Studio native APIs."""

    id: str
    name: str
    model_type: str = ""
    context_limit: Optional[int] = None
    dimensions: Optional[int] = None
    capabilities: list[str] | None = None
    supported_endpoints: list[str] | None = None
    loaded: bool = False
    loaded_instances: list[dict[str, Any]] = field(default_factory=list)
    state: Optional[str] = None
    architecture: Optional[str] = None
    quantization: Optional[str] = None
    format: Optional[str] = None


def normalize_base_url(base_url: str | None, default: str = DEFAULT_BASE_URL) -> str:
    """Normalize an LM Studio base URL for endpoint joins."""
    value = str(base_url or "").strip() or default
    return value.rstrip("/")


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


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("models", "data"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_quantization(row: dict[str, Any]) -> str | None:
    quantization = row.get("quantization")
    if isinstance(quantization, dict):
        return _string_or_none(quantization.get("name"))
    return _string_or_none(quantization)


def _extract_capabilities(row: dict[str, Any]) -> list[str]:
    capabilities = row.get("capabilities")
    if isinstance(capabilities, list):
        return [str(item).lower() for item in capabilities if item]
    if isinstance(capabilities, dict):
        return [str(key).lower() for key, enabled in capabilities.items() if enabled]
    return []


def _extract_loaded_instances(row: dict[str, Any]) -> list[dict[str, Any]]:
    loaded_instances = row.get("loaded_instances")
    if isinstance(loaded_instances, list):
        return [item for item in loaded_instances if isinstance(item, dict)]
    return []


def _is_loaded(row: dict[str, Any], loaded_instances: list[dict[str, Any]]) -> bool:
    state = str(row.get("state") or "").strip().lower()
    if state in {"loaded", "loading"}:
        return True
    return bool(loaded_instances)


def parse_model_row(row: dict[str, Any]) -> LmStudioModelInfo | None:
    """Parse one native LM Studio model row into normalized metadata."""
    model_id = str(row.get("key") or row.get("id") or row.get("model") or "").strip()
    if not model_id:
        return None

    loaded_instances = _extract_loaded_instances(row)
    return LmStudioModelInfo(
        id=model_id,
        name=str(row.get("display_name") or row.get("name") or model_id),
        model_type=str(row.get("type") or "").strip().lower(),
        context_limit=_coerce_positive_int(row.get("max_context_length")),
        dimensions=(
            _coerce_positive_int(row.get("embedding_dimension"))
            or _coerce_positive_int(row.get("dimensions"))
        ),
        capabilities=_extract_capabilities(row) or None,
        supported_endpoints=list(OPENAI_COMPATIBLE_CHAT_ENDPOINTS),
        loaded=_is_loaded(row, loaded_instances),
        loaded_instances=loaded_instances,
        state=_string_or_none(row.get("state")),
        architecture=_string_or_none(row.get("architecture") or row.get("arch")),
        quantization=_extract_quantization(row),
        format=_string_or_none(row.get("format") or row.get("compatibility_type")),
    )


def _extract_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return response.text[:500] or f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or payload)[:500]
        if isinstance(error, str):
            return error[:500]
        message = payload.get("message")
        if message:
            return str(message)[:500]
    return str(payload)[:500]


async def is_reachable(
    base_url: str, timeout: float = 5.0
) -> tuple[bool, Optional[str]]:
    """Check LM Studio reachability using native endpoints first."""
    normalized_base = normalize_base_url(base_url)
    endpoints = ["/api/v1/models", "/api/v0/models", "/v1/models"]
    last_status: int | None = None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for endpoint in endpoints:
                response = await client.get(f"{normalized_base}{endpoint}")
                if response.status_code < 500 and response.status_code != 404:
                    return True, None
                last_status = response.status_code
    except httpx.ConnectError:
        return (
            False,
            f"Failed to connect to LM Studio at {normalized_base}. Make sure the LM Studio server is running and reachable.",
        )
    except httpx.TimeoutException:
        return False, f"Connection to LM Studio at {normalized_base} timed out."
    except Exception as exc:
        return False, f"Error connecting to LM Studio at {normalized_base}: {exc}"

    return (
        False,
        f"LM Studio at {normalized_base} did not expose a usable models endpoint (last status {last_status}).",
    )


async def list_native_models(base_url: str) -> list[LmStudioModelInfo]:
    """List all models using LM Studio native metadata, preferring /api/v1."""
    normalized_base = normalize_base_url(base_url)
    reachable, error = await is_reachable(normalized_base)
    if not reachable:
        raise ConnectionError(
            error or f"Cannot connect to LM Studio at {normalized_base}"
        )

    last_error: str | None = None
    async with httpx.AsyncClient(timeout=10.0) as client:
        for endpoint in ("/api/v1/models", "/api/v0/models"):
            try:
                response = await client.get(f"{normalized_base}{endpoint}")
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                models = [
                    info
                    for row in _extract_rows(response.json())
                    if (info := parse_model_row(row)) is not None
                ]
                if models:
                    return models
            except Exception as exc:
                last_error = str(exc)

    raise RuntimeError(last_error or "LM Studio returned no native models")


async def list_chat_models(base_url: str) -> list[LmStudioModelInfo]:
    """List LM Studio chat-capable models."""
    models = await list_native_models(base_url)
    return [model for model in models if model.model_type in CHAT_MODEL_TYPES]


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


async def probe_embedding_dimension(base_url: str, model: str) -> int | None:
    """Probe /v1/embeddings and return the output vector length."""
    normalized_base = normalize_base_url(base_url)
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
) -> list[LmStudioModelInfo]:
    """List LM Studio embedding models and probe dimensions when possible."""
    normalized_base = normalize_base_url(base_url)
    models = [
        model
        for model in await list_native_models(normalized_base)
        if model.model_type in EMBEDDING_MODEL_TYPES
    ]

    selected = str(selected_model or "").strip()
    for model in models:
        if model.dimensions is not None:
            continue
        if selected and model.id != selected:
            continue
        try:
            model.dimensions = await probe_embedding_dimension(
                normalized_base, model.id
            )
        except Exception as exc:
            logger.debug(
                "Failed to probe LM Studio embedding dimensions for %s: %s",
                model.id,
                exc,
            )

    return models


async def get_model_context_length(model: str, base_url: str) -> int | None:
    """Return LM Studio context metadata for a model if discoverable."""
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


async def load_model(
    base_url: str,
    model: str,
    *,
    context_length: int | None = None,
) -> dict[str, Any]:
    """Load an LM Studio model through the native REST API."""
    normalized_base = normalize_base_url(base_url)
    payload: dict[str, Any] = {"model": str(model or "").strip()}
    if not payload["model"]:
        raise ValueError("model is required")
    parsed_context = _coerce_positive_int(context_length)
    if parsed_context:
        payload["context_length"] = parsed_context

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{normalized_base}/api/v1/models/load", json=payload
        )
        if not response.is_success:
            raise RuntimeError(_extract_error_message(response))
        data = response.json() if response.content else {}
        return data if isinstance(data, dict) else {"data": data}


async def unload_model(base_url: str, instance_id: str) -> dict[str, Any]:
    """Unload an LM Studio model instance through the native REST API."""
    normalized_base = normalize_base_url(base_url)
    payload = {"instance_id": str(instance_id or "").strip()}
    if not payload["instance_id"]:
        raise ValueError("instance_id is required")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{normalized_base}/api/v1/models/unload", json=payload
        )
        if not response.is_success:
            raise RuntimeError(_extract_error_message(response))
        data = response.json() if response.content else {}
        return data if isinstance(data, dict) else {"data": data}
