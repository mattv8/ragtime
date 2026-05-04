"""
oMLX OpenAI-compatible model discovery helpers.

oMLX exposes standard OpenAI-compatible endpoints under /v1. Keep the small
provider-specific pieces here so routes and runtime code can use a stable shape.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = "http://host.docker.internal:8000"
OPENAI_COMPATIBLE_CHAT_ENDPOINTS = ["/chat/completions"]
OPENAI_COMPATIBLE_EMBEDDING_ENDPOINTS = ["/embeddings"]
CAPABILITY_CACHE_TTL_SECONDS = 30.0
CAPABILITY_PROBE_CONCURRENCY = 2
CHAT_MODEL_TYPES = {"llm", "vlm"}
EMBEDDING_MODEL_TYPES = {"embedding", "embeddings"}


@dataclass
class OmlxModelInfo:
    """Model metadata discovered from oMLX OpenAI-compatible APIs."""

    id: str
    name: str
    context_limit: Optional[int] = None
    dimensions: Optional[int] = None
    capabilities: list[str] | None = None
    supported_endpoints: list[str] | None = None
    model_type: str = ""
    engine_type: str = ""
    loaded: bool = False


@dataclass
class _CapabilityCacheEntry:
    expires_at: float
    chat_models: list[OmlxModelInfo]
    embedding_models: list[OmlxModelInfo]


_CAPABILITY_CACHE: dict[tuple[str, str, str], _CapabilityCacheEntry] = {}


def normalize_base_url(base_url: str | None, default: str = DEFAULT_BASE_URL) -> str:
    """Normalize an oMLX base URL for endpoint joins."""
    value = str(base_url or "").strip() or default
    return value.rstrip("/")


def _auth_headers(api_key: str | None) -> dict[str, str]:
    key = str(api_key or "").strip()
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def _cache_key(
    base_url: str,
    api_key: str | None,
    extra_embedding_candidates: tuple[str, ...] = (),
) -> tuple[str, str, str]:
    key = str(api_key or "").strip()
    fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12] if key else ""
    return (
        normalize_base_url(base_url),
        fingerprint,
        ",".join(extra_embedding_candidates),
    )


def _clone_model_info(model: OmlxModelInfo) -> OmlxModelInfo:
    return OmlxModelInfo(
        id=model.id,
        name=model.name,
        context_limit=model.context_limit,
        dimensions=model.dimensions,
        capabilities=list(model.capabilities) if model.capabilities else None,
        supported_endpoints=(
            list(model.supported_endpoints) if model.supported_endpoints else None
        ),
        model_type=model.model_type,
        engine_type=model.engine_type,
        loaded=model.loaded,
    )


def _clone_model_list(models: list[OmlxModelInfo]) -> list[OmlxModelInfo]:
    return [_clone_model_info(model) for model in models]


def _as_chat_model(model: OmlxModelInfo) -> OmlxModelInfo:
    chat_model = _clone_model_info(model)
    chat_model.capabilities = ["chat"]
    chat_model.supported_endpoints = list(OPENAI_COMPATIBLE_CHAT_ENDPOINTS)
    return chat_model


def _model_type_tokens(model: OmlxModelInfo) -> set[str]:
    return {
        str(value or "").strip().lower()
        for value in (model.model_type, model.engine_type)
        if str(value or "").strip()
    }


def _is_embedding_model(model: OmlxModelInfo) -> bool:
    return bool(_model_type_tokens(model) & EMBEDDING_MODEL_TYPES)


def _is_chat_model(model: OmlxModelInfo) -> bool:
    return bool(_model_type_tokens(model) & CHAT_MODEL_TYPES)


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("data") or payload.get("models")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _parse_model_row(row: dict[str, Any]) -> OmlxModelInfo | None:
    model_id = str(row.get("id") or row.get("model") or "").strip()
    if not model_id:
        return None
    return OmlxModelInfo(
        id=model_id,
        name=str(row.get("name") or model_id),
        model_type=str(row.get("model_type") or "").strip().lower(),
        engine_type=str(row.get("engine_type") or "").strip().lower(),
        loaded=bool(row.get("loaded")),
        supported_endpoints=list(OPENAI_COMPATIBLE_CHAT_ENDPOINTS),
    )


def _parse_status_model_row(row: dict[str, Any]) -> OmlxModelInfo | None:
    model = _parse_model_row(row)
    if model is None:
        return None
    model.context_limit = _coerce_positive_int(
        row.get("max_context_window") or row.get("max_tokens")
    )
    return model


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


def _is_invalid_request_error(response: httpx.Response, text: str) -> bool:
    if response.status_code != 400:
        return False
    try:
        payload = response.json()
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    if not isinstance(error, dict):
        return False
    error_type = str(error.get("type") or "").strip().lower()
    error_message = str(error.get("message") or "").strip().lower()
    return error_type == "invalid_request_error" and text in error_message


async def is_reachable(
    base_url: str, api_key: str | None = None, timeout: float = 5.0
) -> tuple[bool, Optional[str]]:
    """Check oMLX reachability using the standard OpenAI models endpoint."""
    normalized_base = normalize_base_url(base_url)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"{normalized_base}/v1/models", headers=_auth_headers(api_key)
            )
            if response.is_success:
                return True, None
            return (
                False,
                f"oMLX at {normalized_base} returned HTTP {response.status_code}: {_extract_error_message(response)}",
            )
    except httpx.ConnectError:
        return (
            False,
            f"Failed to connect to oMLX at {normalized_base}. Make sure the oMLX server is running and reachable.",
        )
    except httpx.TimeoutException:
        return False, f"Connection to oMLX at {normalized_base} timed out."
    except Exception as exc:
        return False, f"Error connecting to oMLX at {normalized_base}: {exc}"


async def list_models(base_url: str, api_key: str | None = None) -> list[OmlxModelInfo]:
    """List models using the standard OpenAI-compatible /v1/models endpoint."""
    normalized_base = normalize_base_url(base_url)
    reachable, error = await is_reachable(normalized_base, api_key=api_key)
    if not reachable:
        raise ConnectionError(error or f"Cannot connect to oMLX at {normalized_base}")

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{normalized_base}/v1/models", headers=_auth_headers(api_key)
        )
        if not response.is_success:
            raise RuntimeError(_extract_error_message(response))
        return [
            info
            for row in _extract_rows(response.json())
            if (info := _parse_model_row(row)) is not None
        ]


async def list_status_models(
    base_url: str, api_key: str | None = None
) -> list[OmlxModelInfo]:
    """List models with oMLX status metadata, including model_type."""
    normalized_base = normalize_base_url(base_url)
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{normalized_base}/v1/models/status", headers=_auth_headers(api_key)
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        payload = response.json()
        rows = []
        if isinstance(payload, dict):
            models = payload.get("models")
            if isinstance(models, list):
                rows = [row for row in models if isinstance(row, dict)]
        return [
            info for row in rows if (info := _parse_status_model_row(row)) is not None
        ]


async def list_chat_models(
    base_url: str, api_key: str | None = None
) -> list[OmlxModelInfo]:
    """List oMLX chat-capable models using endpoint capability probes."""
    chat_models, _embedding_models = await classify_models(base_url, api_key=api_key)
    return chat_models


async def probe_chat_capability(
    base_url: str, model: str, api_key: str | None = None
) -> bool:
    """Probe /v1/chat/completions and return whether the model can chat."""
    normalized_base = normalize_base_url(base_url)
    model_id = str(model or "").strip()
    if not model_id:
        return False

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{normalized_base}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Reply with OK only."}],
                "max_tokens": 1,
                "temperature": 0,
            },
            headers=_auth_headers(api_key),
        )
        if _is_invalid_request_error(response, "not an llm / chat model"):
            return False
        response.raise_for_status()
        return response.is_success


async def probe_embedding_dimension(
    base_url: str, model: str, api_key: str | None = None
) -> int | None:
    """Probe /v1/embeddings and return the output vector length."""
    normalized_base = normalize_base_url(base_url)
    model_id = str(model or "").strip()
    if not model_id:
        return None

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{normalized_base}/v1/embeddings",
            json={"model": model_id, "input": "test"},
            headers=_auth_headers(api_key),
        )
        if response.status_code in {400, 507}:
            if response.status_code == 507:
                # oMLX can return 507 for models that cannot serve embeddings.
                return None
            if _is_invalid_request_error(response, "not an embedding model"):
                return None
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return extract_embedding_dimension(payload)
    return None


async def _classify_one_model(
    base_url: str,
    model: OmlxModelInfo,
    api_key: str | None,
    semaphore: asyncio.Semaphore,
    assume_embedding: bool = False,
) -> tuple[OmlxModelInfo | None, OmlxModelInfo | None]:
    async with semaphore:
        try:
            dimensions = await probe_embedding_dimension(
                base_url, model.id, api_key=api_key
            )
        except Exception as exc:
            logger.debug(
                "Failed to probe oMLX embedding dimensions for %s: %s",
                model.id,
                exc,
            )
            dimensions = None

        if dimensions is None and not assume_embedding:
            return None, None

        embedding_model = _clone_model_info(model)
        embedding_model.dimensions = dimensions
        embedding_model.capabilities = ["embeddings"]
        embedding_model.supported_endpoints = list(
            OPENAI_COMPATIBLE_EMBEDDING_ENDPOINTS
        )

        if assume_embedding:
            return None, embedding_model

        try:
            chat_capable = await probe_chat_capability(
                base_url, model.id, api_key=api_key
            )
        except Exception as exc:
            logger.debug(
                "Failed to probe oMLX chat capability for %s: %s", model.id, exc
            )
            chat_capable = False

        if not chat_capable:
            return None, embedding_model

        chat_model = _clone_model_info(embedding_model)
        chat_model.capabilities = ["chat", "embeddings"]
        chat_model.supported_endpoints = list(
            OPENAI_COMPATIBLE_CHAT_ENDPOINTS + OPENAI_COMPATIBLE_EMBEDDING_ENDPOINTS
        )
        return chat_model, embedding_model


async def classify_models(
    base_url: str,
    api_key: str | None = None,
    extra_embedding_candidates: set[str] | None = None,
) -> tuple[list[OmlxModelInfo], list[OmlxModelInfo]]:
    """Classify oMLX models by probing OpenAI-compatible endpoints."""
    normalized_base = normalize_base_url(base_url)
    extra_candidates = tuple(sorted(extra_embedding_candidates or set()))
    cache_key = _cache_key(normalized_base, api_key, extra_candidates)
    cached = _CAPABILITY_CACHE.get(cache_key)
    now = time.monotonic()
    if cached and cached.expires_at > now:
        return (
            _clone_model_list(cached.chat_models),
            _clone_model_list(cached.embedding_models),
        )

    models = await list_status_models(normalized_base, api_key=api_key)
    if not models:
        models = await list_models(normalized_base, api_key=api_key)

    typed_embedding_models = [model for model in models if _is_embedding_model(model)]
    typed_chat_models = [model for model in models if _is_chat_model(model)]
    if typed_embedding_models or typed_chat_models:
        embedding_candidates = typed_embedding_models
        if extra_candidates:
            embedding_candidates = [
                model
                for model in models
                if model.id in {candidate.id for candidate in typed_embedding_models}
                or model.id in extra_candidates
            ]
    else:
        embedding_candidates = [
            model for model in models if model.id in extra_candidates
        ]

    semaphore = asyncio.Semaphore(CAPABILITY_PROBE_CONCURRENCY)
    results = await asyncio.gather(
        *[
            _classify_one_model(
                normalized_base,
                model,
                api_key,
                semaphore,
                assume_embedding=_is_embedding_model(model),
            )
            for model in embedding_candidates
        ]
    )
    chat_models_by_id = {
        chat.id: chat for chat, _embedding in results if chat is not None
    }
    embedding_models = [
        embedding for _chat, embedding in results if embedding is not None
    ]
    embedding_only_ids = {
        embedding.id
        for chat, embedding in results
        if embedding is not None and chat is None
    }
    chat_models = []
    source_chat_models = typed_chat_models if typed_chat_models else models
    for model in source_chat_models:
        if model.id in embedding_only_ids:
            continue
        chat_models.append(chat_models_by_id.get(model.id) or _as_chat_model(model))
    _CAPABILITY_CACHE[cache_key] = _CapabilityCacheEntry(
        expires_at=now + CAPABILITY_CACHE_TTL_SECONDS,
        chat_models=_clone_model_list(chat_models),
        embedding_models=_clone_model_list(embedding_models),
    )
    return chat_models, embedding_models


async def list_embedding_models(
    base_url: str,
    selected_model: str | None = None,
    api_key: str | None = None,
) -> list[OmlxModelInfo]:
    """List oMLX embedding-capable models using endpoint capability probes."""
    selected = str(selected_model or "").strip()
    extra_candidates = {selected} if selected else None
    _chat_models, embedding_models = await classify_models(
        base_url,
        api_key=api_key,
        extra_embedding_candidates=extra_candidates,
    )
    if selected:
        return [model for model in embedding_models if model.id == selected]
    return embedding_models


async def get_model_context_length(
    model: str, base_url: str, api_key: str | None = None
) -> int | None:
    """Return context metadata for an oMLX model if discoverable."""
    _ = (model, base_url, api_key)
    return None
