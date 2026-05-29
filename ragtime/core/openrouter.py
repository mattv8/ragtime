"""OpenRouter model catalog helpers."""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{DEFAULT_BASE_URL}/models"
OPENROUTER_EMBEDDING_MODELS_URL = f"{DEFAULT_BASE_URL}/embeddings/models"
OPENROUTER_EMBEDDINGS_URL = f"{DEFAULT_BASE_URL}/embeddings"

CHAT_INPUT_TOKENS = {"text", "image", "images", "vision"}
CHAT_ENDPOINT_TOKENS = {"/chat/completions", "chat/completions", "/responses", "responses"}
VISION_INPUT_TOKENS = {"image", "images", "vision"}
TEXT_OUTPUT_TOKENS = {"text"}
EMBEDDING_OUTPUT_TOKENS = {"embedding", "embeddings", "vector", "vectors"}
VISION_CAPABILITY_TOKENS = {"image", "images", "image_input", "vision", "vlm"}
EMBEDDING_CAPABILITY_TOKENS = {"embedding", "embeddings"}
EMBEDDING_ENDPOINT_TOKENS = {"/embeddings", "embeddings"}
EXCLUDED_CHAT_MODEL_ID_TERMS = ("embed", "embedding", "rerank", "moderation", "whisper", "tts")


async def _list_catalog_rows(url: str, api_key: str, *, timeout: float) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()

    data = response.json()
    rows = data.get("data", []) if isinstance(data, dict) else []
    return [row for row in rows if isinstance(row, dict)]


async def list_models(api_key: str, *, timeout: float = 20.0) -> list[dict[str, Any]]:
    """Fetch OpenRouter chat/vision model catalog rows."""
    return await _list_catalog_rows(OPENROUTER_MODELS_URL, api_key, timeout=timeout)


async def list_embedding_models(api_key: str, *, timeout: float = 20.0) -> list[dict[str, Any]]:
    """Fetch OpenRouter embedding model catalog rows."""
    return await _list_catalog_rows(OPENROUTER_EMBEDDING_MODELS_URL, api_key, timeout=timeout)


def _tokens(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(item or "").strip().lower() for item in values if str(item or "").strip()}


def input_modalities(row: dict[str, Any]) -> set[str]:
    """Return input modality tokens from OpenRouter-style metadata."""
    architecture = row.get("architecture")
    if isinstance(architecture, dict):
        values = _tokens(architecture.get("input_modalities"))
        if values:
            return values

    modalities = row.get("modalities")
    if isinstance(modalities, dict):
        values = _tokens(modalities.get("input"))
        if values:
            return values

    return _tokens(row.get("supported_input_modalities"))


def output_modalities(row: dict[str, Any]) -> set[str]:
    """Return output modality tokens from OpenRouter-style metadata."""
    architecture = row.get("architecture")
    if isinstance(architecture, dict):
        values = _tokens(architecture.get("output_modalities"))
        if values:
            return values

    modalities = row.get("modalities")
    if isinstance(modalities, dict):
        values = _tokens(modalities.get("output"))
        if values:
            return values

    return _tokens(row.get("supported_output_modalities"))


def supported_endpoints(row: dict[str, Any]) -> set[str]:
    """Return normalized endpoint tokens from provider metadata."""
    endpoints = row.get("supportedEndpoints")
    if endpoints is None:
        endpoints = row.get("supported_endpoints")
    if endpoints is None:
        endpoints = row.get("endpoints")
    return _tokens(endpoints)


def supported_endpoints_list(row: dict[str, Any]) -> list[str]:
    """Return endpoint tokens in a stable list shape for API responses."""
    return sorted(supported_endpoints(row))


def capabilities(row: dict[str, Any]) -> list[str]:
    """Return normalized capability tokens from provider metadata."""
    values: list[str] = []
    capabilities_obj = row.get("capabilities")
    if isinstance(capabilities_obj, list):
        values.extend(str(item).strip().lower() for item in capabilities_obj if item)
    elif isinstance(capabilities_obj, dict):
        supports_obj = capabilities_obj.get("supports")
        if isinstance(supports_obj, list):
            values.extend(str(item).strip().lower() for item in supports_obj if item)
        elif isinstance(supports_obj, dict):
            values.extend(str(flag).strip().lower() for flag, enabled in supports_obj.items() if enabled)
        values.extend(str(flag).strip().lower() for flag, enabled in capabilities_obj.items() if enabled is True and flag != "supports")

    if input_modalities(row) & VISION_INPUT_TOKENS:
        values.append("image_input")
    if output_modalities(row) & EMBEDDING_OUTPUT_TOKENS or supported_endpoints(row) & EMBEDDING_ENDPOINT_TOKENS:
        values.append("embeddings")

    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def supports_vision(row: dict[str, Any]) -> bool:
    """Return whether catalog metadata explicitly reports image input support."""
    outputs = output_modalities(row)
    if outputs and not outputs & TEXT_OUTPUT_TOKENS:
        return False
    return bool(input_modalities(row) & VISION_INPUT_TOKENS or set(capabilities(row)) & VISION_CAPABILITY_TOKENS)


def supports_embeddings(row: dict[str, Any]) -> bool:
    """Return whether catalog metadata explicitly reports embedding support."""
    return bool(
        output_modalities(row) & EMBEDDING_OUTPUT_TOKENS
        or supported_endpoints(row) & EMBEDDING_ENDPOINT_TOKENS
        or set(capabilities(row)) & EMBEDDING_CAPABILITY_TOKENS
    )


def supports_chat(row: dict[str, Any]) -> bool:
    """Return whether catalog metadata describes a chat-capable model."""
    candidate_id = model_id(row).lower()
    if not candidate_id:
        return False

    endpoints = supported_endpoints(row)
    if endpoints and not endpoints & CHAT_ENDPOINT_TOKENS:
        return False
    if supports_embeddings(row) and not endpoints & CHAT_ENDPOINT_TOKENS:
        return False
    if any(term in candidate_id for term in EXCLUDED_CHAT_MODEL_ID_TERMS):
        return False

    inputs = input_modalities(row)
    if inputs:
        return bool(inputs & CHAT_INPUT_TOKENS)
    return True


def model_id(row: dict[str, Any]) -> str:
    """Return the catalog model identifier."""
    return str(row.get("id") or "").strip()


def model_name(row: dict[str, Any]) -> str:
    """Return a display name, falling back to the model identifier."""
    row_id = model_id(row)
    name = str(row.get("name") or "").strip()
    return name or row_id


def tokenizer(row: dict[str, Any]) -> str | None:
    """Return the tokenizer/family token from architecture metadata."""
    architecture = row.get("architecture")
    if isinstance(architecture, dict) and architecture.get("tokenizer"):
        return str(architecture.get("tokenizer"))
    return None


def supported_parameters(row: dict[str, Any]) -> list[str]:
    """Return normalized supported parameter names."""
    parameters = row.get("supported_parameters")
    if not isinstance(parameters, list):
        return []
    return [str(item).strip().lower() for item in parameters if str(item or "").strip()]


def positive_int(value: Any) -> int | None:
    """Coerce positive numeric metadata to an integer."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def context_limit(row: dict[str, Any]) -> int | None:
    limit = row.get("limit")
    if isinstance(limit, dict):
        parsed = positive_int(limit.get("context"))
        if parsed is not None:
            return parsed

    parsed = positive_int(row.get("context_length"))
    if parsed is not None:
        return parsed

    top_provider = row.get("top_provider")
    if isinstance(top_provider, dict):
        return positive_int(top_provider.get("context_length"))
    return None


def created(row: dict[str, Any]) -> int | None:
    return positive_int(row.get("created"))


def embedding_dimensions(row: dict[str, Any]) -> int | None:
    limit = row.get("limit")
    if not isinstance(limit, dict):
        return None
    return positive_int(limit.get("output"))
