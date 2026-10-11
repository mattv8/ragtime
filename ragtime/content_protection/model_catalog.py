"""Jev model catalog lookup for configured classifier transports."""

from __future__ import annotations

from typing import Any, Literal

import httpx

from ragtime.content_protection.models import ContentProtectionError
from ragtime.content_protection.provider import _resolved_model

Transport = Literal["auto", "typesafe", "openrouter"]
ResolvedTransport = Literal["typesafe", "openrouter"]

_CATALOG_TIMEOUT_SECONDS = 10.0
_TYPE_SAFE_MODELS_URL = "https://api.typesafe.ai/v1/models"
_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models?output_modalities=decisions"
_DEFAULT_MODEL = "jev-latest"
_MISSING_CREDENTIALS_MESSAGE = "Enter or save a TypeSafe or OpenRouter API key first."
_MAX_API_KEY_LENGTH = 512


def _make_http_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


def detect_key_transport(key: str) -> ResolvedTransport | None:
    """Identify a Jev vendor from a known API key prefix."""
    normalized = key.strip()
    if normalized.startswith("sk-or-"):
        return "openrouter"
    if normalized.startswith(("apikey_", "ts_", "ts-")):
        return "typesafe"
    return None


def _failure(transport: Transport, message: str) -> dict[str, Any]:
    return {"success": False, "transport": transport, "models": [], "default_model": _DEFAULT_MODEL, "message": message}


def _select_credential(transport: Transport, api_key: str | None, settings: dict[str, Any]) -> tuple[ResolvedTransport, str] | str:
    draft_key = (api_key or "").strip()
    if len(draft_key) > _MAX_API_KEY_LENGTH:
        return "API key is too long."
    if draft_key:
        detected = detect_key_transport(draft_key)
        if transport == "auto":
            return (detected, draft_key) if detected else "Choose TypeSafe or OpenRouter for this key."
        if detected is not None and detected != transport:
            return "The API key does not match the selected provider."
        return transport, draft_key

    typesafe_key = str(settings.get("typesafe_api_key") or "").strip()
    openrouter_key = str(settings.get("openrouter_api_key") or "").strip()
    if transport == "auto":
        if typesafe_key:
            return "typesafe", typesafe_key
        if openrouter_key:
            return "openrouter", openrouter_key
    elif transport == "typesafe" and typesafe_key:
        return "typesafe", typesafe_key
    elif transport == "openrouter" and openrouter_key:
        return "openrouter", openrouter_key
    return _MISSING_CREDENTIALS_MESSAGE


def _model_rows(transport: ResolvedTransport, payload: object) -> list[dict[str, str | None]]:
    key = "models" if transport == "typesafe" else "data"
    entries = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("invalid model payload")

    rows: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        identifier = entry.get("name") if transport == "typesafe" else entry.get("id")
        if not isinstance(identifier, str):
            continue
        try:
            resolved_identifier = _resolved_model(transport, identifier)
        except ContentProtectionError:
            continue
        if resolved_identifier in seen:
            continue
        seen.add(resolved_identifier)
        catalog_identifier = _DEFAULT_MODEL if resolved_identifier == _resolved_model(transport, _DEFAULT_MODEL) else identifier
        rows.append(
            {
                "id": catalog_identifier,
                "name": entry["name"] if isinstance(entry.get("name"), str) else catalog_identifier,
                "description": entry["description"] if isinstance(entry.get("description"), str) else None,
                "release_date": entry["release_date"] if isinstance(entry.get("release_date"), str) else None,
            }
        )
    return rows


def _with_default_model(rows: list[dict[str, str | None]]) -> list[dict[str, str | None]]:
    default = {"id": _DEFAULT_MODEL, "name": _DEFAULT_MODEL, "description": None, "release_date": None}
    vendor_default = next((row for row in rows if row["id"] == _DEFAULT_MODEL), default)
    return [vendor_default, *[row for row in rows if row["id"] != _DEFAULT_MODEL]]


async def list_jev_models(transport: Transport, api_key: str | None, settings: dict[str, Any]) -> dict[str, Any]:
    """Load validated Jev model choices without exposing provider details."""
    selected = _select_credential(transport, api_key, settings)
    if isinstance(selected, str):
        return _failure(transport, selected)
    resolved_transport, credential = selected
    vendor = "TypeSafe" if resolved_transport == "typesafe" else "OpenRouter"
    url = _TYPE_SAFE_MODELS_URL if resolved_transport == "typesafe" else _OPENROUTER_MODELS_URL
    try:
        async with _make_http_client(_CATALOG_TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers={"Authorization": f"Bearer {credential}"})
    except httpx.HTTPError:
        return _failure(resolved_transport, f"Could not load models from {vendor}.")
    if response.status_code in {401, 403}:
        return _failure(resolved_transport, f"The {vendor} API key was rejected.")
    if response.is_error:
        return _failure(resolved_transport, f"Could not load models from {vendor}.")
    try:
        rows = _with_default_model(_model_rows(resolved_transport, response.json()))
    except (TypeError, ValueError, httpx.DecodingError):
        return _failure(resolved_transport, f"Could not load models from {vendor}.")
    return {"success": True, "transport": resolved_transport, "models": rows, "default_model": _DEFAULT_MODEL, "message": None}
