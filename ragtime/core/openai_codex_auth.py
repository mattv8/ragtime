"""OpenAI Codex subscription OAuth helpers."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

OPENAI_CODEX_ISSUER = "https://auth.openai.com"
OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
OPENAI_CODEX_RESPONSES_ENDPOINT = f"{OPENAI_CODEX_DEFAULT_BASE_URL}/responses"
OPENAI_CODEX_MODELS_ENDPOINT = f"{OPENAI_CODEX_DEFAULT_BASE_URL}/models"
OPENAI_CODEX_MODELS_CLIENT_VERSION = "1.0.0"


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        value = json.loads(decoded.decode("utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def extract_openai_codex_account_id(tokens: dict[str, Any]) -> str:
    """Extract the ChatGPT account identifier from OpenAI OAuth token claims."""
    candidates: list[Any] = [
        tokens.get("chatgpt_account_id"),
        tokens.get("https://api.openai.com/auth.chatgpt_account_id"),
    ]

    id_token = str(tokens.get("id_token") or "").strip()
    if id_token:
        claims = _decode_jwt_payload(id_token)
        candidates.extend(
            [
                claims.get("chatgpt_account_id"),
                claims.get("https://api.openai.com/auth.chatgpt_account_id"),
            ]
        )
        organizations = claims.get("organizations")
        if isinstance(organizations, list) and organizations:
            first = organizations[0]
            if isinstance(first, dict):
                candidates.append(first.get("id"))

    for candidate in candidates:
        value = str(candidate or "").strip()
        if value:
            return value
    return ""


async def refresh_openai_codex_tokens(refresh_token: str) -> dict[str, Any]:
    """Refresh OpenAI Codex OAuth tokens."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{OPENAI_CODEX_ISSUER}/oauth/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": OPENAI_CODEX_CLIENT_ID,
            },
        )
        response.raise_for_status()
        data = response.json()
    return data if isinstance(data, dict) else {}


async def ensure_openai_codex_token_fresh(settings: Any | None = None, repository: Any | None = None) -> str:
    """Return a usable Codex access token, refreshing persisted credentials when needed."""
    repo = repository
    if repo is None:
        from ragtime.indexer.repository import repository as default_repository

        repo = default_repository

    if settings is None:
        settings = await repo.get_settings()

    access_token = str(getattr(settings, "openai_codex_access_token", "") or "").strip()
    refresh_token = str(getattr(settings, "openai_codex_refresh_token", "") or "").strip()
    expires_at = getattr(settings, "openai_codex_token_expires_at", None)

    if access_token and isinstance(expires_at, datetime):
        expires_utc = expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=timezone.utc)
        if expires_utc > datetime.now(timezone.utc) + timedelta(minutes=5):
            return access_token

    if not refresh_token:
        return access_token

    tokens = await refresh_openai_codex_tokens(refresh_token)
    refreshed_access = str(tokens.get("access_token") or "").strip()
    if not refreshed_access:
        return access_token

    refreshed_refresh = str(tokens.get("refresh_token") or refresh_token).strip()
    expires_in = int(tokens.get("expires_in") or 3600)
    refreshed_expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(expires_in, 60))
    account_id = extract_openai_codex_account_id(tokens) or str(getattr(settings, "openai_codex_account_id", "") or "").strip()

    await repo.update_settings(
        {
            "openai_codex_access_token": refreshed_access,
            "openai_codex_refresh_token": refreshed_refresh,
            "openai_codex_token_expires_at": refreshed_expires_at,
            "openai_codex_account_id": account_id,
        }
    )
    from ragtime.core.app_settings import invalidate_settings_cache

    invalidate_settings_cache()
    return refreshed_access
