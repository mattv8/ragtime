"""Copilot HMAC token management — exchange and proactive refresh."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from ragtime.core.app_settings import invalidate_settings_cache
from ragtime.core.logging import get_logger
from ragtime.indexer.repository import repository

logger = get_logger(__name__)

# Refresh the HMAC token when it expires within this window.
_REFRESH_BUFFER = timedelta(minutes=5)

# Singleflight lock: prevents concurrent callers from all triggering a refresh.
_refresh_lock = asyncio.Lock()


async def exchange_github_token_for_copilot_token(
    github_token: str,
) -> tuple[str, Optional[datetime]]:
    """Exchange a GitHub OAuth token for a Copilot HMAC bearer token.

    Returns ``(copilot_token, expires_at)``; raises on failure.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers={
                "Authorization": f"token {github_token}",
                "Accept": "application/json",
                "User-Agent": "ragtime",
            },
        )
        response.raise_for_status()
        data = response.json()

    copilot_token = str(data.get("token", "")).strip()
    if not copilot_token:
        raise ValueError("GitHub Copilot token exchange did not return a token")

    expires_at: Optional[datetime] = None
    raw_expires = data.get("expires_at")
    if isinstance(raw_expires, (int, float)):
        try:
            expires_at = datetime.fromtimestamp(float(raw_expires), tz=timezone.utc)
        except Exception:
            pass

    return copilot_token, expires_at


async def ensure_copilot_token_fresh() -> Optional[str]:
    """Proactively refresh the stored Copilot HMAC token if it is near expiry.

    Returns the (possibly refreshed) access token, or ``None`` when no
    Copilot OAuth connection exists.  Settings and cache are updated
    transparently on refresh.
    """
    settings = await repository.get_settings()
    access_token = (settings.github_copilot_access_token or "").strip()
    refresh_token = (settings.github_copilot_refresh_token or "").strip()

    if not access_token:
        return None

    expires_at = settings.github_copilot_token_expires_at
    if expires_at is None or not refresh_token:
        # No expiry tracked or no refresh token — use what we have.
        return access_token

    now = datetime.now(timezone.utc)
    if isinstance(expires_at, datetime) and expires_at > now + _REFRESH_BUFFER:
        return access_token  # Still fresh.

    # Token is expired or about to expire — serialize concurrent refresh attempts.
    async with _refresh_lock:
        # Re-read settings after acquiring the lock; another caller may have
        # already completed the refresh while we were waiting.
        settings = await repository.get_settings()
        access_token = (settings.github_copilot_access_token or "").strip()
        refresh_token = (settings.github_copilot_refresh_token or "").strip()
        expires_at = settings.github_copilot_token_expires_at

        now = datetime.now(timezone.utc)
        if isinstance(expires_at, datetime) and expires_at > now + _REFRESH_BUFFER:
            return access_token  # Refreshed by another caller.

        try:
            new_token, new_expires = await exchange_github_token_for_copilot_token(
                refresh_token
            )
            await repository.update_settings(
                {
                    "github_copilot_access_token": new_token,
                    "github_copilot_token_expires_at": new_expires,
                }
            )
            invalidate_settings_cache()
            logger.info("Refreshed Copilot HMAC token (expires %s)", new_expires)
            return new_token
        except Exception as exc:
            logger.warning("Failed to refresh Copilot HMAC token: %s", exc)
            return access_token
