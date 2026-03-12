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

# GitHub App client ID used for the OAuth device flow (Copilot extension).
_GITHUB_COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"


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


async def _refresh_oauth_access_token(
    oauth_refresh_token: str,
) -> tuple[str, Optional[str], Optional[datetime]]:
    """Use an OAuth refresh token (ghr_*) to obtain a new access token (ghu_*).

    GitHub Apps with token expiration return short-lived access tokens (~8h)
    alongside a long-lived refresh token (~180 days).  This function uses the
    standard ``refresh_token`` grant to obtain a fresh access token.

    Returns ``(new_access_token, new_refresh_token_or_none, expires_at_or_none)``.
    Raises on failure.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            "https://github.com/login/oauth/access_token",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "ragtime",
            },
            json={
                "client_id": _GITHUB_COPILOT_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": oauth_refresh_token,
            },
        )
        response.raise_for_status()
        data = response.json()

    error = data.get("error")
    if error:
        raise ValueError(
            f"OAuth refresh failed: {error} - {data.get('error_description', '')}"
        )

    new_access_token = str(data.get("access_token", "")).strip()
    if not new_access_token:
        raise ValueError("OAuth refresh did not return an access_token")

    # GitHub may rotate the refresh token on each use.
    new_refresh_token = (data.get("refresh_token") or "").strip() or None

    expires_at: Optional[datetime] = None
    expires_in = data.get("expires_in")
    if isinstance(expires_in, (int, float)) and int(expires_in) > 0:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))

    return new_access_token, new_refresh_token, expires_at


async def _try_oauth_refresh_and_exchange(
    oauth_refresh_token: str,
) -> Optional[str]:
    """Attempt to refresh the expired OAuth access token and exchange for a new HMAC token.

    Returns the new HMAC Copilot token on success, or ``None`` on failure.
    Settings/cache are updated on success.
    """
    try:
        new_ghu, rotated_refresh, _oauth_expires = await _refresh_oauth_access_token(
            oauth_refresh_token
        )
    except Exception as exc:
        logger.warning("OAuth refresh_token grant failed: %s", exc)
        return None

    # Exchange the fresh ghu_* token for an HMAC Copilot token.
    try:
        new_hmac, hmac_expires = await exchange_github_token_for_copilot_token(new_ghu)
    except Exception as exc:
        logger.warning("Copilot token exchange failed after OAuth refresh: %s", exc)
        return None

    update_fields: dict = {
        "github_copilot_access_token": new_hmac,
        "github_copilot_refresh_token": new_ghu,
        "github_copilot_token_expires_at": hmac_expires,
    }
    if rotated_refresh:
        update_fields["github_copilot_oauth_refresh_token"] = rotated_refresh

    await repository.update_settings(update_fields)
    invalidate_settings_cache()
    logger.info(
        "Refreshed Copilot credentials via OAuth refresh_token (HMAC expires %s)",
        hmac_expires,
    )
    return new_hmac


async def ensure_copilot_token_fresh() -> Optional[str]:
    """Proactively refresh the stored Copilot HMAC token if it is near expiry.

    Returns the (possibly refreshed) access token, or ``None`` when no
    Copilot OAuth connection exists.  Settings and cache are updated
    transparently on refresh.
    """
    settings = await repository.get_settings()
    access_token = (settings.github_copilot_access_token or "").strip()
    refresh_token = (settings.github_copilot_refresh_token or "").strip()
    oauth_refresh_token = (
        getattr(settings, "github_copilot_oauth_refresh_token", "") or ""
    )
    oauth_refresh_token = oauth_refresh_token.strip()

    if not access_token:
        # Recover when only the long-lived GitHub OAuth token is present.
        if refresh_token:
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
                logger.info(
                    "Exchanged GitHub OAuth token for Copilot token (expires %s)",
                    new_expires,
                )
                return new_token
            except Exception as exc:
                logger.warning(
                    "Failed to exchange GitHub OAuth token for Copilot token: %s",
                    exc,
                )
                # The ghu_* token may be expired — try the OAuth refresh_token.
                if oauth_refresh_token:
                    result = await _try_oauth_refresh_and_exchange(oauth_refresh_token)
                    if result:
                        return result
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
        oauth_refresh_token = (
            getattr(settings, "github_copilot_oauth_refresh_token", "") or ""
        )
        oauth_refresh_token = oauth_refresh_token.strip()
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

            # The ghu_* token may have expired — try OAuth refresh_token.
            if oauth_refresh_token:
                result = await _try_oauth_refresh_and_exchange(oauth_refresh_token)
                if result:
                    return result

            # Avoid reusing a known-expired token (causes "unauthorized: token expired").
            if isinstance(expires_at, datetime) and expires_at <= now:
                return None
            # If still valid but within refresh buffer, keep using it temporarily.
            return access_token
