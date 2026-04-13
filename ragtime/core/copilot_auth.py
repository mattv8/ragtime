"""Copilot HMAC token management — exchange and proactive refresh."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import httpx

from ragtime.core.app_settings import invalidate_settings_cache
from ragtime.core.logging import get_logger
from ragtime.indexer.repository import repository

logger = get_logger(__name__)

# Refresh the HMAC token when it expires within this window.
_REFRESH_BUFFER = timedelta(minutes=5)

# Singleflight lock: prevents concurrent callers from all triggering a refresh.
_refresh_lock = asyncio.Lock()
_background_refresh_task_holder: dict[str, Optional[asyncio.Task]] = {"task": None}

# GitHub App client ID used for the OAuth device flow (Copilot extension).
_GITHUB_COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"


def _extract_copilot_token_state(
    settings: object,
) -> tuple[str, str, str, Optional[datetime]]:
    """Extract normalized Copilot token state from settings."""
    access_token = (getattr(settings, "github_copilot_access_token", "") or "").strip()
    refresh_token = (
        getattr(settings, "github_copilot_refresh_token", "") or ""
    ).strip()
    oauth_refresh_token = (
        getattr(settings, "github_copilot_oauth_refresh_token", "") or ""
    ).strip()
    expires_at = getattr(settings, "github_copilot_token_expires_at", None)
    return access_token, refresh_token, oauth_refresh_token, expires_at


def _is_token_expiring_soon(expires_at: Optional[datetime]) -> bool:
    """Return True when a token is at/inside the refresh buffer."""
    if not isinstance(expires_at, datetime):
        return False
    now = datetime.now(timezone.utc)
    return expires_at <= now + _REFRESH_BUFFER


def _is_token_fresh(expires_at: Optional[datetime]) -> bool:
    """Return True when the token is still usable outside the refresh buffer."""
    return isinstance(expires_at, datetime) and not _is_token_expiring_soon(expires_at)


def _is_token_expired(expires_at: Optional[datetime]) -> bool:
    """Return True when the token has already expired."""
    if not isinstance(expires_at, datetime):
        return False
    now = datetime.now(timezone.utc)
    return expires_at <= now


def _build_copilot_settings_update(
    *,
    access_token: str,
    expires_at: Optional[datetime],
    refresh_token: Optional[str] = None,
    oauth_refresh_token: Optional[str] = None,
) -> dict[str, object]:
    """Build a settings update payload for persisted Copilot token state."""
    updates: dict[str, object] = {
        "github_copilot_access_token": access_token,
        "github_copilot_token_expires_at": expires_at,
    }
    if refresh_token is not None:
        updates["github_copilot_refresh_token"] = refresh_token
    if oauth_refresh_token is not None:
        updates["github_copilot_oauth_refresh_token"] = oauth_refresh_token
    return updates


async def _persist_copilot_token_state(
    *,
    access_token: str,
    expires_at: Optional[datetime],
    log_message: str,
    refresh_token: Optional[str] = None,
    oauth_refresh_token: Optional[str] = None,
) -> None:
    """Persist refreshed Copilot credentials and invalidate cached settings."""
    await repository.update_settings(
        _build_copilot_settings_update(
            access_token=access_token,
            expires_at=expires_at,
            refresh_token=refresh_token,
            oauth_refresh_token=oauth_refresh_token,
        )
    )
    invalidate_settings_cache()
    logger.info(log_message, expires_at)


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

    await _persist_copilot_token_state(
        access_token=new_hmac,
        refresh_token=new_ghu,
        oauth_refresh_token=rotated_refresh,
        expires_at=hmac_expires,
        log_message="Refreshed Copilot credentials via OAuth refresh_token (HMAC expires %s)",
    )
    return new_hmac


def _should_refresh_in_background(
    access_token: str,
    refresh_token: str,
    oauth_refresh_token: str,
    expires_at: Optional[datetime],
) -> bool:
    """Determine whether a background refresh should be scheduled."""
    if not access_token:
        return bool(refresh_token or oauth_refresh_token)
    return _is_token_expiring_soon(expires_at)


async def ensure_copilot_token_fresh(
    mode: Literal["blocking", "background"] = "blocking",
) -> Optional[str]:
    """Return a Copilot token, optionally refreshing credentials.

    `mode="blocking"` performs the full refresh flow synchronously.
    `mode="background"` schedules refresh asynchronously and returns quickly.
    """
    settings = await repository.get_settings()
    access_token, refresh_token, oauth_refresh_token, expires_at = (
        _extract_copilot_token_state(settings)
    )

    if mode == "background":
        if _should_refresh_in_background(
            access_token,
            refresh_token,
            oauth_refresh_token,
            expires_at,
        ):
            _schedule_background_refresh()
        return access_token or None

    if not access_token:
        # Recover when only the long-lived GitHub OAuth token is present.
        if refresh_token:
            try:
                new_token, new_expires = await exchange_github_token_for_copilot_token(
                    refresh_token
                )
                await _persist_copilot_token_state(
                    access_token=new_token,
                    expires_at=new_expires,
                    log_message="Exchanged GitHub OAuth token for Copilot token (expires %s)",
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

    if expires_at is None or not refresh_token:
        # No expiry tracked or no refresh token — use what we have.
        return access_token

    if _is_token_fresh(expires_at):
        return access_token  # Still fresh.

    # Token is expired or about to expire — serialize concurrent refresh attempts.
    async with _refresh_lock:
        # Re-read settings after acquiring the lock; another caller may have
        # already completed the refresh while we were waiting.
        settings = await repository.get_settings()
        access_token, refresh_token, oauth_refresh_token, expires_at = (
            _extract_copilot_token_state(settings)
        )

        if _is_token_fresh(expires_at):
            return access_token  # Refreshed by another caller.

        try:
            new_token, new_expires = await exchange_github_token_for_copilot_token(
                refresh_token
            )
            await _persist_copilot_token_state(
                access_token=new_token,
                expires_at=new_expires,
                log_message="Refreshed Copilot HMAC token (expires %s)",
            )
            return new_token
        except Exception as exc:
            logger.warning("Failed to refresh Copilot HMAC token: %s", exc)

            # The ghu_* token may have expired — try OAuth refresh_token.
            if oauth_refresh_token:
                result = await _try_oauth_refresh_and_exchange(oauth_refresh_token)
                if result:
                    return result

            # Avoid reusing a known-expired token (causes "unauthorized: token expired").
            if _is_token_expired(expires_at):
                return None
            # If still valid but within refresh buffer, keep using it temporarily.
            return access_token


def _schedule_background_refresh() -> None:
    """Schedule a best-effort refresh task if one is not already running."""
    task = _background_refresh_task_holder.get("task")
    if task and not task.done():
        return

    async def _runner() -> None:
        try:
            await ensure_copilot_token_fresh()
        except Exception as exc:
            logger.debug("Background Copilot token refresh failed: %s", exc)

    try:
        _background_refresh_task_holder["task"] = asyncio.create_task(_runner())
    except RuntimeError:
        # No running event loop in this context.
        return


def is_copilot_token_refresh_in_progress() -> bool:
    """Return whether a background Copilot token refresh task is currently running."""
    task = _background_refresh_task_holder.get("task")
    return bool(task and not task.done())
