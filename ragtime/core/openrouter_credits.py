"""Bounded, best-effort OpenRouter credit monitoring.

This module deliberately keeps provider credentials and raw error responses
inside the HTTP boundary.  Callers receive only coarse state and balances.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ragtime.core.app_settings import get_app_settings
from ragtime.core.logging import get_logger
from ragtime.core.openrouter import DEFAULT_BASE_URL

logger = get_logger(__name__)

OPENROUTER_KEY_URL = f"{DEFAULT_BASE_URL}/key"
OPENROUTER_CREDITS_URL = f"{DEFAULT_BASE_URL}/credits"
_CACHE_SECONDS = 60.0
_REQUEST_TIMEOUT_SECONDS = 10.0
_MAX_BACKOFF_SECONDS = 300.0

_cached_status: dict[str, Any] | None = None
_cached_at: datetime | None = None
_refresh_task: asyncio.Task[dict[str, Any]] | None = None
_monitor_task: asyncio.Task[None] | None = None
_failure_count = 0
_next_refresh_at: datetime | None = None
_payment_required_note: str | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _settings_value(settings: object, name: str, default: Any = None) -> Any:
    if isinstance(settings, dict):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _base_status(*, enabled: bool, threshold: float) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "state": "unknown",
        "key_remaining_usd": None,
        "wallet_remaining_usd": None,
        "threshold_usd": threshold,
        "checked_at": None,
        "stale": False,
        "warning": None,
    }


def _with_payment_warning(status: dict[str, Any]) -> dict[str, Any]:
    result = dict(status)
    if _payment_required_note:
        result["warning"] = _payment_required_note
    return result


def _status_from_cache(*, stale: bool) -> dict[str, Any] | None:
    if _cached_status is None:
        return None
    result = dict(_cached_status)
    result["stale"] = stale
    return _with_payment_warning(result)


def _remaining_from_key(payload: object) -> float | None:
    data = payload.get("data", payload) if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return None
    for name in ("limit_remaining", "remaining", "remaining_credits"):
        value = _number(data.get(name))
        if value is not None:
            return value
    limit, usage = _number(data.get("limit")), _number(data.get("usage"))
    return limit - usage if limit is not None and usage is not None else None


def _remaining_from_wallet(payload: object) -> float | None:
    data = payload.get("data", payload) if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return None
    for name in ("remaining", "remaining_credits", "credits_remaining", "balance"):
        value = _number(data.get(name))
        if value is not None:
            return value
    credits, usage = _number(data.get("total_credits")), _number(data.get("total_usage"))
    return credits - usage if credits is not None and usage is not None else None


async def _get_json(client: httpx.AsyncClient, url: str, key: str) -> tuple[int, object | None]:
    response = await client.get(url, headers={"Authorization": f"Bearer {key}"})
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return response.status_code, payload


async def _refresh(settings: object, threshold: float, inference_key: str, management_key: str | None) -> dict[str, Any]:
    global _cached_status, _cached_at, _failure_count, _next_refresh_at, _payment_required_note
    status = _base_status(enabled=True, threshold=threshold)
    checked_at = _now()
    status["checked_at"] = _iso(checked_at)
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            key_status, key_payload = await _get_json(client, OPENROUTER_KEY_URL, inference_key)
            if key_status in {401, 403}:
                status["state"] = "unknown"
                status["warning"] = "OpenRouter key credit information is unavailable."
            elif key_status >= 400:
                status["state"] = "error"
                status["warning"] = "OpenRouter credit check failed."
            else:
                status["key_remaining_usd"] = _remaining_from_key(key_payload)
                if status["key_remaining_usd"] is None:
                    status["state"] = "unknown"
                    status["warning"] = "OpenRouter key credit information is unavailable."
                else:
                    status["state"] = "exhausted" if status["key_remaining_usd"] <= 0 else ("low" if status["key_remaining_usd"] <= threshold else "ok")

            if management_key:
                wallet_status, wallet_payload = await _get_json(client, OPENROUTER_CREDITS_URL, management_key)
                if wallet_status < 400:
                    status["wallet_remaining_usd"] = _remaining_from_wallet(wallet_payload)
                    if status["wallet_remaining_usd"] is None and status["state"] == "ok":
                        status["state"] = "unknown"
                        status["warning"] = "OpenRouter wallet credit information is unavailable."
                    elif status["wallet_remaining_usd"] is not None:
                        wallet_remaining = status["wallet_remaining_usd"]
                        if wallet_remaining <= 0:
                            status["state"] = "exhausted"
                        elif wallet_remaining <= threshold and status["state"] == "ok":
                            status["state"] = "low"
                        # A successful positive wallet result is credible recovery.
                        if wallet_remaining > 0:
                            _payment_required_note = None
                elif wallet_status in {401, 403}:
                    if status["state"] == "ok":
                        status["state"] = "unknown"
                    status["warning"] = "OpenRouter wallet credit information is unavailable."
                else:
                    status["warning"] = "OpenRouter wallet credit check failed."

        _failure_count = 0
        _next_refresh_at = None
        _cached_status = status
        _cached_at = checked_at
        return _with_payment_warning(status)
    except (httpx.HTTPError, OSError, ValueError):
        _failure_count += 1
        delay = min(_CACHE_SECONDS * (2 ** min(_failure_count - 1, 3)), _MAX_BACKOFF_SECONDS)
        _next_refresh_at = checked_at + timedelta(seconds=delay)
        cached = _status_from_cache(stale=True)
        if cached is not None:
            return cached
        status["state"] = "error"
        status["stale"] = True
        status["warning"] = "OpenRouter credit check is unavailable."
        return _with_payment_warning(status)


async def get_openrouter_credit_status(force_refresh: bool = False) -> dict:
    """Return a cached or refreshed, safe OpenRouter credit status."""
    global _refresh_task
    settings = await get_app_settings()
    enabled = bool(_settings_value(settings, "openrouter_credit_monitor_enabled", False))
    threshold = max(0.0, _number(_settings_value(settings, "openrouter_low_credit_threshold_usd", 5.0)) or 0.0)
    if not enabled:
        status = _base_status(enabled=False, threshold=threshold)
        status["state"] = "disabled"
        return _with_payment_warning(status)

    inference_key = str(_settings_value(settings, "openrouter_api_key", "") or "").strip()
    if not inference_key:
        status = _base_status(enabled=True, threshold=threshold)
        status["state"] = "unconfigured"
        status["warning"] = "OpenRouter is not configured."
        return _with_payment_warning(status)
    management_key = str(_settings_value(settings, "openrouter_management_api_key", "") or "").strip() or None

    now = _now()
    cached = _status_from_cache(stale=False)
    if not force_refresh and cached is not None and _cached_at and (now - _cached_at).total_seconds() < _CACHE_SECONDS:
        return cached
    if not force_refresh and _next_refresh_at and now < _next_refresh_at:
        return _status_from_cache(stale=True) or _with_payment_warning(_base_status(enabled=True, threshold=threshold))
    if _refresh_task is None or _refresh_task.done():
        _refresh_task = asyncio.create_task(_refresh(settings, threshold, inference_key, management_key))
    return await asyncio.shield(_refresh_task)


def note_openrouter_payment_required() -> str | None:
    """Record a coarse payment alert without a provider request."""
    global _payment_required_note
    _payment_required_note = "Recent OpenRouter requests require available payment credit."
    return _payment_required_note


def get_openrouter_credit_warning() -> str | None:
    """Return only a coarse cached warning suitable for non-admin task output."""
    if _payment_required_note:
        return _payment_required_note
    status = _status_from_cache(stale=_cached_at is None or (_now() - _cached_at).total_seconds() >= _CACHE_SECONDS)
    return str(status.get("warning")) if status and status.get("warning") else None


async def _monitor_loop() -> None:
    while True:
        try:
            await get_openrouter_credit_status()
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive lifecycle guard
            logger.warning("OpenRouter credit monitor refresh failed")
        await asyncio.sleep(_CACHE_SECONDS)


def start_openrouter_credit_monitor() -> None:
    """Start one background monitor task when an event loop is available."""
    global _monitor_task
    if _monitor_task is None or _monitor_task.done():
        _monitor_task = asyncio.create_task(_monitor_loop())


async def stop_openrouter_credit_monitor() -> None:
    """Cancel and await the monitor task without leaking cancellation."""
    global _monitor_task
    task, _monitor_task = _monitor_task, None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
