"""Shared HTTP timeout helpers."""

from typing import Any

from ragtime.core.app_settings import get_app_settings

# Keep synchronous HTTP request handlers below common reverse-proxy/CDN read
# timeouts. Cloudflare's default 524 cliff is about 100 seconds, so handlers
# that must return a structured in-app error should cap themselves below it.
DEFAULT_HTTP_PROXY_SAFE_TIMEOUT_SECONDS = 90.0


def normalize_http_proxy_safe_timeout_seconds(value: Any) -> float:
    try:
        timeout_seconds = float(value)
    except (TypeError, ValueError):
        return DEFAULT_HTTP_PROXY_SAFE_TIMEOUT_SECONDS
    if timeout_seconds <= 0:
        return DEFAULT_HTTP_PROXY_SAFE_TIMEOUT_SECONDS
    return max(1.0, min(timeout_seconds, 3600.0))


async def get_http_proxy_safe_timeout_seconds() -> float:
    app_settings = await get_app_settings()
    return normalize_http_proxy_safe_timeout_seconds(app_settings.get("http_proxy_safe_timeout_seconds"))
