"""Pure authentication lifetime policy helpers."""

from typing import Any

WEB_SESSION_HOURS_MIN = 1
WEB_SESSION_HOURS_MAX = 720
MCP_ACCESS_TOKEN_MINUTES_MIN = 5
MCP_ACCESS_TOKEN_MINUTES_MAX = 1440
MCP_ACCESS_TOKEN_MINUTES_DEFAULT = 60
MCP_AUTHORIZATION_DAYS_MIN = 1
MCP_AUTHORIZATION_DAYS_MAX = 90
MCP_AUTHORIZATION_DAYS_DEFAULT = 30


def resolve_web_session_hours(config: Any, fallback_hours: int) -> int:
    """Return a valid persisted web-session override or the legacy fallback."""
    value = getattr(config, "web_session_hours", getattr(config, "webSessionHours", None))
    if value is None:
        return int(fallback_hours)
    try:
        hours = int(value)
    except (TypeError, ValueError):
        return int(fallback_hours)
    if WEB_SESSION_HOURS_MIN <= hours <= WEB_SESSION_HOURS_MAX:
        return hours
    return int(fallback_hours)
