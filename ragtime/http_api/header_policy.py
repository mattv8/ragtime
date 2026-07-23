from __future__ import annotations

HEADER_NAME_ALLOWED_CHARS = frozenset("!#$%&'*+-.^_`|~")

HTTP_API_BLOCKED_HEADER_NAMES = {
    "authorization",
    "connection",
    "content-length",
    "cookie",
    "forwarded",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "x-forwarded-for",
}

HTTP_API_BLOCKED_HEADER_PREFIXES = ("x-forwarded-",)
HTTP_API_CONFIGURED_ALLOWED_HEADER_NAMES = {"authorization", "authentication"}


def _require_header_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def is_blocked_header_name(name: str) -> bool:
    lowered = name.strip().lower()
    return lowered in HTTP_API_BLOCKED_HEADER_NAMES or any(lowered.startswith(prefix) for prefix in HTTP_API_BLOCKED_HEADER_PREFIXES)


def validate_header_name(name: object, *, allow_configured_auth_headers: bool) -> str:
    candidate = _require_header_string(name, field_name="Header name").strip()
    if not candidate:
        raise ValueError("Header name is required")
    if any(ch in {"\r", "\n", ":"} for ch in candidate):
        raise ValueError(f"Header '{candidate}' has an invalid name")
    if not all(ch.isascii() and (ch.isalnum() or ch in HEADER_NAME_ALLOWED_CHARS) for ch in candidate):
        raise ValueError(f"Header '{candidate}' has an invalid name")
    lowered = candidate.lower()
    if is_blocked_header_name(candidate) and (not allow_configured_auth_headers or lowered not in HTTP_API_CONFIGURED_ALLOWED_HEADER_NAMES):
        raise ValueError(f"Header '{candidate}' is not allowed")
    return candidate


def validate_header_value(value: object, *, allow_blank: bool) -> str:
    candidate = _require_header_string(value, field_name="Header value")
    if "\r" in candidate or "\n" in candidate:
        raise ValueError("Header values must not contain CR or LF characters")
    if not allow_blank and candidate.strip() == "":
        raise ValueError("Header value is required")
    return candidate
