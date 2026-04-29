"""Shared OAuth redirect trust and CORS-origin defaults."""

from __future__ import annotations

from urllib.parse import urlparse

TRUSTED_NATIVE_REDIRECT_SCHEMES: frozenset[str] = frozenset(
    {
        "vscode",
        "vscode-insiders",
        "cursor",
        "windsurf",
        "jetbrains",
    }
)

TRUSTED_IDE_REDIRECT_HOSTS: frozenset[str] = frozenset(
    {
        "vscode.dev",
        "insiders.vscode.dev",
        "github.dev",
        "account.jetbrains.com",
    }
)

DEFAULT_TRUSTED_REDIRECT_URIS: tuple[str, ...] = (
    "https://claude.ai/oauth/callback",
    "https://claude.ai/api/mcp/auth_callback",
)

LOOPBACK_REDIRECT_HOSTS: frozenset[str] = frozenset(
    {"127.0.0.1", "localhost", "::1", "[::1]"}
)

LOOPBACK_ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://127.0.0.1",
    "http://localhost",
    "http://[::1]",
)

DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = tuple(
    sorted(
        {
            f"{parsed.scheme.lower()}://{parsed.hostname.lower()}"
            for uri in DEFAULT_TRUSTED_REDIRECT_URIS
            if (parsed := urlparse(uri)).scheme in {"http", "https"}
            and parsed.hostname
        }
    )
)


def build_allowed_origins(configured_origins: str) -> list[str]:
    """Resolve configured CORS origins plus built-in safe defaults."""
    if configured_origins == "*":
        return ["*"]

    origins = [
        origin.strip() for origin in configured_origins.split(",") if origin.strip()
    ]
    resolved = list(DEFAULT_ALLOWED_ORIGINS)

    for origin in origins:
        if origin not in resolved:
            resolved.append(origin)

    for loopback_origin in LOOPBACK_ALLOWED_ORIGINS:
        if loopback_origin not in resolved:
            resolved.append(loopback_origin)

    return resolved