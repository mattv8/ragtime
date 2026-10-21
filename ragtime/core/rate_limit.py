"""
Rate limiting for authentication endpoints.

Uses slowapi to prevent brute-force attacks on login.
"""

import time
from dataclasses import dataclass
from urllib.parse import parse_qsl

from limits import parse
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ragtime.config.settings import settings

# Configure rate limiter using client IP address
# In production behind a reverse proxy, ensure X-Forwarded-For is set
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],  # No default limits - apply explicitly per endpoint
    enabled=not settings.debug_mode,  # Disable in debug mode for easier testing
)

# Rate limit constants
LOGIN_RATE_LIMIT = "5/minute"  # 5 login attempts per minute per IP
OAUTH_TOKEN_PASSWORD_RATE_LIMIT = LOGIN_RATE_LIMIT
OAUTH_TOKEN_EXCHANGE_RATE_LIMIT = "60/minute"
SHARE_AUTH_RATE_LIMIT = LOGIN_RATE_LIMIT
# v1 runtime-bridge calls are keyed per IP, which is acceptable because they
# originate from the runtime host rather than arbitrary browser clients.
RUNTIME_BRIDGE_RATE_LIMIT = "120/minute"
MAX_OAUTH_TOKEN_REQUEST_BODY_BYTES = 64 * 1024


@dataclass(frozen=True)
class OAuthTokenRateLimitBucket:
    scope: str
    limit: str


OAUTH_TOKEN_PASSWORD_BUCKET = OAuthTokenRateLimitBucket("oauth_token_password", OAUTH_TOKEN_PASSWORD_RATE_LIMIT)
OAUTH_TOKEN_EXCHANGE_BUCKET = OAuthTokenRateLimitBucket("oauth_token_exchange", OAUTH_TOKEN_EXCHANGE_RATE_LIMIT)


class _OAuthTokenRequestDisconnected(Exception):
    """Stop processing when the client disconnects before sending a full body."""


def oauth_token_rate_limit_bucket(content_type: str, body: bytes) -> OAuthTokenRateLimitBucket:
    """Classify only an unambiguous URL-encoded non-password grant as an exchange."""
    if content_type.split(";", 1)[0].strip().lower() != "application/x-www-form-urlencoded":
        return OAUTH_TOKEN_PASSWORD_BUCKET
    try:
        fields = parse_qsl(body.decode("utf-8"), keep_blank_values=True, errors="strict")
    except UnicodeDecodeError:
        return OAUTH_TOKEN_PASSWORD_BUCKET
    grant_types = [value for name, value in fields if name == "grant_type"]
    if len(grant_types) != 1 or grant_types[0] == "password":
        return OAUTH_TOKEN_PASSWORD_BUCKET
    return OAUTH_TOKEN_EXCHANGE_BUCKET


async def _read_oauth_token_request_body(receive: Receive, content_length: str | None) -> tuple[bytes, Receive]:
    try:
        declared_length = int(content_length) if content_length is not None else None
    except ValueError:
        declared_length = None
    if declared_length is not None and declared_length > MAX_OAUTH_TOKEN_REQUEST_BODY_BYTES:
        raise ValueError("OAuth token request body is too large")

    chunks: list[bytes] = []
    total_size = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise _OAuthTokenRequestDisconnected
        chunk = message.get("body", b"")
        total_size += len(chunk)
        if total_size > MAX_OAUTH_TOKEN_REQUEST_BODY_BYTES:
            raise ValueError("OAuth token request body is too large")
        chunks.append(chunk)
        if not message.get("more_body", False):
            break

    body = b"".join(chunks)
    sent = False

    async def replay_receive() -> Message:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return body, replay_receive


def consume_oauth_token_rate_limit(scope: Scope, bucket: OAuthTokenRateLimitBucket) -> int | None:
    """Consume the shared OAuth token quota and return a retry delay when exhausted."""
    if not limiter.enabled:
        return None

    client = scope.get("client")
    remote_address = client[0] if client else "127.0.0.1"
    keys = [remote_address, bucket.scope]
    if limiter._key_prefix:
        keys.insert(0, limiter._key_prefix)
    limit = parse(bucket.limit)
    # SlowAPI exposes this configured backend and prefix only as internals; use
    # them here so both buckets share its production storage rather than adding
    # another in-memory request tracker.
    if limiter.limiter.hit(limit, *keys):
        return None

    reset_at, _remaining = limiter.limiter.get_window_stats(limit, *keys)
    return max(1, int(reset_at - time.time()) + 1)


class OAuthTokenRateLimitMiddleware:
    """Bound and rate-limit OAuth token request bodies while replaying valid input."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in {"/token", "/auth/oauth2/token"}:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        try:
            body, replay_receive = await _read_oauth_token_request_body(receive, headers.get("content-length"))
        except _OAuthTokenRequestDisconnected:
            await JSONResponse(
                status_code=400,
                content={"error": "invalid_request", "error_description": "Incomplete token request body"},
                headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
            )(scope, receive, send)
            return
        except ValueError:
            await JSONResponse(
                status_code=413,
                content={"error": "invalid_request", "error_description": "Token request body is too large"},
                headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
            )(scope, receive, send)
            return

        bucket = oauth_token_rate_limit_bucket(headers.get("content-type", ""), body)
        retry_after = consume_oauth_token_rate_limit(scope, bucket)
        if retry_after is not None:
            await JSONResponse(
                status_code=429,
                content={"error": "too_many_requests", "error_description": "Too many token requests"},
                headers={"Cache-Control": "no-store", "Pragma": "no-cache", "Retry-After": str(retry_after)},
            )(scope, replay_receive, send)
            return
        await self.app(scope, replay_receive, send)
