"""
OAuth 2.0 Client Credentials grant for MCP routes.

Implements RFC 6749 §4.4 (client_credentials) so that third-party MCP clients
(e.g. Claude custom connectors, Cowork) can authenticate to Ragtime-hosted MCP
routes with a pre-shared ``client_id`` + ``client_secret`` pair.

Two compatible authentication surfaces are provided:

1. Direct HTTP Basic on every MCP request:
       Authorization: Basic base64(client_id:client_secret)
   Useful for simple clients that don't implement token exchange.

2. OAuth 2.0 token endpoint:
       POST /mcp/oauth/token
       POST /mcp/{route_path}/oauth/token
   Accepts ``grant_type=client_credentials`` and returns a short-lived JWT
   Bearer token scoped to the target route. Subsequent MCP calls send:
       Authorization: Bearer <access_token>

Discovery metadata is exposed via:
    GET /.well-known/oauth-authorization-server
    GET /mcp/{route_path}/.well-known/oauth-authorization-server
    GET /.well-known/oauth-protected-resource (MCP spec; references the above)

Security notes:
- ``client_secret`` is stored Fernet-encrypted in the database (``authPassword``
  column, reused across auth methods).
- Credential comparison uses ``hmac.compare_digest`` to prevent timing oracles.
- Tokens are HS256-signed with the same key as session tokens; the ``aud`` and
  ``scope`` claims bind the token to one MCP route so a token minted for
  ``/mcp/alpha`` cannot be replayed against ``/mcp/beta``.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, urlsplit

from starlette.types import Receive, Scope, Send

from ragtime.config.settings import settings
from ragtime.core.app_settings import get_app_settings
from ragtime.core.auth import decode_jwt_payload, encode_jwt_payload
from ragtime.core.encryption import decrypt_secret
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# In-memory brute-force throttle for MCP auth paths.
#
# Tracks failed credential attempts per source IP. After _THROTTLE_MAX_FAILURES
# failures within _THROTTLE_WINDOW_SECONDS the IP is blocked for
# _THROTTLE_BLOCK_SECONDS. Successful authentication resets the counter.
#
# This is intentionally lightweight (no external dependency) and covers the
# ASGI-level token endpoint and direct Basic-auth paths that cannot use the
# slowapi decorator. The limiter in ragtime.core.rate_limit covers FastAPI
# routes; this covers the raw ASGI handlers.
# ---------------------------------------------------------------------------

_THROTTLE_WINDOW_SECONDS = 60
_THROTTLE_MAX_FAILURES = 10
_THROTTLE_BLOCK_SECONDS = 300

# {ip: [timestamp_of_failure, ...]}
_auth_failure_timestamps: dict[str, list[float]] = defaultdict(list)
# {ip: block_until_timestamp}
_auth_blocked_until: dict[str, float] = {}


def _get_client_ip(scope: Scope) -> str:
    """Extract the client IP from an ASGI scope for brute-force throttling."""
    client = scope.get("client")
    if client:
        return str(client[0])

    headers = dict(scope.get("headers", []))
    forwarded_for = headers.get(b"x-forwarded-for", b"").decode(errors="ignore").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return "unknown"


def _is_throttled(ip: str) -> bool:
    """Return True if the IP is currently blocked due to too many failures."""
    now = time.monotonic()
    blocked_until = _auth_blocked_until.get(ip, 0.0)
    if blocked_until > now:
        return True
    # Prune old failure timestamps outside the window.
    window_start = now - _THROTTLE_WINDOW_SECONDS
    timestamps = _auth_failure_timestamps.get(ip)
    if timestamps is not None:
        pruned = [ts for ts in timestamps if ts >= window_start]
        if pruned != timestamps:
            _auth_failure_timestamps[ip] = pruned
    return False


def _record_auth_failure(ip: str) -> None:
    """Record a failed auth attempt; block the IP if the threshold is exceeded."""
    now = time.monotonic()
    window_start = now - _THROTTLE_WINDOW_SECONDS
    timestamps = _auth_failure_timestamps[ip]
    # Prune stale entries.
    _auth_failure_timestamps[ip] = [ts for ts in timestamps if ts >= window_start]
    _auth_failure_timestamps[ip].append(now)
    if len(_auth_failure_timestamps[ip]) >= _THROTTLE_MAX_FAILURES:
        _auth_blocked_until[ip] = now + _THROTTLE_BLOCK_SECONDS
        logger.warning(
            "MCP auth: IP %s blocked for %ds after %d failures in %ds window",
            ip,
            _THROTTLE_BLOCK_SECONDS,
            _THROTTLE_MAX_FAILURES,
            _THROTTLE_WINDOW_SECONDS,
        )


def _record_auth_success(ip: str) -> None:
    """Clear failure counters for an IP on successful authentication."""
    _auth_failure_timestamps.pop(ip, None)
    _auth_blocked_until.pop(ip, None)


# Token lifetime for MCP client-credentials access tokens. Kept intentionally
# short; clients are expected to refresh by re-issuing a token request.
_CLIENT_CREDENTIALS_TOKEN_TTL = timedelta(hours=1)

# ``aud`` prefix used to bind issued tokens to a specific route. "__default__"
# represents the default /mcp route; custom routes use their route_path.
_AUD_PREFIX = "mcp:"
_DEFAULT_ROUTE_AUD = f"{_AUD_PREFIX}__default__"


def _route_audience(route_path: str | None) -> str:
    """Return the audience claim used for a route."""
    if not route_path:
        return _DEFAULT_ROUTE_AUD
    return f"{_AUD_PREFIX}{route_path}"


def _decode_basic_auth(auth_header: str) -> tuple[str, str] | None:
    """Decode an ``Authorization: Basic`` header into (user, password)."""
    if not auth_header.lower().startswith("basic "):
        return None
    try:
        raw = base64.b64decode(auth_header[6:].strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if ":" not in raw:
        return None
    client_id, client_secret = raw.split(":", 1)
    return client_id, client_secret


async def _read_form_body(receive: Receive) -> dict[str, str]:
    """Collect and parse a small application/x-www-form-urlencoded body."""
    chunks: list[bytes] = []
    # Cap body size defensively (RFC 6749 token requests are tiny).
    total = 0
    limit = 16 * 1024
    while True:
        message = await receive()
        if message["type"] != "http.request":
            break
        body = message.get("body") or b""
        total += len(body)
        if total > limit:
            return {}
        chunks.append(body)
        if not message.get("more_body"):
            break
    try:
        raw = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError:
        return {}
    parsed = parse_qs(raw, keep_blank_values=True)
    # parse_qs yields list-of-values; collapse to the first value per key.
    return {k: v[0] for k, v in parsed.items() if v}


def _extract_client_credentials(scope: Scope, form: dict[str, str] | None = None) -> tuple[str, str] | None:
    """
    Extract ``(client_id, client_secret)`` from the request.

    Supports (in order):
    - ``Authorization: Basic base64(id:secret)`` header (RFC 6749 §2.3.1 preferred).
    - ``client_id`` / ``client_secret`` form fields (RFC 6749 §2.3.1 alternative).
    """
    headers = dict(scope.get("headers", []))
    auth_header = headers.get(b"authorization", b"").decode(errors="ignore")
    basic = _decode_basic_auth(auth_header)
    if basic:
        return basic

    if form is not None:
        cid = form.get("client_id")
        secret = form.get("client_secret")
        if cid and secret:
            return cid, secret
    return None


def _credentials_match(
    provided: tuple[str, str],
    expected_client_id: str | None,
    encrypted_secret: str | None,
) -> bool:
    """Constant-time compare provided vs. configured client credentials."""
    if not expected_client_id or not encrypted_secret:
        return False
    stored_secret = decrypt_secret(encrypted_secret)
    if not stored_secret:
        logger.warning("MCP client_credentials secret decryption failed - encryption key may have changed")
        return False
    # Evaluate both halves to avoid short-circuiting timing differences.
    cid_ok = hmac.compare_digest(provided[0], expected_client_id)
    secret_ok = hmac.compare_digest(provided[1], stored_secret)
    return cid_ok and secret_ok


async def validate_client_credentials_basic(
    scope: Scope,
    expected_client_id: str | None,
    encrypted_secret: str | None,
) -> bool:
    """
    Validate Basic-auth-style client credentials directly on an MCP request.

    On success, annotates the ASGI scope with ``_mcp_client_id`` for logging.
    Failed attempts are counted against the source IP for brute-force
    throttling; callers should check ``_is_throttled`` before invoking this
    function when they want to short-circuit early.
    """
    client_ip = _get_client_ip(scope)
    if _is_throttled(client_ip):
        logger.warning("MCP Basic auth: request from %s blocked by brute-force throttle", client_ip)
        return False
    provided = _extract_client_credentials(scope)
    if not provided:
        return False
    if not _credentials_match(provided, expected_client_id, encrypted_secret):
        _record_auth_failure(client_ip)
        return False
    _record_auth_success(client_ip)
    scope["_mcp_client_id"] = provided[0]
    scope["_mcp_username"] = f"client:{provided[0]}"
    return True


def issue_client_credentials_token(client_id: str, route_path: str | None) -> dict[str, Any]:
    """
    Mint a short-lived JWT bearer token for a client_credentials grant.

    The token is bound to one MCP route via the ``aud`` claim.
    """
    now = datetime.now(timezone.utc)
    expires = now + _CLIENT_CREDENTIALS_TOKEN_TTL
    aud = _route_audience(route_path)
    payload = {
        "sub": f"client:{client_id}",
        "client_id": client_id,
        "aud": aud,
        "scope": aud,
        "grant": "client_credentials",
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }
    token = encode_jwt_payload(payload)
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": int(_CLIENT_CREDENTIALS_TOKEN_TTL.total_seconds()),
        "scope": aud,
    }


def validate_client_credentials_bearer(scope: Scope, route_path: str | None) -> bool:
    """
    Validate a Bearer token previously issued by the token endpoint for
    this route.

    Accepts the token only if:
    - it is signed with the server key,
    - it was issued as a ``client_credentials`` grant,
    - its ``aud`` matches the requested route.
    """
    headers = dict(scope.get("headers", []))
    auth_header = headers.get(b"authorization", b"").decode(errors="ignore")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header[7:]
    expected_aud = _route_audience(route_path)
    payload = decode_jwt_payload(token, audience=expected_aud)
    if payload is None:
        logger.debug("MCP client_credentials bearer rejected")
        return False
    if payload.get("grant") != "client_credentials":
        return False
    client_id = payload.get("client_id")
    if client_id:
        scope["_mcp_client_id"] = client_id
        scope["_mcp_username"] = f"client:{client_id}"
    return True


# ---------------------------------------------------------------------------
# Token endpoint + discovery metadata helpers
# ---------------------------------------------------------------------------


async def _send_json(
    send: Send,
    status: int,
    payload: dict[str, Any],
    headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    base_headers = [
        (b"content-type", b"application/json"),
        (b"cache-control", b"no-store"),
        (b"pragma", b"no-cache"),
    ]
    if headers:
        base_headers.extend(headers)
    await send({"type": "http.response.start", "status": status, "headers": base_headers})
    await send({"type": "http.response.body", "body": body})


async def _token_error(send: Send, status: int, error: str, description: str) -> None:
    await _send_json(
        send,
        status,
        {"error": error, "error_description": description},
        headers=[(b"www-authenticate", b'Basic realm="mcp"')],
    )


async def _get_route_client_credentials(
    route_path: str | None,
) -> tuple[str | None, str | None] | None:
    """
    Resolve (expected_client_id, encrypted_secret) for a route.

    Returns None if the target route is unknown or does not have
    ``client_credentials`` auth configured.
    """
    if route_path:
        # Custom route lookup via DB (avoids an import cycle with mcp.server).
        from ragtime.core.database import get_db

        db = await get_db()
        route = await db.mcprouteconfig.find_unique(where={"routePath": route_path})
        if not route or not route.enabled or not route.requireAuth:
            return None
        if (route.authMethod or "password") != "client_credentials":
            return None
        return route.authClientId, route.authPassword

    app_settings = await get_app_settings()
    if not app_settings.get("mcp_default_route_auth"):
        return None
    if app_settings.get("mcp_default_route_auth_method") != "client_credentials":
        return None
    return (
        app_settings.get("mcp_default_route_client_id"),
        app_settings.get("mcp_default_route_password"),
    )


async def handle_token_request(scope: Scope, receive: Receive, send: Send, route_path: str | None) -> None:
    """
    RFC 6749 §4.4 token endpoint for MCP routes.

    Expects ``POST`` with ``application/x-www-form-urlencoded`` body containing
    at minimum ``grant_type=client_credentials``. Client credentials may be
    provided either via HTTP Basic auth or ``client_id``/``client_secret``
    form fields. Returns a short-lived Bearer token bound to the route.

    Brute-force protection: failed credential attempts are tracked per source
    IP. After ``_THROTTLE_MAX_FAILURES`` failures within the sliding window the
    IP is blocked for ``_THROTTLE_BLOCK_SECONDS``.
    """
    if scope.get("method", "").upper() != "POST":
        await _token_error(send, 405, "invalid_request", "Token endpoint requires POST")
        return

    form = await _read_form_body(receive)
    grant_type = form.get("grant_type", "")
    if grant_type != "client_credentials":
        await _token_error(
            send,
            400,
            "unsupported_grant_type",
            "Only grant_type=client_credentials is supported on this endpoint",
        )
        return

    client_ip = _get_client_ip(scope)
    if _is_throttled(client_ip):
        logger.warning("MCP token endpoint: request from %s blocked by brute-force throttle", client_ip)
        await _token_error(send, 429, "too_many_requests", "Too many failed authentication attempts. Try again later.")
        return

    creds = _extract_client_credentials(scope, form=form)
    if not creds:
        _record_auth_failure(client_ip)
        await _token_error(
            send,
            401,
            "invalid_client",
            "Provide client_id and client_secret via HTTP Basic auth or form body",
        )
        return

    configured = await _get_route_client_credentials(route_path)
    if configured is None:
        # Don't leak whether the route exists or is misconfigured.
        _record_auth_failure(client_ip)
        await _token_error(
            send,
            401,
            "invalid_client",
            "Client credentials authentication is not enabled for this route",
        )
        return

    expected_client_id, encrypted_secret = configured
    if not _credentials_match(creds, expected_client_id, encrypted_secret):
        _record_auth_failure(client_ip)
        await _token_error(send, 401, "invalid_client", "Invalid client credentials")
        return

    _record_auth_success(client_ip)
    payload = issue_client_credentials_token(creds[0], route_path)
    logger.info(
        "Issued MCP client_credentials token for client_id=%s route=%s",
        creds[0],
        route_path or "__default__",
    )
    await _send_json(send, 200, payload)


def _resource_base(scope: Scope) -> str:
    """Reconstruct the scheme://host base URL for metadata documents.

    When ``EXTERNAL_BASE_URL`` is configured it is used as the authoritative
    base so that OAuth metadata always points at the canonical public URL
    regardless of what ``Host`` or ``X-Forwarded-Host`` headers say.  This
    prevents a hostile reverse-proxy or request header from steering MCP
    clients at an attacker-controlled authorization server.

    When no canonical URL is configured the function falls back to the
    direct request Host. ``X-Forwarded-Host`` is only accepted when it names
    the same host family as Host, so hostile forwarded headers cannot poison
    issuer/token metadata.
    """
    configured = str(getattr(settings, "external_base_url", "") or "").strip().rstrip("/")
    if configured:
        return configured

    scheme = scope.get("scheme", "http")
    headers = dict(scope.get("headers", []))
    # Respect reverse-proxy forwarded scheme if present.
    forwarded_proto = headers.get(b"x-forwarded-proto", b"").decode(errors="ignore")
    if forwarded_proto:
        scheme = forwarded_proto.split(",")[0].strip() or scheme
    if scheme not in {"http", "https"}:
        scheme = "http"

    direct_host = headers.get(b"host", b"").decode(errors="ignore").split(",")[0].strip()
    forwarded_host = headers.get(b"x-forwarded-host", b"").decode(errors="ignore").split(",")[0].strip()
    host = direct_host or "localhost"

    if forwarded_host and direct_host:
        direct_hostname = urlsplit(f"{scheme}://{direct_host}").hostname
        forwarded_hostname = urlsplit(f"{scheme}://{forwarded_host}").hostname
        if direct_hostname and forwarded_hostname and forwarded_hostname.lower() == direct_hostname.lower():
            host = forwarded_host

    parsed_host = urlsplit(f"{scheme}://{host}")
    if not parsed_host.hostname or any(char in host for char in "\r\n"):
        host = "localhost"
    return f"{scheme}://{host}"


def build_authorization_server_metadata(base: str, route_path: str | None) -> dict[str, Any]:
    """Build RFC 8414 authorization-server metadata for an MCP route."""
    if route_path:
        issuer = f"{base}/mcp/{route_path}"
        token_endpoint = f"{base}/mcp/{route_path}/oauth/token"
    else:
        issuer = f"{base}/mcp"
        token_endpoint = f"{base}/mcp/oauth/token"
    return {
        "issuer": issuer,
        "token_endpoint": token_endpoint,
        "grant_types_supported": ["client_credentials"],
        "token_endpoint_auth_methods_supported": [
            "client_secret_basic",
            "client_secret_post",
        ],
        "response_types_supported": [],
        "scopes_supported": [_route_audience(route_path)],
    }


def build_interactive_authorization_server_metadata(base: str) -> dict[str, Any]:
    """Build auth-code OAuth metadata for routes that use user authentication."""
    return {
        "issuer": base,
        "authorization_endpoint": f"{base}/authorize",
        "token_endpoint": f"{base}/token",
        "registration_endpoint": f"{base}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "password"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "client_id_metadata_document_supported": True,
        "scopes_supported": [],
        "response_modes_supported": ["query"],
        "service_documentation": f"{base}/docs",
    }


def build_interactive_protected_resource_metadata(base: str, route_path: str | None) -> dict[str, Any]:
    """Build protected-resource metadata for routes that use app-level OAuth."""
    if route_path:
        resource = f"{base}/mcp/{route_path}"
    else:
        resource = f"{base}/mcp"
    return {
        "resource": resource,
        "authorization_servers": [base],
        "scopes_supported": [],
        "bearer_methods_supported": ["header"],
    }


def build_protected_resource_metadata(base: str, route_path: str | None) -> dict[str, Any]:
    """Build RFC 9728 protected-resource metadata for an MCP route."""
    if route_path:
        resource = f"{base}/mcp/{route_path}"
        authorization_server = f"{base}/mcp/{route_path}"
    else:
        resource = f"{base}/mcp"
        authorization_server = f"{base}/mcp"
    return {
        "resource": resource,
        "authorization_servers": [authorization_server],
        "bearer_methods_supported": ["header"],
    }


async def handle_authorization_server_metadata(scope: Scope, _receive: Receive, send: Send, route_path: str | None) -> None:
    """
    RFC 8414 OAuth 2.0 Authorization Server Metadata document.

    Advertises the token endpoint + supported grants so MCP clients that
    discover auth configuration automatically (e.g. Claude Web) can locate
    the token endpoint without manual setup.
    """
    base = _resource_base(scope)
    if await _get_route_client_credentials(route_path) is None:
        payload = build_interactive_authorization_server_metadata(base)
    else:
        payload = build_authorization_server_metadata(base, route_path)
    await _send_json(send, 200, payload)


async def handle_protected_resource_metadata(scope: Scope, _receive: Receive, send: Send, route_path: str | None) -> None:
    """
    RFC 9728 Protected Resource Metadata, aligned with the MCP
    authorization specification. Points clients at the authorization server
    metadata document above.
    """
    base = _resource_base(scope)
    if await _get_route_client_credentials(route_path) is None:
        payload = build_interactive_protected_resource_metadata(base, route_path)
    else:
        payload = build_protected_resource_metadata(base, route_path)
    await _send_json(send, 200, payload)
