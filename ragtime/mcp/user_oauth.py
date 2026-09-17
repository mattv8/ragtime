"""Grant-backed OAuth tokens for interactive MCP clients.

This module deliberately does not create web sessions.  Interactive MCP access
tokens are short lived JWTs backed by a durable OAuth grant and rotating opaque
refresh tokens.
"""

from __future__ import annotations

import base64
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from jose import JWTError, jwt
from prisma.models import User

from ragtime.config.settings import settings
from ragtime.core.app_settings import get_app_settings
from ragtime.core.auth import TokenData, encode_jwt_payload, get_auth_provider_config, hash_token
from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.core.mfa import mfa_needed_for_user
from ragtime.core.oauth_grants import (
    OAuthGrantError,
    create_grant,
    get_active_grant,
    lookup_refresh_grant,
    revoke_grant,
    revoke_refresh_token,
    rotate_refresh_token,
)

_MCP_SERVICE_AUDIENCE = "mcp-service"
_MCP_TOKEN_USE = "mcp_access"
logger = get_logger(__name__)


class McpOAuthError(Exception):
    """Safe protocol error translated by the OAuth HTTP handlers."""

    def __init__(self, error: str, description: str, status_code: int = 400) -> None:
        self.error = error
        self.description = description
        self.status_code = status_code
        super().__init__(description)


def is_mcp_access_token_candidate(token: str) -> bool:
    """Untrusted shape check used only to prevent unsafe legacy fallback."""
    try:
        payload_part = token.split(".")[1]
        payload_part += "=" * (-len(payload_part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_part))
        if not isinstance(payload, dict):
            return False
        return payload.get("token_use") == _MCP_TOKEN_USE or bool(payload.get("grant_id")) or payload.get("aud") == _MCP_SERVICE_AUDIENCE
    except (AttributeError, IndexError, TypeError, ValueError, UnicodeDecodeError):
        return False


def _canonical_origin(raw: str) -> str:
    if not raw or any(ord(character) < 32 or ord(character) == 127 for character in raw) or "\\" in raw:
        raise McpOAuthError("invalid_target", "Invalid MCP resource")
    try:
        parsed = urlsplit(raw)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
    except (TypeError, ValueError):
        raise McpOAuthError("invalid_target", "Invalid MCP resource") from None
    if scheme not in {"http", "https"} or not hostname or parsed.username or parsed.password:
        raise McpOAuthError("invalid_target", "Invalid MCP resource")
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host if port in {None, 80 if scheme == "http" else 443} else f"{host}:{port}"
    return urlunsplit((scheme, netloc, "", "", ""))


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


async def normalize_mcp_resource(resource: str | None, *, base_url: str) -> str:
    """Return the canonical interactive MCP audience for a requested resource."""
    if resource is None or not resource.strip():
        return _MCP_SERVICE_AUDIENCE
    if any(ord(character) < 32 or ord(character) == 127 for character in resource) or "\\" in resource or "%" in resource or ".." in unquote(resource):
        raise McpOAuthError("invalid_target", "Invalid MCP resource")
    try:
        parsed = urlsplit(resource)
    except (TypeError, ValueError):
        raise McpOAuthError("invalid_target", "Invalid MCP resource") from None
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise McpOAuthError("invalid_target", "Invalid MCP resource")
    origin = _canonical_origin(base_url)
    candidate_origin = _canonical_origin(resource)
    if candidate_origin != origin:
        raise McpOAuthError("invalid_target", "MCP resource is not served by this authorization server")
    path = parsed.path.rstrip("/") or "/"
    if path == "/mcp":
        app_settings = await get_app_settings()
        if (
            app_settings.get("mcp_enabled", False)
            and app_settings.get("mcp_default_route_auth", False)
            and _enum_value(app_settings.get("mcp_default_route_auth_method", "oauth2")) == "oauth2"
        ):
            return f"{origin}/mcp"
        raise McpOAuthError("invalid_target", "MCP resource is not eligible for interactive authorization")
    if not path.startswith("/mcp/") or path.count("/") != 2:
        raise McpOAuthError("invalid_target", "Invalid MCP resource")
    route_path = path[len("/mcp/") :]
    app_settings = await get_app_settings()
    if not app_settings.get("mcp_enabled", False):
        raise McpOAuthError("invalid_target", "MCP is disabled")
    db = await get_db()
    route = await db.mcprouteconfig.find_unique(where={"routePath": route_path})
    if not route or not route.enabled or not route.requireAuth or _enum_value(route.authMethod) != "oauth2":
        raise McpOAuthError("invalid_target", "MCP resource is not eligible for interactive authorization")
    return f"{origin}{path}"


async def _current_user(user_id: str) -> User:
    db = await get_db()
    user = await db.user.find_unique(where={"id": user_id})
    if not user:
        raise McpOAuthError("invalid_grant", "The authorization is no longer valid")
    return user


async def _require_current_mfa(user: User, mfa_verified: bool) -> None:
    if await mfa_needed_for_user(user) and not mfa_verified:
        raise McpOAuthError("invalid_grant", "Multi-factor authentication is required")


async def _token_lifetime_minutes() -> int:
    config = await get_auth_provider_config()
    return int(getattr(config, "mcp_access_token_minutes", 60))


async def _grant_lifetime_days() -> int:
    config = await get_auth_provider_config()
    return int(getattr(config, "mcp_authorization_days", 30))


def _mint_access_token(*, grant: Any, user: User, issuer: str, expires_at: datetime) -> str:
    now = datetime.now(timezone.utc)
    return encode_jwt_payload(
        {
            "token_use": _MCP_TOKEN_USE,
            "grant_id": grant.id,
            "sub": grant.user_id,
            "client_id": grant.client_id,
            "aud": grant.audience,
            "iss": issuer,
            "scope": grant.scope,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "security_generation": grant.security_generation,
        }
    )


def _build_pair(*, grant: Any, user: User, issuer: str, refresh_token: str, access_minutes: int) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    expires_at = min(now + timedelta(minutes=access_minutes), grant.expires_at)
    if expires_at <= now:
        raise McpOAuthError("invalid_grant", "The authorization has expired")
    access_token = _mint_access_token(grant=grant, user=user, issuer=issuer, expires_at=expires_at)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": int((expires_at - now).total_seconds()),
        "scope": grant.scope,
    }


async def issue_mcp_token_pair(
    *, user_id: str, client_id: str, audience: str, scope: str, security_generation: int, mfa_verified: bool, auth_methods: list[str], issuer: str
) -> dict[str, Any]:
    access_minutes = await _token_lifetime_minutes()
    grant_days = await _grant_lifetime_days()
    user = await _current_user(user_id)
    if int(getattr(user, "securityGeneration", 0)) != security_generation:
        raise McpOAuthError("invalid_grant", "The authorization is no longer valid")
    await _require_current_mfa(user, mfa_verified)
    now = datetime.now(timezone.utc)
    refresh_token = secrets.token_urlsafe(48)
    try:
        grant = await create_grant(
            user_id=user_id,
            client_id=client_id,
            audience=audience,
            scope=scope or "",
            expires_at=now + timedelta(days=grant_days),
            security_generation=security_generation,
            mfa_verified_at=now if mfa_verified else None,
            auth_methods=auth_methods,
            refresh_hash=hash_token(refresh_token),
            now=now,
        )
    except OAuthGrantError as exc:
        logger.info("MCP OAuth grant issuance rejected: %s", exc.reason)
        raise McpOAuthError("invalid_grant", "The authorization is no longer valid") from exc
    return _build_pair(grant=grant, user=user, issuer=issuer, refresh_token=refresh_token, access_minutes=access_minutes)


async def refresh_mcp_token_pair(*, refresh_token: str, client_id: str, resource: str | None, scope: str | None, issuer: str) -> dict[str, Any]:
    # Resolve policy and fail an obviously expired authorization before the
    # single-use refresh is consumed.  The store repeats the expiry check while
    # holding its lock, closing the race between this preflight and rotation.
    access_minutes = await _token_lifetime_minutes()
    refresh_hash = hash_token(refresh_token)
    grant = await lookup_refresh_grant(refresh_hash)
    if not grant or grant.client_id != client_id:
        logger.info("MCP OAuth refresh rejected: unknown_refresh_token")
        raise McpOAuthError("invalid_grant", "Invalid refresh token")
    if grant.expires_at <= datetime.now(timezone.utc):
        logger.info("MCP OAuth refresh rejected: expired_grant")
        raise McpOAuthError("invalid_grant", "The authorization has expired")
    canonical_resource = None if resource is None else await normalize_mcp_resource(resource, base_url=issuer)
    if canonical_resource is not None and canonical_resource != grant.audience:
        raise McpOAuthError("invalid_grant", "Refresh resource does not match the authorization")
    if scope is not None and scope != grant.scope:
        raise McpOAuthError("invalid_grant", "Refresh scope does not match the authorization")
    user = await _current_user(grant.user_id)
    if int(getattr(user, "securityGeneration", 0)) != grant.security_generation:
        raise McpOAuthError("invalid_grant", "The authorization is no longer valid")
    await _require_current_mfa(user, grant.mfa_verified_at is not None)
    next_refresh = secrets.token_urlsafe(48)
    try:
        rotated = await rotate_refresh_token(
            refresh_hash=refresh_hash,
            next_refresh_hash=hash_token(next_refresh),
            client_id=client_id,
            audience=canonical_resource,
            scope=scope,
        )
    except OAuthGrantError as exc:
        logger.info("MCP OAuth refresh rejected: %s", exc.reason)
        raise McpOAuthError("invalid_grant", "Invalid refresh token") from exc
    return _build_pair(grant=rotated, user=user, issuer=issuer, refresh_token=next_refresh, access_minutes=access_minutes)


def _decode_mcp_access(token: str, *, issuer: str) -> dict[str, Any] | None:
    try:
        payload = jwt.decode(token, settings.encryption_key, algorithms=[settings.jwt_algorithm], issuer=issuer, options={"verify_aud": False})
    except JWTError:
        return None
    required = {"token_use", "grant_id", "sub", "client_id", "aud", "iss", "scope", "iat", "exp", "security_generation"}
    if not required.issubset(payload) or payload.get("token_use") != _MCP_TOKEN_USE:
        return None
    if any(not isinstance(payload[key], str) for key in ("grant_id", "sub", "client_id", "aud", "iss", "scope")):
        return None
    if any(isinstance(payload[key], bool) or not isinstance(payload[key], int) for key in ("iat", "exp", "security_generation")):
        return None
    return dict(payload)


async def revoke_mcp_token(*, token: str, client_id: str | None, issuer: str) -> None:
    payload = _decode_mcp_access(token, issuer=issuer)
    if payload:
        if client_id is not None and client_id != payload["client_id"]:
            return
        await revoke_grant(str(payload["grant_id"]), client_id=client_id)
        return
    # Opaque refresh token.  The store makes unknown values a no-op.
    await revoke_refresh_token(hash_token(token), client_id=client_id)


async def validate_mcp_token_and_fetch_user(token: str, *, resource: str, issuer: str) -> tuple[TokenData | None, User | None]:
    payload = _decode_mcp_access(token, issuer=issuer)
    if not payload:
        return None, None
    audience = payload.get("aud")
    if audience not in {resource, _MCP_SERVICE_AUDIENCE}:
        return None, None
    grant = await get_active_grant(str(payload["grant_id"]))
    if not grant or (
        grant.user_id != payload["sub"]
        or grant.client_id != payload["client_id"]
        or grant.audience != audience
        or grant.scope != payload["scope"]
        or grant.security_generation != payload["security_generation"]
    ):
        return None, None
    try:
        user = await _current_user(grant.user_id)
    except McpOAuthError:
        return None, None
    if int(getattr(user, "securityGeneration", 0)) != grant.security_generation:
        return None, None
    if await mfa_needed_for_user(user) and grant.mfa_verified_at is None:
        return None, None
    return TokenData(
        user_id=user.id,
        username=user.username,
        role=user.role,
        exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        mfa_verified=grant.mfa_verified_at is not None,
        auth_methods=grant.auth_methods,
        security_generation=grant.security_generation,
    ), user
