"""Authentication primitives for the external workspace-development surface."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, FrozenSet

from fastapi import HTTPException, Request, status
from prisma import Json

from ragtime.core.database import get_db
from ragtime.core.datetimes import utc_now
from ragtime.core.security import extract_bearer_token, get_current_user

DEVELOPMENT_CREDENTIAL_PREFIX = "rtdev"
DEVELOPMENT_SCOPES = frozenset({"read", "write", "exec"})
_SELECTOR_BYTES = 16
_SECRET_BYTES = 32


@dataclass(frozen=True)
class DevelopmentPrincipal:
    user_id: str
    is_admin: bool
    credential_id: str | None = None
    workspace_id: str | None = None
    scopes: FrozenSet[str] = frozenset({"read", "write", "exec"})


def _unauthorized() -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid development credential", headers={"WWW-Authenticate": "Bearer"})


def _parse_development_token(token: str) -> tuple[str, str]:
    parts = str(token or "").split("_", 2)
    if len(parts) != 3 or parts[0] != DEVELOPMENT_CREDENTIAL_PREFIX or len(parts[1]) != _SELECTOR_BYTES * 2 or not parts[2]:
        raise _unauthorized()
    if any(char not in "0123456789abcdef" for char in parts[1]):
        raise _unauthorized()
    return parts[1], parts[2]


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _normalize_scopes(scopes: Any) -> FrozenSet[str]:
    if not isinstance(scopes, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(scope) for scope in scopes if str(scope) in DEVELOPMENT_SCOPES)


def build_development_credential_token() -> tuple[str, str, str]:
    selector = secrets.token_hex(_SELECTOR_BYTES)
    token = f"{DEVELOPMENT_CREDENTIAL_PREFIX}_{selector}_{secrets.token_urlsafe(_SECRET_BYTES)}"
    return token, selector, _hash_token(token)


async def resolve_development_principal(request: Request) -> DevelopmentPrincipal:
    """Resolve a session user or a fresh, scoped development credential.

    A token with the ``rtdev_`` prefix is never handed to session JWT parsing.
    Other bearer values retain the existing session validation behavior.
    """
    bearer = extract_bearer_token(request.headers)
    if bearer and bearer.startswith(f"{DEVELOPMENT_CREDENTIAL_PREFIX}_"):
        selector, _secret = _parse_development_token(bearer)
        db = await get_db()
        credential = await db.workspacedevelopmentcredential.find_unique(where={"selector": selector})
        if credential is None or not hmac.compare_digest(str(getattr(credential, "tokenHash", "")), _hash_token(bearer)):
            raise _unauthorized()
        now = utc_now()
        expires_at = getattr(credential, "expiresAt", None)
        if getattr(credential, "revokedAt", None) is not None:
            raise _unauthorized()
        if expires_at is not None and (not isinstance(expires_at, datetime) or expires_at <= now):
            raise _unauthorized()
        user = await db.user.find_unique(where={"id": credential.userId})
        if user is None:
            raise _unauthorized()
        scopes = _normalize_scopes(getattr(credential, "scopes", None))
        if not scopes:
            raise _unauthorized()
        return DevelopmentPrincipal(
            user_id=str(user.id),
            is_admin=str(getattr(user, "role", "")) == "admin",
            credential_id=str(credential.id),
            workspace_id=str(credential.workspaceId),
            scopes=scopes,
        )

    # Preserve the existing fail-closed session JWT parser for cookie/JWT auth.
    token = request.cookies.get("ragtime_session") or bearer
    user = await get_current_user(token=token)
    return DevelopmentPrincipal(user_id=str(user.id), is_admin=str(getattr(user, "role", "")) == "admin")


async def create_workspace_development_credential(
    *, workspace_id: str, user_id: str, name: str, scopes: list[str], expires_at: datetime | None
) -> dict[str, Any]:
    cleaned_name = str(name or "").strip()
    normalized_scopes = _normalize_scopes(scopes)
    if not (1 <= len(cleaned_name) <= 100):
        raise HTTPException(status_code=400, detail="name must be 1-100 characters")
    if not normalized_scopes:
        raise HTTPException(status_code=400, detail="At least one of read, write, or exec scopes is required")
    if expires_at is not None and expires_at <= utc_now():
        raise HTTPException(status_code=400, detail="expires_at must be in the future")
    token, selector, token_hash = build_development_credential_token()
    db = await get_db()
    row = await db.workspacedevelopmentcredential.create(
        data={
            "id": str(uuid.uuid4()),
            "workspace": {"connect": {"id": workspace_id}},
            "user": {"connect": {"id": user_id}},
            "selector": selector,
            "tokenHash": token_hash,
            "name": cleaned_name,
            "scopes": Json(sorted(normalized_scopes)),
            "expiresAt": expires_at,
        }
    )
    payload = development_credential_response(row)
    payload["token"] = token
    return payload


def development_credential_response(row: Any) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "workspace_id": str(row.workspaceId),
        "user_id": str(row.userId),
        "name": str(row.name),
        "scopes": sorted(_normalize_scopes(getattr(row, "scopes", []))),
        "expires_at": row.expiresAt.isoformat() if getattr(row, "expiresAt", None) else None,
        "revoked_at": row.revokedAt.isoformat() if getattr(row, "revokedAt", None) else None,
        "created_at": row.createdAt.isoformat() if getattr(row, "createdAt", None) else None,
        "updated_at": row.updatedAt.isoformat() if getattr(row, "updatedAt", None) else None,
    }


async def rotate_workspace_development_credential(*, workspace_id: str, credential_id: str) -> dict[str, Any]:
    db = await get_db()
    row = await db.workspacedevelopmentcredential.find_unique(where={"id": credential_id})
    if row is None or str(row.workspaceId) != workspace_id:
        raise HTTPException(status_code=404, detail="Development credential not found")
    if getattr(row, "revokedAt", None) is not None:
        raise HTTPException(status_code=400, detail="Revoked credentials cannot be rotated")
    token, selector, token_hash = build_development_credential_token()
    updated = await db.workspacedevelopmentcredential.update(where={"id": credential_id}, data={"selector": selector, "tokenHash": token_hash})
    payload = development_credential_response(updated)
    payload["token"] = token
    return payload


async def revoke_workspace_development_credential(*, workspace_id: str, credential_id: str) -> dict[str, Any]:
    db = await get_db()
    row = await db.workspacedevelopmentcredential.find_unique(where={"id": credential_id})
    if row is None or str(row.workspaceId) != workspace_id:
        raise HTTPException(status_code=404, detail="Development credential not found")
    updated = await db.workspacedevelopmentcredential.update(where={"id": credential_id}, data={"revokedAt": utc_now()})
    return development_credential_response(updated)
