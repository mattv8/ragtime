"""Helpers for normalizing user identity for public and prompt-facing use."""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Mapping

from ragtime.config.settings import settings

_LOCAL_USERNAME_PREFIX = "local:"
USER_FINGERPRINT_SCOPE_WORKSPACE = "workspace"
USER_FINGERPRINT_VERSION = "v1"
_USER_FINGERPRINT_NAMESPACE = "userspace_audit_fingerprint"


def normalize_user_identity(
    username: str | None,
    display_name: str | None = None,
    *,
    lowercase_username: bool = False,
) -> tuple[str, str | None]:
    """Normalize stored user identity for user-facing rendering.

    Local users are stored with a ``local:`` username prefix to avoid
    collisions with LDAP users. User-facing surfaces should not expose that
    storage detail, so this helper strips the prefix, trims whitespace, and
    optionally lowercases the public username for slug/share-path use.
    """

    normalized_username = str(username or "").strip()
    if normalized_username[: len(_LOCAL_USERNAME_PREFIX)].lower() == _LOCAL_USERNAME_PREFIX:
        normalized_username = normalized_username[len(_LOCAL_USERNAME_PREFIX) :].strip()

    if lowercase_username:
        normalized_username = normalized_username.lower()

    normalized_display_name = str(display_name or "").strip() or None
    return normalized_username, normalized_display_name


def build_user_fingerprint_subject(
    *,
    user_id: str,
    username: str | None = None,
    auth_provider: Any = None,
    ldap_dn: str | None = None,
    source_provider: Any = None,
    source_id: str | None = None,
) -> str:
    """Return a stable account subject for workspace-scoped fingerprints."""
    provider = str(getattr(source_provider, "value", source_provider) or getattr(auth_provider, "value", auth_provider) or "user").strip().lower()
    stable_id = str(source_id or "").strip()
    if stable_id:
        return f"{provider}:source:{stable_id}"

    normalized_ldap_dn = str(ldap_dn or "").strip().lower()
    if normalized_ldap_dn:
        return f"{provider}:ldap_dn:{normalized_ldap_dn}"

    normalized_username, _ = normalize_user_identity(username, lowercase_username=True)
    if normalized_username:
        return f"{provider}:username:{normalized_username}"

    return f"ragtime:user_id:{str(user_id or '').strip()}"


def build_workspace_user_fingerprint(
    *,
    user_id: str,
    workspace_id: str,
    workspace_fingerprint_namespace: str | None = None,
    user_identity_subject: str | None = None,
) -> str:
    """Return a deterministic workspace-scoped audit fingerprint for a user."""
    normalized_user_id = str(user_id or "").strip()
    normalized_workspace_id = str(workspace_id or "").strip()
    if not normalized_user_id or not normalized_workspace_id:
        return ""

    normalized_namespace = str(workspace_fingerprint_namespace or "").strip()
    normalized_subject = str(user_identity_subject or "").strip() or f"ragtime:user_id:{normalized_user_id}"
    if normalized_namespace:
        payload = f"{_USER_FINGERPRINT_NAMESPACE}:{USER_FINGERPRINT_VERSION}:{normalized_subject}".encode("utf-8")
        digest = hmac.new(normalized_namespace.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        return f"{USER_FINGERPRINT_VERSION}:{digest}"

    key = settings.encryption_key.encode("utf-8")
    payload = f"{_USER_FINGERPRINT_NAMESPACE}:{USER_FINGERPRINT_VERSION}:{normalized_workspace_id}:{normalized_user_id}".encode("utf-8")
    digest = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return f"{USER_FINGERPRINT_VERSION}:{digest}"


def add_workspace_user_fingerprint(
    payload: Mapping[str, Any] | None,
    *,
    user_id: str | None,
    workspace_id: str,
    workspace_fingerprint_namespace: str | None = None,
    user_identity_subject: str | None = None,
) -> dict[str, Any]:
    """Add workspace-scoped fingerprint metadata to an audit payload."""
    enriched = dict(payload or {})
    fingerprint = build_workspace_user_fingerprint(
        user_id=str(user_id or ""),
        workspace_id=workspace_id,
        workspace_fingerprint_namespace=workspace_fingerprint_namespace,
        user_identity_subject=user_identity_subject,
    )
    if not fingerprint:
        return enriched

    enriched.setdefault("user_fingerprint", fingerprint)
    enriched.setdefault("user_fingerprint_scope", USER_FINGERPRINT_SCOPE_WORKSPACE)
    enriched.setdefault("user_fingerprint_version", USER_FINGERPRINT_VERSION)
    return enriched
