from __future__ import annotations

import asyncio
from typing import Any, TypedDict

from prisma.enums import AuthProvider

from ragtime.config.settings import settings
from ragtime.core.auth import (
    discover_ldap_structure,
    get_auth_provider_config,
    get_ldap_config,
)
from ragtime.core.database import get_db
from ragtime.core.encryption import decrypt_secret
from ragtime.core.logging import get_logger

logger = get_logger(__name__)


class AuthMethodStatusPayload(TypedDict):
    key: str
    label: str
    configured: bool
    available: bool
    status: str
    detail: str | None


_AUTH_METHOD_STATUS_NOT_CONFIGURED = "not_configured"
_AUTH_METHOD_STATUS_UNAVAILABLE = "unavailable"
_AUTH_METHOD_STATUS_AVAILABLE = "available"


def normalize_auth_method_key(value: Any) -> str:
    """Normalize auth provider values into stable auth method keys."""
    return str(getattr(value, "value", value) or "").strip().lower()


async def _build_local_auth_method_status() -> AuthMethodStatusPayload:
    is_configured = bool(settings.local_admin_password)
    if not is_configured:
        return {
            "key": "local",
            "label": "Local Admin",
            "configured": False,
            "available": False,
            "status": _AUTH_METHOD_STATUS_NOT_CONFIGURED,
            "detail": "Not configured",
        }

    return {
        "key": "local",
        "label": "Local Admin",
        "configured": True,
        "available": True,
        "status": _AUTH_METHOD_STATUS_AVAILABLE,
        "detail": "Ready",
    }


async def _build_local_managed_auth_method_status() -> AuthMethodStatusPayload:
    config = await get_auth_provider_config()
    if not config.local_users_enabled:
        return {
            "key": "local_managed",
            "label": "Internal Users",
            "configured": True,
            "available": False,
            "status": _AUTH_METHOD_STATUS_UNAVAILABLE,
            "detail": "Disabled",
        }

    db = await get_db()
    user_count = await db.user.count(where={"authProvider": AuthProvider.local_managed})
    return {
        "key": "local_managed",
        "label": "Internal Users",
        "configured": True,
        "available": True,
        "status": _AUTH_METHOD_STATUS_AVAILABLE,
        "detail": f"{user_count} managed user{'s' if user_count != 1 else ''}",
    }


async def _build_ldap_auth_method_status() -> AuthMethodStatusPayload:
    ldap_config = await get_ldap_config()
    if not ldap_config.serverUrl:
        return {
            "key": "ldap",
            "label": "LDAP",
            "configured": False,
            "available": False,
            "status": _AUTH_METHOD_STATUS_NOT_CONFIGURED,
            "detail": "Not configured",
        }

    if not ldap_config.bindDn or not ldap_config.bindPassword:
        return {
            "key": "ldap",
            "label": "LDAP",
            "configured": True,
            "available": False,
            "status": _AUTH_METHOD_STATUS_UNAVAILABLE,
            "detail": "Configuration incomplete",
        }

    bind_password = decrypt_secret(ldap_config.bindPassword)
    if not bind_password:
        return {
            "key": "ldap",
            "label": "LDAP",
            "configured": True,
            "available": False,
            "status": _AUTH_METHOD_STATUS_UNAVAILABLE,
            "detail": "Bind password unavailable",
        }

    try:
        discovery = await asyncio.wait_for(
            discover_ldap_structure(
                server_url=ldap_config.serverUrl,
                bind_dn=ldap_config.bindDn,
                bind_password=bind_password,
                allow_self_signed=ldap_config.allowSelfSigned,
            ),
            timeout=5.0,
        )
    except TimeoutError:
        return {
            "key": "ldap",
            "label": "LDAP",
            "configured": True,
            "available": False,
            "status": _AUTH_METHOD_STATUS_UNAVAILABLE,
            "detail": "Connection check timed out",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("LDAP availability check failed: %s", exc)
        return {
            "key": "ldap",
            "label": "LDAP",
            "configured": True,
            "available": False,
            "status": _AUTH_METHOD_STATUS_UNAVAILABLE,
            "detail": "Connection check failed",
        }

    if discovery.success:
        return {
            "key": "ldap",
            "label": "LDAP",
            "configured": True,
            "available": True,
            "status": _AUTH_METHOD_STATUS_AVAILABLE,
            "detail": "Reachable",
        }

    return {
        "key": "ldap",
        "label": "LDAP",
        "configured": True,
        "available": False,
        "status": _AUTH_METHOD_STATUS_UNAVAILABLE,
        "detail": "Unreachable",
    }


async def build_auth_method_statuses() -> list[AuthMethodStatusPayload]:
    """Return platform auth method availability for UI/userspace primitives."""
    statuses = await asyncio.gather(
        _build_ldap_auth_method_status(),
        _build_local_managed_auth_method_status(),
        _build_local_auth_method_status(),
    )
    return list(statuses)
