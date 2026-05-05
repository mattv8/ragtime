"""
Authentication module for LDAP and local auth with JWT sessions.

Supports:
- LDAP authentication against Active Directory or OpenLDAP
- Local fallback admin account (env-based, prefixed with "local:")
- JWT tokens in httpOnly cookies
- User sync to PostgreSQL on successful auth
- Role-based access control (user/admin)
"""

import asyncio
import base64
import hashlib
import hmac
import re
import ssl
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import secrets
from fastapi import Request
from jose import JWTError, jwt  # type: ignore[import-untyped]
from ldap3 import ALL, AUTO_BIND_TLS_BEFORE_BIND, SUBTREE, Connection, Server, Tls
from ldap3 import AUTO_BIND_NO_TLS  # type: ignore[import-untyped]
from ldap3.core.exceptions import (  # type: ignore[import-untyped]
    LDAPBindError,
    LDAPException,
)
from ldap3.utils.conv import escape_filter_chars  # type: ignore[import-untyped]
from prisma import Json
from prisma.enums import AuthProvider, UserRole
from pydantic import BaseModel

from ragtime.config.settings import settings
from ragtime.core.database import get_db
from ragtime.core.encryption import decrypt_secret
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

_BROWSER_ORIGIN_HINT_HEADER = "x-ragtime-browser-origin"

# =============================================================================
# Models
# =============================================================================


class TokenData(BaseModel):
    """JWT token payload data."""

    user_id: str
    username: str
    role: str
    exp: datetime


class AuthResult(BaseModel):
    """Result of authentication attempt."""

    success: bool
    user_id: Optional[str] = None
    username: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    role: str = "user"
    error: Optional[str] = None


class LdapDiscoveryResult(BaseModel):
    """Result of LDAP discovery."""

    success: bool
    base_dn: Optional[str] = None
    user_ous: list[str] = []
    groups: list[dict] = []  # [{dn: str, name: str}, ...]
    error: Optional[str] = None


class AuthProviderConfigData(BaseModel):
    """Provider-neutral authentication behavior flags."""

    local_users_enabled: bool = True
    ldap_lazy_sync_enabled: bool = True
    manual_role_override_wins: bool = True
    cache_ttl_minutes: int = 240


class AuthUserProfile(BaseModel):
    """Provider-neutral identity projection used by auth providers."""

    username: str
    source_provider: str
    source_id: Optional[str] = None
    source_dn: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    role: str = "user"
    groups: list[str] = []


PBKDF2_ALGORITHM = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 310_000


# =============================================================================
# JWT Token Management
# =============================================================================


def _is_loopback_hostname(hostname: str | None) -> bool:
    normalized = str(hostname or "").strip().strip(".").lower().strip("[]")
    if not normalized:
        return False
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    return normalized.endswith(".localhost")


def _host_matches_family(
    candidate_host: str | None,
    request_host: str | None,
) -> bool:
    """Return True if candidate_host belongs to a trusted host family.

    Trusted means: identical bare hostname to the active request, a
    loopback alias in both positions, or a subdomain of the configured
    USERSPACE_PREVIEW_BASE_DOMAIN. This constrains how much a client-
    supplied forwarding/browser header can shift the minted origin when
    EXTERNAL_BASE_URL is unset.
    """
    candidate = str(candidate_host or "").strip().lower().strip("[]")
    request_bare = str(request_host or "").strip().lower().strip("[]")
    if not candidate:
        return False
    if candidate == request_bare:
        return True
    if _is_loopback_hostname(candidate) and _is_loopback_hostname(request_bare):
        return True
    preview_base = (
        str(getattr(settings, "userspace_preview_base_domain", "") or "")
        .strip()
        .lower()
    )
    if preview_base:
        if candidate == preview_base or candidate.endswith("." + preview_base):
            return True
    return False


def get_external_origin(request: Request) -> str:
    """Derive the public-facing origin for the current request.

    When EXTERNAL_BASE_URL is set it is authoritative. Otherwise forwarded
    headers are only honored if the host they describe matches the active
    request's host family; any mismatch falls back to the direct request
    origin so hostile Host/X-Forwarded-Host values cannot redirect minted
    preview/share URLs onto unintended hostnames.
    """
    configured = str(getattr(settings, "external_base_url", "") or "").strip()
    if configured:
        return configured.rstrip("/")

    request_scheme = request.url.scheme or "http"
    request_host_header = request.headers.get("host", "") or ""
    request_host_only = urlsplit(f"{request_scheme}://{request_host_header}").hostname

    forwarded_host = request.headers.get("x-forwarded-host", "").strip()
    forwarded_proto = request.headers.get("x-forwarded-proto", "").lower().strip()
    if forwarded_host:
        forwarded_host_only = urlsplit(
            f"{forwarded_proto or request_scheme}://{forwarded_host}"
        ).hostname
        if _host_matches_family(forwarded_host_only, request_host_only):
            return f"{forwarded_proto or request_scheme}://{forwarded_host}"

    if request_host_header:
        return f"{request_scheme}://{request_host_header}"

    return str(request.base_url).rstrip("/")


def _normalize_origin_candidate(origin: str | None) -> str | None:
    raw = str(origin or "").strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def _origin_from_referer(referer: str | None) -> str | None:
    parsed = urlsplit(str(referer or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def _format_origin_host(hostname: str, port: int | None, scheme: str) -> str:
    normalized_host = hostname
    if ":" in normalized_host and not normalized_host.startswith("["):
        normalized_host = f"[{normalized_host}]"
    default_port = (scheme == "https" and port in {None, 443}) or (
        scheme == "http" and port in {None, 80}
    )
    return normalized_host if default_port else f"{normalized_host}:{port}"


def get_browser_matched_origin(
    request: Request,
    *,
    browser_origin: str | None = None,
) -> str:
    """Derive a public origin that preserves the browser-visible host when proxied.

    When EXTERNAL_BASE_URL is configured, it remains the source of truth.
    Otherwise, if the request originated from the UI dev server or another proxy,
    prefer the browser-visible scheme/host while keeping the backend server port.
    """
    server_origin = get_external_origin(request)
    configured = str(getattr(settings, "external_base_url", "") or "").strip()
    if configured:
        return configured.rstrip("/")

    browser_candidate = (
        _normalize_origin_candidate(browser_origin)
        or _normalize_origin_candidate(request.headers.get(_BROWSER_ORIGIN_HINT_HEADER))
        or _normalize_origin_candidate(request.headers.get("origin"))
        or _origin_from_referer(request.headers.get("referer"))
    )
    if not browser_candidate:
        return server_origin

    browser_parts = urlsplit(browser_candidate)
    server_parts = urlsplit(server_origin)
    # Only honor browser-supplied origin hints when they describe a host
    # in the same family as the active request; otherwise fall back to
    # the server-derived origin. This prevents attacker-controlled
    # Origin/Referer/x-ragtime-browser-origin headers from steering
    # minted preview/share URLs onto unrelated hostnames.
    request_host_only = urlsplit(
        f"{request.url.scheme}://{request.headers.get('host', '')}"
    ).hostname
    if not _host_matches_family(browser_parts.hostname, request_host_only):
        return server_origin
    scheme = browser_parts.scheme or server_parts.scheme or request.url.scheme or "http"
    hostname = browser_parts.hostname or server_parts.hostname
    if not hostname:
        return server_origin
    netloc = _format_origin_host(hostname, server_parts.port, scheme)
    return urlunsplit((scheme, netloc, "", "", "")).rstrip("/")


def create_access_token(user_id: str, username: str, role: str) -> str:
    """Create a JWT access token."""
    expire = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expire_hours)
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": expire,
    }
    return encode_jwt_payload(payload)


def encode_jwt_payload(payload: dict[str, Any]) -> str:
    """Encode a JWT payload with Ragtime's configured signing settings."""
    return jwt.encode(
        payload, settings.encryption_key, algorithm=settings.jwt_algorithm
    )


def decode_jwt_payload(
    token: str, *, audience: str | None = None
) -> Optional[dict[str, Any]]:
    """Decode a JWT payload with Ragtime's configured verification settings."""
    try:
        decode_kwargs: dict[str, Any] = {
            "algorithms": [settings.jwt_algorithm],
        }
        if audience is not None:
            decode_kwargs["audience"] = audience
        payload = jwt.decode(token, settings.encryption_key, **decode_kwargs)
        return dict(payload)
    except JWTError as e:
        logger.debug(f"JWT decode error: {e}")
        return None


def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT access token."""
    payload = decode_jwt_payload(token)
    if payload is None:
        return None
    return TokenData(
        user_id=payload["sub"],
        username=payload["username"],
        role=payload["role"],
        exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
    )


def hash_token(token: str) -> str:
    """Create a hash of a token for database storage."""
    return hashlib.sha256(token.encode()).hexdigest()


def hash_local_password(password: str) -> str:
    """Hash a local managed user password with PBKDF2-HMAC-SHA256."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii").rstrip("=")
    digest_b64 = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"{PBKDF2_ALGORITHM}${PBKDF2_ITERATIONS}${salt_b64}${digest_b64}"


def verify_local_password(password: str, stored_hash: str | None) -> bool:
    """Verify a local managed user password against the stored PBKDF2 hash."""
    if not stored_hash:
        return False

    try:
        algorithm, iterations_raw, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if algorithm != PBKDF2_ALGORITHM:
            return False
        iterations = int(iterations_raw)
        salt = base64.urlsafe_b64decode(salt_b64 + "=" * (-len(salt_b64) % 4))
        expected = base64.urlsafe_b64decode(digest_b64 + "=" * (-len(digest_b64) % 4))
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


async def get_auth_provider_config() -> AuthProviderConfigData:
    """Get auth provider policy flags, creating defaults if needed."""
    db = await get_db()
    config = await db.authproviderconfig.find_unique(where={"id": "default"})
    if not config:
        config = await db.authproviderconfig.create(data={"id": "default"})
    return AuthProviderConfigData(
        local_users_enabled=bool(config.localUsersEnabled),
        ldap_lazy_sync_enabled=bool(config.ldapLazySyncEnabled),
        manual_role_override_wins=bool(config.manualRoleOverrideWins),
        cache_ttl_minutes=max(int(config.cacheTtlMinutes or 240), 1),
    )


async def update_auth_provider_config(
    *,
    local_users_enabled: bool | None = None,
    ldap_lazy_sync_enabled: bool | None = None,
    manual_role_override_wins: bool | None = None,
    cache_ttl_minutes: int | None = None,
) -> AuthProviderConfigData:
    """Update provider-neutral authentication policy flags."""
    data: dict[str, Any] = {}
    if local_users_enabled is not None:
        data["localUsersEnabled"] = local_users_enabled
    if ldap_lazy_sync_enabled is not None:
        data["ldapLazySyncEnabled"] = ldap_lazy_sync_enabled
    if manual_role_override_wins is not None:
        data["manualRoleOverrideWins"] = manual_role_override_wins
    if cache_ttl_minutes is not None:
        data["cacheTtlMinutes"] = max(int(cache_ttl_minutes), 1)

    db = await get_db()
    await db.authproviderconfig.upsert(
        where={"id": "default"},
        data={"create": {"id": "default", **data}, "update": data},
    )
    return await get_auth_provider_config()


def _group_key(provider: str, identifier: str) -> str:
    normalized = identifier.strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"{provider}:{digest}"


def _provider_value(provider: Any) -> str:
    return str(getattr(provider, "value", provider) or "")


def _display_name_from_group_identifier(identifier: str) -> str:
    first_part = identifier.split(",", 1)[0]
    if "=" in first_part:
        return first_part.split("=", 1)[1] or identifier
    return first_part or identifier


def _expires_at_is_active(expires_at: Any) -> bool:
    if not expires_at:
        return True
    if isinstance(expires_at, datetime):
        comparable = expires_at
        if comparable.tzinfo is None:
            comparable = comparable.replace(tzinfo=timezone.utc)
        return comparable > datetime.now(timezone.utc)
    return True


def _entry_group_dns(user_entry: Any) -> list[str]:
    if hasattr(user_entry, "memberOf") and user_entry.memberOf:
        return [str(group_dn) for group_dn in user_entry.memberOf]
    return []


def _entry_primary_group_id(user_entry: Any) -> int | None:
    if not hasattr(user_entry, "primaryGroupID") or not user_entry.primaryGroupID:
        return None
    try:
        return int(str(user_entry.primaryGroupID))
    except (TypeError, ValueError):
        return None


def _group_entry_rid(group_entry: Any) -> int | None:
    if hasattr(group_entry, "primaryGroupToken") and group_entry.primaryGroupToken:
        try:
            return int(str(group_entry.primaryGroupToken))
        except (TypeError, ValueError):
            return None

    if hasattr(group_entry, "objectSid") and group_entry.objectSid:
        sid_value = group_entry.objectSid.value
        if isinstance(sid_value, bytes) and len(sid_value) >= 4:
            return int.from_bytes(sid_value[-4:], byteorder="little")

    return None


def _ldap_group_rid(
    ldap_config: Any,
    bind_password: str,
    group_dn: str,
) -> int | None:
    group_conn = _get_ldap_connection(
        ldap_config.serverUrl,
        ldap_config.bindDn,
        bind_password,
        ldap_config.allowSelfSigned,
    )
    if not group_conn:
        return None

    try:
        group_conn.search(
            search_base=group_dn,
            search_filter="(objectClass=*)",
            search_scope="BASE",
            attributes=["primaryGroupToken", "objectSid"],
        )
        if group_conn.entries:
            return _group_entry_rid(group_conn.entries[0])
    except LDAPException as e:
        logger.debug(f"Failed to get RID for group {group_dn}: {e}")
    finally:
        if group_conn.bound:
            group_conn.unbind()

    return None


def _ldap_entry_has_group_dn(
    *,
    user_entry: Any,
    group_dn: str,
    ldap_config: Any | None = None,
    bind_password: str | None = None,
) -> bool:
    """Check direct memberOf and, when possible, AD primary-group membership."""
    group_dn_lower = group_dn.lower()
    member_dns = {group.lower() for group in _entry_group_dns(user_entry)}
    if group_dn_lower in member_dns:
        return True

    if ldap_config is None or bind_password is None:
        return False

    primary_group_id = _entry_primary_group_id(user_entry)
    if not primary_group_id:
        return False

    return _ldap_group_rid(ldap_config, bind_password, group_dn) == primary_group_id


async def _record_auth_sync_event(
    *,
    username: str,
    source_provider: AuthProvider,
    action: str,
    status: str,
    detail: str = "",
    user_id: str | None = None,
) -> None:
    db = await get_db()
    try:
        await db.authsyncevent.create(
            data={
                "userId": user_id,
                "username": username,
                "sourceProvider": source_provider,
                "action": action,
                "status": status,
                "detail": detail,
            }
        )
    except Exception as exc:
        logger.debug(f"Failed to record auth sync event for {username}: {exc}")


# =============================================================================
# LDAP Operations
# =============================================================================


async def get_ldap_config():
    """Get LDAP configuration from database."""
    db = await get_db()
    config = await db.ldapconfig.find_unique(where={"id": "default"})
    if not config:
        # Create default config
        config = await db.ldapconfig.create(data={"id": "default"})
    return config


def _get_ldap_connection(
    server_url: str,
    bind_dn: str,
    bind_password: str,
    allow_self_signed: bool = False,
    connect_timeout: int = 5,
    max_retries: int = 2,
) -> Optional[Connection]:
    """Create an LDAP connection with timeout and retry logic."""
    last_error = None

    for attempt in range(max_retries):
        try:
            # Parse server URL
            use_ssl = server_url.startswith("ldaps://")

            # Configure TLS for self-signed certificates if needed
            tls_config = None
            if use_ssl and allow_self_signed:
                tls_config = Tls(validate=ssl.CERT_NONE)

            server = Server(
                server_url,
                get_info=ALL,
                use_ssl=use_ssl,
                tls=tls_config,
                connect_timeout=connect_timeout,
            )

            auto_bind = AUTO_BIND_TLS_BEFORE_BIND if use_ssl else AUTO_BIND_NO_TLS
            conn = Connection(
                server,
                user=bind_dn,
                password=bind_password,
                auto_bind=auto_bind,
                raise_exceptions=True,
                receive_timeout=connect_timeout,
            )
            return conn
        except LDAPException as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"LDAP connection attempt {attempt + 1} failed: {e}, retrying..."
                )
            continue

    logger.error(f"LDAP connection failed after {max_retries} attempts: {last_error}")
    return None


def _is_invalid_attribute_error(error: LDAPException) -> bool:
    """Return True when an LDAP error indicates an unsupported attribute."""
    error_text = str(error).lower()
    return (
        "invalid attribute type" in error_text
        or "undefined attribute type" in error_text
    )


def _split_username_variants(username: str) -> tuple[str, str]:
    """Return full and short username variants for directory searches."""
    full_username = username.strip()
    normalized = full_username

    if "\\" in normalized:
        normalized = normalized.split("\\", 1)[1]

    short_username = normalized.split("@", 1)[0].strip()
    return full_username, short_username


def _build_default_user_search_filters(username: str) -> list[str]:
    """Build broad LDAP/AD-compatible filter candidates for a username."""
    full_username, short_username = _split_username_variants(username)
    filters: list[str] = []

    def add_filter(attr_name: str, value: str) -> None:
        if not value:
            return
        filter_value = f"({attr_name}={escape_filter_chars(value)})"
        if filter_value not in filters:
            filters.append(filter_value)

    # LDAP-first attributes, followed by AD-specific attributes.
    add_filter("uid", short_username)
    add_filter("mail", full_username)
    add_filter("mailPrimaryAddress", full_username)
    add_filter("mailAlternativeAddress", full_username)
    add_filter("cn", short_username)
    add_filter("sAMAccountName", short_username)
    add_filter("userPrincipalName", full_username)
    return filters


def _build_default_user_search_attributes() -> list[str]:
    """Return ordered fallback LDAP/AD attributes used for user lookup."""
    return [
        "uid",
        "mail",
        "mailPrimaryAddress",
        "mailAlternativeAddress",
        "cn",
        "sAMAccountName",
        "userPrincipalName",
    ]


def _build_ldap_typeahead_filters(query: str) -> list[str]:
    """Build ordered LDAP filters for typeahead across common uid attributes."""
    full_query, short_query = _split_username_variants(query)
    filters: list[str] = []
    attributes = _build_default_user_search_attributes()

    def add_filter(attr_name: str, value: str, *, contains: bool) -> None:
        if not value:
            return
        escaped = escape_filter_chars(value)
        rendered = (
            f"({attr_name}=*{escaped}*)" if contains else f"({attr_name}={escaped}*)"
        )
        if rendered not in filters:
            filters.append(rendered)

    for value in (short_query, full_query):
        for attr_name in attributes:
            add_filter(attr_name, value, contains=False)

    for value in (short_query, full_query):
        for attr_name in attributes:
            add_filter(attr_name, value, contains=True)

    return filters


def _get_first_entry_attribute_value(entry: Any, *attribute_names: str) -> str | None:
    """Return the first populated LDAP entry attribute value as a string."""
    for attribute_name in attribute_names:
        if not hasattr(entry, attribute_name):
            continue

        attribute_value = getattr(entry, attribute_name)
        if attribute_value:
            return str(attribute_value)

    return None


def _get_user_entry_search_attributes() -> list[str]:
    """Return LDAP attributes needed for user lookup and role checks."""
    return ["*", "memberOf"]


def _build_user_search_filters(configured_filter: str, username: str) -> list[str]:
    """Build ordered filter candidates from configured filter + safe fallbacks."""
    filters: list[str] = []

    if configured_filter:
        full_username, short_username = _split_username_variants(username)
        rendered_full = configured_filter.replace(
            "{username}", escape_filter_chars(full_username)
        )
        filters.append(rendered_full)

        if "{username}" in configured_filter and short_username != full_username:
            rendered_short = configured_filter.replace(
                "{username}", escape_filter_chars(short_username)
            )
            if rendered_short not in filters:
                filters.append(rendered_short)

    for fallback_filter in _build_default_user_search_filters(username):
        if fallback_filter not in filters:
            filters.append(fallback_filter)

    return filters


def _search_first_matching_entry(
    conn: Connection,
    search_base: str,
    search_filters: list[str],
    attributes: list[str],
    context: str,
) -> Optional[Any]:
    """Try search filters in order and return first matching LDAP entry."""
    for search_filter in search_filters:
        try:
            conn.search(
                search_base=search_base,
                search_filter=search_filter,
                search_scope=SUBTREE,
                attributes=attributes,
            )
            if conn.entries:
                return conn.entries[0]
        except LDAPException as e:
            if _is_invalid_attribute_error(e):
                logger.debug(
                    f"LDAP filter skipped in {context} due to unsupported attribute: "
                    f"{search_filter} ({e})"
                )
                continue

            logger.debug(
                f"LDAP search failed in {context} for filter {search_filter}: {e}"
            )

    return None


def _ldap_profile_from_entry(
    *,
    user_entry: Any,
    input_username: str,
    role: UserRole,
) -> AuthUserProfile:
    """Convert an LDAP entry into the provider-neutral profile projection."""
    ldap_username = input_username
    if hasattr(user_entry, "sAMAccountName") and user_entry.sAMAccountName:
        ldap_username = str(user_entry.sAMAccountName)
    elif hasattr(user_entry, "uid") and user_entry.uid:
        ldap_username = str(user_entry.uid)

    display_name = _get_first_entry_attribute_value(user_entry, "displayName", "cn")
    email = _get_first_entry_attribute_value(
        user_entry,
        "mail",
        "mailPrimaryAddress",
        "mailAlternativeAddress",
    )

    return AuthUserProfile(
        username=ldap_username,
        source_provider="ldap",
        source_id=str(user_entry.entry_dn),
        source_dn=str(user_entry.entry_dn),
        display_name=display_name or ldap_username,
        email=email,
        role=role,
        groups=_entry_group_dns(user_entry),
    )


async def _sync_user_auth_groups(
    *,
    user_id: str,
    source_provider: AuthProvider,
    group_identifiers: list[str],
    expires_at: datetime | None,
) -> None:
    """Project provider group identifiers into generic AuthGroup memberships."""
    db = await get_db()
    now = datetime.now(timezone.utc)
    group_ids: list[str] = []

    for identifier in group_identifiers:
        normalized_identifier = str(identifier or "").strip()
        if not normalized_identifier:
            continue
        source_provider_value = _provider_value(source_provider)
        key = _group_key(source_provider_value, normalized_identifier)
        create_data: dict[str, Any] = {
            "key": key,
            "displayName": _display_name_from_group_identifier(normalized_identifier),
            "provider": source_provider,
            "sourceId": normalized_identifier,
            "sourceDn": (
                normalized_identifier if source_provider_value == "ldap" else None
            ),
        }
        update_data: dict[str, Any] = {
            "displayName": _display_name_from_group_identifier(normalized_identifier),
            "sourceId": normalized_identifier,
            "sourceDn": (
                normalized_identifier if source_provider_value == "ldap" else None
            ),
        }
        if source_provider_value == "ldap":
            ldap_config = await get_ldap_config()
            normalized_dn = normalized_identifier.casefold()
            admin_group_dns = _normalized_group_dns(
                getattr(ldap_config, "adminGroupDns", [])
            )
            user_group_dns = _normalized_group_dns(
                getattr(ldap_config, "userGroupDns", [])
            )
            create_data["role"] = (
                UserRole.admin if normalized_dn in admin_group_dns else None
            )
            create_data["isLogonGroup"] = normalized_dn in user_group_dns
            update_data["role"] = (
                UserRole.admin if normalized_dn in admin_group_dns else None
            )
            update_data["isLogonGroup"] = normalized_dn in user_group_dns
        group = await db.authgroup.upsert(
            where={"key": key},
            data={
                "create": create_data,
                "update": update_data,
            },
        )
        group_ids.append(group.id)
        await db.authgroupmembership.upsert(
            where={"userId_groupId": {"userId": user_id, "groupId": group.id}},
            data={
                "create": {
                    "userId": user_id,
                    "groupId": group.id,
                    "sourceProvider": source_provider,
                    "sourceSyncedAt": now,
                    "expiresAt": expires_at,
                },
                "update": {
                    "sourceProvider": source_provider,
                    "sourceSyncedAt": now,
                    "expiresAt": expires_at,
                },
            },
        )

    existing = await db.authgroupmembership.find_many(
        where={"userId": user_id, "sourceProvider": source_provider}
    )
    keep_group_ids = set(group_ids)
    for membership in existing:
        if membership.groupId not in keep_group_ids:
            await db.authgroupmembership.delete(where={"id": membership.id})


async def _apply_local_group_role(user: Any, base_role: UserRole) -> UserRole:
    """Apply local managed group role grants to a user's provider-derived role."""
    db = await get_db()
    memberships = await db.authgroupmembership.find_many(
        where={"userId": user.id, "sourceProvider": AuthProvider.local_managed}
    )
    if not memberships:
        return base_role

    for membership in memberships:
        group = await db.authgroup.find_unique(where={"id": membership.groupId})
        if (
            group
            and _provider_value(group.provider) == "local_managed"
            and group.role == UserRole.admin
        ):
            return UserRole.admin
    return base_role


async def _passes_local_logon_gate(user: Any) -> bool:
    """Return whether a local managed user is in a required logon gate group."""
    db = await get_db()
    gate_groups = await db.authgroup.find_many(
        where={"provider": AuthProvider.local_managed, "isLogonGroup": True}
    )
    if not gate_groups:
        return True

    memberships = await db.authgroupmembership.find_many(
        where={"userId": user.id, "sourceProvider": AuthProvider.local_managed}
    )
    gate_group_ids = {group.id for group in gate_groups}
    return any(membership.groupId in gate_group_ids for membership in memberships)


def _normalized_group_dns(values: list[str] | None) -> set[str]:
    return {
        str(value).strip().casefold() for value in (values or []) if str(value).strip()
    }


async def _upsert_provider_user_profile(
    profile: AuthUserProfile,
    *,
    provider: AuthProvider,
    password_hash: str | None = None,
    mark_login: bool = True,
) -> Any:
    """Create/update a user record from a provider-neutral identity projection."""
    db = await get_db()
    auth_config = await get_auth_provider_config()
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=auth_config.cache_ttl_minutes)
    existing_user = await db.user.find_unique(where={"username": profile.username})

    role = UserRole.admin if profile.role == "admin" else UserRole.user
    update_data: dict[str, Any] = {
        "authProvider": provider,
        "ldapDn": profile.source_dn if provider == AuthProvider.ldap else None,
        "sourceProvider": provider,
        "sourceId": profile.source_id,
        "email": profile.email,
        "displayName": profile.display_name,
        "cachedGroups": Json(profile.groups),
        "sourceSyncedAt": now,
        "sourceExpiresAt": expires_at,
    }
    if mark_login:
        update_data["lastLoginAt"] = now
    if password_hash is not None:
        update_data["passwordHash"] = password_hash

    if existing_user:
        if not (
            auth_config.manual_role_override_wins and existing_user.roleManuallySet
        ):
            update_data["role"] = await _apply_local_group_role(existing_user, role)
        user = await db.user.update(where={"id": existing_user.id}, data=update_data)
    else:
        create_data = {
            "username": profile.username,
            **update_data,
            "role": role,
        }
        user = await db.user.create(data=create_data)

    await _sync_user_auth_groups(
        user_id=user.id,
        source_provider=provider,
        group_identifiers=profile.groups,
        expires_at=expires_at,
    )

    if not (auth_config.manual_role_override_wins and user.roleManuallySet):
        effective_role = await _apply_local_group_role(user, user.role)
        if effective_role != user.role:
            user = await db.user.update(
                where={"id": user.id},
                data={"role": effective_role},
            )

    await _record_auth_sync_event(
        username=profile.username,
        source_provider=provider,
        action="upsert_profile",
        status="success",
        user_id=user.id,
    )
    return user


def _discover_ldap_structure_sync(
    server_url: str,
    bind_dn: str,
    bind_password: str,
    allow_self_signed: bool = False,
) -> LdapDiscoveryResult:
    """
    Synchronous LDAP discovery - runs in executor.

    This is the blocking part that must run in a thread pool.
    """
    conn = _get_ldap_connection(server_url, bind_dn, bind_password, allow_self_signed)
    if not conn:
        return LdapDiscoveryResult(
            success=False, error="Failed to connect to LDAP server"
        )

    try:
        # Get base DN from server info
        base_dn = ""
        if conn.server.info and conn.server.info.naming_contexts:
            # Use the first naming context as base DN
            base_dn = str(conn.server.info.naming_contexts[0])
        elif conn.server.info and conn.server.info.other:
            # Try rootDomainNamingContext for AD
            root_nc = conn.server.info.other.get("rootDomainNamingContext")
            if root_nc:
                base_dn = str(root_nc[0])

        if not base_dn:
            return LdapDiscoveryResult(
                success=False, error="Could not determine base DN"
            )

        # Discover user OUs and containers
        ous_and_containers: list[str] = []

        # Search for organizational units
        try:
            conn.search(
                search_base=base_dn,
                search_filter="(objectClass=organizationalUnit)",
                search_scope=SUBTREE,
                attributes=["distinguishedName", "ou"],
            )
            for entry in conn.entries:
                ous_and_containers.append(str(entry.entry_dn))
        except LDAPException as e:
            logger.debug(f"OU search failed (may not be supported): {e}")

        # Also check for CN=Users container (common in AD, may not exist on OpenLDAP)
        try:
            conn.search(
                search_base=base_dn,
                search_filter="(&(objectClass=container)(cn=Users))",
                search_scope=SUBTREE,
                attributes=["distinguishedName"],
            )
            for entry in conn.entries:
                dn = str(entry.entry_dn)
                if dn not in ous_and_containers:
                    ous_and_containers.append(dn)
        except LDAPException as e:
            # container objectClass doesn't exist on OpenLDAP/UCS - this is expected
            logger.debug(f"Container search failed (may not be supported): {e}")

        # Sort by DN depth (fewer components = higher in hierarchy)
        # This creates a tree-like order: DC=..., then OU=...,DC=..., etc.
        def dn_depth(dn: str) -> tuple[int, str]:
            """Return (depth, dn) for sorting - fewer commas = higher level."""
            return (dn.count(","), dn.lower())

        ous_and_containers.sort(key=dn_depth)

        # Always include the base DN as the first/top-level option
        user_ous: list[str] = [base_dn]
        for dn in ous_and_containers:
            if dn != base_dn and dn not in user_ous:
                user_ous.append(dn)

        # Discover groups - search each objectClass separately to handle servers
        # that don't support all objectClasses (AD uses 'group', OpenLDAP uses
        # 'groupOfNames'/'groupOfUniqueNames', POSIX systems use 'posixGroup')
        groups = []
        seen_dns: set[str] = set()

        # List of group objectClasses to try - each searched separately
        # to avoid "invalid class in objectClass" errors on servers that
        # don't have a particular schema
        group_classes = [
            "group",  # Active Directory
            "groupOfNames",  # Standard LDAP
            "groupOfUniqueNames",  # Standard LDAP variant
            "posixGroup",  # POSIX/Unix
        ]

        for obj_class in group_classes:
            try:
                conn.search(
                    search_base=base_dn,
                    search_filter=f"(objectClass={obj_class})",
                    search_scope=SUBTREE,
                    attributes=["distinguishedName", "cn"],
                    size_limit=200,  # Limit per search to avoid timeouts
                )
                for entry in conn.entries:
                    dn = str(entry.entry_dn)
                    if dn not in seen_dns:
                        seen_dns.add(dn)
                        cn = (
                            str(entry.cn)
                            if hasattr(entry, "cn")
                            else dn.split(",")[0].replace("CN=", "")
                        )
                        groups.append({"dn": dn, "name": cn})
            except LDAPException as e:
                # This objectClass doesn't exist on this server - expected
                logger.debug(f"Group search for {obj_class} failed: {e}")

        conn.unbind()

        return LdapDiscoveryResult(
            success=True,
            base_dn=base_dn,
            user_ous=user_ous[:100],  # Limit to 100 for larger directories
            groups=groups[:200],  # Limit to 200
        )

    except LDAPException as e:
        logger.error(f"LDAP discovery error: {e}")
        return LdapDiscoveryResult(success=False, error=str(e))
    finally:
        if conn.bound:
            conn.unbind()


async def discover_ldap_structure(
    server_url: str,
    bind_dn: str,
    bind_password: str,
    allow_self_signed: bool = False,
) -> LdapDiscoveryResult:
    """
    Discover LDAP structure: base DN, user OUs, and groups.

    This auto-discovers the LDAP structure so users don't need to manually
    configure search bases and filters.

    Uses asyncio executor with timeout to prevent blocking the event loop.
    """
    # Parse host and port from server URL for pre-check
    match = re.match(r"ldaps?://([^:]+)(?::(\d+))?", server_url)
    if not match:
        return LdapDiscoveryResult(success=False, error="Invalid LDAP server URL")

    host = match.group(1)
    default_port = 636 if server_url.startswith("ldaps://") else 389
    port = int(match.group(2)) if match.group(2) else default_port

    # Quick socket check to fail fast if server is unreachable
    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=5.0
        )
        writer.close()
        await writer.wait_closed()
    except asyncio.TimeoutError:
        return LdapDiscoveryResult(
            success=False, error=f"Connection timeout to {host}:{port}"
        )
    except ConnectionRefusedError:
        return LdapDiscoveryResult(
            success=False, error=f"Connection refused: {host}:{port}"
        )
    except OSError as e:
        return LdapDiscoveryResult(
            success=False, error=f"Cannot reach {host}:{port}: {e}"
        )

    # Run blocking LDAP discovery in executor with timeout
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: _discover_ldap_structure_sync(
                    server_url, bind_dn, bind_password, allow_self_signed
                ),
            ),
            timeout=15.0,  # 15 second timeout for full discovery
        )
        return result
    except asyncio.TimeoutError:
        return LdapDiscoveryResult(
            success=False, error="LDAP discovery timed out (15s)"
        )


class BindDnLookupResult(BaseModel):
    """Result of bind DN lookup."""

    success: bool
    bind_dn: Optional[str] = None
    display_name: Optional[str] = None
    error: Optional[str] = None


async def lookup_bind_dn(
    server_url: str,
    username: str,
    password: str,
) -> BindDnLookupResult:
    """
    Look up the full DN for a user given their username.

    This attempts to bind with the username directly (works for AD UPN and
    domain\\username formats), and then tries directory searches using
    LDAP/AD-compatible filters.
    """
    try:
        # Parse server URL
        use_ssl = server_url.startswith("ldaps://")
        server = Server(server_url, get_info=ALL, use_ssl=use_ssl)

        # First, try direct bind with username (AD supports this)
        # Try formats: username@domain, domain\username, or just username
        auto_bind = AUTO_BIND_TLS_BEFORE_BIND if use_ssl else AUTO_BIND_NO_TLS

        # Attempt 1: Direct bind (works if username is already a DN or AD-style)
        try:
            conn = Connection(
                server,
                user=username,
                password=password,
                auto_bind=auto_bind,
                raise_exceptions=True,
            )
            # If we get here, bind succeeded - now find the actual DN
            if conn.extend.standard.who_am_i():
                who = conn.extend.standard.who_am_i()
                # Response is like "u:DOMAIN\username" or "dn:CN=..."
                if who and who.startswith("dn:"):
                    bind_dn = who[3:]
                    conn.unbind()
                    return BindDnLookupResult(success=True, bind_dn=bind_dn)

            # Search for our own entry with compatibility fallback filters.
            entry = _search_first_matching_entry(
                conn=conn,
                search_base="",
                search_filters=_build_user_search_filters("", username),
                attributes=["*"],
                context="bind DN lookup (direct bind)",
            )
            if entry:
                bind_dn = str(entry.entry_dn)
                display_name = (
                    str(entry.displayName)
                    if hasattr(entry, "displayName") and entry.displayName
                    else None
                )
                conn.unbind()
                return BindDnLookupResult(
                    success=True, bind_dn=bind_dn, display_name=display_name
                )

            conn.unbind()
        except LDAPBindError:
            pass  # Direct bind failed, will try other methods
        except LDAPException as e:
            logger.debug(f"Direct bind attempt failed: {e}")

        # Attempt 2: Try with rootDSE to get base DN, then search
        try:
            # Connect without binding to get server info
            conn = Connection(server, auto_bind=auto_bind)
            base_dn = ""
            if conn.server.info and conn.server.info.naming_contexts:
                base_dn = str(conn.server.info.naming_contexts[0])

            if base_dn:
                # Now try to bind with username and search
                conn.rebind(user=username, password=password)
                entry = _search_first_matching_entry(
                    conn=conn,
                    search_base=base_dn,
                    search_filters=_build_user_search_filters("", username),
                    attributes=["*"],
                    context="bind DN lookup (base DN)",
                )
                if entry:
                    bind_dn = str(entry.entry_dn)
                    display_name = (
                        str(entry.displayName)
                        if hasattr(entry, "displayName") and entry.displayName
                        else None
                    )
                    conn.unbind()
                    return BindDnLookupResult(
                        success=True, bind_dn=bind_dn, display_name=display_name
                    )
            conn.unbind()
        except LDAPException as e:
            logger.debug(f"Base DN search attempt failed: {e}")

        return BindDnLookupResult(
            success=False,
            error="Could not discover bind DN. Please enter the full DN manually.",
        )

    except LDAPException as e:
        logger.error(f"Bind DN lookup error: {e}")
        return BindDnLookupResult(success=False, error=str(e))


async def authenticate_ldap(username: str, password: str) -> AuthResult:
    """
    Authenticate user against LDAP.

    Steps:
    1. Bind with service account
    2. Search for user by configured filter + LDAP/AD fallback attributes
    3. Attempt bind with user's credentials
    4. Check group membership for role assignment
    5. Sync user to database
    """
    ldap_config = await get_ldap_config()

    if not ldap_config.serverUrl or not ldap_config.bindDn:
        return AuthResult(success=False, error="LDAP not configured")

    # Decrypt bind password if encrypted
    bind_password = decrypt_secret(ldap_config.bindPassword)

    # Connect with service account
    conn = _get_ldap_connection(
        ldap_config.serverUrl,
        ldap_config.bindDn,
        bind_password,
        ldap_config.allowSelfSigned,
    )
    if not conn:
        return AuthResult(success=False, error="Failed to connect to LDAP server")

    try:
        # Build search base
        search_base = ldap_config.userSearchBase or ldap_config.baseDn
        if not search_base:
            return AuthResult(success=False, error="LDAP search base not configured")

        search_filters = _build_user_search_filters(
            ldap_config.userSearchFilter or "(uid={username})",
            username,
        )
        user_entry = _search_first_matching_entry(
            conn=conn,
            search_base=search_base,
            search_filters=search_filters,
            attributes=_get_user_entry_search_attributes(),
            context="LDAP authentication",
        )

        if not user_entry:
            return AuthResult(success=False, error="User not found")

        user_dn = str(user_entry.entry_dn)

        conn.unbind()

        # Verify user's password by binding as the user
        user_conn = _get_ldap_connection(
            ldap_config.serverUrl, user_dn, password, ldap_config.allowSelfSigned
        )
        if not user_conn:
            return AuthResult(success=False, error="Invalid credentials")
        user_conn.unbind()

        # Determine role based on group membership
        try:
            role = _determine_ldap_role_for_entry(
                ldap_config=ldap_config,
                bind_password=bind_password,
                user_entry=user_entry,
                ldap_username=username,
            )
        except ValueError:
            logger.warning(
                f"User {username} not in authorized group. "
                f"required userGroupDns={ldap_config.userGroupDns}"
            )
            return AuthResult(success=False, error="User not in authorized group")

        profile = _ldap_profile_from_entry(
            user_entry=user_entry,
            input_username=username,
            role=role,
        )

        # Sync user to database as a short-lived LDAP cache projection.
        auth_config = await get_auth_provider_config()
        if auth_config.ldap_lazy_sync_enabled:
            user = await _upsert_provider_user_profile(
                profile,
                provider=AuthProvider.ldap,
                mark_login=True,
            )
        else:
            db = await get_db()
            existing_user = await db.user.find_unique(
                where={"username": profile.username}
            )
            if existing_user:
                user = await db.user.update(
                    where={"id": existing_user.id},
                    data={"lastLoginAt": datetime.now(timezone.utc)},
                )
            else:
                user = await _upsert_provider_user_profile(
                    profile,
                    provider=AuthProvider.ldap,
                    mark_login=True,
                )

        if not user:
            return AuthResult(success=False, error="Failed to sync user to database")

        return AuthResult(
            success=True,
            user_id=user.id,
            username=user.username,
            display_name=user.displayName,
            email=user.email,
            role=user.role,
        )

    except LDAPBindError:
        return AuthResult(success=False, error="Invalid credentials")
    except LDAPException as e:
        logger.error(f"LDAP auth error: {e}")
        return AuthResult(success=False, error=f"LDAP error: {str(e)}")
    finally:
        if conn.bound:
            conn.unbind()


async def search_ldap_user_profile(username: str) -> AuthUserProfile | None:
    """Search LDAP for one user and return the provider-neutral profile."""
    ldap_config = await get_ldap_config()
    if not ldap_config.serverUrl or not ldap_config.bindDn:
        return None

    bind_password = decrypt_secret(ldap_config.bindPassword)
    conn = _get_ldap_connection(
        ldap_config.serverUrl,
        ldap_config.bindDn,
        bind_password,
        ldap_config.allowSelfSigned,
    )
    if not conn:
        return None

    try:
        search_base = ldap_config.userSearchBase or ldap_config.baseDn
        if not search_base:
            return None

        user_entry = _search_first_matching_entry(
            conn=conn,
            search_base=search_base,
            search_filters=_build_user_search_filters(
                ldap_config.userSearchFilter or "(uid={username})",
                username,
            ),
            attributes=_get_user_entry_search_attributes(),
            context="LDAP user search",
        )
        if not user_entry:
            return None

        role = _determine_ldap_role_for_entry(
            ldap_config=ldap_config,
            bind_password=bind_password,
            user_entry=user_entry,
            ldap_username=username,
        )
        return _ldap_profile_from_entry(
            user_entry=user_entry,
            input_username=username,
            role=role,
        )
    finally:
        if conn.bound:
            conn.unbind()


async def search_ldap_user_profiles(
    query: str, *, limit: int = 8
) -> list[AuthUserProfile]:
    """Search LDAP for multiple user profiles for admin typeahead workflows."""
    ldap_config = await get_ldap_config()
    if not ldap_config.serverUrl or not ldap_config.bindDn:
        return []

    bind_password = decrypt_secret(ldap_config.bindPassword)
    conn = _get_ldap_connection(
        ldap_config.serverUrl,
        ldap_config.bindDn,
        bind_password,
        ldap_config.allowSelfSigned,
    )
    if not conn:
        return []

    try:
        search_base = ldap_config.userSearchBase or ldap_config.baseDn
        if not search_base:
            return []

        safe_limit = max(1, min(int(limit), 25))
        entries_by_dn: dict[str, Any] = {}

        for search_filter in _build_ldap_typeahead_filters(query):
            try:
                conn.search(
                    search_base=search_base,
                    search_filter=search_filter,
                    search_scope=SUBTREE,
                    attributes=_get_user_entry_search_attributes(),
                    size_limit=max(safe_limit * 3, 20),
                )
            except LDAPException as e:
                if _is_invalid_attribute_error(e):
                    logger.debug(
                        "LDAP typeahead filter skipped due to unsupported attribute: "
                        f"{search_filter} ({e})"
                    )
                    continue
                logger.debug(
                    f"LDAP typeahead search failed for filter {search_filter}: {e}"
                )
                continue

            for entry in conn.entries:
                dn = str(entry.entry_dn)
                if dn not in entries_by_dn:
                    entries_by_dn[dn] = entry
                if len(entries_by_dn) >= safe_limit:
                    break

            if len(entries_by_dn) >= safe_limit:
                break

        profiles: list[AuthUserProfile] = []
        for entry in entries_by_dn.values():
            try:
                role = _determine_ldap_role_for_entry(
                    ldap_config=ldap_config,
                    bind_password=bind_password,
                    user_entry=entry,
                    ldap_username=query,
                )
            except ValueError:
                # Skip users outside configured authorization group constraints.
                continue

            profiles.append(
                _ldap_profile_from_entry(
                    user_entry=entry,
                    input_username=query,
                    role=role,
                )
            )
            if len(profiles) >= safe_limit:
                break

        return profiles
    finally:
        if conn.bound:
            conn.unbind()


async def import_ldap_user_profile(username: str) -> Any | None:
    """Import one LDAP identity into the local user cache without storing a password."""
    profile = await search_ldap_user_profile(username)
    if profile is None:
        return None
    return await _upsert_provider_user_profile(
        profile,
        provider=AuthProvider.ldap,
        mark_login=False,
    )


def _determine_ldap_role_for_entry(
    *,
    ldap_config: Any,
    bind_password: str,
    user_entry: Any,
    ldap_username: str,
) -> UserRole:
    """Resolve LDAP role for a user entry using configured group mappings."""
    role: UserRole = UserRole.user
    admin_group_dns = [
        dn
        for dn in (getattr(ldap_config, "adminGroupDns", []) or [])
        if str(dn).strip()
    ]
    user_group_dns = [
        dn for dn in (getattr(ldap_config, "userGroupDns", []) or []) if str(dn).strip()
    ]

    def is_member_of_group(group_dn: str) -> bool:
        return _ldap_entry_has_group_dn(
            user_entry=user_entry,
            group_dn=group_dn,
            ldap_config=ldap_config,
            bind_password=bind_password,
        )

    member_of = _entry_group_dns(user_entry)
    primary_group_id = _entry_primary_group_id(user_entry)

    logger.debug(
        f"Group membership check for {ldap_username}: "
        f"memberOf={member_of}, primaryGroupID={primary_group_id}, "
        f"adminGroupDns={admin_group_dns}, userGroupDns={user_group_dns}"
    )

    if user_group_dns and not any(
        is_member_of_group(group_dn) for group_dn in user_group_dns
    ):
        raise ValueError("User not in authorized group")

    if any(is_member_of_group(group_dn) for group_dn in admin_group_dns):
        return UserRole.admin

    return role


async def resolve_ldap_role_for_user_dn(
    user_dn: str,
    *,
    ldap_username_hint: str | None = None,
) -> tuple[UserRole | None, str | None]:
    """Resolve current LDAP-derived role for a specific LDAP DN."""
    ldap_config = await get_ldap_config()
    if not ldap_config.serverUrl or not ldap_config.bindDn:
        return None, "LDAP not configured"

    bind_password = decrypt_secret(ldap_config.bindPassword)
    conn = _get_ldap_connection(
        ldap_config.serverUrl,
        ldap_config.bindDn,
        bind_password,
        ldap_config.allowSelfSigned,
    )
    if not conn:
        return None, "Failed to connect to LDAP server"

    try:
        conn.search(
            search_base=user_dn,
            search_filter="(objectClass=*)",
            search_scope="BASE",
            attributes=_get_user_entry_search_attributes(),
        )
        if not conn.entries:
            return None, "User not found"

        user_entry = conn.entries[0]
        ldap_username = ldap_username_hint or user_dn
        if hasattr(user_entry, "sAMAccountName") and user_entry.sAMAccountName:
            ldap_username = str(user_entry.sAMAccountName)
        elif hasattr(user_entry, "uid") and user_entry.uid:
            ldap_username = str(user_entry.uid)

        try:
            role = _determine_ldap_role_for_entry(
                ldap_config=ldap_config,
                bind_password=bind_password,
                user_entry=user_entry,
                ldap_username=ldap_username,
            )
        except ValueError as exc:
            return None, str(exc)

        return role, None
    except LDAPException as e:
        logger.error(f"LDAP role resolution error for {user_dn}: {e}")
        return None, f"LDAP error: {str(e)}"
    finally:
        if conn.bound:
            conn.unbind()


async def ldap_user_is_member_of_group(user_dn: str, group_dn: str) -> bool:
    """Check live LDAP membership for a user DN, including AD primary groups."""
    ldap_config = await get_ldap_config()
    if not ldap_config.serverUrl or not ldap_config.bindDn or not group_dn:
        return False

    bind_password = decrypt_secret(ldap_config.bindPassword)
    conn = _get_ldap_connection(
        ldap_config.serverUrl,
        ldap_config.bindDn,
        bind_password,
        ldap_config.allowSelfSigned,
    )
    if not conn:
        return False

    try:
        conn.search(
            search_base=user_dn,
            search_filter="(objectClass=*)",
            search_scope="BASE",
            attributes=_get_user_entry_search_attributes(),
        )
        if not conn.entries:
            return False

        entry = conn.entries[0]
        return _ldap_entry_has_group_dn(
            user_entry=entry,
            group_dn=group_dn,
            ldap_config=ldap_config,
            bind_password=bind_password,
        )
    except Exception as exc:
        logger.debug(f"LDAP membership check failed for {user_dn}: {exc}")
        return False
    finally:
        if conn.bound:
            conn.unbind()


# =============================================================================
# Local Authentication (Fallback)
# =============================================================================


async def authenticate_local(username: str, password: str) -> AuthResult:
    """
    Authenticate against local admin account.

    The local admin username is prefixed with "local:" in the database
    to avoid collision with LDAP usernames.
    """
    # Check if local admin is configured
    if not settings.local_admin_password:
        return AuthResult(success=False, error="Local admin not configured")

    # Verify credentials
    if (
        username != settings.local_admin_user
        or password != settings.local_admin_password
    ):
        return AuthResult(success=False, error="Invalid credentials")

    # Internal username has "local:" prefix
    internal_username = f"local:{username}"

    # Sync local admin to database
    db = await get_db()
    existing_user = await db.user.find_unique(where={"username": internal_username})

    if existing_user:
        user = await db.user.update(
            where={"id": existing_user.id},
            data={"lastLoginAt": datetime.now(timezone.utc)},
        )
    else:
        user = await db.user.create(
            data={
                "username": internal_username,
                "authProvider": AuthProvider.local,
                "displayName": "Local Admin",
                "role": UserRole.admin,
                "lastLoginAt": datetime.now(timezone.utc),
            }
        )

    if not user:
        return AuthResult(success=False, error="Failed to sync user to database")

    return AuthResult(
        success=True,
        user_id=user.id,
        username=user.username,
        display_name=user.displayName,
        role="admin",
    )


async def authenticate_local_managed(username: str, password: str) -> AuthResult:
    """Authenticate an internally managed local user."""
    auth_config = await get_auth_provider_config()
    if not auth_config.local_users_enabled:
        return AuthResult(success=False, error="Internal users are disabled")

    db = await get_db()
    user = await db.user.find_unique(where={"username": username})
    if not user or _provider_value(user.authProvider) != "local_managed":
        return AuthResult(success=False, error="Invalid credentials")

    if not verify_local_password(password, getattr(user, "passwordHash", None)):
        return AuthResult(success=False, error="Invalid credentials")

    if not await _passes_local_logon_gate(user):
        return AuthResult(success=False, error="User not in authorized group")

    role = await _apply_local_group_role(user, user.role)
    update_data: dict[str, Any] = {"lastLoginAt": datetime.now(timezone.utc)}
    if not (auth_config.manual_role_override_wins and user.roleManuallySet):
        update_data["role"] = role
    user = await db.user.update(where={"id": user.id}, data=update_data)

    return AuthResult(
        success=True,
        user_id=user.id,
        username=user.username,
        display_name=user.displayName,
        email=user.email,
        role=user.role,
    )


async def create_or_update_local_managed_user(
    *,
    username: str,
    password: str | None = None,
    display_name: str | None = None,
    email: str | None = None,
    role: UserRole = UserRole.user,
    user_id: str | None = None,
) -> Any:
    """Create or update an internal managed user."""
    normalized_username = username.strip()
    if not normalized_username:
        raise ValueError("Username is required")

    db = await get_db()
    existing = None
    if user_id:
        existing = await db.user.find_unique(where={"id": user_id})
    else:
        existing = await db.user.find_unique(where={"username": normalized_username})

    data: dict[str, Any] = {
        "username": normalized_username,
        "authProvider": AuthProvider.local_managed,
        "ldapDn": None,
        "sourceProvider": AuthProvider.local_managed,
        "sourceId": normalized_username,
        "cachedGroups": Json([]),
        "sourceSyncedAt": None,
        "sourceExpiresAt": None,
        "displayName": display_name or normalized_username,
        "email": email,
        "role": role,
    }
    if password:
        data["passwordHash"] = hash_local_password(password)

    if existing:
        if _provider_value(existing.authProvider) not in {"local_managed", "ldap"}:
            raise ValueError("Cannot convert this account type")
        if not password:
            data.pop("passwordHash", None)
        user = await db.user.update(where={"id": existing.id}, data=data)
    else:
        if not password:
            raise ValueError("Password is required for new internal users")
        user = await db.user.create(data=data)

    await _record_auth_sync_event(
        username=user.username,
        source_provider=AuthProvider.local_managed,
        action="upsert_local_user",
        status="success",
        user_id=user.id,
    )
    return user


async def user_matches_group_identifier(user: Any, group_identifier: str) -> bool:
    """Return whether a user matches a provider-neutral group key or LDAP DN."""
    normalized_identifier = str(group_identifier or "").strip()
    if not normalized_identifier:
        return True

    if _provider_value(user.authProvider) == "local" and user.role == "admin":
        return True

    identifier_lower = normalized_identifier.lower()
    cached_groups = getattr(user, "cachedGroups", None)
    if isinstance(cached_groups, list) and _expires_at_is_active(
        getattr(user, "sourceExpiresAt", None)
    ):
        if identifier_lower in {str(group).lower() for group in cached_groups}:
            return True

    db = await get_db()
    candidate_groups = await db.authgroup.find_many(
        where={
            "OR": [
                {"key": normalized_identifier},
                {"sourceDn": normalized_identifier},
                {"sourceId": normalized_identifier},
            ]
        }
    )
    for group in candidate_groups:
        membership = await db.authgroupmembership.find_unique(
            where={"userId_groupId": {"userId": user.id, "groupId": group.id}}
        )
        if membership and _expires_at_is_active(getattr(membership, "expiresAt", None)):
            return True

    if getattr(user, "ldapDn", None):
        return await ldap_user_is_member_of_group(user.ldapDn, normalized_identifier)

    return False


# =============================================================================
# Combined Authentication
# =============================================================================


async def authenticate(username: str, password: str) -> AuthResult:
    """
    Authenticate user via local managed, LDAP, or local admin fallback.

    Order:
    1. Local admin env account when the username matches it
    2. Internal managed user database
    3. LDAP live password verification and lazy identity/group sync
    4. Local admin fallback for legacy behavior
    """
    # Check for local admin login first (username matches local admin)
    if username == settings.local_admin_user and settings.local_admin_password:
        local_result = await authenticate_local(username, password)
        if local_result.success:
            return local_result

    local_managed_result = await authenticate_local_managed(username, password)
    if local_managed_result.success:
        return local_managed_result

    # Try LDAP if configured in database (serverUrl is set)
    ldap_config = await get_ldap_config()
    if ldap_config.serverUrl:
        ldap_result = await authenticate_ldap(username, password)
        if ldap_result.success:
            return ldap_result
        # If LDAP explicitly fails (not just "not configured"), return error
        if ldap_result.error and ldap_result.error != "LDAP not configured":
            # Still try local fallback for local admin
            if username == settings.local_admin_user:
                return await authenticate_local(username, password)
            return ldap_result

    # Fallback to local admin
    if username == settings.local_admin_user:
        return await authenticate_local(username, password)

    return AuthResult(success=False, error="Authentication failed")


# =============================================================================
# Session Management
# =============================================================================


async def create_session(
    user_id: str,
    token: str,
    user_agent: Optional[str] = None,
    ip_address: Optional[str] = None,
):
    """Create a session record in the database."""
    db = await get_db()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expire_hours)

    await db.session.create(
        data={
            "userId": user_id,
            "tokenHash": hash_token(token),
            "expiresAt": expires_at,
            "userAgent": user_agent,
            "ipAddress": ip_address,
        }
    )


async def validate_session(token: str) -> Optional[TokenData]:
    """Validate a session token against the database."""
    token_data = decode_access_token(token)
    if not token_data:
        return None

    # Check if session exists and is not expired
    db = await get_db()
    session = await db.session.find_first(
        where={
            "tokenHash": hash_token(token),
            "expiresAt": {"gt": datetime.now(timezone.utc)},
        }
    )

    if not session:
        return None

    return token_data


async def invalidate_session(token: str):
    """Invalidate a session (logout)."""
    db = await get_db()
    await db.session.delete_many(where={"tokenHash": hash_token(token)})


async def invalidate_all_sessions(user_id: str):
    """Invalidate all sessions for a user."""
    db = await get_db()
    await db.session.delete_many(where={"userId": user_id})


async def cleanup_expired_sessions():
    """Remove expired sessions from the database."""
    db = await get_db()
    result = await db.session.delete_many(
        where={"expiresAt": {"lt": datetime.now(timezone.utc)}}
    )
    if result:
        logger.info(f"Cleaned up {result} expired sessions")
