"""
Authentication API routes.

Endpoints for login, logout, user info, and LDAP configuration.
"""

import base64
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional, Tuple, cast
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

from fastapi import (
    APIRouter,
    Body,
    Depends,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse
from prisma import Json
from prisma.enums import AuthProvider, UserRole
from prisma.models import User
from pydantic import BaseModel, Field

from ragtime.config.settings import settings
from ragtime.core.api_accounting import (
    get_api_daily_trend,
    get_api_provider_model_breakdown,
)
from ragtime.core.app_setting_defaults import (
    DEFAULT_AUTHENTICATED_WEBGL_BACKGROUND_ENABLED,
    DEFAULT_CHAT_AUTO_COMPACTION_THRESHOLD_PERCENT,
    DEFAULT_CHAT_COMPACTION_THRESHOLD_PERCENT,
    DEFAULT_SERVER_NAME,
)
from ragtime.core.app_settings import get_app_settings, invalidate_settings_cache
from ragtime.core.auth import (
    _UNSET,
    authenticate,
    create_or_update_local_managed_user,
    discover_ldap_structure,
    get_auth_provider_config,
    get_ldap_config,
    import_ldap_user_profile,
    invalidate_all_sessions,
    invalidate_session,
    issue_authenticated_session,
    lookup_bind_dn,
    recompute_auth_group_member_roles,
    recompute_user_effective_role,
    resolve_ldap_role_for_user_dn,
    search_ldap_user_profile,
    search_ldap_user_profiles,
    update_auth_provider_config,
)
from ragtime.core.auth_methods import build_auth_method_statuses
from ragtime.core.database import get_db
from ragtime.core.encryption import decrypt_secret, encrypt_secret
from ragtime.core.logging import get_logger
from ragtime.core.mcp_accounting import (
    get_mcp_daily_trend,
    get_mcp_usage_by_route,
    get_mcp_usage_by_user,
)
from ragtime.core.mfa import (
    MFA_TRUST_COOKIE_NAME,
    RECOVERY_CODE_COUNT,
    WEBAUTHN_METHOD,
    begin_totp_enrollment,
    confirm_totp_enrollment,
    create_pending_mfa_token,
    create_trusted_device,
    decode_pending_mfa_token,
    generate_recovery_code,
    generate_totp_code,
    get_allowed_mfa_methods,
    get_enabled_totp_factor,
    hash_recovery_code,
    mfa_needed_for_user,
    regenerate_recovery_codes,
    reset_user_mfa,
    resolve_preferred_mfa_method,
    trusted_device_satisfies_mfa,
    user_allowed_enrolled_methods,
    user_has_enabled_totp,
    verify_user_mfa_code,
)
from ragtime.core.rate_limit import LOGIN_RATE_LIMIT, limiter
from ragtime.core.security import (
    get_current_user,
    get_current_user_optional,
    get_session_token,
    require_admin,
)
from ragtime.core.theme import canonicalize_theme_pack_id
from ragtime.core.usage_accounting import (
    get_daily_provider_failures,
    get_daily_usage_trend,
    get_provider_model_breakdown,
    get_usage_earliest_date,
    get_user_daily_usage_series,
    get_user_usage_summary,
)
from ragtime.core.webauthn_mfa import (
    WebauthnError,
    begin_webauthn_authentication,
    begin_webauthn_registration,
    complete_webauthn_authentication,
    complete_webauthn_registration,
    delete_webauthn_credential,
    list_webauthn_credentials,
    rename_webauthn_credential,
    user_has_enabled_webauthn,
)
from ragtime.oauth_redirects import (
    DEFAULT_TRUSTED_REDIRECT_URIS,
    LOOPBACK_REDIRECT_HOSTS,
    TRUSTED_IDE_REDIRECT_HOSTS,
    TRUSTED_NATIVE_REDIRECT_SCHEMES,
    build_allowed_origins,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# In-memory storage for authorization codes (short-lived, 10 min expiry)
# Format: {code: {"client_id": str, "redirect_uri": str, "code_challenge": str,
#                 "user_id": str, "username": str, "role": str, "expires": float}}
_auth_codes: dict[str, dict] = {}
AUTH_CODE_EXPIRY = 600  # 10 minutes
MAX_AUTH_CODES = 10000  # Prevent memory exhaustion from code accumulation


@dataclass(frozen=True)
class RedirectUriValidationResult:
    is_valid: bool
    summary: str = ""
    log_message: str = ""
    normalized_uri: str = ""
    callback_origin: str = ""
    next_steps: Tuple[str, ...] = ()


def _usage_window_start(days: int) -> datetime:
    """Return the UTC start-of-day for an inclusive N-day usage window."""
    today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return today_utc - timedelta(days=days - 1)


def _normalize_redirect_uri(uri: str) -> str:
    parsed_uri = urlparse(uri)
    if parsed_uri.hostname is None:
        return ""

    host = parsed_uri.hostname.lower()
    port = f":{parsed_uri.port}" if parsed_uri.port is not None else ""
    path = parsed_uri.path or ""
    return f"{parsed_uri.scheme.lower()}://{host}{port}{path}"


def _redirect_uri_origin(uri: str) -> str:
    parsed_uri = urlparse(uri)
    if parsed_uri.hostname is None or not parsed_uri.scheme:
        return ""
    port = f":{parsed_uri.port}" if parsed_uri.port is not None else ""
    return f"{parsed_uri.scheme.lower()}://{parsed_uri.hostname.lower()}{port}"


def build_oauth_redirect_url(redirect_uri: str, code: str, state: str = "") -> str:
    """Append OAuth success parameters while preserving the callback URI."""
    parsed_uri = urlsplit(redirect_uri)
    query_params = [(key, value) for key, value in parse_qsl(parsed_uri.query, keep_blank_values=True) if key not in {"code", "state"}]
    query_params.append(("code", code))
    if state:
        query_params.append(("state", state))
    return urlunsplit(
        (
            parsed_uri.scheme,
            parsed_uri.netloc,
            parsed_uri.path,
            urlencode(query_params, doseq=True),
            parsed_uri.fragment,
        )
    )


def _redirect_uri_allow_origin_step(origin: str) -> Optional[str]:
    if not origin or settings.allowed_origins == "*":
        return None
    if origin in build_allowed_origins(settings.allowed_origins):
        return None
    return f"If this callback exchanges the authorization code from a browser on '{origin}', also add '{origin}' to ALLOWED_ORIGINS."


def validate_redirect_uri(redirect_uri: str) -> RedirectUriValidationResult:
    """
    Validate redirect_uri for OAuth2 security.

    For MCP clients (VS Code, JetBrains, etc.), allow:
    - Registered private-use URI schemes for native apps (RFC 8252 §7.1)
      e.g. vscode://, cursor://, windsurf://, jetbrains://
    - Loopback addresses per RFC 8252 Section 7.3
    - Trusted IDE redirect domains (vscode.dev, etc.)
    - Exact URIs listed in OAUTH_TRUSTED_REDIRECT_URIS (any scheme)

    This prevents open redirect attacks while supporting OAuth flows from
    various development tools and native desktop clients.

    RFC 8252 Section 7.3: Loopback Interface Redirection
    - http://127.0.0.1:<port>/<path>
    - http://localhost:<port>/<path>
    - http://[::1]:<port>/<path>
    """
    try:
        parsed = urlparse(redirect_uri)
        normalized_redirect_uri = _normalize_redirect_uri(redirect_uri)
        callback_origin = _redirect_uri_origin(redirect_uri)

        if parsed.scheme in TRUSTED_NATIVE_REDIRECT_SCHEMES:
            if settings.debug_mode:
                logger.debug(f"OAuth2 redirect_uri validated (native app scheme): {redirect_uri}")
            return RedirectUriValidationResult(
                is_valid=True,
                normalized_uri=normalized_redirect_uri,
                callback_origin=callback_origin,
            )

        if parsed.scheme not in ("http", "https"):
            result = RedirectUriValidationResult(
                is_valid=False,
                summary=(
                    "Invalid redirect_uri. Use a loopback http(s) callback, a trusted IDE callback domain, or a registered native-app scheme such as vscode://."
                ),
                log_message=(
                    "OAuth2 redirect_uri rejected: scheme must be http or https or a "
                    f"registered native-app scheme, got {parsed.scheme!r} "
                    f"(redirect_uri={redirect_uri!r})"
                ),
                normalized_uri=normalized_redirect_uri,
                callback_origin=callback_origin,
            )
            logger.warning(result.log_message)
            return result

        hostname = parsed.hostname
        if hostname is None:
            result = RedirectUriValidationResult(
                is_valid=False,
                summary="Invalid redirect_uri. The callback URL must include a hostname.",
                log_message=(f"OAuth2 redirect_uri rejected: no hostname (redirect_uri={redirect_uri!r})"),
                normalized_uri=normalized_redirect_uri,
                callback_origin=callback_origin,
            )
            logger.warning(result.log_message)
            return result

        loopback_hosts = LOOPBACK_REDIRECT_HOSTS
        trusted_domains = TRUSTED_IDE_REDIRECT_HOSTS
        trusted_redirect_uris = set(DEFAULT_TRUSTED_REDIRECT_URIS)

        if settings.oauth_trusted_redirect_uris.strip():
            for raw_uri in settings.oauth_trusted_redirect_uris.split(","):
                candidate = raw_uri.strip()
                if not candidate:
                    continue
                normalized_candidate = _normalize_redirect_uri(candidate)
                if normalized_candidate:
                    trusted_redirect_uris.add(normalized_candidate)

        is_loopback = hostname in loopback_hosts
        is_trusted = hostname in trusted_domains
        is_trusted_uri = normalized_redirect_uri in trusted_redirect_uris

        if not is_loopback and not is_trusted and not is_trusted_uri:
            next_steps = [f"Add '{normalized_redirect_uri}' to OAUTH_TRUSTED_REDIRECT_URIS if this callback should be allowed."]
            allow_origin_step = _redirect_uri_allow_origin_step(callback_origin)
            if allow_origin_step:
                next_steps.append(allow_origin_step)
            result = RedirectUriValidationResult(
                is_valid=False,
                summary=(
                    f"Invalid redirect_uri '{normalized_redirect_uri}'. Ragtime only allows "
                    "loopback callbacks, trusted IDE callback domains, or exact callback URLs "
                    "listed in OAUTH_TRUSTED_REDIRECT_URIS."
                ),
                log_message=(
                    "OAuth2 redirect_uri rejected: '%s' is not a loopback address, trusted "
                    "domain, or trusted callback URI. %s" % (hostname, " ".join(next_steps))
                ),
                normalized_uri=normalized_redirect_uri,
                callback_origin=callback_origin,
                next_steps=tuple(next_steps),
            )
            logger.warning(result.log_message)
            return result

        if is_loopback and parsed.port is not None and (parsed.port < 1 or parsed.port > 65535):
            result = RedirectUriValidationResult(
                is_valid=False,
                summary=(f"Invalid redirect_uri '{normalized_redirect_uri}'. Loopback callback ports must be between 1 and 65535."),
                log_message=(f"OAuth2 redirect_uri rejected: invalid loopback port {parsed.port} (redirect_uri={redirect_uri!r})"),
                normalized_uri=normalized_redirect_uri,
                callback_origin=callback_origin,
            )
            logger.warning(result.log_message)
            return result

        if settings.debug_mode:
            logger.debug(f"OAuth2 redirect_uri validated: {redirect_uri}")

        return RedirectUriValidationResult(
            is_valid=True,
            normalized_uri=normalized_redirect_uri,
            callback_origin=callback_origin,
        )

    except Exception as exc:
        result = RedirectUriValidationResult(
            is_valid=False,
            summary="Invalid redirect_uri. Ragtime could not validate the callback URL.",
            log_message=f"OAuth2 redirect_uri validation error: {exc}",
        )
        logger.warning(result.log_message)
        return result


# =============================================================================
# Request/Response Models
# =============================================================================


class LoginRequest(BaseModel):
    """Login request body."""

    username: Optional[str] = Field(
        None,
        min_length=1,
        description="Username (uid, email/UPN, sAMAccountName, or local admin)",
    )
    password: Optional[str] = Field(None, min_length=1, description="Password")
    mfa_challenge_token: Optional[str] = Field(None, description="Scoped pending-MFA token returned after primary auth")
    totp_code: Optional[str] = Field(None, min_length=1, max_length=64, description="TOTP or recovery code")
    remember_device: bool = Field(False, description="Remember this browser for MFA for the configured duration")


class LoginResponse(BaseModel):
    """Login response with user info."""

    success: bool
    user_id: Optional[str] = None
    username: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    role: str = "user"
    error: Optional[str] = None
    mfa_required: bool = False
    mfa_enrollment_required: bool = False
    mfa_challenge_token: Optional[str] = None
    mfa_methods: list[str] = Field(default_factory=list, description="Allowed MFA methods the user has enrolled, ordered webauthn-first")
    mfa_enroll_methods: list[str] = Field(default_factory=list, description="MFA methods the user is allowed to enroll")
    mfa_preferred_method: Optional[str] = Field(default=None, description="Resolved MFA method to present first in the challenge UI")


class UserResponse(BaseModel):
    """Current user info response."""

    id: str
    username: str
    display_name: Optional[str]
    email: Optional[str]
    role: str
    auth_provider: str
    theme_pack: Optional[str] = None
    role_manually_set: bool = False
    source_provider: Optional[str] = None
    source_synced_at: Optional[datetime] = None
    source_expires_at: Optional[datetime] = None
    cached_groups: list[str] = Field(default_factory=list)
    manual_group_ids: list[str] = Field(default_factory=list)
    ldap_group_ids: list[str] = Field(default_factory=list)
    local_group_ids: list[str] = Field(
        default_factory=list,
        description="Deprecated alias for manual_group_ids.",
    )
    mfa_enabled: bool = False
    mfa_required: bool = False
    recovery_codes_remaining: int = 0


class UserListResponse(BaseModel):
    """Paginated user list response."""

    users: list[UserResponse]
    total: int
    skip: int
    take: int


class UserDirectoryEntryResponse(BaseModel):
    """Minimal user info returned to all authenticated users for member-picker use."""

    id: str
    username: str
    display_name: Optional[str] = None


class UserDirectoryResponse(BaseModel):
    """Response for the user directory endpoint."""

    users: list[UserDirectoryEntryResponse]


class UpdateUserRoleRequest(BaseModel):
    """User role update request payload."""

    role: Literal["user", "admin"] = Field(
        ...,
        description="New role value ('user' or 'admin').",
    )


class ResetUserRoleResponse(BaseModel):
    """Reset role override response payload."""

    success: bool
    role: Literal["user", "admin"]
    role_manually_set: bool


class AuthProviderConfigResponse(BaseModel):
    """Provider-neutral authentication policy response."""

    local_users_enabled: bool
    ldap_lazy_sync_enabled: bool
    manual_role_override_wins: bool
    cache_ttl_minutes: int
    totp_policy: Literal["optional", "required_all", "required_admins_groups"] = "optional"
    totp_required_group_ids: list[str] = Field(default_factory=list)
    totp_remember_device_days: int = 30
    mfa_allowed_methods: list[str] = Field(default=["totp"], description="Allowed MFA methods for the instance")
    mfa_default_method: Optional[str] = Field(None, description="Default MFA method for the instance when the user has no preference")


class UpdateAuthProviderConfigRequest(BaseModel):
    """Update provider-neutral authentication policy flags."""

    local_users_enabled: Optional[bool] = None
    ldap_lazy_sync_enabled: Optional[bool] = None
    manual_role_override_wins: Optional[bool] = None
    cache_ttl_minutes: Optional[int] = Field(None, ge=1, le=10080)
    totp_policy: Optional[Literal["optional", "required_all", "required_admins_groups"]] = None
    totp_required_group_ids: Optional[list[str]] = None
    totp_remember_device_days: Optional[int] = Field(None, ge=1, le=365)
    mfa_allowed_methods: Optional[list[str]] = Field(None, description="Allowed MFA methods; must be a non-empty subset of ['totp', 'webauthn']")
    mfa_default_method: Optional[str] = Field(None, description="Default MFA method; must be null or one of the effective allowed methods")


class MfaEnrollStartRequest(BaseModel):
    mfa_challenge_token: Optional[str] = Field(None, description="Pending MFA enrollment token from login")


class MfaEnrollStartResponse(BaseModel):
    secret: str
    otpauth_uri: str
    enrollment_token: str


class MfaEnrollCompleteRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=64)
    enrollment_token: str = Field(..., description="Scoped TOTP enrollment token returned by /mfa/enroll/start")
    mfa_challenge_token: Optional[str] = None
    remember_device: bool = False


class MfaEnrollCompleteResponse(BaseModel):
    success: bool
    recovery_codes: list[str] = Field(default_factory=list)
    user: Optional[UserResponse] = None


class MfaVerifyRequest(BaseModel):
    mfa_challenge_token: str
    code: str = Field(..., min_length=1, max_length=64)
    remember_device: bool = False


class MfaStepUpRequest(BaseModel):
    verification_code: str = Field(..., min_length=1, max_length=64, description="Current TOTP code or a recovery code proving control of the account")


class TotpRotateCompleteRequest(BaseModel):
    enrollment_token: str = Field(..., description="Rotation enrollment token returned by /mfa/totp/rotate/start")
    code: str = Field(..., min_length=1, max_length=64, description="Code from the newly configured authenticator app")


class RecoveryCodesResponse(BaseModel):
    recovery_codes: list[str] = Field(default_factory=list)


class MfaPreferredMethodRequest(BaseModel):
    method: Optional[str] = Field(None, description="Preferred MFA method, or null to inherit the instance default")


class MfaStatusResponse(BaseModel):
    enabled: bool
    required: bool
    recovery_codes_remaining: int
    methods_enrolled: list[str] = Field(default_factory=list, description="All MFA methods the user has enrolled, independent of current policy")
    allowed_methods: list[str] = Field(default=["totp"], description="MFA methods currently allowed by the provider policy")
    webauthn_credential_count: int = Field(default=0, description="Number of WebAuthn credentials registered for the user")
    preferred_method: Optional[str] = Field(None, description="User's preferred MFA method if it is currently enrolled")
    default_method: Optional[str] = Field(None, description="Instance default MFA method")


class WebauthnRegisterStartRequest(BaseModel):
    mfa_challenge_token: Optional[str] = Field(None, description="Pending MFA enrollment token from login")


class WebauthnRegisterStartResponse(BaseModel):
    options: dict[str, Any] = Field(default_factory=dict, description="WebAuthn PublicKeyCredentialCreationOptionsJSON")
    registration_token: str = Field(..., description="Short-lived signed challenge token for this registration")


class WebauthnRegisterCompleteRequest(BaseModel):
    registration_token: str = Field(..., description="Signed challenge token from /auth/mfa/webauthn/register/start")
    credential: dict[str, Any] = Field(..., description="WebAuthn credential response from the authenticator")
    name: Optional[str] = Field(None, description="Human-readable name for the new passkey")
    mfa_challenge_token: Optional[str] = Field(None, description="Pending MFA enrollment token from login")
    remember_device: bool = Field(False, description="Trust this browser after enrollment")


class WebauthnRegisterCompleteResponse(BaseModel):
    success: bool
    credential_id: str
    name: str
    recovery_codes: Optional[list[str]] = Field(None, description="One-time recovery codes when this was the user's first MFA factor")


class WebauthnAuthenticateStartRequest(BaseModel):
    mfa_challenge_token: str = Field(..., description="Pending MFA challenge token from primary authentication")


class WebauthnAuthenticateStartResponse(BaseModel):
    options: dict[str, Any] = Field(default_factory=dict, description="WebAuthn PublicKeyCredentialRequestOptionsJSON")
    authentication_token: str = Field(..., description="Short-lived signed challenge token for this authentication")


class WebauthnAuthenticateCompleteRequest(BaseModel):
    mfa_challenge_token: str = Field(..., description="Pending MFA challenge token from primary authentication")
    authentication_token: str = Field(..., description="Signed challenge token from /auth/mfa/webauthn/authenticate/start")
    credential: dict[str, Any] = Field(..., description="WebAuthn assertion response from the authenticator")
    remember_device: bool = Field(False, description="Trust this browser after authentication")


class WebauthnCredentialResponse(BaseModel):
    id: str
    name: str
    created_at: str
    last_used_at: Optional[str] = None
    transports: list[str] = Field(default_factory=list)


class WebauthnCredentialListResponse(BaseModel):
    credentials: list[WebauthnCredentialResponse]


class WebauthnCredentialRenameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="New display name for the credential")


class LocalUserCreateRequest(BaseModel):
    """Create an internal managed user."""

    username: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=8, max_length=1024)
    display_name: Optional[str] = Field(None, max_length=255)
    email: Optional[str] = Field(None, max_length=255)
    role: Literal["user", "admin"] = "user"


class LocalUserUpdateRequest(BaseModel):
    """Update an internal managed user."""

    password: Optional[str] = Field(default=None, min_length=8, max_length=1024)
    display_name: Optional[str] = Field(default=None, max_length=255)
    email: Optional[str] = Field(default=None, max_length=255)
    role: Optional[Literal["user", "admin"]] = None


class AuthGroupResponse(BaseModel):
    """Auth group response."""

    id: str
    key: str
    display_name: str
    description: str
    provider: str
    source_id: Optional[str] = None
    source_dn: Optional[str] = None
    role: Optional[str] = None
    member_count: int = 0
    manual_member_count: int = 0
    ldap_member_count: int = 0
    member_previews: list["AuthGroupMemberPreview"] = Field(default_factory=list)
    member_labels: list[str] = Field(default_factory=list)
    is_logon_group: bool = False


class AuthGroupMemberPreview(BaseModel):
    """Member identity preview for auth group hover popovers."""

    username: str
    display_name: str


class AuthGroupListResponse(BaseModel):
    groups: list[AuthGroupResponse]


class AuthGroupUpsertRequest(BaseModel):
    """Create or update a local auth group."""

    key: Optional[str] = Field(None, max_length=255)
    display_name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=1000)
    role: Optional[Literal["user", "admin"]] = None
    is_logon_group: bool = False


class SetUserGroupsRequest(BaseModel):
    """Replace a user's manual internal group memberships."""

    group_ids: list[str] = Field(default_factory=list)


class LdapUserSearchRequest(BaseModel):
    """Search for one LDAP user using configured bind credentials."""

    username: str = Field(..., min_length=1, max_length=255)


class LdapUserProfileResponse(BaseModel):
    username: str
    source_dn: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    role: str
    groups: list[str] = Field(default_factory=list)


class LdapUserTypeaheadRequest(BaseModel):
    """Typeahead lookup for LDAP users using fallback uid/email attributes."""

    query: str = Field(..., min_length=1, max_length=255)
    limit: int = Field(8, ge=1, le=25)


class LdapUserTypeaheadResponse(BaseModel):
    users: list[LdapUserProfileResponse] = Field(default_factory=list)


class LdapUserImportResponse(BaseModel):
    user: UserResponse


class LdapConfigRequest(BaseModel):
    """LDAP configuration update request."""

    server_url: Optional[str] = Field(None, description="LDAP server URL (ldap://host:389 or ldaps://host:636)")
    bind_dn: Optional[str] = Field(None, description="Bind DN or bind username for service account")
    bind_password: Optional[str] = Field(None, description="Bind password (leave empty to keep existing)")
    allow_self_signed: Optional[bool] = Field(None, description="Allow self-signed SSL certificates")
    user_search_base: Optional[str] = Field(None, description="User search base DN (auto-discovered if empty)")
    user_search_filter: Optional[str] = Field(None, description="User search filter (use {username} placeholder)")
    admin_group_dns: Optional[list[str]] = Field(None, description="Admin group DNs")
    user_group_dns: Optional[list[str]] = Field(None, description="User group DNs that are allowed to log in")


class LdapConfigResponse(BaseModel):
    """LDAP configuration response."""

    server_url: str
    bind_dn: str
    allow_self_signed: bool
    base_dn: str
    user_search_base: str
    user_search_filter: str
    admin_group_dns: list[str]
    user_group_dns: list[str]
    discovered_ous: list[str]
    discovered_groups: list[dict]


class LdapDiscoverRequest(BaseModel):
    """Request to discover LDAP structure."""

    server_url: str = Field(..., description="LDAP server URL")
    bind_dn: str = Field(..., description="Bind DN or bind username")
    bind_password: str = Field(..., description="Bind password")
    allow_self_signed: bool = Field(False, description="Allow self-signed SSL certificates")


class LdapDiscoverResponse(BaseModel):
    """LDAP discovery result."""

    success: bool
    base_dn: Optional[str] = None
    user_ous: list[str] = []
    groups: list[dict] = []
    error: Optional[str] = None


class LdapBindDnLookupRequest(BaseModel):
    """Request to look up bind DN from username."""

    server_url: str = Field(..., description="LDAP server URL (ldap:// or ldaps://)")
    username: str = Field(..., description="Username (uid, email/UPN, cn, or sAMAccountName)")
    password: str = Field(..., description="Password")


class LdapBindDnLookupResponse(BaseModel):
    """Bind DN lookup result."""

    success: bool
    bind_dn: Optional[str] = None
    display_name: Optional[str] = None
    error: Optional[str] = None


class AuthMethodStatus(BaseModel):
    """Status of an individual authentication method for login UI display."""

    key: str = Field(..., description="Stable auth method key (e.g. ldap, local, oidc)")
    label: str = Field(..., description="Human-readable auth method label")
    configured: bool = Field(..., description="Whether this auth method is configured")
    available: bool = Field(..., description="Whether the method is currently available")
    status: str = Field(
        ...,
        description="Availability status: available, unavailable, or not_configured",
    )
    detail: Optional[str] = Field(None, description="Short operator-facing detail for login status UI")


class AuthStatusResponse(BaseModel):
    """Authentication status response."""

    authenticated: bool
    ldap_configured: bool
    local_admin_enabled: bool
    debug_mode: bool = False
    debug_username: Optional[str] = None
    debug_password: Optional[str] = None
    debug_totp_code: Optional[str] = None
    cookie_warning: Optional[str] = None  # Warning about cookie/protocol mismatch
    # Security status for UI banner
    api_key_configured: bool = False
    session_cookie_secure: bool = False
    allowed_origins_open: bool = False  # True if ALLOWED_ORIGINS=*
    # True when RUNTIME_AUTH_TOKEN is missing, a known generic default, or
    # resolved via the deprecated legacy variable bridge (migration needed).
    runtime_auth_token_warning: bool = False
    auth_methods: list[AuthMethodStatus] = Field(default_factory=list)
    server_name: str = Field(
        default=DEFAULT_SERVER_NAME,
        description="Configured server branding name",
    )
    default_theme_pack: str = Field(
        default="default",
        description="Instance-wide default app theme pack for users without a personal choice.",
    )
    authenticated_webgl_background_enabled: bool = Field(
        default=DEFAULT_AUTHENTICATED_WEBGL_BACKGROUND_ENABLED,
        description="If True, show the animated WebGL gradient behind authenticated app pages.",
    )
    chat_compaction_threshold_percent: int = Field(
        default=DEFAULT_CHAT_COMPACTION_THRESHOLD_PERCENT,
        description="Show the chat compact button once effective conversation context usage reaches this percentage.",
    )
    chat_auto_compaction_threshold_percent: int = Field(
        default=DEFAULT_CHAT_AUTO_COMPACTION_THRESHOLD_PERCENT,
        description="Automatically compact the conversation once effective context usage reaches this percentage. Set to 100 to disable auto-compaction.",
    )


async def _user_response(user: User) -> UserResponse:
    cached_groups = getattr(user, "cachedGroups", None)
    db = await get_db()
    memberships = await db.authgroupmembership.find_many(where={"userId": user.id})
    group_ids = [membership.groupId for membership in memberships]
    groups = await db.authgroup.find_many(where={"id": {"in": group_ids}}) if group_ids else []
    groups_by_id = {group.id: group for group in groups}
    manual_group_ids: list[str] = []
    ldap_group_ids: list[str] = []
    for membership in memberships:
        group = groups_by_id.get(membership.groupId)
        if not group:
            continue
        membership_provider = _auth_provider_value(membership.sourceProvider)
        group_provider = _auth_provider_value(group.provider)
        if membership_provider == "local_managed" and group_provider == "local_managed":
            manual_group_ids.append(membership.groupId)
        elif membership_provider == "ldap":
            ldap_group_ids.append(membership.groupId)
    mfa_enabled = await user_has_enabled_totp(user.id)
    mfa_required = await mfa_needed_for_user(user)
    recovery_codes_remaining = await db.usermfarecoverycode.count(where={"userId": user.id, "usedAt": None})

    return UserResponse(
        id=user.id,
        username=user.username,
        display_name=user.displayName,
        email=user.email,
        role=user.role,
        auth_provider=user.authProvider,
        theme_pack=canonicalize_theme_pack_id(getattr(user, "themePack", None)),
        role_manually_set=user.roleManuallySet,
        source_provider=getattr(user, "sourceProvider", None),
        source_synced_at=getattr(user, "sourceSyncedAt", None),
        source_expires_at=getattr(user, "sourceExpiresAt", None),
        cached_groups=cached_groups if isinstance(cached_groups, list) else [],
        manual_group_ids=manual_group_ids,
        ldap_group_ids=ldap_group_ids,
        local_group_ids=manual_group_ids,
        mfa_enabled=mfa_enabled,
        mfa_required=mfa_required,
        recovery_codes_remaining=recovery_codes_remaining,
    )


def _prefetched_group_ids_for_user(
    user_id: str,
    memberships_by_user_id: dict[str, list[Any]],
    groups_by_id: dict[str, Any],
) -> tuple[list[str], list[str]]:
    manual_group_ids: list[str] = []
    ldap_group_ids: list[str] = []
    for membership in memberships_by_user_id.get(user_id, []):
        group = groups_by_id.get(membership.groupId)
        if not group:
            continue
        membership_provider = _auth_provider_value(membership.sourceProvider)
        group_provider = _auth_provider_value(group.provider)
        if membership_provider == "local_managed" and group_provider == "local_managed":
            manual_group_ids.append(membership.groupId)
        elif membership_provider == "ldap":
            ldap_group_ids.append(membership.groupId)
    return manual_group_ids, ldap_group_ids


def _prefetched_user_requires_mfa(
    user: User,
    auth_config: Any,
    memberships_by_user_id: dict[str, list[Any]],
    totp_enabled_user_ids: set[str],
    webauthn_enabled_user_ids: set[str],
) -> bool:
    if user.id in totp_enabled_user_ids or user.id in webauthn_enabled_user_ids:
        return True

    policy = str(getattr(auth_config, "totp_policy", "optional") or "optional")
    if policy == "required_all":
        return True
    if policy != "required_admins_groups":
        return False
    if _auth_provider_value(getattr(user, "role", "")) == "admin":
        return True

    required_group_ids = set(getattr(auth_config, "totp_required_group_ids", []) or [])
    if not required_group_ids:
        return False

    return any(membership.groupId in required_group_ids for membership in memberships_by_user_id.get(user.id, []))


async def _bulk_user_responses(users: list[User]) -> list[UserResponse]:
    if not users:
        return []

    db = await get_db()
    auth_config = await get_auth_provider_config()
    user_ids = [user.id for user in users]

    memberships = await db.authgroupmembership.find_many(where={"userId": {"in": user_ids}})
    memberships_by_user_id: dict[str, list[Any]] = {}
    for membership in memberships:
        memberships_by_user_id.setdefault(membership.userId, []).append(membership)

    group_ids = list(dict.fromkeys(membership.groupId for membership in memberships))
    groups = await db.authgroup.find_many(where={"id": {"in": group_ids}}) if group_ids else []
    groups_by_id = {group.id: group for group in groups}

    enabled_totp_factors = await db.usermfafactor.find_many(where={"userId": {"in": user_ids}, "factorType": "totp", "enabled": True})
    totp_enabled_user_ids = {factor.userId for factor in enabled_totp_factors if getattr(factor, "enabled", False) and getattr(factor, "secretEncrypted", None)}

    webauthn_credentials = await db.userwebauthncredential.find_many(where={"userId": {"in": user_ids}})
    webauthn_enabled_user_ids = {credential.userId for credential in webauthn_credentials}

    unused_recovery_codes = await db.usermfarecoverycode.find_many(where={"userId": {"in": user_ids}, "usedAt": None})
    recovery_code_counts: dict[str, int] = {}
    for recovery_code in unused_recovery_codes:
        recovery_code_counts[recovery_code.userId] = recovery_code_counts.get(recovery_code.userId, 0) + 1

    responses: list[UserResponse] = []
    for user in users:
        cached_groups = getattr(user, "cachedGroups", None)
        manual_group_ids, ldap_group_ids = _prefetched_group_ids_for_user(user.id, memberships_by_user_id, groups_by_id)
        responses.append(
            UserResponse(
                id=user.id,
                username=user.username,
                display_name=user.displayName,
                email=user.email,
                role=user.role,
                auth_provider=user.authProvider,
                theme_pack=canonicalize_theme_pack_id(getattr(user, "themePack", None)),
                role_manually_set=user.roleManuallySet,
                source_provider=getattr(user, "sourceProvider", None),
                source_synced_at=getattr(user, "sourceSyncedAt", None),
                source_expires_at=getattr(user, "sourceExpiresAt", None),
                cached_groups=cached_groups if isinstance(cached_groups, list) else [],
                manual_group_ids=manual_group_ids,
                ldap_group_ids=ldap_group_ids,
                local_group_ids=manual_group_ids,
                mfa_enabled=user.id in totp_enabled_user_ids,
                mfa_required=_prefetched_user_requires_mfa(
                    user,
                    auth_config,
                    memberships_by_user_id,
                    totp_enabled_user_ids,
                    webauthn_enabled_user_ids,
                ),
                recovery_codes_remaining=recovery_code_counts.get(user.id, 0),
            )
        )
    return responses


def _auth_provider_config_response(config) -> AuthProviderConfigResponse:
    return AuthProviderConfigResponse(
        local_users_enabled=config.local_users_enabled,
        ldap_lazy_sync_enabled=config.ldap_lazy_sync_enabled,
        manual_role_override_wins=config.manual_role_override_wins,
        cache_ttl_minutes=config.cache_ttl_minutes,
        totp_policy=config.totp_policy,
        totp_required_group_ids=config.totp_required_group_ids,
        totp_remember_device_days=config.totp_remember_device_days,
        mfa_allowed_methods=config.mfa_allowed_methods,
        mfa_default_method=config.mfa_default_method,
    )


def _auth_provider_value(provider) -> str:
    return str(getattr(provider, "value", provider) or "")


def _role_from_form_value(role: str | None) -> UserRole | None:
    if role == "admin":
        return UserRole.admin
    if role == "user":
        return UserRole.user
    return None


def _normalize_dn_list(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        dn = str(value or "").strip()
        key = dn.casefold()
        if not dn or key in seen:
            continue
        seen.add(key)
        normalized.append(dn)
    return normalized


async def _sync_ldap_auth_group_assignments(*, admin_group_dns: list[str], user_group_dns: list[str]) -> None:
    db = await get_db()
    ldap_groups = await db.authgroup.find_many(where={"provider": AuthProvider.ldap})
    admin_dns = {dn.casefold() for dn in admin_group_dns}
    user_dns = {dn.casefold() for dn in user_group_dns}
    for group in ldap_groups:
        source_dn = (group.sourceDn or "").strip().casefold()
        await db.authgroup.update(
            where={"id": group.id},
            data={
                "role": UserRole.admin if source_dn in admin_dns else None,
                "isLogonGroup": source_dn in user_dns,
            },
        )


async def _set_ldap_group_assignment(group, *, role: UserRole | None, is_logon_group: bool) -> None:
    if _auth_provider_value(group.provider) != "ldap" or not group.sourceDn:
        return

    db = await get_db()
    config = await db.ldapconfig.find_first()
    if not config:
        return

    source_dn = group.sourceDn.strip()
    source_key = source_dn.casefold()
    admin_group_dns = [dn for dn in _normalize_dn_list(getattr(config, "adminGroupDns", [])) if dn.casefold() != source_key]
    user_group_dns = [dn for dn in _normalize_dn_list(getattr(config, "userGroupDns", [])) if dn.casefold() != source_key]
    if role == UserRole.admin:
        admin_group_dns.append(source_dn)
    if is_logon_group:
        user_group_dns.append(source_dn)

    await db.ldapconfig.update(
        where={"id": config.id},
        data={
            "adminGroupDns": admin_group_dns,
            "userGroupDns": user_group_dns,
        },
    )


async def _auth_group_response(group) -> AuthGroupResponse:
    db = await get_db()
    memberships = await db.authgroupmembership.find_many(where={"groupId": group.id})
    member_count = len(memberships)
    manual_member_count = sum(1 for membership in memberships if _auth_provider_value(membership.sourceProvider) == "local_managed")
    ldap_member_count = sum(1 for membership in memberships if _auth_provider_value(membership.sourceProvider) == "ldap")
    user_ids = list({membership.userId for membership in memberships})
    users = await db.user.find_many(where={"id": {"in": user_ids}}) if user_ids else []
    users_by_id = {user.id: user for user in users}
    member_labels: list[str] = []
    member_previews: list[AuthGroupMemberPreview] = []
    seen_labels: set[str] = set()
    seen_usernames: set[str] = set()
    for membership in memberships:
        user = users_by_id.get(membership.userId)
        if not user:
            continue
        username = (user.username or "").strip()
        if username:
            normalized_username = username.casefold()
            if normalized_username not in seen_usernames:
                seen_usernames.add(normalized_username)
                member_previews.append(
                    AuthGroupMemberPreview(
                        username=username,
                        display_name=(user.displayName or username).strip() or username,
                    )
                )
        label = (user.displayName or user.username or "").strip()
        if not label:
            continue
        normalized = label.casefold()
        if normalized in seen_labels:
            continue
        seen_labels.add(normalized)
        member_labels.append(label)

    member_labels.sort(key=str.casefold)
    member_previews.sort(
        key=lambda preview: (
            preview.display_name.casefold(),
            preview.username.casefold(),
        )
    )

    return AuthGroupResponse(
        id=group.id,
        key=group.key,
        display_name=group.displayName,
        description=group.description,
        provider=group.provider,
        source_id=group.sourceId,
        source_dn=group.sourceDn,
        role=group.role,
        member_count=member_count,
        manual_member_count=manual_member_count,
        ldap_member_count=ldap_member_count,
        member_previews=member_previews,
        member_labels=member_labels,
        is_logon_group=bool(getattr(group, "isLogonGroup", False)),
    )


# =============================================================================
# Public Endpoints (No Auth Required)
# =============================================================================


def _detect_cookie_mismatch(request: Request) -> Optional[str]:
    """Detect cookie secure flag vs protocol mismatch."""
    # Check if request came over HTTPS (directly or via proxy)
    # Check multiple headers that proxies might set
    is_https = (
        request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto", "").lower() == "https"
        or request.headers.get("x-forwarded-ssl", "").lower() == "on"
        or request.headers.get("x-scheme", "").lower() == "https"
    )

    if settings.session_cookie_secure and not is_https:
        return (
            "Security misconfiguration: SESSION_COOKIE_SECURE=true but you are "
            "connecting over HTTP. Cookies will not be sent, causing auth to fail silently. "
            "Either set SESSION_COOKIE_SECURE=false or access via HTTPS."
        )

    if not settings.session_cookie_secure and is_https:
        return (
            "Security notice: You are connecting over HTTPS but SESSION_COOKIE_SECURE=false. Consider setting SESSION_COOKIE_SECURE=true for better security."
        )

    return None


def _request_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


async def _user_from_pending_mfa_token(token: str, *, purpose: Literal["challenge", "enroll"]) -> User:
    claims = decode_pending_mfa_token(token, expected_purpose=purpose)
    if claims is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired MFA challenge")
    db = await get_db()
    user = await db.user.find_unique(where={"id": claims.user_id})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


async def _set_trusted_device_if_requested(
    response: Response,
    request: Request,
    *,
    user_id: str,
    remember_device: bool,
) -> None:
    if not remember_device:
        return
    auth_config = await get_auth_provider_config()
    token, _expires_at = await create_trusted_device(
        user_id=user_id,
        user_agent=request.headers.get("User-Agent"),
        ip_address=_request_ip(request),
        days=auth_config.totp_remember_device_days,
    )
    response.set_cookie(
        key=MFA_TRUST_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite=settings.session_cookie_samesite,
        max_age=auth_config.totp_remember_device_days * 86400,
        path="/",
    )


async def _issue_login_session(
    response: Response,
    request: Request,
    *,
    user_id: str,
    username: str,
    role: str,
    mfa_verified: bool,
    auth_methods: list[str],
) -> str:
    return await issue_authenticated_session(
        response,
        user_id=user_id,
        username=username,
        role=role,
        user_agent=request.headers.get("User-Agent"),
        ip_address=_request_ip(request),
        mfa_verified=mfa_verified,
        auth_methods=auth_methods,
    )


def _webauthn_credential_response(cred: Any) -> WebauthnCredentialResponse:
    created_at = getattr(cred, "createdAt", None)
    last_used_at = getattr(cred, "lastUsedAt", None)
    return WebauthnCredentialResponse(
        id=getattr(cred, "id", ""),
        name=getattr(cred, "name", ""),
        created_at=created_at.isoformat() if created_at else "",
        last_used_at=last_used_at.isoformat() if last_used_at else None,
        transports=list(getattr(cred, "transports", []) or []),
    )


async def _generate_and_store_recovery_codes(user_id: str) -> list[str]:
    codes = [generate_recovery_code() for _ in range(RECOVERY_CODE_COUNT)]
    db = await get_db()
    await db.usermfarecoverycode.delete_many(where={"userId": user_id})
    for code in codes:
        await db.usermfarecoverycode.create(
            data={
                "userId": user_id,
                "codeHash": hash_recovery_code(code),
            }
        )
    return codes


async def _login_response_for_authenticated_primary(
    *,
    request: Request,
    response: Response,
    result: Any,
    trusted_device_token: str | None,
) -> LoginResponse:
    if not result.user_id or not result.username:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication succeeded but user data is missing",
        )

    db = await get_db()
    user = await db.user.find_unique(where={"id": result.user_id})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    if await mfa_needed_for_user(user):
        allowed_methods = await get_allowed_mfa_methods()
        enrolled_allowed = await user_allowed_enrolled_methods(user.id)

        if enrolled_allowed and await trusted_device_satisfies_mfa(user.id, trusted_device_token):
            await _issue_login_session(
                response,
                request,
                user_id=result.user_id,
                username=result.username,
                role=result.role,
                mfa_verified=True,
                auth_methods=["password", "mfa_trust"],
            )
        elif enrolled_allowed:
            return LoginResponse(
                success=False,
                mfa_required=True,
                mfa_methods=enrolled_allowed,
                mfa_preferred_method=await resolve_preferred_mfa_method(user, enrolled_allowed),
                mfa_challenge_token=create_pending_mfa_token(
                    user_id=result.user_id,
                    username=result.username,
                    role=result.role,
                    purpose="challenge",
                ),
            )
        else:
            return LoginResponse(
                success=False,
                mfa_enrollment_required=True,
                mfa_enroll_methods=allowed_methods,
                mfa_challenge_token=create_pending_mfa_token(
                    user_id=result.user_id,
                    username=result.username,
                    role=result.role,
                    purpose="enroll",
                ),
            )
    else:
        await _issue_login_session(
            response,
            request,
            user_id=result.user_id,
            username=result.username,
            role=result.role,
            mfa_verified=False,
            auth_methods=["password"],
        )

    logger.info(f"User '{result.username}' logged in successfully (role: {result.role})")
    return LoginResponse(
        success=True,
        user_id=result.user_id,
        username=result.username,
        display_name=result.display_name,
        email=result.email,
        role=result.role,
    )


async def _get_debug_totp_code() -> Optional[str]:
    """Compute the local admin's current TOTP code when running in debug mode.

    This is intentionally defensive: the auth status endpoint must never fail
    because the development convenience helper cannot compute a code.
    The TOTP secret itself is never returned; only the current 6-digit code.
    """
    if not settings.debug_mode:
        return None
    if not settings.local_admin_user or not settings.local_admin_password:
        return None
    try:
        db = await get_db()
        user = await db.user.find_unique(where={"username": f"local:{settings.local_admin_user}"})
        if not user:
            return None
        factor = await get_enabled_totp_factor(user.id)
        if not factor or not getattr(factor, "enabled", False):
            return None
        secret_encrypted = getattr(factor, "secretEncrypted", None) or getattr(factor, "secret_encrypted", None)
        if not secret_encrypted:
            return None
        secret = decrypt_secret(secret_encrypted)
        if not secret:
            return None
        return generate_totp_code(secret)
    except Exception as exc:
        logger.debug("Failed to compute debug TOTP code: %s", exc)
        return None


@router.get("/status", response_model=AuthStatusResponse)
async def get_auth_status(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """Get authentication system status.

    Unauthenticated callers receive only the fields needed to render the
    login page (auth methods, server branding, cookie/protocol warning,
    and debug creds when DEBUG_MODE is explicitly enabled). Operator
    posture flags such as api_key_configured, session_cookie_secure,
    allowed_origins_open, and debug_mode are only returned to
    authenticated users to avoid public reconnaissance signals.
    """
    cookie_warning = _detect_cookie_mismatch(request)
    auth_methods = [AuthMethodStatus(**status) for status in await build_auth_method_statuses()]
    ldap_configured = any(method.key == "ldap" and method.configured for method in auth_methods)
    server_name = DEFAULT_SERVER_NAME
    default_theme_pack = "default"
    authenticated_webgl_background_enabled = DEFAULT_AUTHENTICATED_WEBGL_BACKGROUND_ENABLED
    chat_compaction_threshold_percent = DEFAULT_CHAT_COMPACTION_THRESHOLD_PERCENT
    chat_auto_compaction_threshold_percent = DEFAULT_CHAT_AUTO_COMPACTION_THRESHOLD_PERCENT

    try:
        # Invalidate the settings cache before reading to ensure fresh values.
        # The cache is per-worker, so a setting saved on one worker may be stale
        # on another in multi-worker deployments.
        invalidate_settings_cache()
        app_settings = await get_app_settings()
        configured_server_name = str(app_settings.get("server_name") or "").strip()
        if configured_server_name:
            server_name = configured_server_name
        default_theme_pack = str(app_settings.get("default_theme_pack") or "default").strip() or "default"
        authenticated_webgl_background_enabled = bool(
            app_settings.get(
                "authenticated_webgl_background_enabled",
                DEFAULT_AUTHENTICATED_WEBGL_BACKGROUND_ENABLED,
            )
        )
        chat_compaction_threshold_percent = max(
            1,
            min(
                100,
                int(
                    app_settings.get(
                        "chat_compaction_threshold_percent",
                        DEFAULT_CHAT_COMPACTION_THRESHOLD_PERCENT,
                    )
                ),
            ),
        )
        chat_auto_compaction_threshold_percent = max(
            1,
            min(
                100,
                int(
                    app_settings.get(
                        "chat_auto_compaction_threshold_percent",
                        DEFAULT_CHAT_AUTO_COMPACTION_THRESHOLD_PERCENT,
                    )
                ),
            ),
        )
    except Exception as exc:
        logger.debug("Failed to load server branding for auth status: %s", exc)

    is_authenticated = current_user is not None

    return AuthStatusResponse(
        authenticated=is_authenticated,
        ldap_configured=ldap_configured,
        local_admin_enabled=bool(settings.local_admin_password),
        debug_mode=settings.debug_mode if is_authenticated else False,
        debug_username=settings.local_admin_user if settings.debug_mode else None,
        # Debug mode is development-only; frontend uses this for local login autofill.
        debug_password=settings.local_admin_password if settings.debug_mode else None,
        debug_totp_code=await _get_debug_totp_code(),
        cookie_warning=cookie_warning,
        api_key_configured=bool(settings.api_key) if is_authenticated else False,
        session_cookie_secure=(settings.session_cookie_secure if is_authenticated else False),
        allowed_origins_open=((settings.allowed_origins == "*") if is_authenticated else False),
        runtime_auth_token_warning=(settings.runtime_auth_token_warning() if is_authenticated else False),
        auth_methods=auth_methods,
        server_name=server_name,
        default_theme_pack=default_theme_pack,
        authenticated_webgl_background_enabled=authenticated_webgl_background_enabled,
        chat_compaction_threshold_percent=chat_compaction_threshold_percent,
        chat_auto_compaction_threshold_percent=chat_auto_compaction_threshold_percent,
    )


@router.post("/login", response_model=LoginResponse)
@limiter.limit(LOGIN_RATE_LIMIT)
async def login(
    request: Request,
    response: Response,
    body: LoginRequest,
):
    """
    Authenticate user and create session.

    Tries LDAP first (if enabled), then falls back to local admin.
    Sets httpOnly cookie with JWT token.

    Rate limited to 5 attempts per minute per IP to prevent brute-force attacks.
    """
    if body.mfa_challenge_token:
        if not body.totp_code:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="TOTP or recovery code is required")
        user = await _user_from_pending_mfa_token(body.mfa_challenge_token, purpose="challenge")
        verified, method = await verify_user_mfa_code(user, body.totp_code)
        if not verified:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid MFA code")
        await _set_trusted_device_if_requested(
            response,
            request,
            user_id=user.id,
            remember_device=body.remember_device,
        )
        await _issue_login_session(
            response,
            request,
            user_id=user.id,
            username=user.username,
            role=str(user.role),
            mfa_verified=True,
            auth_methods=["password", method or "totp"],
        )
        return LoginResponse(
            success=True,
            user_id=user.id,
            username=user.username,
            display_name=user.displayName,
            email=user.email,
            role=str(user.role),
        )

    if not body.username or not body.password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username and password are required")

    result = await authenticate(body.username, body.password)

    if not result.success:
        logger.warning(f"Login failed for user '{body.username}': {result.error}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=result.error or "Authentication failed",
        )

    return await _login_response_for_authenticated_primary(
        request=request,
        response=response,
        result=result,
        trusted_device_token=request.cookies.get(MFA_TRUST_COOKIE_NAME),
    )


@router.post("/logout")
async def logout(
    response: Response,
    token: Optional[str] = Depends(get_session_token),
):
    """Logout and invalidate session."""
    if token:
        await invalidate_session(token)

    # Clear cookie
    response.delete_cookie(
        key=settings.session_cookie_name,
        httponly=settings.session_cookie_httponly,
        secure=settings.session_cookie_secure,
        samesite=settings.session_cookie_samesite,
    )

    return {"success": True}


# =============================================================================
# OAuth2 Authorization Code Flow with PKCE (for VS Code MCP clients)
# =============================================================================


def _cleanup_expired_auth_codes():
    """Remove expired authorization codes and enforce storage limit."""
    now = time.time()
    expired = [code for code, data in _auth_codes.items() if data["expires"] < now]
    for code in expired:
        del _auth_codes[code]

    # Enforce max limit - evict oldest codes if over limit
    if len(_auth_codes) > MAX_AUTH_CODES:
        # Sort by expiry time and remove oldest
        sorted_codes = sorted(_auth_codes.items(), key=lambda x: x[1]["expires"])
        to_remove = len(_auth_codes) - MAX_AUTH_CODES
        for code, _ in sorted_codes[:to_remove]:
            del _auth_codes[code]
        logger.warning(f"Auth code storage limit reached, evicted {to_remove} codes")


def _verify_pkce(code_verifier: str, code_challenge: str) -> bool:
    """Verify PKCE code_verifier against stored code_challenge (S256 method)."""
    # S256: BASE64URL(SHA256(code_verifier)) == code_challenge
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    # Base64url encoding without padding
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return computed == code_challenge


# =============================================================================
# OAuth2 Token Endpoint (for MCP clients)
# =============================================================================


class OAuth2TokenRequest(BaseModel):
    """OAuth2 token request (Resource Owner Password Credentials grant)."""

    grant_type: str = Field(..., description="Grant type (must be 'password')")
    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=1, description="Password")
    scope: Optional[str] = Field(default=None, description="Requested scopes (optional)")


class OAuth2TokenResponse(BaseModel):
    """OAuth2 token response."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int  # Seconds until expiration
    scope: Optional[str] = None


class OAuth2ErrorResponse(BaseModel):
    """OAuth2 error response per RFC 6749."""

    error: str
    error_description: Optional[str] = None


@router.post(
    "/oauth2/token",
    response_model=OAuth2TokenResponse,
    responses={
        400: {"model": OAuth2ErrorResponse},
        401: {"model": OAuth2ErrorResponse},
    },
    tags=["OAuth2"],
)
@limiter.limit(LOGIN_RATE_LIMIT)
async def oauth2_token(
    request: Request,
    response: Response,
    grant_type: str = Form(..., description="Grant type ('password' or 'authorization_code')"),
    username: Optional[str] = Form(default=None, description="Username (for password grant)"),
    password: Optional[str] = Form(default=None, description="Password (for password grant)"),
    totp_code: Optional[str] = Form(default=None, description="TOTP or recovery code (required when MFA applies)"),
    remember_device: bool = Form(default=False, description="Remember this client browser for MFA"),
    code: Optional[str] = Form(default=None, description="Authorization code (for authorization_code grant)"),
    code_verifier: Optional[str] = Form(default=None, description="PKCE code verifier (for authorization_code grant)"),
    redirect_uri: Optional[str] = Form(default=None, description="Redirect URI (for authorization_code grant)"),
    client_id: Optional[str] = Form(default=None, description="Client ID (for authorization_code grant)"),
    scope: Optional[str] = Form(default=None, description="Requested scopes (optional)"),
):
    """
    OAuth2 Token Endpoint.

    Supports two grant types:
    - **password**: Resource Owner Password Credentials grant (direct username/password)
    - **authorization_code**: Authorization Code grant with PKCE (for browser-based flows)

    **Usage with MCP clients (Authorization Code flow):**
    1. Client redirects to /authorize with PKCE code_challenge
    2. User authenticates, gets redirected back with authorization code
    3. Client exchanges code for token at this endpoint with code_verifier

    **Usage with direct clients (Password flow):**
    1. POST to /auth/oauth2/token with grant_type=password, username, and password
    2. Receive an access_token

    Rate limited to prevent brute-force attacks.
    """

    # OAuth2 error responses must be top-level JSON per RFC 6749 Section 5.2;
    # FastAPI's HTTPException wraps the body in {"detail": ...}, which breaks
    # strict OAuth2 clients (e.g. Claude MCP), so emit JSONResponse directly.
    def _oauth_error(error: str, description: str, status_code: int = 400) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={"error": error, "error_description": description},
        )

    # Some MCP OAuth clients validate token payloads strictly and reject
    # `scope: null`; normalize to an OAuth-compatible string value.
    response_scope = (scope or "").strip()

    # Handle authorization_code grant (PKCE flow)
    if grant_type == "authorization_code":
        if not code or not code_verifier:
            return _oauth_error(
                "invalid_request",
                "code and code_verifier are required for authorization_code grant",
            )

        # Cleanup expired codes
        _cleanup_expired_auth_codes()

        # Look up the authorization code
        auth_data = _auth_codes.get(code)
        if not auth_data:
            return _oauth_error("invalid_grant", "Invalid or expired authorization code")

        # Verify PKCE
        if not _verify_pkce(code_verifier, auth_data["code_challenge"]):
            # Remove the code on PKCE failure (prevent replay)
            del _auth_codes[code]
            return _oauth_error("invalid_grant", "PKCE verification failed")

        # Verify client binding. OAuth auth codes are issued for one client and
        # redirect URI; a supplied mismatched value must not bypass that binding.
        if not client_id:
            return _oauth_error("invalid_request", "client_id is required for authorization_code grant")
        if client_id != auth_data["client_id"]:
            del _auth_codes[code]
            logger.warning(f"OAuth2 client_id mismatch: expected '{auth_data['client_id']}', got '{client_id}'")
            return _oauth_error("invalid_grant", "client_id mismatch")

        if redirect_uri and redirect_uri != auth_data["redirect_uri"]:
            del _auth_codes[code]
            return _oauth_error("invalid_grant", "redirect_uri mismatch")

        # Code is valid - consume it (one-time use)
        del _auth_codes[code]

        token = await _issue_login_session(
            response,
            request,
            user_id=auth_data["user_id"],
            username=auth_data["username"],
            role=auth_data["role"],
            mfa_verified=bool(auth_data.get("mfa_verified", False)),
            auth_methods=list(auth_data.get("auth_methods") or ["password"]),
        )

        logger.info(f"OAuth2 token issued for '{auth_data['username']}' via authorization_code grant")

        return OAuth2TokenResponse(
            access_token=token,
            token_type="Bearer",
            expires_in=settings.jwt_expire_hours * 3600,
            scope=response_scope,
        )

    # Handle password grant
    elif grant_type == "password":
        if not username or not password:
            return _oauth_error(
                "invalid_request",
                "username and password are required for password grant",
            )

        # Authenticate through the shared provider dispatch. This keeps OAuth2
        # clients on the same endpoint while allowing internal users, LDAP users,
        # and future providers to coexist behind authenticate().
        result = await authenticate(username, password)

        if not result.success:
            logger.warning(f"OAuth2 token request failed for '{username}': {result.error}")
            return _oauth_error(
                "invalid_grant",
                result.error or "Authentication failed",
                status_code=401,
            )

        if not result.user_id or not result.username:
            return _oauth_error(
                "server_error",
                "Authentication succeeded but user data is missing",
                status_code=500,
            )

        db = await get_db()
        user = await db.user.find_unique(where={"id": result.user_id})
        if not user:
            return _oauth_error("invalid_grant", "User not found", status_code=401)

        auth_methods = ["password"]
        mfa_verified = False
        if await mfa_needed_for_user(user):
            if await trusted_device_satisfies_mfa(user.id, request.cookies.get(MFA_TRUST_COOKIE_NAME)):
                auth_methods.append("mfa_trust")
                mfa_verified = True
            elif not await user_has_enabled_totp(user.id):
                return _oauth_error(
                    "mfa_enrollment_required",
                    "TOTP enrollment is required before this user can obtain a token",
                    status_code=401,
                )
            elif not totp_code:
                return _oauth_error(
                    "mfa_required",
                    "totp_code is required for this user",
                    status_code=401,
                )
            else:
                verified, method = await verify_user_mfa_code(user, totp_code)
                if not verified:
                    return _oauth_error("invalid_grant", "Invalid MFA code", status_code=401)
                auth_methods.append(method or "totp")
                mfa_verified = True
                await _set_trusted_device_if_requested(
                    response,
                    request,
                    user_id=user.id,
                    remember_device=remember_device,
                )

        token = await _issue_login_session(
            response,
            request,
            user_id=result.user_id,
            username=result.username,
            role=result.role,
            mfa_verified=mfa_verified,
            auth_methods=auth_methods,
        )

        logger.info(f"OAuth2 token issued for '{result.username}' via password grant")

        return OAuth2TokenResponse(
            access_token=token,
            token_type="Bearer",
            expires_in=settings.jwt_expire_hours * 3600,
            scope=response_scope,
        )

    else:
        return _oauth_error(
            "unsupported_grant_type",
            "Supported grant types: 'password', 'authorization_code'",
        )


# =============================================================================
# Authenticated Endpoints
# =============================================================================


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(user: User = Depends(get_current_user)):
    """Get current authenticated user info."""
    return await _user_response(user)


class UpdateMePreferencesRequest(BaseModel):
    """Self-service update of the current user's UI preferences."""

    theme_pack: Optional[str] = Field(
        default=None,
        max_length=32,
        description="App theme pack id, or null to inherit the global default.",
    )


def _normalize_theme_pack(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = canonicalize_theme_pack_id(value) or value.strip().lower()
    if not normalized:
        return None
    if not all(ch.isalnum() or ch in {"-", "_"} for ch in normalized):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid theme pack identifier",
        )
    return normalized


@router.patch("/me", response_model=UserResponse)
async def update_current_user_preferences(
    body: UpdateMePreferencesRequest,
    user: User = Depends(get_current_user),
):
    """Update the current user's own UI preferences (theme pack)."""
    theme_pack = _normalize_theme_pack(body.theme_pack)
    db = await get_db()
    updated = await db.user.update(where={"id": user.id}, data={"themePack": theme_pack})
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return await _user_response(updated)


async def _build_mfa_status(user: User) -> MfaStatusResponse:
    db = await get_db()
    methods_enrolled: list[str] = []
    if await user_has_enabled_webauthn(user.id):
        methods_enrolled.append("webauthn")
    if await user_has_enabled_totp(user.id):
        methods_enrolled.append("totp")
    preferred_method = str(getattr(user, "mfaPreferredMethod", None) or "").lower().strip() or None
    if preferred_method not in methods_enrolled:
        preferred_method = None
    auth_config = await get_auth_provider_config()
    return MfaStatusResponse(
        enabled=bool(methods_enrolled),
        required=await mfa_needed_for_user(user),
        recovery_codes_remaining=await db.usermfarecoverycode.count(where={"userId": user.id, "usedAt": None}),
        methods_enrolled=methods_enrolled,
        allowed_methods=await get_allowed_mfa_methods(),
        webauthn_credential_count=await db.userwebauthncredential.count(where={"userId": user.id}),
        preferred_method=preferred_method,
        default_method=auth_config.mfa_default_method,
    )


@router.get("/mfa/status", response_model=MfaStatusResponse)
async def get_mfa_status(user: User = Depends(get_current_user)):
    """Get the current user's MFA status."""
    return await _build_mfa_status(user)


@router.put("/mfa/preferred-method", response_model=MfaStatusResponse)
async def set_preferred_mfa_method(
    body: MfaPreferredMethodRequest,
    user: User = Depends(get_current_user),
):
    """Set or clear the current user's preferred MFA method."""
    if body.method is not None:
        enrolled_allowed = await user_allowed_enrolled_methods(user.id)
        if body.method not in enrolled_allowed:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Method is not enrolled")

    db = await get_db()
    updated = await db.user.update(where={"id": user.id}, data={"mfaPreferredMethod": body.method})
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return await _build_mfa_status(updated)


@router.post("/mfa/enroll/start", response_model=MfaEnrollStartResponse)
async def start_mfa_enrollment(
    body: MfaEnrollStartRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """Start TOTP enrollment for a logged-in user or a pending forced-enrollment login."""
    user = current_user
    if body.mfa_challenge_token:
        user = await _user_from_pending_mfa_token(body.mfa_challenge_token, purpose="enroll")
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    allowed_methods = await get_allowed_mfa_methods()
    if "totp" not in allowed_methods:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="TOTP enrollment is not allowed")
    try:
        setup = await begin_totp_enrollment(user)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return MfaEnrollStartResponse(**setup)


@router.post("/mfa/enroll/complete", response_model=MfaEnrollCompleteResponse)
@limiter.limit(LOGIN_RATE_LIMIT)
async def complete_mfa_enrollment(
    request: Request,
    response: Response,
    body: MfaEnrollCompleteRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """Confirm TOTP enrollment, return one-time recovery codes, and issue an MFA-complete session."""
    user = current_user
    if body.mfa_challenge_token:
        user = await _user_from_pending_mfa_token(body.mfa_challenge_token, purpose="enroll")
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    success, recovery_codes = await confirm_totp_enrollment(user, body.code, body.enrollment_token)
    if not success:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid TOTP code")

    await invalidate_all_sessions(user.id)
    await _set_trusted_device_if_requested(
        response,
        request,
        user_id=user.id,
        remember_device=body.remember_device,
    )
    await _issue_login_session(
        response,
        request,
        user_id=user.id,
        username=user.username,
        role=str(user.role),
        mfa_verified=True,
        auth_methods=["password", "totp"],
    )
    db = await get_db()
    refreshed = await db.user.find_unique(where={"id": user.id})
    return MfaEnrollCompleteResponse(
        success=True,
        recovery_codes=recovery_codes,
        user=await _user_response(refreshed or user),
    )


@router.post("/mfa/totp/rotate/start", response_model=MfaEnrollStartResponse)
@limiter.limit(LOGIN_RATE_LIMIT)
async def start_totp_rotation(
    request: Request,
    body: MfaStepUpRequest,
    user: User = Depends(get_current_user),
):
    """Begin replacing an already-enrolled authenticator after step-up verification.

    The old TOTP secret stays valid until the new one is confirmed via
    /mfa/totp/rotate/complete, so the account never loses MFA during rotation.
    """
    if "totp" not in await get_allowed_mfa_methods():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="TOTP enrollment is not allowed")
    if not await user_has_enabled_totp(user.id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No authenticator app is enrolled to replace")
    verified, _ = await verify_user_mfa_code(user, body.verification_code)
    if not verified:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Verification failed")
    setup = await begin_totp_enrollment(user, allow_replace=True)
    return MfaEnrollStartResponse(**setup)


@router.post("/mfa/totp/rotate/complete", response_model=MfaEnrollCompleteResponse)
@limiter.limit(LOGIN_RATE_LIMIT)
async def complete_totp_rotation(
    request: Request,
    body: TotpRotateCompleteRequest,
    user: User = Depends(get_current_user),
):
    """Confirm a rotated authenticator, swapping the stored TOTP secret in place.

    Unlike /mfa/enroll/complete this does not reissue or invalidate sessions; it is an
    in-session secret swap. Recovery codes are preserved (not regenerated) on rotation.
    """
    success, recovery_codes = await confirm_totp_enrollment(user, body.code, body.enrollment_token)
    if not success:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid TOTP code")
    return MfaEnrollCompleteResponse(
        success=True,
        recovery_codes=recovery_codes,
        user=await _user_response(user),
    )


@router.post("/mfa/recovery-codes/regenerate", response_model=RecoveryCodesResponse)
@limiter.limit(LOGIN_RATE_LIMIT)
async def regenerate_recovery_codes_route(
    request: Request,
    body: MfaStepUpRequest,
    user: User = Depends(get_current_user),
):
    """Issue a fresh set of recovery codes after step-up verification, invalidating old ones."""
    if not (await user_has_enabled_totp(user.id) or await user_has_enabled_webauthn(user.id)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No MFA is enrolled")
    verified, _ = await verify_user_mfa_code(user, body.verification_code)
    if not verified:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Verification failed")
    codes = await regenerate_recovery_codes(user.id)
    return RecoveryCodesResponse(recovery_codes=codes)


@router.post("/mfa/verify", response_model=LoginResponse)
@limiter.limit(LOGIN_RATE_LIMIT)
async def verify_mfa_challenge(
    request: Request,
    response: Response,
    body: MfaVerifyRequest,
):
    """Verify a pending MFA challenge and issue an authenticated app session."""
    user = await _user_from_pending_mfa_token(body.mfa_challenge_token, purpose="challenge")
    verified, method = await verify_user_mfa_code(user, body.code)
    if not verified:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid MFA code")
    await _set_trusted_device_if_requested(
        response,
        request,
        user_id=user.id,
        remember_device=body.remember_device,
    )
    await _issue_login_session(
        response,
        request,
        user_id=user.id,
        username=user.username,
        role=str(user.role),
        mfa_verified=True,
        auth_methods=["password", method or "totp"],
    )
    return LoginResponse(
        success=True,
        user_id=user.id,
        username=user.username,
        display_name=user.displayName,
        email=user.email,
        role=str(user.role),
    )


@router.post("/mfa/webauthn/register/start", response_model=WebauthnRegisterStartResponse)
async def start_webauthn_registration(
    request: Request,
    body: WebauthnRegisterStartRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """Begin registering a new WebAuthn credential for the current user or a pending enrollment."""
    user = current_user
    if body.mfa_challenge_token:
        user = await _user_from_pending_mfa_token(body.mfa_challenge_token, purpose="enroll")
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    allowed_methods = await get_allowed_mfa_methods()
    if WEBAUTHN_METHOD not in allowed_methods:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="WebAuthn enrollment is not allowed")

    try:
        options, registration_token = await begin_webauthn_registration(user, request)
    except WebauthnError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return WebauthnRegisterStartResponse(options=options, registration_token=registration_token)


@router.post("/mfa/webauthn/register/complete", response_model=WebauthnRegisterCompleteResponse)
@limiter.limit(LOGIN_RATE_LIMIT)
async def complete_webauthn_registration_route(
    request: Request,
    response: Response,
    body: WebauthnRegisterCompleteRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """Finish registering a WebAuthn credential and, when applicable, issue an MFA-complete session."""
    user = current_user
    if body.mfa_challenge_token:
        user = await _user_from_pending_mfa_token(body.mfa_challenge_token, purpose="enroll")
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    allowed_methods = await get_allowed_mfa_methods()
    if WEBAUTHN_METHOD not in allowed_methods:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="WebAuthn enrollment is not allowed")

    is_first_factor = not await user_has_enabled_totp(user.id) and not await user_has_enabled_webauthn(user.id)

    try:
        cred = await complete_webauthn_registration(
            user,
            request,
            body.registration_token,
            body.credential,
            body.name,
        )
    except WebauthnError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    recovery_codes: list[str] | None = None
    if is_first_factor:
        recovery_codes = await _generate_and_store_recovery_codes(user.id)

    if body.mfa_challenge_token:
        await invalidate_all_sessions(user.id)
        await _set_trusted_device_if_requested(
            response,
            request,
            user_id=user.id,
            remember_device=body.remember_device,
        )
        await _issue_login_session(
            response,
            request,
            user_id=user.id,
            username=user.username,
            role=str(user.role),
            mfa_verified=True,
            auth_methods=["password", "webauthn"],
        )

    return WebauthnRegisterCompleteResponse(
        success=True,
        credential_id=getattr(cred, "credentialId", ""),
        name=getattr(cred, "name", ""),
        recovery_codes=recovery_codes if recovery_codes else None,
    )


@router.post("/mfa/webauthn/authenticate/start", response_model=WebauthnAuthenticateStartResponse)
async def start_webauthn_authentication(
    request: Request,
    body: WebauthnAuthenticateStartRequest,
):
    """Begin WebAuthn authentication for a pending MFA challenge."""
    user = await _user_from_pending_mfa_token(body.mfa_challenge_token, purpose="challenge")

    allowed_methods = await get_allowed_mfa_methods()
    if WEBAUTHN_METHOD not in allowed_methods:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="WebAuthn authentication is not allowed")
    if not await user_has_enabled_webauthn(user.id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No WebAuthn credentials found")

    try:
        options, authentication_token = await begin_webauthn_authentication(user, request)
    except WebauthnError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return WebauthnAuthenticateStartResponse(options=options, authentication_token=authentication_token)


@router.post("/mfa/webauthn/authenticate/complete", response_model=LoginResponse)
@limiter.limit(LOGIN_RATE_LIMIT)
async def complete_webauthn_authentication_route(
    request: Request,
    response: Response,
    body: WebauthnAuthenticateCompleteRequest,
):
    """Verify a WebAuthn assertion and issue an authenticated app session."""
    user = await _user_from_pending_mfa_token(body.mfa_challenge_token, purpose="challenge")

    allowed_methods = await get_allowed_mfa_methods()
    if WEBAUTHN_METHOD not in allowed_methods:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="WebAuthn authentication is not allowed")

    try:
        await complete_webauthn_authentication(
            user,
            request,
            body.authentication_token,
            body.credential,
        )
    except WebauthnError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    await _set_trusted_device_if_requested(
        response,
        request,
        user_id=user.id,
        remember_device=body.remember_device,
    )
    await _issue_login_session(
        response,
        request,
        user_id=user.id,
        username=user.username,
        role=str(user.role),
        mfa_verified=True,
        auth_methods=["password", "webauthn"],
    )

    return LoginResponse(
        success=True,
        user_id=user.id,
        username=user.username,
        display_name=user.displayName,
        email=user.email,
        role=str(user.role),
    )


@router.get("/mfa/webauthn/credentials", response_model=WebauthnCredentialListResponse)
async def list_webauthn_credentials_route(user: User = Depends(get_current_user)):
    """List the current user's WebAuthn credentials."""
    creds = await list_webauthn_credentials(user.id)
    return WebauthnCredentialListResponse(credentials=[_webauthn_credential_response(c) for c in creds])


@router.patch("/mfa/webauthn/credentials/{credential_id}", response_model=WebauthnCredentialResponse)
async def rename_webauthn_credential_route(
    credential_id: str,
    body: WebauthnCredentialRenameRequest,
    user: User = Depends(get_current_user),
):
    """Rename a WebAuthn credential."""
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Credential name is required")
    cred = await rename_webauthn_credential(user.id, credential_id, name)
    if not cred:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Credential not found")
    return _webauthn_credential_response(cred)


@router.delete("/mfa/webauthn/credentials/{credential_id}")
async def delete_webauthn_credential_route(
    credential_id: str,
    user: User = Depends(get_current_user),
):
    """Delete a WebAuthn credential."""
    deleted = await delete_webauthn_credential(user.id, credential_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Credential not found")
    await invalidate_all_sessions(user.id)
    return {"success": True}


# =============================================================================
# Admin Endpoints (LDAP Configuration)
# =============================================================================


@router.post("/ldap/discover", response_model=LdapDiscoverResponse)
async def discover_ldap(
    body: LdapDiscoverRequest,
    _user: User = Depends(require_admin),
):
    """
    Discover LDAP structure (base DN, user OUs, groups).

    Used to populate dropdowns for LDAP configuration.
    """
    result = await discover_ldap_structure(
        server_url=body.server_url,
        bind_dn=body.bind_dn,
        bind_password=body.bind_password,
        allow_self_signed=body.allow_self_signed,
    )

    return LdapDiscoverResponse(
        success=result.success,
        base_dn=result.base_dn,
        user_ous=result.user_ous,
        groups=result.groups,
        error=result.error,
    )


@router.get("/ldap/discover", response_model=LdapDiscoverResponse)
async def discover_ldap_with_stored_credentials(
    _user: User = Depends(require_admin),
):
    """
    Discover LDAP structure using stored credentials from database.

    Useful for auto-populating dropdowns on settings page load without
    requiring the user to re-enter their bind password.
    """
    config = await get_ldap_config()

    if not config.serverUrl or not config.bindDn or not config.bindPassword:
        return LdapDiscoverResponse(
            success=False,
            error="LDAP not fully configured. Please provide server URL, bind DN, and password.",
        )

    # Decrypt bind password before using
    bind_password = decrypt_secret(config.bindPassword)

    result = await discover_ldap_structure(
        server_url=config.serverUrl,
        bind_dn=config.bindDn,
        bind_password=bind_password,
        allow_self_signed=config.allowSelfSigned,
    )

    return LdapDiscoverResponse(
        success=result.success,
        base_dn=result.base_dn,
        user_ous=result.user_ous,
        groups=result.groups,
        error=result.error,
    )


@router.post("/ldap/lookup-bind-dn", response_model=LdapBindDnLookupResponse)
async def lookup_ldap_bind_dn(
    body: LdapBindDnLookupRequest,
    _user: User = Depends(require_admin),
):
    """
    Look up the full DN for a bind account given just the username.

    Attempts to authenticate with the provided credentials and discover
    the full distinguished name. Useful for simplifying LDAP setup.
    """
    result = await lookup_bind_dn(
        server_url=body.server_url,
        username=body.username,
        password=body.password,
    )

    return LdapBindDnLookupResponse(
        success=result.success,
        bind_dn=result.bind_dn,
        display_name=result.display_name,
        error=result.error,
    )


@router.get("/ldap/config", response_model=LdapConfigResponse)
async def get_ldap_configuration(_user: User = Depends(require_admin)):
    """Get current LDAP configuration."""
    config = await get_ldap_config()

    return LdapConfigResponse(
        server_url=config.serverUrl,
        bind_dn=config.bindDn,
        allow_self_signed=config.allowSelfSigned,
        base_dn=config.baseDn,
        user_search_base=config.userSearchBase,
        user_search_filter=config.userSearchFilter,
        admin_group_dns=_normalize_dn_list(config.adminGroupDns),
        user_group_dns=_normalize_dn_list(config.userGroupDns),
        discovered_ous=(config.discoveredOus if isinstance(config.discoveredOus, list) else []),
        discovered_groups=(config.discoveredGroups if isinstance(config.discoveredGroups, list) else []),
    )


@router.put("/ldap/config", response_model=LdapConfigResponse)
async def update_ldap_configuration(
    body: LdapConfigRequest,
    _user: User = Depends(require_admin),
):
    """Update LDAP configuration."""
    db = await get_db()

    # Get existing config
    existing = await get_ldap_config()

    # Merge with existing values for optional fields
    server_url = body.server_url if body.server_url is not None else existing.serverUrl
    bind_dn = body.bind_dn if body.bind_dn is not None else existing.bindDn
    # Keep existing (encrypted) password if not provided, otherwise encrypt new password
    if body.bind_password:
        bind_password = encrypt_secret(body.bind_password)
        bind_password_for_discovery = body.bind_password
    else:
        bind_password = existing.bindPassword
        bind_password_for_discovery = decrypt_secret(existing.bindPassword)
    allow_self_signed = body.allow_self_signed if body.allow_self_signed is not None else existing.allowSelfSigned
    user_search_filter = body.user_search_filter if body.user_search_filter is not None else existing.userSearchFilter
    admin_group_dns = _normalize_dn_list(body.admin_group_dns if body.admin_group_dns is not None else existing.adminGroupDns)
    user_group_dns = _normalize_dn_list(body.user_group_dns if body.user_group_dns is not None else existing.userGroupDns)

    # Discover structure if user_search_base not provided and we have connection details
    discovery = None
    base_dn = existing.baseDn or ""
    user_search_base = body.user_search_base if body.user_search_base is not None else existing.userSearchBase
    discovered_ous = existing.discoveredOus if isinstance(existing.discoveredOus, list) else []
    discovered_groups = existing.discoveredGroups if isinstance(existing.discoveredGroups, list) else []

    if server_url and bind_dn and bind_password and not user_search_base:
        discovery = await discover_ldap_structure(
            server_url=server_url,
            bind_dn=bind_dn,
            bind_password=bind_password_for_discovery,
            allow_self_signed=allow_self_signed,
        )
        if discovery.success:
            base_dn = discovery.base_dn or ""
            user_search_base = discovery.user_ous[0] if discovery.user_ous else ""
            discovered_ous = discovery.user_ous
            discovered_groups = discovery.groups

    config = await db.ldapconfig.upsert(
        where={"id": "default"},
        data={
            "create": {
                "id": "default",
                "serverUrl": server_url or "",
                "bindDn": bind_dn or "",
                "bindPassword": bind_password or "",
                "allowSelfSigned": allow_self_signed,
                "baseDn": base_dn,
                "userSearchBase": user_search_base or "",
                "userSearchFilter": user_search_filter or "(uid={username})",
                "adminGroupDns": admin_group_dns,
                "userGroupDns": user_group_dns,
                "discoveredOus": Json(discovered_ous),
                "discoveredGroups": Json(discovered_groups),
            },
            "update": {
                "serverUrl": server_url or "",
                "bindDn": bind_dn or "",
                "bindPassword": bind_password or "",
                "allowSelfSigned": allow_self_signed,
                "baseDn": base_dn,
                "userSearchBase": user_search_base or "",
                "userSearchFilter": user_search_filter or "(uid={username})",
                "adminGroupDns": admin_group_dns,
                "userGroupDns": user_group_dns,
                "discoveredOus": Json(discovered_ous),
                "discoveredGroups": Json(discovered_groups),
            },
        },
    )
    await _sync_ldap_auth_group_assignments(
        admin_group_dns=admin_group_dns,
        user_group_dns=user_group_dns,
    )

    logger.info("LDAP configuration updated")

    return LdapConfigResponse(
        server_url=config.serverUrl,
        bind_dn=config.bindDn,
        allow_self_signed=config.allowSelfSigned,
        base_dn=config.baseDn,
        user_search_base=config.userSearchBase,
        user_search_filter=config.userSearchFilter,
        admin_group_dns=_normalize_dn_list(config.adminGroupDns),
        user_group_dns=_normalize_dn_list(config.userGroupDns),
        discovered_ous=(config.discoveredOus if isinstance(config.discoveredOus, list) else []),
        discovered_groups=(config.discoveredGroups if isinstance(config.discoveredGroups, list) else []),
    )


@router.post("/ldap/test")
async def test_ldap_connection(
    body: LdapDiscoverRequest,
    _user: User = Depends(require_admin),
):
    """Test LDAP connection with provided credentials."""
    result = await discover_ldap_structure(
        server_url=body.server_url,
        bind_dn=body.bind_dn,
        bind_password=body.bind_password,
        allow_self_signed=body.allow_self_signed,
    )

    if result.success:
        return {
            "success": True,
            "message": f"Connected successfully. Base DN: {result.base_dn}",
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Connection failed",
        )


@router.post("/ldap/search-user", response_model=LdapUserProfileResponse)
async def search_ldap_user(
    body: LdapUserSearchRequest,
    _user: User = Depends(require_admin),
):
    """Search configured LDAP for one user without importing it."""
    profile = await search_ldap_user_profile(body.username)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LDAP user not found or not authorized by configured groups",
        )
    return LdapUserProfileResponse(
        username=profile.username,
        source_dn=profile.source_dn,
        display_name=profile.display_name,
        email=profile.email,
        role=profile.role,
        groups=profile.groups,
    )


@router.post("/ldap/search-users", response_model=LdapUserTypeaheadResponse)
async def search_ldap_users(
    body: LdapUserTypeaheadRequest,
    _user: User = Depends(require_admin),
):
    """Search configured LDAP for multiple users (typeahead/autocomplete)."""
    profiles = await search_ldap_user_profiles(body.query, limit=body.limit)
    return LdapUserTypeaheadResponse(
        users=[
            LdapUserProfileResponse(
                username=profile.username,
                source_dn=profile.source_dn,
                display_name=profile.display_name,
                email=profile.email,
                role=profile.role,
                groups=profile.groups,
            )
            for profile in profiles
        ]
    )


@router.post("/ldap/import-user", response_model=LdapUserImportResponse)
async def import_ldap_user(
    body: LdapUserSearchRequest,
    _user: User = Depends(require_admin),
):
    """Import one LDAP identity into the local cache without storing a password."""
    imported = await import_ldap_user_profile(body.username)
    if imported is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LDAP user not found or not authorized by configured groups",
        )
    return LdapUserImportResponse(user=await _user_response(imported))


@router.get("/provider/config", response_model=AuthProviderConfigResponse)
async def get_provider_config(_user: User = Depends(require_admin)):
    """Get provider-neutral authentication policy flags."""
    return _auth_provider_config_response(await get_auth_provider_config())


@router.put("/provider/config", response_model=AuthProviderConfigResponse)
async def update_provider_config(
    body: UpdateAuthProviderConfigRequest,
    _user: User = Depends(require_admin),
):
    """Update provider-neutral authentication policy flags."""
    current_config = await get_auth_provider_config()
    mfa_methods = body.mfa_allowed_methods
    if mfa_methods is not None:
        normalized = {str(m).lower().strip() for m in mfa_methods}
        valid = {"totp", "webauthn"}
        if not normalized or not normalized.issubset(valid):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="mfa_allowed_methods must be a non-empty subset of ['totp', 'webauthn']",
            )
        mfa_methods = sorted(normalized & valid)

    effective_allowed = set(mfa_methods if mfa_methods is not None else current_config.mfa_allowed_methods)
    mfa_default_method: Any = _UNSET
    if "mfa_default_method" in body.model_fields_set:
        if body.mfa_default_method is not None and body.mfa_default_method not in effective_allowed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="mfa_default_method must be null or one of the effective allowed methods",
            )
        mfa_default_method = body.mfa_default_method

    updated = await update_auth_provider_config(
        local_users_enabled=body.local_users_enabled,
        ldap_lazy_sync_enabled=body.ldap_lazy_sync_enabled,
        manual_role_override_wins=body.manual_role_override_wins,
        cache_ttl_minutes=body.cache_ttl_minutes,
        totp_policy=body.totp_policy,
        totp_required_group_ids=body.totp_required_group_ids,
        totp_remember_device_days=body.totp_remember_device_days,
        mfa_allowed_methods=mfa_methods,
        mfa_default_method=mfa_default_method,
    )
    return _auth_provider_config_response(updated)


@router.post("/local/users", response_model=UserResponse)
async def create_local_user(
    body: LocalUserCreateRequest,
    _user: User = Depends(require_admin),
):
    """Create an internal managed user."""
    role = UserRole.admin if body.role == "admin" else UserRole.user
    try:
        user = await create_or_update_local_managed_user(
            username=body.username,
            password=body.password,
            display_name=body.display_name,
            email=body.email,
            role=role,
            role_manually_set=role == UserRole.admin,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return await _user_response(user)


@router.patch("/local/users/{user_id}", response_model=UserResponse)
async def update_local_user(
    user_id: str,
    body: LocalUserUpdateRequest,
    _user: User = Depends(require_admin),
):
    """Update an internal managed user profile or password."""
    db = await get_db()
    existing = await db.user.find_unique(where={"id": user_id})
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if _auth_provider_value(existing.authProvider) != "local_managed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only internal managed users can be edited here",
        )
    role = existing.role
    if body.role:
        role = UserRole.admin if body.role == "admin" else UserRole.user
    try:
        user = await create_or_update_local_managed_user(
            user_id=user_id,
            username=existing.username,
            password=body.password,
            display_name=(body.display_name if body.display_name is not None else existing.displayName),
            email=(body.email if body.email is not None else existing.email),
            role=role,
            role_manually_set=True if body.role is not None else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if body.password is not None:
        await invalidate_all_sessions(user_id)
    return await _user_response(user)


def _normalize_group_key(value: str) -> str:
    normalized = "-".join(value.strip().lower().split())
    normalized = "".join(ch for ch in normalized if ch.isalnum() or ch in {"-", "_", "."})
    return normalized or "group"


@router.get("/groups", response_model=AuthGroupListResponse)
async def list_auth_groups(_user: User = Depends(require_admin)):
    """List Group Memberships."""
    db = await get_db()
    groups = await db.authgroup.find_many(order={"displayName": "asc"})
    return AuthGroupListResponse(groups=[await _auth_group_response(group) for group in groups])


@router.post("/groups", response_model=AuthGroupResponse)
async def create_auth_group(
    body: AuthGroupUpsertRequest,
    _user: User = Depends(require_admin),
):
    """Create a local managed auth group."""
    db = await get_db()
    key = body.key or _normalize_group_key(body.display_name)
    role = _role_from_form_value(body.role)
    try:
        group = await db.authgroup.create(
            data={
                "key": key,
                "displayName": body.display_name,
                "description": body.description,
                "provider": AuthProvider.local_managed,
                "role": role,
                "isLogonGroup": body.is_logon_group,
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create group: {exc}",
        )
    return await _auth_group_response(group)


@router.patch("/groups/{group_id}", response_model=AuthGroupResponse)
async def update_auth_group(
    group_id: str,
    body: AuthGroupUpsertRequest,
    _user: User = Depends(require_admin),
):
    """Update auth group role and logon-gate assignments."""
    db = await get_db()
    group = await db.authgroup.find_unique(where={"id": group_id})
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found")
    role = _role_from_form_value(body.role)
    group_provider = _auth_provider_value(group.provider)
    update_data: dict[str, Any] = {
        "role": role,
        "isLogonGroup": body.is_logon_group,
    }
    if group_provider == "local_managed":
        update_data.update(
            {
                "key": body.key or group.key,
                "displayName": body.display_name,
                "description": body.description,
            }
        )
    updated = await db.authgroup.update(where={"id": group_id}, data=cast(Any, update_data))
    await _set_ldap_group_assignment(
        updated,
        role=role,
        is_logon_group=body.is_logon_group,
    )
    await recompute_auth_group_member_roles(group_id)
    return await _auth_group_response(updated)


@router.delete("/groups/{group_id}")
async def delete_auth_group(
    group_id: str,
    _user: User = Depends(require_admin),
):
    """Delete an auth group and its memberships."""
    db = await get_db()
    group = await db.authgroup.find_unique(where={"id": group_id})
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found")
    if _auth_provider_value(group.provider) != "local_managed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=("LDAP-synced groups cannot be deleted manually because they are recreated from LDAP membership sync"),
        )
    memberships = await db.authgroupmembership.find_many(where={"groupId": group_id})
    affected_user_ids = sorted({membership.userId for membership in memberships})
    await db.authgroup.delete(where={"id": group_id})
    for affected_user_id in affected_user_ids:
        await recompute_user_effective_role(affected_user_id)
    return {"success": True}


@router.put("/users/{user_id}/groups", response_model=UserResponse)
async def set_user_groups(
    user_id: str,
    body: SetUserGroupsRequest,
    current_user: User = Depends(require_admin),
):
    """Replace a user's manual internal group memberships."""
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own groups",
        )
    db = await get_db()
    user = await db.user.find_unique(where={"id": user_id})
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    desired = set(body.group_ids)
    desired_groups = await db.authgroup.find_many(where={"id": {"in": list(desired)}}) if desired else []
    found_group_ids = {group.id for group in desired_groups}
    unknown_group_ids = sorted(desired - found_group_ids)
    if unknown_group_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown group id: {unknown_group_ids[0]}",
        )
    for group in desired_groups:
        if _auth_provider_value(group.provider) != "local_managed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Manual group assignment only supports internal groups",
            )

    existing = await db.authgroupmembership.find_many(where={"userId": user_id, "sourceProvider": AuthProvider.local_managed})
    for membership in existing:
        if membership.groupId not in desired:
            await db.authgroupmembership.delete(where={"id": membership.id})

    for group in desired_groups:
        await db.authgroupmembership.upsert(
            where={"userId_groupId": {"userId": user_id, "groupId": group.id}},
            data={
                "create": {
                    "userId": user_id,
                    "groupId": group.id,
                    "sourceProvider": AuthProvider.local_managed,
                },
                "update": {"sourceProvider": AuthProvider.local_managed},
            },
        )

    refreshed = await recompute_user_effective_role(user_id)
    if not refreshed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return await _user_response(refreshed)


# =============================================================================
# User Management
# =============================================================================


@router.get("/users/directory", response_model=UserDirectoryResponse)
async def list_users_directory(
    _user: User = Depends(get_current_user),
):
    """List all users with minimal info (available to all authenticated users)."""
    db = await get_db()
    users = await db.user.find_many(
        order={"username": "asc"},
    )
    return UserDirectoryResponse(
        users=[
            UserDirectoryEntryResponse(
                id=u.id,
                username=u.username,
                display_name=u.displayName,
            )
            for u in users
        ]
    )


@router.get("/users", response_model=UserListResponse)
async def list_users(
    skip: int = Query(0, ge=0),
    take: int = Query(50, ge=1, le=500),
    _user: User = Depends(require_admin),
):
    """List all users (admin only)."""
    db = await get_db()
    total = await db.user.count()
    users = await db.user.find_many(
        order={"createdAt": "desc"},
        skip=skip,
        take=take,
    )

    return UserListResponse(
        users=await _bulk_user_responses(users),
        total=total,
        skip=skip,
        take=take,
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_admin),
):
    """Delete a user (admin only)."""
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete yourself",
        )

    db = await get_db()
    user = await db.user.find_unique(where={"id": user_id})

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    await db.user.delete(where={"id": user_id})
    logger.info(f"User '{user.username}' deleted by admin '{current_user.username}'")

    return {"success": True}


@router.delete("/users/{user_id}/mfa")
async def reset_user_mfa_by_admin(
    user_id: str,
    current_user: User = Depends(require_admin),
):
    """Reset a user's MFA enrollment and remembered devices (admin only)."""
    db = await get_db()
    user = await db.user.find_unique(where={"id": user_id})
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    await reset_user_mfa(user_id)
    await invalidate_all_sessions(user_id)
    logger.info(f"User '{user.username}' MFA reset by admin '{current_user.username}'")
    return {"success": True}


@router.patch("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    request: UpdateUserRoleRequest = Body(...),
    current_user: User = Depends(require_admin),
):
    """Update user role (admin only)."""
    resolved_role = request.role

    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role",
        )

    # Convert string to enum
    role_enum = UserRole.admin if resolved_role == "admin" else UserRole.user

    db = await get_db()
    user = await db.user.update(
        where={"id": user_id},
        data={"role": role_enum, "roleManuallySet": True},
    )

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    logger.info(f"User '{user.username}' role changed to '{resolved_role}' by admin '{current_user.username}'")

    return {"success": True, "role": resolved_role}


@router.post("/users/{user_id}/role/reset", response_model=ResetUserRoleResponse)
async def reset_user_role_override(
    user_id: str,
    current_user: User = Depends(require_admin),
):
    """Reset manual role override and re-apply LDAP-derived role (admin only)."""
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot reset your own role override",
        )

    db = await get_db()
    user = await db.user.find_unique(where={"id": user_id})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if _auth_provider_value(user.authProvider) != "ldap":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role reset is only supported for LDAP users",
        )

    if not user.ldapDn:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LDAP DN is missing for this user",
        )

    resolved_role, error = await resolve_ldap_role_for_user_dn(
        user.ldapDn,
        ldap_username_hint=user.username,
    )
    if error or resolved_role is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error or "Failed to resolve LDAP role",
        )

    updated_user = await db.user.update(
        where={"id": user_id},
        data={"role": resolved_role, "roleManuallySet": False},
    )
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    role_value: Literal["user", "admin"] = "admin" if updated_user.role == UserRole.admin else "user"

    logger.info(f"User '{updated_user.username}' role override reset by admin '{current_user.username}'")

    return ResetUserRoleResponse(
        success=True,
        role=role_value,
        role_manually_set=False,
    )


# =============================================================================
# Usage Dashboard (Admin)
# =============================================================================


@router.get("/usage/summary")
async def usage_summary(
    days: int = Query(30, ge=1, le=365),
    _user: User = Depends(require_admin),
):
    """Per-user usage summary for the admin dashboard."""
    since = _usage_window_start(days)
    rows = await get_user_usage_summary(since=since)
    return {"users": rows, "days": days}


@router.get("/usage/providers")
async def usage_providers(
    days: int = Query(30, ge=1, le=365),
    _user: User = Depends(require_admin),
):
    """Usage breakdown by provider and model."""
    since = _usage_window_start(days)
    ui_rows = await get_provider_model_breakdown(since=since)
    api_rows = await get_api_provider_model_breakdown(since=since)
    rows = ui_rows + api_rows
    return {"providers": rows, "days": days}


@router.get("/usage/daily")
async def usage_daily(
    days: int = Query(30, ge=1, le=365),
    _user: User = Depends(require_admin),
):
    """Daily usage trend for the admin dashboard."""
    since = _usage_window_start(days)
    rows = await get_daily_usage_trend(since=since)
    return {"daily": rows, "days": days}


@router.get("/usage/users/daily")
async def usage_users_daily(
    days: int = Query(30, ge=1, le=365),
    _user: User = Depends(require_admin),
):
    """Per-user daily usage trend for the admin dashboard."""
    since = _usage_window_start(days)
    rows = await get_user_daily_usage_series(since=since)
    return {"series": rows, "days": days}


@router.get("/usage/range")
async def usage_range(
    _user: User = Depends(require_admin),
):
    """Return the earliest usage data date for dynamic range selection."""
    earliest = await get_usage_earliest_date()
    return {"earliest_date": earliest}


@router.get("/usage/providers/daily-failures")
async def usage_provider_daily_failures(
    days: int = Query(30, ge=1, le=365),
    _user: User = Depends(require_admin),
):
    """Daily failure/interrupted counts by provider and model."""
    since = _usage_window_start(days)
    rows = await get_daily_provider_failures(since=since)
    return {"cells": rows, "days": days}


@router.get("/usage/mcp")
async def usage_mcp(
    days: int = Query(30, ge=1, le=365),
    _user: User = Depends(require_admin),
):
    """MCP request usage summary for the admin dashboard."""
    since = _usage_window_start(days)
    users, daily, routes = (
        await get_mcp_usage_by_user(since=since),
        await get_mcp_daily_trend(since=since),
        await get_mcp_usage_by_route(since=since),
    )
    return {"users": users, "daily": daily, "routes": routes, "days": days}


@router.get("/usage/api")
async def usage_api(
    days: int = Query(30, ge=1, le=365),
    _user: User = Depends(require_admin),
):
    """API request daily usage summary for the admin dashboard."""
    since = _usage_window_start(days)
    daily = await get_api_daily_trend(since=since)
    return {"daily": daily, "days": days}
