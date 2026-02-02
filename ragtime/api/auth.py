"""
Authentication API routes.

Endpoints for login, logout, user info, and LDAP configuration.
"""

import base64
import hashlib
import time
from typing import Optional
from urllib.parse import urlparse

from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from prisma import Json
from prisma.enums import UserRole
from prisma.models import User
from pydantic import BaseModel, Field

from ragtime.config.settings import settings
from ragtime.core.auth import (
    authenticate,
    create_access_token,
    create_session,
    discover_ldap_structure,
    get_ldap_config,
    invalidate_session,
    lookup_bind_dn,
)
from ragtime.core.database import get_db
from ragtime.core.encryption import decrypt_secret, encrypt_secret
from ragtime.core.logging import get_logger
from ragtime.core.rate_limit import LOGIN_RATE_LIMIT, limiter
from ragtime.core.security import get_current_user, get_session_token, require_admin

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# In-memory storage for authorization codes (short-lived, 10 min expiry)
# Format: {code: {"client_id": str, "redirect_uri": str, "code_challenge": str,
#                 "user_id": str, "username": str, "role": str, "expires": float}}
_auth_codes: dict[str, dict] = {}
AUTH_CODE_EXPIRY = 600  # 10 minutes
MAX_AUTH_CODES = 10000  # Prevent memory exhaustion from code accumulation


def validate_redirect_uri(redirect_uri: str) -> bool:
    """
    Validate redirect_uri for OAuth2 security.

    For MCP clients (VS Code, JetBrains, etc.), allow:
    - Loopback addresses per RFC 8252 Section 7.3
    - Trusted IDE redirect domains (vscode.dev, etc.)

    This prevents open redirect attacks while supporting OAuth flows from
    various development tools.

    RFC 8252 Section 7.3: Loopback Interface Redirection
    - http://127.0.0.1:<port>/<path>
    - http://localhost:<port>/<path>
    - http://[::1]:<port>/<path>
    """
    try:
        parsed = urlparse(redirect_uri)

        # Must be http or https
        if parsed.scheme not in ("http", "https"):
            logger.warning(
                f"OAuth2 redirect_uri rejected: scheme must be http or https, got {parsed.scheme}"
            )
            return False

        # Extract hostname (remove port if present)
        hostname = parsed.hostname
        if hostname is None:
            logger.warning("OAuth2 redirect_uri rejected: no hostname")
            return False

        # Allow loopback addresses (RFC 8252 for native apps)
        loopback_hosts = {"127.0.0.1", "localhost", "::1", "[::1]"}

        # Allow trusted IDE redirect domains
        # These are official OAuth redirect endpoints for IDEs
        trusted_domains = {
            "vscode.dev",  # VS Code web/desktop OAuth callback
            "insiders.vscode.dev",  # VS Code Insiders
            "github.dev",  # GitHub Codespaces
            "jetbrains.com",  # JetBrains IDEs
            "account.jetbrains.com",
        }

        # Check if it's a loopback or trusted domain
        is_loopback = hostname in loopback_hosts
        is_trusted = hostname in trusted_domains or any(
            hostname.endswith(f".{domain}") for domain in trusted_domains
        )

        if not is_loopback and not is_trusted:
            logger.warning(
                f"OAuth2 redirect_uri rejected: '{hostname}' is not a loopback address or trusted domain"
            )
            return False

        # For loopback, port can be any valid port (MCP clients pick available ports)
        if (
            is_loopback
            and parsed.port is not None
            and (parsed.port < 1 or parsed.port > 65535)
        ):
            logger.warning(f"OAuth2 redirect_uri rejected: invalid port {parsed.port}")
            return False

        # In debug mode, log successful validation for troubleshooting
        if settings.debug_mode:
            logger.debug(f"OAuth2 redirect_uri validated: {redirect_uri}")

        return True

    except Exception as e:
        logger.warning(f"OAuth2 redirect_uri validation error: {e}")
        return False


# =============================================================================
# Request/Response Models
# =============================================================================


class LoginRequest(BaseModel):
    """Login request body."""

    username: str = Field(
        ..., min_length=1, description="Username (sAMAccountName, uid, or local admin)"
    )
    password: str = Field(..., min_length=1, description="Password")


class LoginResponse(BaseModel):
    """Login response with user info."""

    success: bool
    user_id: Optional[str] = None
    username: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    role: str = "user"
    error: Optional[str] = None


class UserResponse(BaseModel):
    """Current user info response."""

    id: str
    username: str
    display_name: Optional[str]
    email: Optional[str]
    role: str
    auth_provider: str


class LdapConfigRequest(BaseModel):
    """LDAP configuration update request."""

    server_url: Optional[str] = Field(
        None, description="LDAP server URL (ldap://host:389 or ldaps://host:636)"
    )
    bind_dn: Optional[str] = Field(
        None, description="Bind DN or sAMAccountName for service account"
    )
    bind_password: Optional[str] = Field(
        None, description="Bind password (leave empty to keep existing)"
    )
    allow_self_signed: Optional[bool] = Field(
        None, description="Allow self-signed SSL certificates"
    )
    user_search_base: Optional[str] = Field(
        None, description="User search base DN (auto-discovered if empty)"
    )
    user_search_filter: Optional[str] = Field(
        None, description="User search filter (use {username} placeholder)"
    )
    admin_group_dn: Optional[str] = Field(None, description="Admin group DN")
    user_group_dn: Optional[str] = Field(None, description="User group DN (optional)")


class LdapConfigResponse(BaseModel):
    """LDAP configuration response."""

    server_url: str
    bind_dn: str
    allow_self_signed: bool
    base_dn: str
    user_search_base: str
    user_search_filter: str
    admin_group_dn: str
    user_group_dn: str
    discovered_ous: list[str]
    discovered_groups: list[dict]


class LdapDiscoverRequest(BaseModel):
    """Request to discover LDAP structure."""

    server_url: str = Field(..., description="LDAP server URL")
    bind_dn: str = Field(..., description="Bind DN or sAMAccountName")
    bind_password: str = Field(..., description="Bind password")
    allow_self_signed: bool = Field(
        False, description="Allow self-signed SSL certificates"
    )


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
    username: str = Field(..., description="Username (sAMAccountName, uid, or UPN)")
    password: str = Field(..., description="Password")


class LdapBindDnLookupResponse(BaseModel):
    """Bind DN lookup result."""

    success: bool
    bind_dn: Optional[str] = None
    display_name: Optional[str] = None
    error: Optional[str] = None


class AuthStatusResponse(BaseModel):
    """Authentication status response."""

    authenticated: bool
    ldap_configured: bool
    local_admin_enabled: bool
    debug_mode: bool = False
    debug_username: Optional[str] = None
    debug_password: Optional[str] = None
    cookie_warning: Optional[str] = None  # Warning about cookie/protocol mismatch
    # Security status for UI banner
    api_key_configured: bool = False
    session_cookie_secure: bool = False
    allowed_origins_open: bool = False  # True if ALLOWED_ORIGINS=*


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
            "Security notice: You are connecting over HTTPS but SESSION_COOKIE_SECURE=false. "
            "Consider setting SESSION_COOKIE_SECURE=true for better security."
        )

    return None


@router.get("/status", response_model=AuthStatusResponse)
async def get_auth_status(request: Request):
    """Get authentication system status."""
    ldap_config = await get_ldap_config()
    cookie_warning = _detect_cookie_mismatch(request)

    return AuthStatusResponse(
        authenticated=False,  # This endpoint is for unauthenticated users
        ldap_configured=bool(ldap_config.serverUrl),
        local_admin_enabled=bool(settings.local_admin_password),
        debug_mode=settings.debug_mode,
        debug_username=settings.local_admin_user if settings.debug_mode else None,
        debug_password=settings.local_admin_password if settings.debug_mode else None,
        cookie_warning=cookie_warning,
        api_key_configured=bool(settings.api_key),
        session_cookie_secure=settings.session_cookie_secure,
        allowed_origins_open=settings.allowed_origins == "*",
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
    result = await authenticate(body.username, body.password)

    if not result.success:
        logger.warning(f"Login failed for user '{body.username}': {result.error}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=result.error or "Authentication failed",
        )

    # These are guaranteed to be set on success
    if not result.user_id or not result.username:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication succeeded but user data is missing",
        )

    # Create JWT token
    token = create_access_token(result.user_id, result.username, result.role)

    # Create session in database
    await create_session(
        user_id=result.user_id,
        token=token,
        user_agent=request.headers.get("User-Agent"),
        ip_address=request.client.host if request.client else None,
    )

    # Set httpOnly cookie
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        httponly=settings.session_cookie_httponly,
        secure=settings.session_cookie_secure,
        samesite=settings.session_cookie_samesite,
        max_age=settings.jwt_expire_hours * 3600,
    )

    logger.info(
        f"User '{result.username}' logged in successfully (role: {result.role})"
    )

    return LoginResponse(
        success=True,
        user_id=result.user_id,
        username=result.username,
        display_name=result.display_name,
        email=result.email,
        role=result.role,
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
    scope: Optional[str] = Field(
        default=None, description="Requested scopes (optional)"
    )


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
    grant_type: str = Form(
        ..., description="Grant type ('password' or 'authorization_code')"
    ),
    username: Optional[str] = Form(
        default=None, description="Username (for password grant)"
    ),
    password: Optional[str] = Form(
        default=None, description="Password (for password grant)"
    ),
    code: Optional[str] = Form(
        default=None, description="Authorization code (for authorization_code grant)"
    ),
    code_verifier: Optional[str] = Form(
        default=None, description="PKCE code verifier (for authorization_code grant)"
    ),
    redirect_uri: Optional[str] = Form(
        default=None, description="Redirect URI (for authorization_code grant)"
    ),
    client_id: Optional[str] = Form(
        default=None, description="Client ID (for authorization_code grant)"
    ),
    scope: Optional[str] = Form(
        default=None, description="Requested scopes (optional)"
    ),
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
    # Handle authorization_code grant (PKCE flow)
    if grant_type == "authorization_code":
        if not code or not code_verifier:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_request",
                    "error_description": "code and code_verifier are required for authorization_code grant",
                },
            )

        # Cleanup expired codes
        _cleanup_expired_auth_codes()

        # Look up the authorization code
        auth_data = _auth_codes.get(code)
        if not auth_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_grant",
                    "error_description": "Invalid or expired authorization code",
                },
            )

        # Verify PKCE
        if not _verify_pkce(code_verifier, auth_data["code_challenge"]):
            # Remove the code on PKCE failure (prevent replay)
            del _auth_codes[code]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_grant",
                    "error_description": "PKCE verification failed",
                },
            )

        # Verify client_id matches (if provided) - prevents code theft
        if client_id and client_id != auth_data["client_id"]:
            del _auth_codes[code]
            logger.warning(
                f"OAuth2 client_id mismatch: expected '{auth_data['client_id']}', got '{client_id}'"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_grant",
                    "error_description": "client_id mismatch",
                },
            )

        # Verify redirect_uri matches (if provided)
        if redirect_uri and redirect_uri != auth_data["redirect_uri"]:
            del _auth_codes[code]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_grant",
                    "error_description": "redirect_uri mismatch",
                },
            )

        # Code is valid - consume it (one-time use)
        del _auth_codes[code]

        # Create access token
        token = create_access_token(
            auth_data["user_id"],
            auth_data["username"],
            auth_data["role"],
        )

        # Create session in database
        await create_session(
            user_id=auth_data["user_id"],
            token=token,
            user_agent=request.headers.get("User-Agent"),
            ip_address=request.client.host if request.client else None,
        )

        logger.info(
            f"OAuth2 token issued for '{auth_data['username']}' via authorization_code grant"
        )

        return OAuth2TokenResponse(
            access_token=token,
            token_type="Bearer",
            expires_in=settings.jwt_expire_hours * 3600,
            scope=scope,
        )

    # Handle password grant
    elif grant_type == "password":
        if not username or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_request",
                    "error_description": "username and password are required for password grant",
                },
            )

        # Check if LDAP is configured (OAuth2 requires LDAP)
        ldap_config = await get_ldap_config()
        if not ldap_config.serverUrl or not ldap_config.bindDn:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "server_error",
                    "error_description": "OAuth2 requires LDAP to be configured",
                },
            )

        # Authenticate via LDAP
        result = await authenticate(username, password)

        if not result.success:
            logger.warning(
                f"OAuth2 token request failed for '{username}': {result.error}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "invalid_grant",
                    "error_description": result.error or "Authentication failed",
                },
            )

        if not result.user_id or not result.username:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "server_error",
                    "error_description": "Authentication succeeded but user data is missing",
                },
            )

        # Create access token
        token = create_access_token(result.user_id, result.username, result.role)

        # Create session in database
        await create_session(
            user_id=result.user_id,
            token=token,
            user_agent=request.headers.get("User-Agent"),
            ip_address=request.client.host if request.client else None,
        )

        logger.info(f"OAuth2 token issued for '{result.username}' via password grant")

        return OAuth2TokenResponse(
            access_token=token,
            token_type="Bearer",
            expires_in=settings.jwt_expire_hours * 3600,
            scope=scope,
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "unsupported_grant_type",
                "error_description": "Supported grant types: 'password', 'authorization_code'",
            },
        )


# =============================================================================
# Authenticated Endpoints
# =============================================================================


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(user: User = Depends(get_current_user)):
    """Get current authenticated user info."""
    return UserResponse(
        id=user.id,
        username=user.username,
        display_name=user.displayName,
        email=user.email,
        role=user.role,
        auth_provider=user.authProvider,
    )


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
        admin_group_dn=config.adminGroupDn,
        user_group_dn=config.userGroupDn,
        discovered_ous=(
            config.discoveredOus if isinstance(config.discoveredOus, list) else []
        ),
        discovered_groups=(
            config.discoveredGroups if isinstance(config.discoveredGroups, list) else []
        ),
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
    else:
        bind_password = existing.bindPassword
    allow_self_signed = (
        body.allow_self_signed
        if body.allow_self_signed is not None
        else existing.allowSelfSigned
    )
    user_search_filter = (
        body.user_search_filter
        if body.user_search_filter is not None
        else existing.userSearchFilter
    )
    # Empty string means "clear the value", None means "keep existing"
    admin_group_dn = (
        body.admin_group_dn
        if body.admin_group_dn is not None
        else existing.adminGroupDn
    )
    user_group_dn = (
        body.user_group_dn if body.user_group_dn is not None else existing.userGroupDn
    )
    # Normalize empty strings to empty (not None) for storage
    admin_group_dn = admin_group_dn or ""
    user_group_dn = user_group_dn or ""

    # Discover structure if user_search_base not provided and we have connection details
    discovery = None
    base_dn = existing.baseDn or ""
    user_search_base = (
        body.user_search_base
        if body.user_search_base is not None
        else existing.userSearchBase
    )
    discovered_ous = (
        existing.discoveredOus if isinstance(existing.discoveredOus, list) else []
    )
    discovered_groups = (
        existing.discoveredGroups if isinstance(existing.discoveredGroups, list) else []
    )

    if server_url and bind_dn and bind_password and not user_search_base:
        discovery = await discover_ldap_structure(
            server_url=server_url,
            bind_dn=bind_dn,
            bind_password=bind_password,
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
                "userSearchFilter": user_search_filter
                or "(|(sAMAccountName={username})(uid={username}))",
                "adminGroupDn": admin_group_dn or "",
                "userGroupDn": user_group_dn or "",
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
                "userSearchFilter": user_search_filter
                or "(|(sAMAccountName={username})(uid={username}))",
                "adminGroupDn": admin_group_dn or "",
                "userGroupDn": user_group_dn or "",
                "discoveredOus": Json(discovered_ous),
                "discoveredGroups": Json(discovered_groups),
            },
        },
    )

    logger.info("LDAP configuration updated")

    return LdapConfigResponse(
        server_url=config.serverUrl,
        bind_dn=config.bindDn,
        allow_self_signed=config.allowSelfSigned,
        base_dn=config.baseDn,
        user_search_base=config.userSearchBase,
        user_search_filter=config.userSearchFilter,
        admin_group_dn=config.adminGroupDn,
        user_group_dn=config.userGroupDn,
        discovered_ous=(
            config.discoveredOus if isinstance(config.discoveredOus, list) else []
        ),
        discovered_groups=(
            config.discoveredGroups if isinstance(config.discoveredGroups, list) else []
        ),
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


# =============================================================================
# User Management (Admin Only)
# =============================================================================


@router.get("/users")
async def list_users(_user: User = Depends(require_admin)):
    """List all users."""
    db = await get_db()
    users = await db.user.find_many(
        order={"createdAt": "desc"},
    )

    return [
        UserResponse(
            id=u.id,
            username=u.username,
            display_name=u.displayName,
            email=u.email,
            role=u.role,
            auth_provider=u.authProvider,
        )
        for u in users
    ]


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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    await db.user.delete(where={"id": user_id})
    logger.info(f"User '{user.username}' deleted by admin '{current_user.username}'")

    return {"success": True}


@router.patch("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    role: str,
    current_user: User = Depends(require_admin),
):
    """Update user role (admin only)."""
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role",
        )

    if role not in ("user", "admin"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role. Must be 'user' or 'admin'",
        )

    # Convert string to enum
    role_enum = UserRole.admin if role == "admin" else UserRole.user

    db = await get_db()
    user = await db.user.update(
        where={"id": user_id},
        data={"role": role_enum},
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    logger.info(
        f"User '{user.username}' role changed to '{role}' by admin '{current_user.username}'"
    )

    return {"success": True, "role": role}
