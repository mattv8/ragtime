"""
Authentication API routes.

Endpoints for login, logout, user info, and LDAP configuration.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
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
)
from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.core.security import (
    get_current_user,
    get_session_token,
    require_admin,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# =============================================================================
# Request/Response Models
# =============================================================================


class LoginRequest(BaseModel):
    """Login request body."""
    username: str = Field(..., min_length=1, description="Username (sAMAccountName, uid, or local admin)")
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
    server_url: str = Field(..., description="LDAP server URL (ldap://host:389 or ldaps://host:636)")
    bind_dn: str = Field(..., description="Bind DN or sAMAccountName for service account")
    bind_password: str = Field(..., description="Bind password")
    user_search_base: Optional[str] = Field(None, description="User search base DN (auto-discovered if empty)")
    user_search_filter: str = Field(
        "(|(sAMAccountName={username})(uid={username}))",
        description="User search filter (use {username} placeholder)"
    )
    admin_group_dn: Optional[str] = Field(None, description="Admin group DN")
    user_group_dn: Optional[str] = Field(None, description="User group DN (optional)")


class LdapConfigResponse(BaseModel):
    """LDAP configuration response."""
    server_url: str
    bind_dn: str
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


class LdapDiscoverResponse(BaseModel):
    """LDAP discovery result."""
    success: bool
    base_dn: Optional[str] = None
    user_ous: list[str] = []
    groups: list[dict] = []
    error: Optional[str] = None


class AuthStatusResponse(BaseModel):
    """Authentication status response."""
    authenticated: bool
    ldap_configured: bool
    local_admin_enabled: bool


# =============================================================================
# Public Endpoints (No Auth Required)
# =============================================================================


@router.get("/status", response_model=AuthStatusResponse)
async def get_auth_status():
    """Get authentication system status."""
    ldap_config = await get_ldap_config()

    return AuthStatusResponse(
        authenticated=False,  # This endpoint is for unauthenticated users
        ldap_configured=bool(ldap_config.serverUrl),
        local_admin_enabled=bool(settings.local_admin_password),
    )


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    response: Response,
    body: LoginRequest,
):
    """
    Authenticate user and create session.

    Tries LDAP first (if enabled), then falls back to local admin.
    Sets httpOnly cookie with JWT token.
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

    logger.info(f"User '{result.username}' logged in successfully (role: {result.role})")

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
    )

    return LdapDiscoverResponse(
        success=result.success,
        base_dn=result.base_dn,
        user_ous=result.user_ous,
        groups=result.groups,
        error=result.error,
    )


@router.get("/ldap/config", response_model=LdapConfigResponse)
async def get_ldap_configuration(_user: User = Depends(require_admin)):
    """Get current LDAP configuration."""
    config = await get_ldap_config()

    return LdapConfigResponse(
        server_url=config.serverUrl,
        bind_dn=config.bindDn,
        base_dn=config.baseDn,
        user_search_base=config.userSearchBase,
        user_search_filter=config.userSearchFilter,
        admin_group_dn=config.adminGroupDn,
        user_group_dn=config.userGroupDn,
        discovered_ous=config.discoveredOus if isinstance(config.discoveredOus, list) else [],
        discovered_groups=config.discoveredGroups if isinstance(config.discoveredGroups, list) else [],
    )


@router.put("/ldap/config", response_model=LdapConfigResponse)
async def update_ldap_configuration(
    body: LdapConfigRequest,
    _user: User = Depends(require_admin),
):
    """Update LDAP configuration."""
    db = await get_db()

    # Discover structure if not provided
    discovery = None
    if not body.user_search_base:
        discovery = await discover_ldap_structure(
            server_url=body.server_url,
            bind_dn=body.bind_dn,
            bind_password=body.bind_password,
        )
        if not discovery.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"LDAP discovery failed: {discovery.error}",
            )

    # Prepare values with proper types
    base_dn = (discovery.base_dn or "") if discovery else ""
    user_search_base = body.user_search_base or (discovery.user_ous[0] if discovery and discovery.user_ous else "")
    discovered_ous = Json(discovery.user_ous) if discovery else Json([])
    discovered_groups = Json(discovery.groups) if discovery else Json([])

    config = await db.ldapconfig.upsert(
        where={"id": "default"},
        data={
            "create": {
                "id": "default",
                "serverUrl": body.server_url,
                "bindDn": body.bind_dn,
                "bindPassword": body.bind_password,
                "baseDn": base_dn,
                "userSearchBase": user_search_base,
                "userSearchFilter": body.user_search_filter,
                "adminGroupDn": body.admin_group_dn or "",
                "userGroupDn": body.user_group_dn or "",
                "discoveredOus": discovered_ous,
                "discoveredGroups": discovered_groups,
            },
            "update": {
                "serverUrl": body.server_url,
                "bindDn": body.bind_dn,
                "bindPassword": body.bind_password,
                "baseDn": base_dn,
                "userSearchBase": user_search_base,
                "userSearchFilter": body.user_search_filter,
                "adminGroupDn": body.admin_group_dn or "",
                "userGroupDn": body.user_group_dn or "",
                "discoveredOus": discovered_ous,
                "discoveredGroups": discovered_groups,
            },
        },
    )

    logger.info("LDAP configuration updated")

    return LdapConfigResponse(
        server_url=config.serverUrl,
        bind_dn=config.bindDn,
        base_dn=config.baseDn,
        user_search_base=config.userSearchBase,
        user_search_filter=config.userSearchFilter,
        admin_group_dn=config.adminGroupDn,
        user_group_dn=config.userGroupDn,
        discovered_ous=config.discoveredOus if isinstance(config.discoveredOus, list) else [],
        discovered_groups=config.discoveredGroups if isinstance(config.discoveredGroups, list) else [],
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
    )

    if result.success:
        return {"success": True, "message": f"Connected successfully. Base DN: {result.base_dn}"}
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    logger.info(f"User '{user.username}' role changed to '{role}' by admin '{current_user.username}'")

    return {"success": True, "role": role}
