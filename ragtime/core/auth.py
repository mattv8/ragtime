"""
Authentication module for LDAP and local auth with JWT sessions.

Supports:
- LDAP authentication against Active Directory or OpenLDAP
- Local fallback admin account (env-based, prefixed with "local:")
- JWT tokens in httpOnly cookies
- User sync to PostgreSQL on successful auth
- Role-based access control (user/admin)
"""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt  # type: ignore[import-untyped]
from ldap3 import (  # type: ignore[import-untyped]
    Connection,
    Server,
    ALL,
    SUBTREE,
    AUTO_BIND_NO_TLS,
    AUTO_BIND_TLS_BEFORE_BIND,
)
from ldap3.core.exceptions import LDAPException, LDAPBindError  # type: ignore[import-untyped]
from prisma.enums import AuthProvider, UserRole
from pydantic import BaseModel

from ragtime.config.settings import settings
from ragtime.core.database import get_db
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

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


# =============================================================================
# JWT Token Management
# =============================================================================


def create_access_token(user_id: str, username: str, role: str) -> str:
    """Create a JWT access token."""
    expire = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expire_hours)
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        return TokenData(
            user_id=payload["sub"],
            username=payload["username"],
            role=payload["role"],
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        )
    except JWTError as e:
        logger.debug(f"JWT decode error: {e}")
        return None


def hash_token(token: str) -> str:
    """Create a hash of a token for database storage."""
    return hashlib.sha256(token.encode()).hexdigest()


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


def _get_ldap_connection(server_url: str, bind_dn: str, bind_password: str) -> Optional[Connection]:
    """Create an LDAP connection."""
    try:
        # Parse server URL
        use_ssl = server_url.startswith("ldaps://")
        server = Server(server_url, get_info=ALL, use_ssl=use_ssl)

        auto_bind = AUTO_BIND_TLS_BEFORE_BIND if use_ssl else AUTO_BIND_NO_TLS
        conn = Connection(
            server,
            user=bind_dn,
            password=bind_password,
            auto_bind=auto_bind,
            raise_exceptions=True,
        )
        return conn
    except LDAPException as e:
        logger.error(f"LDAP connection error: {e}")
        return None


async def discover_ldap_structure(
    server_url: str,
    bind_dn: str,
    bind_password: str,
) -> LdapDiscoveryResult:
    """
    Discover LDAP structure: base DN, user OUs, and groups.

    This auto-discovers the LDAP structure so users don't need to manually
    configure search bases and filters.
    """
    conn = _get_ldap_connection(server_url, bind_dn, bind_password)
    if not conn:
        return LdapDiscoveryResult(success=False, error="Failed to connect to LDAP server")

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
            return LdapDiscoveryResult(success=False, error="Could not determine base DN")

        # Discover user OUs
        user_ous: list[str] = []
        conn.search(
            search_base=base_dn,
            search_filter="(objectClass=organizationalUnit)",
            search_scope=SUBTREE,
            attributes=["distinguishedName", "ou"],
        )
        for entry in conn.entries:
            dn = str(entry.entry_dn)
            # Prioritize OUs that look like user containers
            if any(kw in dn.lower() for kw in ["user", "people", "account"]):
                user_ous.insert(0, dn)
            else:
                user_ous.append(dn)

        # Also check for CN=Users container (common in AD)
        conn.search(
            search_base=base_dn,
            search_filter="(&(objectClass=container)(cn=Users))",
            search_scope=SUBTREE,
            attributes=["distinguishedName"],
        )
        for entry in conn.entries:
            user_ous.insert(0, str(entry.entry_dn))

        # Discover groups
        groups = []
        conn.search(
            search_base=base_dn,
            search_filter="(|(objectClass=group)(objectClass=groupOfNames)(objectClass=posixGroup))",
            search_scope=SUBTREE,
            attributes=["distinguishedName", "cn"],
        )
        for entry in conn.entries:
            dn = str(entry.entry_dn)
            cn = str(entry.cn) if hasattr(entry, "cn") else dn.split(",")[0].replace("CN=", "")
            groups.append({"dn": dn, "name": cn})

        conn.unbind()

        return LdapDiscoveryResult(
            success=True,
            base_dn=base_dn,
            user_ous=user_ous[:50],  # Limit to 50
            groups=groups[:100],  # Limit to 100
        )

    except LDAPException as e:
        logger.error(f"LDAP discovery error: {e}")
        return LdapDiscoveryResult(success=False, error=str(e))
    finally:
        if conn.bound:
            conn.unbind()


async def authenticate_ldap(username: str, password: str) -> AuthResult:
    """
    Authenticate user against LDAP.

    Steps:
    1. Bind with service account
    2. Search for user by sAMAccountName or uid
    3. Attempt bind with user's credentials
    4. Check group membership for role assignment
    5. Sync user to database
    """
    ldap_config = await get_ldap_config()

    if not ldap_config.serverUrl or not ldap_config.bindDn:
        return AuthResult(success=False, error="LDAP not configured")

    # Connect with service account
    conn = _get_ldap_connection(
        ldap_config.serverUrl,
        ldap_config.bindDn,
        ldap_config.bindPassword,
    )
    if not conn:
        return AuthResult(success=False, error="Failed to connect to LDAP server")

    try:
        # Build search base
        search_base = ldap_config.userSearchBase or ldap_config.baseDn
        if not search_base:
            return AuthResult(success=False, error="LDAP search base not configured")

        # Search for user
        search_filter = ldap_config.userSearchFilter.replace("{username}", username)
        conn.search(
            search_base=search_base,
            search_filter=search_filter,
            search_scope=SUBTREE,
            attributes=["distinguishedName", "sAMAccountName", "uid", "mail", "displayName", "memberOf"],
        )

        if not conn.entries:
            return AuthResult(success=False, error="User not found")

        user_entry = conn.entries[0]
        user_dn = str(user_entry.entry_dn)
        user_mail = str(user_entry.mail) if hasattr(user_entry, "mail") and user_entry.mail else None
        user_display = str(user_entry.displayName) if hasattr(user_entry, "displayName") and user_entry.displayName else username

        # Get username from LDAP (prefer sAMAccountName, fall back to uid)
        ldap_username = username
        if hasattr(user_entry, "sAMAccountName") and user_entry.sAMAccountName:
            ldap_username = str(user_entry.sAMAccountName)
        elif hasattr(user_entry, "uid") and user_entry.uid:
            ldap_username = str(user_entry.uid)

        conn.unbind()

        # Verify user's password by binding as the user
        user_conn = _get_ldap_connection(ldap_config.serverUrl, user_dn, password)
        if not user_conn:
            return AuthResult(success=False, error="Invalid credentials")
        user_conn.unbind()

        # Determine role based on group membership
        role: UserRole = UserRole.user
        if ldap_config.adminGroupDn:
            member_of: list[str] = []
            if hasattr(user_entry, "memberOf") and user_entry.memberOf:
                member_of = [str(g) for g in user_entry.memberOf]

            if ldap_config.adminGroupDn in member_of:
                role = UserRole.admin
            elif ldap_config.userGroupDn and ldap_config.userGroupDn not in member_of:
                # If user group is specified, user must be a member
                return AuthResult(success=False, error="User not in authorized group")

        # Sync user to database
        db = await get_db()
        existing_user = await db.user.find_unique(where={"username": ldap_username})

        if existing_user:
            # Update existing user
            user = await db.user.update(
                where={"id": existing_user.id},
                data={
                    "ldapDn": user_dn,
                    "email": user_mail,
                    "displayName": user_display,
                    "role": role,
                    "lastLoginAt": datetime.now(timezone.utc),
                },
            )
        else:
            # Create new user
            user = await db.user.create(
                data={
                    "username": ldap_username,
                    "authProvider": AuthProvider.ldap,
                    "ldapDn": user_dn,
                    "email": user_mail,
                    "displayName": user_display,
                    "role": role,
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
    if username != settings.local_admin_user or password != settings.local_admin_password:
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


# =============================================================================
# Combined Authentication
# =============================================================================


async def authenticate(username: str, password: str) -> AuthResult:
    """
    Authenticate user via LDAP or local fallback.

    Order:
    1. If LDAP is enabled, try LDAP first
    2. If LDAP fails or is disabled, try local admin
    """
    # Check for local admin login first (username matches local admin)
    if username == settings.local_admin_user and settings.local_admin_password:
        local_result = await authenticate_local(username, password)
        if local_result.success:
            return local_result

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


async def create_session(user_id: str, token: str, user_agent: Optional[str] = None, ip_address: Optional[str] = None):
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
