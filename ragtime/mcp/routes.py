"""
MCP HTTP Transport Routes - Streamable HTTP transport for MCP.

Provides HTTP endpoints for MCP clients using the Streamable HTTP transport.
This allows MCP clients to connect via URL configuration like:

    "url": "http://localhost:8000/mcp"              # Default route (all tools)
    "url": "http://localhost:8000/mcp/my_toolset"   # Custom route (subset)

Supports both SSE streaming responses and JSON responses for stateless operation.
Custom routes support optional Bearer token authentication.

Default route supports LDAP group-based tool filtering when OAuth2 auth is enabled:
- Filters are configured via McpDefaultRouteFilter in the database
- Users see tools based on their LDAP group membership
- Higher priority filters take precedence when user is in multiple groups
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
from anyio.abc import TaskStatus
from fastapi import APIRouter, Request
from mcp.server.lowlevel.server import Server as MCPServer
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from ragtime.core.app_settings import get_app_settings
from ragtime.core.logging import get_logger
from ragtime.mcp.server import (
    get_custom_route_server,
    get_default_route_filtered_server,
    mcp_server,
)
from ragtime.mcp.tools import mcp_tool_adapter

logger = get_logger(__name__)

router = APIRouter(prefix="/mcp-debug", tags=["MCP Debug"])

# Session manager for Streamable HTTP transport (default route)
# Stateless mode with JSON responses for maximum compatibility
_session_manager: StreamableHTTPSessionManager | None = None

# Session managers for custom routes (created on demand)
_custom_session_managers: dict[str, StreamableHTTPSessionManager] = {}

# Cached filtered MCP servers (keyed by filter ID)
# We use the low-level server directly for filtered requests since
# StreamableHTTPSessionManager requires run() to be called before handling requests
_default_filter_servers: dict[str, MCPServer[Any, Any]] = {}


def get_session_manager() -> StreamableHTTPSessionManager:
    """Get or create the MCP session manager for the default route."""
    global _session_manager
    if _session_manager is None:
        _session_manager = StreamableHTTPSessionManager(
            app=mcp_server,
            event_store=None,  # Stateless mode - no event persistence
            json_response=True,  # JSON responses for broad client compatibility
            stateless=True,  # Stateless mode - each request is independent
        )
        logger.info("Created MCP StreamableHTTP session manager (stateless mode)")
    return _session_manager


async def get_filtered_server(
    filter_id: str,
) -> MCPServer[Any, Any] | None:
    """
    Get or create an MCP server for a default route filter.

    Args:
        filter_id: The filter ID

    Returns:
        MCP server configured with the filter's tool restrictions, or None
    """
    global _default_filter_servers

    # Check cache
    if filter_id in _default_filter_servers:
        return _default_filter_servers[filter_id]

    # Get filtered server
    result = await get_default_route_filtered_server(filter_id)
    if not result:
        return None

    server, _ = result
    _default_filter_servers[filter_id] = server
    logger.info(f"Cached filtered MCP server for default route filter: {filter_id}")

    return server


async def handle_filtered_request(
    server: MCPServer[Any, Any],
    scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    """
    Handle a filtered MCP request using the low-level server directly.

    This bypasses StreamableHTTPSessionManager since filtered servers are
    created on-demand and the session manager requires run() to be called first.
    We use stateless mode with a fresh transport per request.
    """
    logger.debug("Handling filtered request with direct server")

    # Create transport for this request
    http_transport = StreamableHTTPServerTransport(
        mcp_session_id=None,  # No session tracking in stateless mode
        is_json_response_enabled=True,
        event_store=None,
        security_settings=None,
    )

    # Run the server and handle the request
    async with anyio.create_task_group() as tg:

        async def run_stateless_server(
            *, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED
        ) -> None:
            async with http_transport.connect() as streams:
                read_stream, write_stream = streams
                task_status.started()
                try:
                    await server.run(
                        read_stream,
                        write_stream,
                        server.create_initialization_options(),
                        stateless=True,
                    )
                except Exception:
                    logger.exception("Filtered stateless session crashed")

        # Start the server task
        await tg.start(run_stateless_server)

        # Handle the HTTP request
        await http_transport.handle_request(scope, receive, send)

        # Terminate transport to clean up
        await http_transport.terminate()

        # Cancel task group to clean up
        tg.cancel_scope.cancel()


async def get_custom_session_manager(
    route_path: str,
) -> (
    tuple[StreamableHTTPSessionManager, bool, str | None, str | None, str | None] | None
):
    """
    Get or create a session manager for a custom route.

    Args:
        route_path: The route path suffix

    Returns:
        Tuple of (session_manager, require_auth, auth_password, auth_method, allowed_group)
        or None if route not found
    """
    # Check cache
    if route_path in _custom_session_managers:
        # Get require_auth, password, auth_method, allowed_group from route config
        result = await get_custom_route_server(route_path)
        if result:
            _, _, require_auth, auth_password, auth_method, allowed_group = result
            return (
                _custom_session_managers[route_path],
                require_auth,
                auth_password,
                auth_method,
                allowed_group,
            )
        return None

    # Get or create the server
    result = await get_custom_route_server(route_path)
    if not result:
        return None

    server, _, require_auth, auth_password, auth_method, allowed_group = result

    # Create session manager
    manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=True,
        stateless=True,
    )
    _custom_session_managers[route_path] = manager
    logger.info(f"Created custom MCP session manager for route: /mcp/{route_path}")

    return manager, require_auth, auth_password, auth_method, allowed_group


@asynccontextmanager
async def mcp_lifespan_manager() -> AsyncIterator[None]:
    """Lifespan manager for MCP session manager."""
    session_manager = get_session_manager()
    async with session_manager.run():
        logger.info("MCP StreamableHTTP session manager started")
        try:
            yield
        finally:
            # Also stop custom session managers
            for path in _custom_session_managers:
                logger.debug(f"Stopping custom session manager: {path}")
            _custom_session_managers.clear()
            # Clear default filter server cache
            for filter_id in _default_filter_servers:
                logger.debug(f"Clearing filter server cache: {filter_id}")
            _default_filter_servers.clear()
            logger.info("MCP StreamableHTTP session manager stopped")


async def _validate_bearer_token(scope: Scope) -> bool:
    """
    Validate Bearer token from Authorization header using session validation.

    Uses the same session validation as the rest of Ragtime.
    Used for default /mcp route when mcp_default_route_auth is enabled.

    Returns:
        True if valid, False otherwise
    """
    from ragtime.core.auth import validate_session

    # Get Authorization header
    headers = dict(scope.get("headers", []))
    auth_header = headers.get(b"authorization", b"").decode()

    if not auth_header.startswith("Bearer "):
        return False

    token = auth_header[7:]  # Remove "Bearer " prefix

    try:
        token_data = await validate_session(token)
        return token_data is not None
    except Exception as e:
        logger.debug(f"Token validation failed: {e}")
        return False


async def _validate_oauth2_token(
    scope: Scope, allowed_group_dn: str | None = None
) -> bool:
    """
    Validate OAuth2 Bearer token and optionally check LDAP group membership.

    Args:
        scope: ASGI scope
        allowed_group_dn: Optional LDAP group DN that user must be a member of

    Returns:
        True if valid and user is in allowed group (if specified), False otherwise

    Note:
        Local admin users bypass LDAP group restrictions since they don't have
        LDAP group membership. Regular local users are denied if group restriction
        is configured.
    """
    from ragtime.core.auth import validate_session
    from ragtime.core.database import get_db

    # Get Authorization header
    headers = dict(scope.get("headers", []))
    auth_header = headers.get(b"authorization", b"").decode()

    if not auth_header.startswith("Bearer "):
        return False

    token = auth_header[7:]  # Remove "Bearer " prefix

    try:
        token_data = await validate_session(token)
        if token_data is None:
            return False

        # If no group restriction, token is valid
        if not allowed_group_dn:
            return True

        # Check if user is in the allowed LDAP group
        db = await get_db()
        user = await db.user.find_unique(where={"id": token_data.user_id})
        if not user:
            logger.debug(f"User {token_data.user_id} not found in database")
            return False

        # Local admin users bypass LDAP group restrictions
        # They are trusted and don't have LDAP group membership
        if user.authProvider == "local" and user.role == "admin":
            logger.debug(
                f"Local admin '{user.username}' bypasses LDAP group restriction"
            )
            return True

        # Get user's LDAP DN and check group membership
        if not user.ldapDn:
            # Non-admin local users cannot access group-restricted routes
            logger.debug(
                f"User {user.username} has no LDAP DN and is not local admin - denying access"
            )
            return False

        # Re-check LDAP group membership for the user
        from ragtime.core.auth import _get_ldap_connection, get_ldap_config
        from ragtime.core.encryption import decrypt_secret

        ldap_config = await get_ldap_config()
        if not ldap_config.serverUrl:
            logger.debug("LDAP not configured - denying OAuth2 access")
            return False

        bind_password = decrypt_secret(ldap_config.bindPassword)
        conn = _get_ldap_connection(
            ldap_config.serverUrl,
            ldap_config.bindDn,
            bind_password,
            ldap_config.allowSelfSigned,
        )
        if not conn:
            logger.warning("Failed to connect to LDAP for group membership check")
            return False

        try:
            # Get user's memberOf and primaryGroupID
            conn.search(
                search_base=user.ldapDn,
                search_filter="(objectClass=*)",
                search_scope="BASE",
                attributes=["memberOf", "primaryGroupID"],
            )

            if not conn.entries:
                conn.unbind()
                return False

            entry = conn.entries[0]
            member_of = []
            if hasattr(entry, "memberOf") and entry.memberOf:
                member_of = [str(g).lower() for g in entry.memberOf]

            primary_group_id = None
            if hasattr(entry, "primaryGroupID") and entry.primaryGroupID:
                primary_group_id = int(str(entry.primaryGroupID))

            # Case-insensitive DN comparison
            allowed_group_lower = allowed_group_dn.lower()

            # Check direct membership
            if allowed_group_lower in member_of:
                conn.unbind()
                logger.debug(f"User {user.username} is member of {allowed_group_dn}")
                return True

            # Check primary group
            if primary_group_id:
                conn.search(
                    search_base=allowed_group_dn,
                    search_filter="(objectClass=*)",
                    search_scope="BASE",
                    attributes=["primaryGroupToken"],
                )
                if conn.entries:
                    group_entry = conn.entries[0]
                    if (
                        hasattr(group_entry, "primaryGroupToken")
                        and group_entry.primaryGroupToken
                    ):
                        group_rid = int(str(group_entry.primaryGroupToken))
                        if group_rid == primary_group_id:
                            conn.unbind()
                            logger.debug(
                                f"User {user.username}'s primary group (RID {primary_group_id}) "
                                f"matches {allowed_group_dn}"
                            )
                            return True

            conn.unbind()
            logger.debug(f"User {user.username} is NOT a member of {allowed_group_dn}")
            return False

        except Exception as e:
            logger.debug(f"LDAP group check failed: {e}")
            if conn.bound:
                conn.unbind()
            return False

    except Exception as e:
        logger.debug(f"OAuth2 token validation failed: {e}")
        return False


async def _get_user_matching_filter(scope: Scope) -> str | None:
    """
    Get the ID of the highest-priority default route filter matching the user's LDAP groups.

    This function:
    1. Validates the OAuth2 Bearer token to get the user
    2. Queries the user's LDAP group memberships
    3. Finds enabled default route filters whose LDAP group matches
    4. Returns the ID of the highest priority matching filter

    Args:
        scope: ASGI scope containing the Authorization header

    Returns:
        Filter ID if a matching filter is found, None otherwise (show all tools)
    """
    from ragtime.core.auth import validate_session
    from ragtime.core.database import get_db

    logger.debug("_get_user_matching_filter: Starting filter check")

    # Get Authorization header
    headers = dict(scope.get("headers", []))
    auth_header = headers.get(b"authorization", b"").decode()

    if not auth_header.startswith("Bearer "):
        logger.debug("_get_user_matching_filter: No Bearer token found")
        return None

    token = auth_header[7:]  # Remove "Bearer " prefix

    try:
        token_data = await validate_session(token)
        if token_data is None:
            logger.debug("_get_user_matching_filter: Token validation failed")
            return None

        db = await get_db()
        user = await db.user.find_unique(where={"id": token_data.user_id})
        if not user:
            logger.debug("_get_user_matching_filter: User not found in database")
            return None

        logger.debug(
            f"_get_user_matching_filter: User={user.username}, provider={user.authProvider}, role={user.role}"
        )

        # Local admin users bypass filtering - see all tools
        if user.authProvider == "local" and user.role == "admin":
            logger.debug(
                f"Local admin '{user.username}' bypasses default route filtering"
            )
            return None

        # Get user's LDAP groups
        if not user.ldapDn:
            logger.debug(
                f"_get_user_matching_filter: User {user.username} has no LDAP DN"
            )
            # Non-LDAP users see all tools (no filtering)
            return None

        logger.debug(f"_get_user_matching_filter: User LDAP DN={user.ldapDn}")

        # Get all enabled default route filters, ordered by priority descending
        filters = await db.mcpdefaultroutefilter.find_many(
            where={"enabled": True},
            order={"priority": "desc"},
        )

        logger.debug(f"_get_user_matching_filter: Found {len(filters)} enabled filters")

        if not filters:
            # No filters configured - show all tools
            return None

        # Get LDAP connection to check group membership
        from ragtime.core.auth import _get_ldap_connection, get_ldap_config
        from ragtime.core.encryption import decrypt_secret

        ldap_config = await get_ldap_config()
        if not ldap_config.serverUrl:
            logger.debug("_get_user_matching_filter: No LDAP server URL configured")
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
            # Get user's memberOf and primaryGroupID
            conn.search(
                search_base=user.ldapDn,
                search_filter="(objectClass=*)",
                search_scope="BASE",
                attributes=["memberOf", "primaryGroupID"],
            )

            if not conn.entries:
                conn.unbind()
                return None

            entry = conn.entries[0]
            user_groups: set[str] = set()
            if hasattr(entry, "memberOf") and entry.memberOf:
                user_groups = {str(g).lower() for g in entry.memberOf}

            primary_group_id = None
            if hasattr(entry, "primaryGroupID") and entry.primaryGroupID:
                primary_group_id = int(str(entry.primaryGroupID))

            # Check each filter in priority order
            for f in filters:
                filter_group_dn = f.ldapGroupDn.lower()

                # Check direct membership
                if filter_group_dn in user_groups:
                    conn.unbind()
                    logger.debug(
                        f"User {user.username} matches filter '{f.name}' "
                        f"via group {f.ldapGroupDn}"
                    )
                    return f.id

                # Check primary group
                if primary_group_id:
                    try:
                        conn.search(
                            search_base=f.ldapGroupDn,
                            search_filter="(objectClass=*)",
                            search_scope="BASE",
                            attributes=["primaryGroupToken"],
                        )
                        if conn.entries:
                            group_entry = conn.entries[0]
                            if (
                                hasattr(group_entry, "primaryGroupToken")
                                and group_entry.primaryGroupToken
                            ):
                                group_rid = int(str(group_entry.primaryGroupToken))
                                if group_rid == primary_group_id:
                                    conn.unbind()
                                    logger.debug(
                                        f"User {user.username} matches filter '{f.name}' "
                                        f"via primary group"
                                    )
                                    return f.id
                    except Exception:
                        # Continue to next filter if this one fails
                        pass

            conn.unbind()
            logger.debug(f"User {user.username} has no matching default route filter")
            return None

        except Exception as e:
            logger.debug(f"LDAP group check for filter matching failed: {e}")
            if conn.bound:
                conn.unbind()
            return None

    except Exception as e:
        logger.debug(f"Failed to get matching filter for user: {e}")
        return None


async def _validate_route_password(scope: Scope, encrypted_password: str) -> bool:
    """
    Validate password from Authorization header or MCP-Password header.

    Accepts:
    - Authorization: Bearer <password>
    - MCP-Password: <password>  (preferred for VS Code to avoid OAuth prompts)

    Used for custom MCP routes with authentication enabled.

    Args:
        scope: ASGI scope
        encrypted_password: The Fernet-encrypted password

    Returns:
        True if valid, False otherwise
    """
    from ragtime.core.encryption import decrypt_secret

    # Get headers
    headers = dict(scope.get("headers", []))

    # Try MCP-Password header first (avoids VS Code OAuth prompts)
    mcp_password = headers.get(b"mcp-password", b"").decode()
    if mcp_password:
        provided_password = mcp_password
    else:
        # Fall back to Authorization: Bearer <password>
        auth_header = headers.get(b"authorization", b"").decode()
        if not auth_header.startswith("Bearer "):
            return False
        provided_password = auth_header[7:]  # Remove "Bearer " prefix

    # Decrypt the stored password and compare
    stored_password = decrypt_secret(encrypted_password)
    if not stored_password:
        # Decryption failed (key changed?)
        logger.warning("Failed to decrypt route password - key may have changed")
        return False

    return provided_password == stored_password


async def _check_default_route_auth() -> tuple[bool, str, str | None, str | None]:
    """
    Check if default /mcp route requires authentication and get auth configuration.

    Returns:
        Tuple of (require_auth, auth_method, encrypted_password, allowed_group_dn)
    """
    app_settings = await get_app_settings()
    require_auth = app_settings.get("mcp_default_route_auth", False)
    auth_method = app_settings.get("mcp_default_route_auth_method", "password")
    encrypted_password = app_settings.get("mcp_default_route_password")
    allowed_group = app_settings.get("mcp_default_route_allowed_group")
    return require_auth, auth_method, encrypted_password, allowed_group


async def _check_mcp_enabled() -> bool:
    """Check if MCP server is enabled in database settings."""
    app_settings = await get_app_settings()
    return app_settings.get("mcp_enabled", False)


class MCPTransportEndpoint:
    """
    ASGI endpoint for MCP Streamable HTTP transport.

    This wraps the StreamableHTTPSessionManager as a proper ASGI app
    to avoid double-response issues with FastAPI's api_route decorator.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle MCP protocol request by delegating to session manager."""
        # Check if MCP is enabled
        if not await _check_mcp_enabled():
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [[b"content-type", b"application/json"]],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error": "MCP server is disabled. Enable it in Settings."}',
                }
            )
            return

        # Check if default route requires auth and get auth config
        require_auth, auth_method, encrypted_password, allowed_group = (
            await _check_default_route_auth()
        )

        if require_auth:
            is_valid = False

            if auth_method == "oauth2":
                # OAuth2 with LDAP - validate token and check group membership
                is_valid = await _validate_oauth2_token(scope, allowed_group)
            elif encrypted_password:
                # Password-based auth
                is_valid = await _validate_route_password(scope, encrypted_password)
            else:
                # Fall back to session token validation
                is_valid = await _validate_bearer_token(scope)

            if not is_valid:
                # Send 401 Unauthorized
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"www-authenticate", b'Bearer realm="mcp", MCP-Password'),
                        ],
                    }
                )
                if auth_method == "oauth2":
                    detail = "OAuth2 Bearer token required - token invalid, expired, or user not authorized"
                elif encrypted_password:
                    detail = "Password required (use MCP-Password header or Authorization: Bearer)"
                else:
                    detail = "Bearer token required"
                await send(
                    {
                        "type": "http.response.body",
                        "body": f'{{"error": "Unauthorized", "detail": "{detail}"}}'.encode(),
                    }
                )
                return

        # If using OAuth2 auth, check for LDAP group-based tool filtering
        if require_auth and auth_method == "oauth2":
            # Try to find a matching default route filter for this user
            matching_filter_id = await _get_user_matching_filter(scope)
            if matching_filter_id:
                # Use filtered server directly (bypasses session manager issue)
                filtered_server = await get_filtered_server(matching_filter_id)
                if filtered_server:
                    await handle_filtered_request(filtered_server, scope, receive, send)
                    return
                # Filter found but couldn't create server - fall back to default

        # Use the default session manager
        await get_session_manager().handle_request(scope, receive, send)


class MCPCustomRouteEndpoint:
    """
    ASGI endpoint for custom MCP routes.

    Handles requests to /mcp/{route_path} with optional authentication.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle MCP protocol request for a custom route."""
        # Check if MCP is enabled
        if not await _check_mcp_enabled():
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [[b"content-type", b"application/json"]],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error": "MCP server is disabled. Enable it in Settings."}',
                }
            )
            return

        # Extract route_path from path
        path = scope.get("path", "")
        # Path format: /mcp/{route_path}
        if path.startswith("/mcp/"):
            route_path = path[5:]  # Remove "/mcp/" prefix
            # Remove any trailing slashes or query params
            if "?" in route_path:
                route_path = route_path.split("?")[0]
            route_path = route_path.rstrip("/")
        else:
            route_path = ""

        if not route_path:
            # Send 404 Not Found
            await send(
                {
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error": "Not Found", "detail": "Route path required"}',
                }
            )
            return

        # Get session manager for this route
        result = await get_custom_session_manager(route_path)
        if not result:
            await send(
                {
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error": "Not Found", "detail": "Custom MCP route not found or disabled"}',
                }
            )
            return

        session_manager, require_auth, auth_password, auth_method, allowed_group = (
            result
        )

        # Check authentication if required
        if require_auth:
            is_valid = False

            if auth_method == "oauth2":
                # OAuth2 with LDAP - validate token and check group membership
                is_valid = await _validate_oauth2_token(scope, allowed_group)
            elif auth_password:
                # Password-based auth
                is_valid = await _validate_route_password(scope, auth_password)
            else:
                # Auth required but no method configured
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"www-authenticate", b'Bearer realm="mcp"'),
                        ],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"error": "Unauthorized", "detail": "Authentication required but not configured"}',
                    }
                )
                return

            if not is_valid:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"www-authenticate", b'Bearer realm="mcp"'),
                        ],
                    }
                )
                if auth_method == "oauth2":
                    detail = "OAuth2 Bearer token required - token invalid, expired, or user not authorized"
                else:
                    detail = "Invalid password"
                await send(
                    {
                        "type": "http.response.body",
                        "body": f'{{"error": "Unauthorized", "detail": "{detail}"}}'.encode(),
                    }
                )
                return

        await session_manager.handle_request(scope, receive, send)


# MCP Transport Route - uses ASGI app directly to avoid double response
# This is mounted as a Route instead of using APIRouter to properly handle
# the session manager's direct ASGI response
mcp_transport_route = Route(
    "/mcp",
    endpoint=MCPTransportEndpoint(),
    methods=["GET", "POST", "DELETE", "OPTIONS"],
)

# Custom route endpoint - catches /mcp/{anything}
mcp_custom_route = Route(
    "/mcp/{route_path:path}",
    endpoint=MCPCustomRouteEndpoint(),
    methods=["GET", "POST", "DELETE", "OPTIONS"],
)


def get_mcp_routes() -> list[Route]:
    """Get all MCP-related routes for mounting in the app."""
    return [mcp_transport_route, mcp_custom_route]


@router.get("/tools")
async def list_mcp_tools(include_unhealthy: bool = False):
    """
    List available MCP tools (debug endpoint).

    This is a convenience endpoint for debugging/testing.
    The actual MCP protocol uses the HTTP transport at /mcp.

    Args:
        include_unhealthy: If True, include tools that failed heartbeat check
    """
    tools = await mcp_tool_adapter.get_available_tools(
        include_unhealthy=include_unhealthy
    )

    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                "tool_type": t.tool_config.get("tool_type"),
            }
            for t in tools
        ],
        "count": len(tools),
    }


@router.post("/tools/{tool_name}/call")
async def call_mcp_tool(tool_name: str, request: Request):
    """
    Call an MCP tool directly (debug endpoint).

    This is a convenience endpoint for debugging/testing.
    The actual MCP protocol uses the HTTP transport at /mcp.
    """
    body = await request.json()
    arguments = body.get("arguments", {})

    result = await mcp_tool_adapter.execute_tool(tool_name, arguments)

    return {
        "tool": tool_name,
        "result": result,
    }


@router.post("/invalidate-cache")
async def invalidate_mcp_cache():
    """
    Invalidate the MCP tool cache.

    Forces a fresh heartbeat check on the next tool list request.
    Useful after adding/modifying tool configurations.
    """
    mcp_tool_adapter.invalidate_cache()
    # Also clear custom session managers so they're recreated
    _custom_session_managers.clear()
    # Clear default filter server cache
    _default_filter_servers.clear()
    return {"status": "cache_invalidated"}


@router.get("/health")
async def mcp_health():
    """
    MCP server health check.

    Returns the number of available tools and their status.
    """
    try:
        all_tools = await mcp_tool_adapter.get_available_tools(include_unhealthy=True)
        healthy_tools = await mcp_tool_adapter.get_available_tools(
            include_unhealthy=False
        )

        return {
            "status": "healthy",
            "total_tools": len(all_tools),
            "healthy_tools": len(healthy_tools),
            "unhealthy_tools": len(all_tools) - len(healthy_tools),
        }
    except Exception as e:
        logger.exception(f"MCP health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


@router.get("/routes")
async def list_mcp_routes_debug():
    """
    List all custom MCP routes (debug endpoint).

    Returns configured custom routes and their settings.
    """
    from ragtime.core.database import get_db

    db = await get_db()
    routes = await db.mcprouteconfig.find_many(
        order={"createdAt": "desc"},
        include={"toolSelections": True},
    )

    return {
        "routes": [
            {
                "name": r.name,
                "path": f"/mcp/{r.routePath}",
                "enabled": r.enabled,
                "require_auth": r.requireAuth,
                "tool_count": len(r.toolSelections) if r.toolSelections else 0,
                "include_knowledge_search": r.includeKnowledgeSearch,
                "include_git_history": r.includeGitHistory,
            }
            for r in routes
        ],
        "count": len(routes),
    }
