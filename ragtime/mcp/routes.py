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

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
from anyio.abc import TaskStatus
from fastapi import APIRouter, Depends, Request
from mcp.server.lowlevel.server import Server as MCPServer
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from prisma.models import User
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from ragtime.core.app_settings import get_app_settings
from ragtime.core.auth import user_matches_group_identifier, validate_session
from ragtime.core.database import get_db
from ragtime.core.encryption import decrypt_secret
from ragtime.core.logging import get_logger
from ragtime.core.mcp_accounting import log_mcp_request
from ragtime.core.security import require_admin
from ragtime.mcp.oauth import (
    handle_authorization_server_metadata,
    handle_protected_resource_metadata,
    handle_token_request,
    validate_client_credentials_basic,
    validate_client_credentials_bearer,
)
from ragtime.mcp.server import (
    get_custom_route_server,
    get_default_route_filtered_server,
    get_mcp_server,
    notify_tools_changed,
    register_tools_changed_callback,
)
from ragtime.mcp.tools import mcp_tool_adapter

logger = get_logger(__name__)

router = APIRouter(prefix="/mcp-debug", tags=["MCP Debug"])

# Session manager for Streamable HTTP transport (default route)
# Stateless mode with JSON responses for maximum compatibility
_mcp_http_state: dict[str, Any] = {
    "session_manager": None,
}

# Cached custom route MCP servers (keyed by route path)
# We use the low-level server directly since StreamableHTTPSessionManager
# requires run() to be called before handling requests
_custom_route_servers: dict[str, MCPServer[Any, Any]] = {}

# Cached filtered MCP servers (keyed by filter ID)
# We use the low-level server directly for filtered requests since
# StreamableHTTPSessionManager requires run() to be called before handling requests
_default_filter_servers: dict[str, MCPServer[Any, Any]] = {}


def invalidate_http_route_cache() -> None:
    """Invalidate HTTP MCP route server caches."""
    _custom_route_servers.clear()
    _default_filter_servers.clear()


# Ensure notify_tools_changed() also clears HTTP route-level caches.
register_tools_changed_callback(invalidate_http_route_cache)


async def get_session_manager() -> StreamableHTTPSessionManager:
    """Get or create the MCP session manager for the default route."""
    session_manager = _mcp_http_state["session_manager"]

    if session_manager is None:
        default_server = await get_mcp_server()
        session_manager = StreamableHTTPSessionManager(
            app=default_server,
            event_store=None,  # Stateless mode - no event persistence
            json_response=True,  # JSON responses for broad client compatibility
            stateless=True,  # Stateless mode - each request is independent
        )
        _mcp_http_state["session_manager"] = session_manager
        logger.info("Created MCP StreamableHTTP session manager (stateless mode)")
    return session_manager


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


async def get_custom_route_server_cached(
    route_path: str,
) -> (
    tuple[MCPServer[Any, Any], bool, str | None, str | None, str | None, str | None]
    | None
):
    """
    Get or create an MCP server for a custom route.

    Args:
        route_path: The route path suffix

    Returns:
        Tuple of (server, require_auth, auth_password, auth_method, allowed_group,
        auth_client_id) or None if route not found
    """
    # Get route config (always need fresh auth info)
    result = await get_custom_route_server(route_path)
    if not result:
        return None

    (
        server,
        _,
        require_auth,
        auth_password,
        auth_method,
        allowed_group,
        auth_client_id,
    ) = result

    # Cache the server if not already cached
    if route_path not in _custom_route_servers:
        _custom_route_servers[route_path] = server
        logger.info(f"Cached MCP server for custom route: /mcp/{route_path}")

    return (
        _custom_route_servers[route_path],
        require_auth,
        auth_password,
        auth_method,
        allowed_group,
        auth_client_id,
    )


@asynccontextmanager
async def mcp_lifespan_manager() -> AsyncIterator[None]:
    """Lifespan manager for MCP session manager."""
    session_manager = await get_session_manager()
    async with session_manager.run():
        logger.info("MCP StreamableHTTP session manager started")
        try:
            yield
        finally:
            # Clear custom route server cache
            for path in _custom_route_servers:
                logger.debug(f"Clearing custom route server cache: {path}")
            _custom_route_servers.clear()
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
    # Get Authorization header
    headers = dict(scope.get("headers", []))
    auth_header = headers.get(b"authorization", b"").decode()

    if not auth_header.startswith("Bearer "):
        return False

    token = auth_header[7:]  # Remove "Bearer " prefix

    try:
        token_data = await validate_session(token)
        if token_data is not None:
            # Annotate scope for downstream MCP request logging
            scope["_mcp_user_id"] = token_data.user_id
            return True
        return False
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

        # Annotate scope for downstream MCP request logging
        scope["_mcp_user_id"] = token_data.user_id

        # If no group restriction, token is valid
        if not allowed_group_dn:
            return True

        # Check if user is in the allowed provider group. LDAP users are checked
        # live when a DN is supplied; internal/local groups use cached memberships.
        db = await get_db()
        user = await db.user.find_unique(where={"id": token_data.user_id})
        if not user:
            logger.debug(f"User {token_data.user_id} not found in database")
            return False
        matched = await user_matches_group_identifier(user, allowed_group_dn)
        if matched:
            logger.debug(f"User {user.username} matches group {allowed_group_dn}")
        else:
            logger.debug(
                f"User {user.username} does not match group {allowed_group_dn}"
            )
        return matched

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
        if (
            str(getattr(user.authProvider, "value", user.authProvider)) == "local"
            and user.role == "admin"
        ):
            logger.debug(
                f"Local admin '{user.username}' bypasses default route filtering"
            )
            return None

        # Get all enabled default route filters, ordered by priority descending
        filters = await db.mcpdefaultroutefilter.find_many(
            where={"enabled": True},
            order={"priority": "desc"},
        )

        logger.debug(f"_get_user_matching_filter: Found {len(filters)} enabled filters")

        if not filters:
            # No filters configured - show all tools
            return None
        for f in filters:
            if await user_matches_group_identifier(user, f.ldapGroupDn):
                logger.debug(
                    f"User {user.username} matches filter '{f.name}' via group {f.ldapGroupDn}"
                )
                return f.id

        logger.debug(f"User {user.username} has no matching default route filter")
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

    return hmac.compare_digest(provided_password, stored_password)


async def _check_default_route_auth() -> (
    tuple[bool, str, str | None, str | None, str | None]
):
    """
    Check if default /mcp route requires authentication and get auth configuration.

    Returns:
        Tuple of (require_auth, auth_method, encrypted_password, allowed_group_dn,
        client_id)
    """
    app_settings = await get_app_settings()
    require_auth = app_settings.get("mcp_default_route_auth", False)
    auth_method = app_settings.get("mcp_default_route_auth_method", "password")
    encrypted_password = app_settings.get("mcp_default_route_password")
    allowed_group = app_settings.get("mcp_default_route_allowed_group")
    client_id = app_settings.get("mcp_default_route_client_id")
    return require_auth, auth_method, encrypted_password, allowed_group, client_id


async def _check_mcp_enabled() -> bool:
    """Check if MCP server is enabled in database settings."""
    app_settings = await get_app_settings()
    return app_settings.get("mcp_enabled", False)


async def _log_mcp(
    scope: Scope,
    *,
    route_name: str = "default",
    auth_method: str = "none",
    status_code: int = 200,
) -> None:
    """Fire-and-forget MCP request log entry."""
    http_method = str(scope.get("method", "POST")).upper()
    if http_method == "OPTIONS":
        # CORS preflight requests are transport noise, not MCP protocol usage.
        return

    user_id = scope.get("_mcp_user_id")
    username = scope.get("_mcp_username")
    await log_mcp_request(
        user_id=user_id,
        username=username,
        route_name=route_name,
        auth_method=auth_method,
        http_method=http_method,
        status_code=status_code,
    )


def _wrap_send_for_status(
    send: Send,
) -> tuple[Send, dict[str, int | None]]:
    """Wrap ASGI send to capture the final response status code."""
    response_state: dict[str, int | None] = {"status_code": None}

    async def send_with_status(message: dict[str, Any]) -> None:
        if message.get("type") == "http.response.start":
            response_state["status_code"] = int(message.get("status", 200))
        await send(message)

    return send_with_status, response_state


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
            await _log_mcp(scope, status_code=503)
            return

        # Check if default route requires auth and get auth config
        require_auth, auth_method, encrypted_password, allowed_group, client_id = (
            await _check_default_route_auth()
        )

        resolved_auth_method = "none"
        if require_auth:
            is_valid = False

            if auth_method == "oauth2":
                resolved_auth_method = "oauth2"
                # OAuth2 with LDAP - validate token and check group membership
                is_valid = await _validate_oauth2_token(scope, allowed_group)
            elif auth_method == "client_credentials":
                resolved_auth_method = "client_credentials"
                # Accept either direct Basic auth or a Bearer token issued by
                # the client_credentials token endpoint for the default route.
                is_valid = validate_client_credentials_bearer(scope, None)
                if not is_valid:
                    is_valid = await validate_client_credentials_basic(
                        scope, client_id, encrypted_password
                    )
            elif encrypted_password:
                resolved_auth_method = "password"
                # Password-based auth
                is_valid = await _validate_route_password(scope, encrypted_password)
            else:
                resolved_auth_method = "bearer"
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
                elif auth_method == "client_credentials":
                    detail = (
                        "OAuth2 client_credentials required - provide client_id/client_secret "
                        "via HTTP Basic auth or exchange them for a Bearer token at /mcp/oauth/token"
                    )
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
                await _log_mcp(scope, auth_method=resolved_auth_method, status_code=401)
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
                    await _log_mcp(scope, auth_method=resolved_auth_method)
                    return
                # Filter found but couldn't create server - fall back to default

        # Use the default session manager
        session_manager = await get_session_manager()
        send_with_status, response_state = _wrap_send_for_status(send)
        try:
            await session_manager.handle_request(scope, receive, send_with_status)
        except Exception:
            await _log_mcp(
                scope,
                auth_method=resolved_auth_method,
                status_code=response_state["status_code"] or 500,
            )
            raise
        await _log_mcp(
            scope,
            auth_method=resolved_auth_method,
            status_code=response_state["status_code"] or 200,
        )


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
            await _log_mcp(scope, status_code=503)
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

        # Handle OAuth2 helper sub-paths for either the default or custom routes.
        # These are matched by the catch-all; strip the suffix to locate the
        # owning route (empty = default /mcp).
        if route_path in ("oauth/token",):
            await handle_token_request(scope, receive, send, None)
            return
        if route_path in (
            ".well-known/oauth-authorization-server",
            ".well-known/oauth-protected-resource",
        ):
            if route_path.endswith("oauth-authorization-server"):
                await handle_authorization_server_metadata(scope, receive, send, None)
            else:
                await handle_protected_resource_metadata(scope, receive, send, None)
            return
        if route_path.endswith("/oauth/token"):
            base_route = route_path[: -len("/oauth/token")]
            await handle_token_request(scope, receive, send, base_route or None)
            return
        if route_path.endswith("/.well-known/oauth-authorization-server"):
            base_route = route_path[: -len("/.well-known/oauth-authorization-server")]
            await handle_authorization_server_metadata(
                scope, receive, send, base_route or None
            )
            return
        if route_path.endswith("/.well-known/oauth-protected-resource"):
            base_route = route_path[: -len("/.well-known/oauth-protected-resource")]
            await handle_protected_resource_metadata(
                scope, receive, send, base_route or None
            )
            return

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
            await _log_mcp(scope, status_code=404)
            return

        # Get server for this route
        result = await get_custom_route_server_cached(route_path)
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
            await _log_mcp(scope, route_name=route_path, status_code=404)
            return

        (
            server,
            require_auth,
            auth_password,
            auth_method,
            allowed_group,
            auth_client_id,
        ) = result

        # Check authentication if required
        resolved_auth_method = "none"
        if require_auth:
            is_valid = False

            if auth_method == "oauth2":
                resolved_auth_method = "oauth2"
                # OAuth2 with LDAP - validate token and check group membership
                is_valid = await _validate_oauth2_token(scope, allowed_group)
            elif auth_method == "client_credentials":
                resolved_auth_method = "client_credentials"
                # Accept either direct Basic auth or a Bearer token issued by
                # the client_credentials token endpoint for this route.
                is_valid = validate_client_credentials_bearer(scope, route_path)
                if not is_valid:
                    is_valid = await validate_client_credentials_basic(
                        scope, auth_client_id, auth_password
                    )
            elif auth_password:
                resolved_auth_method = "password"
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
                await _log_mcp(scope, route_name=route_path, status_code=401)
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
                elif auth_method == "client_credentials":
                    detail = (
                        "OAuth2 client_credentials required - provide client_id/client_secret "
                        f"via HTTP Basic auth or exchange them for a Bearer token at "
                        f"/mcp/{route_path}/oauth/token"
                    )
                else:
                    detail = "Invalid password"
                await send(
                    {
                        "type": "http.response.body",
                        "body": f'{{"error": "Unauthorized", "detail": "{detail}"}}'.encode(),
                    }
                )
                await _log_mcp(
                    scope,
                    route_name=route_path,
                    auth_method=resolved_auth_method,
                    status_code=401,
                )
                return

        # Handle request directly with the server (bypasses session manager)
        send_with_status, response_state = _wrap_send_for_status(send)
        try:
            await handle_filtered_request(server, scope, receive, send_with_status)
        except Exception:
            await _log_mcp(
                scope,
                route_name=route_path,
                auth_method=resolved_auth_method,
                status_code=response_state["status_code"] or 500,
            )
            raise
        await _log_mcp(
            scope,
            route_name=route_path,
            auth_method=resolved_auth_method,
            status_code=response_state["status_code"] or 200,
        )


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


class _OAuthTokenEndpoint:
    """ASGI endpoint wrapping ``handle_token_request`` for default/custom routes."""

    def __init__(self, *, default: bool) -> None:
        self._default = default

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        route_path = (
            None if self._default else scope.get("path_params", {}).get("route_path")
        )
        await handle_token_request(scope, receive, send, route_path)


class _OAuthASMetadataEndpoint:
    """ASGI endpoint wrapping ``handle_authorization_server_metadata``."""

    def __init__(self, *, default: bool) -> None:
        self._default = default

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        route_path = _normalize_well_known_route_path(
            None if self._default else scope.get("path_params", {}).get("route_path")
        )
        await handle_authorization_server_metadata(scope, receive, send, route_path)


class _OAuthPRMetadataEndpoint:
    """ASGI endpoint wrapping ``handle_protected_resource_metadata``."""

    def __init__(self, *, default: bool) -> None:
        self._default = default

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        route_path = (
            None if self._default else scope.get("path_params", {}).get("route_path")
        )
        await handle_protected_resource_metadata(scope, receive, send, route_path)


def _normalize_well_known_route_path(route_path: str | None) -> str | None:
    """Normalize RFC 8414 path-form issuer suffixes to an MCP route path."""
    normalized = (route_path or "").strip("/")
    if not normalized or normalized == "mcp":
        return None
    if normalized.startswith("mcp/"):
        return normalized[len("mcp/") :]
    return normalized


# OAuth 2.0 token + discovery routes for MCP client_credentials auth.
# Order matters: these must be registered BEFORE ``mcp_custom_route`` so the
# greedy ``/mcp/{route_path:path}`` pattern doesn't swallow them.
mcp_default_token_route = Route(
    "/mcp/oauth/token",
    endpoint=_OAuthTokenEndpoint(default=True),
    methods=["POST", "OPTIONS"],
)
mcp_custom_token_route = Route(
    "/mcp/{route_path:path}/oauth/token",
    endpoint=_OAuthTokenEndpoint(default=False),
    methods=["POST", "OPTIONS"],
)
mcp_custom_as_metadata_route = Route(
    "/mcp/{route_path:path}/.well-known/oauth-authorization-server",
    endpoint=_OAuthASMetadataEndpoint(default=False),
    methods=["GET", "OPTIONS"],
)
mcp_prefixed_as_metadata_route = Route(
    "/.well-known/oauth-authorization-server/{route_path:path}",
    endpoint=_OAuthASMetadataEndpoint(default=False),
    methods=["GET", "OPTIONS"],
)
mcp_prefixed_oidc_metadata_route = Route(
    "/.well-known/openid-configuration/{route_path:path}",
    endpoint=_OAuthASMetadataEndpoint(default=False),
    methods=["GET", "OPTIONS"],
)
mcp_custom_oidc_metadata_route = Route(
    "/mcp/{route_path:path}/.well-known/openid-configuration",
    endpoint=_OAuthASMetadataEndpoint(default=False),
    methods=["GET", "OPTIONS"],
)
mcp_custom_pr_metadata_route = Route(
    "/mcp/{route_path:path}/.well-known/oauth-protected-resource",
    endpoint=_OAuthPRMetadataEndpoint(default=False),
    methods=["GET", "OPTIONS"],
)


def get_mcp_routes() -> list[Route]:
    """Get all MCP-related routes for mounting in the app."""
    return [
        # OAuth helper endpoints registered before the greedy custom route
        # so that ``/mcp/{path}/oauth/token`` etc. aren't swallowed.
        mcp_default_token_route,
        mcp_custom_token_route,
        mcp_custom_as_metadata_route,
        mcp_prefixed_as_metadata_route,
        mcp_prefixed_oidc_metadata_route,
        mcp_custom_oidc_metadata_route,
        mcp_custom_pr_metadata_route,
        mcp_transport_route,
        mcp_custom_route,
    ]


@router.get("/tools")
async def list_mcp_tools(
    include_unhealthy: bool = False, _user: User = Depends(require_admin)
):
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
async def call_mcp_tool(
    tool_name: str, request: Request, _user: User = Depends(require_admin)
):
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
async def invalidate_mcp_cache(_user: User = Depends(require_admin)):
    """
    Invalidate the MCP tool cache.

    Forces a fresh heartbeat check on the next tool list request.
    Useful after adding/modifying tool configurations.
    """
    notify_tools_changed()
    return {"status": "cache_invalidated"}


@router.get("/health")
async def mcp_health(_user: User = Depends(require_admin)):
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
async def list_mcp_routes_debug(_user: User = Depends(require_admin)):
    """
    List all custom MCP routes (debug endpoint).

    Returns configured custom routes and their settings.
    """
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
