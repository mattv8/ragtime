"""
MCP HTTP Transport Routes - Streamable HTTP transport for MCP.

Provides HTTP endpoints for MCP clients using the Streamable HTTP transport.
This allows MCP clients to connect via URL configuration like:

    "url": "http://localhost:8000/mcp"              # Default route (all tools)
    "url": "http://localhost:8000/mcp/my_toolset"   # Custom route (subset)

Supports both SSE streaming responses and JSON responses for stateless operation.
Custom routes support optional Bearer token authentication.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Request
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from ragtime.core.app_settings import get_app_settings
from ragtime.core.logging import get_logger
from ragtime.mcp.server import get_custom_route_server, mcp_server
from ragtime.mcp.tools import mcp_tool_adapter

logger = get_logger(__name__)

router = APIRouter(prefix="/mcp-debug", tags=["MCP Debug"])

# Session manager for Streamable HTTP transport (default route)
# Stateless mode with JSON responses for maximum compatibility
_session_manager: StreamableHTTPSessionManager | None = None

# Session managers for custom routes (created on demand)
_custom_session_managers: dict[str, StreamableHTTPSessionManager] = {}


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


async def get_custom_session_manager(
    route_path: str,
) -> tuple[StreamableHTTPSessionManager, bool, str | None] | None:
    """
    Get or create a session manager for a custom route.

    Args:
        route_path: The route path suffix

    Returns:
        Tuple of (session_manager, require_auth, auth_password_hash) or None if route not found
    """
    # Check cache
    if route_path in _custom_session_managers:
        # Get require_auth and password from route config
        result = await get_custom_route_server(route_path)
        if result:
            _, _, require_auth, auth_password = result
            return _custom_session_managers[route_path], require_auth, auth_password
        return None

    # Get or create the server
    result = await get_custom_route_server(route_path)
    if not result:
        return None

    server, _, require_auth, auth_password = result

    # Create session manager
    manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=True,
        stateless=True,
    )
    _custom_session_managers[route_path] = manager
    logger.info(f"Created custom MCP session manager for route: /mcp/{route_path}")

    return manager, require_auth, auth_password


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


async def _check_default_route_auth() -> tuple[bool, str | None]:
    """
    Check if default /mcp route requires authentication and get the encrypted password.

    Returns:
        Tuple of (require_auth, encrypted_password)
    """
    app_settings = await get_app_settings()
    require_auth = app_settings.get("mcp_default_route_auth", False)
    encrypted_password = app_settings.get("mcp_default_route_password")
    return require_auth, encrypted_password


class MCPTransportEndpoint:
    """
    ASGI endpoint for MCP Streamable HTTP transport.

    This wraps the StreamableHTTPSessionManager as a proper ASGI app
    to avoid double-response issues with FastAPI's api_route decorator.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle MCP protocol request by delegating to session manager."""
        # Check if default route requires auth and get password
        require_auth, encrypted_password = await _check_default_route_auth()

        if require_auth:
            # Determine auth method - password or session token
            if encrypted_password:
                # Use password-based auth
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
                detail = (
                    "Password required (use MCP-Password header or Authorization: Bearer)"
                    if encrypted_password
                    else "Bearer token required"
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": f'{{"error": "Unauthorized", "detail": "{detail}"}}'.encode(),
                    }
                )
                return

        session_manager = get_session_manager()
        await session_manager.handle_request(scope, receive, send)


class MCPCustomRouteEndpoint:
    """
    ASGI endpoint for custom MCP routes.

    Handles requests to /mcp/{route_path} with optional authentication.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle MCP protocol request for a custom route."""
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

        session_manager, require_auth, auth_password = result

        # Check authentication if required
        if require_auth:
            # Validate against the route's password
            if not auth_password:
                # Auth required but no password set - deny access
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

            if not await _validate_route_password(scope, auth_password):
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
                        "body": b'{"error": "Unauthorized", "detail": "Invalid password"}',
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
