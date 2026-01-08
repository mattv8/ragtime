"""
MCP HTTP Transport Routes - Streamable HTTP transport for MCP.

Provides HTTP endpoints for MCP clients using the Streamable HTTP transport.
This allows MCP clients to connect via URL configuration like:

    "url": "http://localhost:8000/mcp"

Supports both SSE streaming responses and JSON responses for stateless operation.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Request
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from ragtime.core.logging import get_logger
from ragtime.mcp.server import mcp_server
from ragtime.mcp.tools import mcp_tool_adapter

logger = get_logger(__name__)

router = APIRouter(prefix="/mcp-debug", tags=["MCP Debug"])

# Session manager for Streamable HTTP transport
# Stateless mode with JSON responses for maximum compatibility
_session_manager: StreamableHTTPSessionManager | None = None


def get_session_manager() -> StreamableHTTPSessionManager:
    """Get or create the MCP session manager."""
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


@asynccontextmanager
async def mcp_lifespan_manager() -> AsyncIterator[None]:
    """Lifespan manager for MCP session manager."""
    session_manager = get_session_manager()
    async with session_manager.run():
        logger.info("MCP StreamableHTTP session manager started")
        try:
            yield
        finally:
            logger.info("MCP StreamableHTTP session manager stopped")


class MCPTransportEndpoint:
    """
    ASGI endpoint for MCP Streamable HTTP transport.

    This wraps the StreamableHTTPSessionManager as a proper ASGI app
    to avoid double-response issues with FastAPI's api_route decorator.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle MCP protocol request by delegating to session manager."""
        session_manager = get_session_manager()
        await session_manager.handle_request(scope, receive, send)


# MCP Transport Route - uses ASGI app directly to avoid double response
# This is mounted as a Route instead of using APIRouter to properly handle
# the session manager's direct ASGI response
mcp_transport_route = Route(
    "/mcp",
    endpoint=MCPTransportEndpoint(),
    methods=["GET", "POST", "DELETE", "OPTIONS"],
)


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
