"""
MCP SSE Routes - Server-Sent Events transport for MCP.

Provides HTTP endpoints for MCP clients that use SSE transport instead of stdio.
Useful for web-based MCP clients and testing.
"""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from ragtime.core.logging import get_logger
from ragtime.mcp.server import mcp_server
from ragtime.mcp.tools import mcp_tool_adapter

logger = get_logger(__name__)

router = APIRouter(prefix="/mcp", tags=["MCP"])


@router.get("/tools")
async def list_mcp_tools(include_unhealthy: bool = False):
    """
    List available MCP tools.

    This is a convenience endpoint for debugging/testing.
    The actual MCP protocol uses the SSE transport.

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
    Call an MCP tool directly.

    This is a convenience endpoint for debugging/testing.
    The actual MCP protocol uses the SSE transport.
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
