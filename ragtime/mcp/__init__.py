"""
MCP (Model Context Protocol) Server Module.

Exposes Ragtime tools as an MCP server for use by MCP clients like Claude Desktop.
Tools are dynamically discovered from the ToolConfig database and filtered by
heartbeat status to only expose healthy, reachable tools.

Supports custom MCP routes with tool selection and optional authentication.

Usage:
    # Stdio transport (for Claude Desktop)
    python -m ragtime.mcp

    # SSE transport (mounted in FastAPI, see main.py)
    # Available at /mcp endpoint when enabled
    # Custom routes available at /mcp/{route_path}
"""

from ragtime.mcp.server import get_custom_route_server, mcp_server, run_mcp_server
from ragtime.mcp.tools import McpRouteFilter, MCPToolAdapter

__all__ = [
    "mcp_server",
    "run_mcp_server",
    "MCPToolAdapter",
    "McpRouteFilter",
    "get_custom_route_server",
]
