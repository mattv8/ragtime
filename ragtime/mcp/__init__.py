"""
MCP (Model Context Protocol) Server Module.

Exposes Ragtime tools as an MCP server for use by MCP clients like Claude Desktop.
Tools are dynamically discovered from the ToolConfig database and filtered by
heartbeat status to only expose healthy, reachable tools.

Usage:
    # Stdio transport (for Claude Desktop)
    python -m ragtime.mcp

    # SSE transport (mounted in FastAPI, see main.py)
    # Available at /mcp endpoint when enabled
"""

from ragtime.mcp.server import mcp_server, run_mcp_server
from ragtime.mcp.tools import MCPToolAdapter

__all__ = ["mcp_server", "run_mcp_server", "MCPToolAdapter"]
