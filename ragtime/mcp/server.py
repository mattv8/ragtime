"""
MCP Server - Model Context Protocol server for Ragtime.

Exposes Ragtime tools (postgres, odoo, ssh, filesystem search) via MCP protocol.
Supports both stdio (for Claude Desktop) and SSE (for web clients) transports.

Features:
- Dynamic tool discovery from ToolConfig database
- Heartbeat-based filtering (only exposes healthy tools)
- Automatic tool refresh on config changes
- Extensible architecture for adding new tool types
"""

import asyncio
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from ragtime.core.app_settings import get_app_settings
from ragtime.core.logging import get_logger
from ragtime.mcp.tools import mcp_tool_adapter

logger = get_logger(__name__)

# MCP server instance - created lazily with dynamic name
_mcp_server: Server | None = None


async def get_mcp_server() -> Server:
    """Get or create MCP server with dynamic name from settings."""
    global _mcp_server
    if _mcp_server is None:
        app_settings = await get_app_settings()
        server_name = app_settings.get("server_name", "Ragtime")
        _mcp_server = Server(server_name.lower().replace(" ", "-"))
        _register_handlers(_mcp_server)
        logger.info(f"Created MCP server with name: {server_name}")
    return _mcp_server


def _register_handlers(server: Server) -> None:
    """Register MCP protocol handlers on the server instance."""

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """
        List all available tools.

        Tools are dynamically discovered from the ToolConfig database.
        Only healthy tools (passing heartbeat check) are exposed.
        """
        tools: list[Tool] = []

        try:
            tool_definitions = await mcp_tool_adapter.get_available_tools()

            for tool_def in tool_definitions:
                tools.append(
                    Tool(
                        name=tool_def.name,
                        description=tool_def.description,
                        inputSchema=tool_def.input_schema,
                    )
                )

            logger.debug(f"MCP list_tools: exposing {len(tools)} tools")

        except Exception as e:
            logger.exception(f"Error listing MCP tools: {e}")
            # Return empty list on error - better than crashing

        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """
        Execute a tool and return the result.

        Args:
            name: Tool name (e.g., "query_production_db")
            arguments: Tool arguments from MCP client

        Returns:
            List containing a single TextContent with the result
        """
        logger.info(f"MCP call_tool: {name}")
        logger.debug(f"MCP call_tool arguments: {arguments}")

        try:
            result = await mcp_tool_adapter.execute_tool(name, arguments)

            return [TextContent(type="text", text=result)]

        except Exception as e:
            logger.exception(f"MCP tool execution error: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]


# For backward compatibility - create a default instance with handlers
mcp_server = Server("ragtime")
_register_handlers(mcp_server)


async def run_mcp_server(transport: str = "stdio") -> None:
    """
    Run the MCP server with the specified transport.

    Args:
        transport: "stdio" for Claude Desktop, "sse" for web clients
    """
    from ragtime.core.database import connect_db, disconnect_db
    from ragtime.rag import rag

    logger.info(f"Starting MCP server with {transport} transport")

    # Initialize database connection
    await connect_db()

    # Initialize RAG components (needed for knowledge search)
    try:
        await rag.initialize()
    except Exception as e:
        logger.warning(f"RAG initialization failed (knowledge search unavailable): {e}")

    try:
        if transport == "stdio":
            from mcp.server.stdio import stdio_server

            async with stdio_server() as (read_stream, write_stream):
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )
        else:
            raise ValueError(f"Unsupported transport: {transport}")
    finally:
        await disconnect_db()
        logger.info("MCP server stopped")


def main() -> None:
    """Entry point for running MCP server from command line."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Ragtime MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    args = parser.parse_args()

    # Set up basic logging for standalone mode
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # MCP uses stdout for protocol, logs go to stderr
    )

    try:
        asyncio.run(run_mcp_server(args.transport))
    except KeyboardInterrupt:
        logger.info("MCP server interrupted")
    except Exception as e:
        logger.exception(f"MCP server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
