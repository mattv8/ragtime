"""
MCP Server - Model Context Protocol server for Ragtime.

Exposes Ragtime tools (postgres, odoo, ssh, filesystem search) via MCP protocol.
Supports both stdio (for Claude Desktop) and SSE (for web clients) transports.

Features:
- Dynamic tool discovery from ToolConfig database
- Heartbeat-based filtering (only exposes healthy tools)
- Automatic tool refresh on config changes
- Custom MCP routes with tool selection
- Optional authorization for routes
- Extensible architecture for adding new tool types
"""

import argparse
import asyncio
import logging
import sys
import weakref
from collections.abc import Callable
from typing import Any

from mcp.server import Server
from mcp.server.session import ServerSession
from mcp.types import TextContent, Tool

from ragtime.core.app_settings import get_app_settings
from ragtime.core.database import connect_db, disconnect_db, get_db
from ragtime.core.logging import get_logger
from ragtime.mcp.tools import McpRouteFilter, MCPToolAdapter, mcp_tool_adapter
from ragtime.rag import rag

logger = get_logger(__name__)

# MCP server instance - created lazily with dynamic name
_mcp_server: Server | None = None

# Cache of custom route servers
_custom_route_servers: dict[str, Server] = {}
_custom_route_adapters: dict[str, MCPToolAdapter] = {}

# Cache of default route filter servers (keyed by filter ID)
_default_filter_servers: dict[str, Server] = {}
_default_filter_adapters: dict[str, MCPToolAdapter] = {}

# Active MCP sessions (for best-effort tools/list_changed notifications)
_active_sessions: weakref.WeakSet[ServerSession] = weakref.WeakSet()

# Callbacks invoked when tools/routes change (used by HTTP transport layer caches)
_tools_changed_callbacks: list[Callable[[], None]] = []


def register_tools_changed_callback(callback: Callable[[], None]) -> None:
    """Register a callback to run when MCP tools/routes are invalidated."""
    if callback in _tools_changed_callbacks:
        return
    _tools_changed_callbacks.append(callback)


def _track_active_session(server: Server) -> None:
    """Track the current MCP session if running inside a request context."""
    try:
        session = server.request_context.session
    except LookupError:
        return
    _active_sessions.add(session)


async def _notify_active_sessions_tool_list_changed() -> None:
    """Best-effort broadcast of tools/list_changed to active MCP sessions."""
    if not _active_sessions:
        return

    for session in list(_active_sessions):
        try:
            await session.send_tool_list_changed()
        except Exception:
            logger.debug("Failed to send tools/list_changed to MCP session")


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


def notify_tools_changed() -> None:
    """
    Notify that tools have changed.

    This invalidates the MCP tool cache so the next tools/list request
    returns the updated list. In stateless mode, clients poll for changes
    when they see listChanged: true in server capabilities.
    """
    mcp_tool_adapter.invalidate_cache()
    # Also invalidate all custom route adapters
    for adapter in _custom_route_adapters.values():
        adapter.invalidate_cache()
    # Invalidate default filter adapters
    for adapter in _default_filter_adapters.values():
        adapter.invalidate_cache()
    # Clear all server caches (they'll be recreated with new config)
    _custom_route_servers.clear()
    _custom_route_adapters.clear()
    _default_filter_servers.clear()
    _default_filter_adapters.clear()

    # Notify additional cache owners (e.g., HTTP route-level server caches)
    for callback in _tools_changed_callbacks:
        try:
            callback()
        except Exception:
            logger.exception("Failed to run MCP tools-changed callback")

    # Best-effort notify connected sessions (if transport/session supports it)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        loop.create_task(_notify_active_sessions_tool_list_changed())

    logger.info("MCP tool cache invalidated - tools list has changed")


def _register_handlers(
    server: Server,
    adapter: MCPToolAdapter | None = None,
    route_filter: McpRouteFilter | None = None,
) -> None:
    """Register MCP protocol handlers on the server instance."""

    # Use provided adapter or default
    tool_adapter = adapter if adapter else mcp_tool_adapter

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """
        List all available tools.

        Tools are dynamically discovered from the ToolConfig database.
        Only healthy tools (passing heartbeat check) are exposed.
        """
        tools: list[Tool] = []
        _track_active_session(server)

        try:
            tool_definitions = await tool_adapter.get_available_tools(
                route_filter=route_filter
            )

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
        _track_active_session(server)
        logger.info(f"MCP call_tool: {name}")
        logger.debug(f"MCP call_tool arguments: {arguments}")

        try:
            # If a route filter is active, validate the tool is allowed
            if route_filter is not None:
                allowed_tools = await tool_adapter.get_available_tools(
                    route_filter=route_filter
                )
                allowed_names = {t.name for t in allowed_tools}
                if name not in allowed_names:
                    logger.warning(f"Tool '{name}' not allowed by route filter")
                    return [
                        TextContent(
                            type="text",
                            text=f"Error: Tool '{name}' is not available",
                        )
                    ]

            result = await tool_adapter.execute_tool(name, arguments)

            return [TextContent(type="text", text=result)]

        except Exception as e:
            logger.exception(f"MCP tool execution error: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]


async def get_custom_route_server(
    route_path: str,
) -> tuple[Server, MCPToolAdapter, bool, str | None, str | None, str | None] | None:
    """
    Get or create an MCP server for a custom route.

    Args:
        route_path: The route path suffix (e.g., 'my_toolset')

    Returns:
        Tuple of (Server, MCPToolAdapter, require_auth, auth_password, auth_method, allowed_group)
        or None if route not found
    """
    # Check cache first
    if route_path in _custom_route_servers:
        adapter = _custom_route_adapters.get(route_path, mcp_tool_adapter)
        # Need to fetch require_auth, authPassword, authMethod, allowedLdapGroup from db
        db = await get_db()
        route = await db.mcprouteconfig.find_unique(where={"routePath": route_path})
        if route:
            return (
                _custom_route_servers[route_path],
                adapter,
                route.requireAuth,
                route.authPassword,
                route.authMethod,
                route.allowedLdapGroup,
            )
        return None

    db = await get_db()

    # Find the route configuration
    route = await db.mcprouteconfig.find_unique(
        where={"routePath": route_path},
        include={"toolSelections": True},
    )

    if not route or not route.enabled:
        return None

    # Get tool config IDs
    tool_ids = (
        [sel.toolConfigId for sel in route.toolSelections]
        if route.toolSelections
        else []
    )

    # Create route filter with index selections
    route_filter = McpRouteFilter(
        tool_config_ids=tool_ids,
        include_knowledge_search=route.includeKnowledgeSearch,
        include_git_history=route.includeGitHistory,
        selected_document_indexes=route.selectedDocumentIndexes or None,
        selected_filesystem_indexes=route.selectedFilesystemIndexes or None,
        selected_schema_indexes=route.selectedSchemaIndexes or None,
    )

    # Create new adapter and server for this route
    adapter = MCPToolAdapter()
    _custom_route_adapters[route_path] = adapter

    app_settings = await get_app_settings()
    server_name = app_settings.get("server_name", "Ragtime")
    route_server_name = f"{server_name.lower().replace(' ', '-')}-{route_path}"

    server = Server(route_server_name)
    _register_handlers(server, adapter, route_filter)
    _custom_route_servers[route_path] = server

    logger.info(f"Created custom MCP server for route: /mcp/{route_path}")

    return (
        server,
        adapter,
        route.requireAuth,
        route.authPassword,
        route.authMethod,
        route.allowedLdapGroup,
    )


async def get_default_route_filtered_server(
    filter_id: str,
) -> tuple[Server, MCPToolAdapter] | None:
    """
    Get or create an MCP server for a default route filter.

    This is used when OAuth2 auth is enabled on the default route and the user
    is a member of an LDAP group that has a configured filter.

    Args:
        filter_id: The filter ID

    Returns:
        Tuple of (Server, MCPToolAdapter) or None if filter not found
    """
    # Check cache first
    if filter_id in _default_filter_servers:
        adapter = _default_filter_adapters.get(filter_id, mcp_tool_adapter)
        return (_default_filter_servers[filter_id], adapter)

    db = await get_db()

    # Find the filter configuration
    f = await db.mcpdefaultroutefilter.find_unique(
        where={"id": filter_id},
        include={"toolSelections": True},
    )

    if not f or not f.enabled:
        return None

    # Get tool config IDs
    tool_ids = (
        [sel.toolConfigId for sel in f.toolSelections] if f.toolSelections else []
    )

    # Create route filter with index selections
    route_filter = McpRouteFilter(
        tool_config_ids=tool_ids,
        include_knowledge_search=f.includeKnowledgeSearch,
        include_git_history=f.includeGitHistory,
        selected_document_indexes=f.selectedDocumentIndexes or None,
        selected_filesystem_indexes=f.selectedFilesystemIndexes or None,
        selected_schema_indexes=f.selectedSchemaIndexes or None,
    )

    # Create new adapter and server for this filter
    adapter = MCPToolAdapter()
    _default_filter_adapters[filter_id] = adapter

    app_settings = await get_app_settings()
    server_name = app_settings.get("server_name", "Ragtime")
    filter_server_name = (
        f"{server_name.lower().replace(' ', '-')}-filter-{filter_id[:8]}"
    )

    server = Server(filter_server_name)
    _register_handlers(server, adapter, route_filter)
    _default_filter_servers[filter_id] = server

    logger.info(f"Created filtered MCP server for default route filter: {f.name}")

    return (server, adapter)


# For backward compatibility - create a default instance with handlers
mcp_server = Server("ragtime")
_register_handlers(mcp_server)


async def run_mcp_server(transport: str = "stdio") -> None:
    """
    Run the MCP server with the specified transport.

    Args:
        transport: "stdio" for Claude Desktop, "sse" for web clients
    """
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
            from mcp.server.lowlevel import NotificationOptions
            from mcp.server.stdio import stdio_server

            async with stdio_server() as (read_stream, write_stream):
                # Advertise listChanged: true so clients know to poll for tool updates
                notification_options = NotificationOptions(tools_changed=True)
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(
                        notification_options=notification_options
                    ),
                )
        else:
            raise ValueError(f"Unsupported transport: {transport}")
    finally:
        await disconnect_db()
        logger.info("MCP server stopped")


def main() -> None:
    """Entry point for running MCP server from command line."""
    parser = argparse.ArgumentParser(description="Ragtime MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    args = parser.parse_args()

    # Set up basic logging for standalone mode
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
