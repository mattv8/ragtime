"""
MCP Tool Adapter - Converts Ragtime tools to MCP format.

This module provides the translation layer between Ragtime's LangChain-based
tools and MCP's tool format. It supports:
- Dynamic tool discovery from database
- Heartbeat-based filtering of unhealthy tools
- JSON Schema generation for MCP inputSchema
- Extensible tool type registration for future tools
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from ragtime.config.settings import settings
from ragtime.core.app_settings import get_app_settings, get_tool_configs
from ragtime.core.logging import get_logger
from ragtime.indexer.schema_service import search_schema_index
from ragtime.rag import rag
from ragtime.rag.components import RAGComponents
from ragtime.tools.git_history import search_git_history

logger = get_logger(__name__)


# Tool input schemas for MCP (JSON Schema format)
# These define the parameters each tool type accepts
TOOL_INPUT_SCHEMAS: dict[str, dict] = {
    "postgres": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute. Must include LIMIT clause for SELECT queries.",
            },
            "reason": {
                "type": "string",
                "description": "Brief description of what this query retrieves",
            },
        },
        "required": ["query"],
    },
    "odoo_shell": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute in Odoo shell using ORM methods (env['model'].search_read, etc.)",
            },
            "reason": {
                "type": "string",
                "description": "Brief description of what this code does",
            },
        },
        "required": ["code"],
    },
    "ssh_shell": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute on the remote server",
            },
            "reason": {
                "type": "string",
                "description": "Brief description of what this command does",
            },
        },
        "required": ["command"],
    },
    "filesystem_indexer": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query to find relevant documents/files",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-50, default 10)",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
            },
            "max_chars_per_result": {
                "type": "integer",
                "description": "Maximum characters per result (default: 500). Use 0 for full content when you need complete file content. Increase when results are truncated.",
                "default": 500,
                "minimum": 0,
                "maximum": 10000,
            },
        },
        "required": ["query"],
    },
    "mssql": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute. Must include TOP n for SELECT queries.",
            },
            "reason": {
                "type": "string",
                "description": "Brief description of what this query retrieves",
            },
        },
        "required": ["query"],
    },
    "mysql": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute. Must include LIMIT clause for SELECT queries.",
            },
            "reason": {
                "type": "string",
                "description": "Brief description of what this query retrieves",
            },
        },
        "required": ["query"],
    },
    # Knowledge search tool (built-in, always available when indexes exist)
    "knowledge_search": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant documentation, code, or technical information",
            },
            "index_name": {
                "type": "string",
                "description": "Optional: specific index to search (leave empty to search all indexes)",
                "default": "",
            },
        },
        "required": ["query"],
    },
    # Schema search tool (for database tools with schema indexing enabled)
    "schema_search": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The natural language query used to search the database schema for relevant table and column information.",
            },
            "limit": {
                "type": "integer",
                "description": "The maximum number of matches to return. Defaults to 5.",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["prompt"],
    },
    # Git history search tool (for git-based filesystem indexes)
    "git_history": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "The git action to perform. One of: "
                    "'search_commits' - Search commit messages for keywords, "
                    "'get_commit' - Get detailed info about a specific commit, "
                    "'file_history' - Get commit history for a specific file, "
                    "'blame' - Show who last modified each line of a file, "
                    "'find_files' - Find files matching a pattern (use before file_history/blame if unsure of path)"
                ),
                "enum": [
                    "search_commits",
                    "get_commit",
                    "file_history",
                    "blame",
                    "find_files",
                ],
            },
            "query": {
                "type": "string",
                "description": "Search query for 'search_commits' action - keywords to find in commit messages",
            },
            "commit_hash": {
                "type": "string",
                "description": "Commit hash for 'get_commit' action (full or abbreviated)",
            },
            "file_path": {
                "type": "string",
                "description": "File path for 'file_history'/'blame' actions, or pattern for 'find_files' (e.g., 'index.php', '*.py')",
            },
            "index_name": {
                "type": "string",
                "description": "Optional: specific index/repo to search (searches all git repos if not specified)",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return for search operations",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": ["action"],
    },
}


@dataclass
class ToolHealthStatus:
    """Health status for a single tool."""

    tool_id: str
    alive: bool
    latency_ms: float | None = None
    error: str | None = None
    checked_at: datetime | None = None


@dataclass
class MCPToolDefinition:
    """MCP tool definition ready for registration."""

    name: str
    description: str
    input_schema: dict
    tool_config: dict  # Original ToolConfig data
    execute_fn: Callable[..., Awaitable[str]]


@dataclass
class McpRouteFilter:
    """Filter configuration for custom MCP routes."""

    tool_config_ids: list[str]  # IDs of tool configs to include
    include_knowledge_search: bool = True
    include_git_history: bool = True
    selected_document_indexes: list[str] | None = (
        None  # None = all, [] = none, list = specific
    )
    selected_filesystem_indexes: list[str] | None = (
        None  # Filter by filesystem tool IDs
    )
    selected_schema_indexes: list[str] | None = None  # Filter by schema tool IDs


class MCPToolAdapter:
    """
    Adapter that converts Ragtime tools to MCP format.

    Features:
    - Dynamic tool discovery from ToolConfig database
    - Heartbeat-based filtering (unhealthy tools are hidden)
    - Caching with TTL for performance
    - Extensible tool type registration
    - Support for filtered tool lists (custom MCP routes)
    """

    def __init__(self, heartbeat_cache_ttl: float = 30.0):
        """
        Initialize the adapter.

        Args:
            heartbeat_cache_ttl: How long to cache heartbeat results (seconds)
        """
        self._heartbeat_cache_ttl = heartbeat_cache_ttl
        self._heartbeat_cache: dict[str, ToolHealthStatus] = {}
        self._last_heartbeat_check: datetime | None = None
        self._tool_executors: dict[str, Callable[..., Awaitable[str]]] = {}

    async def get_available_tools(
        self,
        include_unhealthy: bool = False,
        route_filter: McpRouteFilter | None = None,
    ) -> list[MCPToolDefinition]:
        """
        Get all available tools as MCP definitions.

        Args:
            include_unhealthy: If True, include tools that failed heartbeat
            route_filter: Optional filter for custom MCP routes

        Returns:
            List of MCP tool definitions ready for registration
        """
        tools: list[MCPToolDefinition] = []

        # Get tool configs from database
        tool_configs = await get_tool_configs()

        # Apply route filter if provided
        if route_filter is not None:
            tool_configs = [
                c for c in tool_configs if c.get("id") in route_filter.tool_config_ids
            ]

        # Check heartbeat status
        health_status = await self._check_heartbeats(tool_configs)

        for config in tool_configs:
            tool_id = config.get("id", "")
            tool_type = config.get("tool_type", "")
            enabled = config.get("enabled", True)
            tool_name = config.get("name", "Unknown")

            # Skip disabled tools
            if not enabled:
                logger.info(f"MCP: Skipping disabled tool: {tool_name}")
                continue

            # Skip unhealthy tools unless explicitly requested
            status = health_status.get(tool_id)
            if status and not status.alive and not include_unhealthy:
                logger.info(
                    f"MCP: Skipping unhealthy tool: {tool_name} ({status.error})"
                )
                continue

            # For filesystem_indexer tools, check selected_filesystem_indexes filter
            if tool_type == "filesystem_indexer" and route_filter is not None:
                if route_filter.selected_filesystem_indexes is not None:
                    if tool_id not in route_filter.selected_filesystem_indexes:
                        continue

            # Create MCP tool definition
            tool_def = await self._create_tool_definition(config)
            if tool_def:
                tools.append(tool_def)
                logger.info(f"MCP: Added tool: {tool_def.name} (type: {tool_type})")
            else:
                logger.info(
                    f"MCP: Failed to create tool definition for: {tool_name} (type: {tool_type})"
                )

            # Create schema search tool if schema indexing is enabled for this tool
            # Apply selected_schema_indexes filter if provided
            if (
                route_filter is not None
                and route_filter.selected_schema_indexes is not None
            ):
                if tool_id not in route_filter.selected_schema_indexes:
                    continue
            schema_tool_def = await self._create_schema_search_tool_definition(config)
            if schema_tool_def:
                tools.append(schema_tool_def)

        # Add knowledge search tool(s) based on aggregate_search setting
        # Skip if route_filter explicitly excludes knowledge search
        include_knowledge = (
            route_filter is None or route_filter.include_knowledge_search
        )

        if include_knowledge:
            app_settings = await get_app_settings()
            aggregate_search = app_settings.get("aggregate_search", True)

            if aggregate_search:
                # Single aggregated search_knowledge tool
                # When aggregate_search is true, always include all indexes (don't filter)
                knowledge_tool = await self._create_knowledge_search_tool(
                    selected_indexes=None
                )
                if knowledge_tool:
                    tools.append(knowledge_tool)
            else:
                # Separate search_<index_name> tools for each index
                # Apply document index filter when in per-index mode
                per_index_tools = await self._create_per_index_search_tools(
                    selected_indexes=(
                        route_filter.selected_document_indexes if route_filter else None
                    )
                )
                tools.extend(per_index_tools)
        else:
            app_settings = await get_app_settings()
            aggregate_search = app_settings.get("aggregate_search", True)

        # Add git history search tool(s) if we have git repos
        # Skip if route_filter explicitly excludes git history
        include_git = route_filter is None or route_filter.include_git_history

        if include_git:
            git_history_tools = await self._create_git_history_tools(aggregate_search)
            tools.extend(git_history_tools)

        return tools

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool by name with the given arguments.

        Args:
            tool_name: The MCP tool name (e.g., "query_production_db")
            arguments: Tool arguments from MCP client

        Returns:
            Tool execution result as string
        """
        # Check if we have a cached executor
        if tool_name in self._tool_executors:
            return await self._tool_executors[tool_name](**arguments)

        # Otherwise, we need to find and build the tool
        tool_configs = await get_tool_configs()

        # Parse tool name to find matching config
        # Tool names follow patterns like: query_<name>, odoo_<name>, ssh_<name>, search_<name>
        for config in tool_configs:
            expected_name = self._build_tool_name(config)
            if expected_name == tool_name:
                tool_def = await self._create_tool_definition(config)
                if tool_def:
                    self._tool_executors[tool_name] = tool_def.execute_fn
                    return await tool_def.execute_fn(**arguments)

        # Check for knowledge search
        if tool_name == "search_knowledge":
            knowledge_tool = await self._create_knowledge_search_tool()
            if knowledge_tool:
                self._tool_executors[tool_name] = knowledge_tool.execute_fn
                return await knowledge_tool.execute_fn(**arguments)

        # Check for schema search tools (pattern: search_<name>_schema)
        if tool_name.startswith("search_") and tool_name.endswith("_schema"):
            for config in tool_configs:
                schema_tool_name = self._build_schema_tool_name(config)
                if schema_tool_name == tool_name:
                    schema_tool_def = await self._create_schema_search_tool_definition(
                        config
                    )
                    if schema_tool_def:
                        self._tool_executors[tool_name] = schema_tool_def.execute_fn
                        return await schema_tool_def.execute_fn(**arguments)

        # Check for per-index search tools (pattern: search_<index_name>)
        # These are created when aggregate_search is disabled
        if tool_name.startswith("search_") and not tool_name.endswith("_schema"):
            per_index_tools = await self._create_per_index_search_tools()
            for tool_def in per_index_tools:
                if tool_def.name == tool_name:
                    self._tool_executors[tool_name] = tool_def.execute_fn
                    return await tool_def.execute_fn(**arguments)

        # Check for git history tools
        if tool_name == "search_git_history" or tool_name.startswith(
            "search_git_history_"
        ):
            app_settings = await get_app_settings()
            aggregate_search = app_settings.get("aggregate_search", True)
            git_tools = await self._create_git_history_tools(aggregate_search)
            for tool_def in git_tools:
                if tool_def.name == tool_name:
                    self._tool_executors[tool_name] = tool_def.execute_fn
                    return await tool_def.execute_fn(**arguments)

        return f"Error: Unknown tool '{tool_name}'"

    async def _check_heartbeats(
        self, tool_configs: list[dict]
    ) -> dict[str, ToolHealthStatus]:
        """
        Check heartbeat status for all tools, using cache when possible.

        Returns:
            Dictionary mapping tool_id to health status
        """
        now = datetime.now(timezone.utc)

        # Check if cache is still valid
        if (
            self._last_heartbeat_check
            and (now - self._last_heartbeat_check).total_seconds()
            < self._heartbeat_cache_ttl
        ):
            return self._heartbeat_cache

        # Import here to avoid circular imports
        try:
            from ragtime.indexer.routes import _heartbeat_check
        except ImportError:
            pass  # Expected circular import context

        async def check_single(config: dict) -> tuple[str, ToolHealthStatus]:
            # SSH tunnel connections need more time to authenticate and establish
            has_ssh_tunnel = conn_config.get("ssh_tunnel_enabled", False)
            heartbeat_timeout = 15.0 if has_ssh_tunnel else 5.0

            try:
                result = await asyncio.wait_for(
                    _heartbeat_check(tool_type, conn_config),
                    timeout=heartbeat_timeout,
                )
                latency = (asyncio.get_event_loop().time() - start_time) * 1000

                return tool_id, ToolHealthStatus(
                    tool_id=tool_id,
                    alive=result.success,
                    latency_ms=round(latency, 1) if result.success else None,
                    error=result.message if not result.success else None,
                    checked_at=now,
                )
            except asyncio.TimeoutError:
                timeout_msg = f"Heartbeat timeout ({int(heartbeat_timeout)}s)"
                return tool_id, ToolHealthStatus(
                    tool_id=tool_id,
                    alive=False,
                    error=timeout_msg,
                    checked_at=now,
                )
            except Exception as e:
                return tool_id, ToolHealthStatus(
                    tool_id=tool_id,
                    alive=False,
                    error=str(e),
                    checked_at=now,
                )

        # Check all tools concurrently
        results = await asyncio.gather(
            *[check_single(config) for config in tool_configs],
            return_exceptions=True,
        )

        # Update cache
        self._heartbeat_cache = {}
        for result in results:
            if isinstance(result, tuple):
                tool_id, status = result
                self._heartbeat_cache[tool_id] = status

        self._last_heartbeat_check = now
        return self._heartbeat_cache

    def _build_tool_name(self, config: dict) -> str:
        """Build MCP tool name from config."""
        tool_type = config.get("tool_type", "")
        # Tool-safe name: lowercase and replace any non-alphanumeric sequence with underscore
        raw_name = (config.get("name", "") or "").strip()
        name = re.sub(r"[^a-zA-Z0-9]+", "_", raw_name).strip("_").lower()

        # Prefix based on tool type
        prefixes = {
            "postgres": "query_",
            "mssql": "query_",
            "mysql": "query_",
            "odoo_shell": "odoo_",
            "ssh_shell": "ssh_",
            "filesystem_indexer": "search_",
        }
        prefix = prefixes.get(tool_type, "tool_")
        return f"{prefix}{name}"

    def _build_schema_tool_name(self, config: dict) -> str | None:
        """Build MCP schema search tool name from config, if applicable."""
        tool_type = config.get("tool_type", "")
        # Only database tools can have schema search
        if tool_type not in ("postgres", "mssql", "mysql"):
            return None

        # Tool-safe name
        raw_name = (config.get("name", "") or "").strip()
        name = re.sub(r"[^a-zA-Z0-9]+", "_", raw_name).strip("_").lower()
        return f"search_{name}_schema"

    async def _create_schema_search_tool_definition(
        self, config: dict
    ) -> MCPToolDefinition | None:
        """
        Create an MCP tool definition for schema search if enabled.

        Returns:
            MCPToolDefinition or None if schema indexing not enabled
        """
        tool_type = config.get("tool_type", "")
        # Only database tools can have schema search
        if tool_type not in ("postgres", "mssql", "mysql"):
            return None

        conn_config = config.get("connection_config", {})
        schema_index_enabled = conn_config.get("schema_index_enabled", False)
        if not schema_index_enabled:
            return None

        tool_name = self._build_schema_tool_name(config)
        if not tool_name:
            return None

        # Get safe name for index lookup
        raw_name = (config.get("name", "") or "").strip()
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", raw_name).strip("_").lower()
        index_name = f"schema_{safe_name}"

        # Build description
        db_name = config.get("name", "database")
        description = config.get("description", "")
        tool_description = (
            f"This retrieves relevant {db_name} schema entries based on a natural language query. "
            f"Use this BEFORE writing SQL queries to understand available tables and columns."
        )
        if description:
            tool_description += f" Database contains: {description}"

        # Create executor
        async def search_schema(prompt: str, limit: int = 5, **_: Any) -> str:
            """Execute schema search."""
            logger.debug(f"MCP schema search for {db_name}: {prompt[:100]}")
            result = await search_schema_index(
                query=prompt,
                index_name=index_name,
                max_results=min(limit, 20),
            )
            return result

        return MCPToolDefinition(
            name=tool_name,
            description=tool_description,
            input_schema=TOOL_INPUT_SCHEMAS["schema_search"],
            tool_config=config,
            execute_fn=search_schema,
        )

    async def _create_tool_definition(self, config: dict) -> MCPToolDefinition | None:
        """
        Create an MCP tool definition from a ToolConfig.

        Returns:
            MCPToolDefinition or None if tool type is unknown
        """
        tool_type = config.get("tool_type", "")
        tool_name = self._build_tool_name(config)

        # Get input schema for tool type
        input_schema = TOOL_INPUT_SCHEMAS.get(tool_type)
        if not input_schema:
            logger.warning(f"Unknown tool type: {tool_type}")
            return None

        # Build description
        description = self._build_tool_description(config)

        # Create executor function
        execute_fn = await self._create_executor(config, tool_type)

        return MCPToolDefinition(
            name=tool_name,
            description=description,
            input_schema=input_schema,
            tool_config=config,
            execute_fn=execute_fn,
        )

    def _build_tool_description(self, config: dict) -> str:
        """Build tool description for MCP."""
        tool_type = config.get("tool_type", "")
        name = config.get("name", "Unknown")
        description = config.get("description", "")

        type_descriptions = {
            "postgres": f"Query the '{name}' PostgreSQL database using SQL.",
            "mssql": f"Query the '{name}' MSSQL/SQL Server database using SQL.",
            "odoo_shell": f"Execute Python ORM code in '{name}' Odoo shell.",
            "ssh_shell": f"Execute shell commands on '{name}' via SSH.",
            "filesystem_indexer": f"Search indexed documents from '{name}'.",
        }

        base_desc = type_descriptions.get(
            tool_type, f"Execute {tool_type} tool '{name}'"
        )

        if description:
            base_desc += f" This resource contains: {description}"

        return base_desc

    async def _create_executor(
        self, config: dict, tool_type: str
    ) -> Callable[..., Awaitable[str]]:
        """
        Create an async executor function for the tool.

        This reuses the existing tool execution logic from components.py.
        """
        # Import components for execution logic
        # Create a temporary RAGComponents instance to build the tool
        # This is a bit wasteful but ensures we reuse all validation/security logic
        rag_temp = RAGComponents()
        rag_temp._tool_configs = [config]  # pyright: ignore[reportPrivateUsage]

        tool_name = config.get("name", "").replace(" ", "_").lower()
        # Ensure executor uses the exact same tool-safe naming convention
        tool_name = (
            re.sub(r"[^a-zA-Z0-9]+", "_", (config.get("name", "") or "").strip())
            .strip("_")
            .lower()
        )
        tool_id = config.get("id", "")

        if tool_type == "postgres":
            tool = await rag_temp._create_postgres_tool(
                config, tool_name, tool_id, include_metadata=False
            )  # pyright: ignore[reportPrivateUsage]
        elif tool_type == "mssql":
            tool = await rag_temp._create_mssql_tool(
                config, tool_name, tool_id, include_metadata=False
            )  # pyright: ignore[reportPrivateUsage]
        elif tool_type == "mysql":
            tool = await rag_temp._create_mysql_tool(
                config, tool_name, tool_id, include_metadata=False
            )  # pyright: ignore[reportPrivateUsage]
        elif tool_type == "odoo_shell":
            tool = await rag_temp._create_odoo_tool(
                config, tool_name, tool_id
            )  # pyright: ignore[reportPrivateUsage]
        elif tool_type == "ssh_shell":
            tool = await rag_temp._create_ssh_tool(
                config, tool_name, tool_id
            )  # pyright: ignore[reportPrivateUsage]
        elif tool_type == "filesystem_indexer":
            tool = await rag_temp._create_filesystem_tool(
                config, tool_name, tool_id
            )  # pyright: ignore[reportPrivateUsage]
        else:

            async def unknown_executor(**_kwargs: Any) -> str:
                return f"Error: Unknown tool type '{tool_type}'"

            return unknown_executor

        if tool is None:

            async def failed_executor(**_kwargs: Any) -> str:
                return f"Error: Failed to create tool '{tool_name}'"

            return failed_executor

        # Return the tool's coroutine
        async def executor(**kwargs: Any) -> str:
            try:
                result = await tool.ainvoke(kwargs)
                return str(result)
            except Exception as e:
                logger.exception(f"Tool execution error: {e}")
                return f"Error: {str(e)}"

        return executor

    async def _create_knowledge_search_tool(
        self, selected_indexes: list[str] | None = None
    ) -> MCPToolDefinition | None:
        """Create the knowledge search tool if indexes are available.

        Args:
            selected_indexes: Optional list of index names to include.
                              None = all indexes, [] = no indexes, list = specific indexes
        """
        # Allow partial availability - core ready is enough, indexes may still be loading
        if not rag.is_ready:
            return None

        # If indexes are still loading but none available yet, skip for now
        if not rag.retrievers:
            if rag._indexes_loading:  # pyright: ignore[reportPrivateUsage]
                logger.debug("Knowledge search tool skipped - indexes still loading")
            return None

        # Filter FAISS dbs based on selected_indexes
        if selected_indexes is not None:
            if not selected_indexes:
                return None  # Empty list means no document indexes selected
            available_dbs = {
                name: db
                for name, db in rag.faiss_dbs.items()
                if name in selected_indexes
            }
            if not available_dbs:
                return None
        else:
            available_dbs = rag.faiss_dbs

        index_names = list(available_dbs.keys())

        # Get default k and MMR settings
        app_settings = rag._app_settings or {}  # pyright: ignore[reportPrivateUsage]
        default_k = app_settings.get("search_results_k", 5)
        use_mmr = app_settings.get("search_use_mmr", True)
        mmr_lambda = app_settings.get("search_mmr_lambda", 0.5)

        description = (
            "Search indexed documentation and codebase for relevant information. "
            f"Available indexes: {', '.join(index_names)}. "
            "Use for finding code examples, schema definitions, configuration details. "
            "Use 'k' to control number of results. "
            "Use 'max_chars_per_result' to control content length (0 for full content when results are truncated)."
        )

        # Capture available_dbs in closure
        dbs_snapshot = available_dbs

        async def search_knowledge(
            query: str,
            index_name: str = "",
            k: int = default_k,
            max_chars_per_result: int = 500,
            **_: Any,
        ) -> str:
            """Execute knowledge search with dynamic k and content length."""
            results = []
            errors = []

            # Clamp parameters
            k = max(1, min(50, k))
            max_chars_per_result = max(0, min(10000, max_chars_per_result))

            # Log the search attempt for debugging
            logger.debug(
                f"MCP search_knowledge called with query='{query[:50] if query else ''}...', "
                f"index_name='{index_name}', k={k}, max_chars={max_chars_per_result}"
            )
            logger.debug(f"Available FAISS dbs: {list(dbs_snapshot.keys())}")

            # Determine which dbs to search
            if index_name and index_name in dbs_snapshot:
                dbs_to_search = {index_name: dbs_snapshot[index_name]}
            else:
                dbs_to_search = dbs_snapshot

            if not dbs_to_search:
                logger.warning("No FAISS dbs available for MCP search_knowledge")
                return "No knowledge indexes are currently loaded. Please index some documents first."

            for name, db in dbs_to_search.items():
                try:
                    logger.debug(
                        f"MCP searching index '{name}' with query: {query[:50] if query else ''}..., k={k}"
                    )
                    # Use MMR or similarity search based on settings
                    if use_mmr:
                        fetch_k = max(k * 4, 20)  # Get more candidates for diversity
                        docs = db.max_marginal_relevance_search(
                            query, k=k, fetch_k=fetch_k, lambda_mult=mmr_lambda
                        )
                    else:
                        docs = db.similarity_search(query, k=k)

                    logger.debug(f"MCP index '{name}' returned {len(docs)} documents")
                    for doc in docs:
                        source = doc.metadata.get("source", "unknown")
                        content = doc.page_content
                        # Apply truncation if max_chars_per_result > 0
                        if (
                            max_chars_per_result > 0
                            and len(content) > max_chars_per_result
                        ):
                            content = content[:max_chars_per_result] + "... (truncated)"
                        results.append(f"[{name}] {source}:\n{content}")
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Error searching {name}: {e}", exc_info=True)
                    # Detect Ollama connectivity issues
                    if (
                        "ollama" in error_msg.lower()
                        or "failed to connect" in error_msg.lower()
                    ):
                        errors.append(
                            f"[{name}] Embedding service unavailable - Cannot connect to Ollama. "
                            "Check that Ollama is running and the URL in Settings is accessible from the server "
                            "(use 'host.docker.internal' instead of 'localhost' when running in Docker)."
                        )
                    else:
                        errors.append(f"[{name}] Search error: {error_msg}")

            if results:
                logger.debug(f"MCP search_knowledge found {len(results)} results")
                return (
                    f"Found {len(results)} relevant documents:\n\n"
                    + "\n\n---\n\n".join(results)
                )

            # Return errors if we had any, otherwise generic no results message
            if errors:
                logger.warning(f"MCP search_knowledge failed with errors: {errors}")
                return "Search failed:\n" + "\n".join(errors)

            logger.debug("MCP search_knowledge found no results")
            return "No relevant documentation found for this query."

        # Create schema with available indexes in index_name description
        knowledge_schema = {
            "type": "object",
            "properties": {
                "query": TOOL_INPUT_SCHEMAS["knowledge_search"]["properties"]["query"],
                "index_name": {
                    "type": "string",
                    "description": (
                        "Optional: specific index to search (leave empty to search all indexes). "
                        f"Available indexes: {', '.join(index_names)}"
                    ),
                    "default": "",
                },
                "k": {
                    "type": "integer",
                    "description": (
                        f"Number of results to retrieve (default: {default_k}, max: 50). "
                        "Increase for broader searches, decrease for focused results."
                    ),
                    "default": default_k,
                    "minimum": 1,
                    "maximum": 50,
                },
                "max_chars_per_result": {
                    "type": "integer",
                    "description": (
                        "Maximum characters per result (default: 500). "
                        "Use 0 for full content when you need complete code/file content. "
                        "Increase when results are truncated."
                    ),
                    "default": 500,
                    "minimum": 0,
                    "maximum": 10000,
                },
            },
            "required": ["query"],
        }

        return MCPToolDefinition(
            name="search_knowledge",
            description=description,
            input_schema=knowledge_schema,
            tool_config={"tool_type": "knowledge_search", "name": "knowledge"},
            execute_fn=search_knowledge,
        )

    async def _create_per_index_search_tools(
        self, selected_indexes: list[str] | None = None
    ) -> list[MCPToolDefinition]:
        """Create separate search tools for each index.

        When aggregate_search is disabled, this creates search_<index_name>
        tools that give the AI granular control over which index to search.

        Args:
            selected_indexes: Optional list of index names to include.
                              None = all indexes, [] = no indexes, list = specific indexes
        """
        # Allow partial availability - core ready is enough
        if not rag.is_ready:
            return []

        # If indexes are still loading but none available yet, skip for now
        if not rag.faiss_dbs:
            if rag._indexes_loading:  # pyright: ignore[reportPrivateUsage]
                logger.debug("Per-index search tools skipped - indexes still loading")
            return []

        # Filter FAISS dbs based on selected_indexes
        if selected_indexes is not None:
            if not selected_indexes:
                return []  # Empty list means no document indexes selected
            available_dbs = {
                name: db
                for name, db in rag.faiss_dbs.items()
                if name in selected_indexes
            }
            if not available_dbs:
                return []
        else:
            available_dbs = rag.faiss_dbs

        tools: list[MCPToolDefinition] = []

        # Get settings for default k and MMR
        app_settings = rag._app_settings or {}  # pyright: ignore[reportPrivateUsage]
        default_k = app_settings.get("search_results_k", 5)
        use_mmr = app_settings.get("search_use_mmr", True)
        mmr_lambda = app_settings.get("search_mmr_lambda", 0.5)

        # Get index metadata for descriptions
        index_descriptions: dict[str, str] = {}
        for idx in rag._index_metadata or []:
            if idx.get("enabled", True):
                name = idx.get("name", "")
                index_descriptions[name] = idx.get("description", "")

        # Create input schema for per-index search with k and max_chars_per_result
        per_index_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant documentation, code, or technical information",
                },
                "k": {
                    "type": "integer",
                    "description": (
                        f"Number of results to retrieve (default: {default_k}, max: 50). "
                        "Increase for broader searches."
                    ),
                    "default": default_k,
                    "minimum": 1,
                    "maximum": 50,
                },
                "max_chars_per_result": {
                    "type": "integer",
                    "description": (
                        "Maximum characters per result (default: 500). "
                        "Use 0 for full content when results are truncated."
                    ),
                    "default": 500,
                    "minimum": 0,
                    "maximum": 10000,
                },
            },
            "required": ["query"],
        }

        for index_name, db in available_dbs.items():
            # Sanitize index name for tool name
            safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", index_name).strip("_").lower()
            tool_name = f"search_{safe_name}"

            # Build description
            idx_desc = index_descriptions.get(index_name, "")
            tool_description = (
                f"Search the '{index_name}' index for relevant information. "
                "Use 'k' to control results count, 'max_chars_per_result' for content length (0=full)."
            )
            if idx_desc:
                tool_description += f" {idx_desc}"

            # Create executor for this specific index
            def make_search_func(
                idx_name: str,
                idx_db,
                use_mmr_: bool,
                mmr_lambda_: float,
                default_k_: int,
            ):
                async def search_index(
                    query: str,
                    k: int = default_k_,
                    max_chars_per_result: int = 500,
                    **_: Any,
                ) -> str:
                    """Execute search on a specific index with dynamic k."""
                    results = []

                    # Clamp parameters
                    k = max(1, min(50, k))
                    max_chars_per_result = max(0, min(10000, max_chars_per_result))

                    logger.debug(
                        f"MCP search_{idx_name} called with query='{query[:50] if query else ''}...', k={k}, max_chars={max_chars_per_result}"
                    )

                    try:
                        # Use MMR or similarity search based on settings
                        if use_mmr_:
                            fetch_k = max(k * 4, 20)
                            docs = idx_db.max_marginal_relevance_search(
                                query, k=k, fetch_k=fetch_k, lambda_mult=mmr_lambda_
                            )
                        else:
                            docs = idx_db.similarity_search(query, k=k)

                        logger.debug(
                            f"MCP index '{idx_name}' returned {len(docs)} documents"
                        )
                        for doc in docs:
                            source = doc.metadata.get("source", "unknown")
                            content = doc.page_content
                            # Apply truncation if max_chars_per_result > 0
                            if (
                                max_chars_per_result > 0
                                and len(content) > max_chars_per_result
                            ):
                                content = (
                                    content[:max_chars_per_result] + "... (truncated)"
                                )
                            results.append(f"{source}:\n{content}")
                    except Exception as e:
                        error_msg = str(e)
                        logger.warning(
                            f"Error searching {idx_name}: {e}", exc_info=True
                        )
                        if (
                            "ollama" in error_msg.lower()
                            or "failed to connect" in error_msg.lower()
                        ):
                            return (
                                "Embedding service unavailable - Cannot connect to Ollama. "
                                "Check that Ollama is running and accessible."
                            )
                        return f"Search error: {error_msg}"

                    if results:
                        return (
                            f"Found {len(results)} relevant documents:\n\n"
                            + "\n\n---\n\n".join(results)
                        )

                    return (
                        f"No relevant documents found in {idx_name} for query: {query}"
                    )

                return search_index

            tools.append(
                MCPToolDefinition(
                    name=tool_name,
                    description=tool_description,
                    input_schema=per_index_schema,
                    tool_config={"tool_type": "per_index_search", "name": index_name},
                    execute_fn=make_search_func(
                        index_name, db, use_mmr, mmr_lambda, default_k
                    ),
                )
            )

        return tools

    async def _create_git_history_tools(
        self, aggregate_search: bool
    ) -> list[MCPToolDefinition]:
        """Create git history search tool(s) for git-based indexes.

        When aggregate_search is enabled: creates a single search_git_history tool
        When aggregate_search is disabled: creates search_git_history_<name> per index
        """
        tools: list[MCPToolDefinition] = []
        index_base = Path(settings.index_data_path)

        # Find all git repos and match them with index metadata
        # Only include repos where git_history_depth != 1 (shallow clone has no history)
        git_repos: list[tuple[str, Path, str]] = []  # (name, path, description)

        if index_base.exists():
            for index_dir in index_base.iterdir():
                if not index_dir.is_dir():
                    continue

                git_repo = index_dir / ".git_repo"
                if git_repo.exists() and (git_repo / ".git").exists():
                    # Get metadata including config_snapshot to check git_history_depth
                    description = ""
                    git_history_depth = 0  # Default to full history
                    for idx in rag._index_metadata or []:
                        if idx.get("name") == index_dir.name:
                            description = idx.get("description", "")
                            # Get git_history_depth from config_snapshot
                            config = idx.get("config_snapshot") or {}
                            git_history_depth = config.get("git_history_depth", 0)
                            break

                    # Only expose git history tool if depth != 1
                    # depth=0 means full history, depth>1 means we have commits to search
                    # depth=1 is a shallow clone with only the latest commit (not useful)
                    if git_history_depth != 1:
                        git_repos.append((index_dir.name, git_repo, description))
                    else:
                        logger.debug(
                            f"Skipping git history tool for {index_dir.name}: "
                            "shallow clone (depth=1)"
                        )

        if not git_repos:
            return []

        if aggregate_search:
            # Single tool for all git repos
            repo_names = [name for name, _, _ in git_repos]

            async def execute_aggregate(**kwargs: Any) -> str:
                return await search_git_history(**kwargs)

            # Create schema with available repos in index_name description
            git_history_schema = {
                "type": "object",
                "properties": {
                    **TOOL_INPUT_SCHEMAS["git_history"]["properties"],
                    "index_name": {
                        "type": "string",
                        "description": (
                            "Optional: specific index/repo name to search "
                            "(searches all git repos if not specified). "
                            f"Available repos: {', '.join(repo_names)}"
                        ),
                    },
                },
                "required": ["action"],
            }

            tools.append(
                MCPToolDefinition(
                    name="search_git_history",
                    description=(
                        "Search git repository history for detailed commit information. "
                        "Actions: 'search_commits' (find commits by message keywords), "
                        "'get_commit' (show full commit details), "
                        "'file_history' (show commits that modified a file), "
                        "'blame' (show who last modified each line), "
                        "'find_files' (find files matching a pattern - use before file_history/blame if unsure of path). "
                        f"Available repos: {', '.join(repo_names)}. "
                        "Use this to understand code evolution, find when bugs were introduced, "
                        "or identify who worked on specific features."
                    ),
                    input_schema=git_history_schema,
                    tool_config={"tool_type": "git_history", "name": "aggregate"},
                    execute_fn=execute_aggregate,
                )
            )
            logger.info(
                f"MCP: Added search_git_history tool for {len(git_repos)} repo(s)"
            )
        else:
            # Separate tool per repo
            for name, repo_path, description in git_repos:
                # Create per-index schema (no index_name param needed)
                per_index_schema = {
                    "type": "object",
                    "properties": {
                        "action": TOOL_INPUT_SCHEMAS["git_history"]["properties"][
                            "action"
                        ],
                        "query": TOOL_INPUT_SCHEMAS["git_history"]["properties"][
                            "query"
                        ],
                        "commit_hash": TOOL_INPUT_SCHEMAS["git_history"]["properties"][
                            "commit_hash"
                        ],
                        "file_path": TOOL_INPUT_SCHEMAS["git_history"]["properties"][
                            "file_path"
                        ],
                        "max_results": TOOL_INPUT_SCHEMAS["git_history"]["properties"][
                            "max_results"
                        ],
                    },
                    "required": ["action"],
                }

                safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
                tool_name = f"search_git_history_{safe_name}"

                tool_description = f"Search git history for the '{name}' repository. "
                if description:
                    tool_description += f"{description} "
                tool_description += "Actions: 'search_commits', 'get_commit', 'file_history', 'blame', 'find_files'."

                # Create executor for this specific repo
                def make_git_search_func(idx_name: str):
                    async def search_repo(**kwargs: Any) -> str:
                        # Force index_name to this repo
                        kwargs["index_name"] = idx_name
                        return await search_git_history(**kwargs)

                    return search_repo

                tools.append(
                    MCPToolDefinition(
                        name=tool_name,
                        description=tool_description,
                        input_schema=per_index_schema,
                        tool_config={"tool_type": "git_history", "name": name},
                        execute_fn=make_git_search_func(name),
                    )
                )

            logger.info(f"MCP: Added {len(tools)} per-index git history search tools")

        return tools

    def invalidate_cache(self) -> None:
        """Invalidate the heartbeat cache, forcing a fresh check on next call."""
        self._heartbeat_cache = {}
        self._last_heartbeat_check = None
        self._tool_executors = {}


# Global adapter instance
mcp_tool_adapter = MCPToolAdapter()
