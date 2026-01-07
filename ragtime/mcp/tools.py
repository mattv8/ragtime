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
from typing import Any, Awaitable, Callable

from ragtime.core.app_settings import get_tool_configs
from ragtime.core.logging import get_logger

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


class MCPToolAdapter:
    """
    Adapter that converts Ragtime tools to MCP format.

    Features:
    - Dynamic tool discovery from ToolConfig database
    - Heartbeat-based filtering (unhealthy tools are hidden)
    - Caching with TTL for performance
    - Extensible tool type registration
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
        self, include_unhealthy: bool = False
    ) -> list[MCPToolDefinition]:
        """
        Get all available tools as MCP definitions.

        Args:
            include_unhealthy: If True, include tools that failed heartbeat

        Returns:
            List of MCP tool definitions ready for registration
        """
        tools: list[MCPToolDefinition] = []

        # Get tool configs from database
        tool_configs = await get_tool_configs()

        # Check heartbeat status
        health_status = await self._check_heartbeats(tool_configs)

        for config in tool_configs:
            tool_id = config.get("id", "")
            enabled = config.get("enabled", True)

            # Skip disabled tools
            if not enabled:
                continue

            # Skip unhealthy tools unless explicitly requested
            status = health_status.get(tool_id)
            if status and not status.alive and not include_unhealthy:
                logger.debug(
                    f"Skipping unhealthy tool: {config.get('name')} ({status.error})"
                )
                continue

            # Create MCP tool definition
            tool_def = await self._create_tool_definition(config)
            if tool_def:
                tools.append(tool_def)

            # Create schema search tool if schema indexing is enabled for this tool
            schema_tool_def = await self._create_schema_search_tool_definition(config)
            if schema_tool_def:
                tools.append(schema_tool_def)

        # Add knowledge search tool if indexes are available
        knowledge_tool = await self._create_knowledge_search_tool()
        if knowledge_tool:
            tools.append(knowledge_tool)

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
        from ragtime.indexer.routes import _heartbeat_check

        async def check_single(config: dict) -> tuple[str, ToolHealthStatus]:
            tool_id = config.get("id", "")
            tool_type = config.get("tool_type", "")
            conn_config = config.get("connection_config", {})
            start_time = asyncio.get_event_loop().time()

            try:
                result = await asyncio.wait_for(
                    _heartbeat_check(tool_type, conn_config),
                    timeout=5.0,
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
                return tool_id, ToolHealthStatus(
                    tool_id=tool_id,
                    alive=False,
                    error="Heartbeat timeout (5s)",
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
        if tool_type not in ("postgres", "mssql"):
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
        if tool_type not in ("postgres", "mssql"):
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
            from ragtime.indexer.schema_service import search_schema_index

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
        from ragtime.rag.components import RAGComponents

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
                config, tool_name, tool_id
            )  # pyright: ignore[reportPrivateUsage]
        elif tool_type == "mssql":
            tool = await rag_temp._create_mssql_tool(
                config, tool_name, tool_id
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

    async def _create_knowledge_search_tool(self) -> MCPToolDefinition | None:
        """Create the knowledge search tool if indexes are available."""
        from ragtime.rag import rag

        if not rag.is_ready or not rag.retrievers:
            return None

        index_names = list(rag.retrievers.keys())

        description = (
            "Search indexed documentation and codebase for relevant information. "
            f"Available indexes: {', '.join(index_names)}. "
            "Use for finding code examples, schema definitions, configuration details."
        )

        async def search_knowledge(query: str, index_name: str = "", **_: Any) -> str:
            """Execute knowledge search."""
            results = []
            errors = []

            # Log the search attempt for debugging
            logger.debug(
                f"MCP search_knowledge called with query='{query[:50] if query else ''}...', index_name='{index_name}'"
            )
            logger.debug(f"Available retrievers: {list(rag.retrievers.keys())}")

            # Determine which retrievers to search
            if index_name and index_name in rag.retrievers:
                retrievers_to_search = {index_name: rag.retrievers[index_name]}
            else:
                retrievers_to_search = rag.retrievers

            if not retrievers_to_search:
                logger.warning("No retrievers available for MCP search_knowledge")
                return "No knowledge indexes are currently loaded. Please index some documents first."

            for name, retriever in retrievers_to_search.items():
                try:
                    logger.debug(
                        f"MCP searching index '{name}' with query: {query[:50] if query else ''}..."
                    )
                    docs = retriever.invoke(query)
                    logger.debug(f"MCP index '{name}' returned {len(docs)} documents")
                    for doc in docs:
                        source = doc.metadata.get("source", "unknown")
                        content = (
                            doc.page_content[:500] + "..."
                            if len(doc.page_content) > 500
                            else doc.page_content
                        )
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

        return MCPToolDefinition(
            name="search_knowledge",
            description=description,
            input_schema=TOOL_INPUT_SCHEMAS["knowledge_search"],
            tool_config={"tool_type": "knowledge_search", "name": "knowledge"},
            execute_fn=search_knowledge,
        )

    def invalidate_cache(self) -> None:
        """Invalidate the heartbeat cache, forcing a fresh check on next call."""
        self._heartbeat_cache = {}
        self._last_heartbeat_check = None
        self._tool_executors = {}


# Global adapter instance
mcp_tool_adapter = MCPToolAdapter()
