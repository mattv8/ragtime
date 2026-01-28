"""
Tool definitions for LangChain agents.

Tools are auto-discovered from this package. To add a new tool:
1. Create a new file (e.g., my_tool.py)
2. Define a StructuredTool named: my_tool_tool (or just 'tool')
3. Enable "my_tool" via the Settings UI at /indexes/ui

See registry.py for the discovery mechanism.
"""

from .registry import (
    register_tool,
    get_tool,
    get_all_tools,
    get_enabled_tools,
    discover_tools,
)

# Re-export individual tools for direct import if needed
# NOTE: odoo_tool is deprecated - use tool configs instead
from .postgres import postgres_tool, PostgresQueryInput, execute_postgres_query

__all__ = [
    # Registry functions
    "register_tool",
    "get_tool",
    "get_all_tools",
    "get_enabled_tools",
    "discover_tools",
    # Individual tools (postgres kept for backwards compatibility)
    "postgres_tool",
    "PostgresQueryInput",
    "execute_postgres_query",
]
