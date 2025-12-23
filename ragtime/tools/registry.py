"""
Tool registry with auto-discovery.

Tools are automatically discovered from this package. To add a new tool:
1. Create a new file in ragtime/tools/ (e.g., my_tool.py)
2. Define a StructuredTool named following the pattern: my_tool_tool
3. Enable "my_tool" via the Settings UI at /indexes/ui

The registry will automatically find and load it.
"""

import importlib
import pkgutil
from typing import Dict, List, Optional

from langchain_core.tools import StructuredTool

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Global tool registry
_tools: Dict[str, StructuredTool] = {}


def register_tool(name: str, tool: StructuredTool) -> None:
    """Register a tool in the global registry."""
    _tools[name] = tool
    logger.debug(f"Registered tool: {name}")


def get_tool(name: str) -> Optional[StructuredTool]:
    """Get a tool by name."""
    return _tools.get(name)


def get_all_tools() -> Dict[str, StructuredTool]:
    """Get all registered tools."""
    return _tools.copy()


def get_enabled_tools(enabled_list: List[str]) -> List[StructuredTool]:
    """
    Get tools that are in the enabled list.

    Args:
        enabled_list: List of tool names to enable (e.g., ["odoo", "postgres"])

    Returns:
        List of enabled StructuredTool instances.
    """
    tools = []
    for name in enabled_list:
        name = name.strip().lower()
        if name in _tools:
            tools.append(_tools[name])
            logger.info(f"Enabled tool: {name}")
        else:
            logger.warning(f"Tool not found: {name} (available: {list(_tools.keys())})")
    return tools


def discover_tools() -> None:
    """
    Auto-discover and register all tools in the tools package.

    Looks for StructuredTool instances in each module that follow
    the naming convention: <module_name>_tool
    """
    import ragtime.tools as tools_package

    logger.info("Discovering tools...")

    for importer, modname, ispkg in pkgutil.iter_modules(tools_package.__path__):
        if modname.startswith("_") or modname == "registry":
            continue

        try:
            module = importlib.import_module(f"ragtime.tools.{modname}")

            # Look for tool following naming convention
            tool_attr_name = f"{modname}_tool"
            if hasattr(module, tool_attr_name):
                tool = getattr(module, tool_attr_name)
                if isinstance(tool, StructuredTool):
                    register_tool(modname, tool)

            # Also check for a 'tool' attribute (simpler naming)
            elif hasattr(module, "tool"):
                tool = getattr(module, "tool")
                if isinstance(tool, StructuredTool):
                    register_tool(modname, tool)

        except Exception as e:
            logger.error(f"Failed to load tool module {modname}: {e}")

    logger.info(f"Discovered {len(_tools)} tools: {list(_tools.keys())}")


# Auto-discover on import
discover_tools()
