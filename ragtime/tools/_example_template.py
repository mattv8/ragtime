"""
Example Tool Template
=====================

Copy this file to create a new tool. The tool will be auto-discovered
if you follow the naming conventions.

Naming conventions:
1. File name: my_tool.py
2. Tool variable: my_tool_tool (or just 'tool')

To enable the tool, add "my_tool" to enabled_tools via the Settings UI
at http://localhost:8001/indexes/ui
"""

import asyncio
from typing import Optional
from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

from ragtime.config import settings
from ragtime.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# INPUT SCHEMA
# =============================================================================
# Define the input parameters your tool accepts.
# Use Pydantic models with Field descriptions for LLM guidance.

class ExampleToolInput(BaseModel):
    """Input schema for the example tool."""

    query: str = Field(
        description="The query or request to process. Be specific about what data you need."
    )
    limit: int = Field(
        default=10,
        description="Maximum number of results to return.",
        ge=1,
        le=100
    )
    format: Optional[str] = Field(
        default="table",
        description="Output format: 'table', 'json', or 'list'"
    )


# =============================================================================
# TOOL IMPLEMENTATION
# =============================================================================

async def execute_example_query(
    query: str,
    limit: int = 10,
    format: str = "table"
) -> str:
    """
    Execute the example tool logic.

    This is where you implement your tool's functionality.
    It should be an async function for non-blocking execution.

    Args:
        query: The user's query.
        limit: Maximum results.
        format: Output format.

    Returns:
        String result to be shown to the user.
    """
    logger.info(f"Example tool called with query: {query[:50]}...")

    # Your implementation here
    # Examples:
    # - Call an external API
    # - Query a database
    # - Execute a command
    # - Process some data

    try:
        # Simulate some async work
        await asyncio.sleep(0.1)

        # Return formatted result
        result = f"Example tool processed: {query}\nLimit: {limit}, Format: {format}"

        return result

    except Exception as e:
        logger.exception("Error in example tool")
        return f"Error: {str(e)}"


# =============================================================================
# TOOL DEFINITION
# =============================================================================
# Create the LangChain StructuredTool. The name should follow the pattern:
# <filename>_tool (e.g., example_tool for example.py)

example_tool = StructuredTool.from_function(
    coroutine=execute_example_query,
    name="example_query",
    description="""Example tool for demonstration purposes.

Use this tool when:
- The user asks for example data
- Testing the tool system

IMPORTANT:
- This is just a template
- Replace with your actual functionality
- Add proper documentation

Example usage: "Get example data with limit 5"
""",
    args_schema=ExampleToolInput
)


# =============================================================================
# ALTERNATIVE: Sync function
# =============================================================================
# If your tool doesn't need async, you can use a regular function:
#
# def sync_example_function(query: str) -> str:
#     return f"Processed: {query}"
#
# example_tool = StructuredTool.from_function(
#     func=sync_example_function,
#     name="example",
#     description="...",
#     args_schema=ExampleToolInput
# )
