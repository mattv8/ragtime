"""
DataTable visualization tool for UI-only rendering.

This tool allows the AI to create interactive data tables that the frontend
renders using DataTables.js. It is NOT exposed to MCP or external API clients -
only the internal chat UI uses this tool.

The tool returns a JSON specification that the frontend parses and renders
as an interactive table with sorting, searching, and pagination.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ragtime.core.logging import get_logger

logger = get_logger(__name__)


class CreateDataTableInput(BaseModel):
    """Input schema for creating an interactive data table."""

    title: str = Field(description="Table title displayed above the table")
    columns: list[str] = Field(description="Column headers for the table")
    data: list[list[Any]] = Field(
        description="2D array of table data - each inner array is a row"
    )
    description: str = Field(
        default="",
        description="Brief description of what this table shows",
    )
    raw_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional raw DataTables.js configuration object for advanced customization. "
            "Allows full access to DataTables options like columnDefs, order, pageLength, etc."
        ),
    )


async def create_datatable(
    title: str,
    columns: list[str],
    data: list[list[Any]],
    description: str = "",
    raw_config: dict[str, Any] | None = None,
) -> str:
    """
    Create an interactive data table specification.

    This tool generates a DataTables.js-compatible configuration that the frontend
    renders as an interactive table with sorting, searching, and pagination.

    Args:
        title: Table title
        columns: Column headers
        data: 2D array of row data
        description: Accessibility description
        raw_config: Optional raw DataTables.js config for advanced customization

    Returns:
        JSON string containing the table specification for frontend rendering.
    """
    logger.info(f"Creating datatable: {title} ({len(data)} rows)")

    # Build DataTables configuration
    table_config: dict[str, Any] = {
        "columns": [{"title": col} for col in columns],
        "data": data,
        "pageLength": 10 if len(data) > 10 else len(data),
        "searching": len(data) > 5,
        "ordering": True,
        "paging": len(data) > 10,
        "info": len(data) > 5,
    }

    # Merge in raw config if provided (allows full customization)
    if raw_config:
        table_config.update(raw_config)

    # Wrap in our marker format for frontend detection
    output = {
        "__datatable__": True,
        "title": title,
        "config": table_config,
        "description": description,
    }

    return json.dumps(output, indent=2)


# Create LangChain tool - this is only added to the agent for UI requests
create_datatable_tool = StructuredTool.from_function(
    coroutine=create_datatable,
    name="create_datatable",
    description="""Create an interactive data table with sorting, searching, and pagination.
Use this tool when presenting tabular data that the user might want to explore.

Best used for:
- Query results with multiple columns and rows
- Data that users might want to sort or search
- Lists of records, items, or entities
- Any tabular data with >3 rows

The table will have:
- Sortable columns (click header to sort)
- Search/filter box (for larger tables)
- Pagination (for tables with many rows)
- Responsive design

Example:
{
  "title": "Active Users",
  "columns": ["ID", "Name", "Email", "Status"],
  "data": [
    [1, "Alice", "alice@example.com", "Active"],
    [2, "Bob", "bob@example.com", "Active"],
    [3, "Carol", "carol@example.com", "Pending"]
  ]
}

For advanced customization, use raw_config with DataTables.js options:
{
  "title": "Custom Table",
  "columns": ["Name", "Value"],
  "data": [["A", 100], ["B", 200]],
  "raw_config": {
    "order": [[1, "desc"]],
    "pageLength": 25,
    "columnDefs": [{"className": "text-right", "targets": [1]}]
  }
}
""",
    args_schema=CreateDataTableInput,
)
