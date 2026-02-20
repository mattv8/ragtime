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
from pydantic import BaseModel, Field, model_validator

from ragtime.core.logging import get_logger

logger = get_logger(__name__)


def _normalize_data_connection(
    data_connection: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize data connection metadata to strict component schema."""
    if not isinstance(data_connection, dict):
        return None

    component_id = (
        str(data_connection.get("component_id") or "").strip()
        or str(data_connection.get("source_tool_config_id") or "").strip()
    )
    component_name = (
        str(data_connection.get("component_name") or "").strip()
        or str(data_connection.get("source_tool") or "").strip()
        or None
    )
    component_type = (
        str(data_connection.get("component_type") or "").strip()
        or str(data_connection.get("source_tool_type") or "").strip()
        or None
    )

    request_payload = data_connection.get("request")
    if request_payload is None:
        request_payload = data_connection.get("source_input")
    if request_payload is None:
        request_payload = {}
    if not isinstance(request_payload, dict):
        request_payload = {"value": request_payload}

    refresh_raw = data_connection.get("refresh_interval_seconds")
    refresh_interval_seconds: int | None = None
    if refresh_raw is not None:
        try:
            refresh_interval_seconds = max(1, int(refresh_raw))
        except (TypeError, ValueError):
            refresh_interval_seconds = None

    component_kind = (
        str(data_connection.get("component_kind") or "").strip() or "tool_config"
    )

    normalized: dict[str, Any] = {
        "component_kind": component_kind,
        "request": request_payload,
    }
    if component_id:
        normalized["component_id"] = component_id
    if component_name:
        normalized["component_name"] = component_name
    if component_type:
        normalized["component_type"] = component_type
    if refresh_interval_seconds is not None:
        normalized["refresh_interval_seconds"] = refresh_interval_seconds

    return normalized


class CreateDataTableInput(BaseModel):
    """Input schema for creating an interactive data table."""

    title: str = Field(description="Table title displayed above the table")
    columns: list[str] = Field(description="Column headers for the table")
    data: list[list[Any]] = Field(
        description=(
            "2D array of table data where each inner array is a row. "
            "Can also be provided as 'rows' or as a list of objects with column keys."
        ),
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
    data_connection: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional internal data-connection component reference metadata. "
            "In User Space mode, this should reference admin-configured tool components (for example tool_config IDs)."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, values: Any) -> Any:
        """
        Normalize input to accept common LLM variations:
        - 'rows' as alias for 'data'
        - List of dicts converted to list of lists
        """
        if not isinstance(values, dict):
            return values

        # Accept 'rows' as alias for 'data'
        if "data" not in values and "rows" in values:
            values["data"] = values.pop("rows")

        # Convert list-of-dicts to list-of-lists using column order
        data = values.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            columns = values.get("columns", [])
            if columns:
                values["data"] = [[row.get(col) for col in columns] for row in data]
            else:
                values["data"] = [list(row.values()) for row in data]

        return values


async def create_datatable(
    title: str,
    columns: list[str],
    data: list[list[Any]],
    description: str = "",
    raw_config: dict[str, Any] | None = None,
    data_connection: dict[str, Any] | None = None,
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

    row_count = len(data)
    page_length = 10 if row_count > 10 else max(row_count, 1)

    # Build DataTables configuration
    table_config: dict[str, Any] = {
        "columns": [{"title": col} for col in columns],
        "data": data,
        "pageLength": page_length,
        "searching": row_count > 5,
        "ordering": True,
        "paging": row_count > 10,
        "info": row_count > 5,
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
        "data_connection": _normalize_data_connection(data_connection),
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

REQUIRED PARAMETERS:
- title: Table title string
- columns: Array of column header strings
- data: 2D array of actual row values (REQUIRED - you must pass the data from your query results)

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

When running in User Space mode, follow system prompt instructions for persistent
data_connection components configured by admins in Settings.
""",
    args_schema=CreateDataTableInput,
)
