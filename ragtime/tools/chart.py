"""
Chart visualization tool for UI-only rendering.

This tool allows the AI to create data visualizations that the frontend
renders using Chart.js. It is NOT exposed to MCP or external API clients -
only the internal chat UI uses this tool.

The tool returns a JSON specification that the frontend parses and renders
as an interactive chart inline in the chat stream.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragtime.core.logging import get_logger
from ragtime.userspace.live_data import (
    normalize_live_data_connection,
    validate_live_data_connection,
)

logger = get_logger(__name__)


USERSPACE_CHART_DESCRIPTION_SUFFIX = """

User Space mode override:
- For chat rendering, pass explicit values in labels/datasets for the current response.
- For persistent dashboard files, do NOT hard-code full query snapshots as the long-term data source.
- Persist a live `data_connection` component reference (tool_config-based) with request payload so data can refresh.
- If a live connection cannot be wired with current context, report the blocker instead of silently embedding static data.
"""


CHAT_CHART_DESCRIPTION_SUFFIX = """

Chat mode override:
- For SQL-backed charts, pass the raw successful query result as `source_data` with `columns` and `rows`.
- Do not manually build `labels` or `datasets`; this tool formats Chart.js data from `source_data`.
- `data_connection` must include component_kind=tool_config, component_id, request, and a `result_mapping` that maps row fields to chart labels/datasets.
- In `result_mapping`, use `label_field` for labels and `data_field` or `dataset` for the numeric value field.
- Validate that `source_data.columns` contains every field referenced by `data_connection.result_mapping` before calling this tool.
"""


class ChartDataset(BaseModel):
    """A single dataset for a chart."""

    label: str = Field(description="Label for this dataset (shown in legend)")
    data: list[float | int] = Field(description="Data values for this dataset")
    backgroundColor: str | list[str] | None = Field(
        default=None,
        description="Background color(s) - single color or array for each data point",
    )
    borderColor: str | None = Field(
        default=None, description="Border color for line/bar charts"
    )
    borderWidth: int | None = Field(default=None, description="Border width in pixels")


class CreateChartInput(BaseModel):
    """Input schema for creating a chart visualization."""

    chart_type: Literal[
        "bar", "line", "pie", "doughnut", "scatter", "radar", "polarArea"
    ] = Field(
        description="Type of chart to create: 'bar', 'line', 'pie', 'doughnut', 'scatter', 'radar', or 'polarArea'"
    )
    title: str = Field(description="Chart title displayed above the chart")
    labels: list[str] = Field(
        description="Labels for the X-axis (bar/line) or segments (pie/doughnut)"
    )
    datasets: list[dict[str, Any]] = Field(
        description=(
            "Array of datasets. Each dataset should have: "
            "'label' (string), 'data' (array of numbers), "
            "and optionally 'backgroundColor', 'borderColor', 'borderWidth', 'fill', 'tension'. "
            "Can also be provided as 'dataset' (single dataset) or 'data' (single-series shorthand)."
        )
    )
    description: str = Field(
        default="",
        description="Brief description of what this chart shows (for accessibility)",
    )
    raw_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional Chart.js options for advanced customization. "
            "Use this to customize scales, tooltips, plugins, etc. "
            "The options are merged with the auto-generated config (does not replace data)."
        ),
    )
    data_connection: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional internal data-connection component reference metadata. "
            "When sourced from a query tool, reference the admin-configured tool_config ID, "
            "the exact request payload, and optional result_mapping metadata for refresh."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, values: Any) -> Any:
        """
        Normalize common LLM payload variations:
        - 'dataset' (single dataset object) -> 'datasets'
        - 'data' shorthand -> 'datasets' for single-series charts
        - Chart.js-style nested data object: {'data': {'labels': [...], 'datasets': [...]}}
        """
        if not isinstance(values, dict):
            return values

        # Accept singular alias
        if "datasets" not in values and "dataset" in values:
            dataset_value = values.pop("dataset")
            if isinstance(dataset_value, list):
                values["datasets"] = dataset_value
            elif isinstance(dataset_value, dict):
                values["datasets"] = [dataset_value]

        # Accept Chart.js-style nested payload
        if "datasets" not in values and isinstance(values.get("data"), dict):
            nested_data = values.get("data") or {}
            nested_labels = nested_data.get("labels")
            nested_datasets = nested_data.get("datasets")
            if "labels" not in values and isinstance(nested_labels, list):
                values["labels"] = nested_labels
            if isinstance(nested_datasets, list):
                values["datasets"] = nested_datasets

        # Accept single-series shorthand payload: 'data': [1, 2, 3]
        if "datasets" not in values and "data" in values:
            raw_data = values.get("data")
            if isinstance(raw_data, list):
                if raw_data and isinstance(raw_data[0], dict) and "data" in raw_data[0]:
                    # Already dataset-like payload but under 'data' key
                    values["datasets"] = raw_data
                else:
                    values["datasets"] = [
                        {
                            "label": values.get("title") or "Series 1",
                            "data": raw_data,
                        }
                    ]

        # Ensure datasets is always a list if provided as a dict
        if isinstance(values.get("datasets"), dict):
            values["datasets"] = [values["datasets"]]

        return values


def _coerce_chart_number(value: Any, field: str) -> float | int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Chart field {field} contains non-numeric values")
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        try:
            number = float(cleaned)
        except ValueError as exc:
            raise ValueError(f"Chart field {field} contains non-numeric values") from exc
        return int(number) if number.is_integer() else number
    raise ValueError(f"Chart field {field} contains non-numeric values")


def _source_data_to_rows(source_data: Any) -> tuple[list[str], list[dict[str, Any]]]:
    if not isinstance(source_data, dict):
        raise ValueError("source_data must be an object with columns and rows")

    raw_columns = source_data.get("columns")
    raw_rows = source_data.get("rows", source_data.get("data"))
    if not isinstance(raw_columns, list) or not raw_columns:
        raise ValueError("source_data.columns must be a non-empty array")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("source_data.rows must be a non-empty array")

    columns = [str(column).strip() for column in raw_columns]
    if any(not column for column in columns):
        raise ValueError("source_data.columns cannot contain empty names")

    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if isinstance(raw_row, dict):
            rows.append(dict(raw_row))
        elif isinstance(raw_row, (list, tuple)):
            values = list(raw_row)
            rows.append(
                {
                    column: values[index] if index < len(values) else None
                    for index, column in enumerate(columns)
                }
            )
        else:
            raise ValueError("source_data.rows must contain objects or arrays")

    if not rows:
        raise ValueError("source_data.rows must contain at least one valid row")
    return columns, rows


def _normalize_dataset_mappings(mapping: dict[str, Any]) -> list[dict[str, str]]:
    raw_datasets = mapping.get("datasets")
    normalized: list[dict[str, str]] = []
    root_field = str(
        mapping.get("data_field")
        or mapping.get("dataField")
        or mapping.get("dataset")
        or mapping.get("dataset_field")
        or mapping.get("datasetField")
        or mapping.get("y_field")
        or mapping.get("yField")
        or mapping.get("value_field")
        or mapping.get("valueField")
        or ""
    ).strip()
    if isinstance(raw_datasets, list):
        for entry in raw_datasets:
            if not isinstance(entry, dict):
                continue
            field = str(
                entry.get("data_field")
                or entry.get("dataField")
                or entry.get("dataset")
                or entry.get("dataset_field")
                or entry.get("datasetField")
                or entry.get("field")
                or entry.get("y_field")
                or entry.get("yField")
                or entry.get("value_field")
                or entry.get("valueField")
                or ""
            ).strip()
            if field:
                normalized.append(
                    {"field": field, "label": str(entry.get("label") or field)}
                )
    raw_fields = mapping.get("dataset_fields") or mapping.get("datasetFields")
    if isinstance(raw_fields, dict):
        for label, field_value in raw_fields.items():
            field = str(field_value or "").strip()
            if field:
                normalized.append({"field": field, "label": str(label or field)})
    elif isinstance(raw_fields, list):
        for field_value in raw_fields:
            field = str(field_value or "").strip()
            if field:
                normalized.append({"field": field, "label": field})
    if not normalized and root_field:
        normalized.append(
            {
                "field": root_field,
                "label": str(
                    mapping.get("dataset_label")
                    or mapping.get("label")
                    or root_field
                ),
            }
        )
    return normalized


def _build_chart_data_from_source(
    source_data: Any,
    data_connection: dict[str, Any] | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    normalized_connection = validate_live_data_connection(
        data_connection,
        require_result_mapping=True,
    )
    mapping = normalized_connection.get("result_mapping") or normalized_connection.get(
        "visualization_mapping"
    )
    if not isinstance(mapping, dict):
        raise ValueError("data_connection.result_mapping is required for live charts")

    label_field = str(
        mapping.get("label_field")
        or mapping.get("labelField")
        or mapping.get("x_field")
        or mapping.get("xField")
        or mapping.get("label")
        or ""
    ).strip()
    if not label_field:
        raise ValueError("data_connection.result_mapping is missing label_field")

    columns, rows = _source_data_to_rows(source_data)
    if label_field not in columns and not any(label_field in row for row in rows):
        raise ValueError(f"source_data is missing label field {label_field}")

    dataset_mappings = _normalize_dataset_mappings(mapping)
    if not dataset_mappings:
        raise ValueError("data_connection.result_mapping is missing dataset field mappings")

    labels = [str(row.get(label_field, "")) for row in rows]
    datasets: list[dict[str, Any]] = []
    for dataset_mapping in dataset_mappings:
        field = dataset_mapping["field"]
        if field not in columns and not any(field in row for row in rows):
            raise ValueError(f"source_data is missing dataset field {field}")
        datasets.append(
            {
                "label": str(dataset_mapping.get("label") or field),
                "data": [_coerce_chart_number(row.get(field), field) for row in rows],
            }
        )
    return labels, datasets


class CreateLiveChartInput(BaseModel):
    """Chat-mode chart schema that formats raw query rows."""

    model_config = ConfigDict(extra="forbid")

    chart_type: Literal[
        "bar", "line", "pie", "doughnut", "scatter", "radar", "polarArea"
    ] = Field(
        description="Type of chart to create: 'bar', 'line', 'pie', 'doughnut', 'scatter', 'radar', or 'polarArea'"
    )
    title: str = Field(description="Chart title displayed above the chart")
    source_data: dict[str, Any] = Field(
        description=(
            "Raw successful query result payload with columns and rows. Use the query "
            "tool's returned table data directly; do not pre-build labels or datasets."
        ),
    )
    description: str = Field(
        default="",
        description="Brief description of what this chart shows (for accessibility)",
    )
    raw_config: dict[str, Any] | None = Field(
        default=None,
        description="Optional Chart.js options to merge after chart data is generated.",
    )

    data_connection: dict[str, Any] = Field(
        ...,
        description=(
            "Required live data-connection metadata. Must include component_kind=tool_config, "
            "component_id, the exact successful query payload as request, and result_mapping."
        ),
    )

    @field_validator("data_connection", mode="before")
    @classmethod
    def require_live_data_connection(cls, value: Any) -> dict[str, Any]:
        return validate_live_data_connection(value, require_result_mapping=True)

    @model_validator(mode="after")
    def require_valid_source_mapping(self) -> "CreateLiveChartInput":
        _build_chart_data_from_source(self.source_data, self.data_connection)
        return self


# Default color palettes for charts
CHART_COLORS = [
    "rgba(54, 162, 235, 0.8)",  # Blue
    "rgba(255, 99, 132, 0.8)",  # Red/Pink
    "rgba(75, 192, 192, 0.8)",  # Teal
    "rgba(255, 206, 86, 0.8)",  # Yellow
    "rgba(153, 102, 255, 0.8)",  # Purple
    "rgba(255, 159, 64, 0.8)",  # Orange
    "rgba(199, 199, 199, 0.8)",  # Gray
    "rgba(83, 102, 255, 0.8)",  # Indigo
]

CHART_BORDER_COLORS = [
    "rgba(54, 162, 235, 1)",
    "rgba(255, 99, 132, 1)",
    "rgba(75, 192, 192, 1)",
    "rgba(255, 206, 86, 1)",
    "rgba(153, 102, 255, 1)",
    "rgba(255, 159, 64, 1)",
    "rgba(199, 199, 199, 1)",
    "rgba(83, 102, 255, 1)",
]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """
    Recursively merge override dict into base dict (mutates base in-place).

    For nested dicts, merges recursively. For other values, override replaces base.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _apply_default_colors(
    datasets: list[dict[str, Any]], chart_type: str
) -> list[dict[str, Any]]:
    """Apply default colors to datasets that don't have colors specified."""
    result = []
    for i, dataset in enumerate(datasets):
        ds = dict(dataset)  # Copy to avoid mutating original

        if chart_type in ("pie", "doughnut"):
            # Pie/doughnut charts need an array of colors for segments
            if not ds.get("backgroundColor"):
                num_segments = len(ds.get("data", []))
                ds["backgroundColor"] = [
                    CHART_COLORS[j % len(CHART_COLORS)] for j in range(num_segments)
                ]
            if not ds.get("borderColor"):
                ds["borderColor"] = "rgba(255, 255, 255, 1)"
            if ds.get("borderWidth") is None:
                ds["borderWidth"] = 2
        else:
            # Bar/line charts use single colors per dataset
            if not ds.get("backgroundColor"):
                ds["backgroundColor"] = CHART_COLORS[i % len(CHART_COLORS)]
            if not ds.get("borderColor"):
                ds["borderColor"] = CHART_BORDER_COLORS[i % len(CHART_BORDER_COLORS)]
            if ds.get("borderWidth") is None:
                ds["borderWidth"] = 1

        result.append(ds)

    return result


async def create_chart(
    chart_type: str,
    title: str,
    labels: list[str] | None = None,
    datasets: list[dict[str, Any]] | None = None,
    description: str = "",
    raw_config: dict[str, Any] | None = None,
    data_connection: dict[str, Any] | None = None,
    source_data: dict[str, Any] | None = None,
) -> str:
    """
    Create a chart visualization specification.

    This tool generates a Chart.js-compatible configuration that the frontend
    renders as an interactive chart. Use this when data would be better
    understood as a visual representation.

    Args:
        chart_type: Type of chart ('bar', 'line', 'pie', 'doughnut', 'scatter', 'radar', 'polarArea')
        title: Chart title
        labels: X-axis labels (bar/line) or segment labels (pie/doughnut)
        datasets: Array of datasets with label, data, and optional colors
        description: Accessibility description
        raw_config: Optional raw Chart.js config for advanced customization

    Returns:
        JSON string containing the chart specification for frontend rendering.
    """
    if source_data is not None:
        try:
            labels, datasets = _build_chart_data_from_source(source_data, data_connection)
        except ValueError as exc:
            return f"Error: Cannot create chart '{title}' - {exc}"

    logger.info(f"Creating {chart_type} chart: {title}")

    # Validate that we have actual data
    if not labels or len(labels) == 0:
        return f"Error: Cannot create chart '{title}' - no labels provided. Charts require actual data from query results."

    if not datasets or len(datasets) == 0:
        return f"Error: Cannot create chart '{title}' - no datasets provided. Charts require actual data from query results."

    # Check if datasets have valid data points (not just empty arrays or placeholder values)
    total_data_points = 0
    for ds in datasets:
        data = ds.get("data", [])
        if not data:
            continue
        for point in data:
            # For scatter charts, check if x/y objects have actual values
            if isinstance(point, dict):
                x_val = point.get("x")
                y_val = point.get("y")
                if x_val is not None and y_val is not None:
                    total_data_points += 1
            # For other chart types, count valid numeric values
            elif isinstance(point, (int, float)):
                total_data_points += 1

    if total_data_points == 0:
        return f"Error: Cannot create chart '{title}' - no valid data points found. Charts require actual numeric data from query results."

    # Apply default colors if not specified
    colored_datasets = _apply_default_colors(datasets, chart_type)

    # Build Chart.js configuration
    chart_config: dict[str, Any] = {
        "type": chart_type,
        "data": {
            "labels": labels,
            "datasets": colored_datasets,
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": True,
            "plugins": {
                "title": {
                    "display": True,
                    "text": title,
                    "font": {"size": 16, "weight": "bold"},
                },
                "legend": {
                    "display": len(colored_datasets) > 1
                    or chart_type in ("pie", "doughnut", "polarArea"),
                    "position": "bottom",
                },
            },
        },
    }

    # Add axis labels for bar/line/scatter charts
    if chart_type in ("bar", "line", "scatter"):
        options = chart_config["options"]
        if isinstance(options, dict):
            options["scales"] = {
                "y": {"beginAtZero": True},
            }

    # Merge raw_config options if provided (for custom axis labels, tooltips, etc.)
    if raw_config:
        # Deep merge the options - raw_config.options overrides/extends our defaults
        if "options" in raw_config and isinstance(raw_config["options"], dict):
            _deep_merge(chart_config["options"], raw_config["options"])
        # Allow overriding data section if explicitly provided in raw_config
        if "data" in raw_config and isinstance(raw_config["data"], dict):
            _deep_merge(chart_config["data"], raw_config["data"])

    # Wrap in our marker format for frontend detection
    output = {
        "__chart__": True,
        "config": chart_config,
        "description": description,
        "data_connection": normalize_live_data_connection(data_connection),
    }

    return json.dumps(output, indent=2)


# Create LangChain tool - this is only added to the agent for UI requests
create_chart_tool = StructuredTool.from_function(
    coroutine=create_chart,
    name="create_chart",
    handle_validation_error=True,
    description="""Create a data visualization chart to display to the user.
PROACTIVELY use this tool when you retrieve numeric data that could be visualized.

Best used for:
- Comparing values across categories (bar chart)
- Showing trends over time (line chart)
- Showing proportions of a whole (pie/doughnut chart)
- Multi-dimensional comparisons (radar chart)
- Correlation analysis (scatter chart)

Chart types:
- 'bar': Compare values across categories
- 'line': Show trends/changes over time
- 'pie': Show parts of a whole (use for <7 categories)
- 'doughnut': Like pie but with center hole
- 'scatter': Show correlation between variables
- 'radar': Multi-dimensional comparison
- 'polarArea': Like pie but with equal angles

Example for bar chart:
{
  "chart_type": "bar",
  "title": "Sales by Region",
  "labels": ["North", "South", "East", "West"],
  "datasets": [{"label": "Q1 Sales", "data": [120, 190, 300, 250]}]
}

For advanced customization, use raw_config with any Chart.js options:
{
  "chart_type": "line",
  "title": "Revenue Trend",
  "labels": ["Jan", "Feb", "Mar"],
  "datasets": [{"label": "Revenue", "data": [100, 150, 200]}],
  "raw_config": {
    "options": {
      "scales": {"y": {"title": {"display": true, "text": "USD"}}},
      "plugins": {"tooltip": {"mode": "index"}}
    }
  }
}

""",
    args_schema=CreateChartInput,
)
