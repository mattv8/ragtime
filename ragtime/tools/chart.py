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
from pydantic import BaseModel, Field

from ragtime.core.logging import get_logger

logger = get_logger(__name__)


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
            "and optionally 'backgroundColor', 'borderColor', 'borderWidth', 'fill', 'tension'"
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
    labels: list[str],
    datasets: list[dict[str, Any]],
    description: str = "",
    raw_config: dict[str, Any] | None = None,
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
    chart_config = {
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
    }

    return json.dumps(output, indent=2)


# Create LangChain tool - this is only added to the agent for UI requests
create_chart_tool = StructuredTool.from_function(
    coroutine=create_chart,
    name="create_chart",
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
