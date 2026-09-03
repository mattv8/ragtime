"""Shared registry of UI-only chat visualization tool names and types.

Chart, datatable, and HTML component tools are rendered inline by the chat UI
from a JSON envelope carrying a marker key. This module is the single source of
truth for their names so backend registration, live refresh, export context,
and prompts stay in sync. It intentionally has no imports from ``ragtime.tools``
or ``ragtime.rag`` to avoid import cycles.
"""

from __future__ import annotations

from typing import Literal

CHART_TOOL_NAME = "create_chart"
DATATABLE_TOOL_NAME = "create_datatable"
HTML_COMPONENT_TOOL_NAME = "create_html_component"

VISUALIZATION_TOOL_NAMES: frozenset[str] = frozenset({CHART_TOOL_NAME, DATATABLE_TOOL_NAME, HTML_COMPONENT_TOOL_NAME})

VisualizationToolType = Literal["chart", "datatable", "html_component"]
RetryableVisualizationToolType = Literal["chart", "datatable"]

VISUALIZATION_TOOL_NAME_BY_TYPE: dict[str, str] = {
    "chart": CHART_TOOL_NAME,
    "datatable": DATATABLE_TOOL_NAME,
    "html_component": HTML_COMPONENT_TOOL_NAME,
}

VISUALIZATION_MARKER_BY_TYPE: dict[str, str] = {
    "chart": "__chart__",
    "datatable": "__datatable__",
    "html_component": "__html_component__",
}


def visualization_tool_name_for_type(tool_type: str) -> str:
    """Return the tool name for a visualization type, raising on unknown types."""
    try:
        return VISUALIZATION_TOOL_NAME_BY_TYPE[tool_type]
    except KeyError as exc:
        raise ValueError(f"Unknown visualization tool type: {tool_type!r}") from exc


def is_visualization_tool_name(tool_name: str | None) -> bool:
    """Return True when the tool name is one of the inline chat visualization tools."""
    return bool(tool_name) and tool_name in VISUALIZATION_TOOL_NAMES
