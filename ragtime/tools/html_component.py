"""
Sandboxed HTML component tool for UI-only rendering.

This tool lets the AI build arbitrary HTML/JS/CSS components (maps, heatmaps,
network graphs, gauges, timelines, custom layouts) that the frontend renders
inline in chat inside a sandboxed iframe. It is NOT exposed to MCP or external
API clients - only the internal chat UI uses this tool.

The tool returns a JSON envelope (marker key ``__html_component__``) that the
frontend parses. Tabular data travels separately from the markup so live
refresh can replace ``data`` and push it into the running component without
reloading the frame.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator, model_validator

from ragtime.core.logging import get_logger
from ragtime.core.visualization_tools import HTML_COMPONENT_TOOL_NAME
from ragtime.tools.chart import source_data_to_rows
from ragtime.tools.visualization_validation import format_visualization_validation_error
from ragtime.userspace.live_data import (
    normalize_live_data_connection,
    validate_live_data_connection,
)

logger = get_logger(__name__)


HTML_COMPONENT_MAX_HTML_CHARS = 200_000
HTML_COMPONENT_MAX_DATA_CHARS = 1_000_000
HTML_COMPONENT_MIN_HEIGHT = 200
HTML_COMPONENT_MAX_HEIGHT = 1200

# Mixed content: the chat app is served over https, so http:// assets are blocked by the browser anyway.
_HTTP_ASSET_URL_PATTERN = re.compile(r"""(?:src|href)\s*=\s*["']http://""", re.IGNORECASE)
_HTML_TAG_PATTERN = re.compile(r"<html[\s>]", re.IGNORECASE)

_HTML_ALIAS_KEYS = ("code", "markup", "content")


USERSPACE_HTML_COMPONENT_DESCRIPTION_SUFFIX = """

User Space mode override: this tool is not available for persistent dashboards; build User Space modules with the runtime bridge instead.
"""


CHAT_HTML_COMPONENT_DESCRIPTION_SUFFIX = """

Chat mode override:
- Prefer create_chart or create_datatable whenever they can express the visual; call this tool only for maps, heatmaps, network graphs, gauges, timelines, and custom layouts they cannot.
- For query-backed components, pass the raw successful query result as `source_data` with `columns` and `rows`, and read it in the HTML from `window.ragtime.data` (subscribe with `window.ragtime.onData`).
- Include `data_connection` with component_kind=tool_config, component_id, and the exact request payload used to fetch the rows so Refresh can re-run the query.
- `data_connection` must reference an executable selected tool component. Do not use descriptive metadata such as component_kind=web_research, research dates, source labels, or notes in place of component_id/request.
- For static or synthesized reference data (coordinates, thresholds, lookups), pass it as `data` and omit `data_connection`.
- Never call fetch() for Ragtime data from the component; the iframe is sandboxed and has no API access.
"""


HTML_COMPONENT_EXPECTED_INPUT_SHAPE = """Live/query-backed chat component:
{
    "title": "Component title",
    "html": "<div id=\\"root\\"></div><script>window.ragtime.onData(function (data) { /* render data.rows */ });</script>",
    "source_data": {"columns": ["column_a"], "rows": [["value"]]},
    "data_connection": {
        "component_kind": "tool_config",
        "component_id": "<selected ToolConfig ID>",
        "request": {"query": "<exact successful query payload>"}
    },
    "height": 420
}

Static component when live refresh is not required:
{
    "title": "Component title",
    "html": "<div id=\\"root\\"></div><script>/* read window.ragtime.data */</script>",
    "data": {"any": "json"}
}"""


def _format_html_component_validation_error(error: Exception) -> str:
    return format_visualization_validation_error(
        error,
        tool_name=HTML_COMPONENT_TOOL_NAME,
        expected_shape=HTML_COMPONENT_EXPECTED_INPUT_SHAPE,
        guidance=[
            "Pass the markup in `html` (a fragment or a full document); load libraries only from https:// CDN URLs.",
            "For query-backed components, pass the raw successful query result in source_data.columns/source_data.rows and read it from window.ragtime.data.",
            "data_connection must identify a real selected tool_config component with component_id and the exact successful request payload.",
            f"Keep html under {HTML_COMPONENT_MAX_HTML_CHARS:,} characters, data under {HTML_COMPONENT_MAX_DATA_CHARS:,} characters, and height between {HTML_COMPONENT_MIN_HEIGHT} and {HTML_COMPONENT_MAX_HEIGHT}.",
            "If create_chart or create_datatable can express the visual, use that tool instead of retrying this one.",
        ],
    )


def _validate_html_markup(html: Any) -> str:
    """Validate raw component markup, raising ValueError with model-facing feedback."""
    if not isinstance(html, str) or not html.strip():
        raise ValueError("html must be a non-empty string")
    if len(html) > HTML_COMPONENT_MAX_HTML_CHARS:
        raise ValueError(f"html exceeds {HTML_COMPONENT_MAX_HTML_CHARS:,} characters")
    if _HTTP_ASSET_URL_PATTERN.search(html):
        raise ValueError("html must not reference http:// assets in src/href attributes (mixed content); use https:// URLs")
    return html


def _wrap_html_document(html: str) -> str:
    """Return a full HTML document, wrapping fragments that lack an <html> tag."""
    if _HTML_TAG_PATTERN.search(html):
        return html
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"></head>'
        f"<body>{html}</body></html>"
    )


def normalize_component_html(html: Any) -> str:
    """Validate user- or model-supplied markup and return it as a full document.

    Shared by the tool and the in-chat source editor so edited components obey the
    same limits and mixed-content rules as freshly generated ones.
    """
    return _wrap_html_document(_validate_html_markup(html))


def _is_tabular_payload(data: Any) -> bool:
    return isinstance(data, dict) and isinstance(data.get("columns"), list) and isinstance(data.get("rows"), list)


def _canonicalize_tabular(source: Any) -> dict[str, Any]:
    columns, rows = source_data_to_rows(source)
    return {"columns": columns, "rows": rows, "row_count": len(rows)}


class CreateHtmlComponentInput(BaseModel):
    """Input schema for creating a sandboxed inline HTML component."""

    title: str = Field(description="Component title displayed in the header above the component")
    html: str = Field(
        description=(
            "HTML/JS/CSS for the component. A fragment is wrapped into a full document automatically. "
            "Load libraries only via https CDN <script>/<link> tags. Read data from window.ragtime.data."
        ),
    )
    data: dict[str, Any] | list[Any] | None = Field(
        default=None,
        description=(
            "Optional static or synthesized JSON data exposed to the component as window.ragtime.data. Use source_data instead for query-backed rows."
        ),
    )
    source_data: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Raw successful query result payload with columns and rows. Canonicalized to {columns, rows, row_count} "
            "and exposed as window.ragtime.data; rows are objects keyed by column name."
        ),
    )
    description: str = Field(
        default="",
        description="Brief description of what this component shows (for accessibility)",
    )
    height: int | None = Field(
        default=None,
        ge=HTML_COMPONENT_MIN_HEIGHT,
        le=HTML_COMPONENT_MAX_HEIGHT,
        description=f"Optional initial frame height hint in pixels ({HTML_COMPONENT_MIN_HEIGHT}-{HTML_COMPONENT_MAX_HEIGHT}); the frame auto-sizes to content afterwards.",
    )
    data_connection: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional live data-connection metadata. When source_data came from a query tool, reference the "
            "admin-configured tool_config ID and exact request payload so Refresh can re-run the query."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, values: Any) -> Any:
        """
        Normalize common LLM payload variations:
        - 'code' / 'markup' / 'content' -> 'html'
        - top-level 'columns' + 'rows' -> 'source_data'

        The dict is mutated in place on purpose: LangChain's StructuredTool only
        forwards keys present in the original tool input to the coroutine.
        """
        if not isinstance(values, dict):
            return values

        if "html" not in values:
            for alias in _HTML_ALIAS_KEYS:
                if alias in values:
                    values["html"] = values.pop(alias)
                    break

        if "source_data" not in values and isinstance(values.get("columns"), list) and isinstance(values.get("rows"), list):
            values["source_data"] = {"columns": values.pop("columns"), "rows": values.pop("rows")}

        return values

    @field_validator("html", mode="before")
    @classmethod
    def validate_html(cls, value: Any) -> str:
        return _validate_html_markup(value)


async def create_html_component(
    title: str,
    html: str,
    data: dict[str, Any] | list[Any] | None = None,
    source_data: dict[str, Any] | None = None,
    description: str = "",
    height: int | None = None,
    data_connection: dict[str, Any] | None = None,
) -> str:
    """
    Create a sandboxed inline HTML component specification.

    Args:
        title: Component title
        html: Markup (fragment or full document)
        data: Static/synthesized JSON data, or a {columns, rows} table to canonicalize
        source_data: Raw query result ({columns, rows}) canonicalized to tabular data
        description: Accessibility description
        height: Initial frame height hint (200-1200)
        data_connection: Live data-connection metadata for refresh

    Returns:
        JSON string containing the component envelope for frontend rendering.
    """

    def _error(message: str) -> str:
        return f"Error: Cannot create component '{title}' - {message}"

    try:
        html = _validate_html_markup(html)
    except ValueError as exc:
        return _error(str(exc))

    if height is not None and not (HTML_COMPONENT_MIN_HEIGHT <= height <= HTML_COMPONENT_MAX_HEIGHT):
        return _error(f"height must be between {HTML_COMPONENT_MIN_HEIGHT} and {HTML_COMPONENT_MAX_HEIGHT}")

    payload: dict[str, Any] | list[Any] | None = data
    try:
        if source_data is not None:
            payload = _canonicalize_tabular(source_data)
        elif _is_tabular_payload(data):
            payload = _canonicalize_tabular(data)
    except ValueError as exc:
        return _error(str(exc))

    normalized_connection: dict[str, Any] | None = None
    if data_connection is not None:
        try:
            validate_live_data_connection(data_connection)
        except ValueError as exc:
            return _error(str(exc))
        if not _is_tabular_payload(payload):
            return _error("data_connection requires tabular source_data (columns/rows) so refresh can replace window.ragtime.data")
        normalized_connection = normalize_live_data_connection(data_connection)

    if payload is not None and len(json.dumps(payload, default=str)) > HTML_COMPONENT_MAX_DATA_CHARS:
        return _error(f"data exceeds {HTML_COMPONENT_MAX_DATA_CHARS:,} characters")

    row_count = payload.get("row_count") if isinstance(payload, dict) else None
    logger.info(f"Creating html component: {title} ({row_count if row_count is not None else 'non-tabular'} rows)")

    output = {
        "__html_component__": True,
        "title": title,
        "html": _wrap_html_document(html),
        "data": payload,
        "description": description,
        "height": height,
        "data_connection": normalized_connection,
    }

    return json.dumps(output, indent=2)


# Create LangChain tool - this is only added to the agent for UI requests.
# Deliberately NOT named html_component_tool so ragtime/tools/registry.py autodiscovery skips it.
create_html_component_tool = StructuredTool.from_function(
    coroutine=create_html_component,
    name=HTML_COMPONENT_TOOL_NAME,
    handle_validation_error=_format_html_component_validation_error,
    description=f"""Create a custom sandboxed HTML/JS component rendered inline in the chat.
Use this ONLY when create_chart or create_datatable cannot express the visual: maps and geographic heatmaps,
network/graph diagrams, gauges, timelines, custom dashboards or layouts.

Rules:
- Prefer create_chart/create_datatable whenever they fit; use this tool only for visuals they cannot express.
- The component runs in a sandboxed iframe: no cookies, storage, Ragtime API calls, or navigation.
  Load libraries only via https CDN <script>/<link> tags (Leaflet, D3, ECharts, etc.); http:// assets are rejected.
- Never call fetch() for Ragtime data. Read `window.ragtime.data` and subscribe with `window.ragtime.onData(cb)`
  so Refresh re-renders with new rows without reloading the frame.
- Make the root container responsive (100% width) with an explicit pixel height (for example `#map {{ height: 420px }}`); avoid `100vh`.
- Use `window.ragtime.theme.tokens` or the `--ragtime-*` CSS variables for colors and fonts so the component matches light/dark mode.

Data:
- Query-backed rows go in `source_data` ({{"columns": [...], "rows": [...]}}) and are exposed as
  `window.ragtime.data` = {{"columns": [...], "rows": [{{column: value}}], "row_count": n}}.
- Add `data_connection` (component_kind=tool_config, component_id, exact request) when the rows came from a selected query tool
  so the user can refresh; refresh replaces only the data, not the html.
- Static or synthesized reference data (coordinates, thresholds) goes in `data`; omit `data_connection` for it.
- A fragment is wrapped into a full document automatically; a full document is passed through unchanged.

Limits: html <= {HTML_COMPONENT_MAX_HTML_CHARS // 1000} KB, data <= {HTML_COMPONENT_MAX_DATA_CHARS // 1_000_000} MB,
`height` {HTML_COMPONENT_MIN_HEIGHT}-{HTML_COMPONENT_MAX_HEIGHT} (optional initial frame height; the frame auto-sizes to content).

Example (Leaflet heatmap from a query):
{{
  "title": "Shipments by origin",
  "description": "Shipment origins, last 30 days",
  "height": 480,
  "source_data": {{"columns": ["lat", "lng", "shipments"], "rows": [[31.9, -99.9, 412], [40.7, -74.0, 288]]}},
  "data_connection": {{
    "component_kind": "tool_config",
    "component_id": "<selected ToolConfig ID>",
    "request": {{"query": "SELECT lat, lng, COUNT(*) AS shipments FROM shipments GROUP BY lat, lng"}}
  }},
  "html": "<link rel=\\"stylesheet\\" href=\\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\\">
<script src=\\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\\"></script>
<script src=\\"https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js\\"></script>
<style>#map {{ width: 100%; height: 460px; }}</style>
<div id=\\"map\\"></div>
<script>
  var map = L.map('map').setView([39.5, -98.35], 4);
  L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ attribution: '&copy; OpenStreetMap' }}).addTo(map);
  var heat = L.heatLayer([], {{ radius: 25 }}).addTo(map);
  window.ragtime.onData(function (data) {{
    heat.setLatLngs((data && data.rows || []).map(function (r) {{ return [r.lat, r.lng, r.shipments]; }}));
  }});
</script>"
}}

""",
    args_schema=CreateHtmlComponentInput,
)
