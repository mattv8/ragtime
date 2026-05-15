"""Live component-backed refresh helpers for chat visualizations."""


from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Literal

from ragtime.userspace.live_data import normalize_live_data_connection
from ragtime.userspace.models import ExecuteComponentRequest, ExecuteComponentResponse

class LiveVisualizationRefreshError(ValueError):
    """Raised when a visualization cannot be refreshed from live metadata."""


def _load_visualization_payload(output: str, tool_type: Literal["chart", "datatable"]) -> dict[str, Any]:
    try:
        payload = json.loads(output)
    except Exception as exc:
        raise LiveVisualizationRefreshError("Visualization output is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise LiveVisualizationRefreshError("Visualization output is not an object.")

    marker = "__chart__" if tool_type == "chart" else "__datatable__"
    if payload.get(marker) is not True or not isinstance(payload.get("config"), dict):
        raise LiveVisualizationRefreshError("Target event is not a refreshable visualization payload.")
    return payload


def _extract_data_connection(payload: dict[str, Any]) -> dict[str, Any]:
    connection = payload.get("data_connection")
    if not isinstance(connection, dict):
        raise LiveVisualizationRefreshError("This visualization does not have live data metadata.")
    return connection


def build_component_request_from_visualization(
    output: str,
    tool_type: Literal["chart", "datatable"],
) -> ExecuteComponentRequest:
    """Build an ExecuteComponentRequest from a persisted visualization payload."""
    payload = _load_visualization_payload(output, tool_type)
    connection = _extract_data_connection(payload)
    try:
        normalized = normalize_live_data_connection(
            connection,
            require_component_id=True,
            require_request=True,
        )
    except ValueError as exc:
        raise LiveVisualizationRefreshError(str(exc)) from exc

    assert normalized is not None
    component_id = normalized["component_id"]
    request_payload = normalized["request"]
    if not isinstance(request_payload, (dict, str)):
        raise LiveVisualizationRefreshError("Live data request payload must be an object or string.")

    return ExecuteComponentRequest(component_id=component_id, request=request_payload)


def render_refreshed_visualization(
    *,
    original_output: str,
    tool_type: Literal["chart", "datatable"],
    component_response: ExecuteComponentResponse,
) -> str:
    """Render a refreshed visualization using returned component rows.

    This preserves the existing Chart.js/DataTables payload shape and options,
    replacing only the data-bearing fields.
    """
    if component_response.error:
        raise LiveVisualizationRefreshError(component_response.error)
    payload = _load_visualization_payload(original_output, tool_type)
    if tool_type == "datatable":
        refreshed = _refresh_datatable_payload(payload, component_response)
    else:
        refreshed = _refresh_chart_payload(payload, component_response)
    return json.dumps(refreshed, indent=2, default=str)


def _refresh_datatable_payload(
    payload: dict[str, Any],
    component_response: ExecuteComponentResponse,
) -> dict[str, Any]:
    refreshed = deepcopy(payload)
    config = dict(refreshed.get("config") or {})
    columns = list(component_response.columns or [])
    if not columns and component_response.rows:
        columns = list(component_response.rows[0].keys())
    if not columns:
        raise LiveVisualizationRefreshError("Live query returned no columns for the table.")

    existing_columns = config.get("columns") if isinstance(config.get("columns"), list) else None
    if (
        existing_columns
        and len(existing_columns) == len(columns)
        and all(isinstance(col, dict) for col in existing_columns)
    ):
        # Preserve LLM-provided column metadata (titles, render hints) when the
        # SQL column count is unchanged; refresh only the row data below.
        config["columns"] = [dict(col) for col in existing_columns]
    else:
        config["columns"] = [{"title": column} for column in columns]
    config["data"] = [
        [row.get(column) for column in columns]
        for row in component_response.rows
    ]
    config.setdefault("pageLength", 10 if len(component_response.rows) > 10 else max(len(component_response.rows), 1))
    config.setdefault("searching", len(component_response.rows) > 5)
    config.setdefault("ordering", True)
    config.setdefault("paging", len(component_response.rows) > 10)
    config.setdefault("info", len(component_response.rows) > 5)
    refreshed["config"] = config
    return refreshed


def _refresh_chart_payload(
    payload: dict[str, Any],
    component_response: ExecuteComponentResponse,
) -> dict[str, Any]:
    connection = _extract_data_connection(payload)
    mapping = connection.get("result_mapping") or connection.get("visualization_mapping")
    if not isinstance(mapping, dict):
        raise LiveVisualizationRefreshError("Chart live data metadata is missing result_mapping.")

    label_field = str(
        mapping.get("label_field")
        or mapping.get("labelField")
        or mapping.get("x_field")
        or mapping.get("xField")
        or mapping.get("label")
        or ""
    ).strip()
    if not label_field:
        raise LiveVisualizationRefreshError("Chart result_mapping is missing label_field.")

    dataset_mappings = _normalize_dataset_mappings(mapping)
    if not dataset_mappings:
        raise LiveVisualizationRefreshError("Chart result_mapping is missing dataset field mappings.")

    refreshed = deepcopy(payload)
    config = dict(refreshed.get("config") or {})
    data = dict(config.get("data") or {})
    rows = component_response.rows
    data["labels"] = [str(row.get(label_field, "")) for row in rows]

    existing_datasets = data.get("datasets")
    if not isinstance(existing_datasets, list):
        existing_datasets = []

    datasets: list[dict[str, Any]] = []
    for index, dataset_mapping in enumerate(dataset_mappings):
        field = dataset_mapping["field"]
        base = dict(existing_datasets[index]) if index < len(existing_datasets) and isinstance(existing_datasets[index], dict) else {}
        base["label"] = str(dataset_mapping.get("label") or base.get("label") or field)
        base["data"] = [_coerce_chart_number(row.get(field), field) for row in rows]
        datasets.append(base)

    data["datasets"] = datasets
    config["data"] = data
    refreshed["config"] = config
    return refreshed


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
    root_mapping: dict[str, str] | None = None
    if root_field:
        root_mapping = {
            "field": root_field,
            "label": str(mapping.get("dataset_label") or mapping.get("label") or root_field),
        }
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
                normalized.append({"field": field, "label": str(entry.get("label") or field)})
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
    if not normalized and root_mapping:
        normalized.append(root_mapping)
    return normalized


def _coerce_chart_number(value: Any, field: str) -> float | int:
    if isinstance(value, bool) or value is None:
        raise LiveVisualizationRefreshError(f"Chart field {field} contains non-numeric values.")
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        try:
            number = float(cleaned)
        except ValueError as exc:
            raise LiveVisualizationRefreshError(f"Chart field {field} contains non-numeric values.") from exc
        return int(number) if number.is_integer() else number
    raise LiveVisualizationRefreshError(f"Chart field {field} contains non-numeric values.")
