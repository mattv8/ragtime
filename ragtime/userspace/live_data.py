"""Shared helpers for User Space live-data connection metadata."""


from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ragtime.userspace.models import UserSpaceLiveDataConnection

_COMPONENT_ID_KEYS = ("component_id", "source_tool_config_id", "tool_config_id")
_REQUEST_KEYS = ("request", "source_input")
_MAPPING_KEYS = ("result_mapping", "visualization_mapping")


def _first_non_empty_string(source: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(source.get(key) or "").strip()
        if value:
            return value
    return ""


def _first_present_value(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in source and source.get(key) is not None:
            return source.get(key)
    return None


def _is_empty_request(value: Any) -> bool:
    return (isinstance(value, str) and not value.strip()) or (
        isinstance(value, dict) and not value
    )


def _validate_canonical_connection(payload: dict[str, Any]) -> None:
    try:
        UserSpaceLiveDataConnection.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("data_connection must match the live data connection schema") from exc


def normalize_live_data_connection(
    data_connection: Any,
    *,
    require_component_id: bool = False,
    require_request: bool = False,
    require_result_mapping: bool = False,
    include_mappings: bool = True,
) -> dict[str, Any] | None:
    """Normalize chat visualization metadata to the User Space component schema."""
    if not isinstance(data_connection, dict):
        if require_component_id or require_request or require_result_mapping:
            raise ValueError("data_connection must be an object")
        return None

    component_id = _first_non_empty_string(data_connection, _COMPONENT_ID_KEYS)
    if require_component_id and not component_id:
        raise ValueError("data_connection.component_id is required")

    request_payload = _first_present_value(data_connection, _REQUEST_KEYS)
    if require_request and request_payload is None:
        raise ValueError("data_connection.request is required")
    if require_request and _is_empty_request(request_payload):
        raise ValueError("data_connection.request cannot be empty")
    if require_request and not isinstance(request_payload, (dict, str)):
        raise ValueError("data_connection.request must be an object or string")
    if request_payload is None:
        request_payload = {}
    elif not isinstance(request_payload, (dict, str)):
        request_payload = {"value": request_payload}

    component_kind = str(data_connection.get("component_kind") or "").strip()
    normalized: dict[str, Any] = {
        "component_kind": component_kind or "tool_config",
        "request": request_payload,
    }
    if component_id:
        normalized["component_id"] = component_id

    component_name = (
        str(data_connection.get("component_name") or "").strip()
        or str(data_connection.get("source_tool") or "").strip()
        or str(data_connection.get("tool_config_name") or "").strip()
    )
    if component_name:
        normalized["component_name"] = component_name

    component_type = (
        str(data_connection.get("component_type") or "").strip()
        or str(data_connection.get("source_tool_type") or "").strip()
        or str(data_connection.get("tool_type") or "").strip()
    )
    if component_type:
        normalized["component_type"] = component_type

    refresh_raw = data_connection.get("refresh_interval_seconds")
    if refresh_raw is not None:
        try:
            normalized["refresh_interval_seconds"] = max(1, int(refresh_raw))
        except (TypeError, ValueError):
            pass

    if require_component_id or require_request:
        _validate_canonical_connection(normalized)

    mapping_found = False
    if include_mappings:
        for mapping_key in _MAPPING_KEYS:
            mapping = data_connection.get(mapping_key)
            if isinstance(mapping, dict) and mapping:
                normalized[mapping_key] = mapping
                mapping_found = True

    if require_result_mapping and not mapping_found:
        raise ValueError("data_connection.result_mapping is required for live charts")

    return normalized


def validate_live_data_connection(
    data_connection: Any,
    *,
    require_result_mapping: bool = False,
) -> dict[str, Any]:
    """Validate and normalize required live data metadata for tool schemas."""
    normalized = normalize_live_data_connection(
        data_connection,
        require_component_id=True,
        require_request=True,
        require_result_mapping=require_result_mapping,
    )
    assert normalized is not None
    return normalized