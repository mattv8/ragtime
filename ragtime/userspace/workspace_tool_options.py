"""Pure helpers for per-workspace tool option normalization and write policy."""

from __future__ import annotations

from typing import Any, Optional

_KNOWN_OPTION_KEYS = {"write_access_enabled"}


def _coerce_options_dict(options: Any) -> dict[str, Any] | None:
    if isinstance(options, dict):
        return options
    model_dump = getattr(options, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return None


def normalize_workspace_tool_options(
    raw: Optional[dict[str, Any]],
    *,
    selected_tool_ids: set[str],
    write_capable_tool_ids: set[str],
) -> dict[str, dict[str, bool]]:
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, dict[str, bool]] = {}
    for tool_config_id, options in raw.items():
        if not isinstance(tool_config_id, str) or not tool_config_id.strip():
            continue
        if tool_config_id not in selected_tool_ids or tool_config_id not in write_capable_tool_ids:
            continue

        options_dict = _coerce_options_dict(options)
        if not isinstance(options_dict, dict):
            continue

        cleaned: dict[str, bool] = {}
        for key in _KNOWN_OPTION_KEYS:
            value = options_dict.get(key)
            if isinstance(value, bool) and value:
                cleaned[key] = value

        if cleaned:
            normalized[tool_config_id] = cleaned

    return normalized


def load_workspace_tool_options(raw_options: Optional[dict[str, Any]]) -> dict[str, bool]:
    options_dict = _coerce_options_dict(raw_options)
    if not isinstance(options_dict, dict):
        return {}

    cleaned: dict[str, bool] = {}
    for key in _KNOWN_OPTION_KEYS:
        value = options_dict.get(key)
        if isinstance(value, bool) and value:
            cleaned[key] = value
    return cleaned


def resolve_workspace_tool_write_access(
    global_allow_write: bool,
    options: Optional[dict[str, bool]],
) -> bool:
    _ = global_allow_write
    return bool(options and options.get("write_access_enabled") is True)
