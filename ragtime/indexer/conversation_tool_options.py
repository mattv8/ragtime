"""Pure helpers for per-conversation tool option normalization and write policy."""

from __future__ import annotations

from typing import Any, Optional

# Keys considered part of the conversation tool-options contract.
_KNOWN_OPTION_KEYS = {"write_access_enabled", "read_only_enabled"}


def normalize_conversation_tool_options(raw: Optional[dict[str, Any]]) -> dict[str, dict[str, bool]]:
    """Normalize raw per-conversation tool options from client input.

    Keeps only the recognized boolean option keys and drops rows that are not
    mappings or that contain no truthy option values. Unknown keys are dropped.
    """
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, dict[str, bool]] = {}
    for tool_config_id, options in raw.items():
        if not isinstance(tool_config_id, str) or not tool_config_id.strip():
            continue
        if not isinstance(options, dict):
            continue

        cleaned: dict[str, bool] = {}
        for key in _KNOWN_OPTION_KEYS:
            value = options.get(key)
            if isinstance(value, bool) and value:
                cleaned[key] = value

        if cleaned:
            normalized[tool_config_id] = cleaned

    return normalized


def resolve_effective_allow_write(global_allow_write: bool, options: Optional[dict[str, bool]]) -> bool:
    """Return the request-scoped write policy for a configured tool.

    Priority:
      1. If ``read_only_enabled`` is True, force False.
      2. Else if ``write_access_enabled`` is True, force True.
      3. Otherwise fall back to the tool's global ``allow_write`` value.
    """
    if options:
        if options.get("read_only_enabled"):
            return False
        if options.get("write_access_enabled"):
            return True
    return bool(global_allow_write)


def load_conversation_tool_options(raw_options: Optional[dict[str, Any]]) -> dict[str, bool]:
    """Load and normalize one persisted conversation-tool option row."""
    if not isinstance(raw_options, dict):
        return {}

    cleaned: dict[str, bool] = {}
    for key in _KNOWN_OPTION_KEYS:
        value = raw_options.get(key)
        if isinstance(value, bool) and value:
            cleaned[key] = value
    return cleaned
