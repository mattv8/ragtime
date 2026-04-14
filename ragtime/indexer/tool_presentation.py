from __future__ import annotations

from typing import Any, Mapping


TERMINAL_PRESENTATION_KIND = "terminal"
USERSPACE_EXEC_RERUN_KIND = "userspace_exec"
TERMINAL_CONNECTION_TOOL_TYPES = frozenset({"ssh_shell", "odoo_shell"})


def _string_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def normalize_tool_presentation(
    tool_name: str | None,
    connection: Mapping[str, Any] | None = None,
    presentation: Mapping[str, Any] | None = None,
) -> dict[str, str] | None:
    """Normalize or derive UI presentation metadata for a tool call."""

    normalized: dict[str, str] = {}
    if isinstance(presentation, Mapping):
        for key, value in presentation.items():
            key_str = _string_value(key)
            value_str = _string_value(value)
            if not key_str or not value_str:
                continue
            if key_str in {"kind", "rerun_kind"}:
                normalized[key_str] = value_str.lower()
            else:
                normalized[key_str] = value_str
        if normalized:
            return normalized

    normalized_tool_name = _string_value(tool_name)
    tool_type = ""
    if isinstance(connection, Mapping):
        tool_type = _string_value(connection.get("tool_type")).lower()

    if normalized_tool_name == "run_terminal_command":
        return {
            "kind": TERMINAL_PRESENTATION_KIND,
            "rerun_kind": USERSPACE_EXEC_RERUN_KIND,
        }

    if tool_type in TERMINAL_CONNECTION_TOOL_TYPES:
        return {"kind": TERMINAL_PRESENTATION_KIND}

    return None