"""Pure external-instruction fact and tool-vocabulary helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


def build_env_var_turn_hint(env_vars: Sequence[object] | None) -> str:
    """Render caller-safe environment key/status facts, never values."""
    items = list(env_vars or [])
    parts: list[str] = []
    for item in items[:10]:
        key = item.get("key") if isinstance(item, Mapping) else getattr(item, "key", "")
        has_value = item.get("has_value") if isinstance(item, Mapping) else getattr(item, "has_value", False)
        if str(key or "").strip():
            parts.append(f"{str(key).strip()}({'set' if has_value else 'missing'})")
    if not parts:
        return ""
    suffix = "" if len(items) <= 10 else f", +{len(items) - 10} more"
    return "- Workspace env vars (keys only): " + ", ".join(parts) + suffix + ".\n"


_TOOL_TRANSLATIONS = {
    "assay_userspace_code": "`files_list`, `file_read`, and `code_search`",
    "search_userspace_code": "`code_search` (or `files_list` and `file_read`)",
    "list_userspace_files": "`files_list`",
    "patch_userspace_file": "`file_patch`",
    "upsert_userspace_file": "`file_write`",
    "validate_userspace_code": "`validate`",
    "create_userspace_snapshot": "`snapshot_create`",
    "run_terminal_command": "`exec_start` followed by `exec_get`",
    "get_app_runtime_status": "`runtime_status`",
    "restart_app_runtime": "`runtime_stop` then `runtime_start`",
    "userspace_diagnostics": "the caller-safe diagnostics facts in `context`",
    "upsert_userspace_env_var": "ask a workspace owner to create the environment-variable placeholder",
    "discover_userspace_primitives": "consult the supplied documented primitive contract",
    "configure_userspace_identity_entitlements": "ask a workspace owner or administrator to configure identity entitlements",
    "search_knowledge": "`index_search` for an explicitly granted index",
}
_TOOL_REFERENCE = re.compile(r"(?<![A-Za-z0-9_])`?(" + "|".join(map(re.escape, _TOOL_TRANSLATIONS)) + r")`?(?![A-Za-z0-9_])")


def translate_internal_tool_references(text: str) -> str:
    """Translate internal chat tool names to actual external operations."""
    return _TOOL_REFERENCE.sub(lambda match: _TOOL_TRANSLATIONS[match.group(1)], text)


def normalize_facts(value: Any) -> Any:
    """Convert caller-safe Pydantic/ORM values to JSON-shaped facts."""
    if hasattr(value, "model_dump"):
        return normalize_facts(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): normalize_facts(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_facts(item) for item in value]
    return value
