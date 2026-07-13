from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

VALID_TOOL_SELECTION_MODES = frozenset({"default_all", "custom"})


def require_valid_tool_selection_mode(value: Any) -> str:
    mode = str(value or "").strip()
    if mode not in VALID_TOOL_SELECTION_MODES:
        raise ValueError("tool_selection_mode must be 'default_all' or 'custom'")
    return mode


def intersect_tool_ids(primary: Sequence[str], secondary: Sequence[str]) -> list[str]:
    secondary_ids = {str(tool_id or "").strip() for tool_id in secondary if str(tool_id or "").strip()}
    return [tool_id for tool_id in primary if tool_id in secondary_ids]


async def resolve_effective_tool_ids(
    *,
    tool_selection_mode: Any,
    selected_tool_ids: Sequence[str] | None,
    selected_tool_group_ids: Sequence[str] | None,
    list_healthy_enabled_tool_ids: Callable[[], Awaitable[list[str]]],
    get_tool_ids_for_groups: Callable[[list[str]], Awaitable[list[str]]],
) -> list[str]:
    mode = require_valid_tool_selection_mode(tool_selection_mode)
    healthy_tool_ids = list(await list_healthy_enabled_tool_ids())
    if mode == "default_all":
        return healthy_tool_ids

    resolved_tool_ids: list[str] = []
    seen: set[str] = set()

    def add_tool_id(tool_id: Any) -> None:
        normalized = str(tool_id or "").strip()
        if normalized and normalized not in seen:
            resolved_tool_ids.append(normalized)
            seen.add(normalized)

    for tool_id in selected_tool_ids or []:
        add_tool_id(tool_id)

    group_ids = [str(group_id or "").strip() for group_id in (selected_tool_group_ids or []) if str(group_id or "").strip()]
    if group_ids:
        for tool_id in await get_tool_ids_for_groups(group_ids):
            add_tool_id(tool_id)

    healthy_set = set(healthy_tool_ids)
    return [tool_id for tool_id in resolved_tool_ids if tool_id in healthy_set]
