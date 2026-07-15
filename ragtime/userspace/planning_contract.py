"""Versioned, curated planning contract exposed to external agents.

This is the ONLY prompt-adjacent content served to external agents. It must
never include the full internal system prompt. Rules that must stay aligned
with the internal builder prompt are derived from constants in
``ragtime.rag.prompts`` and covered by drift tests.
"""

from __future__ import annotations

from typing import Any

from ragtime.rag.prompts import _WORKSPACE_CONTINUITY_EXISTING_RULES

PLANNING_CONTRACT_VERSION = "1"

_STATIC_BUILDER_RULES: list[str] = [
    "The runtime entrypoint lives in .ragtime/runtime-entrypoint.json; launch commands must use $PORT and bind 0.0.0.0.",
    "When the workspace has selected data tools, dashboards must be wired to live tool data; hardcoded/mock datasets are rejected by validation.",
    "Browser-side live data (context.components[...].execute) is always read-only; writes only happen from workspace server code via the runtime bridge, subject to workspace write policy.",
    "SQLite local persistence (.ragtime/db/app.sqlite3 with numbered migrations) supplements live data for local app state; it never replaces live dashboard datasets.",
    "Match the app theme using runtime CSS variables/tokens; avoid hardcoded palettes and custom font stacks.",
    "The preview iframe is cross-origin; workspace code must never read context from window.parent or window.top.",
]


def continuity_rules() -> list[str]:
    """Continuity rules shared verbatim with the internal builder prompt."""
    return [rule.lstrip("- ").strip() for rule in _WORKSPACE_CONTINUITY_EXISTING_RULES]


def build_builder_contract(*, sqlite_persistence_mode: str, has_live_data_tools: bool) -> dict[str, Any]:
    """Curated rule set describing how the internal builder will behave."""
    return {
        "contract_version": PLANNING_CONTRACT_VERSION,
        "sqlite_persistence_mode": sqlite_persistence_mode,
        "has_live_data_tools": has_live_data_tools,
        "rules": continuity_rules() + list(_STATIC_BUILDER_RULES),
    }


def build_recommended_workflow() -> list[str]:
    """Ordered guidance surfaced to external agents in context + manifest."""
    return [
        "Fetch /context and read the architecture, selected tools, and rules before proposing work.",
        "List files and read only the ones relevant to the change; prefer extending existing code over replacing it.",
        "Draft a build brief with the user: objective, concrete requirements, acceptance criteria, files to preserve, and required data component IDs.",
        "POST the brief to /tasks with a fresh idempotency_key; the internal builder starts immediately in a new workspace conversation.",
        "Poll /tasks/{task_id} every 10-30 seconds; long builds are normal. If the final result ends with a question, answer it via /tasks/{task_id}/reply.",
    ]
