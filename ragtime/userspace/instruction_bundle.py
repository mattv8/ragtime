"""Deterministic User Space instructions for an authorized external harness."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from ragtime.core.entrypoint_status import EntrypointState, EntrypointStatus
from ragtime.rag.prompts import (
    build_current_user_turn_reminder_line,
    build_index_system_prompt,
    build_tool_system_prompt,
    build_userspace_diagnostics_turn_reminder_line,
    build_userspace_instruction_sections,
    build_userspace_turn_reminder_with_env_vars,
    build_workspace_continuity_context,
)
from ragtime.userspace.instruction_content import content_protection_refusal_relay_guidance
from ragtime.userspace.instruction_facts import build_env_var_turn_hint, normalize_facts, translate_internal_tool_references


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _items(value: object) -> list[dict[str, Any]]:
    return [_mapping(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _entrypoint(context: Mapping[str, Any]) -> tuple[EntrypointStatus, bool]:
    architecture = _mapping(context.get("architecture"))
    runtime = _mapping(context.get("runtime"))
    facts = {**architecture, **runtime, **_mapping(context.get("entrypoint"))}
    state_value = str(facts.get("entrypoint_state") or facts.get("state") or "missing")
    if state_value == "missing":
        state: EntrypointState = "missing"
    elif state_value == "valid":
        state = "valid"
    else:
        state = "invalid"
    return (
        EntrypointStatus(
            state=state,
            framework=facts.get("framework"),
            framework_known=bool(facts.get("framework_known", facts.get("framework"))),
            command=str(facts.get("command") or ""),
            cwd=str(facts.get("cwd") or "."),
            error=facts.get("error"),
        ),
        bool(facts.get("is_default_static", False)),
    )


def _revision(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def build_instruction_bundle(context: dict) -> dict:
    """Build one complete, ACL-input-only instruction bundle.

    ``context`` must already be filtered by the caller.  This function neither
    queries storage nor discovers global tools, indexes, identities, or secrets.
    """

    source = _mapping(context)
    workspace = _mapping(source.get("workspace"))
    architecture = _mapping(source.get("architecture"))
    user = _mapping(source.get("user"))
    if not user:
        user = {key: workspace[key] for key in ("username", "display_name", "caller_role") if key in workspace}
    selected_tools = _items(source.get("selected_tools") or source.get("authorized_tools"))
    authorized_indexes = _items(source.get("authorized_indexes"))
    authorized_resources = _mapping(source.get("authorized_resources"))
    mounts = _items(authorized_resources.get("mounts") or source.get("mounts"))
    buckets = _items(authorized_resources.get("object_storage_buckets") or source.get("object_storage_buckets"))
    shared_sqlite = _items(authorized_resources.get("shared_sqlite_databases") or source.get("shared_sqlite_databases"))
    credentials = source.get("authorized_build_credentials", source.get("build_credentials", []))
    credentials = credentials if isinstance(credentials, list) else []
    env_vars = _items(source.get("env_vars") or source.get("environment_variables"))
    status, is_default_static = _entrypoint(source)
    snapshot = _mapping(source.get("snapshot_summary"))
    continuity = build_workspace_continuity_context(
        file_count=int(architecture.get("file_count") or 0),
        key_files=[str(value) for value in architecture.get("key_files", []) if isinstance(value, str)],
        framework=status.framework,
        entrypoint_valid=status.state == "valid" and not is_default_static,
        last_snapshot_message=snapshot.get("last_message"),
        recent_failure_summaries=[str(value) for value in source.get("recent_failure_summaries", []) if value],
    )
    sqlite_mode = str(workspace.get("sqlite_persistence_mode") or source.get("sqlite_persistence_mode") or "exclude")
    capabilities = _mapping(source.get("capabilities"))
    tool_names = capabilities.get("tool_names")
    available_tool_names = {str(name) for name in tool_names} if isinstance(tool_names, list) else None
    sections = build_userspace_instruction_sections(
        include_sqlite_persistence=sqlite_mode == "include",
        has_live_data_tools=bool(selected_tools),
        workspace_continuity=continuity,
        entrypoint_status=status,
        is_default_static=is_default_static,
        username=user.get("username"),
        display_name=user.get("display_name"),
        available_tool_names=available_tool_names,
        shared_sqlite_databases=shared_sqlite,
        mounts_enabled=bool(mounts or authorized_resources.get("mounts_enabled", source.get("mounts_enabled", False))),
        mounts=mounts,
        object_storage_enabled=bool(buckets or authorized_resources.get("object_storage_enabled", source.get("object_storage_enabled", False))),
        object_storage_buckets=buckets,
    )
    sections = {key: translate_internal_tool_references(value) for key, value in sections.items()}
    sections["authorized_indexes"] = build_index_system_prompt(authorized_indexes, search_tool_name="index_search")
    sections["authorized_tools"] = build_tool_system_prompt(selected_tools, no_tools_selected=True)
    env_line = str(source.get("env_var_reminder_line") or "")
    if not env_line:
        env_line = build_env_var_turn_hint(env_vars)
    turn = build_current_user_turn_reminder_line(user.get("username"), user.get("display_name"))
    turn += build_userspace_turn_reminder_with_env_vars(
        include_sqlite_persistence=sqlite_mode == "include",
        env_var_reminder_line=env_line,
        runtime_status_reminder_line=str(source.get("runtime_status_reminder_line") or ""),
        diagnostics_reminder_line=str(source.get("diagnostics_reminder_line") or "")
        or build_userspace_diagnostics_turn_reminder_line(source.get("diagnostics"), available_tool_names=available_tool_names),
    )
    turn += "\n" + content_protection_refusal_relay_guidance()
    turn = translate_internal_tool_references(turn)
    unsupported = [
        {"item": "global catalog discovery", "reason": "only caller-authorized resources supplied in context are exported"},
        {"item": "internal subagent and model APIs", "reason": "translated to external harness responsibilities"},
        {"item": "unprovided runtime facts", "reason": "caller did not supply an ACL-filtered value"},
    ]
    bundle = {
        "workspace": workspace,
        "user": user,
        "system_instructions": sections,
        "turn_instructions": turn,
        "capabilities": {
            **capabilities,
            "authorized_tools": selected_tools,
            "authorized_indexes": authorized_indexes,
            "authorized_build_credentials": credentials,
        },
        "facts": {
            "entrypoint": {
                "state": status.state,
                "framework": status.framework,
                "framework_known": status.framework_known,
                "command": status.command,
                "cwd": status.cwd,
                "error": status.error,
                "is_default_static": is_default_static,
            },
            "continuity": {
                "file_count": int(architecture.get("file_count") or 0),
                "key_files": [str(value) for value in architecture.get("key_files", []) if isinstance(value, str)],
                "snapshot_summary": normalize_facts(snapshot),
            },
            "environment_variables": normalize_facts(env_vars),
            "mounts": normalize_facts(mounts),
            "object_storage_buckets": normalize_facts(buckets),
            "shared_sqlite_databases": normalize_facts(shared_sqlite),
            "runtime_status": normalize_facts(source.get("runtime_status")),
            "runtime_blocker": str(source.get("runtime_status_reminder_line") or ""),
            "diagnostics": normalize_facts(source.get("diagnostics") or []),
            "recent_failure_summaries": [str(value) for value in source.get("recent_failure_summaries", []) if value],
        },
        "unsupported": unsupported,
    }
    bundle["context_revision"] = _revision(bundle)
    return bundle
