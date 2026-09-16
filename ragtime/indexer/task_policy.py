"""Durable execution-policy and outcome helpers for chat tasks."""

from __future__ import annotations

from typing import Any, Literal

TaskType = Literal["build", "general"]


def make_execution_policy(task_type: TaskType, *, source: str) -> dict[str, Any]:
    return {
        "version": 1,
        "task_type": task_type,
        "require_action": task_type == "build",
        "source": source,
    }


def policy_requires_action(policy: object) -> bool:
    return isinstance(policy, dict) and policy.get("version") == 1 and bool(policy.get("require_action"))


def activity_summary(tool_calls: list[dict[str, Any]]) -> dict[str, int]:
    """Count real completed tool executions, excluding synthetic recovery rows."""
    attempted = succeeded = failed = 0
    for call in tool_calls:
        if call.get("synthetic") or call.get("recovery"):
            continue
        attempted += 1
        if call.get("success") is False or call.get("failed") is True:
            failed += 1
        else:
            succeeded += 1
    return {"attempted": attempted, "succeeded": succeeded, "failed": failed}


def required_action_termination(policy: object, activity: dict[str, int], hit_max_iterations: bool) -> str | None:
    if not policy_requires_action(policy):
        return None
    if hit_max_iterations:
        return "max_iterations"
    if activity["attempted"] == 0:
        return "no_actions"
    if activity["succeeded"] == 0:
        return "all_tools_failed"
    return None
