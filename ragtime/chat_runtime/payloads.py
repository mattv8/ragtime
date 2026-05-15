"""Shared payload helpers for chat diagnostics tool results."""


from __future__ import annotations

from typing import Any, Mapping

from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_COMMAND_TIMEOUT_MAX_SECONDS
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTIC_COMMAND_TOOL_ID

def resolve_chat_diagnostic_conversation_id(
    conversation_id: str | None,
    user_id: str | None,
) -> str:
    effective_conversation_id = (conversation_id or "").strip()
    if effective_conversation_id:
        return effective_conversation_id
    effective_user_id = (user_id or "anonymous").strip() or "anonymous"
    return f"chat-anon-{effective_user_id}"


def normalize_chat_diagnostic_timeout_seconds(value: Any, default: int = 30) -> int:
    max_seconds = max(1, min(120, int(CHAT_DIAGNOSTICS_COMMAND_TIMEOUT_MAX_SECONDS)))
    try:
        timeout_seconds = int(value)
    except (TypeError, ValueError):
        timeout_seconds = default
    return max(1, min(max_seconds, timeout_seconds))


def build_chat_diagnostic_rejection_payload(
    *,
    command: str,
    timeout_seconds: int,
    error: str,
) -> dict[str, Any]:
    return {
        "tool": CHAT_DIAGNOSTIC_COMMAND_TOOL_ID,
        "status": "rejected_not_persisted",
        "command": (command or "").strip(),
        "cwd": ".",
        "timeout_seconds": int(timeout_seconds),
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "truncated": False,
        "error": error,
    }


def build_chat_diagnostic_command_payload(
    *,
    command: str,
    timeout_seconds: int,
    response: Mapping[str, Any],
    duration_ms: int | None = None,
    reason: str = "",
) -> dict[str, Any]:
    exit_code = int(response.get("exit_code", 0) or 0)
    timed_out = bool(response.get("timed_out", False))
    command_failed = timed_out or exit_code != 0

    payload: dict[str, Any] = {
        "tool": CHAT_DIAGNOSTIC_COMMAND_TOOL_ID,
        "status": (
            "command_timed_out"
            if timed_out
            else "command_failed" if command_failed else "completed"
        ),
        "command": (command or "").strip(),
        "cwd": ".",
        "timeout_seconds": int(timeout_seconds),
        "exit_code": exit_code,
        "stdout": response.get("stdout") or "",
        "stderr": response.get("stderr") or "",
        "timed_out": timed_out,
        "truncated": bool(response.get("truncated", False)),
    }
    if duration_ms is not None:
        payload["duration_ms"] = int(duration_ms)
    if reason:
        payload["reason"] = reason
    if command_failed:
        payload["error"] = (
            "Diagnostic command timed out."
            if timed_out
            else "Diagnostic command finished with an error."
        )
    return payload