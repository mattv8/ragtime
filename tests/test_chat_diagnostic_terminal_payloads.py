from ragtime.chat_runtime.payloads import build_chat_diagnostic_command_payload
from ragtime.chat_runtime.payloads import build_chat_diagnostic_rejection_payload
from ragtime.chat_runtime.payloads import normalize_chat_diagnostic_timeout_seconds
from ragtime.chat_runtime.payloads import resolve_chat_diagnostic_conversation_id
from ragtime.indexer.tool_presentation import CHAT_DIAGNOSTIC_RERUN_KIND
from ragtime.indexer.tool_presentation import normalize_tool_presentation


def test_chat_diagnostic_payload_marks_failed_commands_consistently() -> None:
    payload = build_chat_diagnostic_command_payload(
        command=" curl -I https://example.com ",
        timeout_seconds=12,
        response={"exit_code": 7, "stdout": "", "stderr": "failed"},
        duration_ms=42,
        reason="probe",
    )

    assert payload["tool"] == "run_chat_diagnostic_command"
    assert payload["status"] == "command_failed"
    assert payload["command"] == "curl -I https://example.com"
    assert payload["timeout_seconds"] == 12
    assert payload["stderr"] == "failed"
    assert payload["duration_ms"] == 42
    assert payload["reason"] == "probe"
    assert payload["error"] == "Diagnostic command finished with an error."


def test_chat_diagnostic_rejection_payload_is_terminal_shaped() -> None:
    payload = build_chat_diagnostic_rejection_payload(
        command="rm -rf /tmp/example",
        timeout_seconds=5,
        error="blocked",
    )

    assert payload == {
        "tool": "run_chat_diagnostic_command",
        "status": "rejected_not_persisted",
        "command": "rm -rf /tmp/example",
        "cwd": ".",
        "timeout_seconds": 5,
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "truncated": False,
        "error": "blocked",
    }


def test_chat_diagnostic_helpers_normalize_timeout_and_conversation_id() -> None:
    assert normalize_chat_diagnostic_timeout_seconds("0") == 1
    assert normalize_chat_diagnostic_timeout_seconds("999") == 30
    assert resolve_chat_diagnostic_conversation_id(" conv ", "user") == "conv"
    assert resolve_chat_diagnostic_conversation_id("", " user ") == "chat-anon-user"


def test_chat_diagnostic_tool_presentation_declares_rerun_kind() -> None:
    presentation = normalize_tool_presentation("run_chat_diagnostic_command")

    assert presentation == {
        "kind": "terminal",
        "rerun_kind": CHAT_DIAGNOSTIC_RERUN_KIND,
    }