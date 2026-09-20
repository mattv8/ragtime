"""Canonical static User Space guidance for external instruction consumers."""

from __future__ import annotations

from ragtime.rag.prompts import (
    _USERSPACE_DATA_WIRING_BLOCK,
    _USERSPACE_MODE_PROMPT_TEMPLATE,
    _USERSPACE_OBJECT_STORAGE_SDK_GUIDANCE,
    _USERSPACE_RUNTIME_BRIDGE_BLOCK,
    BASE_USERSPACE_SYSTEM_PROMPT,
    USERSPACE_ENTRYPOINT_SETUP_PROMPT,
    USERSPACE_EXTERNAL_HARNESS_GUIDANCE_PROMPT,
    USERSPACE_SHARED_LIVE_DATA_GUARDRAILS,
    build_userspace_data_and_persistence_boundaries_fragment,
    build_userspace_turn_reminder,
)
from ragtime.userspace.instruction_facts import translate_internal_tool_references

_CONTENT_PROTECTION_REFUSAL_RELAY_GUIDANCE = """
## Content-protection refusals

If a platform tool returns a content-protection refusal, it includes a displayable `reason` and `next_step` alongside its code and request ID. Relay that reason and next step to the end user, stop the blocked operation, and do not automatically retry it. Parent agents must preserve this information when reporting a child-agent refusal. This is guidance for harness behavior; unrelated third-party UIs may not obey it.
"""


def content_protection_refusal_relay_guidance() -> str:
    """Return the shared external-harness refusal relay contract."""
    return _CONTENT_PROTECTION_REFUSAL_RELAY_GUIDANCE


def _render_workspace_template() -> str:
    """Render canonical placeholders once with neutral fact references.

    ``str.format`` preserves the template's escaped JSON/braced path examples,
    unlike ad-hoc replacement which leaked doubled braces to external clients.
    """
    return _USERSPACE_MODE_PROMPT_TEMPLATE.format(
        workspace_continuity="Workspace continuity, selected tools, identity, environment keys, runtime state, mounts, and storage are supplied as separately authorized live facts.",
        sqlite_persistence_block="Persistence rules apply only when workspace facts declare the corresponding persistence mode.",
        data_wiring_block="Live-data rules apply only when caller-authorized workspace facts list selected live-data tools.",
    )


def _topic(text: str, start: str, end: str | None = None) -> str:
    """Take a named canonical template region without maintaining a copy."""
    begin = text.index(start)
    finish = text.index(end, begin) if end else len(text)
    return text[begin:finish]


def build_external_guidance_documents() -> dict[str, str]:
    """Canonical generic platform guidance by stable topic, without live facts.

    Callers append their separately authorized workspace facts.  In particular,
    this function never renders an entrypoint state, identity, selected tool, or
    storage configuration merely to make a document look complete.
    """

    template = _render_workspace_template()
    documents = {
        "workspace": BASE_USERSPACE_SYSTEM_PROMPT
        + USERSPACE_EXTERNAL_HARNESS_GUIDANCE_PROMPT
        + content_protection_refusal_relay_guidance()
        + _topic(template, "## USER SPACE WORKSPACE CONTEXT", "#### Terminal tool")
        + _topic(template, "### File tool workflow", "### Theme + CSS rules")
        + "\n## Per-turn completion checklist\n"
        + build_userspace_turn_reminder(include_sqlite_persistence=False),
        "runtime": USERSPACE_ENTRYPOINT_SETUP_PROMPT
        + "\n\n"
        + _USERSPACE_RUNTIME_BRIDGE_BLOCK
        + _topic(template, "#### Terminal tool", "### File tool workflow"),
        "live-data": _USERSPACE_DATA_WIRING_BLOCK.format(userspace_shared_live_data_guardrails=USERSPACE_SHARED_LIVE_DATA_GUARDRAILS.strip()),
        "persistence": (
            "## Persistence applicability\n\n"
            "Apply Lane A only when caller-authorized workspace facts list selected live-data tools. "
            "Apply Lane B only when caller-authorized workspace facts declare SQLite persistence enabled. "
            "Shared targets are supplied separately.\n"
            + build_userspace_data_and_persistence_boundaries_fragment(include_sqlite_persistence=None, has_live_data_tools=None, shared_sqlite_databases=[])
        ),
        "identity": _topic(template, "#### Optional platform primitives", "#### Terminal tool"),
        "ui": _topic(template, "### Theme + CSS rules"),
        "storage": (
            "\n### Mount and object-storage facts\n\nOnly caller-authorized facts enumerate mount paths, bucket names, public/private roots, source availability, and writeability. Do not infer unlisted filesystem or object-storage access.\n"
            + _USERSPACE_OBJECT_STORAGE_SDK_GUIDANCE
        ),
    }
    return {topic: translate_internal_tool_references(text) for topic, text in documents.items()}
