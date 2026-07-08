from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

CHAT_EVENT_CHANNELS_BY_TYPE = {
    "reasoning": "analysis",
    "tool": "commentary",
    "content": "final",
    "error": "final",
}


def channel_for_event_type(event_type: str | None) -> str:
    """Return the display channel for a persisted chat event type."""
    return CHAT_EVENT_CHANNELS_BY_TYPE.get(str(event_type or ""), "final")


def with_event_channel(event: dict[str, Any]) -> dict[str, Any]:
    """Attach a first-class channel to a chat event while preserving compatibility."""
    normalized = dict(event)
    normalized.setdefault("channel", channel_for_event_type(normalized.get("type")))
    return normalized


def sanitize_reasoning_text(text: str) -> str:
    """Remove provider/rendering separators from reasoning summaries."""
    if not text or "<!--" not in text:
        return text

    cleaned_lines: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if re.match(r"^\s*(```|~~~)", line):
            cleaned_lines.append(line)
            in_fence = not in_fence
            continue
        if in_fence:
            cleaned_lines.append(line)
            continue
        cleaned_lines.append(re.sub(r"<!--\s*-->", "", line))

    return re.sub(r"\n{3,}", "\n\n", "".join(cleaned_lines))


def append_reasoning_event(
    events: list[dict[str, Any]],
    content: str,
    block_started_at: datetime | None,
) -> datetime:
    """Append reasoning content while preserving the active block start time."""
    now = datetime.now(timezone.utc)
    if block_started_at is None:
        block_started_at = now

    if events and events[-1].get("type") == "reasoning":
        events[-1].setdefault("channel", "analysis")
        events[-1]["content"] = sanitize_reasoning_text(f"{events[-1].get('content', '')}{content}")
    else:
        events.append({"type": "reasoning", "channel": "analysis", "content": sanitize_reasoning_text(content)})

    return block_started_at


def finalize_reasoning_block(
    events: list[dict[str, Any]],
    block_started_at: datetime | None,
) -> None:
    """Store final elapsed seconds on the latest reasoning event in a block."""
    if block_started_at is None:
        return

    latest_reasoning_event = next(
        (event for event in reversed(events) if isinstance(event, dict) and event.get("type") == "reasoning"),
        None,
    )
    if latest_reasoning_event is None:
        return

    elapsed_seconds = max(
        int((datetime.now(timezone.utc) - block_started_at).total_seconds()),
        0,
    )
    latest_reasoning_event["duration_seconds"] = elapsed_seconds
