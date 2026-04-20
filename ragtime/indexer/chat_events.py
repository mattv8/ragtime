
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

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
        events[-1]["content"] = f"{events[-1].get('content', '')}{content}"
    else:
        events.append({"type": "reasoning", "content": content})

    return block_started_at


def finalize_reasoning_block(
    events: list[dict[str, Any]],
    block_started_at: datetime | None,
) -> None:
    """Store final elapsed seconds on the latest reasoning event in a block."""
    if block_started_at is None:
        return

    latest_reasoning_event = next(
        (
            event
            for event in reversed(events)
            if isinstance(event, dict) and event.get("type") == "reasoning"
        ),
        None,
    )
    if latest_reasoning_event is None:
        return

    elapsed_seconds = max(
        int((datetime.now(timezone.utc) - block_started_at).total_seconds()),
        0,
    )
    latest_reasoning_event["duration_seconds"] = elapsed_seconds
