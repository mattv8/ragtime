"""Safe, durable chat notices for content-protection denials."""

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from ragtime.content_protection.models import CONTENT_PROTECTION_PUBLIC_DEFAULTS, ContentProtectionError
from ragtime.core.datetimes import utc_now
from ragtime.core.event_bus import task_event_bus
from ragtime.core.logging import get_logger

DENIAL_MESSAGE_CONTENT = "Request blocked."
DENIAL_EVENT_TYPE = "content_protection_denial"
_DENIAL_EVENT_FIELDS = ("type", "channel", "code", "public_reason", "request_id", "timestamp")

logger = get_logger(__name__)


def denial_dedup_key(conversation_id: str, error: ContentProtectionError, task_id: str | None = None) -> str:
    """Return the stable per-attempt ID used to deduplicate a denial notice."""
    attempt_id = task_id or error.request_id
    return str(uuid5(NAMESPACE_URL, f"ragtime:content-protection-denial:{conversation_id}:{attempt_id}"))


def build_denial_message(conversation_id: str, error: ContentProtectionError, *, task_id: str | None = None) -> dict[str, Any]:
    """Build the only server-controlled message shape allowed for a denial."""
    if not isinstance(error, ContentProtectionError):
        raise TypeError("error must be a ContentProtectionError")
    if error.code not in CONTENT_PROTECTION_PUBLIC_DEFAULTS:
        raise ValueError("unsupported content protection error code")

    timestamp = utc_now().isoformat()
    public_reason = str(error.public_detail()["message"]).replace("\x00", "")
    event = {
        "type": DENIAL_EVENT_TYPE,
        "channel": "final",
        "code": error.code,
        "public_reason": public_reason,
        "request_id": str(error.request_id).replace("\x00", ""),
        "timestamp": timestamp,
    }
    return {
        "role": "assistant",
        "content": DENIAL_MESSAGE_CONTENT,
        "timestamp": timestamp,
        "message_id": denial_dedup_key(conversation_id, error, task_id),
        "events": [event],
    }


def is_denial_notice(message: Any) -> bool:
    """Return whether a persisted message has the server denial event type."""
    events = message.get("events") if isinstance(message, dict) else getattr(message, "events", None)
    return isinstance(events, list) and any(isinstance(event, dict) and event.get("type") == DENIAL_EVENT_TYPE for event in events)


def latest_denial_message_id(messages: list[Any]) -> str | None:
    """Return the newest typed denial message's opaque ID."""
    for message in reversed(messages):
        message_id = message.get("message_id") if isinstance(message, dict) else getattr(message, "message_id", None)
        if is_denial_notice(message) and isinstance(message_id, str) and message_id:
            return message_id
    return None


def denial_preview_events(message: dict[str, Any]) -> list[dict[str, str]] | None:
    """Return only the frozen typed event fields for a deferred denial preview."""
    if not is_denial_notice(message):
        return None
    events = message.get("events")
    if not isinstance(events, list):
        return None
    return [
        {field: str(event[field]).replace("\x00", "") for field in _DENIAL_EVENT_FIELDS if field in event}
        for event in events
        if isinstance(event, dict) and event.get("type") == DENIAL_EVENT_TYPE
    ] or None


async def persist_chat_denial(conversation_id: str, error: ContentProtectionError, *, task_id: str | None = None) -> str | None:
    """Persist a denial notice and announce its safe opaque identifiers."""
    from ragtime.indexer.repository import repository

    try:
        conversation = await repository.record_content_protection_denial(conversation_id, error, task_id=task_id)
    except Exception:
        logger.warning("Could not persist content-protection denial notice")
        return None
    if conversation is None:
        return None

    message_id = denial_dedup_key(conversation_id, error, task_id)
    error.denial_message_id = message_id
    try:
        await task_event_bus.publish(
            f"conversation:{conversation_id}",
            {"event": "denial_persisted", "message_id": message_id, "request_id": error.request_id},
        )
    except Exception:
        logger.warning("Could not publish content-protection denial notice")
    return message_id
