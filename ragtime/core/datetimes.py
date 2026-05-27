from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def coerce_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_utc_iso_datetime(value: str) -> datetime:
    return coerce_utc_datetime(datetime.fromisoformat(value.replace("Z", "+00:00")))
