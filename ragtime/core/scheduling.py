from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ragtime.core.datetimes import coerce_utc_datetime, utc_now

MINUTES_PER_DAY = 24 * 60


def clamp_start_minute(value: int | None) -> int | None:
    """Return a valid minutes-after-midnight value, or None when unset."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed < 0 or parsed >= MINUTES_PER_DAY:
        return None
    return parsed


def normalize_timezone_name(value: str | None) -> str | None:
    """Validate an IANA timezone name and return None for unset/invalid values."""
    normalized = str(value or "").strip()
    if not normalized:
        return None
    try:
        ZoneInfo(normalized)
    except ZoneInfoNotFoundError:
        return None
    return normalized


def next_anchored_run_after(
    *,
    interval_seconds: int | float,
    start_minute: int | None,
    timezone_name: str | None,
    after: datetime | None = None,
    now: datetime | None = None,
) -> datetime | None:
    """Compute the next UTC run time for a local wall-clock anchor.

    Returns None when the schedule is not anchored. This lets callers keep
    existing completion-based interval behavior when start_minute/timezone are
    unset or invalid.
    """
    anchor_minute = clamp_start_minute(start_minute)
    tz_name = normalize_timezone_name(timezone_name)
    if anchor_minute is None or tz_name is None:
        return None

    interval = max(1, int(interval_seconds))
    tz = ZoneInfo(tz_name)
    reference_utc = coerce_utc_datetime(after or now or utc_now())
    reference_local = reference_utc.astimezone(tz)

    anchor_hour = anchor_minute // 60
    anchor_min = anchor_minute % 60
    candidate_local = reference_local.replace(
        hour=anchor_hour,
        minute=anchor_min,
        second=0,
        microsecond=0,
    )

    # Walk forward from the anchor on the reference date until the next slot
    # after the reference is found.
    while candidate_local.astimezone(timezone.utc) <= reference_utc:
        candidate_local = candidate_local + timedelta(seconds=interval)

    return candidate_local.astimezone(timezone.utc)


def is_anchored_schedule_due(
    *,
    interval_seconds: int | float,
    start_minute: int | None,
    timezone_name: str | None,
    last_run_at: datetime | None,
    now: datetime | None = None,
) -> bool | None:
    """Return due state for an anchored schedule, or None when unanchored."""
    current = coerce_utc_datetime(now or utc_now())
    if last_run_at is None:
        # If the item has never run, wait for the first anchored slot at or
        # before now instead of running immediately on process startup.
        first_due = next_anchored_run_after(
            interval_seconds=interval_seconds,
            start_minute=start_minute,
            timezone_name=timezone_name,
            after=current - timedelta(seconds=max(1, int(interval_seconds))),
            now=current,
        )
    else:
        first_due = next_anchored_run_after(
            interval_seconds=interval_seconds,
            start_minute=start_minute,
            timezone_name=timezone_name,
            after=last_run_at,
            now=current,
        )
    if first_due is None:
        return None
    return current >= first_due
