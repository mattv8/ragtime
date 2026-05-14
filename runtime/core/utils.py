from __future__ import annotations

import os
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_positive_int_env(name: str, default_value: int) -> int:
    raw_value = os.getenv(name, str(default_value)).strip()
    try:
        parsed = int(raw_value)
        return parsed if parsed > 0 else default_value
    except Exception:
        return default_value