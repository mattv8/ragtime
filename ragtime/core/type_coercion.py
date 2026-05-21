import math
from typing import Any


def coerce_int_metadata(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return default
        return int(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        try:
            parsed = float(text)
        except ValueError:
            return default
        if not math.isfinite(parsed):
            return default
        return int(parsed)


def coerce_bool_metadata(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return value != 0
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def coerce_positive_int_metadata(value: Any) -> int | None:
    """Coerce metadata to a strictly positive int, or None when not parseable.

    Used for provider-supplied numeric limits where a missing, zero, negative,
    or malformed value should be ignored rather than silently zeroed.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        coerced = int(value)
        return coerced if coerced > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            parsed = int(text)
            return parsed if parsed > 0 else None
    return None
