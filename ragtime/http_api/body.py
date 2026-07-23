from __future__ import annotations

from collections.abc import Sequence
from typing import Any

BODY_MEDIA_TYPES = {
    "json": "application/json",
    "form": "application/x-www-form-urlencoded",
    "multipart": "multipart/form-data",
}


def _row_value(row: object, key: str) -> Any:
    return row.get(key) if isinstance(row, dict) else getattr(row, key, None)


def _media_type(value: str) -> str:
    return value.split(";", 1)[0].strip().lower()


def validate_configured_content_type(
    headers: Sequence[object] | None,
    body_format: object,
    *,
    has_body: bool,
    section: str,
) -> None:
    if not has_body:
        return
    configured = next(
        (str(_row_value(row, "value") or "") for row in headers or [] if str(_row_value(row, "name") or "").strip().lower() == "content-type"),
        "",
    )
    if not configured:
        return
    normalized_format = str(getattr(body_format, "value", body_format))
    expected = BODY_MEDIA_TYPES[normalized_format]
    if normalized_format == "multipart":
        raise ValueError(f"{section} Content-Type is generated automatically for multipart bodies")
    if _media_type(configured) != expected:
        raise ValueError(f"{section} Content-Type conflicts with the selected body format")
