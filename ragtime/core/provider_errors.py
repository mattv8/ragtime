"""Provider-safe error classification helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator

_PAYMENT_ERROR_TYPES = {
    "payment_required",
    "insufficient_credit",
    "insufficient_credits",
    "credit_exhausted",
    "credits_exhausted",
    "balance_exhausted",
}


def _walk_payload(value: object) -> Iterator[object]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_payload(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_payload(nested)


def _parse_payload(value: object) -> object | None:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Providers may deliver a final error as an SSE data event.
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            return json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
    return None


def _payload_is_payment_required(payload: object) -> bool:
    for item in _walk_payload(payload):
        if not isinstance(item, dict):
            continue
        for field in ("error_type", "type", "code"):
            value = item.get(field)
            if isinstance(value, str) and value.strip().lower().replace("-", "_") in _PAYMENT_ERROR_TYPES:
                return True
    return False


def _error_status(error: object) -> int | None:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    # Narrow to int-convertible types before attempting conversion
    if isinstance(status_code, (str, bytes, bytearray, float)):
        try:
            return int(status_code)
        except (ValueError, TypeError):
            pass
    return None


def _error_payload(error: object) -> object | None:
    if isinstance(error, (dict, list, str, bytes)):
        return _parse_payload(error)
    response = getattr(error, "response", None)
    if response is not None:
        try:
            return response.json()
        except (ValueError, json.JSONDecodeError):
            return _parse_payload(getattr(response, "text", None) or getattr(response, "content", None))
    return _parse_payload(getattr(error, "body", None) or getattr(error, "content", None))


def classify_provider_error(error: object, provider: str | None = None) -> str | None:
    """Classify known provider failures without retaining provider error text.

    ``provider`` is reserved for future provider-specific classifications.  A
    402 is unambiguous for every HTTP provider; otherwise only structured,
    explicit credit error types qualify as payment failures.
    """
    del provider
    if _error_status(error) == 402:
        return "payment_required"
    if _payload_is_payment_required(_error_payload(error)):
        return "payment_required"
    return None


def provider_error_message(code: str) -> str:
    """Return a safe user-facing message for a classified provider error."""
    if code == "payment_required":
        return "The provider requires available payment credit before this request can continue."
    return "The provider request could not be completed."
