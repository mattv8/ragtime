from __future__ import annotations

import hmac
from collections.abc import Mapping


def bearer_token(headers: Mapping[str, str]) -> str | None:
    value = headers.get("authorization") or headers.get("Authorization")
    if not value:
        return None
    scheme, _, token = value.strip().partition(" ")
    return token if scheme.lower() == "bearer" and token else None


def constant_time_token_matches(expected: str, provided: str | None) -> bool:
    if provided is None:
        return False
    try:
        return hmac.compare_digest(expected, provided)
    except TypeError:
        return False
