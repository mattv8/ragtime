"""Hash-only administrator MFA recovery-pass primitives.

The module intentionally keeps recovery continuations separate from ordinary
MFA tokens.  A continuation is proof to replace a factor, never an app session.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import uuid4

from jose import JWTError, jwt  # type: ignore[import-untyped]

from ragtime.config.settings import settings

RECOVERY_PASS_TTL = timedelta(minutes=30)
RECOVERY_CONTINUATION_TTL = timedelta(minutes=10)
MAX_PASS_ATTEMPTS = 5

RecoveryStatus = Literal["issued", "redeemed", "completed", "revoked", "expired"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def generate_pass() -> str:
    # 32 random bytes is 256 bits, comfortably above the required 128 bits.
    return secrets.token_urlsafe(32)


def hash_pass(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verify_pass(value: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_pass(value), stored_hash)


def _token(*, purpose: str, user_id: str, grant_id: str, generation: int, expires: datetime, **extra: Any) -> str:
    return jwt.encode(
        {"sub": user_id, "grant_id": grant_id, "generation": generation, "purpose": purpose, "exp": expires, **extra},
        settings.encryption_key,
        algorithm=settings.jwt_algorithm,
    )


def create_recovery_token(*, user_id: str, grant_id: str, generation: int, expires: datetime) -> str:
    return _token(purpose="mfa:recovery_continuation", user_id=user_id, grant_id=grant_id, generation=generation, expires=expires)


def create_totp_token(*, user_id: str, grant_id: str, generation: int, secret: str, expires: datetime) -> str:
    return _token(purpose="mfa:recovery_totp_enrollment", user_id=user_id, grant_id=grant_id, generation=generation, expires=expires, secret=secret)


def create_webauthn_token(*, user_id: str, grant_id: str, generation: int, challenge: str, jti: str, expires: datetime) -> str:
    """Create a recovery-only WebAuthn registration challenge token."""
    return _token(
        purpose="mfa:recovery_webauthn_enrollment",
        user_id=user_id,
        grant_id=grant_id,
        generation=generation,
        challenge=challenge,
        jti=jti,
        expires=expires,
        security_generation=generation,
    )


def decode_token(token: str, *, purpose: str) -> dict[str, Any] | None:
    try:
        claims = jwt.decode(token, settings.encryption_key, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None
    if not isinstance(claims, dict) or claims.get("purpose") != purpose:
        return None
    return claims


def recovery_status(row: dict[str, Any], now: datetime | None = None) -> RecoveryStatus:
    now = now or utcnow()
    if row.get("revoked_at"):
        return "revoked"
    if row.get("completed_at"):
        return "completed"
    expires = row.get("expires_at")
    if isinstance(expires, str):
        expires = datetime.fromisoformat(expires.replace("Z", "+00"))
    if expires and (expires if expires.tzinfo else expires.replace(tzinfo=timezone.utc)) <= now:
        return "expired"
    if row.get("redeemed_at"):
        return "redeemed"
    return "issued"
