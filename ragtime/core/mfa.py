from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from urllib.parse import quote

from jose import JWTError, jwt  # type: ignore[import-untyped]
from prisma import Json

from ragtime.config.settings import settings
from ragtime.core.database import get_db
from ragtime.core.encryption import decrypt_secret, encrypt_secret
from ragtime.core.logging import get_logger
from ragtime.core.webauthn_mfa import user_has_enabled_webauthn

logger = get_logger(__name__)

TOTP_METHOD = "totp"
WEBAUTHN_METHOD = "webauthn"
VALID_MFA_METHODS = (TOTP_METHOD, WEBAUTHN_METHOD)

TOTP_PERIOD_SECONDS = 30
TOTP_DIGITS = 6
TOTP_SECRET_BYTES = 20
RECOVERY_CODE_COUNT = 8
PENDING_MFA_EXPIRY_SECONDS = 600
MFA_TRUST_COOKIE_NAME = "ragtime_mfa_trust"

TotpPolicyValue = Literal["optional", "required_all", "required_admins_groups"]
PendingMfaPurpose = Literal["challenge", "enroll"]


@dataclass(frozen=True)
class TotpVerificationResult:
    valid: bool
    time_step: int | None = None


@dataclass(frozen=True)
class PendingMfaClaims:
    user_id: str
    username: str
    role: str
    purpose: PendingMfaPurpose
    exp: datetime


@dataclass(frozen=True)
class TotpEnrollmentClaims:
    user_id: str
    username: str
    role: str
    secret: str
    exp: datetime
    rotation: bool = False


def _b64decode_padded(value: str) -> bytes:
    return base64.b32decode(value.upper() + "=" * (-len(value) % 8), casefold=True)


def base32_secret_from_bytes(secret: bytes) -> str:
    return base64.b32encode(secret).decode("ascii").rstrip("=")


def generate_totp_secret() -> str:
    return base32_secret_from_bytes(secrets.token_bytes(TOTP_SECRET_BYTES))


def generate_totp_code(
    secret: str,
    *,
    for_time: int | float | None = None,
    digits: int = TOTP_DIGITS,
    period_seconds: int = TOTP_PERIOD_SECONDS,
) -> str:
    timestamp = int(time.time() if for_time is None else for_time)
    counter = timestamp // period_seconds
    key = _b64decode_padded(secret)
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code_int = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(code_int % (10**digits)).zfill(digits)


def verify_totp_code(
    secret: str,
    code: str,
    *,
    for_time: int | float | None = None,
    last_used_step: int | None = None,
    window: int = 1,
) -> TotpVerificationResult:
    normalized = "".join(ch for ch in str(code or "") if ch.isdigit())
    if len(normalized) != TOTP_DIGITS:
        return TotpVerificationResult(valid=False)

    timestamp = int(time.time() if for_time is None else for_time)
    current_step = timestamp // TOTP_PERIOD_SECONDS
    for offset in range(-window, window + 1):
        step = current_step + offset
        if last_used_step is not None and step <= int(last_used_step):
            continue
        expected = generate_totp_code(secret, for_time=step * TOTP_PERIOD_SECONDS)
        if hmac.compare_digest(expected, normalized):
            return TotpVerificationResult(valid=True, time_step=step)
    return TotpVerificationResult(valid=False)


def generate_recovery_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    raw = "".join(secrets.choice(alphabet) for _ in range(16))
    return f"{raw[:4]}-{raw[4:8]}-{raw[8:12]}-{raw[12:]}"


def _normalize_recovery_code(code: str) -> str:
    return "".join(ch for ch in str(code or "").upper() if ch.isalnum())


def hash_recovery_code(code: str) -> str:
    salt = secrets.token_bytes(16)
    normalized = _normalize_recovery_code(code)
    digest = hashlib.pbkdf2_hmac("sha256", normalized.encode("utf-8"), salt, 310_000)
    salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii").rstrip("=")
    digest_b64 = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"pbkdf2_sha256$310000${salt_b64}${digest_b64}"


def verify_recovery_code(code: str, stored_hash: str | None) -> bool:
    if not code or not stored_hash:
        return False
    try:
        algorithm, iterations_raw, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_b64 + "=" * (-len(salt_b64) % 4))
        expected = base64.urlsafe_b64decode(digest_b64 + "=" * (-len(digest_b64) % 4))
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            _normalize_recovery_code(code).encode("utf-8"),
            salt,
            int(iterations_raw),
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def create_pending_mfa_token(
    *,
    user_id: str,
    username: str,
    role: str,
    purpose: PendingMfaPurpose,
) -> str:
    expire = datetime.now(timezone.utc) + timedelta(seconds=PENDING_MFA_EXPIRY_SECONDS)
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "purpose": f"mfa:{purpose}",
        "exp": expire,
    }
    return jwt.encode(payload, settings.encryption_key, algorithm=settings.jwt_algorithm)


def decode_pending_mfa_token(
    token: str,
    *,
    expected_purpose: PendingMfaPurpose,
) -> PendingMfaClaims | None:
    try:
        payload = jwt.decode(token, settings.encryption_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        logger.debug("Pending MFA token decode failed: %s", exc)
        return None

    if payload.get("purpose") != f"mfa:{expected_purpose}":
        return None
    try:
        return PendingMfaClaims(
            user_id=str(payload["sub"]),
            username=str(payload["username"]),
            role=str(payload["role"]),
            purpose=expected_purpose,
            exp=datetime.fromtimestamp(int(payload["exp"]), tz=timezone.utc),
        )
    except (KeyError, TypeError, ValueError):
        return None


def create_totp_enrollment_token(
    *,
    user_id: str,
    username: str,
    role: str,
    secret: str,
    rotation: bool = False,
) -> str:
    expire = datetime.now(timezone.utc) + timedelta(seconds=PENDING_MFA_EXPIRY_SECONDS)
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "secret": secret,
        "purpose": "mfa:totp_enrollment",
        "rotation": bool(rotation),
        "exp": expire,
    }
    return jwt.encode(payload, settings.encryption_key, algorithm=settings.jwt_algorithm)


def decode_totp_enrollment_token(token: str) -> TotpEnrollmentClaims | None:
    try:
        payload = jwt.decode(token, settings.encryption_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        logger.debug("TOTP enrollment token decode failed: %s", exc)
        return None

    if payload.get("purpose") != "mfa:totp_enrollment":
        return None
    try:
        return TotpEnrollmentClaims(
            user_id=str(payload["sub"]),
            username=str(payload["username"]),
            role=str(payload["role"]),
            secret=str(payload["secret"]),
            exp=datetime.fromtimestamp(int(payload["exp"]), tz=timezone.utc),
            rotation=bool(payload.get("rotation", False)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def build_otpauth_uri(*, issuer: str, username: str, secret: str) -> str:
    label = quote(f"{issuer}:{username}")
    return f"otpauth://totp/{label}?secret={quote(secret)}&issuer={quote(issuer)}&algorithm=SHA1&digits={TOTP_DIGITS}&period={TOTP_PERIOD_SECONDS}"


def _provider_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


async def user_requires_totp(user: Any, *, config: Any | None = None) -> bool:
    if config is None:
        db = await get_db()
        config = await db.authproviderconfig.find_unique(where={"id": "default"})

    policy = _provider_value(getattr(config, "totp_policy", getattr(config, "totpPolicy", "optional")))
    if policy == "required_all":
        return True
    if policy != "required_admins_groups":
        return False
    if _provider_value(getattr(user, "role", "")) == "admin":
        return True

    group_ids = list(getattr(config, "totp_required_group_ids", getattr(config, "totpRequiredGroupIds", [])) or [])
    if not group_ids:
        return False

    db = await get_db()
    membership = await db.authgroupmembership.find_first(
        where={
            "userId": user.id,
            "groupId": {"in": group_ids},
        }
    )
    return membership is not None


async def get_enabled_totp_factor(user_id: str) -> Any | None:
    db = await get_db()
    return await db.usermfafactor.find_unique(where={"userId_factorType": {"userId": user_id, "factorType": "totp"}})


async def user_has_enabled_totp(user_id: str) -> bool:
    factor = await get_enabled_totp_factor(user_id)
    return bool(factor and getattr(factor, "enabled", False) and getattr(factor, "secretEncrypted", None))


async def get_allowed_mfa_methods() -> list[str]:
    """From AuthProviderConfig.mfaAllowedMethods; defaults to [\"totp\"] when unset/empty/invalid."""
    db = await get_db()
    config = await db.authproviderconfig.find_unique(where={"id": "default"})
    raw = getattr(config, "mfaAllowedMethods", getattr(config, "mfa_allowed_methods", None)) if config else None
    methods = [str(v).lower().strip() for v in (raw or []) if str(v).lower().strip() in VALID_MFA_METHODS]
    return methods if methods else [TOTP_METHOD]


async def user_allowed_enrolled_methods(user_id: str) -> list[str]:
    """Intersection of allowed methods and the user's enrolled factors.

    Order is ["webauthn", "totp"] when both are present.
    """
    allowed = await get_allowed_mfa_methods()
    enrolled: list[str] = []
    if WEBAUTHN_METHOD in allowed and await user_has_enabled_webauthn(user_id):
        enrolled.append(WEBAUTHN_METHOD)
    if TOTP_METHOD in allowed and await user_has_enabled_totp(user_id):
        enrolled.append(TOTP_METHOD)
    return enrolled


async def resolve_preferred_mfa_method(
    user,
    enrolled_allowed: list[str],
    *,
    config=None,
) -> str | None:
    """Resolve the MFA method to present first in a challenge.

    Priority: user preference -> provider default -> None.
    Only returns a method that is both enrolled and currently allowed.
    """
    pref = str(getattr(user, "mfaPreferredMethod", None) or "").lower().strip()
    if pref in enrolled_allowed:
        return pref

    if config is None:
        db = await get_db()
        config = await db.authproviderconfig.find_unique(where={"id": "default"})

    default = str(getattr(config, "mfaDefaultMethod", getattr(config, "mfa_default_method", None)) or "").lower().strip()
    if default in enrolled_allowed:
        return default

    return None


async def mfa_needed_for_user(user: Any, *, config: Any | None = None) -> bool:
    return await user_requires_totp(user, config=config) or await user_has_enabled_totp(user.id) or await user_has_enabled_webauthn(user.id)


async def regenerate_recovery_codes(user_id: str) -> list[str]:
    """Replace all of a user's recovery codes with a fresh set, returning the plaintext codes."""
    db = await get_db()
    recovery_codes = [generate_recovery_code() for _ in range(RECOVERY_CODE_COUNT)]
    await db.usermfarecoverycode.delete_many(where={"userId": user_id})
    for recovery_code in recovery_codes:
        await db.usermfarecoverycode.create(
            data={
                "userId": user_id,
                "codeHash": hash_recovery_code(recovery_code),
            }
        )
    return recovery_codes


async def begin_totp_enrollment(user: Any, *, allow_replace: bool = False) -> dict[str, str]:
    secret = generate_totp_secret()
    db = await get_db()
    existing = await db.usermfafactor.find_unique(where={"userId_factorType": {"userId": user.id, "factorType": "totp"}})
    if existing and getattr(existing, "enabled", False) and not allow_replace:
        raise ValueError("TOTP is already enabled. Ask an administrator to reset MFA before re-enrolling.")
    issuer = str(getattr(settings, "server_name", "") or "Ragtime").strip() or "Ragtime"
    return {
        "secret": secret,
        "otpauth_uri": build_otpauth_uri(issuer=issuer, username=user.username, secret=secret),
        "enrollment_token": create_totp_enrollment_token(
            user_id=user.id,
            username=user.username,
            role=str(getattr(user, "role", "user") or "user"),
            secret=secret,
            rotation=allow_replace,
        ),
    }


async def confirm_totp_enrollment(user: Any, code: str, enrollment_token: str) -> tuple[bool, list[str]]:
    db = await get_db()
    claims = decode_totp_enrollment_token(enrollment_token)
    if not claims or claims.user_id != user.id:
        return False, []

    factor = await db.usermfafactor.find_unique(where={"userId_factorType": {"userId": user.id, "factorType": "totp"}})
    if factor and getattr(factor, "enabled", False) and not claims.rotation:
        return False, []

    result = verify_totp_code(claims.secret, code, last_used_step=None)
    if not result.valid or result.time_step is None:
        return False, []

    is_first_factor = not await user_has_enabled_totp(user.id) and not await user_has_enabled_webauthn(user.id)
    recovery_codes: list[str] = []
    if is_first_factor:
        recovery_codes = await regenerate_recovery_codes(user.id)
    await db.usermfafactor.upsert(
        where={"userId_factorType": {"userId": user.id, "factorType": "totp"}},
        data={
            "create": {
                "userId": user.id,
                "factorType": "totp",
                "label": "Authenticator app",
                "secretEncrypted": encrypt_secret(claims.secret),
                "enabled": True,
                "confirmedAt": datetime.now(timezone.utc),
                "lastUsedStep": result.time_step,
                "lastUsedAt": datetime.now(timezone.utc),
            },
            "update": {
                "secretEncrypted": encrypt_secret(claims.secret),
                "enabled": True,
                "confirmedAt": datetime.now(timezone.utc),
                "lastUsedStep": result.time_step,
                "lastUsedAt": datetime.now(timezone.utc),
            },
        },
    )
    return True, recovery_codes


async def verify_user_mfa_code(user: Any, code: str) -> tuple[bool, str | None]:
    db = await get_db()
    allowed_methods = await get_allowed_mfa_methods()

    if TOTP_METHOD in allowed_methods:
        factor = await get_enabled_totp_factor(user.id)
        if factor and getattr(factor, "enabled", False) and getattr(factor, "secretEncrypted", None):
            secret = decrypt_secret(factor.secretEncrypted)
            result = verify_totp_code(
                secret,
                code,
                last_used_step=getattr(factor, "lastUsedStep", None),
            )
            if result.valid and result.time_step is not None:
                await db.usermfafactor.update(
                    where={"id": factor.id},
                    data={"lastUsedStep": result.time_step, "lastUsedAt": datetime.now(timezone.utc)},
                )
                return True, "totp"

    recovery_codes = await db.usermfarecoverycode.find_many(where={"userId": user.id, "usedAt": None})
    for recovery_code in recovery_codes:
        if verify_recovery_code(code, recovery_code.codeHash):
            await db.usermfarecoverycode.update(
                where={"id": recovery_code.id},
                data={"usedAt": datetime.now(timezone.utc)},
            )
            return True, "recovery_code"
    return False, None


def hash_trusted_device_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def trusted_device_satisfies_mfa(user_id: str, token: str | None) -> bool:
    if not token:
        return False
    db = await get_db()
    trusted = await db.usermfatrusteddevice.find_unique(where={"tokenHash": hash_trusted_device_token(token)})
    if not trusted or trusted.userId != user_id:
        return False
    expires_at = trusted.expiresAt
    if isinstance(expires_at, datetime):
        comparable = expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=timezone.utc)
        if comparable <= datetime.now(timezone.utc):
            await db.usermfatrusteddevice.delete(where={"id": trusted.id})
            return False
    await db.usermfatrusteddevice.update(where={"id": trusted.id}, data={"lastUsedAt": datetime.now(timezone.utc)})
    return True


async def create_trusted_device(
    *,
    user_id: str,
    user_agent: str | None,
    ip_address: str | None,
    days: int = 30,
) -> tuple[str, datetime]:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=max(1, min(int(days), 365)))
    db = await get_db()
    await db.usermfatrusteddevice.create(
        data={
            "userId": user_id,
            "tokenHash": hash_trusted_device_token(token),
            "userAgent": user_agent,
            "ipAddress": ip_address,
            "expiresAt": expires_at,
        }
    )
    return token, expires_at


async def reset_user_mfa(user_id: str) -> None:
    db = await get_db()
    await db.usermfafactor.delete_many(where={"userId": user_id})
    await db.usermfarecoverycode.delete_many(where={"userId": user_id})
    await db.usermfatrusteddevice.delete_many(where={"userId": user_id})
    await db.userwebauthncredential.delete_many(where={"userId": user_id})
