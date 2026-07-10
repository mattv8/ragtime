from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import Request
from jose import JWTError, jwt  # type: ignore[import-untyped]
from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import (
    base64url_to_bytes,
    bytes_to_base64url,
    options_to_json,
)
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    AuthenticatorTransport,
    PublicKeyCredentialDescriptor,
    UserVerificationRequirement,
)

from ragtime.config.settings import settings
from ragtime.core.database import get_db
from ragtime.core.logging import get_logger

try:
    from prisma.errors import UniqueViolationError
except ImportError:  # pragma: no cover - defensive fallback for older Prisma client stubs
    UniqueViolationError = None  # type: ignore[assignment]

logger = get_logger(__name__)

WEBAUTHN_REGISTER_PURPOSE = "webauthn_register"
WEBAUTHN_AUTHN_PURPOSE = "webauthn_authn"
WEBAUTHN_CHALLENGE_TTL_SECONDS = 300
WEBAUTHN_DEFAULT_CREDENTIAL_NAME = "Passkey"
WEBAUTHN_CREDENTIAL_NAME_MAX_LENGTH = 100

_consumed_jtis: dict[str, float] = {}


class WebauthnError(Exception):
    """User-facing error with a message safe to return as an API detail."""


@dataclass(frozen=True)
class WebauthnChallengeClaims:
    user_id: str
    purpose: str
    challenge: bytes
    jti: str
    exp: datetime


def _prune_consumed_jtis(now: float | None = None) -> None:
    now = now if now is not None else time.time()
    expired = [jti for jti, exp in _consumed_jtis.items() if exp <= now]
    for jti in expired:
        _consumed_jtis.pop(jti, None)


async def _consume_jti(jti: str, exp: float) -> None:
    db = await get_db()
    challenge_table = getattr(db, "userwebauthnchallenge", None)
    if challenge_table is not None:
        now = datetime.now(timezone.utc)
        await challenge_table.delete_many(where={"expiresAt": {"lte": now}})
        try:
            await challenge_table.create(
                data={
                    "id": str(uuid4()),
                    "jti": jti,
                    "expiresAt": datetime.fromtimestamp(exp, tz=timezone.utc),
                }
            )
        except Exception as exc:
            if UniqueViolationError is not None and isinstance(exc, UniqueViolationError):
                raise WebauthnError("This WebAuthn challenge has already been used.") from exc
            if "duplicate jti" in str(exc).casefold():
                raise WebauthnError("This WebAuthn challenge has already been used.") from exc
            raise
        return

    _prune_consumed_jtis()
    if jti in _consumed_jtis:
        raise WebauthnError("This WebAuthn challenge has already been used.")
    _consumed_jtis[jti] = exp


def _create_challenge_token(
    *,
    user_id: str,
    purpose: str,
    challenge: bytes,
) -> str:
    expire = datetime.now(timezone.utc) + timedelta(seconds=WEBAUTHN_CHALLENGE_TTL_SECONDS)
    jti = str(uuid4())
    payload = {
        "sub": user_id,
        "purpose": purpose,
        "challenge": bytes_to_base64url(challenge),
        "jti": jti,
        "exp": expire,
    }
    return jwt.encode(payload, settings.encryption_key, algorithm=settings.jwt_algorithm)


def _decode_challenge_token(token: str, *, expected_purpose: str) -> WebauthnChallengeClaims | None:
    try:
        payload = jwt.decode(token, settings.encryption_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        logger.debug("WebAuthn challenge token decode failed: %s", exc)
        return None

    if payload.get("purpose") != expected_purpose:
        return None
    try:
        return WebauthnChallengeClaims(
            user_id=str(payload["sub"]),
            purpose=str(payload["purpose"]),
            challenge=base64url_to_bytes(str(payload["challenge"])),
            jti=str(payload["jti"]),
            exp=datetime.fromtimestamp(int(payload["exp"]), tz=timezone.utc),
        )
    except (KeyError, TypeError, ValueError):
        return None


def resolve_rp(request: Request) -> tuple[str, list[str]]:
    """Determine the RP ID and expected origins for WebAuthn.

    RP ID is the hostname of EXTERNAL_BASE_URL when configured (the canonical,
    admin-anchored host). Without it, the browser-reported Origin header is
    the ground truth for the user-facing host: dev servers and reverse
    proxies (for example the Vite proxy with changeOrigin) rewrite the Host
    header on the way to the backend, so the request hostname is only a last
    resort. Expected origins include the external origin (when set) plus the
    request Origin header when its hostname matches the RP ID or is a
    subdomain of it.
    """
    expected_origins: list[str] = []

    external_url = (settings.external_base_url or "").strip()
    rp_id: str | None = None
    if external_url:
        parsed = urlparse(external_url)
        rp_id = parsed.hostname
        if parsed.scheme and parsed.netloc:
            external_origin = f"{parsed.scheme}://{parsed.netloc}".lower().rstrip("/")
            if external_origin not in expected_origins:
                expected_origins.append(external_origin)

    origin_header = (request.headers.get("origin") or "").strip()
    parsed_origin = urlparse(origin_header) if origin_header else None
    origin_hostname = parsed_origin.hostname if parsed_origin else None

    if not rp_id:
        rp_id = origin_hostname or request.url.hostname

    if not rp_id:
        raise WebauthnError("Unable to determine a valid WebAuthn RP ID for this request.")

    if parsed_origin and origin_hostname:
        suffix = f".{rp_id}"
        if origin_hostname == rp_id or origin_hostname.endswith(suffix):
            origin_value = f"{parsed_origin.scheme}://{parsed_origin.netloc}".lower().rstrip("/")
            if origin_value not in expected_origins:
                expected_origins.append(origin_value)

    if not expected_origins:
        raise WebauthnError("Unable to determine a valid WebAuthn origin for this request.")

    return rp_id, expected_origins


def _normalize_credential_name(name: str | None) -> str:
    name = (name or WEBAUTHN_DEFAULT_CREDENTIAL_NAME).strip()
    return name[:WEBAUTHN_CREDENTIAL_NAME_MAX_LENGTH] or WEBAUTHN_DEFAULT_CREDENTIAL_NAME


def _credential_id_b64(cred: Any) -> str | None:
    return getattr(cred, "credentialId", None) or getattr(cred, "credential_id", None)


def _build_exclude_credentials(credentials: list[Any]) -> list[PublicKeyCredentialDescriptor]:
    descriptors: list[PublicKeyCredentialDescriptor] = []
    for cred in credentials:
        credential_id_b64 = _credential_id_b64(cred)
        if not credential_id_b64:
            continue
        descriptors.append(
            PublicKeyCredentialDescriptor(
                id=base64url_to_bytes(str(credential_id_b64)),
            )
        )
    return descriptors


def _build_allow_credentials(credentials: list[Any]) -> list[PublicKeyCredentialDescriptor]:
    descriptors: list[PublicKeyCredentialDescriptor] = []
    for cred in credentials:
        credential_id_b64 = _credential_id_b64(cred)
        transports_raw = getattr(cred, "transports", None) or []
        transports: list[AuthenticatorTransport] = []
        for t in transports_raw:
            try:
                transports.append(AuthenticatorTransport(str(t)))
            except ValueError:
                logger.debug("Ignoring unknown WebAuthn transport: %s", t)
        if credential_id_b64:
            descriptors.append(
                PublicKeyCredentialDescriptor(
                    id=base64url_to_bytes(str(credential_id_b64)),
                    transports=transports or None,
                )
            )
    return descriptors


async def begin_webauthn_registration(user: Any, request: Request) -> tuple[dict, str]:
    db = await get_db()
    existing = await db.userwebauthncredential.find_many(where={"userId": user.id})
    exclude_credentials = _build_exclude_credentials(existing)

    rp_id, _expected_origins = resolve_rp(request)
    challenge = secrets.token_bytes(32)

    server_name = str(getattr(settings, "server_name", "") or "").strip() or "Ragtime"
    options = generate_registration_options(
        rp_id=rp_id,
        rp_name=server_name,
        user_name=user.username,
        user_id=user.id.encode("utf-8"),
        user_display_name=getattr(user, "displayName", None) or getattr(user, "display_name", None) or user.username,
        challenge=challenge,
        authenticator_selection=AuthenticatorSelectionCriteria(
            user_verification=UserVerificationRequirement.PREFERRED,
        ),
        exclude_credentials=exclude_credentials or None,
    )

    options_dict: dict = json.loads(options_to_json(options))
    registration_token = _create_challenge_token(
        user_id=user.id,
        purpose=WEBAUTHN_REGISTER_PURPOSE,
        challenge=challenge,
    )
    return options_dict, registration_token


async def complete_webauthn_registration(
    user: Any,
    request: Request,
    registration_token: str,
    credential: dict,
    name: str | None,
) -> Any:
    claims = _decode_challenge_token(registration_token, expected_purpose=WEBAUTHN_REGISTER_PURPOSE)
    if not claims or claims.user_id != user.id:
        raise WebauthnError("Invalid or expired WebAuthn registration token.")

    await _consume_jti(claims.jti, claims.exp.timestamp())

    rp_id, expected_origins = resolve_rp(request)
    try:
        verification = verify_registration_response(
            credential=credential,
            expected_challenge=claims.challenge,
            expected_rp_id=rp_id,
            expected_origin=expected_origins,
        )
    except Exception as exc:
        logger.debug("WebAuthn registration verification failed: %s", exc)
        raise WebauthnError("WebAuthn registration could not be verified.") from exc

    credential_id_b64 = bytes_to_base64url(verification.credential_id)
    public_key_b64 = bytes_to_base64url(verification.credential_public_key)
    aaguid = getattr(verification, "aaguid", None)
    if aaguid is not None:
        aaguid = str(aaguid)

    transports: list[str] = []
    response = credential.get("response", {}) if isinstance(credential, dict) else {}
    if isinstance(response, dict):
        raw_transports = response.get("transports")
        if isinstance(raw_transports, list):
            transports = [str(t) for t in raw_transports]

    db = await get_db()
    try:
        row = await db.userwebauthncredential.create(
            data={
                "userId": user.id,
                "credentialId": credential_id_b64,
                "publicKey": public_key_b64,
                "signCount": int(verification.sign_count or 0),
                "transports": transports,
                "aaguid": aaguid or None,
                "name": _normalize_credential_name(name),
            }
        )
    except Exception as exc:
        logger.error("Failed to store WebAuthn credential: %s", exc)
        raise WebauthnError("Failed to store WebAuthn credential.") from exc

    return row


async def begin_webauthn_authentication(user: Any, request: Request) -> tuple[dict, str]:
    db = await get_db()
    credentials = await db.userwebauthncredential.find_many(where={"userId": user.id})
    if not credentials:
        raise WebauthnError("No WebAuthn credentials found for this user.")

    rp_id, _expected_origins = resolve_rp(request)
    challenge = secrets.token_bytes(32)
    allow_credentials = _build_allow_credentials(credentials)

    options = generate_authentication_options(
        rp_id=rp_id,
        challenge=challenge,
        allow_credentials=allow_credentials,
        user_verification=UserVerificationRequirement.PREFERRED,
    )

    options_dict: dict = json.loads(options_to_json(options))
    authentication_token = _create_challenge_token(
        user_id=user.id,
        purpose=WEBAUTHN_AUTHN_PURPOSE,
        challenge=challenge,
    )
    return options_dict, authentication_token


async def complete_webauthn_authentication(
    user: Any,
    request: Request,
    authentication_token: str,
    credential: dict,
) -> None:
    claims = _decode_challenge_token(authentication_token, expected_purpose=WEBAUTHN_AUTHN_PURPOSE)
    if not claims or claims.user_id != user.id:
        raise WebauthnError("Invalid or expired WebAuthn authentication token.")

    await _consume_jti(claims.jti, claims.exp.timestamp())

    credential_id_b64 = credential.get("id") if isinstance(credential, dict) else None
    if not credential_id_b64:
        raise WebauthnError("Missing credential ID in WebAuthn response.")

    db = await get_db()
    row = await db.userwebauthncredential.find_first(where={"userId": user.id, "credentialId": str(credential_id_b64)})
    if not row:
        raise WebauthnError("Unknown WebAuthn credential.")

    rp_id, expected_origins = resolve_rp(request)
    try:
        verification = verify_authentication_response(
            credential=credential,
            expected_challenge=claims.challenge,
            expected_rp_id=rp_id,
            expected_origin=expected_origins,
            credential_public_key=base64url_to_bytes(row.publicKey),
            credential_current_sign_count=int(row.signCount or 0),
        )
    except Exception as exc:
        logger.debug("WebAuthn authentication verification failed: %s", exc)
        raise WebauthnError("WebAuthn authentication could not be verified.") from exc

    await db.userwebauthncredential.update(
        where={"id": row.id},
        data={
            "signCount": int(verification.new_sign_count),
            "lastUsedAt": datetime.now(timezone.utc),
        },
    )


async def list_webauthn_credentials(user_id: str) -> list[Any]:
    db = await get_db()
    return await db.userwebauthncredential.find_many(
        where={"userId": user_id},
        order={"createdAt": "desc"},
    )


async def rename_webauthn_credential(user_id: str, cred_id: str, name: str) -> Any | None:
    db = await get_db()
    row = await db.userwebauthncredential.find_unique(where={"id": cred_id})
    if not row or getattr(row, "userId", None) != user_id:
        return None
    try:
        return await db.userwebauthncredential.update(
            where={"id": cred_id},
            data={"name": _normalize_credential_name(name)},
        )
    except Exception as exc:
        logger.debug("Rename WebAuthn credential failed: %s", exc)
        return None


async def delete_webauthn_credential(user_id: str, cred_id: str) -> bool:
    db = await get_db()
    try:
        result = await db.userwebauthncredential.delete_many(where={"id": cred_id, "userId": user_id})
        return getattr(result, "count", 0) > 0
    except Exception as exc:
        logger.debug("Delete WebAuthn credential failed: %s", exc)
        return False


async def user_has_enabled_webauthn(user_id: str) -> bool:
    db = await get_db()
    count = await db.userwebauthncredential.count(where={"userId": user_id})
    return count > 0
