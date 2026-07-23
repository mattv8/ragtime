from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

from ragtime.http_api.models import (
    HttpApiConnectionConfig,
    HttpApiOAuthClientAuthMethod,
    HttpApiOAuthFlow,
)
from ragtime.http_api.security import validate_base_url

_SESSION_TTL = 15 * 60
_MAX_SESSIONS = 128


@dataclass(frozen=True)
class OAuthDiscoveryResult:
    issuer: str
    authorization_endpoint: str = ""
    device_authorization_endpoint: str = ""
    token_endpoint: str = ""
    grant_types_supported: list[str] = field(default_factory=list)
    code_challenge_methods_supported: list[str] = field(default_factory=list)
    scopes_supported: list[str] = field(default_factory=list)
    token_endpoint_auth_methods_supported: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OAuthCredentialUpdate:
    access_token: str
    refresh_token: str = ""
    token_type: str = "Bearer"
    expires_at: str = ""

    def as_config(self) -> dict[str, str]:
        return {
            "oauth_access_token": self.access_token,
            "oauth_refresh_token": self.refresh_token,
            "oauth_token_type": self.token_type,
            "oauth_token_expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class OAuthStartResult:
    session_id: str
    status: str
    authorization_url: str = ""
    verification_uri: str = ""
    verification_uri_complete: str = ""
    user_code: str = ""
    expires_in: int | None = None
    interval: int = 5
    callback_url: str = ""


@dataclass(frozen=True)
class OAuthPollResult:
    status: str
    session_id: str = ""
    retry_after: int = 0
    verification_uri: str = ""
    verification_uri_complete: str = ""
    user_code: str = ""


@dataclass
class _OAuthSession:
    session_id: str
    owner_id: str
    config: HttpApiConnectionConfig
    callback_url: str
    expires_at: float
    state: str = ""
    verifier: str = ""
    device_code: str = ""
    interval: int = 5
    next_poll_at: float = 0.0
    credentials: OAuthCredentialUpdate | None = None
    status: str = "pending"
    verification_uri: str = ""
    verification_uri_complete: str = ""
    user_code: str = ""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _normalized_issuer(value: str) -> str:
    parsed = validate_base_url(value.rstrip("/"), debug_mode=False)
    scheme = parsed.scheme.decode() if isinstance(parsed.scheme, bytes) else str(parsed.scheme)
    netloc = parsed.netloc.decode() if isinstance(parsed.netloc, bytes) else str(parsed.netloc)
    return urlunsplit((scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _expiry_iso(expires_in: Any) -> str:
    try:
        seconds = max(0, int(expires_in))
    except (TypeError, ValueError):
        seconds = 0
    return datetime.fromtimestamp(time.time() + seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z") if seconds else ""


def _credential_update(payload: dict[str, Any]) -> OAuthCredentialUpdate:
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise ValueError("OAuth provider did not return an access token")
    return OAuthCredentialUpdate(
        access_token=token,
        refresh_token=str(payload.get("refresh_token") or ""),
        token_type=str(payload.get("token_type") or "Bearer"),
        expires_at=_expiry_iso(payload.get("expires_in")),
    )


class HttpApiOAuthManager:
    def __init__(self, broker: Any | None = None, *, clock: Any | None = None) -> None:
        self._broker = broker
        self._clock = clock or time.monotonic
        self._sessions: dict[str, _OAuthSession] = {}
        self._lock = asyncio.Lock()

    @property
    def broker(self) -> Any:
        if self._broker is None:
            from ragtime.http_api.service import http_api_broker

            self._broker = http_api_broker
        return self._broker

    async def discover(self, issuer_url: str) -> OAuthDiscoveryResult:
        issuer = _normalized_issuer(issuer_url)
        parsed = urlsplit(issuer)
        path = parsed.path.strip("/")
        rfc_path = f"/.well-known/oauth-authorization-server/{path}" if path else "/.well-known/oauth-authorization-server"
        oidc_path = f"/{path}/.well-known/openid-configuration" if path else "/.well-known/openid-configuration"
        status, metadata = await self.broker.oauth_request_json(urlunsplit((parsed.scheme, parsed.netloc, rfc_path, "", "")))
        if status == 404:
            status, metadata = await self.broker.oauth_request_json(urlunsplit((parsed.scheme, parsed.netloc, oidc_path, "", "")))
        if status != 200 or not isinstance(metadata, dict):
            raise ValueError("OAuth provider discovery failed")
        if _normalized_issuer(str(metadata.get("issuer") or "")) != issuer:
            raise ValueError("OAuth provider issuer did not match the requested issuer")
        result = OAuthDiscoveryResult(
            issuer=issuer,
            authorization_endpoint=str(metadata.get("authorization_endpoint") or ""),
            device_authorization_endpoint=str(metadata.get("device_authorization_endpoint") or ""),
            token_endpoint=str(metadata.get("token_endpoint") or ""),
            grant_types_supported=sorted({str(x) for x in metadata.get("grant_types_supported", [])}),
            code_challenge_methods_supported=sorted({str(x) for x in metadata.get("code_challenge_methods_supported", [])}),
            scopes_supported=sorted({str(x) for x in metadata.get("scopes_supported", [])}),
            token_endpoint_auth_methods_supported=sorted({str(x) for x in metadata.get("token_endpoint_auth_methods_supported", [])}),
        )
        for endpoint in (result.authorization_endpoint, result.device_authorization_endpoint, result.token_endpoint):
            if endpoint:
                await self.broker.validate_oauth_url(endpoint)
        return result

    async def start(self, owner_id: str, config: HttpApiConnectionConfig, callback_url: str) -> OAuthStartResult:
        if getattr(config.auth_mode, "value", config.auth_mode) != "oauth2":
            raise ValueError("OAuth configuration is required")
        session = _OAuthSession(secrets.token_urlsafe(24), owner_id, config, callback_url, self._clock() + _SESSION_TTL)
        async with self._lock:
            self._evict()
            if len(self._sessions) >= _MAX_SESSIONS:
                oldest = min(self._sessions, key=lambda key: self._sessions[key].expires_at)
                self._sessions.pop(oldest, None)
            self._sessions[session.session_id] = session
        if config.oauth_flow == HttpApiOAuthFlow.DEVICE_CODE:
            status, payload = await self.broker.oauth_request_json(
                config.oauth_device_authorization_url,
                method="POST",
                form={"client_id": config.oauth_client_id, "scope": " ".join(config.oauth_scopes)},
                client_auth=config.oauth_client_auth_method,
                client_id=config.oauth_client_id,
                client_secret=config.oauth_client_secret,
            )
            if status != 200:
                session.status = "failed"
                raise ValueError("OAuth device authorization failed")
            session.device_code = str(payload.get("device_code") or "")
            if not session.device_code:
                raise ValueError("OAuth device authorization failed")
            raw_interval = payload.get("interval", 5)
            try:
                interval = 5 if raw_interval is None else int(raw_interval)
            except (TypeError, ValueError):
                interval = 5
            session.interval = max(1, min(60, interval))
            session.next_poll_at = self._clock()
            session.verification_uri = str(payload.get("verification_uri") or payload.get("verification_url") or "")
            session.verification_uri_complete = str(payload.get("verification_uri_complete") or "")
            session.user_code = str(payload.get("user_code") or "")
            return OAuthStartResult(
                session.session_id,
                "pending",
                verification_uri=session.verification_uri,
                verification_uri_complete=session.verification_uri_complete,
                user_code=session.user_code,
                expires_in=int(payload.get("expires_in") or 0) or None,
                interval=session.interval,
            )
        session.state = secrets.token_urlsafe(24)
        session.verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(session.verifier.encode()).digest()).rstrip(b"=").decode()
        params = {
            "response_type": "code",
            "client_id": config.oauth_client_id,
            "redirect_uri": callback_url,
            "scope": " ".join(config.oauth_scopes),
            "state": session.state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        return OAuthStartResult(
            session.session_id,
            "pending",
            authorization_url=f"{config.oauth_authorization_url}{'&' if '?' in config.oauth_authorization_url else '?'}{urlencode(params)}",
            callback_url=callback_url,
        )

    async def poll(self, owner_id: str, session_id: str) -> OAuthPollResult:
        session = self._get(owner_id, session_id)
        async with session.lock:
            if self._clock() >= session.expires_at:
                session.status = "expired"
            if session.status != "pending":
                return self._result(session)
            if session.config.oauth_flow == HttpApiOAuthFlow.AUTHORIZATION_CODE_PKCE:
                return self._result(session)
            now = self._clock()
            if now < session.next_poll_at:
                return OAuthPollResult(
                    "pending",
                    session.session_id,
                    max(1, int(session.next_poll_at - now)),
                    session.verification_uri,
                    session.verification_uri_complete,
                    session.user_code,
                )
            status, payload = await self.broker.oauth_request_json(
                session.config.oauth_token_url,
                method="POST",
                form={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": session.device_code,
                    "client_id": session.config.oauth_client_id,
                },
                client_auth=session.config.oauth_client_auth_method,
                client_id=session.config.oauth_client_id,
                client_secret=session.config.oauth_client_secret,
            )
            error = str(payload.get("error") or "")
            if status == 200:
                session.credentials = _credential_update(payload)
                session.status = "connected"
            elif error == "authorization_pending":
                session.next_poll_at = now + session.interval
            elif error == "slow_down":
                session.interval = min(60, session.interval + 5)
                session.next_poll_at = now + session.interval
            elif error in {"access_denied", "expired_token"}:
                session.status = "failed" if error == "access_denied" else "expired"
            else:
                session.status = "failed"
            return self._result(session)

    async def complete_authorization_code(self, state: str, code: str | None, error: str | None) -> OAuthPollResult:
        async with self._lock:
            session = next((item for item in self._sessions.values() if item.state == state), None)
        if session is None:
            return OAuthPollResult("failed")
        async with session.lock:
            if session.state != state or self._clock() >= session.expires_at or session.status != "pending":
                return OAuthPollResult("failed", session.session_id)
            session.state = ""
            if error or not code:
                session.status = "failed"
                return self._result(session)
            form = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": session.callback_url,
                "client_id": session.config.oauth_client_id,
                "code_verifier": session.verifier,
            }
            status, payload = await self.broker.oauth_request_json(
                session.config.oauth_token_url,
                method="POST",
                form=form,
                client_auth=session.config.oauth_client_auth_method,
                client_id=session.config.oauth_client_id,
                client_secret=session.config.oauth_client_secret,
            )
            if status != 200:
                session.status = "failed"
            else:
                session.credentials = _credential_update(payload)
                session.status = "connected"
            return self._result(session)

    async def peek_credentials(self, owner_id: str, session_id: str) -> OAuthCredentialUpdate:
        session = self._get(owner_id, session_id)
        if session.status != "connected" or session.credentials is None:
            raise ValueError("OAuth session is not connected")
        return session.credentials

    async def consume(self, owner_id: str, session_id: str) -> None:
        session = self._get(owner_id, session_id)
        self._sessions.pop(session.session_id, None)

    def _get(self, owner_id: str, session_id: str) -> _OAuthSession:
        self._evict()
        session = self._sessions.get(session_id)
        if session is None or session.owner_id != owner_id:
            raise ValueError("OAuth session is unavailable")
        if self._clock() >= session.expires_at:
            session.status = "expired"
        return session

    def _evict(self) -> None:
        now = self._clock()
        for key in [key for key, session in self._sessions.items() if session.expires_at <= now]:
            self._sessions.pop(key, None)

    def _result(self, session: _OAuthSession) -> OAuthPollResult:
        retry_after = max(0, int(session.next_poll_at - self._clock())) if session.status == "pending" else 0
        return OAuthPollResult(session.status, session.session_id, retry_after, session.verification_uri, session.verification_uri_complete, session.user_code)


http_api_oauth_manager = HttpApiOAuthManager()
