from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import socket
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, cast
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import httpx
from fastapi import HTTPException
from jose import JWTError, jwt  # type: ignore[import-untyped]
from prisma.errors import ForeignKeyViolationError
from starlette.websockets import WebSocket

from prisma import fields as prisma_fields
from ragtime.config import settings
from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.core.workspace_ops import normalize_runtime_file_path
from ragtime.indexer.workspace_state import build_workspace_chat_state
from ragtime.userspace.models import (
    RuntimeOperationPhase,
    RuntimeSessionState,
    UserSpaceCapabilityTokenResponse,
    UserSpaceCollabSnapshotResponse,
    UserSpaceFileInfo,
    UserSpaceFileResponse,
    UserSpacePreviewLaunchResponse,
    UserSpacePreviewWarning,
    UserSpaceRuntimeActionResponse,
    UserSpaceRuntimeSession,
    UserSpaceRuntimeSessionResponse,
    UserSpaceRuntimeStatusResponse,
    UserSpaceWorkspaceTabStateResponse,
)
from ragtime.userspace.preview_probe import (
    build_preview_probe_url,
    is_preview_probe_response,
)
from ragtime.userspace.runtime_errors import RuntimeVersionConflictError
from ragtime.userspace.service import userspace_service

logger = get_logger(__name__)

_RUNTIME_CAPABILITY_TTL_SECONDS = 900
_RUNTIME_PREVIEW_BOOTSTRAP_TTL_SECONDS = 300
_RUNTIME_PREVIEW_SESSION_TTL_SECONDS = 14400
_RUNTIME_PREVIEW_SESSION_COOKIE_NAME = "userspace_preview_session"
_RUNTIME_DEVSERVER_PORT = 5173
_RUNTIME_PREVIEW_DEFAULT_BASE = f"http://127.0.0.1:{_RUNTIME_DEVSERVER_PORT}"
_RUNTIME_PROVIDER_MANAGER = "microvm_pool_v1"
_RUNTIME_PREVIEW_GRANT_KIND = "userspace_preview_grant"
_RUNTIME_PREVIEW_SESSION_KIND = "userspace_preview_session"
_DEFAULT_USERSPACE_PREVIEW_BASE_DOMAIN = "userspace-preview.lvh.me"
_RUNTIME_PREVIEW_UPSTREAM_CACHE_TTL_SECONDS = 300
_RUNTIME_PROVIDER_STATUS_CACHE_TTL_SECONDS = 2.0
_RUNTIME_PROVIDER_STATUS_STALE_FALLBACK_SECONDS = 8.0
_RUNTIME_PREVIEW_PROBE_CACHE_TTL_SECONDS = 3.0
_RUNTIME_PUBLIC_PREVIEW_DNS_CACHE_TTL_SECONDS = 30.0
_RUNTIME_PUBLIC_PREVIEW_PROBE_CACHE_TTL_SECONDS = 30.0


@dataclass
class _CollabDocState:
    workspace_id: str
    file_path: str
    content: str
    version: int = 0
    clients: set[WebSocket] = field(default_factory=set)


@dataclass(frozen=True)
class _RuntimeManagerRequestConfig:
    base_url: str
    headers: dict[str, str]
    timeout_seconds: float
    retry_attempts: int
    retry_base_delay_seconds: float


@dataclass(frozen=True)
class _PreviewUpstreamCacheEntry:
    provider_session_id: str | None
    base_url: str
    cached_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class _ProviderStatusCacheEntry:
    payload: dict[str, Any]
    cached_at: datetime


@dataclass(frozen=True)
class _PreviewProbeCacheEntry:
    ok: bool
    checked_at: datetime


class UserSpaceRuntimeService:
    def __init__(self) -> None:
        self._collab_docs: dict[tuple[str, str], _CollabDocState] = {}
        self._collab_lock = asyncio.Lock()
        self._runtime_cache_lock = asyncio.Lock()
        self._workspace_generation: dict[str, int] = {}
        self._collab_presence: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        self._workspace_events: dict[str, asyncio.Condition] = {}
        self._workspace_event_payload: dict[str, dict[str, Any]] = {}
        self._runtime_watch_workspaces: set[str] = set()
        self._runtime_watch_signatures: dict[str, tuple[Any, ...]] = {}
        self._preview_base_domains: set[str] = set()
        self._preview_upstream_cache: dict[str, _PreviewUpstreamCacheEntry] = {}
        self._provider_status_cache: dict[str, _ProviderStatusCacheEntry] = {}
        self._preview_probe_cache: dict[str, _PreviewProbeCacheEntry] = {}
        self._public_preview_dns_cache: dict[str, _PreviewProbeCacheEntry] = {}
        self._public_preview_probe_cache: dict[str, _PreviewProbeCacheEntry] = {}
        self._probe_tasks: dict[str, asyncio.Task[bool]] = {}
        self._runtime_watch_task: asyncio.Task[None] | None = None
        self._runtime_watch_task_lock = asyncio.Lock()
        self._runtime_watch_interval_seconds = float(
            getattr(settings, "userspace_runtime_watch_interval_seconds", 1.0)
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _control_plane_base_url() -> str:
        scheme = "https" if bool(getattr(settings, "enable_https", False)) else "http"
        port = str(os.getenv("PORT", "8000") or "8000").strip() or "8000"
        return f"{scheme}://127.0.0.1:{port}"

    @staticmethod
    def _default_public_control_plane_origin() -> str:
        configured = str(getattr(settings, "external_base_url", "") or "").strip()
        if configured:
            parsed = urlsplit(configured)
            return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")

        scheme = "https" if bool(getattr(settings, "enable_https", False)) else "http"
        port = int(getattr(settings, "port", 8000) or 8000)
        host = "localhost"
        default_port = (scheme == "https" and port == 443) or (
            scheme == "http" and port == 80
        )
        netloc = host if default_port else f"{host}:{port}"
        return f"{scheme}://{netloc}"

    @staticmethod
    def _preview_host_label(workspace_id: str) -> str:
        normalized = re.sub(r"[^a-z0-9-]", "-", str(workspace_id or "").strip().lower())
        normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
        if not normalized:
            digest = hashlib.sha256(
                f"userspace-preview:{workspace_id}:{settings.encryption_key}".encode(
                    "utf-8"
                )
            ).hexdigest()
            normalized = digest[:20]
        return normalized[:63].rstrip("-")

    @staticmethod
    def _is_localhost_like_host(hostname: str | None) -> bool:
        normalized = str(hostname or "").strip().strip(".").lower().strip("[]")
        return normalized in {"localhost", "127.0.0.1", "::1"} or normalized.endswith(
            ".localhost"
        )

    @staticmethod
    def _normalize_preview_base_domain_candidate(value: str | None) -> str | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        parsed = urlsplit(raw if "://" in raw else f"//{raw}")
        candidate = (parsed.hostname or raw).strip().strip(".").lower()
        if candidate.startswith("*."):
            candidate = candidate[2:]
        if not candidate or any(char.isspace() for char in candidate):
            return None
        return candidate

    def _configured_preview_base_domain(self) -> str | None:
        return self._normalize_preview_base_domain_candidate(
            getattr(settings, "userspace_preview_base_domain", "")
        )

    def _resolve_preview_base_domain(
        self,
        control_plane_origin: str | None,
    ) -> tuple[str, str]:
        configured = self._configured_preview_base_domain()
        if configured:
            return configured, "configured"

        parsed = urlsplit(str(control_plane_origin or "").strip())
        hostname = str(parsed.hostname or "").strip().strip(".").lower()
        if not hostname:
            raise HTTPException(
                status_code=500,
                detail="Unable to derive userspace preview domain from the current origin",
            )

        if bool(
            getattr(settings, "debug_mode", False)
        ) and self._is_localhost_like_host(hostname):
            return _DEFAULT_USERSPACE_PREVIEW_BASE_DOMAIN, "debug_default"
        if self._is_localhost_like_host(hostname):
            return "localhost", "derived"
        return hostname, "derived"

    def _remember_preview_base_domain(self, base_domain: str) -> None:
        normalized = self._normalize_preview_base_domain_candidate(base_domain)
        if normalized:
            self._preview_base_domains.add(normalized)

    def get_preview_base_domains(self) -> set[str]:
        domains = {
            domain
            for domain in self._preview_base_domains
            if self._normalize_preview_base_domain_candidate(domain)
        }
        configured = self._configured_preview_base_domain()
        if configured:
            domains.add(configured)
        if bool(getattr(settings, "debug_mode", False)):
            domains.add(_DEFAULT_USERSPACE_PREVIEW_BASE_DOMAIN)
        return domains

    async def _build_preview_launch_warning(
        self,
        *,
        preview_origin: str,
        resolved_base_domain: str,
        source: str,
    ) -> UserSpacePreviewWarning | None:
        parsed = urlsplit(preview_origin)
        preview_host = str(parsed.hostname or "").strip().lower()
        if not preview_host:
            return None

        if source == "debug_default":
            return None
        if self._is_localhost_like_host(
            resolved_base_domain
        ) or resolved_base_domain.endswith(".localhost"):
            return None
        if resolved_base_domain == _DEFAULT_USERSPACE_PREVIEW_BASE_DOMAIN:
            return UserSpacePreviewWarning(
                issue_code="preview_dev_domain_outside_debug",
                title="Userspace preview domain is using the dev-only default",
                warnings=[
                    "userspace-preview.lvh.me is only meant for local DEBUG_MODE development.",
                    "Either set USERSPACE_PREVIEW_BASE_DOMAIN to a real wildcard DNS/TLS domain, or unset it to auto-derive from your Ragtime origin.",
                ],
                dismiss_key=f"userspace-preview-warning:{resolved_base_domain}:dev-domain",
                resolved_base_domain=resolved_base_domain,
                preview_host=preview_host,
                source=cast(Any, source),
            )

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not await self._resolve_preview_host_cached(preview_host, port):
            return UserSpacePreviewWarning(
                issue_code="preview_dns_unresolvable",
                title="Userspace preview DNS is not resolving",
                warnings=[
                    f"{preview_host} did not resolve during preview launch checks.",
                    f"Either configure wildcard DNS/TLS for *.{resolved_base_domain} pointing back to Ragtime, or set USERSPACE_PREVIEW_BASE_DOMAIN to a domain with working wildcard resolution.",
                ],
                dismiss_key=f"userspace-preview-warning:{resolved_base_domain}:dns",
                resolved_base_domain=resolved_base_domain,
                preview_host=preview_host,
                source=cast(Any, source),
            )

        if not await self._probe_public_preview_origin_cached(preview_origin):
            return UserSpacePreviewWarning(
                issue_code="preview_host_unreachable",
                title="Userspace preview subdomains are not reaching Ragtime",
                warnings=[
                    f"{preview_host} resolves in DNS but does not reach Ragtime's public preview endpoint.",
                    f"Either fix wildcard routing/TLS for *.{resolved_base_domain} to reach Ragtime, or set USERSPACE_PREVIEW_BASE_DOMAIN to a domain with working wildcard proxy.",
                ],
                dismiss_key=f"userspace-preview-warning:{resolved_base_domain}:routing",
                resolved_base_domain=resolved_base_domain,
                preview_host=preview_host,
                source=cast(Any, source),
            )
        return None

    def _build_preview_origin(
        self,
        workspace_id: str,
        *,
        control_plane_origin: str | None = None,
    ) -> str:
        parsed = urlsplit(
            control_plane_origin or self._default_public_control_plane_origin()
        )
        base_domain, _ = self._resolve_preview_base_domain(
            control_plane_origin or self._default_public_control_plane_origin()
        )
        self._remember_preview_base_domain(base_domain)
        host = f"{self._preview_host_label(workspace_id)}.{base_domain}"
        port = parsed.port
        use_default_port = (parsed.scheme == "https" and port in {None, 443}) or (
            parsed.scheme == "http" and port in {None, 80}
        )
        netloc = host if use_default_port else f"{host}:{port}"
        return urlunsplit((parsed.scheme or "http", netloc, "", "", "")).rstrip("/")

    def get_preview_origin(
        self,
        workspace_id: str,
        *,
        control_plane_origin: str | None = None,
    ) -> str:
        return self._build_preview_origin(
            workspace_id,
            control_plane_origin=control_plane_origin,
        )

    def _build_preview_token(
        self,
        claims: dict[str, Any],
        *,
        ttl_seconds: int,
    ) -> tuple[str, datetime]:
        now = self._utc_now()
        expires_at = now + timedelta(seconds=max(60, min(ttl_seconds, 86400)))
        payload = {key: value for key, value in claims.items() if value is not None}
        payload.update(
            {
                "iat": int(now.timestamp()),
                "exp": int(expires_at.timestamp()),
                "jti": str(uuid4()),
            }
        )
        token = jwt.encode(
            payload, settings.encryption_key, algorithm=settings.jwt_algorithm
        )
        return token, expires_at

    def _sanitize_preview_path(self, path: str | None, query: str | None = None) -> str:
        normalized = "/" + (path or "").lstrip("/")
        if normalized.startswith("/__ragtime/"):
            normalized = "/"
        if query:
            normalized = f"{normalized}?{query}"
        return normalized

    async def _build_workspace_preview_launch_response(
        self,
        *,
        workspace_id: str,
        subject_user_id: str | None,
        control_plane_origin: str,
        path: str,
        parent_origin: str | None,
        mode: str,
        share_token: str | None = None,
        owner_username: str | None = None,
        share_slug: str | None = None,
        share_access_mode: str | None = None,
    ) -> UserSpacePreviewLaunchResponse:
        resolved_base_domain, source = self._resolve_preview_base_domain(
            control_plane_origin
        )
        self._remember_preview_base_domain(resolved_base_domain)
        preview_origin = self._build_preview_origin(
            workspace_id,
            control_plane_origin=control_plane_origin,
        )
        preview_host = urlsplit(preview_origin).netloc.lower()
        grant_payload: dict[str, Any] = {
            "kind": _RUNTIME_PREVIEW_GRANT_KIND,
            "sub": subject_user_id,
            "workspace_id": workspace_id,
            "preview_mode": mode,
            "preview_host": preview_host,
            "parent_origin": (parent_origin or "").strip() or None,
            "target_path": path,
            "share_token": share_token,
            "owner_username": owner_username,
            "share_slug": share_slug,
        }
        if share_access_mode:
            grant_payload["share_access_mode"] = share_access_mode
        grant_token, expires_at = self._build_preview_token(
            grant_payload,
            ttl_seconds=_RUNTIME_PREVIEW_BOOTSTRAP_TTL_SECONDS,
        )
        bootstrap_query = urlencode({"grant": grant_token})
        preview_warning = await self._build_preview_launch_warning(
            preview_origin=preview_origin,
            resolved_base_domain=resolved_base_domain,
            source=source,
        )
        return UserSpacePreviewLaunchResponse(
            workspace_id=workspace_id,
            preview_origin=preview_origin,
            preview_url=f"{preview_origin}/__ragtime/bootstrap?{bootstrap_query}",
            expires_at=expires_at,
            preview_warning=preview_warning,
        )

    def build_preview_session_token(
        self, claims: dict[str, Any]
    ) -> tuple[str, datetime]:
        payload = dict(claims)
        payload["kind"] = _RUNTIME_PREVIEW_SESSION_KIND
        return self._build_preview_token(
            payload,
            ttl_seconds=_RUNTIME_PREVIEW_SESSION_TTL_SECONDS,
        )

    def verify_preview_token(self, token: str, *, expected_kind: str) -> dict[str, Any]:
        try:
            claims = cast(
                dict[str, Any],
                jwt.decode(
                    token,
                    settings.encryption_key,
                    algorithms=[settings.jwt_algorithm],
                ),
            )
        except JWTError as exc:
            raise HTTPException(
                status_code=401, detail="Invalid preview token"
            ) from exc

        if str(claims.get("kind") or "").strip() != expected_kind:
            raise HTTPException(status_code=401, detail="Invalid preview token")
        workspace_id = str(claims.get("workspace_id") or "").strip()
        if not workspace_id:
            raise HTTPException(status_code=401, detail="Invalid preview token")
        return claims

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(cast(float, value))
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(cast(str, value).strip())
        return None

    @staticmethod
    def _resolve_provider_last_error(
        payload: dict[str, Any],
        fallback: str | None = None,
    ) -> str | None:
        if "last_error" not in payload:
            return fallback
        raw_value = payload.get("last_error")
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            normalized = raw_value.strip()
            return normalized or None
        normalized = str(raw_value).strip()
        return normalized or None

    def _runtime_session_model(self, db: Any) -> Any:
        return getattr(db, "userspaceruntimesession")

    @staticmethod
    def _missing_runtime_session_field(exc: Exception) -> str | None:
        detail = str(exc)
        marker = "UserSpaceRuntimeSession.data."
        if marker not in detail:
            return None
        tail = detail.split(marker, 1)[1]
        missing_field = tail.split("`", 1)[0].split()[0].strip()
        return missing_field or None

    @staticmethod
    def _sanitize_pg_strings(data: dict[str, Any]) -> dict[str, Any]:
        """Strip null bytes (0x00) that PostgreSQL text columns reject."""
        cleaned: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str) and "\x00" in value:
                value = value.replace("\x00", "")
            cleaned[key] = value
        return cleaned

    async def _prisma_write_with_field_fallback(
        self,
        model: Any,
        operation: str,
        data: dict[str, Any],
        *,
        where: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a Prisma create/update, auto-stripping unknown fields on retry.

        Args:
            model: Prisma model handle.
            operation: ``"create"`` or ``"update"``.
            data: Field payload (will be copied and sanitized).
            where: Required for ``"update"`` operations.
        """
        payload = self._sanitize_pg_strings(dict(data))
        op_fn = getattr(model, operation)
        attempts = 0
        while True:
            try:
                kwargs: dict[str, Any] = {"data": payload}
                if where is not None:
                    kwargs["where"] = where
                return await op_fn(**kwargs)
            except Exception as exc:
                attempts += 1
                missing_field = self._missing_runtime_session_field(exc)
                if not missing_field or missing_field not in payload or attempts > 8:
                    raise
                payload.pop(missing_field, None)
                logger.warning(
                    "Runtime session %s dropped unsupported prisma field '%s'",
                    operation,
                    missing_field,
                )

    async def _runtime_session_update_row(
        self,
        model: Any,
        row_id: str,
        data: dict[str, Any],
    ) -> Any:
        return await self._prisma_write_with_field_fallback(
            model, "update", data, where={"id": row_id}
        )

    async def _runtime_session_create_row(
        self,
        model: Any,
        data: dict[str, Any],
    ) -> Any:
        return await self._prisma_write_with_field_fallback(model, "create", data)

    def _runtime_audit_model(self, db: Any) -> Any:
        return getattr(db, "userspaceruntimeauditevent")

    # Cached audit insert shape index to avoid probing every call.
    _audit_shape_index: int | None = None

    def _collab_doc_model(self, db: Any) -> Any:
        return getattr(db, "userspacecollabdoc")

    def _to_runtime_session(self, row: Any) -> UserSpaceRuntimeSession:
        state_value = getattr(row, "state", "stopped")
        state_text = (
            state_value
            if isinstance(state_value, str)
            else str(getattr(state_value, "value", state_value))
        )
        if state_text not in {"starting", "running", "stopping", "stopped", "error"}:
            state_text = "stopped"
        state = cast(RuntimeSessionState, state_text)
        return UserSpaceRuntimeSession(
            id=str(getattr(row, "id", "")),
            workspace_id=str(getattr(row, "workspaceId", "")),
            leased_by_user_id=str(getattr(row, "leasedByUserId", "")),
            state=state,
            runtime_provider=str(getattr(row, "runtimeProvider", "microvm_pool_v1")),
            provider_session_id=getattr(row, "providerSessionId", None),
            preview_internal_url=getattr(row, "previewInternalUrl", None),
            launch_framework=getattr(row, "launchFramework", None),
            launch_command=getattr(row, "launchCommand", None),
            launch_cwd=getattr(row, "launchCwd", None),
            launch_port=getattr(row, "launchPort", None),
            created_at=getattr(row, "createdAt"),
            updated_at=getattr(row, "updatedAt"),
            last_heartbeat_at=getattr(row, "lastHeartbeatAt", None),
            idle_expires_at=getattr(row, "idleExpiresAt", None),
            ttl_expires_at=getattr(row, "ttlExpiresAt", None),
            last_error=getattr(row, "lastError", None),
        )

    def _merge_provider_status(
        self,
        session: UserSpaceRuntimeSession,
        provider_status: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a Prisma update payload by merging provider status into session fields.

        Returns only fields that actually changed.  The caller can persist
        or discard the delta as appropriate.
        """
        _valid = {"starting", "running", "stopping", "stopped", "error"}
        state_raw = str(provider_status.get("state") or "").strip()
        state = state_raw if state_raw in _valid else session.state

        preview = (
            str(provider_status.get("preview_internal_url") or "").strip()
            or session.preview_internal_url
            or _RUNTIME_PREVIEW_DEFAULT_BASE
        )
        framework = (
            str(provider_status.get("launch_framework") or "").strip()
            or session.launch_framework
        )
        command = (
            str(provider_status.get("launch_command") or "").strip()
            or session.launch_command
        )
        cwd = str(provider_status.get("launch_cwd") or "").strip() or session.launch_cwd
        port = (
            self._optional_int(provider_status.get("launch_port"))
            or session.launch_port
        )
        last_error = self._resolve_provider_last_error(
            provider_status, fallback=session.last_error
        )

        update: dict[str, Any] = {}
        if state != session.state:
            update["state"] = state
        if preview != session.preview_internal_url:
            update["previewInternalUrl"] = preview
        if framework != session.launch_framework:
            update["launchFramework"] = framework
        if command != session.launch_command:
            update["launchCommand"] = command
        if cwd != session.launch_cwd:
            update["launchCwd"] = cwd
        if port != session.launch_port:
            update["launchPort"] = port
        if last_error != session.last_error:
            update["lastError"] = last_error

        return update

    def _normalize_file_path(self, file_path: str) -> str:
        return normalize_runtime_file_path(
            file_path,
            is_reserved_path=userspace_service.is_reserved_internal_path,
        )

    async def _get_active_session_row(self, workspace_id: str) -> Any | None:
        db = await get_db()
        model = self._runtime_session_model(db)
        return await model.find_first(
            where={
                "workspaceId": workspace_id,
                "state": {"in": ["starting", "running"]},
            },
            order={"updatedAt": "desc"},
        )

    async def _get_latest_session_row(self, workspace_id: str) -> Any | None:
        db = await get_db()
        model = self._runtime_session_model(db)
        return await model.find_first(
            where={"workspaceId": workspace_id},
            order={"updatedAt": "desc"},
        )

    def _resolve_preview_base_url(self, session: UserSpaceRuntimeSession | None) -> str:
        value = (session.preview_internal_url or "").strip() if session else ""
        if value.startswith("http://") or value.startswith("https://"):
            return value.rstrip("/")
        return _RUNTIME_PREVIEW_DEFAULT_BASE

    @staticmethod
    def _cache_entry_is_fresh(cached_at: datetime, max_age_seconds: float) -> bool:
        return (datetime.now(timezone.utc) - cached_at).total_seconds() <= max(
            0.0, max_age_seconds
        )

    async def _cache_preview_upstream_session(
        self,
        session: UserSpaceRuntimeSession,
    ) -> None:
        now = self._utc_now()
        async with self._runtime_cache_lock:
            self._preview_upstream_cache[session.workspace_id] = (
                _PreviewUpstreamCacheEntry(
                    provider_session_id=session.provider_session_id,
                    base_url=self._resolve_preview_base_url(session),
                    cached_at=now,
                    expires_at=now
                    + timedelta(seconds=_RUNTIME_PREVIEW_UPSTREAM_CACHE_TTL_SECONDS),
                )
            )

    async def _get_cached_preview_upstream_base_url(
        self,
        workspace_id: str,
    ) -> str | None:
        async with self._runtime_cache_lock:
            entry = self._preview_upstream_cache.get(workspace_id)
            if entry is None:
                return None
            if entry.expires_at <= self._utc_now():
                self._preview_upstream_cache.pop(workspace_id, None)
                return None
            return entry.base_url

    async def _cache_provider_status(
        self,
        provider_session_id: str,
        payload: dict[str, Any],
    ) -> None:
        if not provider_session_id:
            return
        async with self._runtime_cache_lock:
            self._provider_status_cache[provider_session_id] = (
                _ProviderStatusCacheEntry(
                    payload=dict(payload),
                    cached_at=self._utc_now(),
                )
            )

    async def _get_cached_provider_status(
        self,
        provider_session_id: str,
        *,
        max_age_seconds: float,
    ) -> dict[str, Any] | None:
        async with self._runtime_cache_lock:
            entry = self._provider_status_cache.get(provider_session_id)
            if entry is None:
                return None
            if not self._cache_entry_is_fresh(entry.cached_at, max_age_seconds):
                return None
            return dict(entry.payload)

    async def _drop_provider_status_cache(
        self,
        provider_session_id: str | None,
    ) -> None:
        if not provider_session_id:
            return
        async with self._runtime_cache_lock:
            self._provider_status_cache.pop(provider_session_id, None)

    async def invalidate_preview_session_cache(self, workspace_id: str) -> None:
        async with self._runtime_cache_lock:
            preview_entry = self._preview_upstream_cache.pop(workspace_id, None)
            self._preview_probe_cache.pop(workspace_id, None)
            if preview_entry and preview_entry.provider_session_id:
                self._provider_status_cache.pop(
                    preview_entry.provider_session_id,
                    None,
                )

    async def _invalidate_workspace_runtime_caches(
        self,
        workspace_id: str,
        *,
        invalidate_preview_host: bool = False,
    ) -> None:
        await self.invalidate_preview_session_cache(workspace_id)
        if invalidate_preview_host:
            from ragtime.userspace.preview_host import (
                invalidate_preview_sessions_for_workspace,
            )

            await invalidate_preview_sessions_for_workspace(workspace_id)

    async def _get_cached_probe_result(
        self,
        cache: dict[str, _PreviewProbeCacheEntry],
        key: str,
        *,
        max_age_seconds: float,
    ) -> bool | None:
        async with self._runtime_cache_lock:
            entry = cache.get(key)
            if entry is None:
                return None
            if not self._cache_entry_is_fresh(
                entry.checked_at,
                max_age_seconds,
            ):
                return None
            return entry.ok

    async def _cache_probe_result(
        self,
        cache: dict[str, _PreviewProbeCacheEntry],
        key: str,
        ok: bool,
    ) -> None:
        async with self._runtime_cache_lock:
            cache[key] = _PreviewProbeCacheEntry(ok=ok, checked_at=self._utc_now())

    async def _run_cached_probe(
        self,
        *,
        cache: dict[str, _PreviewProbeCacheEntry],
        cache_key: str,
        ttl_seconds: float,
        inflight_key: str,
        probe: Callable[[], Any],
    ) -> bool:
        cached = await self._get_cached_probe_result(
            cache,
            cache_key,
            max_age_seconds=ttl_seconds,
        )
        if cached is not None:
            return cached

        async with self._runtime_cache_lock:
            task = self._probe_tasks.get(inflight_key)
            if task is None:
                task = asyncio.create_task(probe())
                self._probe_tasks[inflight_key] = task

        try:
            ok = bool(await task)
        except Exception:
            ok = False
        finally:
            async with self._runtime_cache_lock:
                if self._probe_tasks.get(inflight_key) is task:
                    self._probe_tasks.pop(inflight_key, None)

        await self._cache_probe_result(cache, cache_key, ok)
        return ok

    async def _probe_preview_base_url_cached(
        self,
        workspace_id: str,
        base_url: str,
    ) -> bool:
        return await self._run_cached_probe(
            cache=self._preview_probe_cache,
            cache_key=workspace_id,
            ttl_seconds=_RUNTIME_PREVIEW_PROBE_CACHE_TTL_SECONDS,
            inflight_key=f"workspace-preview:{workspace_id}",
            probe=lambda: self._probe_preview_base_url(base_url),
        )

    async def _resolve_preview_host_cached(self, preview_host: str, port: int) -> bool:
        cache_key = f"{preview_host}:{port}"
        return await self._run_cached_probe(
            cache=self._public_preview_dns_cache,
            cache_key=cache_key,
            ttl_seconds=_RUNTIME_PUBLIC_PREVIEW_DNS_CACHE_TTL_SECONDS,
            inflight_key=f"dns:{cache_key}",
            probe=lambda: self._resolve_preview_host(preview_host, port),
        )

    async def _resolve_preview_host(self, preview_host: str, port: int) -> bool:
        try:
            await asyncio.to_thread(
                socket.getaddrinfo,
                preview_host,
                port,
                type=socket.SOCK_STREAM,
            )
        except OSError:
            return False
        return True

    async def _probe_public_preview_origin_cached(self, preview_origin: str) -> bool:
        probe_url = build_preview_probe_url(preview_origin)
        return await self._run_cached_probe(
            cache=self._public_preview_probe_cache,
            cache_key=probe_url,
            ttl_seconds=_RUNTIME_PUBLIC_PREVIEW_PROBE_CACHE_TTL_SECONDS,
            inflight_key=f"public-preview:{probe_url}",
            probe=lambda: self._probe_public_preview_origin(probe_url),
        )

    async def _probe_public_preview_origin(self, probe_url: str) -> bool:
        timeout = httpx.Timeout(connect=1.0, read=1.0, write=1.0, pool=0.5)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
            ) as client:
                response = await client.get(probe_url)
        except Exception:
            return False
        return is_preview_probe_response(response.status_code, response.headers)

    async def _probe_preview_base_url(self, base_url: str) -> bool:
        health_candidates = ["/", "/@vite/client"]
        # Reduced timeouts to prevent UI locking during workspace switch
        timeout = httpx.Timeout(connect=0.4, read=0.5, write=0.5, pool=0.4)

        async def check_candidate(client: httpx.AsyncClient, candidate: str) -> bool:
            try:
                response = await client.get(f"{base_url}{candidate}")
                if response.status_code < 500:
                    return True
            except Exception:
                pass
            return False

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            # Check all health candidates in parallel using asyncio.gather
            results = await asyncio.gather(
                *(check_candidate(client, c) for c in health_candidates)
            )
            return any(results)

    def _build_capability_token_response(
        self,
        workspace_id: str,
        user_id: str,
        capabilities: list[str],
        *,
        session_id: str | None = None,
        ttl_seconds: int = _RUNTIME_CAPABILITY_TTL_SECONDS,
    ) -> UserSpaceCapabilityTokenResponse:
        now = self._utc_now()
        expires_at = now + timedelta(seconds=max(60, min(ttl_seconds, 3600)))
        claims = {
            "sub": user_id,
            "workspace_id": workspace_id,
            "session_id": session_id,
            "capabilities": sorted({c for c in capabilities if c}),
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "jti": str(uuid4()),
        }
        token = jwt.encode(
            claims, settings.encryption_key, algorithm=settings.jwt_algorithm
        )
        return UserSpaceCapabilityTokenResponse(
            token=token,
            expires_at=expires_at,
            workspace_id=workspace_id,
            session_id=session_id,
            capabilities=cast(list[str], claims["capabilities"]),
        )

    def _runtime_provider_name(self) -> str:
        return _RUNTIME_PROVIDER_MANAGER

    def _runtime_manager_request_config(self) -> _RuntimeManagerRequestConfig:
        base_url = str(
            getattr(
                settings,
                "userspace_runtime_manager_url",
                "http://runtime:8090",
            )
        ).strip()
        manager_auth_token = str(
            getattr(settings, "userspace_runtime_manager_auth_token", "")
        ).strip()
        headers: dict[str, str] = {}
        if manager_auth_token:
            headers["Authorization"] = f"Bearer {manager_auth_token}"

        timeout_seconds = float(
            getattr(settings, "userspace_runtime_manager_timeout_seconds", 120.0)
        )
        retry_attempts = max(
            1,
            int(
                getattr(
                    settings,
                    "userspace_runtime_manager_retry_attempts",
                    3,
                )
            ),
        )
        retry_base_delay_seconds = float(
            getattr(settings, "userspace_runtime_manager_retry_delay_seconds", 0.2)
        )

        return _RuntimeManagerRequestConfig(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
            retry_base_delay_seconds=retry_base_delay_seconds,
        )

    def _runtime_manager_enabled(self) -> bool:
        manager_url = self._runtime_manager_request_config().base_url
        return manager_url.startswith("http://") or manager_url.startswith("https://")

    def _require_runtime_manager(self) -> None:
        if self._runtime_manager_enabled():
            return
        raise HTTPException(
            status_code=503,
            detail="Runtime manager is required for userspace runtime offload",
        )

    async def _runtime_manager_request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self._runtime_manager_request_config()
        base_url = config.base_url
        url = f"{base_url}/{path.lstrip('/')}"
        timeout = httpx.Timeout(config.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response: httpx.Response | None = None
            for attempt in range(1, config.retry_attempts + 1):
                try:
                    response = await client.request(
                        method,
                        url,
                        json=json_payload,
                        headers=config.headers,
                    )
                except Exception as exc:
                    if attempt < config.retry_attempts:
                        await asyncio.sleep(config.retry_base_delay_seconds * attempt)
                        continue
                    exc_type = exc.__class__.__name__
                    exc_message = str(exc).strip()
                    detail = f"Runtime manager unavailable ({exc_type})"
                    if exc_message:
                        detail = f"{detail}: {exc_message}"
                    raise HTTPException(
                        status_code=502,
                        detail=detail,
                    ) from exc

                if response.status_code >= 500 and attempt < config.retry_attempts:
                    await asyncio.sleep(config.retry_base_delay_seconds * attempt)
                    continue
                break

        if response is None:
            raise HTTPException(
                status_code=502,
                detail="Runtime manager unavailable (no response)",
            )

        if response.status_code >= 400:
            body_preview = response.text[:256]
            raise HTTPException(
                status_code=502,
                detail=(
                    "Runtime manager request failed "
                    f"({response.status_code}): {body_preview}"
                ),
            )

        if not response.content:
            return {}
        try:
            data = response.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    async def _runtime_provider_start_session(
        self,
        workspace_id: str,
        leased_by_user_id: str,
        existing_provider_session_id: str | None = None,
    ) -> dict[str, Any]:
        self._require_runtime_manager()

        workspace_env, workspace_env_visibility, workspace_mounts = (
            await asyncio.gather(
                userspace_service.get_workspace_runtime_environment(workspace_id),
                userspace_service.get_workspace_runtime_environment_visibility(
                    workspace_id
                ),
                userspace_service.resolve_workspace_mounts_for_runtime(workspace_id),
            )
        )
        payload: dict[str, Any] = {
            "workspace_id": workspace_id,
            "leased_by_user_id": leased_by_user_id,
            "workspace_env": workspace_env,
            "workspace_env_visibility": workspace_env_visibility,
            "workspace_mounts": workspace_mounts,
        }
        if existing_provider_session_id:
            payload["provider_session_id"] = existing_provider_session_id

        return await self._runtime_manager_request(
            "POST",
            "/sessions/start",
            json_payload=payload,
        )

    async def _runtime_provider_stop_session(
        self,
        provider_session_id: str | None,
    ) -> None:
        if not provider_session_id:
            return
        self._require_runtime_manager()
        await self._runtime_manager_request(
            "POST",
            f"/sessions/{provider_session_id}/stop",
        )

    async def _runtime_provider_get_status(
        self,
        provider_session_id: str | None,
        *,
        max_age_seconds: float | None = None,
        allow_stale_on_error: bool = False,
    ) -> dict[str, Any] | None:
        if not provider_session_id:
            return None
        if max_age_seconds is not None:
            cached = await self._get_cached_provider_status(
                provider_session_id,
                max_age_seconds=max_age_seconds,
            )
            if cached is not None:
                return cached
        self._require_runtime_manager()
        try:
            response = await self._runtime_manager_request(
                "GET",
                f"/sessions/{provider_session_id}",
            )
            await self._cache_provider_status(provider_session_id, response)
            return response
        except HTTPException as exc:
            if "(404)" in str(exc.detail):
                await self._drop_provider_status_cache(provider_session_id)
                return None
            if allow_stale_on_error:
                cached = await self._get_cached_provider_status(
                    provider_session_id,
                    max_age_seconds=_RUNTIME_PROVIDER_STATUS_STALE_FALLBACK_SECONDS,
                )
                if cached is not None:
                    return cached
            raise

    async def _runtime_provider_restart_devserver(
        self,
        provider_session_id: str | None,
        workspace_env: dict[str, str] | None = None,
        workspace_env_visibility: dict[str, bool] | None = None,
        workspace_mounts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if not provider_session_id:
            return None
        self._require_runtime_manager()
        json_payload: dict[str, Any] | None = None
        if (
            workspace_env is not None
            or workspace_env_visibility is not None
            or workspace_mounts is not None
        ):
            json_payload = {}
            if workspace_env is not None:
                json_payload["workspace_env"] = workspace_env
            if workspace_env_visibility is not None:
                json_payload["workspace_env_visibility"] = workspace_env_visibility
            if workspace_mounts is not None:
                json_payload["workspace_mounts"] = workspace_mounts
        return await self._runtime_manager_request(
            "POST",
            f"/sessions/{provider_session_id}/restart",
            json_payload=json_payload,
        )

    async def _runtime_provider_refresh_mounts(
        self,
        provider_session_id: str | None,
        workspace_mounts: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not provider_session_id:
            return None
        self._require_runtime_manager()
        return await self._runtime_manager_request(
            "POST",
            f"/sessions/{provider_session_id}/mounts/refresh",
            json_payload={"workspace_mounts": workspace_mounts},
        )

    async def _runtime_provider_get_pty_ws_url(
        self,
        provider_session_id: str | None,
    ) -> str:
        if not provider_session_id:
            raise HTTPException(status_code=404, detail="Runtime session unavailable")
        self._require_runtime_manager()
        response = await self._runtime_manager_request(
            "GET",
            f"/sessions/{provider_session_id}/pty/ws-url",
        )
        ws_url = str(response.get("ws_url") or "").strip()
        if not ws_url.startswith("ws://") and not ws_url.startswith("wss://"):
            raise HTTPException(
                status_code=502, detail="Runtime PTY upstream unavailable"
            )
        return ws_url

    async def _runtime_provider_read_file(
        self,
        provider_session_id: str | None,
        file_path: str,
    ) -> dict[str, Any]:
        if not provider_session_id:
            raise HTTPException(status_code=404, detail="Runtime session unavailable")
        self._require_runtime_manager()
        return await self._runtime_manager_request(
            "GET",
            f"/sessions/{provider_session_id}/fs/{quote(file_path, safe='/@._-~')}",
        )

    async def _runtime_provider_write_file(
        self,
        provider_session_id: str | None,
        file_path: str,
        content: str,
    ) -> dict[str, Any]:
        if not provider_session_id:
            raise HTTPException(status_code=404, detail="Runtime session unavailable")
        self._require_runtime_manager()
        return await self._runtime_manager_request(
            "PUT",
            f"/sessions/{provider_session_id}/fs/{quote(file_path, safe='/@._-~')}",
            json_payload={"content": content},
        )

    async def _runtime_provider_delete_file(
        self,
        provider_session_id: str | None,
        file_path: str,
    ) -> dict[str, Any]:
        if not provider_session_id:
            raise HTTPException(status_code=404, detail="Runtime session unavailable")
        self._require_runtime_manager()
        return await self._runtime_manager_request(
            "DELETE",
            f"/sessions/{provider_session_id}/fs/{quote(file_path, safe='/@._-~')}",
        )

    async def _runtime_provider_capture_screenshot(
        self,
        provider_session_id: str | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not provider_session_id:
            raise HTTPException(status_code=404, detail="Runtime session unavailable")
        self._require_runtime_manager()
        return await self._runtime_manager_request(
            "POST",
            f"/sessions/{provider_session_id}/screenshot",
            json_payload=payload,
        )

    async def _runtime_provider_content_probe(
        self,
        provider_session_id: str | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not provider_session_id:
            raise HTTPException(status_code=404, detail="Runtime session unavailable")
        self._require_runtime_manager()
        return await self._runtime_manager_request(
            "POST",
            f"/sessions/{provider_session_id}/content-probe",
            json_payload=payload,
        )

    async def _runtime_provider_exec_command(
        self,
        provider_session_id: str | None,
        command: str,
        timeout_seconds: int = 30,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        if not provider_session_id:
            raise HTTPException(status_code=404, detail="Runtime session unavailable")
        self._require_runtime_manager()
        payload: dict[str, Any] = {
            "command": command,
            "timeout_seconds": timeout_seconds,
        }
        if cwd:
            payload["cwd"] = cwd
        return await self._runtime_manager_request(
            "POST",
            f"/sessions/{provider_session_id}/exec",
            json_payload=payload,
        )

    # -- Workspace-level runtime manager requests (not session-scoped) --

    async def _runtime_workspace_file_list(
        self,
        workspace_id: str,
        *,
        include_dirs: bool = False,
        workspace_mounts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._require_runtime_manager()
        payload: dict[str, Any] = {"include_dirs": include_dirs}
        if workspace_mounts is not None:
            payload["workspace_mounts"] = workspace_mounts
        return await self._runtime_manager_request(
            "POST",
            f"/workspaces/{workspace_id}/files",
            json_payload=payload,
        )

    async def _runtime_workspace_git_command(
        self,
        workspace_id: str,
        *,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self._require_runtime_manager()
        payload: dict[str, Any] = {"args": args}
        if env is not None:
            payload["env"] = env
        return await self._runtime_manager_request(
            "POST",
            f"/workspaces/{workspace_id}/git",
            json_payload=payload,
        )

    async def _runtime_workspace_scm_status(
        self,
        workspace_id: str,
    ) -> dict[str, Any]:
        self._require_runtime_manager()
        return await self._runtime_manager_request(
            "GET",
            f"/workspaces/{workspace_id}/scm-status",
        )

    async def _ensure_session_row(
        self,
        workspace_id: str,
        leased_by_user_id: str,
        *,
        auto_start: bool = False,
    ) -> UserSpaceRuntimeSession:
        db = await get_db()
        model = self._runtime_session_model(db)
        current = await self._get_active_session_row(workspace_id)
        if current:
            session = self._to_runtime_session(current)
            target_provider_name = self._runtime_provider_name()
            provider_status: dict[str, Any] | None = None
            if (
                session.state == "running"
                and session.runtime_provider == target_provider_name
            ):
                provider_status = await self._runtime_provider_get_status(
                    session.provider_session_id,
                    max_age_seconds=_RUNTIME_PROVIDER_STATUS_CACHE_TTL_SECONDS,
                    allow_stale_on_error=True,
                )
                if provider_status is not None:
                    delta = self._merge_provider_status(session, provider_status)
                    if delta:
                        delta["lastHeartbeatAt"] = self._utc_now()
                        current = await self._runtime_session_update_row(
                            model,
                            session.id,
                            delta,
                        )
                        refreshed = self._to_runtime_session(current)
                        await self._cache_preview_upstream_session(refreshed)
                        return refreshed
                    await self._cache_preview_upstream_session(session)
                    return session

                logger.warning(
                    "Runtime provider session missing for workspace %s; recreating transparently",
                    workspace_id,
                )

            provider_data = await self._runtime_provider_start_session(
                workspace_id,
                session.leased_by_user_id or leased_by_user_id,
                existing_provider_session_id=(
                    session.provider_session_id
                    if session.runtime_provider == target_provider_name
                    else None
                ),
            )
            delta = self._merge_provider_status(session, provider_data)
            delta.update(
                {
                    "state": delta.get(
                        "state", str(provider_data.get("state") or "running")
                    ),
                    "runtimeProvider": self._runtime_provider_name(),
                    "providerSessionId": str(
                        provider_data.get("provider_session_id")
                        or session.provider_session_id
                        or ""
                    ),
                    "lastHeartbeatAt": self._utc_now(),
                }
            )
            current = await self._runtime_session_update_row(
                model,
                session.id,
                delta,
            )
            refreshed = self._to_runtime_session(current)
            await self._cache_provider_status(
                refreshed.provider_session_id or "",
                provider_data,
            )
            await self._cache_preview_upstream_session(refreshed)
            return refreshed

        if not auto_start:
            latest = await self._get_latest_session_row(workspace_id)
            if latest:
                latest_session = self._to_runtime_session(latest)
                if latest_session.state in {"stopped", "stopping", "error"}:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Runtime session is stopped. Start it from the workspace runtime controls."
                        ),
                    )
                raise HTTPException(
                    status_code=503,
                    detail="Runtime session is unavailable",
                )
            logger.info(
                "Runtime session missing for workspace %s; creating new session",
                workspace_id,
            )

        now = self._utc_now()
        provider_data = await self._runtime_provider_start_session(
            workspace_id,
            leased_by_user_id,
        )
        try:
            row = await self._runtime_session_create_row(
                model,
                {
                    "id": str(uuid4()),
                    "workspaceId": workspace_id,
                    "leasedByUserId": leased_by_user_id,
                    "state": str(provider_data.get("state") or "running"),
                    "runtimeProvider": self._runtime_provider_name(),
                    "providerSessionId": str(
                        provider_data.get("provider_session_id")
                        or f"runtime-{workspace_id[:8]}-{uuid4().hex[:8]}"
                    ),
                    "previewInternalUrl": str(
                        provider_data.get("preview_internal_url")
                        or _RUNTIME_PREVIEW_DEFAULT_BASE
                    ),
                    "launchFramework": str(provider_data.get("launch_framework") or "")
                    or None,
                    "launchCommand": str(provider_data.get("launch_command") or "")
                    or None,
                    "launchCwd": str(provider_data.get("launch_cwd") or "") or None,
                    "launchPort": self._optional_int(provider_data.get("launch_port")),
                    "createdAt": now,
                    "updatedAt": now,
                    "lastHeartbeatAt": now,
                    "idleExpiresAt": now + timedelta(hours=4),
                    "ttlExpiresAt": now + timedelta(hours=24),
                    "lastError": provider_data.get("last_error"),
                },
            )
        except ForeignKeyViolationError as exc:
            logger.info(
                "Runtime session create race: workspace %s no longer exists",
                workspace_id,
            )
            raise HTTPException(status_code=404, detail="Workspace not found") from exc
        session = self._to_runtime_session(row)
        await self._cache_provider_status(
            session.provider_session_id or "", provider_data
        )
        await self._cache_preview_upstream_session(session)
        return session

    async def ensure_workspace_preview_session(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceRuntimeSession:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer")
        session = await self._ensure_session_row(workspace_id, user_id)
        await self._cache_preview_upstream_session(session)
        return session

    async def ensure_shared_preview_session(
        self,
        workspace_id: str,
    ) -> UserSpaceRuntimeSession:
        db = await get_db()
        workspace = await db.workspace.find_unique(where={"id": workspace_id})
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        owner_user_id = str(getattr(workspace, "ownerUserId", "") or "")
        if not owner_user_id:
            raise HTTPException(status_code=500, detail="Workspace owner unavailable")
        session = await self._ensure_session_row(workspace_id, owner_user_id)
        await self._cache_preview_upstream_session(session)
        return session

    async def list_workspace_files_internal(
        self,
        workspace_id: str,
        *,
        include_dirs: bool = False,
    ) -> list[UserSpaceFileInfo]:
        mount_specs = await userspace_service.resolve_workspace_mounts_for_runtime(
            workspace_id
        )
        payload = await self._runtime_workspace_file_list(
            workspace_id,
            include_dirs=include_dirs,
            workspace_mounts=mount_specs,
        )
        files = payload.get("files") or []
        result: list[UserSpaceFileInfo] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            result.append(
                UserSpaceFileInfo(
                    path=str(item.get("path", "") or ""),
                    size_bytes=int(item.get("size_bytes", 0) or 0),
                    updated_at=self._parse_datetime(item.get("updated_at")),
                    entry_type=cast(Any, str(item.get("entry_type", "file") or "file")),
                )
            )
        return result

    async def run_workspace_git_command_internal(
        self,
        workspace_id: str,
        *,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> tuple[int, bytes, bytes]:
        payload = await self._runtime_workspace_git_command(
            workspace_id,
            args=args,
            env=env,
        )
        try:
            raw_rc = payload.get("returncode")
            returncode = int(raw_rc) if raw_rc is not None else 1
        except Exception:
            returncode = 1
        stdout_bytes = base64.b64decode(str(payload.get("stdout_b64", "") or ""))
        stderr_bytes = base64.b64decode(str(payload.get("stderr_b64", "") or ""))
        return returncode, stdout_bytes, stderr_bytes

    async def get_workspace_scm_status_internal(
        self,
        workspace_id: str,
    ) -> dict[str, Any]:
        return await self._runtime_workspace_scm_status(workspace_id)

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        text = str(value or "").strip()
        if not text:
            return datetime.now(timezone.utc)
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    async def issue_workspace_preview_launch(
        self,
        workspace_id: str,
        user_id: str,
        *,
        control_plane_origin: str,
        path: str = "/",
        parent_origin: str | None = None,
    ) -> UserSpacePreviewLaunchResponse:
        await self.ensure_workspace_preview_session(workspace_id, user_id)
        return await self._build_workspace_preview_launch_response(
            workspace_id=workspace_id,
            subject_user_id=user_id,
            control_plane_origin=control_plane_origin,
            path=self._sanitize_preview_path(path),
            parent_origin=parent_origin,
            mode="workspace",
        )

    async def issue_shared_preview_launch(
        self,
        workspace_id: str,
        *,
        control_plane_origin: str,
        path: str = "/",
        parent_origin: str | None = None,
        share_token: str | None = None,
        owner_username: str | None = None,
        share_slug: str | None = None,
        subject_user_id: str | None = None,
        share_access_mode: str | None = None,
    ) -> UserSpacePreviewLaunchResponse:
        await self.ensure_shared_preview_session(workspace_id)
        mode = "share_token" if share_token else "share_slug"
        return await self._build_workspace_preview_launch_response(
            workspace_id=workspace_id,
            subject_user_id=subject_user_id,
            control_plane_origin=control_plane_origin,
            path=self._sanitize_preview_path(path),
            parent_origin=parent_origin,
            mode=mode,
            share_token=share_token,
            owner_username=owner_username,
            share_slug=share_slug,
            share_access_mode=share_access_mode,
        )

    async def build_workspace_preview_upstream_url(
        self,
        workspace_id: str,
        user_id: str,
        path: str,
        query: str | None = None,
    ) -> str:
        base_url = await self._get_cached_preview_upstream_base_url(workspace_id)
        if not base_url:
            session = await self.ensure_workspace_preview_session(workspace_id, user_id)
            base_url = self._resolve_preview_base_url(session)
        normalized_path = quote((path or "").lstrip("/"), safe="/@._-~")
        upstream = (
            f"{base_url}/{normalized_path}" if normalized_path else f"{base_url}/"
        )
        if query:
            upstream = f"{upstream}?{query}"
        return upstream

    async def build_workspace_pty_upstream_ws_url(
        self,
        workspace_id: str,
        user_id: str,
    ) -> str:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer")
        session = await self.ensure_workspace_preview_session(workspace_id, user_id)
        return await self._runtime_provider_get_pty_ws_url(session.provider_session_id)

    async def build_shared_preview_upstream_url(
        self,
        workspace_id: str,
        path: str,
        query: str | None = None,
    ) -> str:
        base_url = await self._get_cached_preview_upstream_base_url(workspace_id)
        if not base_url:
            session = await self.ensure_shared_preview_session(workspace_id)
            base_url = self._resolve_preview_base_url(session)
        normalized_path = quote((path or "").lstrip("/"), safe="/@._-~")
        upstream = (
            f"{base_url}/{normalized_path}" if normalized_path else f"{base_url}/"
        )
        if query:
            upstream = f"{upstream}?{query}"
        return upstream

    async def _audit(
        self,
        workspace_id: str,
        event_type: str,
        *,
        user_id: str | None,
        session_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        db = await get_db()
        model = self._runtime_audit_model(db)
        normalized_payload = json.loads(json.dumps(payload or {}, default=str))
        payload_json = prisma_fields.Json(normalized_payload)

        def _build_shapes() -> list[dict[str, Any]]:
            shapes: list[dict[str, Any]] = [
                {
                    "workspaceId": workspace_id,
                    "userId": user_id,
                    "sessionId": session_id,
                    "eventType": event_type,
                    "eventPayload": payload_json,
                },
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "event_type": event_type,
                    "event_payload": payload_json,
                },
            ]
            if user_id:
                shapes.append(
                    {
                        "workspace": {"connect": {"id": workspace_id}},
                        "user": {"connect": {"id": user_id}},
                        "sessionId": session_id,
                        "eventType": event_type,
                        "eventPayload": payload_json,
                    }
                )
            return shapes

        try:
            shapes = _build_shapes()
            # Fast path: use cached shape index if we already know which works.
            if self._audit_shape_index is not None:
                idx = self._audit_shape_index
                if idx < len(shapes):
                    try:
                        await model.create(data=shapes[idx])
                        return
                    except Exception:
                        # Cached shape no longer valid; fall through to probe.
                        self._audit_shape_index = None

            for idx, data in enumerate(shapes):
                try:
                    await model.create(data=data)
                    self._audit_shape_index = idx
                    return
                except Exception:
                    continue
            raise RuntimeError("all runtime audit insert shapes failed")
        except Exception as exc:
            logger.debug("Runtime audit insert failed: %s", exc)

    async def get_runtime_session(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceRuntimeSessionResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer")
        active = await self._get_active_session_row(workspace_id)
        return UserSpaceRuntimeSessionResponse(
            workspace_id=workspace_id,
            session=self._to_runtime_session(active) if active else None,
        )

    async def start_runtime_session(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceRuntimeActionResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "editor")

        session = await self._ensure_session_row(
            workspace_id,
            user_id,
            auto_start=True,
        )
        await self._invalidate_workspace_runtime_caches(
            workspace_id,
            invalidate_preview_host=True,
        )
        await self._cache_preview_upstream_session(session)
        await self._audit(
            workspace_id,
            "session_start",
            user_id=user_id,
            session_id=session.id,
            payload={"provider_session_id": session.provider_session_id},
        )

        provider_status = await self._runtime_provider_get_status(
            session.provider_session_id,
            max_age_seconds=_RUNTIME_PROVIDER_STATUS_CACHE_TTL_SECONDS,
            allow_stale_on_error=True,
        )
        operation_id = None
        operation_phase = None
        operation_started_at = None
        operation_updated_at = None
        if provider_status:
            operation_id = provider_status.get("runtime_operation_id")
            operation_phase = provider_status.get("runtime_operation_phase")
            operation_started_at = provider_status.get("runtime_operation_started_at")
            operation_updated_at = provider_status.get("runtime_operation_updated_at")

        return UserSpaceRuntimeActionResponse(
            workspace_id=workspace_id,
            session_id=session.id,
            state=session.state,
            success=True,
            runtime_operation_id=operation_id,
            runtime_operation_phase=operation_phase,
            runtime_operation_started_at=operation_started_at,
            runtime_operation_updated_at=operation_updated_at,
        )

    async def stop_runtime_session(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceRuntimeActionResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "editor")

        active = await self._get_active_session_row(workspace_id)
        if not active:
            raise HTTPException(status_code=404, detail="No active runtime session")

        active_session = self._to_runtime_session(active)
        provider_stop_error: str | None = None
        try:
            await self._runtime_provider_stop_session(
                active_session.provider_session_id
            )
        except HTTPException as exc:
            provider_stop_error = str(exc.detail)
            logger.warning(
                "Runtime provider stop failed for workspace %s: %s",
                workspace_id,
                provider_stop_error,
            )

        db = await get_db()
        model = self._runtime_session_model(db)
        row = await self._runtime_session_update_row(
            model,
            active_session.id,
            {
                "state": "stopped",
                "lastHeartbeatAt": self._utc_now(),
                "lastError": provider_stop_error,
            },
        )
        session = self._to_runtime_session(row)
        await self._invalidate_workspace_runtime_caches(
            workspace_id,
            invalidate_preview_host=True,
        )
        await self._audit(
            workspace_id,
            "session_stop",
            user_id=user_id,
            session_id=session.id,
        )
        return UserSpaceRuntimeActionResponse(
            workspace_id=workspace_id,
            session_id=session.id,
            state=session.state,
            success=True,
        )

    async def get_devserver_status(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceRuntimeStatusResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer")
        active = await self._get_active_session_row(workspace_id)
        if not active:
            return UserSpaceRuntimeStatusResponse(
                workspace_id=workspace_id,
                session_state="stopped",
                session_id=None,
                devserver_running=False,
                devserver_port=_RUNTIME_DEVSERVER_PORT,
                runtime_capabilities=None,
                runtime_has_cap_sys_admin=None,
                preview_url=self._build_preview_origin(workspace_id),
            )

        session = self._to_runtime_session(active)
        provider_status = await self._runtime_provider_get_status(
            session.provider_session_id,
            max_age_seconds=_RUNTIME_PROVIDER_STATUS_CACHE_TTL_SECONDS,
            allow_stale_on_error=True,
        )

        if (
            provider_status is None
            and session.runtime_provider == self._runtime_provider_name()
            and session.state in {"starting", "running"}
        ):
            # Re-read the DB row to guard against a concurrent stop_runtime_session
            # that set the state to "stopped" after our initial read.
            fresh_row = await self._get_active_session_row(workspace_id)
            if not fresh_row:
                return UserSpaceRuntimeStatusResponse(
                    workspace_id=workspace_id,
                    session_state="stopped",
                    session_id=session.id,
                    devserver_running=False,
                    devserver_port=_RUNTIME_DEVSERVER_PORT,
                    runtime_capabilities=None,
                    runtime_has_cap_sys_admin=None,
                    preview_url=self._build_preview_origin(workspace_id),
                )
            logger.warning(
                "Runtime provider status missing for workspace %s; auto-recovering session",
                workspace_id,
            )
            provider_status = await self._runtime_provider_start_session(
                workspace_id,
                session.leased_by_user_id or user_id,
                existing_provider_session_id=session.provider_session_id,
            )

        state_for_response = session.state
        launch_framework = session.launch_framework
        launch_command = session.launch_command
        launch_cwd = session.launch_cwd
        launch_port = session.launch_port
        last_error = session.last_error

        if provider_status:
            delta = self._merge_provider_status(session, provider_status)
            # Apply merged values for local use in this method
            state_for_response = cast(
                RuntimeSessionState,
                delta.get("state", session.state),
            )
            launch_framework = delta.get("launchFramework", session.launch_framework)
            launch_command = delta.get("launchCommand", session.launch_command)
            launch_cwd = delta.get("launchCwd", session.launch_cwd)
            launch_port = delta.get("launchPort", session.launch_port)
            last_error = delta.get("lastError", session.last_error)

            if delta:
                # Guard against a concurrent stop overwrite: only persist if the
                # session is still active in DB.
                fresh_check = await self._get_active_session_row(workspace_id)
                if fresh_check and getattr(fresh_check, "id", None) == session.id:
                    delta["lastHeartbeatAt"] = self._utc_now()
                    db = await get_db()
                    model = self._runtime_session_model(db)
                    updated = await self._runtime_session_update_row(
                        model,
                        session.id,
                        delta,
                    )
                    session = self._to_runtime_session(updated)

        base_url = self._resolve_preview_base_url(session)

        if provider_status and "devserver_running" in provider_status:
            devserver_running = bool(provider_status.get("devserver_running"))
        else:
            devserver_running = state_for_response in {
                "starting",
                "running",
            } and await self._probe_preview_base_url_cached(workspace_id, base_url)

        devserver_port = launch_port or _RUNTIME_DEVSERVER_PORT
        runtime_capabilities: dict[str, Any] | None = None
        runtime_has_cap_sys_admin: bool | None = None
        runtime_operation_id: str | None = None
        runtime_operation_phase: RuntimeOperationPhase | None = None
        runtime_operation_started_at: Any | None = None
        runtime_operation_updated_at: Any | None = None
        allowed_runtime_phases = {
            "queued",
            "provisioning",
            "bootstrapping",
            "deps_install",
            "launching",
            "probing",
            "ready",
            "failed",
            "stopped",
        }
        if provider_status:
            raw_runtime_capabilities = provider_status.get("runtime_capabilities")
            if isinstance(raw_runtime_capabilities, dict):
                runtime_capabilities = raw_runtime_capabilities
                cap_value = raw_runtime_capabilities.get("has_cap_sys_admin")
                if isinstance(cap_value, bool):
                    runtime_has_cap_sys_admin = cap_value
            runtime_operation_id = (
                str(provider_status.get("runtime_operation_id") or "").strip() or None
            )
            phase_value = (
                str(provider_status.get("runtime_operation_phase") or "").strip()
                or None
            )
            if phase_value in allowed_runtime_phases:
                runtime_operation_phase = cast(RuntimeOperationPhase, phase_value)
            else:
                runtime_operation_phase = None
            runtime_operation_started_at = provider_status.get(
                "runtime_operation_started_at"
            )
            runtime_operation_updated_at = provider_status.get(
                "runtime_operation_updated_at"
            )

        return UserSpaceRuntimeStatusResponse(
            workspace_id=workspace_id,
            session_state=state_for_response,
            session_id=session.id,
            devserver_running=devserver_running,
            devserver_port=devserver_port,
            launch_framework=launch_framework,
            launch_command=launch_command,
            launch_cwd=launch_cwd,
            runtime_capabilities=runtime_capabilities,
            runtime_has_cap_sys_admin=runtime_has_cap_sys_admin,
            preview_url=self._build_preview_origin(workspace_id),
            last_error=last_error,
            runtime_operation_id=runtime_operation_id,
            runtime_operation_phase=runtime_operation_phase,
            runtime_operation_started_at=runtime_operation_started_at,
            runtime_operation_updated_at=runtime_operation_updated_at,
        )

    async def get_workspace_tab_state(
        self,
        workspace_id: str,
        user_id: str,
        *,
        selected_conversation_id: str | None = None,
        is_admin: bool = False,
    ) -> UserSpaceWorkspaceTabStateResponse:
        await userspace_service.enforce_workspace_role(
            workspace_id,
            user_id,
            "viewer",
            is_admin=is_admin,
        )
        runtime_status, chat_state = await asyncio.gather(
            self.get_devserver_status(workspace_id, user_id),
            build_workspace_chat_state(
                workspace_id=workspace_id,
                user_id=user_id,
                is_admin=is_admin,
                selected_conversation_id=selected_conversation_id,
            ),
        )
        return UserSpaceWorkspaceTabStateResponse(
            workspace_id=workspace_id,
            runtime_status=runtime_status,
            chat_state=chat_state,
        )

    async def restart_devserver(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceRuntimeActionResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "editor")

        active = await self._get_active_session_row(workspace_id)
        if active:
            active_session = self._to_runtime_session(active)
            provider_stop_error: str | None = None
            try:
                await self._runtime_provider_stop_session(
                    active_session.provider_session_id
                )
                await self._drop_provider_status_cache(
                    active_session.provider_session_id
                )
            except HTTPException as exc:
                provider_stop_error = str(exc.detail)
                logger.warning(
                    "Runtime provider stop failed during devserver restart for workspace %s: %s",
                    workspace_id,
                    provider_stop_error,
                )

            db = await get_db()
            model = self._runtime_session_model(db)
            await self._runtime_session_update_row(
                model,
                active_session.id,
                {
                    "state": "stopped",
                    "lastHeartbeatAt": self._utc_now(),
                    "lastError": provider_stop_error,
                },
            )
            await self._invalidate_workspace_runtime_caches(
                workspace_id,
                invalidate_preview_host=True,
            )

        start = await self.start_runtime_session(workspace_id, user_id)
        await self._audit(
            workspace_id,
            "devserver_restart",
            user_id=user_id,
            session_id=start.session_id,
        )
        return UserSpaceRuntimeActionResponse(
            workspace_id=start.workspace_id,
            session_id=start.session_id,
            state=start.state,
            success=True,
            runtime_operation_id=start.runtime_operation_id,
            runtime_operation_phase=start.runtime_operation_phase,
            runtime_operation_started_at=start.runtime_operation_started_at,
            runtime_operation_updated_at=start.runtime_operation_updated_at,
        )

    async def refresh_runtime_env_vars(self, workspace_id: str) -> None:
        """Hot-apply workspace environment variable changes to a running session.

        Reads the current env vars from DB, then restarts the devserver with
        the updated environment.  No-op if no active runtime session exists.
        """
        active = await self._get_active_session_row(workspace_id)
        if not active:
            return
        session = self._to_runtime_session(active)
        if session.state not in {"running", "starting"}:
            return
        session = await self._ensure_session_row(
            workspace_id,
            session.leased_by_user_id,
            auto_start=False,
        )
        workspace_env, workspace_env_visibility = await asyncio.gather(
            userspace_service.get_workspace_runtime_environment(workspace_id),
            userspace_service.get_workspace_runtime_environment_visibility(
                workspace_id
            ),
        )
        await self._runtime_provider_restart_devserver(
            session.provider_session_id,
            workspace_env=workspace_env,
            workspace_env_visibility=workspace_env_visibility,
        )

    async def refresh_runtime_env_vars_for_all_active_workspaces(self) -> None:
        """Best-effort env-var refresh for all active runtime sessions."""
        db = await get_db()
        model = self._runtime_session_model(db)
        rows = await model.find_many(
            where={"state": {"in": ["starting", "running"]}},
            order={"updatedAt": "desc"},
        )
        workspace_ids: set[str] = set()
        for row in rows:
            workspace_id = str(getattr(row, "workspaceId", "") or "").strip()
            if workspace_id:
                workspace_ids.add(workspace_id)

        for workspace_id in workspace_ids:
            try:
                await self.refresh_runtime_env_vars(workspace_id)
            except Exception as exc:
                logger.warning(
                    "Failed to refresh runtime env vars for workspace %s: %s",
                    workspace_id,
                    exc,
                )

    async def list_active_runtime_workspace_targets(self) -> list[tuple[str, str]]:
        """Return active runtime workspace ids with names, sorted for stable batch order."""
        db = await get_db()
        model = self._runtime_session_model(db)
        rows = await model.find_many(
            where={"state": {"in": ["starting", "running"]}},
            order={"updatedAt": "asc"},
        )
        workspace_ids: list[str] = []
        seen_workspace_ids: set[str] = set()
        for row in rows:
            workspace_id = str(getattr(row, "workspaceId", "") or "").strip()
            if not workspace_id or workspace_id in seen_workspace_ids:
                continue
            seen_workspace_ids.add(workspace_id)
            workspace_ids.append(workspace_id)

        if not workspace_ids:
            return []

        workspace_rows = await db.workspace.find_many(
            where={"id": {"in": workspace_ids}},
            order={"name": "asc"},
        )
        names_by_id = {
            str(getattr(row, "id", "") or "")
            .strip(): str(getattr(row, "name", "") or "")
            .strip()
            for row in workspace_rows
        }
        return [
            (workspace_id, names_by_id.get(workspace_id) or workspace_id)
            for workspace_id in workspace_ids
        ]

    async def restart_runtime_env_vars_and_wait(
        self,
        workspace_id: str,
        *,
        timeout_seconds: float,
        progress_callback: Callable[[RuntimeOperationPhase | None], None] | None = None,
    ) -> None:
        active = await self._get_active_session_row(workspace_id)
        if active is None:
            raise HTTPException(
                status_code=404,
                detail="Runtime session is no longer active",
            )

        session = self._to_runtime_session(active)
        if (
            session.state not in {"running", "starting"}
            or not session.provider_session_id
        ):
            raise HTTPException(
                status_code=404,
                detail="Runtime session is no longer active",
            )

        provider_session_id = session.provider_session_id
        await self.refresh_runtime_env_vars(workspace_id)

        deadline = _time.monotonic() + timeout_seconds
        allowed_runtime_phases = {
            "queued",
            "provisioning",
            "bootstrapping",
            "deps_install",
            "launching",
            "probing",
            "ready",
            "failed",
            "stopped",
        }

        while True:
            provider_status = await self._runtime_provider_get_status(
                provider_session_id,
                max_age_seconds=0,
            )
            if provider_status is None:
                raise HTTPException(
                    status_code=404,
                    detail="Runtime session is no longer active",
                )

            phase_value = str(
                provider_status.get("runtime_operation_phase") or ""
            ).strip()
            runtime_operation_phase: RuntimeOperationPhase | None = None
            if phase_value in allowed_runtime_phases:
                runtime_operation_phase = cast(RuntimeOperationPhase, phase_value)

            if progress_callback is not None:
                progress_callback(runtime_operation_phase)

            state = str(provider_status.get("state") or "").strip() or "running"
            last_error = str(provider_status.get("last_error") or "").strip() or None
            devserver_running = bool(provider_status.get("devserver_running"))

            if state == "running" and (
                runtime_operation_phase in {None, "ready"} or devserver_running
            ):
                return

            if state in {"error", "stopped"} or runtime_operation_phase == "failed":
                raise HTTPException(
                    status_code=502,
                    detail=(
                        last_error
                        or f"Runtime entered {runtime_operation_phase or state} during restart"
                    ),
                )

            if _time.monotonic() >= deadline:
                raise HTTPException(
                    status_code=504,
                    detail=(
                        "Timed out waiting for runtime restart after "
                        f"{int(timeout_seconds)} seconds"
                    ),
                )

            await asyncio.sleep(1.0)

    async def _persist_runtime_mount_refresh_result(
        self,
        active_session: UserSpaceRuntimeSession,
        provider_status: dict[str, Any] | None,
    ) -> tuple[
        RuntimeSessionState,
        str | None,
        RuntimeOperationPhase | None,
        datetime | None,
        datetime | None,
    ]:
        if provider_status:
            delta = self._merge_provider_status(active_session, provider_status)
            delta.update(
                {
                    "state": delta.get(
                        "state",
                        str(provider_status.get("state") or active_session.state),
                    ),
                    "lastHeartbeatAt": self._utc_now(),
                    "lastError": None,
                }
            )
            db = await get_db()
            model = self._runtime_session_model(db)
            await self._runtime_session_update_row(
                model,
                active_session.id,
                delta,
            )
            return (
                cast(
                    RuntimeSessionState,
                    provider_status.get("state") or active_session.state,
                ),
                cast(str | None, provider_status.get("runtime_operation_id")),
                cast(
                    RuntimeOperationPhase | None,
                    provider_status.get("runtime_operation_phase"),
                ),
                cast(
                    datetime | None,
                    provider_status.get("runtime_operation_started_at"),
                ),
                cast(
                    datetime | None,
                    provider_status.get("runtime_operation_updated_at"),
                ),
            )

        return active_session.state, None, None, None, None

    async def refresh_workspace_mount_after_sync(
        self,
        workspace_id: str,
        mount_id: str,
    ) -> str | None:
        active = await self._get_active_session_row(workspace_id)
        if not active:
            return None

        active_session = self._to_runtime_session(active)
        if active_session.state not in {"running", "starting"}:
            return None

        resolved_mounts = await userspace_service.resolve_workspace_mounts_for_runtime(
            workspace_id,
            mount_ids=[mount_id],
        )
        if not resolved_mounts:
            logger.debug(
                "Skipping automatic runtime mount refresh for %s/%s because no runtime mount spec was resolved",
                workspace_id,
                mount_id,
            )
            return None

        try:
            provider_status = await self._runtime_provider_refresh_mounts(
                active_session.provider_session_id,
                resolved_mounts,
            )
            await self._persist_runtime_mount_refresh_result(
                active_session,
                provider_status,
            )
            await self.bump_workspace_generation(workspace_id, "mount_refresh")
            if provider_status is None:
                return "Sync completed, but the active runtime did not report a mount refresh result."
            return None
        except HTTPException as exc:
            detail = str(exc.detail).strip() or "runtime mount refresh failed"
        except Exception as exc:
            detail = str(exc).strip() or "runtime mount refresh failed"

        logger.warning(
            "Automatic runtime mount refresh failed for %s/%s: %s",
            workspace_id,
            mount_id,
            detail,
        )
        return (
            "Sync completed, but the active runtime was not refreshed automatically: "
            f"{detail[:240]}"
        )

    async def refresh_workspace_mount(
        self,
        workspace_id: str,
        user_id: str,
        mount_id: str,
    ) -> UserSpaceRuntimeActionResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "editor")

        active = await self._get_active_session_row(workspace_id)
        if not active:
            raise HTTPException(status_code=409, detail="No active runtime session")

        active_session = self._to_runtime_session(active)
        if active_session.state not in {"running", "starting"}:
            raise HTTPException(status_code=409, detail="Runtime session is not active")

        workspace_mounts = await userspace_service.list_workspace_mounts(
            workspace_id,
            user_id,
        )
        mount = next((item for item in workspace_mounts if item.id == mount_id), None)
        if not mount:
            raise HTTPException(status_code=404, detail="Mount not found")
        if mount.source_type != "ssh":
            raise HTTPException(
                status_code=400,
                detail="Runtime mount refresh is only supported for SFTP-backed mounts",
            )
        if mount.sync_status != "synced":
            raise HTTPException(
                status_code=409,
                detail="Sync the SFTP mount before refreshing the runtime sandbox",
            )

        resolved_mounts = await userspace_service.resolve_workspace_mounts_for_runtime(
            workspace_id,
            mount_ids=[mount_id],
        )
        if not resolved_mounts:
            raise HTTPException(
                status_code=400,
                detail="Mount is not available for runtime refresh",
            )

        provider_status = await self._runtime_provider_refresh_mounts(
            active_session.provider_session_id,
            resolved_mounts,
        )
        (
            state,
            operation_id,
            operation_phase,
            operation_started_at,
            operation_updated_at,
        ) = await self._persist_runtime_mount_refresh_result(
            active_session,
            provider_status,
        )

        await self._audit(
            workspace_id,
            "mount_refresh",
            user_id=user_id,
            session_id=active_session.id,
            payload={"mount_id": mount_id},
        )
        return UserSpaceRuntimeActionResponse(
            workspace_id=workspace_id,
            session_id=active_session.id,
            state=state,
            success=True,
            runtime_operation_id=operation_id,
            runtime_operation_phase=operation_phase,
            runtime_operation_started_at=operation_started_at,
            runtime_operation_updated_at=operation_updated_at,
        )

    async def issue_capability_token(
        self,
        workspace_id: str,
        user_id: str,
        capabilities: list[str],
        session_id: str | None = None,
        ttl_seconds: int = _RUNTIME_CAPABILITY_TTL_SECONDS,
    ) -> UserSpaceCapabilityTokenResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer")
        token_response = self._build_capability_token_response(
            workspace_id,
            user_id,
            capabilities,
            session_id=session_id,
            ttl_seconds=ttl_seconds,
        )
        await self._audit(
            workspace_id,
            "capability_token_issue",
            user_id=user_id,
            session_id=session_id,
            payload={"capabilities": token_response.capabilities},
        )
        return token_response

    def verify_capability_token(
        self,
        token: str,
        workspace_id: str,
        capability: str,
    ) -> dict[str, Any]:
        try:
            claims = jwt.decode(
                token,
                settings.encryption_key,
                algorithms=[settings.jwt_algorithm],
            )
        except JWTError as exc:
            detail = "Invalid workspace authorization token. Refresh the page to reauthorize."
            if exc.__class__.__name__ == "ExpiredSignatureError":
                detail = "Your workspace authorization has expired. Refresh the page to reauthorize."
            raise HTTPException(
                status_code=401,
                detail=detail,
            ) from exc

        if str(claims.get("workspace_id") or "") != workspace_id:
            raise HTTPException(
                status_code=403, detail="Capability token scope mismatch"
            )

        capabilities = set(claims.get("capabilities") or [])
        if capability not in capabilities:
            raise HTTPException(status_code=403, detail="Capability not granted")

        return claims

    async def _load_file_content(
        self,
        workspace_id: str,
        normalized_path: str,
        user_id: str,
    ) -> str:
        normalized_path = (
            await userspace_service.ensure_workspace_path_not_in_disabled_mount(
                workspace_id,
                normalized_path,
            )
        )
        session = await self.ensure_workspace_preview_session(workspace_id, user_id)
        payload = await self._runtime_provider_read_file(
            session.provider_session_id,
            normalized_path,
        )
        return str(payload.get("content", ""))

    async def _persist_file_content(
        self,
        workspace_id: str,
        normalized_path: str,
        content: str,
        user_id: str,
    ) -> None:
        normalized_path = (
            await userspace_service.ensure_workspace_path_not_in_disabled_mount(
                workspace_id,
                normalized_path,
            )
        )
        session = await self.ensure_workspace_preview_session(workspace_id, user_id)
        await self._runtime_provider_write_file(
            session.provider_session_id,
            normalized_path,
            content,
        )
        await userspace_service.clear_workspace_changed_file_acknowledgements_for_paths_for_all_users(
            workspace_id,
            [normalized_path],
        )
        await userspace_service.touch_workspace(workspace_id)

    async def runtime_fs_read(
        self,
        workspace_id: str,
        file_path: str,
        user_id: str,
    ) -> UserSpaceFileResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer")
        normalized_path = self._normalize_file_path(file_path)
        normalized_path = (
            await userspace_service.ensure_workspace_path_not_in_disabled_mount(
                workspace_id,
                normalized_path,
            )
        )
        session = await self.ensure_workspace_preview_session(workspace_id, user_id)
        payload = await self._runtime_provider_read_file(
            session.provider_session_id,
            normalized_path,
        )
        return UserSpaceFileResponse(
            path=normalized_path,
            content=str(payload.get("content", "")),
            artifact_type=None,
            live_data_connections=None,
            live_data_checks=None,
            updated_at=self._utc_now(),
        )

    async def runtime_fs_write(
        self,
        workspace_id: str,
        file_path: str,
        content: str,
        user_id: str,
    ) -> UserSpaceFileResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "editor")
        normalized_path = self._normalize_file_path(file_path)
        normalized_path = (
            await userspace_service.ensure_workspace_path_not_in_disabled_mount(
                workspace_id,
                normalized_path,
            )
        )
        await self._persist_file_content(
            workspace_id,
            normalized_path,
            content,
            user_id,
        )
        await self._audit(
            workspace_id,
            "runtime_fs_write",
            user_id=user_id,
            payload={"file_path": normalized_path},
        )
        return UserSpaceFileResponse(
            path=normalized_path,
            content=content,
            artifact_type=None,
            live_data_connections=None,
            live_data_checks=None,
            updated_at=self._utc_now(),
        )

    async def runtime_fs_delete(
        self,
        workspace_id: str,
        file_path: str,
        user_id: str,
    ) -> dict[str, Any]:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "editor")
        normalized_path = self._normalize_file_path(file_path)
        normalized_path = (
            await userspace_service.ensure_workspace_path_not_in_disabled_mount(
                workspace_id,
                normalized_path,
            )
        )
        session = await self.ensure_workspace_preview_session(workspace_id, user_id)
        await self._runtime_provider_delete_file(
            session.provider_session_id, normalized_path
        )
        await userspace_service.touch_workspace(workspace_id)
        await self._audit(
            workspace_id,
            "runtime_fs_delete",
            user_id=user_id,
            payload={"file_path": normalized_path},
        )
        return {"success": True}

    async def get_collab_snapshot(
        self,
        workspace_id: str,
        file_path: str,
        user_id: str,
    ) -> UserSpaceCollabSnapshotResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer")
        can_edit = True
        try:
            await userspace_service.enforce_workspace_role(
                workspace_id, user_id, "editor"
            )
        except HTTPException:
            can_edit = False

        normalized_path = self._normalize_file_path(file_path)
        normalized_path = (
            await userspace_service.ensure_workspace_path_not_in_disabled_mount(
                workspace_id,
                normalized_path,
            )
        )
        key = (workspace_id, normalized_path)

        async with self._collab_lock:
            state = self._collab_docs.get(key)
        if state is None:
            content = await self._load_file_content(
                workspace_id,
                normalized_path,
                user_id,
            )
            async with self._collab_lock:
                state = self._collab_docs.get(key)
                if state is None:
                    state = _CollabDocState(
                        workspace_id=workspace_id,
                        file_path=normalized_path,
                        content=content,
                        version=1,
                    )
                    self._collab_docs[key] = state

        return UserSpaceCollabSnapshotResponse(
            workspace_id=workspace_id,
            file_path=normalized_path,
            version=state.version,
            content=state.content,
            read_only=not can_edit,
        )

    async def register_collab_client(
        self,
        workspace_id: str,
        file_path: str,
        websocket: WebSocket,
        user_id: str,
    ) -> UserSpaceCollabSnapshotResponse:
        normalized_path = self._normalize_file_path(file_path)
        normalized_path = (
            await userspace_service.ensure_workspace_path_not_in_disabled_mount(
                workspace_id,
                normalized_path,
            )
        )
        key = (workspace_id, normalized_path)
        async with self._collab_lock:
            state = self._collab_docs.get(key)
        if state is None:
            content = await self._load_file_content(
                workspace_id,
                normalized_path,
                user_id,
            )
            async with self._collab_lock:
                state = self._collab_docs.get(key)
                if state is None:
                    state = _CollabDocState(
                        workspace_id=workspace_id,
                        file_path=normalized_path,
                        content=content,
                        version=1,
                    )
                    self._collab_docs[key] = state
                state.clients.add(websocket)
        else:
            async with self._collab_lock:
                state = self._collab_docs.get(key)
                if state is None:
                    raise HTTPException(
                        status_code=409, detail="Collaboration state unavailable"
                    )
                state.clients.add(websocket)

        return UserSpaceCollabSnapshotResponse(
            workspace_id=workspace_id,
            file_path=normalized_path,
            version=state.version,
            content=state.content,
            read_only=False,
        )

    async def update_collab_presence(
        self,
        workspace_id: str,
        file_path: str,
        user_id: str,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        normalized_path = self._normalize_file_path(file_path)
        key = (workspace_id, normalized_path)
        async with self._collab_lock:
            presence_for_doc = self._collab_presence.setdefault(key, {})
            presence_for_doc[user_id] = {
                "user_id": user_id,
                "cursor": payload.get("cursor"),
                "selection": payload.get("selection"),
                "updated_at": self._utc_now().isoformat(),
            }
            return list(presence_for_doc.values())

    async def clear_collab_presence(
        self,
        workspace_id: str,
        file_path: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        normalized_path = self._normalize_file_path(file_path)
        key = (workspace_id, normalized_path)
        async with self._collab_lock:
            presence_for_doc = self._collab_presence.get(key)
            if not presence_for_doc:
                return []
            presence_for_doc.pop(user_id, None)
            if not presence_for_doc:
                self._collab_presence.pop(key, None)
                return []
            return list(presence_for_doc.values())

    async def get_collab_presence(
        self,
        workspace_id: str,
        file_path: str,
    ) -> list[dict[str, Any]]:
        normalized_path = self._normalize_file_path(file_path)
        key = (workspace_id, normalized_path)
        async with self._collab_lock:
            return list(self._collab_presence.get(key, {}).values())

    async def get_collab_clients(
        self,
        workspace_id: str,
        file_path: str,
    ) -> list[WebSocket]:
        normalized_path = self._normalize_file_path(file_path)
        key = (workspace_id, normalized_path)
        async with self._collab_lock:
            state = self._collab_docs.get(key)
            if not state:
                return []
            return list(state.clients)

    async def unregister_collab_client(
        self,
        workspace_id: str,
        file_path: str,
        websocket: WebSocket,
    ) -> None:
        normalized_path = self._normalize_file_path(file_path)
        key = (workspace_id, normalized_path)
        async with self._collab_lock:
            state = self._collab_docs.get(key)
            if not state:
                return
            state.clients.discard(websocket)

    async def apply_collab_update(
        self,
        workspace_id: str,
        file_path: str,
        content: str,
        user_id: str,
        expected_version: int | None = None,
    ) -> UserSpaceCollabSnapshotResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "editor")

        normalized_path = self._normalize_file_path(file_path)
        normalized_path = (
            await userspace_service.ensure_workspace_path_not_in_disabled_mount(
                workspace_id,
                normalized_path,
            )
        )
        key = (workspace_id, normalized_path)

        async with self._collab_lock:
            state = self._collab_docs.get(key)
        if state is None:
            existing = await self._load_file_content(
                workspace_id,
                normalized_path,
                user_id,
            )
            async with self._collab_lock:
                state = self._collab_docs.get(key)
                if state is None:
                    state = _CollabDocState(
                        workspace_id=workspace_id,
                        file_path=normalized_path,
                        content=existing,
                        version=1,
                    )
                    self._collab_docs[key] = state

        async with self._collab_lock:
            state = self._collab_docs.get(key)
            if state is None:
                raise HTTPException(
                    status_code=409, detail="Collaboration state unavailable"
                )
            if expected_version is not None and expected_version != state.version:
                raise RuntimeVersionConflictError(
                    expected_version=expected_version,
                    actual_version=state.version,
                )

            state.content = content
            state.version += 1
            version = state.version
            recipients = list(state.clients)

        await self._persist_file_content(
            workspace_id,
            normalized_path,
            content,
            user_id,
        )
        await self._store_collab_checkpoint(
            workspace_id, normalized_path, content, version
        )
        await self.bump_workspace_generation(workspace_id)
        await self._audit(
            workspace_id,
            "collab_update",
            user_id=user_id,
            payload={
                "file_path": normalized_path,
                "version": version,
                "size_bytes": len(content.encode("utf-8", errors="ignore")),
                "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            },
        )

        message = {
            "type": "update",
            "workspace_id": workspace_id,
            "file_path": normalized_path,
            "version": version,
            "content": content,
        }
        payload = json.dumps(message)

        async def _send(client: WebSocket) -> None:
            try:
                await client.send_text(payload)
            except Exception:
                pass

        if recipients:
            await asyncio.gather(*(_send(client) for client in recipients))

        return UserSpaceCollabSnapshotResponse(
            workspace_id=workspace_id,
            file_path=normalized_path,
            version=version,
            content=content,
            read_only=False,
        )

    async def create_collab_file(
        self,
        workspace_id: str,
        file_path: str,
        user_id: str,
        content: str = "",
    ) -> UserSpaceCollabSnapshotResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "editor")
        normalized_path = self._normalize_file_path(file_path)
        normalized_path = (
            await userspace_service.ensure_workspace_path_not_in_disabled_mount(
                workspace_id,
                normalized_path,
            )
        )
        await self._persist_file_content(
            workspace_id,
            normalized_path,
            content,
            user_id,
        )

        async with self._collab_lock:
            state = _CollabDocState(
                workspace_id=workspace_id,
                file_path=normalized_path,
                content=content,
                version=1,
            )
            self._collab_docs[(workspace_id, normalized_path)] = state

        await self._store_collab_checkpoint(workspace_id, normalized_path, content, 1)
        await self._audit(
            workspace_id,
            "collab_file_create",
            user_id=user_id,
            payload={"file_path": normalized_path},
        )
        return UserSpaceCollabSnapshotResponse(
            workspace_id=workspace_id,
            file_path=normalized_path,
            version=1,
            content=content,
            read_only=False,
        )

    async def rename_collab_file(
        self,
        workspace_id: str,
        old_path: str,
        new_path: str,
        user_id: str,
    ) -> dict[str, Any]:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "editor")
        normalized_old = self._normalize_file_path(old_path)
        normalized_new = self._normalize_file_path(new_path)
        normalized_old = (
            await userspace_service.ensure_workspace_path_not_in_disabled_mount(
                workspace_id,
                normalized_old,
            )
        )
        normalized_new = (
            await userspace_service.ensure_workspace_path_not_in_disabled_mount(
                workspace_id,
                normalized_new,
            )
        )
        session = await self.ensure_workspace_preview_session(workspace_id, user_id)
        source_payload = await self._runtime_provider_read_file(
            session.provider_session_id,
            normalized_old,
        )
        source_exists = bool(source_payload.get("exists", True))
        if not source_exists:
            raise HTTPException(status_code=404, detail="File not found")

        target_payload = await self._runtime_provider_read_file(
            session.provider_session_id,
            normalized_new,
        )
        target_exists = bool(target_payload.get("exists", False))
        if target_exists:
            raise HTTPException(status_code=409, detail="Target file already exists")

        content = str(source_payload.get("content", ""))
        await self._runtime_provider_write_file(
            session.provider_session_id,
            normalized_new,
            content,
        )
        await self._runtime_provider_delete_file(
            session.provider_session_id,
            normalized_old,
        )

        async with self._collab_lock:
            old_key = (workspace_id, normalized_old)
            new_key = (workspace_id, normalized_new)
            state = self._collab_docs.pop(old_key, None)
            if state:
                state.file_path = normalized_new
                self._collab_docs[new_key] = state
            presence = self._collab_presence.pop(old_key, None)
            if presence:
                self._collab_presence[new_key] = presence

        db = await get_db()
        model = self._collab_doc_model(db)
        row = await model.find_first(
            where={"workspaceId": workspace_id, "filePath": normalized_old}
        )
        if row:
            await model.update(
                where={"id": row.id},
                data={"filePath": normalized_new, "updatedAt": self._utc_now()},
            )

        await userspace_service.touch_workspace(workspace_id)
        await self._audit(
            workspace_id,
            "collab_file_rename",
            user_id=user_id,
            payload={"old_path": normalized_old, "new_path": normalized_new},
        )
        return {"old_path": normalized_old, "new_path": normalized_new, "success": True}

    async def delete_collab_file(
        self,
        workspace_id: str,
        file_path: str,
        user_id: str,
    ) -> dict[str, Any]:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "editor")
        normalized_path = self._normalize_file_path(file_path)
        normalized_path = (
            await userspace_service.ensure_workspace_path_not_in_disabled_mount(
                workspace_id,
                normalized_path,
            )
        )
        session = await self.ensure_workspace_preview_session(workspace_id, user_id)
        source_payload = await self._runtime_provider_read_file(
            session.provider_session_id,
            normalized_path,
        )
        source_exists = bool(source_payload.get("exists", True))
        if not source_exists:
            raise HTTPException(status_code=404, detail="File not found")
        await self._runtime_provider_delete_file(
            session.provider_session_id,
            normalized_path,
        )

        async with self._collab_lock:
            key = (workspace_id, normalized_path)
            self._collab_docs.pop(key, None)
            self._collab_presence.pop(key, None)

        db = await get_db()
        model = self._collab_doc_model(db)
        await model.delete_many(
            where={"workspaceId": workspace_id, "filePath": normalized_path}
        )
        await userspace_service.touch_workspace(workspace_id)
        await self._audit(
            workspace_id,
            "collab_file_delete",
            user_id=user_id,
            payload={"file_path": normalized_path},
        )
        return {"file_path": normalized_path, "success": True}

    async def capture_workspace_screenshot(
        self,
        workspace_id: str,
        user_id: str,
        path: str = "",
        width: int = 1440,
        height: int = 900,
        full_page: bool = True,
        timeout_ms: int = 25000,
        wait_for_selector: str = "body",
        capture_element: bool = False,
        clip_padding_px: int = 16,
        wait_after_load_ms: int = 1800,
        refresh_before_capture: bool = True,
    ) -> dict[str, Any]:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer")
        session = await self.ensure_workspace_preview_session(workspace_id, user_id)
        payload: dict[str, Any] = {
            "path": str(path or ""),
            "width": int(width),
            "height": int(height),
            "full_page": bool(full_page),
            "timeout_ms": int(timeout_ms),
            "wait_for_selector": str(wait_for_selector or ""),
            "capture_element": bool(capture_element),
            "clip_padding_px": int(clip_padding_px),
            "wait_after_load_ms": int(wait_after_load_ms),
            "refresh_before_capture": bool(refresh_before_capture),
        }
        return await self._runtime_provider_capture_screenshot(
            session.provider_session_id,
            payload,
        )

    async def probe_workspace_content(
        self,
        workspace_id: str,
        user_id: str,
        path: str = "",
        timeout_ms: int = 15000,
        wait_after_load_ms: int = 2000,
        inject_mock_context: bool = False,
    ) -> dict[str, Any]:
        """Run a lightweight Playwright content probe against the preview."""
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer")
        session = await self.ensure_workspace_preview_session(workspace_id, user_id)
        payload: dict[str, Any] = {
            "path": str(path or ""),
            "timeout_ms": int(timeout_ms),
            "wait_after_load_ms": int(wait_after_load_ms),
            "inject_mock_context": bool(inject_mock_context),
        }
        return await self._runtime_provider_content_probe(
            session.provider_session_id,
            payload,
        )

    async def exec_workspace_command(
        self,
        workspace_id: str,
        user_id: str,
        command: str,
        timeout_seconds: int = 30,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Execute a shell command in the workspace runtime container."""
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "editor")
        session = await self.ensure_workspace_preview_session(workspace_id, user_id)
        return await self._runtime_provider_exec_command(
            session.provider_session_id,
            command,
            timeout_seconds=timeout_seconds,
            cwd=cwd,
        )

    async def _store_collab_checkpoint(
        self,
        workspace_id: str,
        file_path: str,
        content: str,
        version: int,
    ) -> None:
        db = await get_db()
        model = self._collab_doc_model(db)
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        now = self._utc_now()
        row = await model.find_first(
            where={"workspaceId": workspace_id, "filePath": file_path}
        )
        payload = {
            "checkpointVersion": version,
            "docStateBase64": encoded,
            "updatedAt": now,
        }
        if row:
            await model.update(where={"id": row.id}, data=payload)
        else:
            await model.create(
                data={
                    "id": str(uuid4()),
                    "workspaceId": workspace_id,
                    "filePath": file_path,
                    "checkpointVersion": version,
                    "docStateBase64": encoded,
                    "createdAt": now,
                    "updatedAt": now,
                }
            )

    async def invalidate_workspace_runtime_state(self, workspace_id: str) -> None:
        async with self._collab_lock:
            keys_to_drop = [
                key for key in self._collab_docs.keys() if key[0] == workspace_id
            ]
            for key in keys_to_drop:
                self._collab_docs.pop(key, None)
                self._collab_presence.pop(key, None)
        await self._invalidate_workspace_runtime_caches(
            workspace_id,
            invalidate_preview_host=True,
        )
        await self.bump_workspace_generation(workspace_id)

        db = await get_db()
        model = self._runtime_session_model(db)
        sessions = await model.find_many(
            where={
                "workspaceId": workspace_id,
                "state": {"in": ["starting", "running"]},
            },
            order={"updatedAt": "desc"},
        )
        now = self._utc_now()
        provider_stop_errors: dict[str, str | None] = {}
        for session in sessions:
            raw_provider_session_id = getattr(session, "providerSessionId", None)
            provider_session_id: str = (
                str(raw_provider_session_id).strip()
                if raw_provider_session_id is not None
                else ""
            )
            provider_stop_error = provider_stop_errors.get(provider_session_id)
            if provider_session_id and provider_session_id not in provider_stop_errors:
                provider_stop_error = None
                try:
                    await self._runtime_provider_stop_session(provider_session_id)
                    await self._drop_provider_status_cache(provider_session_id)
                except HTTPException as exc:
                    provider_stop_error = str(exc.detail)
                    logger.warning(
                        "Runtime provider stop failed during invalidation for workspace %s: %s",
                        workspace_id,
                        provider_stop_error,
                    )
                provider_stop_errors[provider_session_id] = provider_stop_error

            last_error = "Workspace snapshot restore invalidated active runtime state"
            if provider_stop_error:
                last_error = (
                    f"{last_error}. Provider stop failed: {provider_stop_error}"
                )
            await model.update(
                where={"id": session.id},
                data={
                    "state": "stopped",
                    "lastHeartbeatAt": now,
                    "lastError": last_error,
                },
            )

    async def track_workspace_runtime_events(self, workspace_id: str) -> None:
        self._runtime_watch_workspaces.add(workspace_id)
        await self._ensure_runtime_watch_task()

    async def _ensure_runtime_watch_task(self) -> None:
        task = self._runtime_watch_task
        if task is not None and not task.done():
            return
        async with self._runtime_watch_task_lock:
            task = self._runtime_watch_task
            if task is not None and not task.done():
                return
            self._runtime_watch_task = asyncio.create_task(
                self._runtime_watch_loop(),
                name="userspace-runtime-watch-loop",
            )

    async def _runtime_watch_loop(self) -> None:
        while True:
            await asyncio.sleep(max(0.25, self._runtime_watch_interval_seconds))
            workspace_ids = list(self._runtime_watch_workspaces)
            if not workspace_ids:
                continue
            for workspace_id in workspace_ids:
                try:
                    await self._emit_runtime_phase_change_if_needed(workspace_id)
                except Exception:
                    logger.debug(
                        "Runtime watch loop poll failed for workspace %s",
                        workspace_id,
                        exc_info=True,
                    )

    async def _runtime_phase_signature(
        self,
        workspace_id: str,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        active = await self._get_active_session_row(workspace_id)
        if not active:
            payload: dict[str, Any] = {
                "runtime": {
                    "workspace_id": workspace_id,
                    "session_id": None,
                    "session_state": "stopped",
                    "devserver_running": False,
                    "runtime_operation_id": None,
                    "runtime_operation_phase": "stopped",
                    "last_error": None,
                }
            }
            return (
                (
                    "stopped",
                    False,
                    None,
                    "stopped",
                    None,
                    None,
                ),
                payload,
            )

        session = self._to_runtime_session(active)
        provider_status: dict[str, Any] | None = None
        last_error: str | None = session.last_error
        try:
            provider_status = await self._runtime_provider_get_status(
                session.provider_session_id
            )
        except HTTPException as exc:
            last_error = str(exc.detail)

        state = session.state
        devserver_running = False
        operation_id: str | None = None
        operation_phase: RuntimeOperationPhase | None = None

        if provider_status:
            raw_state = str(provider_status.get("state") or "").strip()
            if raw_state in {"starting", "running", "stopping", "stopped", "error"}:
                state = cast(RuntimeSessionState, raw_state)
            if "devserver_running" in provider_status:
                devserver_running = bool(provider_status.get("devserver_running"))
            operation_id = (
                str(provider_status.get("runtime_operation_id") or "").strip() or None
            )
            raw_phase = (
                str(provider_status.get("runtime_operation_phase") or "").strip()
                or None
            )
            if raw_phase in {
                "queued",
                "provisioning",
                "bootstrapping",
                "deps_install",
                "launching",
                "probing",
                "ready",
                "failed",
                "stopped",
            }:
                operation_phase = cast(RuntimeOperationPhase, raw_phase)
            last_error = self._resolve_provider_last_error(
                provider_status,
                fallback=last_error,
            )

        if not provider_status:
            devserver_running = state == "running"

        mount_rows: list[Any] = []
        try:
            db = await get_db()
            mount_rows = await db.workspacemount.find_many(
                where={"workspaceId": workspace_id},
                order={"createdAt": "asc"},
            )
        except Exception:
            logger.debug(
                "Failed to load runtime mount rows for workspace %s",
                workspace_id,
                exc_info=True,
            )

        mount_target_paths = [
            str(getattr(row, "targetPath", "") or "").strip()
            for row in mount_rows
            if str(getattr(row, "targetPath", "") or "").strip()
        ]
        mount_content_signature = (
            await userspace_service.get_runtime_mount_content_signature(
                workspace_id,
                mount_target_paths,
            )
        )

        payload = {
            "runtime": {
                "workspace_id": workspace_id,
                "session_id": session.id,
                "session_state": state,
                "devserver_running": devserver_running,
                "runtime_operation_id": operation_id,
                "runtime_operation_phase": operation_phase,
                "last_error": last_error,
                "mount_target_paths": mount_target_paths,
            }
        }
        signature = (
            state,
            devserver_running,
            session.id,
            operation_phase,
            operation_id,
            last_error,
            mount_content_signature,
        )
        return signature, payload

    async def _emit_runtime_phase_change_if_needed(self, workspace_id: str) -> None:
        signature, payload = await self._runtime_phase_signature(workspace_id)
        previous = self._runtime_watch_signatures.get(workspace_id)
        if previous == signature:
            return

        mount_contents_changed = previous is None or previous[-1] != signature[-1]
        self._runtime_watch_signatures[workspace_id] = signature
        if mount_contents_changed:
            await userspace_service.stage_runtime_mounts_into_sync_cache(workspace_id)
            userspace_service.invalidate_file_list_cache(workspace_id)
        event_type = (
            "runtime_mount_contents"
            if previous and previous[:-1] == signature[:-1]
            else "runtime_phase"
        )
        await self.bump_workspace_generation(
            workspace_id,
            event_type=event_type,
            payload=payload,
        )

    async def get_workspace_generation(self, workspace_id: str) -> int:
        return self._workspace_generation.get(workspace_id, 0)

    def get_workspace_event_payload(self, workspace_id: str) -> dict[str, Any]:
        generation = self._workspace_generation.get(workspace_id, 0)
        cached = self._workspace_event_payload.get(workspace_id)
        if not cached:
            return {"generation": generation}
        payload = dict(cached)
        payload["generation"] = generation
        return payload

    def _get_workspace_condition(self, workspace_id: str) -> asyncio.Condition:
        cond = self._workspace_events.get(workspace_id)
        if cond is None:
            cond = asyncio.Condition()
            self._workspace_events[workspace_id] = cond
        return cond

    async def bump_workspace_generation(
        self,
        workspace_id: str,
        event_type: str = "update",
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Increment generation counter and wake SSE subscribers."""
        gen = self._workspace_generation.get(workspace_id, 0) + 1
        self._workspace_generation[workspace_id] = gen
        event_payload: dict[str, Any] = {"generation": gen, "event_type": event_type}
        if payload:
            event_payload.update(payload)
        self._workspace_event_payload[workspace_id] = event_payload
        cond = self._get_workspace_condition(workspace_id)
        async with cond:
            cond.notify_all()
        return gen

    async def wait_workspace_generation(
        self,
        workspace_id: str,
        after_generation: int,
        timeout: float = 30.0,
    ) -> int:
        """Block until workspace generation exceeds *after_generation* or timeout."""
        cond = self._get_workspace_condition(workspace_id)
        try:
            async with cond:
                await asyncio.wait_for(
                    cond.wait_for(
                        lambda: self._workspace_generation.get(workspace_id, 0)
                        > after_generation
                    ),
                    timeout=timeout,
                )
        except asyncio.TimeoutError:
            pass
        return self._workspace_generation.get(workspace_id, 0)


userspace_runtime_service = UserSpaceRuntimeService()
