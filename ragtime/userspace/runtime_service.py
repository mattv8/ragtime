from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import posixpath
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from urllib.parse import quote
from uuid import uuid4

import httpx
from fastapi import HTTPException
from jose import JWTError, jwt  # type: ignore[import-untyped]
from starlette.websockets import WebSocket

from prisma import fields as prisma_fields
from ragtime.config import settings
from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.userspace.models import (
    RuntimeSessionState,
    UserSpaceCapabilityTokenResponse,
    UserSpaceCollabSnapshotResponse,
    UserSpaceFileResponse,
    UserSpaceRuntimeActionResponse,
    UserSpaceRuntimeSession,
    UserSpaceRuntimeSessionResponse,
    UserSpaceRuntimeStatusResponse,
)
from ragtime.userspace.service import userspace_service

logger = get_logger(__name__)

_RUNTIME_CAPABILITY_TTL_SECONDS = 900
_RUNTIME_DEVSERVER_PORT = 5173
_RUNTIME_PREVIEW_DEFAULT_BASE = f"http://127.0.0.1:{_RUNTIME_DEVSERVER_PORT}"
_RUNTIME_PROVIDER_MANAGER = "microvm_pool_v1"


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


class RuntimeVersionConflictError(Exception):
    def __init__(self, expected_version: int, actual_version: int) -> None:
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Version conflict: expected {expected_version}, current {actual_version}"
        )


class UserSpaceRuntimeService:
    def __init__(self) -> None:
        self._collab_docs: dict[tuple[str, str], _CollabDocState] = {}
        self._collab_lock = asyncio.Lock()
        self._workspace_generation: dict[str, int] = {}
        self._collab_presence: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

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
        normalized = posixpath.normpath((file_path or "").replace("\\", "/")).strip()
        if (
            normalized.startswith("../")
            or normalized == ".."
            or normalized.startswith("/")
        ):
            raise HTTPException(status_code=400, detail="Invalid file path")
        if normalized in {"", "."}:
            raise HTTPException(status_code=400, detail="Invalid file path")
        if userspace_service.is_reserved_internal_path(normalized):
            raise HTTPException(status_code=400, detail="Invalid file path")
        return normalized

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

    def _resolve_preview_base_url(self, session: UserSpaceRuntimeSession | None) -> str:
        value = (session.preview_internal_url or "").strip() if session else ""
        if value.startswith("http://") or value.startswith("https://"):
            return value.rstrip("/")
        return _RUNTIME_PREVIEW_DEFAULT_BASE

    async def _probe_preview_base_url(self, base_url: str) -> bool:
        health_candidates = ["/", "/@vite/client"]
        timeout = httpx.Timeout(connect=0.6, read=0.8, write=0.8, pool=0.6)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            for candidate in health_candidates:
                try:
                    response = await client.get(f"{base_url}{candidate}")
                except Exception:
                    continue
                if response.status_code < 500:
                    return True
        return False

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

        payload: dict[str, Any] = {
            "workspace_id": workspace_id,
            "leased_by_user_id": leased_by_user_id,
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
    ) -> dict[str, Any] | None:
        if not provider_session_id:
            return None
        self._require_runtime_manager()
        try:
            return await self._runtime_manager_request(
                "GET",
                f"/sessions/{provider_session_id}",
            )
        except HTTPException as exc:
            if "(404)" in str(exc.detail):
                return None
            raise

    async def _runtime_provider_restart_devserver(
        self,
        provider_session_id: str | None,
    ) -> dict[str, Any] | None:
        if not provider_session_id:
            return None
        self._require_runtime_manager()
        return await self._runtime_manager_request(
            "POST",
            f"/sessions/{provider_session_id}/restart",
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

    async def _ensure_session_row(
        self,
        workspace_id: str,
        leased_by_user_id: str,
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
                    session.provider_session_id
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
                        return self._to_runtime_session(current)
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
            return self._to_runtime_session(current)

        now = self._utc_now()
        provider_data = await self._runtime_provider_start_session(
            workspace_id,
            leased_by_user_id,
        )
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
                "launchCommand": str(provider_data.get("launch_command") or "") or None,
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
        return self._to_runtime_session(row)

    async def ensure_workspace_preview_session(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceRuntimeSession:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer")
        session = await self._ensure_session_row(workspace_id, user_id)
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
        return await self._ensure_session_row(workspace_id, owner_user_id)

    async def build_workspace_preview_upstream_url(
        self,
        workspace_id: str,
        user_id: str,
        path: str,
        query: str | None = None,
    ) -> str:
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

        session = await self._ensure_session_row(workspace_id, user_id)
        await self._audit(
            workspace_id,
            "session_start",
            user_id=user_id,
            session_id=session.id,
            payload={"provider_session_id": session.provider_session_id},
        )
        return UserSpaceRuntimeActionResponse(
            workspace_id=workspace_id,
            session_id=session.id,
            state=session.state,
            success=True,
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
                preview_url=f"/indexes/userspace/workspaces/{workspace_id}/preview/",
            )

        session = self._to_runtime_session(active)
        provider_status = await self._runtime_provider_get_status(
            session.provider_session_id
        )

        if (
            provider_status is None
            and session.runtime_provider == self._runtime_provider_name()
            and session.state in {"starting", "running"}
        ):
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
            } and await self._probe_preview_base_url(base_url)

        devserver_port = launch_port or _RUNTIME_DEVSERVER_PORT

        return UserSpaceRuntimeStatusResponse(
            workspace_id=workspace_id,
            session_state=state_for_response,
            session_id=session.id,
            devserver_running=devserver_running,
            devserver_port=devserver_port,
            launch_framework=launch_framework,
            launch_command=launch_command,
            launch_cwd=launch_cwd,
            preview_url=f"/indexes/userspace/workspaces/{workspace_id}/preview/",
            last_error=last_error,
        )

    async def restart_devserver(
        self,
        workspace_id: str,
        user_id: str,
    ) -> UserSpaceRuntimeActionResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "editor")

        active = await self._get_active_session_row(workspace_id)
        start = await self.start_runtime_session(workspace_id, user_id)
        if active:
            active_session = self._to_runtime_session(active)
            provider_restart = await self._runtime_provider_restart_devserver(
                active_session.provider_session_id
            )
            if provider_restart:
                delta = self._merge_provider_status(active_session, provider_restart)
                delta.update(
                    {
                        "state": delta.get(
                            "state", str(provider_restart.get("state") or "running")
                        ),
                        "lastHeartbeatAt": self._utc_now(),
                    }
                )
                db = await get_db()
                model = self._runtime_session_model(db)
                await self._runtime_session_update_row(
                    model,
                    active_session.id,
                    delta,
                )
        await self._audit(
            workspace_id,
            "devserver_restart",
            user_id=user_id,
            session_id=start.session_id,
        )
        return start

    async def issue_capability_token(
        self,
        workspace_id: str,
        user_id: str,
        capabilities: list[str],
        session_id: str | None = None,
        ttl_seconds: int = _RUNTIME_CAPABILITY_TTL_SECONDS,
    ) -> UserSpaceCapabilityTokenResponse:
        await userspace_service.enforce_workspace_role(workspace_id, user_id, "viewer")

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
        await self._audit(
            workspace_id,
            "capability_token_issue",
            user_id=user_id,
            session_id=session_id,
            payload={"capabilities": claims["capabilities"]},
        )
        return UserSpaceCapabilityTokenResponse(
            token=token,
            expires_at=expires_at,
            workspace_id=workspace_id,
            session_id=session_id,
            capabilities=cast(list[str], claims["capabilities"]),
        )

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
            raise HTTPException(
                status_code=401, detail="Invalid capability token"
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
        session = await self.ensure_workspace_preview_session(workspace_id, user_id)
        await self._runtime_provider_write_file(
            session.provider_session_id,
            normalized_path,
            content,
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
        key = (workspace_id, normalized_path)

        async with self._collab_lock:
            state = self._collab_docs.get(key)
            if state is None:
                content = await self._load_file_content(
                    workspace_id,
                    normalized_path,
                    user_id,
                )
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
        key = (workspace_id, normalized_path)
        async with self._collab_lock:
            state = self._collab_docs.get(key)
            if state is None:
                content = await self._load_file_content(
                    workspace_id,
                    normalized_path,
                    user_id,
                )
                state = _CollabDocState(
                    workspace_id=workspace_id,
                    file_path=normalized_path,
                    content=content,
                    version=1,
                )
                self._collab_docs[key] = state
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
        key = (workspace_id, normalized_path)

        async with self._collab_lock:
            state = self._collab_docs.get(key)
            if state is None:
                existing = await self._load_file_content(
                    workspace_id,
                    normalized_path,
                    user_id,
                )
                state = _CollabDocState(
                    workspace_id=workspace_id,
                    file_path=normalized_path,
                    content=existing,
                    version=1,
                )
                self._collab_docs[key] = state

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
        self._workspace_generation[workspace_id] = (
            self._workspace_generation.get(workspace_id, 0) + 1
        )
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
        for client in recipients:
            try:
                await client.send_text(json.dumps(message))
            except Exception:
                continue

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
        wait_after_load_ms: int = 900,
        refresh_before_capture: bool = True,
        filename: str | None = None,
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
            "wait_after_load_ms": int(wait_after_load_ms),
            "refresh_before_capture": bool(refresh_before_capture),
            "filename": filename,
        }
        return await self._runtime_provider_capture_screenshot(
            session.provider_session_id,
            payload,
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
        self._workspace_generation[workspace_id] = (
            self._workspace_generation.get(workspace_id, 0) + 1
        )

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
        for session in sessions:
            await model.update(
                where={"id": session.id},
                data={
                    "state": "stopped",
                    "lastHeartbeatAt": now,
                    "lastError": "Workspace snapshot restore invalidated active runtime state",
                },
            )

    async def get_workspace_generation(self, workspace_id: str) -> int:
        return self._workspace_generation.get(workspace_id, 0)


userspace_runtime_service = UserSpaceRuntimeService()
