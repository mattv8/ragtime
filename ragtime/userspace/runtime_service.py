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
_RUNTIME_PROVIDER_LOCAL = "microvm_pool_v1"
_RUNTIME_PROVIDER_MANAGER = "runtime_manager_v1"


@dataclass
class _CollabDocState:
    workspace_id: str
    file_path: str
    content: str
    version: int = 0
    clients: set[WebSocket] = field(default_factory=set)


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

    def _runtime_session_model(self, db: Any) -> Any:
        return getattr(db, "userspaceruntimesession")

    def _runtime_audit_model(self, db: Any) -> Any:
        return getattr(db, "userspaceruntimeauditevent")

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
            created_at=getattr(row, "createdAt"),
            updated_at=getattr(row, "updatedAt"),
            last_heartbeat_at=getattr(row, "lastHeartbeatAt", None),
            idle_expires_at=getattr(row, "idleExpiresAt", None),
            ttl_expires_at=getattr(row, "ttlExpiresAt", None),
            last_error=getattr(row, "lastError", None),
        )

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
        if self._runtime_manager_enabled():
            return _RUNTIME_PROVIDER_MANAGER
        return _RUNTIME_PROVIDER_LOCAL

    def _runtime_manager_enabled(self) -> bool:
        manager_url = str(
            getattr(settings, "userspace_runtime_manager_url", "")
        ).strip()
        return manager_url.startswith("http://") or manager_url.startswith("https://")

    async def _runtime_manager_request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base_url = str(
            getattr(
                settings,
                "userspace_runtime_manager_url",
                "http://runtime:8090",
            )
        ).rstrip("/")
        url = f"{base_url}/{path.lstrip('/')}"
        headers: dict[str, str] = {}
        manager_auth_token = str(
            getattr(settings, "userspace_runtime_manager_auth_token", "")
        )
        if manager_auth_token:
            headers["Authorization"] = f"Bearer {manager_auth_token}"

        timeout = httpx.Timeout(
            float(getattr(settings, "userspace_runtime_manager_timeout_seconds", 5.0))
        )
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            try:
                response = await client.request(
                    method,
                    url,
                    json=json_payload,
                    headers=headers,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Runtime manager unavailable: {exc}",
                ) from exc

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
        if not self._runtime_manager_enabled():
            provider_session_id = existing_provider_session_id or (
                f"local-{workspace_id[:8]}-{uuid4().hex[:8]}"
            )
            return {
                "provider_session_id": provider_session_id,
                "preview_internal_url": _RUNTIME_PREVIEW_DEFAULT_BASE,
                "state": "running",
                "last_error": None,
            }

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
        if not self._runtime_manager_enabled():
            return
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
        if not self._runtime_manager_enabled():
            return None
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
        if not self._runtime_manager_enabled():
            return None
        return await self._runtime_manager_request(
            "POST",
            f"/sessions/{provider_session_id}/restart",
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
            manager_enabled = self._runtime_manager_enabled()
            target_provider_name = self._runtime_provider_name()
            if (
                session.state == "running"
                and session.runtime_provider == target_provider_name
            ):
                return session

            provider_data = await self._runtime_provider_start_session(
                workspace_id,
                leased_by_user_id,
                existing_provider_session_id=(
                    session.provider_session_id
                    if manager_enabled
                    and session.runtime_provider == target_provider_name
                    else None
                ),
            )
            current = await model.update(
                where={"id": session.id},
                data={
                    "state": str(provider_data.get("state") or "running"),
                    "runtimeProvider": self._runtime_provider_name(),
                    "providerSessionId": str(
                        provider_data.get("provider_session_id")
                        or session.provider_session_id
                        or ""
                    ),
                    "previewInternalUrl": str(
                        provider_data.get("preview_internal_url")
                        or session.preview_internal_url
                        or _RUNTIME_PREVIEW_DEFAULT_BASE
                    ),
                    "lastHeartbeatAt": self._utc_now(),
                    "lastError": provider_data.get("last_error"),
                },
            )
            return self._to_runtime_session(current)

        now = self._utc_now()
        provider_data = await self._runtime_provider_start_session(
            workspace_id,
            leased_by_user_id,
        )
        row = await model.create(
            data={
                "id": str(uuid4()),
                "workspaceId": workspace_id,
                "leasedByUserId": leased_by_user_id,
                "state": str(provider_data.get("state") or "running"),
                "runtimeProvider": self._runtime_provider_name(),
                "providerSessionId": str(
                    provider_data.get("provider_session_id")
                    or f"local-{workspace_id[:8]}-{uuid4().hex[:8]}"
                ),
                "previewInternalUrl": str(
                    provider_data.get("preview_internal_url")
                    or _RUNTIME_PREVIEW_DEFAULT_BASE
                ),
                "createdAt": now,
                "updatedAt": now,
                "lastHeartbeatAt": now,
                "idleExpiresAt": now + timedelta(hours=4),
                "ttlExpiresAt": now + timedelta(hours=24),
                "lastError": provider_data.get("last_error"),
            }
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
        attempts: list[dict[str, Any]] = [
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
            attempts.append(
                {
                    "workspace": {"connect": {"id": workspace_id}},
                    "user": {"connect": {"id": user_id}},
                    "sessionId": session_id,
                    "eventType": event_type,
                    "eventPayload": payload_json,
                }
            )

        try:
            for data in attempts:
                try:
                    await model.create(data=data)
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
        row = await model.update(
            where={"id": active_session.id},
            data={
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

        state_for_response = session.state
        preview_internal_url = session.preview_internal_url
        last_error = session.last_error

        if provider_status:
            state_candidate = str(provider_status.get("state") or "").strip()
            if state_candidate in {
                "starting",
                "running",
                "stopping",
                "stopped",
                "error",
            }:
                state_for_response = cast(RuntimeSessionState, state_candidate)
            preview_candidate = str(
                provider_status.get("preview_internal_url") or ""
            ).strip()
            if preview_candidate:
                preview_internal_url = preview_candidate
            provider_last_error = provider_status.get("last_error")
            if isinstance(provider_last_error, str):
                last_error = provider_last_error

            if (
                state_for_response != session.state
                or preview_internal_url != session.preview_internal_url
                or last_error != session.last_error
            ):
                db = await get_db()
                model = self._runtime_session_model(db)
                updated = await model.update(
                    where={"id": session.id},
                    data={
                        "state": state_for_response,
                        "previewInternalUrl": preview_internal_url,
                        "lastError": last_error,
                        "lastHeartbeatAt": self._utc_now(),
                    },
                )
                session = self._to_runtime_session(updated)

        base_url = self._resolve_preview_base_url(
            UserSpaceRuntimeSession(
                id=session.id,
                workspace_id=session.workspace_id,
                leased_by_user_id=session.leased_by_user_id,
                state=state_for_response,
                runtime_provider=session.runtime_provider,
                provider_session_id=session.provider_session_id,
                preview_internal_url=preview_internal_url,
                created_at=session.created_at,
                updated_at=session.updated_at,
                last_heartbeat_at=session.last_heartbeat_at,
                idle_expires_at=session.idle_expires_at,
                ttl_expires_at=session.ttl_expires_at,
                last_error=last_error,
            )
        )

        if provider_status and "devserver_running" in provider_status:
            devserver_running = bool(provider_status.get("devserver_running"))
        else:
            devserver_running = state_for_response in {
                "starting",
                "running",
            } and await self._probe_preview_base_url(base_url)

        return UserSpaceRuntimeStatusResponse(
            workspace_id=workspace_id,
            session_state=state_for_response,
            session_id=session.id,
            devserver_running=devserver_running,
            devserver_port=_RUNTIME_DEVSERVER_PORT,
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
                db = await get_db()
                model = self._runtime_session_model(db)
                await model.update(
                    where={"id": active_session.id},
                    data={
                        "state": str(provider_restart.get("state") or "running"),
                        "lastHeartbeatAt": self._utc_now(),
                        "lastError": provider_restart.get("last_error"),
                        "previewInternalUrl": str(
                            provider_restart.get("preview_internal_url")
                            or active_session.preview_internal_url
                            or _RUNTIME_PREVIEW_DEFAULT_BASE
                        ),
                    },
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

    async def _load_file_content(self, workspace_id: str, normalized_path: str) -> str:
        target = userspace_service.resolve_workspace_file_path(
            workspace_id, normalized_path
        )
        if not target.exists() or not target.is_file():
            return ""
        try:
            return target.read_text(encoding="utf-8")
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to read file: {exc}"
            ) from exc

    async def _persist_file_content(
        self,
        workspace_id: str,
        normalized_path: str,
        content: str,
    ) -> None:
        target = userspace_service.resolve_workspace_file_path(
            workspace_id, normalized_path
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(content, encoding="utf-8")
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to write file: {exc}"
            ) from exc
        await userspace_service.touch_workspace(workspace_id)

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
                content = await self._load_file_content(workspace_id, normalized_path)
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
    ) -> UserSpaceCollabSnapshotResponse:
        normalized_path = self._normalize_file_path(file_path)
        key = (workspace_id, normalized_path)
        async with self._collab_lock:
            state = self._collab_docs.get(key)
            if state is None:
                content = await self._load_file_content(workspace_id, normalized_path)
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
                existing = await self._load_file_content(workspace_id, normalized_path)
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

        await self._persist_file_content(workspace_id, normalized_path, content)
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
        await self._persist_file_content(workspace_id, normalized_path, content)

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

        old_target = userspace_service.resolve_workspace_file_path(
            workspace_id, normalized_old
        )
        new_target = userspace_service.resolve_workspace_file_path(
            workspace_id, normalized_new
        )
        if not old_target.exists() or not old_target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        if new_target.exists():
            raise HTTPException(status_code=409, detail="Target file already exists")
        new_target.parent.mkdir(parents=True, exist_ok=True)
        old_target.rename(new_target)

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
        target = userspace_service.resolve_workspace_file_path(
            workspace_id, normalized_path
        )
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        target.unlink()

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
