from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import HTTPException

from runtime.manager.models import (
    ManagerSession,
    PoolSlot,
    RuntimeSessionResponse,
    StartSessionRequest,
)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ManagerSession] = {}
        self._workspace_index: dict[str, str] = {}
        self._pool_slots: dict[str, PoolSlot] = {}
        self._lock = asyncio.Lock()
        self._default_preview_url = os.getenv(
            "USERSPACE_RUNTIME_DEFAULT_PREVIEW_URL", "http://127.0.0.1:5173"
        )
        self._warm_pool_size = self._get_positive_int_env(
            "USERSPACE_RUNTIME_POOL_SIZE",
            2,
        )
        self._max_pool_size = self._get_positive_int_env(
            "USERSPACE_RUNTIME_MAX_POOL_SIZE",
            max(4, self._warm_pool_size),
        )
        if self._max_pool_size < self._warm_pool_size:
            self._max_pool_size = self._warm_pool_size
        self._lease_ttl_seconds = self._get_positive_int_env(
            "USERSPACE_RUNTIME_LEASE_TTL_SECONDS",
            3600,
        )
        self._idle_evict_seconds = self._get_positive_int_env(
            "USERSPACE_RUNTIME_IDLE_EVICT_SECONDS",
            1200,
        )
        self._reconcile_interval_seconds = self._get_positive_int_env(
            "USERSPACE_RUNTIME_RECONCILE_INTERVAL_SECONDS",
            15,
        )
        self._reconcile_task: asyncio.Task[None] | None = None

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _get_positive_int_env(name: str, default_value: int) -> int:
        raw_value = os.getenv(name, str(default_value)).strip()
        try:
            parsed = int(raw_value)
            return parsed if parsed > 0 else default_value
        except Exception:
            return default_value

    def _new_slot(self) -> PoolSlot:
        now = self._utc_now()
        slot_id = f"slot-{uuid4().hex[:12]}"
        vm_id = f"microvm-{uuid4().hex[:12]}"
        return PoolSlot(
            slot_id=slot_id,
            vm_id=vm_id,
            lease_provider_session_id=None,
            workspace_id=None,
            state="warm",
            created_at=now,
            updated_at=now,
            last_used_at=now,
        )

    async def startup(self) -> None:
        async with self._lock:
            self._ensure_warm_slots_locked()
        if self._reconcile_task is None:
            self._reconcile_task = asyncio.create_task(self._reconcile_loop())

    async def shutdown(self) -> None:
        task = self._reconcile_task
        self._reconcile_task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    def _ensure_warm_slots_locked(self) -> None:
        warm_count = sum(
            1 for slot in self._pool_slots.values() if slot.state == "warm"
        )
        while (
            warm_count < self._warm_pool_size
            and len(self._pool_slots) < self._max_pool_size
        ):
            slot = self._new_slot()
            self._pool_slots[slot.slot_id] = slot
            warm_count += 1

    def _select_or_create_slot_locked(self) -> PoolSlot | None:
        warm_slots = [
            slot for slot in self._pool_slots.values() if slot.state == "warm"
        ]
        if warm_slots:
            warm_slots.sort(key=lambda slot: slot.last_used_at)
            return warm_slots[0]
        if len(self._pool_slots) >= self._max_pool_size:
            return None
        slot = self._new_slot()
        self._pool_slots[slot.slot_id] = slot
        return slot

    def _lease_slot_locked(
        self,
        slot: PoolSlot,
        provider_session_id: str,
        workspace_id: str,
        now: datetime,
    ) -> None:
        slot.state = "leased"
        slot.lease_provider_session_id = provider_session_id
        slot.workspace_id = workspace_id
        slot.updated_at = now
        slot.last_used_at = now

    def _ensure_session_lease_locked(
        self,
        session: ManagerSession,
        now: datetime,
    ) -> None:
        current_slot: PoolSlot | None = None
        if session.lease_slot_id:
            current_slot = self._pool_slots.get(session.lease_slot_id)

        slot_ready = current_slot is not None and (
            (
                current_slot.state == "warm"
                and (
                    current_slot.lease_provider_session_id is None
                    or current_slot.lease_provider_session_id
                    == session.provider_session_id
                )
            )
            or (
                current_slot.state == "leased"
                and current_slot.lease_provider_session_id
                == session.provider_session_id
                and current_slot.workspace_id == session.workspace_id
            )
        )

        if slot_ready and current_slot is not None:
            self._lease_slot_locked(
                current_slot,
                provider_session_id=session.provider_session_id,
                workspace_id=session.workspace_id,
                now=now,
            )
            session.lease_slot_id = current_slot.slot_id
            session.lease_vm_id = current_slot.vm_id
            session.lease_started_at = now
            return

        selected = self._select_or_create_slot_locked()
        if selected is None:
            raise HTTPException(
                status_code=503,
                detail="Runtime pool exhausted. Retry after an existing session stops.",
            )
        self._lease_slot_locked(
            selected,
            provider_session_id=session.provider_session_id,
            workspace_id=session.workspace_id,
            now=now,
        )
        session.lease_slot_id = selected.slot_id
        session.lease_vm_id = selected.vm_id
        session.lease_started_at = now

    def _release_slot_locked(self, slot_id: str, now: datetime) -> None:
        slot = self._pool_slots.get(slot_id)
        if not slot:
            return
        slot.state = "warm"
        slot.lease_provider_session_id = None
        slot.workspace_id = None
        slot.updated_at = now
        slot.last_used_at = now

    def _cleanup_expired_sessions_locked(self, now: datetime) -> None:
        for provider_session_id, session in list(self._sessions.items()):
            if (
                session.state in {"starting", "running"}
                and session.lease_expires_at <= now
            ):
                session.state = "error"
                session.devserver_running = False
                session.last_error = "Lease expired"
                session.updated_at = now
                if session.lease_slot_id:
                    self._release_slot_locked(session.lease_slot_id, now)

            if session.state in {"stopped", "error"}:
                workspace_provider_session = self._workspace_index.get(
                    session.workspace_id
                )
                if workspace_provider_session == provider_session_id:
                    self._workspace_index.pop(session.workspace_id, None)

    def _evict_idle_warm_slots_locked(self, now: datetime) -> None:
        warm_slots = [
            slot for slot in self._pool_slots.values() if slot.state == "warm"
        ]
        if len(warm_slots) <= self._warm_pool_size:
            return
        warm_slots.sort(key=lambda slot: slot.last_used_at)
        for slot in warm_slots:
            warm_count = sum(
                1 for existing in self._pool_slots.values() if existing.state == "warm"
            )
            if warm_count <= self._warm_pool_size:
                break
            idle_seconds = (now - slot.last_used_at).total_seconds()
            if idle_seconds < self._idle_evict_seconds:
                continue
            self._pool_slots.pop(slot.slot_id, None)

    async def _reconcile_loop(self) -> None:
        while True:
            await asyncio.sleep(self._reconcile_interval_seconds)
            now = self._utc_now()
            async with self._lock:
                self._cleanup_expired_sessions_locked(now)
                self._evict_idle_warm_slots_locked(now)
                self._ensure_warm_slots_locked()

    @staticmethod
    def _as_response(session: ManagerSession) -> RuntimeSessionResponse:
        return RuntimeSessionResponse(
            provider_session_id=session.provider_session_id,
            workspace_id=session.workspace_id,
            state=session.state,
            preview_internal_url=session.preview_internal_url,
            devserver_running=session.devserver_running,
            last_error=session.last_error,
            updated_at=session.updated_at,
        )

    async def start_session(
        self, request: StartSessionRequest
    ) -> RuntimeSessionResponse:
        async with self._lock:
            now = self._utc_now()
            self._cleanup_expired_sessions_locked(now)
            provider_session_id = request.provider_session_id
            if provider_session_id and provider_session_id in self._sessions:
                session = self._sessions[provider_session_id]
            else:
                existing_id = self._workspace_index.get(request.workspace_id)
                if existing_id and existing_id in self._sessions:
                    session = self._sessions[existing_id]
                else:
                    new_id = provider_session_id or (
                        f"mgr-{request.workspace_id[:8]}-{uuid4().hex[:8]}"
                    )
                    session = ManagerSession(
                        provider_session_id=new_id,
                        workspace_id=request.workspace_id,
                        leased_by_user_id=request.leased_by_user_id,
                        preview_internal_url=self._default_preview_url,
                        state="running",
                        devserver_running=True,
                        last_error=None,
                        updated_at=now,
                        lease_slot_id=None,
                        lease_vm_id=None,
                        lease_started_at=now,
                        lease_expires_at=now
                        + timedelta(seconds=self._lease_ttl_seconds),
                    )
                    self._sessions[new_id] = session

            self._ensure_session_lease_locked(session, now)
            self._workspace_index[session.workspace_id] = session.provider_session_id
            session.leased_by_user_id = request.leased_by_user_id
            session.state = "running"
            session.devserver_running = True
            session.last_error = None
            session.updated_at = now
            session.lease_expires_at = now + timedelta(seconds=self._lease_ttl_seconds)
            if session.lease_slot_id and session.lease_slot_id in self._pool_slots:
                self._pool_slots[session.lease_slot_id].last_used_at = now
                self._pool_slots[session.lease_slot_id].updated_at = now
            self._ensure_warm_slots_locked()
            return self._as_response(session)

    async def get_session(self, provider_session_id: str) -> RuntimeSessionResponse:
        async with self._lock:
            now = self._utc_now()
            self._cleanup_expired_sessions_locked(now)
            session = self._sessions.get(provider_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Runtime session not found")
            if session.state in {"running", "starting"}:
                session.lease_expires_at = now + timedelta(
                    seconds=self._lease_ttl_seconds
                )
                session.updated_at = now
                if session.lease_slot_id and session.lease_slot_id in self._pool_slots:
                    self._pool_slots[session.lease_slot_id].last_used_at = now
                    self._pool_slots[session.lease_slot_id].updated_at = now
            return self._as_response(session)

    async def stop_session(self, provider_session_id: str) -> RuntimeSessionResponse:
        async with self._lock:
            now = self._utc_now()
            session = self._sessions.get(provider_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Runtime session not found")
            session.state = "stopped"
            session.devserver_running = False
            session.updated_at = now
            if session.lease_slot_id:
                self._release_slot_locked(session.lease_slot_id, now)
            self._workspace_index.pop(session.workspace_id, None)
            self._ensure_warm_slots_locked()
            return self._as_response(session)

    async def restart_devserver(
        self, provider_session_id: str
    ) -> RuntimeSessionResponse:
        async with self._lock:
            now = self._utc_now()
            session = self._sessions.get(provider_session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Runtime session not found")
            if session.state not in {"running", "starting"}:
                raise HTTPException(
                    status_code=409,
                    detail="Runtime session is not active",
                )
            session.state = "running"
            session.devserver_running = True
            session.last_error = None
            session.updated_at = now
            session.lease_expires_at = now + timedelta(seconds=self._lease_ttl_seconds)
            if session.lease_slot_id and session.lease_slot_id in self._pool_slots:
                self._pool_slots[session.lease_slot_id].last_used_at = now
                self._pool_slots[session.lease_slot_id].updated_at = now
            return self._as_response(session)

    async def pool_status(self) -> dict[str, int]:
        async with self._lock:
            self._cleanup_expired_sessions_locked(self._utc_now())
            warm = sum(1 for slot in self._pool_slots.values() if slot.state == "warm")
            leased = sum(
                1 for slot in self._pool_slots.values() if slot.state == "leased"
            )
            active_sessions = sum(
                1
                for session in self._sessions.values()
                if session.state in {"running", "starting"}
            )
            return {
                "warm_slots": warm,
                "leased_slots": leased,
                "total_slots": len(self._pool_slots),
                "active_sessions": active_sessions,
                "warm_pool_size": self._warm_pool_size,
                "max_pool_size": self._max_pool_size,
            }
