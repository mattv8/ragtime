from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_HEALTH_PATHS = {"/health", "/health/"}
_BACKUP_STATUS_PREFIXES = (
    "/indexes/server-backups/jobs/",
    "/indexes/server-backups/restore-jobs/",
)


@dataclass(frozen=True)
class MaintenanceStatus:
    active: bool
    lease_id: str | None = None
    reason: str | None = None
    retry_after_seconds: int | None = None
    acquired_at: datetime | None = None
    expires_at: datetime | None = None

    @classmethod
    def inactive(cls) -> "MaintenanceStatus":
        return cls(active=False)


def is_maintenance_bypass_path(method: str, path: str) -> bool:
    normalized_method = (method or "GET").upper()
    normalized_path = path or "/"
    if normalized_method not in _SAFE_METHODS:
        return False
    if normalized_path in _HEALTH_PATHS:
        return True
    return any(normalized_path.startswith(prefix) and not normalized_path.endswith("/download") for prefix in _BACKUP_STATUS_PREFIXES)


class ProcessMaintenanceState:
    def __init__(self, *, default_ttl_seconds: int = 900) -> None:
        self._lock = asyncio.Lock()
        self._inactive = asyncio.Event()
        self._inactive.set()
        self._status = MaintenanceStatus.inactive()
        self._default_ttl_seconds = max(1, int(default_ttl_seconds))

    def _expire_locked(self, now: datetime) -> None:
        if not self._status.active:
            return
        if self._status.expires_at is None or self._status.expires_at > now:
            return
        lease_id = self._status.lease_id
        self._status = MaintenanceStatus.inactive()
        self._inactive.set()
        logger.info("Server maintenance lease %s expired", lease_id)

    async def acquire(
        self,
        lease_id: str,
        *,
        reason: str | None = None,
        retry_after_seconds: int = 60,
    ) -> MaintenanceStatus:
        async with self._lock:
            now = datetime.now(timezone.utc)
            self._expire_locked(now)
            if self._status.active:
                if self._status.lease_id == lease_id:
                    return self._status
                raise RuntimeError("Maintenance lease already active")
            self._status = MaintenanceStatus(
                active=True,
                lease_id=lease_id,
                reason=reason,
                retry_after_seconds=retry_after_seconds,
                acquired_at=now,
                expires_at=now + timedelta(seconds=self._default_ttl_seconds),
            )
            self._inactive.clear()
            logger.info("Server maintenance activated for lease %s", lease_id)
            return self._status

    async def release(self, lease_id: str) -> MaintenanceStatus:
        async with self._lock:
            self._expire_locked(datetime.now(timezone.utc))
            if not self._status.active:
                return self._status
            if self._status.lease_id != lease_id:
                raise RuntimeError("Maintenance lease held by another lease")
            self._status = MaintenanceStatus.inactive()
            self._inactive.set()
            logger.info("Server maintenance released for lease %s", lease_id)
            return self._status

    async def renew(self, lease_id: str, ttl_seconds: int | None = None) -> MaintenanceStatus:
        async with self._lock:
            now = datetime.now(timezone.utc)
            self._expire_locked(now)
            if not self._status.active:
                raise RuntimeError("Maintenance lease is not active")
            if self._status.lease_id != lease_id:
                raise RuntimeError("Maintenance lease held by another lease")
            ttl = self._default_ttl_seconds if ttl_seconds is None else max(1, int(ttl_seconds))
            base_time = now
            if self._status.expires_at is not None and self._status.expires_at > base_time:
                base_time = self._status.expires_at
            self._status = MaintenanceStatus(
                active=True,
                lease_id=self._status.lease_id,
                reason=self._status.reason,
                retry_after_seconds=self._status.retry_after_seconds,
                acquired_at=self._status.acquired_at,
                expires_at=base_time + timedelta(seconds=ttl),
            )
            logger.info("Server maintenance renewed for lease %s", lease_id)
            return self._status

    async def snapshot(self) -> MaintenanceStatus:
        async with self._lock:
            self._expire_locked(datetime.now(timezone.utc))
            return self._status

    async def wait_until_inactive(self) -> None:
        await self._inactive.wait()


class MaintenanceModeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, state: ProcessMaintenanceState) -> None:
        super().__init__(app)
        self._state = state

    async def dispatch(self, request: Request, call_next):
        status = await self._state.snapshot()
        if not status.active or is_maintenance_bypass_path(request.method, request.url.path):
            return await call_next(request)
        if request.method.upper() not in _SAFE_METHODS:
            return JSONResponse(
                status_code=503,
                content={"detail": "Server maintenance is active. Retry later."},
                headers={"Retry-After": str(status.retry_after_seconds or 60)},
            )
        return await call_next(request)
