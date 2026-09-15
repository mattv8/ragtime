from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException

from ragtime.config import settings
from ragtime.core.http_client import RejectResponseCookies


@dataclass(frozen=True)
class RuntimeManagerRequestConfig:
    base_url: str
    headers: dict[str, str]
    timeout_seconds: float
    retry_attempts: int
    retry_base_delay_seconds: float


@dataclass
class _RuntimeManagerClientState:
    client: httpx.AsyncClient
    lock: asyncio.Lock
    drained: asyncio.Event
    closed: asyncio.Event
    active_requests: int = 0
    closing: bool = False


_runtime_manager_client_states: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _RuntimeManagerClientState] = weakref.WeakKeyDictionary()


def _new_runtime_manager_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        follow_redirects=True,
        cookies=RejectResponseCookies(),
    )


def _get_runtime_manager_client_state() -> _RuntimeManagerClientState:
    loop = asyncio.get_running_loop()
    state = _runtime_manager_client_states.get(loop)
    if state is None:
        state = _RuntimeManagerClientState(
            client=_new_runtime_manager_client(),
            lock=asyncio.Lock(),
            drained=asyncio.Event(),
            closed=asyncio.Event(),
        )
        state.drained.set()
        _runtime_manager_client_states[loop] = state
    return state


async def _acquire_runtime_manager_client() -> tuple[_RuntimeManagerClientState, httpx.AsyncClient]:
    state = _get_runtime_manager_client_state()
    async with state.lock:
        # A shutdown removes its state before waiting for active requests. A
        # request that races with it must therefore use a fresh client.
        if state.closing:
            state = _get_runtime_manager_client_state()
        state.active_requests += 1
        state.drained.clear()
        return state, state.client


async def _release_runtime_manager_client(state: _RuntimeManagerClientState) -> None:
    async with state.lock:
        state.active_requests -= 1
        if state.active_requests == 0:
            state.drained.set()


async def close_runtime_manager_client() -> None:
    """Close this event loop's pooled runtime-manager HTTP client."""
    loop = asyncio.get_running_loop()
    state = _runtime_manager_client_states.get(loop)
    if state is None:
        return

    close_client = False
    async with state.lock:
        if not state.closing:
            state.closing = True
            if _runtime_manager_client_states.get(loop) is state:
                del _runtime_manager_client_states[loop]
            close_client = True

    if close_client:
        try:
            await state.drained.wait()
            await state.client.aclose()
        finally:
            state.closed.set()
    else:
        await state.closed.wait()


def get_runtime_manager_request_config() -> RuntimeManagerRequestConfig:
    base_url = str(
        getattr(
            settings,
            "userspace_runtime_manager_url",
            "http://runtime:8090",
        )
    ).strip()
    manager_auth_token = str(getattr(settings, "userspace_runtime_auth_token", "")).strip()
    headers: dict[str, str] = {}
    if manager_auth_token:
        headers["Authorization"] = f"Bearer {manager_auth_token}"

    timeout_seconds = float(getattr(settings, "userspace_runtime_manager_timeout_seconds", 120.0))
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
    retry_base_delay_seconds = float(getattr(settings, "userspace_runtime_manager_retry_delay_seconds", 0.2))

    return RuntimeManagerRequestConfig(
        base_url=base_url.rstrip("/"),
        headers=headers,
        timeout_seconds=timeout_seconds,
        retry_attempts=retry_attempts,
        retry_base_delay_seconds=retry_base_delay_seconds,
    )


def runtime_manager_enabled(
    config: RuntimeManagerRequestConfig | None = None,
) -> bool:
    config = config or get_runtime_manager_request_config()
    manager_url = config.base_url
    return manager_url.startswith("http://") or manager_url.startswith("https://")


async def runtime_manager_request(
    method: str,
    path: str,
    *,
    json_payload: dict[str, Any] | None = None,
    timeout_override_seconds: float | None = None,
    retry_safe: bool = True,
    surface_error_status: bool = False,
    unavailable_detail_prefix: str = "Runtime manager unavailable",
    request_failed_detail_prefix: str = "Runtime manager request failed",
) -> dict[str, Any]:
    config = get_runtime_manager_request_config()
    url = f"{config.base_url}/{path.lstrip('/')}"
    timeout = httpx.Timeout(timeout_override_seconds if timeout_override_seconds is not None else config.timeout_seconds)
    state, client = await _acquire_runtime_manager_client()
    response: httpx.Response | None = None
    try:
        attempts = config.retry_attempts if retry_safe else 1
        for attempt in range(1, attempts + 1):
            try:
                response = await client.request(
                    method,
                    url,
                    json=json_payload,
                    headers=config.headers,
                    timeout=timeout,
                )
            except Exception as exc:
                if attempt < attempts:
                    await asyncio.sleep(config.retry_base_delay_seconds * attempt)
                    continue
                exc_type = exc.__class__.__name__
                exc_message = str(exc).strip()
                detail = f"{unavailable_detail_prefix} ({exc_type})"
                if exc_message:
                    detail = f"{detail}: {exc_message}"
                raise HTTPException(status_code=502, detail=detail) from exc

            if response.status_code >= 500 and attempt < attempts:
                await response.aclose()
                response = None
                await asyncio.sleep(config.retry_base_delay_seconds * attempt)
                continue
            break

        if response is None:
            raise HTTPException(
                status_code=502,
                detail=f"{unavailable_detail_prefix} (no response)",
            )

        if response.status_code >= 400:
            body_preview = response.text[:256]
            if surface_error_status and 400 <= response.status_code < 500:
                raise HTTPException(status_code=response.status_code, detail=body_preview or "Runtime manager request rejected")
            raise HTTPException(
                status_code=502,
                detail=(f"{request_failed_detail_prefix} ({response.status_code}): {body_preview}"),
            )

        if not response.content:
            return {}
        try:
            data = response.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    finally:
        if response is not None:
            await response.aclose()
        await _release_runtime_manager_client(state)
