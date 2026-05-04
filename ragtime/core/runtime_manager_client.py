
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException

from ragtime.config import settings

@dataclass(frozen=True)
class RuntimeManagerRequestConfig:
    base_url: str
    headers: dict[str, str]
    timeout_seconds: float
    retry_attempts: int
    retry_base_delay_seconds: float


def get_runtime_manager_request_config() -> RuntimeManagerRequestConfig:
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
    unavailable_detail_prefix: str = "Runtime manager unavailable",
    request_failed_detail_prefix: str = "Runtime manager request failed",
) -> dict[str, Any]:
    config = get_runtime_manager_request_config()
    url = f"{config.base_url}/{path.lstrip('/')}"
    timeout = httpx.Timeout(
        timeout_override_seconds
        if timeout_override_seconds is not None
        else config.timeout_seconds
    )
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
                detail = f"{unavailable_detail_prefix} ({exc_type})"
                if exc_message:
                    detail = f"{detail}: {exc_message}"
                raise HTTPException(status_code=502, detail=detail) from exc

            if response.status_code >= 500 and attempt < config.retry_attempts:
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
        raise HTTPException(
            status_code=502,
            detail=(
                f"{request_failed_detail_prefix} "
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
