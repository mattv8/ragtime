from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ragtime.config import settings
from ragtime.core.rate_limit import limiter
from ragtime.core.security import require_admin
from ragtime.core.webhooks import bearer_token, constant_time_token_matches
from ragtime.indexer.models import ToolType
from ragtime.indexer.repository import repository
from ragtime.pdm_automation.models import PdmWebhookConfigResponse, PdmWebhookEnableResponse, PdmWebhookPayload
from ragtime.pdm_automation.repository import pdm_automation_repository
from ragtime.pdm_automation.service import pdm_automation_service

router = APIRouter(tags=["pdm-automation"])
_MAX_BODY = 65536


def _base_url(request: Request) -> str:
    return str(getattr(settings, "external_base_url", "") or "").strip().rstrip("/") or str(request.base_url).rstrip("/")


async def _pdm_tool(tool_id: str) -> Any:
    tool = await repository.get_tool_config(tool_id)
    if tool is None or tool.tool_type != ToolType.SOLIDWORKS_PDM:
        raise HTTPException(status_code=404, detail="PDM tool not found")
    return tool


async def _read_body(request: Request) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > _MAX_BODY:
            raise HTTPException(status_code=413, detail="Request body too large")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/webhooks/pdm/{webhook_id}")
@limiter.limit("60/minute")
async def receive_pdm_webhook(webhook_id: str, request: Request) -> JSONResponse:
    target = await pdm_automation_repository.resolve_webhook(webhook_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Webhook not found")
    # Authenticate before body handling, then revalidate identity/state in the
    # acceptance transaction so pause/rotation/delete cannot race acceptance.
    from ragtime.core.encryption import decrypt_secret

    if not constant_time_token_matches(decrypt_secret(str(target.get("webhook_secret") or "")), bearer_token(request.headers)):
        raise HTTPException(status_code=401, detail="Invalid webhook credentials")
    body = await _read_body(request)
    try:
        payload = json.loads(body.decode("utf-8") if body else "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Malformed JSON body") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Malformed JSON body")
    try:
        parsed = PdmWebhookPayload.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid webhook payload") from exc
    accepted = await pdm_automation_repository.accept_authenticated_webhook(webhook_id, bearer_token(request.headers), parsed.event_id)
    if accepted == "missing":
        raise HTTPException(status_code=404, detail="Webhook not found")
    if accepted == "unauthorized":
        raise HTTPException(status_code=401, detail="Invalid webhook credentials")
    if accepted == "ignored":
        return JSONResponse(status_code=202, content={"status": "ignored"})
    pdm_automation_service.wake()
    return JSONResponse(status_code=202, content={"status": "accepted"})


@router.get("/indexes/tools/{tool_id}/pdm/webhook", response_model=PdmWebhookConfigResponse)
async def get_pdm_webhook(tool_id: str, request: Request, _user: Any = Depends(require_admin)) -> PdmWebhookConfigResponse:
    await _pdm_tool(tool_id)
    return await pdm_automation_repository.get_status(tool_id, _base_url(request))


@router.post("/indexes/tools/{tool_id}/pdm/webhook", response_model=PdmWebhookEnableResponse)
async def enable_pdm_webhook(tool_id: str, request: Request, _user: Any = Depends(require_admin)) -> PdmWebhookEnableResponse:
    await _pdm_tool(tool_id)
    return await pdm_automation_repository.enable_webhook(tool_id, _base_url(request))


@router.post("/indexes/tools/{tool_id}/pdm/webhook/rotate", response_model=PdmWebhookEnableResponse)
async def rotate_pdm_webhook(tool_id: str, request: Request, _user: Any = Depends(require_admin)) -> PdmWebhookEnableResponse:
    await _pdm_tool(tool_id)
    return await pdm_automation_repository.rotate_webhook(tool_id, _base_url(request))


@router.post("/indexes/tools/{tool_id}/pdm/webhook/pause", response_model=PdmWebhookConfigResponse)
async def pause_pdm_webhook(tool_id: str, request: Request, _user: Any = Depends(require_admin)) -> PdmWebhookConfigResponse:
    await _pdm_tool(tool_id)
    return await pdm_automation_repository.set_paused(tool_id, True, _base_url(request))


@router.post("/indexes/tools/{tool_id}/pdm/webhook/resume", response_model=PdmWebhookConfigResponse)
async def resume_pdm_webhook(tool_id: str, request: Request, _user: Any = Depends(require_admin)) -> PdmWebhookConfigResponse:
    await _pdm_tool(tool_id)
    result = await pdm_automation_repository.set_paused(tool_id, False, _base_url(request))
    pdm_automation_service.wake()
    return result


@router.delete("/indexes/tools/{tool_id}/pdm/webhook", response_model=PdmWebhookConfigResponse)
async def disable_pdm_webhook(tool_id: str, request: Request, _user: Any = Depends(require_admin)) -> PdmWebhookConfigResponse:
    await _pdm_tool(tool_id)
    return await pdm_automation_repository.disable_webhook(tool_id, _base_url(request))
