from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from ragtime.core.rate_limit import limiter
from ragtime.git_webhooks.models import GitPushEvent, GitWebhookTarget
from ragtime.git_webhooks.repository import git_webhook_repository
from ragtime.git_webhooks.service import git_webhook_service
from ragtime.git_webhooks.verification import WebhookAuthenticationError, parse_git_events, verify_webhook_request

router = APIRouter(prefix="/webhooks/git", tags=["git-webhooks"])

_MAX_WEBHOOK_BODY_BYTES = 1024 * 1024


class _BodyTooLargeError(Exception):
    pass


@router.post("/{webhook_id}", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("60/minute")
async def receive_git_webhook(
    webhook_id: str,
    request: Request,
    token: str | None = None,
) -> JSONResponse:
    target = await git_webhook_repository.resolve_target(webhook_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Webhook not found")
    try:
        body = await _read_limited_body(request, _MAX_WEBHOOK_BODY_BYTES)
    except _BodyTooLargeError:
        return JSONResponse(status_code=413, content={"detail": "Request body too large"})
    try:
        verify_webhook_request(target, request.headers, body, token)
    except WebhookAuthenticationError as exc:
        raise HTTPException(status_code=401, detail="Invalid webhook credentials") from exc
    payload = _decode_json_object(body)
    events = parse_git_events(target.provider, request.headers, payload)
    await _accept_matching_events(target, events)
    return JSONResponse(status_code=202, content={"status": "accepted"})


async def _read_limited_body(request: Request, limit: int) -> bytes:
    body_chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise _BodyTooLargeError()
        if chunk:
            body_chunks.append(chunk)
    return b"".join(body_chunks)


def _decode_json_object(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8") if body else "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Malformed JSON body") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Malformed JSON body")
    return payload


async def _accept_matching_events(target: GitWebhookTarget, events: list[GitPushEvent]) -> None:
    configured_branch = (target.branch or "main").strip() or "main"
    for event in events:
        if event.kind == "push" and (event.branch or "") == configured_branch:
            await git_webhook_service.accept_push(target, event)
            return
    for event in events:
        if event.kind != "push":
            await git_webhook_repository.record_ignored(
                target,
                event,
                event.message or "Webhook event ignored.",
            )
            return
    if events:
        event = events[0]
        await git_webhook_repository.record_ignored(
            target,
            event,
            f"Push to branch '{event.branch or 'unknown'}' ignored; configured branch is '{configured_branch}'.",
        )
