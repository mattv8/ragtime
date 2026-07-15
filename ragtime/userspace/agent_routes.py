"""Workspace external-agent HTTP surface.

Public router (``/agent/w/{token}``): consumed by external chat agents via HTTP.
The token in the URL is a scoped, revocable workspace agent-access token;
every action executes as the minting user under normal workspace ACLs.

Management router (``/indexes/userspace/...``): session-authenticated
endpoints the workspace UI uses to enable, disable, and rotate the token.
"""

from __future__ import annotations

from typing import Any, Literal, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ragtime.core.auth import get_browser_matched_origin
from ragtime.core.security import get_current_user
from ragtime.userspace.agent_access import (
    disable_agent_access,
    enable_agent_access,
    get_agent_access_status,
    resolve_agent_access_token,
    rotate_agent_access_token,
)
from ragtime.userspace.agent_briefs import BuildBriefInput
from ragtime.userspace.agent_read_service import agent_read_service
from ragtime.userspace.build_task_service import build_task_service
from ragtime.userspace.planning_contract import build_recommended_workflow
from ragtime.userspace.planning_service import planning_service

agent_router = APIRouter(prefix="/agent/w", tags=["Workspace Agent Access"])
agent_management_router = APIRouter(prefix="/indexes/userspace", tags=["User Space"])

_NO_STORE = {"Cache-Control": "no-store"}
_MANIFEST_TEMPLATE = """# Ragtime Workspace Agent API

You are reading the collaboration manifest for the Ragtime workspace
"{workspace_name}". Ragtime is a dashboard/app builder; the workspace owner
invited you (an external agent) to help plan work and submit build tasks.

All endpoints share this base URL. The URL itself carries a scoped access
token: treat it as a secret, do not echo it into logs or summaries.

    {base_url}

## Endpoints

1. GET  {base_url}/context
   Curated JSON: workspace summary, architecture (framework, entrypoint),
   selected data tools, builder rules, recommended workflow. Read this first.

2. GET  {base_url}/files?prefix=&offset=0&limit=200
   Paginated JSON list of workspace files.

3. GET  {base_url}/file?path=<path>&start_line=1&max_lines=400
   Bounded text read of one file. Follow "next_start_line" to page through
   long files. Non-text files return HTTP 415.

4. GET  {base_url}/conversations?offset=0&limit=50
   Paginated JSON list of top-level workspace conversations. Returns only
   conversation metadata plus explicit subagent child IDs.

5. GET  {base_url}/conversations/{{conversation_id}}?direction=forward|backward&cursor=&limit=20
   Read sanitized transcript pages in chronological order. Cursor paging is
   exclusive: forward starts after the cursor, backward returns messages
   before the cursor. Only user/assistant text and tool inputs are exposed;
   no tool outputs, reasoning, system prompts, or internal metadata.

6. GET  {base_url}/code-search?query=<text>&mode=semantic|symbols|hybrid
   Search the hidden workspace code index only. Returns best-effort read-only
   results with index readiness status. This surface never writes state,
   submits tasks, or exposes tool output; agents must treat the response as
   input-only and produce no side effects.

7. POST {base_url}/tasks
   Submit a structured build brief. Ragtime's internal builder agent starts
   immediately in a new workspace conversation. JSON body:
   {{"idempotency_key": "<fresh unique string per new task>",
     "title": "Add revenue chart",
     "objective": "...",
     "requirements": ["..."],
     "acceptance_criteria": ["..."],
     "constraints": [], "preserve_paths": [], "data_component_ids": [],
     "non_goals": [], "context_revision": "<from /context, optional>"}}
   Response: {{"deduplicated": false, "conversation_id": "...",
   "task_id": "...", "status": "pending"}}. Retrying with the same
   idempotency_key returns the original task instead of duplicating it.

8. GET  {base_url}/tasks/{{task_id}}?max_result_chars=20000
   Poll task status: pending/running/completed/failed/cancelled/interrupted.
   When completed, "result" holds the builder's final message (possibly
   truncated). "possibly_stalled": true means no recent progress. Poll every
   10-30 seconds; long builds are normal.

9. POST {base_url}/tasks/{{task_id}}/reply   body: {{"idempotency_key": "<fresh unique string per new reply>", "message": "..."}}
   Continue the same conversation if the builder's result asks a question.
   Retrying with the same idempotency_key and reply payload returns the same
   follow-up task instead of duplicating it.

## How to work

{workflow}

Errors are JSON ({{"detail": ...}}) with meaningful HTTP status codes:
404 unknown token/file/task/conversation, 400 invalid brief/cursor/query,
403 not permitted, 409 idempotency conflict, 415 non-text file. Code-search
readiness states return HTTP 200 with status fields instead of task errors.
"""


class AgentTaskReplyRequest(BaseModel):
    """Reply payload for an existing build task conversation."""

    idempotency_key: str = Field(
        min_length=1,
        max_length=200,
        description="Fresh idempotency key for this reply attempt.",
    )
    message: str = Field(
        min_length=1,
        max_length=20000,
        description="Reply message for the builder conversation.",
    )


class EnableAgentAccessRequest(BaseModel):
    """Management request to enable external agent access."""

    allow_task_submission: bool = Field(
        default=True,
        description="Whether the token may start builder tasks while enabled.",
    )


def _agent_base_url(request: Request, token: str) -> str:
    return f"{get_browser_matched_origin(request)}/agent/w/{token}"


def _set_no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


def should_apply_agent_no_store(path: str) -> bool:
    normalized = str(path or "").strip()
    if normalized == "/agent/w" or normalized.startswith("/agent/w/"):
        return True
    parts = [part for part in normalized.split("/") if part]
    if len(parts) not in {5, 6}:
        return False
    if parts[0] != "indexes" or parts[1] != "userspace" or parts[2] != "workspaces":
        return False
    if parts[4] != "agent-access":
        return False
    return len(parts) == 5 or parts[5] in {"enable", "disable", "rotate"}


def _raise_http_exception(status_code: int, detail: str) -> None:
    raise HTTPException(status_code=status_code, detail=detail, headers=dict(_NO_STORE))


def _reraise_no_store(exc: HTTPException) -> NoReturn:
    headers = dict(exc.headers or {})
    headers.setdefault("Cache-Control", "no-store")
    raise HTTPException(status_code=exc.status_code, detail=exc.detail, headers=headers) from exc


def _with_agent_url(status: dict[str, Any], request: Request) -> dict[str, Any]:
    token = status.get("token")
    status["agent_url"] = _agent_base_url(request, str(token)) if token else None
    return status


@agent_router.get("/{token}", response_class=PlainTextResponse)
async def get_agent_manifest(token: str, request: Request) -> PlainTextResponse:
    try:
        ctx = await resolve_agent_access_token(token)
        from ragtime.userspace.service import userspace_service

        workspace = await userspace_service.enforce_workspace_role(
            ctx.workspace_id,
            ctx.acting_user_id,
            "viewer",
            is_admin=ctx.acting_user_is_admin,
        )
    except HTTPException as exc:
        _reraise_no_store(exc)
    workflow = "\n".join(f"{index}. {line}" for index, line in enumerate(build_recommended_workflow(), start=1))
    text = _MANIFEST_TEMPLATE.format(
        workspace_name=str(getattr(workspace, "name", "Workspace")),
        base_url=_agent_base_url(request, token),
        workflow=workflow,
    )
    if not ctx.allow_task_submission:
        text += "\nNOTE: Task submission is disabled for this token; /tasks and /tasks/{task_id}/reply return 403.\n"
    return PlainTextResponse(text, media_type="text/markdown", headers=dict(_NO_STORE))


@agent_router.get("/{token}/context")
async def get_agent_context(token: str, response: Response) -> dict[str, Any]:
    _set_no_store(response)
    try:
        ctx = await resolve_agent_access_token(token)
        return await planning_service.get_workspace_context(ctx.workspace_id, ctx.acting_user_id)
    except HTTPException as exc:
        _reraise_no_store(exc)


@agent_router.get("/{token}/files")
async def list_agent_files(
    token: str,
    response: Response,
    prefix: str = Query(default="", max_length=500),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
) -> dict[str, Any]:
    _set_no_store(response)
    try:
        ctx = await resolve_agent_access_token(token)
        return await planning_service.list_files(
            ctx.workspace_id,
            ctx.acting_user_id,
            prefix=prefix,
            offset=offset,
            limit=limit,
        )
    except HTTPException as exc:
        _reraise_no_store(exc)


@agent_router.get("/{token}/file")
async def read_agent_file(
    token: str,
    response: Response,
    path: str = Query(min_length=1, max_length=1000),
    start_line: int = Query(default=1, ge=1),
    max_lines: int = Query(default=400, ge=1, le=2000),
    max_chars: int = Query(default=30000, ge=200, le=100000),
) -> dict[str, Any]:
    _set_no_store(response)
    try:
        ctx = await resolve_agent_access_token(token)
        return await planning_service.read_file(
            ctx.workspace_id,
            ctx.acting_user_id,
            path,
            start_line=start_line,
            max_lines=max_lines,
            max_chars=max_chars,
        )
    except HTTPException as exc:
        if exc.status_code == 415:
            _raise_http_exception(415, "file_not_text")
        _reraise_no_store(exc)


@agent_router.get("/{token}/conversations")
async def list_agent_conversations(
    token: str,
    response: Response,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    _set_no_store(response)
    try:
        ctx = await resolve_agent_access_token(token)
        return await agent_read_service.list_conversations(
            ctx.workspace_id,
            ctx.acting_user_id,
            is_admin=ctx.acting_user_is_admin,
            offset=offset,
            limit=limit,
        )
    except HTTPException as exc:
        _reraise_no_store(exc)


@agent_router.get("/{token}/conversations/{conversation_id}")
async def get_agent_conversation(
    token: str,
    conversation_id: str,
    response: Response,
    direction: Literal["forward", "backward"],
    cursor: int | None = Query(default=None, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    _set_no_store(response)
    try:
        ctx = await resolve_agent_access_token(token)
        return await agent_read_service.get_conversation_transcript(
            ctx.workspace_id,
            ctx.acting_user_id,
            conversation_id,
            is_admin=ctx.acting_user_is_admin,
            direction=direction,
            cursor=cursor,
            limit=limit,
        )
    except HTTPException as exc:
        _reraise_no_store(exc)


@agent_router.get("/{token}/code-search")
async def search_agent_workspace_code(
    token: str,
    response: Response,
    query: str = Query(min_length=1, max_length=2000),
    mode: Literal["semantic", "symbols", "hybrid"] = "hybrid",
    max_results: int = Query(default=8, ge=1, le=25),
    max_chars_per_result: int = Query(default=1200, ge=200, le=6000),
) -> dict[str, Any]:
    _set_no_store(response)
    try:
        ctx = await resolve_agent_access_token(token)
        return await agent_read_service.search_code(
            ctx.workspace_id,
            ctx.acting_user_id,
            is_admin=ctx.acting_user_is_admin,
            query=query,
            mode=mode,
            max_results=max_results,
            max_chars_per_result=max_chars_per_result,
        )
    except HTTPException as exc:
        _reraise_no_store(exc)


@agent_router.post("/{token}/tasks")
async def submit_agent_task(token: str, brief: BuildBriefInput, response: Response) -> dict[str, Any]:
    _set_no_store(response)
    try:
        ctx = await resolve_agent_access_token(token)
        if not ctx.allow_task_submission:
            _raise_http_exception(403, "Task submission is disabled for this agent access token")
        return await build_task_service.start_build_task(ctx.workspace_id, ctx.acting_user_id, brief)
    except HTTPException as exc:
        _reraise_no_store(exc)


@agent_router.get("/{token}/tasks/{task_id}")
async def get_agent_task(
    token: str,
    task_id: str,
    response: Response,
    max_result_chars: int = Query(default=20000, ge=1000, le=100000),
) -> dict[str, Any]:
    _set_no_store(response)
    try:
        ctx = await resolve_agent_access_token(token)
        return await build_task_service.get_build_task(
            ctx.workspace_id,
            ctx.acting_user_id,
            task_id,
            max_result_chars=max_result_chars,
        )
    except HTTPException as exc:
        _reraise_no_store(exc)


@agent_router.post("/{token}/tasks/{task_id}/reply")
async def reply_agent_task(
    token: str,
    task_id: str,
    body: AgentTaskReplyRequest,
    response: Response,
) -> dict[str, Any]:
    _set_no_store(response)
    try:
        ctx = await resolve_agent_access_token(token)
        if not ctx.allow_task_submission:
            _raise_http_exception(403, "Task submission is disabled for this agent access token")
        return await build_task_service.reply_to_build_task(
            ctx.workspace_id,
            ctx.acting_user_id,
            task_id,
            body.message,
            body.idempotency_key,
        )
    except HTTPException as exc:
        _reraise_no_store(exc)


@agent_management_router.get("/workspaces/{workspace_id}/agent-access")
async def get_workspace_agent_access(
    workspace_id: str,
    request: Request,
    response: Response,
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    _set_no_store(response)
    try:
        return _with_agent_url(await get_agent_access_status(workspace_id, user.id), request)
    except HTTPException as exc:
        _reraise_no_store(exc)


@agent_management_router.post("/workspaces/{workspace_id}/agent-access/enable")
async def enable_workspace_agent_access(
    workspace_id: str,
    request: Request,
    response: Response,
    body: EnableAgentAccessRequest | None = None,
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    _set_no_store(response)
    try:
        allow = body.allow_task_submission if body is not None else True
        return _with_agent_url(
            await enable_agent_access(workspace_id, user.id, allow_task_submission=allow),
            request,
        )
    except HTTPException as exc:
        _reraise_no_store(exc)


@agent_management_router.post("/workspaces/{workspace_id}/agent-access/disable")
async def disable_workspace_agent_access(
    workspace_id: str,
    request: Request,
    response: Response,
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    _set_no_store(response)
    try:
        return _with_agent_url(await disable_agent_access(workspace_id, user.id), request)
    except HTTPException as exc:
        _reraise_no_store(exc)


@agent_management_router.post("/workspaces/{workspace_id}/agent-access/rotate")
async def rotate_workspace_agent_access(
    workspace_id: str,
    request: Request,
    response: Response,
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    _set_no_store(response)
    try:
        return _with_agent_url(await rotate_agent_access_token(workspace_id, user.id), request)
    except HTTPException as exc:
        _reraise_no_store(exc)
