from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response

from ragtime.core.auth import validate_session
from ragtime.core.database import get_db
from ragtime.core.security import get_current_user, get_current_user_optional
from ragtime.userspace.models import (
    UserSpaceCapabilityTokenResponse,
    UserSpaceFileResponse,
    UserSpaceRuntimeActionResponse,
    UserSpaceRuntimeSessionResponse,
    UserSpaceRuntimeStatusResponse,
)
from ragtime.userspace.runtime_service import (
    RuntimeVersionConflictError,
    userspace_runtime_service,
)
from ragtime.userspace.service import userspace_service

router = APIRouter(prefix="/indexes/userspace", tags=["User Space Runtime"])


_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


async def _broadcast_collab_message(
    workspace_id: str,
    file_path: str,
    message: dict[str, Any],
) -> None:
    clients = await userspace_runtime_service.get_collab_clients(
        workspace_id, file_path
    )
    payload = json.dumps(message)
    for client in clients:
        with contextlib.suppress(Exception):
            await client.send_text(payload)


def _proxy_request_headers(request: Request) -> dict[str, str]:
    hop_by_hop = {
        "host",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
    }
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in hop_by_hop
    }


def _proxy_response_headers(headers: httpx.Headers) -> dict[str, str]:
    blocked = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
    return {key: value for key, value in headers.items() if key.lower() not in blocked}


def _to_websocket_url(http_url: str) -> str:
    parsed = urlsplit(http_url)
    if parsed.scheme == "https":
        scheme = "wss"
    elif parsed.scheme == "http":
        scheme = "ws"
    else:
        scheme = parsed.scheme
    return urlunsplit(
        (scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment)
    )


async def _proxy_websocket_request(websocket: WebSocket, upstream_url: str) -> None:
    try:
        websockets_module = importlib.import_module("websockets")
    except Exception:
        await websocket.close(code=1011)
        return

    await websocket.accept()
    try:
        async with websockets_module.connect(upstream_url, max_size=None) as upstream:

            async def downstream_to_upstream() -> None:
                while True:
                    message = await websocket.receive()
                    message_type = message.get("type")
                    if message_type == "websocket.disconnect":
                        break
                    text = message.get("text")
                    data = message.get("bytes")
                    if text is not None:
                        await upstream.send(text)
                    elif data is not None:
                        await upstream.send(data)

            async def upstream_to_downstream() -> None:
                while True:
                    message = await upstream.recv()
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(str(message))

            down_task = asyncio.create_task(downstream_to_upstream())
            up_task = asyncio.create_task(upstream_to_downstream())
            done, pending = await asyncio.wait(
                {down_task, up_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1011)


async def _proxy_http_request(request: Request, upstream_url: str) -> Response:
    if request.headers.get("upgrade", "").lower() == "websocket":
        raise HTTPException(
            status_code=501,
            detail="WebSocket proxy upgrades are not available on this route",
        )

    body = await request.body()
    timeout = httpx.Timeout(connect=2.0, read=30.0, write=30.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        try:
            upstream_response = await client.request(
                method=request.method,
                url=upstream_url,
                content=body if body else None,
                headers=_proxy_request_headers(request),
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Runtime preview upstream unavailable: {exc}",
            ) from exc

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_proxy_response_headers(upstream_response.headers),
        media_type=upstream_response.headers.get("content-type"),
    )


async def _websocket_user(websocket: WebSocket) -> Any | None:
    session_token = websocket.cookies.get("ragtime_session")
    if not session_token:
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            session_token = auth_header[7:]

    if not session_token:
        return None

    token_data = await validate_session(session_token)
    if not token_data:
        return None

    db = await get_db()
    return await db.user.find_unique(where={"id": token_data.user_id})


@router.get(
    "/runtime/workspaces/{workspace_id}/session",
    response_model=UserSpaceRuntimeSessionResponse,
)
async def get_runtime_session(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.get_runtime_session(workspace_id, user.id)


@router.post(
    "/runtime/workspaces/{workspace_id}/session/start",
    response_model=UserSpaceRuntimeActionResponse,
)
async def start_runtime_session(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.start_runtime_session(workspace_id, user.id)


@router.post(
    "/runtime/workspaces/{workspace_id}/session/stop",
    response_model=UserSpaceRuntimeActionResponse,
)
async def stop_runtime_session(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.stop_runtime_session(workspace_id, user.id)


@router.get(
    "/runtime/workspaces/{workspace_id}/devserver/status",
    response_model=UserSpaceRuntimeStatusResponse,
)
async def get_devserver_status(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.get_devserver_status(workspace_id, user.id)


@router.post(
    "/runtime/workspaces/{workspace_id}/devserver/start",
    response_model=UserSpaceRuntimeActionResponse,
)
async def start_devserver(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.start_runtime_session(workspace_id, user.id)


@router.post(
    "/runtime/workspaces/{workspace_id}/devserver/restart",
    response_model=UserSpaceRuntimeActionResponse,
)
async def restart_devserver(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.restart_devserver(workspace_id, user.id)


@router.post(
    "/runtime/workspaces/{workspace_id}/capability-token",
    response_model=UserSpaceCapabilityTokenResponse,
)
async def issue_capability_token(
    workspace_id: str,
    payload: dict[str, Any],
    user: Any = Depends(get_current_user),
):
    capabilities = payload.get("capabilities") if isinstance(payload, dict) else []
    if not isinstance(capabilities, list):
        capabilities = []
    session_id = payload.get("session_id") if isinstance(payload, dict) else None
    return await userspace_runtime_service.issue_capability_token(
        workspace_id,
        user.id,
        [str(value) for value in capabilities],
        str(session_id) if session_id else None,
    )


@router.get(
    "/runtime/workspaces/{workspace_id}/fs/{file_path:path}",
    response_model=UserSpaceFileResponse,
)
async def runtime_fs_read(
    workspace_id: str,
    file_path: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.runtime_fs_read(
        workspace_id,
        file_path,
        user.id,
    )


@router.put(
    "/runtime/workspaces/{workspace_id}/fs/{file_path:path}",
    response_model=UserSpaceFileResponse,
)
async def runtime_fs_write(
    workspace_id: str,
    file_path: str,
    payload: dict[str, Any],
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.runtime_fs_write(
        workspace_id,
        file_path,
        str(payload.get("content", "")),
        user.id,
    )


@router.delete("/runtime/workspaces/{workspace_id}/fs/{file_path:path}")
async def runtime_fs_delete(
    workspace_id: str,
    file_path: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_runtime_service.runtime_fs_delete(
        workspace_id,
        file_path,
        user.id,
    )


@router.websocket("/runtime/workspaces/{workspace_id}/pty")
async def runtime_pty(workspace_id: str, websocket: WebSocket):
    user = await _websocket_user(websocket)
    if not user:
        await websocket.close(code=4401)
        return

    try:
        await userspace_service.enforce_workspace_role(workspace_id, user.id, "viewer")
    except HTTPException:
        await websocket.close(code=4403)
        return

    can_write = True
    try:
        await userspace_service.enforce_workspace_role(workspace_id, user.id, "editor")
    except HTTPException:
        can_write = False

    if not can_write:
        await websocket.accept()
        await websocket.send_text(
            json.dumps(
                {
                    "type": "status",
                    "read_only": True,
                    "message": "Read-only terminal session",
                }
            )
        )
        await websocket.close(code=1000)
        return

    upstream_ws_url = (
        await userspace_runtime_service.build_workspace_pty_upstream_ws_url(
            workspace_id,
            user.id,
        )
    )
    await _proxy_websocket_request(websocket, upstream_ws_url)


@router.websocket("/collab/workspaces/{workspace_id}/files/{file_path:path}")
async def collab_file_socket(workspace_id: str, file_path: str, websocket: WebSocket):
    user = await _websocket_user(websocket)
    if not user:
        await websocket.close(code=4401)
        return

    try:
        snapshot = await userspace_runtime_service.get_collab_snapshot(
            workspace_id,
            file_path,
            user.id,
        )
    except HTTPException as exc:
        await websocket.close(code=4403 if exc.status_code == 403 else 4404)
        return

    can_edit = not snapshot.read_only

    await websocket.accept()
    await userspace_runtime_service.register_collab_client(
        workspace_id, snapshot.file_path, websocket, user.id
    )
    await websocket.send_text(
        json.dumps(
            {
                "type": "snapshot",
                "workspace_id": workspace_id,
                "file_path": snapshot.file_path,
                "version": snapshot.version,
                "content": snapshot.content,
                "read_only": snapshot.read_only,
            }
        )
    )
    current_presence = await userspace_runtime_service.get_collab_presence(
        workspace_id,
        snapshot.file_path,
    )
    await websocket.send_text(
        json.dumps(
            {
                "type": "presence",
                "workspace_id": workspace_id,
                "file_path": snapshot.file_path,
                "users": current_presence,
            }
        )
    )

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message_type = payload.get("type")
            if message_type == "presence":
                users = await userspace_runtime_service.update_collab_presence(
                    workspace_id,
                    snapshot.file_path,
                    user.id,
                    payload if isinstance(payload, dict) else {},
                )
                await _broadcast_collab_message(
                    workspace_id,
                    snapshot.file_path,
                    {
                        "type": "presence",
                        "workspace_id": workspace_id,
                        "file_path": snapshot.file_path,
                        "users": users,
                    },
                )
                continue
            if message_type != "update":
                continue
            if not can_edit:
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "message": "Read-only collaboration session"}
                    )
                )
                continue

            content = str(payload.get("content", ""))
            client_version_raw = payload.get("version")
            client_version = (
                int(client_version_raw)
                if isinstance(client_version_raw, (int, float, str))
                and str(client_version_raw).strip().lstrip("-").isdigit()
                else None
            )
            try:
                updated = await userspace_runtime_service.apply_collab_update(
                    workspace_id,
                    snapshot.file_path,
                    content,
                    user.id,
                    expected_version=client_version,
                )
            except RuntimeVersionConflictError as conflict:
                latest = await userspace_runtime_service.get_collab_snapshot(
                    workspace_id,
                    snapshot.file_path,
                    user.id,
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": (
                                f"Version conflict: expected {conflict.expected_version}, "
                                f"current {conflict.actual_version}"
                            ),
                        }
                    )
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "snapshot",
                            "workspace_id": latest.workspace_id,
                            "file_path": latest.file_path,
                            "version": latest.version,
                            "content": latest.content,
                            "read_only": latest.read_only,
                        }
                    )
                )
                continue
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "ack",
                        "workspace_id": updated.workspace_id,
                        "file_path": updated.file_path,
                        "version": updated.version,
                    }
                )
            )

    except WebSocketDisconnect:
        users = await userspace_runtime_service.clear_collab_presence(
            workspace_id,
            snapshot.file_path,
            user.id,
        )
        await userspace_runtime_service.unregister_collab_client(
            workspace_id,
            snapshot.file_path,
            websocket,
        )
        await _broadcast_collab_message(
            workspace_id,
            snapshot.file_path,
            {
                "type": "presence",
                "workspace_id": workspace_id,
                "file_path": snapshot.file_path,
                "users": users,
            },
        )


@router.post("/collab/workspaces/{workspace_id}/files/create")
async def collab_create_file(
    workspace_id: str,
    payload: dict[str, Any],
    user: Any = Depends(get_current_user),
):
    file_path = str(payload.get("file_path", "")).strip()
    content = str(payload.get("content", ""))
    created = await userspace_runtime_service.create_collab_file(
        workspace_id,
        file_path,
        user.id,
        content=content,
    )
    await _broadcast_collab_message(
        workspace_id,
        created.file_path,
        {
            "type": "file_created",
            "workspace_id": workspace_id,
            "file_path": created.file_path,
            "version": created.version,
        },
    )
    return {
        "success": True,
        "workspace_id": workspace_id,
        "file_path": created.file_path,
        "version": created.version,
    }


@router.post("/collab/workspaces/{workspace_id}/files/rename")
async def collab_rename_file(
    workspace_id: str,
    payload: dict[str, Any],
    user: Any = Depends(get_current_user),
):
    old_path = str(payload.get("old_path", "")).strip()
    new_path = str(payload.get("new_path", "")).strip()
    result = await userspace_runtime_service.rename_collab_file(
        workspace_id,
        old_path,
        new_path,
        user.id,
    )
    await _broadcast_collab_message(
        workspace_id,
        result["new_path"],
        {
            "type": "file_renamed",
            "workspace_id": workspace_id,
            "old_path": result["old_path"],
            "new_path": result["new_path"],
        },
    )
    return result


@router.post("/collab/workspaces/{workspace_id}/files/delete")
async def collab_delete_file(
    workspace_id: str,
    payload: dict[str, Any],
    user: Any = Depends(get_current_user),
):
    file_path = str(payload.get("file_path", "")).strip()
    result = await userspace_runtime_service.delete_collab_file(
        workspace_id,
        file_path,
        user.id,
    )
    return result


@router.api_route("/workspaces/{workspace_id}/preview", methods=_PROXY_METHODS)
@router.api_route(
    "/workspaces/{workspace_id}/preview/{path:path}", methods=_PROXY_METHODS
)
async def workspace_preview_proxy(
    workspace_id: str,
    request: Request,
    path: str = "",
    user: Any = Depends(get_current_user),
):
    upstream_url = await userspace_runtime_service.build_workspace_preview_upstream_url(
        workspace_id,
        user.id,
        path,
        query=request.url.query or None,
    )
    return await _proxy_http_request(request, upstream_url)


@router.api_route("/shared/{owner_username}/{share_slug}", methods=_PROXY_METHODS)
@router.api_route(
    "/shared/{owner_username}/{share_slug}/{path:path}", methods=_PROXY_METHODS
)
async def shared_preview_proxy(
    owner_username: str,
    share_slug: str,
    request: Request,
    path: str = "",
    share_password: str | None = Header(
        default=None, alias="X-UserSpace-Share-Password"
    ),
    user: Any | None = Depends(get_current_user_optional),
):
    preview = await userspace_service.get_shared_preview_by_slug(
        owner_username,
        share_slug,
        current_user=user,
        password=share_password,
    )
    upstream_url = await userspace_runtime_service.build_shared_preview_upstream_url(
        preview.workspace_id,
        path,
        query=request.url.query or None,
    )
    return await _proxy_http_request(request, upstream_url)


@router.api_route("/shared/{share_token}/preview", methods=_PROXY_METHODS)
@router.api_route("/shared/{share_token}/preview/{path:path}", methods=_PROXY_METHODS)
async def shared_token_preview_proxy(
    share_token: str,
    request: Request,
    path: str = "",
    share_password: str | None = Header(
        default=None, alias="X-UserSpace-Share-Password"
    ),
    user: Any | None = Depends(get_current_user_optional),
):
    preview = await userspace_service.get_shared_preview(
        share_token,
        current_user=user,
        password=share_password,
    )
    upstream_url = await userspace_runtime_service.build_shared_preview_upstream_url(
        preview.workspace_id,
        path,
        query=request.url.query or None,
    )
    return await _proxy_http_request(request, upstream_url)


@router.websocket("/workspaces/{workspace_id}/preview")
@router.websocket("/workspaces/{workspace_id}/preview/{path:path}")
async def workspace_preview_proxy_websocket(
    workspace_id: str,
    websocket: WebSocket,
    path: str = "",
):
    user = await _websocket_user(websocket)
    if not user:
        await websocket.close(code=4401)
        return

    try:
        await userspace_service.enforce_workspace_role(workspace_id, user.id, "viewer")
    except HTTPException:
        await websocket.close(code=4403)
        return

    upstream_url = await userspace_runtime_service.build_workspace_preview_upstream_url(
        workspace_id,
        user.id,
        path,
        query=websocket.url.query or None,
    )
    await _proxy_websocket_request(websocket, _to_websocket_url(upstream_url))


@router.websocket("/shared/{owner_username}/{share_slug}")
@router.websocket("/shared/{owner_username}/{share_slug}/{path:path}")
async def shared_preview_proxy_websocket(
    owner_username: str,
    share_slug: str,
    websocket: WebSocket,
    path: str = "",
):
    user = await _websocket_user(websocket)
    share_password = websocket.headers.get("x-userspace-share-password")

    try:
        preview = await userspace_service.get_shared_preview_by_slug(
            owner_username,
            share_slug,
            current_user=user,
            password=share_password,
        )
    except HTTPException as exc:
        await websocket.close(code=4403 if exc.status_code == 403 else 4404)
        return

    upstream_url = await userspace_runtime_service.build_shared_preview_upstream_url(
        preview.workspace_id,
        path,
        query=websocket.url.query or None,
    )
    await _proxy_websocket_request(websocket, _to_websocket_url(upstream_url))


@router.websocket("/shared/{share_token}/preview")
@router.websocket("/shared/{share_token}/preview/{path:path}")
async def shared_token_preview_proxy_websocket(
    share_token: str,
    websocket: WebSocket,
    path: str = "",
):
    user = await _websocket_user(websocket)
    share_password = websocket.headers.get("x-userspace-share-password")

    try:
        preview = await userspace_service.get_shared_preview(
            share_token,
            current_user=user,
            password=share_password,
        )
    except HTTPException as exc:
        await websocket.close(code=4403 if exc.status_code == 403 else 4404)
        return

    upstream_url = await userspace_runtime_service.build_shared_preview_upstream_url(
        preview.workspace_id,
        path,
        query=websocket.url.query or None,
    )
    await _proxy_websocket_request(websocket, _to_websocket_url(upstream_url))
