"""Authenticated pairing management plus narrowly scoped public enrollment APIs."""

import asyncio
import base64
import json
from datetime import datetime, timezone
from importlib.resources import files
from typing import Annotated, Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, WebSocket
from pydantic import ValidationError

from ragtime.core.auth import get_external_origin
from ragtime.core.rate_limit import limiter
from ragtime.core.security import require_admin
from ragtime.indexer.repository import repository as tool_repository
from ragtime.remote_client.models import EnrollmentCompleteRequest, OfflineCompleteRequest, PairingCreatedResponse, PairingCreateRequest, PairingResponse
from ragtime.remote_client.repository import remote_pairing_repository
from ragtime.remote_client.service import remote_pairing_service

router = APIRouter(prefix="/indexes/remote", tags=["remote-client"])
_MAX_ENROLLMENT_BODY_BYTES = 64 * 1024

_SCRIPTS = files("ragtime.remote_client.scripts")
_INSTALL_SH = _SCRIPTS.joinpath("install.sh").read_bytes()
_INSTALL_PS1 = _SCRIPTS.joinpath("install.ps1").read_bytes()
_TUNNEL_CLIENT_SH = _SCRIPTS.joinpath("tunnel-client.py").read_bytes()
_TUNNEL_CLIENT_PS1 = _SCRIPTS.joinpath("tunnel-client.ps1").read_bytes()


def _bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer enrollment token required")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Bearer enrollment token required")
    return token


def _require_bounded_body(request: Request) -> None:
    try:
        content_length = int(request.headers.get("content-length", "0"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
    if content_length > _MAX_ENROLLMENT_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Enrollment request body too large")


def _ws_url(request: Request) -> str:
    origin = get_external_origin(request)
    parts = urlsplit(origin)
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, "/indexes/remote/tunnel", "", ""))


def _commands(request: Request, token: str, pairing: dict[str, Any]) -> dict[str, str]:
    origin = get_external_origin(request)
    return {
        "sh": f"curl --fail --location '{origin}/indexes/remote/install.sh' | sh -s -- --server-url '{origin}' --token '{token}'",
        "powershell": f"$p=Join-Path $env:TEMP ragtime-install.ps1; iwr -UseBasicParsing '{origin}/indexes/remote/install.ps1' -OutFile $p; powershell.exe -NoProfile -ExecutionPolicy Bypass -File $p -ServerUrl '{origin}' -Token '{token}'",
        "offline_sh": f"sh ./install.sh --mode offline --pairing-id '{pairing['id']}' --public-key '{pairing['publicKey']}'",
        "offline_powershell": f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\install.ps1 -Mode offline -PairingId '{pairing['id']}' -PublicKey '{pairing['publicKey']}'",
    }


async def _response(pairing: dict[str, Any]) -> PairingResponse:
    status = str(getattr(pairing["status"], "value", pairing["status"]))
    if status == "pending" and pairing["enrollmentExpiresAt"] <= datetime.now(timezone.utc):
        status = "expired"
    tool = await tool_repository.get_tool_config(str(pairing["toolId"]))
    secret_fields = {"key_content", "key_path", "key_passphrase", "password", "tunnel_credential"}
    connection_config = {key: value for key, value in (tool.connection_config if tool else {}).items() if key not in secret_fields}
    return PairingResponse(
        id=str(pairing["id"]),
        tool_id=str(pairing["toolId"]),
        status=status,
        expires_at=pairing["enrollmentExpiresAt"],
        facts=pairing.get("facts"),
        transport=pairing.get("transport"),
        error=pairing.get("error"),
        connection_config=connection_config,
    )


@router.post("/pairings", response_model=PairingCreatedResponse)
async def create_pairing(payload: PairingCreateRequest, request: Request, _user: Any = Depends(require_admin)) -> PairingCreatedResponse:
    try:
        pairing, token, _tunnel_credential = await remote_pairing_service.create(payload.tool_type, payload.tool_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    base = await _response(pairing)
    return PairingCreatedResponse(**base.model_dump(), public_key=pairing["publicKey"], commands=_commands(request, token, pairing))


@router.get("/pairings/{pairing_id}", response_model=PairingResponse)
async def get_pairing(pairing_id: str, _user: Any = Depends(require_admin)) -> PairingResponse:
    pairing = await remote_pairing_repository.get(pairing_id)
    if not pairing:
        raise HTTPException(status_code=404, detail="Pairing not found")
    await remote_pairing_repository.touch(pairing_id)
    return await _response(pairing)


@router.post("/pairings/{pairing_id}/regenerate", response_model=PairingCreatedResponse)
async def regenerate_pairing(pairing_id: str, request: Request, _user: Any = Depends(require_admin)) -> PairingCreatedResponse:
    regenerated = await remote_pairing_service.regenerate(pairing_id)
    if not regenerated:
        raise HTTPException(status_code=404, detail="Pairing not found")
    pairing, token, _tunnel_credential = regenerated
    base = await _response(pairing)
    return PairingCreatedResponse(**base.model_dump(), public_key=pairing["publicKey"], commands=_commands(request, token, pairing))


@router.delete("/pairings/{pairing_id}", status_code=204)
async def revoke_pairing(pairing_id: str, _user: Any = Depends(require_admin)) -> Response:
    pairing = await remote_pairing_repository.revoke(pairing_id)
    if not pairing:
        raise HTTPException(status_code=404, detail="Pairing not found")
    from ragtime.remote_client.relay import relay_manager

    for revoked_pairing_id in pairing.pop("revokedPairingIds", [pairing_id]):
        await relay_manager.revoke(revoked_pairing_id)
    return Response(status_code=204)


@router.post("/pairings/{pairing_id}/complete-offline", response_model=PairingResponse)
async def complete_offline(pairing_id: str, payload: OfflineCompleteRequest, _user: Any = Depends(require_admin)) -> PairingResponse:
    try:
        decoded = base64.b64decode(payload.blob, validate=True)
        data = json.loads(decoded)
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Invalid offline enrollment blob") from None
    if not isinstance(data, dict) or str(data.get("pairing_id")) != pairing_id:
        raise HTTPException(status_code=400, detail="Offline enrollment blob does not match pairing")
    pairing = await remote_pairing_repository.get(pairing_id)
    if not pairing:
        raise HTTPException(status_code=404, detail="Pairing not found")
    if data.get("public_key") != pairing["publicKey"]:
        raise HTTPException(status_code=409, detail="Offline enrollment key is stale")
    offline_nonce = data.get("nonce")
    if not isinstance(offline_nonce, str) or not offline_nonce:
        raise HTTPException(status_code=400, detail="Offline enrollment nonce is required")
    try:
        facts = EnrollmentCompleteRequest(nonce=offline_nonce, facts=data.get("facts"), mode="direct").facts.model_dump()
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="Invalid offline enrollment facts") from exc
    result = await remote_pairing_service.complete_offline(pairing_id, offline_nonce, facts)
    if not result:
        raise HTTPException(status_code=409, detail="Offline enrollment was rejected")
    return await _response(result)


@router.get("/enroll/material")
@limiter.limit("60/minute")
async def enrollment_material(request: Request, authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    token = _bearer_token(authorization)
    pairing = await remote_pairing_repository.get_by_token(token)
    status = str(getattr(pairing["status"], "value", pairing["status"])) if pairing else ""
    recovery_active = (
        pairing and pairing.get("enrollmentNonce") and pairing.get("recoveryExpiresAt") and pairing["recoveryExpiresAt"] > datetime.now(timezone.utc)
    )
    if not pairing or (status != "pending" and not recovery_active):
        raise HTTPException(status_code=404, detail="Enrollment not available")
    return {
        "pairing_id": pairing["id"],
        "public_key": pairing["publicKey"],
        "ws_url": _ws_url(request),
        "artifact_hashes": {
            "install.sh": remote_pairing_service.artifact_hash(_INSTALL_SH),
            "install.ps1": remote_pairing_service.artifact_hash(_INSTALL_PS1),
            "tunnel-client.py": remote_pairing_service.artifact_hash(_TUNNEL_CLIENT_SH),
            "tunnel-client.ps1": remote_pairing_service.artifact_hash(_TUNNEL_CLIENT_PS1),
        },
        "tunnel_client_sha256": remote_pairing_service.artifact_hash(_TUNNEL_CLIENT_SH),
        "tunnel_client_url": f"{get_external_origin(request)}/indexes/remote/tunnel-client.py",
        "tunnel_client_powershell_url": f"{get_external_origin(request)}/indexes/remote/tunnel-client.ps1",
    }


@router.post("/enroll/complete")
@limiter.limit("10/minute")
async def enrollment_complete(request: Request, payload: EnrollmentCompleteRequest, authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    _require_bounded_body(request)
    result = await remote_pairing_service.complete(_bearer_token(authorization), payload.nonce, payload.facts.model_dump(), payload.mode)
    if not result:
        raise HTTPException(status_code=409, detail="Enrollment token expired, revoked, or already used")
    response = {"pairing_id": result["id"], "status": result["status"], "transport": result.get("transport")}
    if result.get("transport") == "tunnel":
        response["credential"] = await remote_pairing_repository.enrollment_credential(str(result["id"]))
    return response


@router.websocket("/tunnel")
async def remote_tunnel(websocket: WebSocket) -> None:
    """Authenticate a tunnel credential before handing the socket to the relay."""
    authorization = websocket.headers.get("authorization")
    try:
        credential = _bearer_token(authorization)
    except HTTPException:
        await websocket.close(code=4401)
        return
    pairing = await remote_pairing_repository.get_by_tunnel_credential(credential)
    if not pairing or str(getattr(pairing["status"], "value", pairing["status"])) == "revoked":
        await websocket.close(code=4403)
        return
    if websocket.headers.get("sec-websocket-protocol", "") != "ragtime-tunnel-v1":
        await websocket.close(code=4406)
        return
    from ragtime.remote_client.relay import relay_manager

    await websocket.accept(subprotocol="ragtime-tunnel-v1")
    pairing_id = str(pairing["id"])
    relay_task = asyncio.create_task(relay_manager.attach(pairing_id, websocket))
    try:
        for _ in range(20):
            if relay_manager.endpoint_for(pairing_id):
                await remote_pairing_service.verify_tunnel(pairing_id)
                break
            await asyncio.sleep(0.05)
        await relay_task
    finally:
        if not relay_task.done():
            relay_task.cancel()


@router.get("/install.sh", include_in_schema=False)
async def install_sh() -> Response:
    return Response(_INSTALL_SH, media_type="text/x-shellscript", headers={"Cache-Control": "public, max-age=300"})


@router.get("/install.ps1", include_in_schema=False)
async def install_ps1() -> Response:
    return Response(_INSTALL_PS1, media_type="text/plain", headers={"Cache-Control": "public, max-age=300"})


@router.get("/tunnel-client.py", include_in_schema=False)
async def tunnel_client_python() -> Response:
    return Response(_TUNNEL_CLIENT_SH, media_type="text/x-python", headers={"Cache-Control": "public, max-age=300"})


@router.get("/tunnel-client.ps1", include_in_schema=False)
async def tunnel_client_powershell() -> Response:
    return Response(_TUNNEL_CLIENT_PS1, media_type="text/plain", headers={"Cache-Control": "public, max-age=300"})
