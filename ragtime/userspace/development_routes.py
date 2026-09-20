"""HTTP adapter for the external development operation registry."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field

from ragtime.userspace.development_access import DevelopmentPrincipal, resolve_development_principal
from ragtime.userspace.development_bootstrap import build_bootstrap_manifest, get_bootstrap_artifact
from ragtime.userspace.development_service import development_service

router = APIRouter(prefix="/indexes/userspace/development/workspaces/{workspace_id}", tags=["User Space Development"])


class DevelopmentOperationRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


@router.get("/context")
async def get_context(workspace_id: str, principal: DevelopmentPrincipal = Depends(resolve_development_principal)) -> dict[str, Any]:
    return await development_service.execute(principal, workspace_id, "context", {})


@router.get("/operations")
async def get_operations(workspace_id: str, principal: DevelopmentPrincipal = Depends(resolve_development_principal)) -> dict[str, Any]:
    return {"operations": await development_service.list_authorized_operations(principal, workspace_id)}


@router.get("/bootstrap")
async def get_bootstrap(workspace_id: str, request: Request, principal: DevelopmentPrincipal = Depends(resolve_development_principal)) -> dict[str, Any]:
    """Discover generated setup assets after fresh workspace authorization."""
    await development_service._workspace(principal, workspace_id, "read")
    return build_bootstrap_manifest(origin=str(request.base_url).rstrip("/"), workspace_id=workspace_id, scopes=principal.scopes)


@router.get("/bootstrap/files/{artifact_id:path}")
async def get_bootstrap_file(
    workspace_id: str, artifact_id: str, request: Request, revision: str | None = None, principal: DevelopmentPrincipal = Depends(resolve_development_principal)
) -> Response:
    """Download one generated artifact; all calls reauthorize the credential."""
    await development_service._workspace(principal, workspace_id, "read")
    artifact = get_bootstrap_artifact(artifact_id=artifact_id, revision=revision, origin=str(request.base_url).rstrip("/"), workspace_id=workspace_id)
    return Response(
        content=artifact["content"].encode("utf-8"),
        media_type=artifact["content_type"],
        headers={"ETag": artifact["sha256"], "X-Ragtime-Guidance-Revision": artifact["guidance_revision"]},
    )


@router.post("/operations/{operation}")
async def execute_operation(
    workspace_id: str, operation: str, body: DevelopmentOperationRequest, principal: DevelopmentPrincipal = Depends(resolve_development_principal)
) -> Any:
    return await development_service.execute(principal, workspace_id, operation, body.arguments)
