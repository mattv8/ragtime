"""HTTP adapter for the external development operation registry."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ragtime.userspace.development_access import DevelopmentPrincipal, resolve_development_principal
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


@router.post("/operations/{operation}")
async def execute_operation(
    workspace_id: str, operation: str, body: DevelopmentOperationRequest, principal: DevelopmentPrincipal = Depends(resolve_development_principal)
) -> Any:
    return await development_service.execute(principal, workspace_id, operation, body.arguments)
