"""Administrative SolidWorks host management endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from ragtime.core.security import require_admin
from ragtime.indexer.models import ToolType
from ragtime.indexer.repository import repository
from ragtime.solidworks.host_artifact import get_host_artifact, verify_host_artifact
from ragtime.solidworks.host_installer import HostArtifactUnavailableError
from ragtime.solidworks.pdm_link import normalize_vault_folder
from ragtime.solidworks.pdm_rights import PdmToolNotFound, pdm_rights_service
from ragtime.solidworks.service import solidworks_service

router = APIRouter(prefix="/indexes/solidworks", tags=["SolidWorks"], dependencies=[Depends(require_admin)])


class PdmRightsRequest(BaseModel):
    pdm_tool_id: str
    folders: list[str] = Field(max_length=25)

    @field_validator("folders")
    @classmethod
    def validate_folders(cls, folders: list[str]) -> list[str]:
        return [normalize_vault_folder(folder) for folder in folders]


class PdmFoldersRequest(BaseModel):
    pdm_tool_id: str
    path: str = Field(default="\\", min_length=1, max_length=260)
    offset: int = Field(default=0, ge=0)
    snapshot: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        if any(ord(char) < 32 or ord(char) == 127 for char in path):
            raise ValueError("A vault path cannot contain control characters.")
        return path


async def _tool_config(tool_id: str) -> dict:
    config = await repository.get_tool_config(tool_id)
    if config is None or config.tool_type != ToolType.SOLIDWORKS:
        raise HTTPException(status_code=404, detail="Tool configuration not found")
    return config.model_dump()


@router.get("/host/artifact")
async def host_artifact():
    artifact = get_host_artifact()
    return {
        "available": artifact is not None,
        "host_version": artifact.host_version if artifact else None,
        "protocol": artifact.protocol if artifact else None,
        "sha256": artifact.sha256 if artifact else None,
        "size_bytes": artifact.size_bytes if artifact else None,
    }


@router.get("/host/download")
async def host_download():
    artifact = get_host_artifact()
    if artifact is None:
        raise HTTPException(status_code=503, detail="SolidWorks host artifact unavailable")
    if not verify_host_artifact(artifact):
        raise HTTPException(status_code=503, detail="SolidWorks host artifact failed its integrity check")
    return FileResponse(artifact.path, media_type="application/zip", filename=artifact.file_name)


@router.get("/tools/{tool_id}/host")
async def host_status(tool_id: str):
    return await solidworks_service.host_status(await _tool_config(tool_id))


@router.post("/tools/{tool_id}/host/install")
async def install_host(tool_id: str):
    config = await _tool_config(tool_id)
    try:
        return await solidworks_service.install_host(config)
    except HostArtifactUnavailableError as exc:
        raise HTTPException(status_code=503, detail="SolidWorks host artifact unavailable") from exc


@router.post("/tools/{tool_id}/host/uninstall")
async def uninstall_host(tool_id: str):
    return await solidworks_service.uninstall_host(await _tool_config(tool_id))


@router.post("/tools/{tool_id}/pdm/rights")
async def pdm_rights(tool_id: str, body: PdmRightsRequest):
    config = await _tool_config(tool_id)
    try:
        return await pdm_rights_service.rights(config, body.pdm_tool_id, body.folders)
    except PdmToolNotFound as exc:
        raise HTTPException(status_code=404, detail="PDM tool not found") from exc


@router.post("/tools/{tool_id}/pdm/folders")
async def pdm_folders(tool_id: str, body: PdmFoldersRequest):
    config = await _tool_config(tool_id)
    try:
        return await pdm_rights_service.folders(config, body.pdm_tool_id, body.path, body.offset, body.snapshot)
    except PdmToolNotFound as exc:
        raise HTTPException(status_code=404, detail="PDM tool not found") from exc
