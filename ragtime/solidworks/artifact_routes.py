"""Authenticated SolidWorks artifact metadata and content."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ragtime.core.security import get_current_user
from ragtime.indexer.repository import repository
from ragtime.solidworks.artifacts import (
    ARTIFACT_URL_PREFIX,
    ArtifactExpired,
    ArtifactForbidden,
    ArtifactNotFound,
    artifact_wire,
    authorize_artifact,
    load_artifact,
    resolve_artifact_content,
    sanitize_artifact_filename,
)

router = APIRouter(prefix=ARTIFACT_URL_PREFIX)
_MEDIA_TYPES = {
    "png": "image/png",
    "step": "model/step",
    "stl": "model/stl",
    "3mf": "model/3mf",
    "glb": "model/gltf-binary",
    "zip": "application/zip",
}


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ArtifactExpired):
        return HTTPException(status_code=410, detail="expired")
    if isinstance(exc, ArtifactForbidden):
        return HTTPException(status_code=403, detail="forbidden")
    return HTTPException(status_code=404, detail="not_found")


@router.get("/{artifact_id}")
async def artifact_metadata(artifact_id: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    try:
        meta, _ = await load_artifact(artifact_id)
        await authorize_artifact(meta, user)
    except (ArtifactNotFound, ArtifactForbidden) as exc:
        raise _http_error(exc) from exc
    tool = await repository.get_tool_config(meta["tool_config_id"]) if meta.get("tool_config_id") else None
    return {**artifact_wire(meta), "expired": meta["expired"], "tool_config_name": tool.name if tool else None}


@router.get("/{artifact_id}/content")
async def artifact_content(artifact_id: str, inline: int = 0, user: Any = Depends(get_current_user)) -> FileResponse:
    try:
        meta, path = await resolve_artifact_content(artifact_id, user)
    except (ArtifactNotFound, ArtifactForbidden, ArtifactExpired) as exc:
        raise _http_error(exc) from exc
    format = meta["format"]
    return FileResponse(
        path,
        media_type=_MEDIA_TYPES.get(format, "application/octet-stream"),
        filename=sanitize_artifact_filename(meta["file_name"], format),
        content_disposition_type="inline" if format == "png" or inline == 1 else "attachment",
        headers={"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"},
    )
