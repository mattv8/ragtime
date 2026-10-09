"""Admin control-plane routes for content protection."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ragtime.content_protection.models import ContentProtectionConfig
from ragtime.content_protection.service import (
    ProtectionContext,
    list_decisions,
    load_config,
    preview_policy,
    probe_readiness,
    save_config,
    test_sample,
)
from ragtime.core import app_settings
from ragtime.core.database import get_db
from ragtime.core.security import require_admin
from ragtime.tools.registry import get_all_tools

router = APIRouter(prefix="/indexes/content-protection", tags=["Content Protection"])

_SURFACES: tuple[tuple[str, str], ...] = (
    ("chat", "Chat"),
    ("workspace_chat", "Workspace chat"),
    ("shared_chat", "Shared chat"),
    ("openai_api", "OpenAI API"),
    ("mcp", "MCP"),
    ("development", "Development"),
    ("component", "Component"),
    ("history", "History"),
)


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SaveConfigRequest(_RequestModel):
    expected_revision: int = Field(ge=0)
    config: ContentProtectionConfig


class PreviewPolicyRequest(_RequestModel):
    user_id: str | None = Field(default=None, max_length=128)
    surface: str = Field(default="chat", min_length=1, max_length=64)
    mcp_route: str | None = Field(default=None, max_length=256)
    tool_id: str | None = Field(default=None, max_length=256)
    public: bool = False
    baseline: Literal["user", "service", "public", "anonymous"] = "user"
    config: ContentProtectionConfig | None = None
    access_level_ids: list[str] | None = None


class SampleTestRequest(_RequestModel):
    config: ContentProtectionConfig
    sample: str = Field(min_length=1, max_length=1_048_576)
    access_level_ids: list[str] = Field(default_factory=list, max_length=24)
    user_id: str | None = Field(default=None, max_length=128)


class ReadinessRequest(_RequestModel):
    config: ContentProtectionConfig


def _catalog_row(identifier: str, name: str) -> dict[str, str]:
    return {"id": identifier, "name": name}


@router.get("/config")
async def get_config(_user: Any = Depends(require_admin)) -> Any:
    """Return the persisted configuration directly. Admin only."""
    try:
        return await load_config()
    except (TypeError, ValueError):
        record = await (await get_db()).contentprotectionconfig.find_unique(where={"id": "default"})
        revision = int(record.revision) if record is not None else 0
        raise HTTPException(status_code=409, detail={"code": "invalid_content_protection_config", "revision": revision}) from None


@router.put("/config")
async def update_config(body: SaveConfigRequest, user: Any = Depends(require_admin)) -> Any:
    """Validate and atomically save a revisioned configuration. Admin only."""
    try:
        return await save_config(body.config, body.expected_revision, str(user.id))
    except Exception as exc:
        # The core owns validation and exposes a stable error code without
        # letting route handlers serialize provider/database details.
        if getattr(exc, "code", None) == "revision_conflict" or (isinstance(exc, ValueError) and str(exc) == "revision_conflict"):
            raise HTTPException(status_code=409, detail="Content protection configuration changed") from exc
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=422, detail="Invalid content protection configuration") from exc
        raise


@router.get("/catalog")
async def get_catalog(_user: Any = Depends(require_admin)) -> dict[str, Any]:
    """Return trusted canonical policy scope choices. Admin only."""
    db = await get_db()
    users, groups, tool_configs, mcp_routes = await asyncio.gather(
        db.user.find_many(order={"username": "asc"}),
        db.authgroup.find_many(order={"displayName": "asc"}),
        db.toolconfig.find_many(order={"name": "asc"}),
        db.mcprouteconfig.find_many(order={"name": "asc"}),
    )

    tools = {identifier: _catalog_row(identifier, identifier) for identifier in get_all_tools()}
    tools.update({str(tool.id): _catalog_row(str(tool.id), str(tool.name)) for tool in tool_configs})
    settings = await app_settings.get_app_settings()
    catalog: dict[str, Any] = {
        "users": [_catalog_row(str(user.id), str(user.displayName or user.username)) for user in users],
        "groups": [_catalog_row(str(group.id), str(group.displayName)) for group in groups],
        "tools": sorted(tools.values(), key=lambda item: (item["name"].lower(), item["id"])),
        "mcp_routes": [_catalog_row("default", "Default MCP route")] + [_catalog_row(str(route.id), str(route.name)) for route in mcp_routes],
        "surfaces": [_catalog_row(identifier, name) for identifier, name in _SURFACES],
        "classifier_status": {
            "typesafe_key_configured": bool(settings.get("typesafe_api_key")),
            "openrouter_key_configured": bool(settings.get("openrouter_api_key")),
        },
    }
    return catalog


@router.post("/preview")
async def preview(body: PreviewPolicyRequest, _user: Any = Depends(require_admin)) -> Any:
    """Resolve policy server-side for the supplied trusted scope identifiers."""
    active_config = body.config
    if body.access_level_ids is not None:
        _validate_synthetic_access_levels(body)
        active_config = active_config or await load_config()
        _validate_access_level_ids(body.access_level_ids, active_config)
    result = await preview_policy(
        ProtectionContext(
            user_id=body.user_id,
            surface=body.surface,
            mcp_route=body.mcp_route,
            tool_id=body.tool_id,
            public=body.public,
            baseline="public" if body.public else body.baseline,
        ),
        active_config,
        access_level_ids=body.access_level_ids,
    )
    from ragtime.rag.prompts import build_access_level_prompt_fragment

    result["prompt_fragment"] = build_access_level_prompt_fragment(result)
    return result


def _validate_synthetic_access_levels(body: PreviewPolicyRequest) -> None:
    if body.user_id is not None or body.public or body.baseline != "user":
        raise HTTPException(status_code=422, detail={"code": "ambiguous_synthetic_access_levels"})


def _validate_access_level_ids(access_level_ids: list[str], config: ContentProtectionConfig) -> None:
    if len(access_level_ids) > 24 or any(not identifier or len(identifier) > 128 for identifier in access_level_ids):
        raise HTTPException(status_code=422, detail={"code": "invalid_access_level_ids"})
    if len(access_level_ids) != len(set(access_level_ids)):
        raise HTTPException(status_code=422, detail={"code": "duplicate_access_level_ids"})
    known_ids = {level.id for level in config.access_levels}
    if unknown_ids := sorted(set(access_level_ids) - known_ids):
        raise HTTPException(status_code=422, detail={"code": "unknown_access_level_ids", "access_level_ids": unknown_ids})


@router.post("/test")
async def test(body: SampleTestRequest, _user: Any = Depends(require_admin)) -> Any:
    """Classify only an administrator-supplied bounded sample. Admin only."""
    try:
        context = ProtectionContext(user_id=body.user_id) if body.user_id else None
        return await test_sample(body.config, body.sample, body.access_level_ids, context=context)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid content protection sample") from exc


@router.post("/readiness")
async def readiness(body: ReadinessRequest, _user: Any = Depends(require_admin)) -> Any:
    """Run the bounded classifier contract probe. Admin only."""
    return await probe_readiness(body.config)


@router.get("/decisions")
async def decisions(_user: Any = Depends(require_admin)) -> dict[str, Any]:
    """Return retained decision metadata, never candidate content. Admin only."""
    return {"items": await list_decisions(limit=50)}
