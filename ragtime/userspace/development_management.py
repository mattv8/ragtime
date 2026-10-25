"""Narrow, transport-neutral workspace management operations."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ragtime.core.tool_access import resolve_tool_access
from ragtime.indexer.repository import repository
from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.models import UpdateWorkspaceRequest, UpsertWorkspaceEnvVarRequest
from ragtime.userspace.service import userspace_service

_MAX_PAGE_BYTES = 24 * 1024
_USER_ONLY_OPERATIONS = frozenset({"workspace_tools_available", "workspace_tools_set"})
_MAX_SELECTED_TOOL_IDS = 500
_MAX_CANDIDATE_FIELD_BYTES = 512


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


MANAGEMENT_OPERATIONS: tuple[tuple[str, str, str, dict[str, Any]], ...] = (
    (
        "workspace_metadata_update",
        "Update only this workspace's name or description.",
        "write",
        _schema(
            {
                "name": {"type": "string", "minLength": 1, "maxLength": 120, "description": "New workspace name."},
                "description": {"type": "string", "maxLength": 1000, "description": "New workspace description."},
            }
        ),
    ),
    (
        "env_placeholder_upsert",
        "Create or describe an environment-variable placeholder without setting its value.",
        "write",
        _schema(
            {
                "key": {"type": "string", "minLength": 1, "maxLength": 128, "description": "Environment variable key; no value is accepted."},
                "description": {"type": "string", "maxLength": 1000, "description": "Optional placeholder description."},
            },
            ["key"],
        ),
    ),
    (
        "workspace_tools_available",
        "List healthy tools that both the caller and workspace owner may use.",
        "read",
        _schema(
            {
                "query": {"type": "string", "maxLength": 120, "description": "Optional case-insensitive candidate search."},
                "offset": {"type": "integer", "minimum": 0, "description": "Zero-based page offset."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Maximum candidates to return."},
            }
        ),
    ),
    (
        "workspace_tools_set",
        "Set the workspace's authorized tool selection without enabling tool writes.",
        "write",
        _schema(
            {
                "mode": {"type": "string", "enum": ["custom", "default_all"], "description": "custom requires selected_tool_ids; default_all forbids them."},
                "selected_tool_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": _MAX_SELECTED_TOOL_IDS,
                    "description": "Authorized IDs for custom mode, including an empty list.",
                },
            },
            ["mode"],
        ),
    ),
)
_MANAGEMENT_OPERATION_NAMES = frozenset(operation[0] for operation in MANAGEMENT_OPERATIONS)


class _ManagementRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _MetadataUpdateRequest(_ManagementRequest):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def require_change(self) -> "_MetadataUpdateRequest":
        if self.name is None and self.description is None:
            raise ValueError("name or description is required")
        if self.name is not None and not self.name.strip():
            raise ValueError("name must not be blank")
        return self


class _EnvPlaceholderRequest(_ManagementRequest):
    key: str = Field(min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=1000)


class _ToolCandidatesRequest(_ManagementRequest):
    query: str | None = Field(default=None, max_length=120)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def normalize_query(self) -> "_ToolCandidatesRequest":
        if self.query is not None:
            self.query = self.query.strip() or None
        return self


class _ToolSelectionRequest(_ManagementRequest):
    mode: Literal["custom", "default_all"]
    selected_tool_ids: list[str] | None = Field(default=None, max_length=_MAX_SELECTED_TOOL_IDS)

    @model_validator(mode="after")
    def validate_mode(self) -> "_ToolSelectionRequest":
        has_selected_ids = "selected_tool_ids" in self.model_fields_set
        if self.mode == "custom" and (not has_selected_ids or self.selected_tool_ids is None):
            raise ValueError("selected_tool_ids is required for custom mode")
        if self.mode == "default_all" and has_selected_ids:
            raise ValueError("selected_tool_ids is not allowed for default_all mode")
        if self.selected_tool_ids is not None:
            self.selected_tool_ids = list(dict.fromkeys(self.selected_tool_ids))
        return self


def management_operation_visible(principal: DevelopmentPrincipal, name: str) -> bool:
    """Return whether an operation may be disclosed to this principal."""
    if name not in _MANAGEMENT_OPERATION_NAMES:
        return False
    return name not in _USER_ONLY_OPERATIONS or getattr(principal, "auth_source", None) == "user"


def _operation(name: str) -> tuple[str, str, str, dict[str, Any]]:
    for operation in MANAGEMENT_OPERATIONS:
        if operation[0] == name:
            return operation
    raise HTTPException(status_code=404, detail="Unknown workspace management operation")


def _validate_request(model: type[_ManagementRequest], arguments: Any) -> _ManagementRequest:
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=422, detail="arguments must be a JSON object")
    try:
        return model.model_validate(arguments)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_input=False, include_context=False, include_url=False)) from exc


async def _enforce_operation_access(principal: DevelopmentPrincipal, workspace_id: str, scope: str) -> Any:
    if principal.workspace_id is not None and principal.workspace_id != workspace_id:
        raise HTTPException(status_code=403, detail="Development credential is bound to another workspace")
    if scope not in principal.scopes:
        raise HTTPException(status_code=403, detail=f"Development credential lacks {scope} scope")
    required_role = "viewer" if scope == "read" else "editor"
    return await userspace_service.enforce_workspace_role(
        workspace_id,
        principal.user_id,
        required_role,
        is_admin=principal.is_admin,
    )


async def _authorized_tool_candidates(principal: DevelopmentPrincipal, workspace_id: str) -> list[Any]:
    workspace = await userspace_service.enforce_workspace_role(
        workspace_id,
        principal.user_id,
        "owner",
        is_admin=principal.is_admin,
    )
    healthy_ids = set(await repository.list_healthy_enabled_tool_ids())
    configs = await repository.list_tool_configs(enabled_only=True)
    candidate_configs = [config for config in configs if str(getattr(config, "id", "")) in healthy_ids]
    candidate_ids = [str(config.id) for config in candidate_configs]
    caller_access = await resolve_tool_access(
        user_id=principal.user_id,
        is_admin=principal.is_admin,
        surface="workspace",
        tool_config_ids=candidate_ids,
    )
    owner_ids = set(await userspace_service.filter_tool_ids_for_workspace_owner(workspace, candidate_ids))
    return [config for config in candidate_configs if str(config.id) in owner_ids and caller_access.get(str(config.id), "deny") != "deny"]


def _candidate_metadata(config: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"component_id": str(config.id)}
    for field in ("name", "description"):
        value = str(getattr(config, field, "") or "")
        if len(value.encode("utf-8")) <= _MAX_CANDIDATE_FIELD_BYTES:
            item[field] = value
        else:
            item[f"{field}_snippet"] = value.encode("utf-8")[:_MAX_CANDIDATE_FIELD_BYTES].decode("utf-8", errors="ignore")
            item[f"{field}_truncated"] = True
    return item


def _matches_query(config: Any, query: str | None) -> bool:
    if query is None:
        return True
    return query in "\n".join((str(config.id), str(getattr(config, "name", "")), str(getattr(config, "description", "") or ""))).casefold()


def _bounded_candidate_page(configs: list[Any], request: _ToolCandidatesRequest) -> dict[str, Any]:
    query = request.query.casefold() if request.query else None
    items = [_candidate_metadata(config) for config in configs if _matches_query(config, query)]
    page = items[request.offset : request.offset + request.limit]
    emitted: list[dict[str, Any]] = []
    for item in page:
        candidate = {"items": [*emitted, item], "total": len(items), "offset": request.offset, "next_offset": None}
        if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > _MAX_PAGE_BYTES:
            break
        emitted.append(item)
    if not emitted and request.offset < len(items):
        raise HTTPException(status_code=413, detail="A tool candidate exceeds bounded delivery limit")
    next_offset = request.offset + len(emitted) if request.offset + len(emitted) < len(items) else None
    return {"items": emitted, "total": len(items), "offset": request.offset, "next_offset": next_offset}


async def execute_management_operation(
    principal: DevelopmentPrincipal,
    workspace_id: str,
    operation: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Execute one bounded management operation through canonical services."""
    _name, _description, scope, _schema_definition = _operation(operation)
    if not management_operation_visible(principal, operation):
        raise HTTPException(status_code=403, detail="Operation is not available for this credential")
    await _enforce_operation_access(principal, workspace_id, scope)

    if operation == "workspace_metadata_update":
        request = _validate_request(_MetadataUpdateRequest, arguments)
        assert isinstance(request, _MetadataUpdateRequest)
        updated_workspace = await userspace_service.update_workspace(
            workspace_id,
            UpdateWorkspaceRequest(name=request.name, description=request.description),
            principal.user_id,
            is_admin=principal.is_admin,
        )
        return {"workspace_id": updated_workspace.id, "name": updated_workspace.name, "description": updated_workspace.description}

    if operation == "env_placeholder_upsert":
        request = _validate_request(_EnvPlaceholderRequest, arguments)
        assert isinstance(request, _EnvPlaceholderRequest)
        updated_env_var = await userspace_service.upsert_workspace_env_var(
            workspace_id,
            principal.user_id,
            UpsertWorkspaceEnvVarRequest(key=request.key, description=request.description, value=None),
        )
        return {"key": updated_env_var.key, "has_value": updated_env_var.has_value, "source": updated_env_var.source}

    if operation == "workspace_tools_available":
        request = _validate_request(_ToolCandidatesRequest, arguments)
        assert isinstance(request, _ToolCandidatesRequest)
        candidates = await _authorized_tool_candidates(principal, workspace_id)
        return _bounded_candidate_page(candidates, request)

    request = _validate_request(_ToolSelectionRequest, arguments)
    assert isinstance(request, _ToolSelectionRequest)
    candidates = await _authorized_tool_candidates(principal, workspace_id)
    candidate_ids = {str(config.id) for config in candidates}
    selected_tool_ids = request.selected_tool_ids
    update_payload: dict[str, Any] = {"tool_selection_mode": request.mode}
    if request.mode == "custom":
        assert selected_tool_ids is not None
        unknown_ids = sorted(set(selected_tool_ids) - candidate_ids)
        if unknown_ids:
            raise HTTPException(status_code=422, detail={"selected_tool_ids": unknown_ids})
        update_payload.update(selected_tool_ids=selected_tool_ids, selected_tool_group_ids=[])
    update_request = UpdateWorkspaceRequest(**update_payload)
    updated = await userspace_service.update_workspace(
        workspace_id,
        update_request,
        principal.user_id,
        is_admin=principal.is_admin,
    )
    response: dict[str, Any] = {"workspace_id": updated.id, "tool_selection_mode": updated.tool_selection_mode}
    if request.mode == "custom":
        response["selected_tool_ids"] = updated.selected_tool_ids
    return response
