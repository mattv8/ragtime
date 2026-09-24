"""Actor-aware external workspace-development operations.

This module is deliberately transport neutral: HTTP and MCP both resolve a
``DevelopmentPrincipal`` and call this dispatcher.  It never manufactures a
conversation or substitutes the workspace owner for the caller.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import posixpath
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from ragtime.content_protection.external import authorize_external_content, public_error_detail
from ragtime.core.database import get_db
from ragtime.core.tool_access import resolve_tool_access
from ragtime.http_api.models import HttpApiConnectionConfig, HttpApiRequest
from ragtime.http_api.openapi import search_openapi_catalog
from ragtime.indexer.models import SCHEMA_INDEXER_CAPABLE_TOOL_TYPES, ToolType
from ragtime.indexer.pdm_service import search_pdm_index
from ragtime.indexer.repository import repository
from ragtime.indexer.schema_service import search_schema_index
from ragtime.rag.components import rag
from ragtime.tools.filesystem_indexer import search_filesystem_index
from ragtime.userspace.agent_read_service import agent_read_service
from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.instruction_bundle import build_instruction_bundle
from ragtime.userspace.instruction_facts import build_env_var_turn_hint, normalize_facts
from ragtime.userspace.models import ExecuteComponentRequest, UpsertWorkspaceFileRequest
from ragtime.userspace.planning_service import planning_service
from ragtime.userspace.service import (
    _USPACE_EXEC_BROWSER_SUPPORTED_TOOL_TYPES,
    _USPACE_RUNTIME_BRIDGE_SUPPORTED_TOOL_TYPES,
    userspace_service,
)


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        result["required"] = required
    return result


def _file_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """Expose the same content/metadata constraints that writes validate."""
    model_schema = UpsertWorkspaceFileRequest.model_json_schema()
    canonical_properties = model_schema.get("properties", {})
    result = _schema({key: canonical_properties.get(key, value) for key, value in properties.items()}, required)
    if "$defs" in model_schema:
        result["$defs"] = model_schema["$defs"]
    return result


_OPERATIONS: tuple[tuple[str, str, str, dict[str, Any]], ...] = (
    ("context", "Complete ACL-filtered external harness instruction bundle.", "read", _schema({})),
    ("operation_describe", "Describe one authorized operation and its input schema.", "read", _schema({"operation": {"type": "string"}}, ["operation"])),
    ("files_list", "List workspace files.", "read", _schema({"prefix": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}})),
    (
        "file_read",
        "Read a UTF-8 workspace file with its content hash.",
        "read",
        _schema({"path": {"type": "string"}, "start_line": {"type": "integer"}, "max_lines": {"type": "integer"}}, ["path"]),
    ),
    (
        "file_write",
        "Write a file when its expected content hash matches.",
        "write",
        _file_schema(
            {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "expected_hash": {"type": ["string", "null"]},
                "artifact_type": {"type": ["string", "null"]},
                "live_data_requested": {"type": "boolean"},
                "live_data_connections": {},
                "live_data_checks": {},
            },
            ["path", "content", "expected_hash"],
        ),
    ),
    (
        "file_patch",
        "Apply an exact text replacement when its expected hash matches.",
        "write",
        _file_schema(
            {
                "path": {"type": "string"},
                "expected_hash": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
                "artifact_type": {"type": ["string", "null"]},
                "live_data_requested": {"type": "boolean"},
                "live_data_connections": {},
                "live_data_checks": {},
            },
            ["path", "expected_hash", "old", "new"],
        ),
    ),
    (
        "file_delete",
        "Delete a file when its expected content hash matches.",
        "write",
        _schema({"path": {"type": "string"}, "expected_hash": {"type": "string"}}, ["path", "expected_hash"]),
    ),
    (
        "code_search",
        "Search the workspace's hidden code index.",
        "read",
        _schema(
            {"query": {"type": "string"}, "mode": {"type": "string", "enum": ["semantic", "symbols", "hybrid"]}, "max_results": {"type": "integer"}}, ["query"]
        ),
    ),
    ("snapshot_create", "Create a workspace snapshot.", "write", _schema({"message": {"type": "string"}})),
    ("snapshots_list", "List workspace snapshots.", "read", _schema({})),
    ("snapshot_restore", "Restore a snapshot.", "write", _schema({"snapshot_id": {"type": "string"}}, ["snapshot_id"])),
    ("validate", "Validate current workspace file and entrypoint state.", "read", _schema({})),
    ("runtime_status", "Read runtime session status.", "read", _schema({})),
    ("runtime_start", "Start the workspace runtime (non-retry-safe).", "exec", _schema({})),
    ("runtime_stop", "Stop the workspace runtime.", "exec", _schema({})),
    ("preview_launch", "Launch an authenticated workspace preview.", "exec", _schema({"control_plane_origin": {"type": "string"}})),
    ("resources", "List caller-authorized selected tools and granted indexes.", "read", _schema({})),
    (
        "http_api_catalog_search",
        "Search an authorized HTTP API OpenAPI catalog.",
        "read",
        _schema({"component_id": {"type": "string"}, "query": {"type": "string"}, "limit": {"type": "integer"}}, ["component_id", "query"]),
    ),
    (
        "execute_component",
        "Execute a caller-authorized selected tool read-only.",
        "read",
        _schema({"component_id": {"type": "string"}, "request": {}}, ["component_id", "request"]),
    ),
    (
        "index_search",
        "Search the code index or an explicitly granted index.",
        "read",
        _schema({"index_name": {"type": "string"}, "query": {"type": "string"}}, ["index_name", "query"]),
    ),
    ("index_grants_list", "List workspace index grants.", "read", _schema({})),
    ("index_grant_create", "Grant a general index to this workspace.", "write", _schema({"index_name": {"type": "string"}}, ["index_name"])),
    ("index_grant_delete", "Remove a general index grant.", "write", _schema({"index_name": {"type": "string"}}, ["index_name"])),
    (
        "exec_start",
        "Start an asynchronous sandbox exec job.",
        "exec",
        _schema({"command": {"type": "string"}, "timeout_seconds": {"type": "integer"}, "cwd": {"type": "string"}}, ["command"]),
    ),
    ("exec_list", "List runtime exec jobs.", "exec", _schema({})),
    (
        "exec_get",
        "Read incremental exec-job output.",
        "exec",
        _schema({"job_id": {"type": "string"}, "cursor": {"type": "integer"}, "limit": {"type": "integer"}}, ["job_id"]),
    ),
    ("exec_cancel", "Cancel a running exec job.", "exec", _schema({"job_id": {"type": "string"}}, ["job_id"])),
)


class DevelopmentService:
    def list_operations(self, scopes: frozenset[str] | None = None) -> list[dict[str, Any]]:
        """Return registry operations, optionally limited to credential scopes."""
        return [
            {"name": name, "description": description, "scope": scope, "input_schema": schema}
            for name, description, scope, schema in _OPERATIONS
            if scopes is None or scope in scopes
        ]

    async def list_authorized_operations(
        self,
        principal: DevelopmentPrincipal,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        """Return operations available to an authorized workspace caller."""
        await self._workspace(principal, workspace_id, "read")
        return self.list_operations(principal.scopes)

    async def _workspace(self, principal: DevelopmentPrincipal, workspace_id: str, scope: str) -> Any:
        if principal.workspace_id and principal.workspace_id != workspace_id:
            raise HTTPException(status_code=403, detail="Development credential is bound to another workspace")
        if scope not in principal.scopes:
            raise HTTPException(status_code=403, detail=f"Development credential lacks {scope} scope")
        role = "viewer" if scope == "read" else "editor"
        return await userspace_service.enforce_workspace_role(workspace_id, principal.user_id, role, is_admin=principal.is_admin)

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_arguments(schema: dict[str, Any], arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise HTTPException(status_code=422, detail="arguments must be a JSON object")
        properties = schema.get("properties", {})
        unknown = set(arguments) - set(properties)
        missing = [key for key in schema.get("required", []) if key not in arguments]
        if unknown or missing:
            raise HTTPException(status_code=422, detail={"unknown": sorted(unknown), "missing": missing})
        for key, value in arguments.items():
            definition = properties[key]
            expected = definition.get("type") if isinstance(definition, dict) else None
            allowed = expected if isinstance(expected, list) else [expected]
            if expected is None:
                continue
            valid = (
                ("string" in allowed and isinstance(value, str))
                or ("integer" in allowed and isinstance(value, int) and not isinstance(value, bool))
                or ("boolean" in allowed and isinstance(value, bool))
                or ("null" in allowed and value is None)
            )
            if not valid:
                raise HTTPException(status_code=422, detail=f"arguments.{key} has an invalid type")
            if isinstance(definition, dict) and "enum" in definition and value not in definition["enum"]:
                raise HTTPException(status_code=422, detail=f"arguments.{key} has an invalid value")
        return arguments

    @staticmethod
    def _build_file_write_request(
        *,
        content: str,
        arguments: dict[str, Any],
        existing_file: Any | None = None,
    ) -> UpsertWorkspaceFileRequest:
        """Use the canonical write model so live-data proof validation remains intact."""
        patch = existing_file is not None
        payload: dict[str, Any] = {
            "content": content,
            "artifact_type": arguments.get("artifact_type", getattr(existing_file, "artifact_type", None) if patch else None),
            "live_data_requested": arguments.get("live_data_requested", False),
            "live_data_connections": arguments.get("live_data_connections", getattr(existing_file, "live_data_connections", None) if patch else None),
            "live_data_checks": arguments.get("live_data_checks", getattr(existing_file, "live_data_checks", None) if patch else None),
        }
        try:
            return UpsertWorkspaceFileRequest(**payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

    async def _read_with_hash(
        self, principal: DevelopmentPrincipal, workspace_id: str, path: str, *, start_line: int = 1, max_lines: int = 400
    ) -> dict[str, Any]:
        file = await userspace_service.get_workspace_file(workspace_id, path, principal.user_id, is_admin=principal.is_admin)
        lines = file.content.splitlines()
        start_line = max(1, start_line)
        max_lines = max(1, min(max_lines, 2000))
        selected = lines[start_line - 1 : start_line - 1 + max_lines]
        end_line = start_line - 1 + len(selected)
        return {
            "path": file.path,
            "content": "\n".join(selected),
            "content_hash": self._hash(file.content),
            "updated_at": file.updated_at,
            "total_lines": len(lines),
            "start_line": start_line,
            "end_line": end_line,
            "next_start_line": end_line + 1 if end_line < len(lines) else None,
            "truncated": end_line < len(lines),
        }

    async def _context(self, principal: DevelopmentPrincipal, workspace_id: str) -> dict[str, Any]:
        from ragtime.userspace.runtime_service import userspace_runtime_service

        context = await planning_service.get_workspace_context(workspace_id, principal.user_id)
        db = await get_db()
        user = await db.user.find_unique(where={"id": principal.user_id})
        context["user"] = {"username": getattr(user, "username", None), "display_name": getattr(user, "displayName", None), "role": getattr(user, "role", None)}
        catalog = await self._resources(principal, workspace_id)
        mounts, mountable_sources, env_vars, diagnostics = await asyncio.gather(
            userspace_service.list_workspace_mounts(workspace_id, principal.user_id, is_admin=principal.is_admin),
            userspace_service.list_mountable_sources(workspace_id, principal.user_id),
            userspace_service.list_workspace_env_var_summaries(workspace_id, principal.user_id),
            userspace_service.list_workspace_preview_diagnostic_summary(workspace_id),
        )
        try:
            runtime_status = await userspace_runtime_service.get_devserver_status(workspace_id, principal.user_id)
        except Exception:
            # Context export must remain available when a transient provider
            # readiness read fails; do not substitute a guessed runtime state.
            runtime_status = None
        storage = await userspace_service.get_workspace_object_storage_summary(workspace_id, principal.user_id)
        shared_sqlite = await userspace_service.list_accessible_cross_workspace_sqlite_targets(workspace_id, principal.user_id)
        context["selected_tools"] = catalog["tools"]
        context["authorized_tools"] = catalog["tools"]
        context["authorized_indexes"] = catalog["indexes"]
        context["authorized_resources"] = {
            "mounts_enabled": bool(mounts or mountable_sources),
            "mounts": [
                {
                    "workspace_relative_path": posixpath.relpath(str(getattr(item, "target_path", "/workspace") or "/workspace"), "/workspace"),
                    "target_path": str(getattr(item, "target_path", "") or "/workspace"),
                    "source_name": getattr(item, "source_name", None),
                    "source_path": getattr(item, "source_path", None),
                    "description": getattr(item, "description", None),
                    "sync_status": getattr(item, "sync_status", None),
                    "enabled": "true" if bool(getattr(item, "enabled", False)) else "false",
                }
                for item in mounts
            ],
            "object_storage_enabled": bool(storage and storage.buckets),
            "object_storage_buckets": [
                {
                    "name": bucket.name,
                    "description": bucket.description,
                    "public_root": f"/{bucket.name}/{getattr(bucket, 'public_prefix', 'public') or 'public'}",
                    "private_root": f"/{bucket.name}/{getattr(bucket, 'private_prefix', 'private') or 'private'}",
                    "is_default": "true" if storage is not None and bucket.name == storage.default_bucket_name else "false",
                }
                for bucket in (storage.buckets if storage else [])
            ],
            "shared_sqlite_databases": shared_sqlite,
        }
        context["env_vars"] = [{"key": item.key, "has_value": item.has_value} for item in env_vars]
        context["diagnostics"] = normalize_facts(diagnostics)
        context["runtime_status"] = normalize_facts(runtime_status)
        from ragtime.rag.components import RAGComponents
        from ragtime.rag.prompts import build_userspace_diagnostics_turn_reminder_line

        context["env_var_reminder_line"] = build_env_var_turn_hint(env_vars)
        context["runtime_status_reminder_line"] = RAGComponents._build_userspace_runtime_status_turn_hint(runtime_status) if runtime_status else ""
        context["diagnostics_reminder_line"] = build_userspace_diagnostics_turn_reminder_line(diagnostics)
        return build_instruction_bundle(context)

    @staticmethod
    def _browser_component_request_schema(tool_type: str) -> dict[str, Any]:
        """Describe the payload accepted by execute_component, not tool-only fields."""
        if tool_type == "http_api":
            return HttpApiRequest.model_json_schema()
        return {
            "oneOf": [
                {"type": "string", "description": "Read-only query text."},
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ]
        }

    async def _enrich_authorized_tool(self, tool_config: Any, access: str) -> dict[str, Any] | None:
        """Build external metadata from the canonical runtime tool after ACL filtering."""
        tool_type = str(getattr(getattr(tool_config, "tool_type", None), "value", getattr(tool_config, "tool_type", "")))
        runtime_config = userspace_service._tool_config_runtime_dict(tool_config, allow_write=False)
        try:
            runtime_tool = await rag.build_primary_runtime_tool_from_config(runtime_config)
        except Exception:
            return None
        if runtime_tool is None:
            return None

        canonical_description = str(getattr(runtime_tool, "description", "") or "")
        browser_supported = tool_type in _USPACE_EXEC_BROWSER_SUPPORTED_TOOL_TYPES
        server_supported = tool_type in _USPACE_RUNTIME_BRIDGE_SUPPORTED_TOOL_TYPES
        browser_lane: dict[str, Any] = {
            "supported": browser_supported,
            "mode": "browser_read_only",
            "write_policy": "Browser/external execution is always read-only.",
        }
        if browser_supported:
            browser_lane["request_schema"] = self._browser_component_request_schema(tool_type)
            browser_lane["instructions"] = canonical_description
            browser_lane["timeout_max_seconds"] = int(getattr(tool_config, "timeout_max_seconds", 300) or 300)
        else:
            browser_lane["reason"] = "This tool type is not supported on the browser/external read-only execute_component surface."

        return {
            "component_id": str(getattr(tool_config, "id", "")),
            "name": str(getattr(tool_config, "name", "") or ""),
            "tool_type": tool_type,
            "description": str(getattr(tool_config, "description", "") or ""),
            "canonical_description": canonical_description,
            "access": access,
            # Retained for planning-contract compatibility. Only the browser
            # lane is callable through this external operation.
            "execute_component": browser_lane,
            "execution_lanes": {
                "browser": browser_lane,
                "server_runtime_bridge": {
                    "supported": server_supported,
                    "mode": "server_runtime_bridge",
                    "write_policy": "Server execution applies workspace owner opt-in and tool access policy.",
                },
            },
        }

    async def _resources(self, principal: DevelopmentPrincipal, workspace_id: str) -> dict[str, Any]:
        workspace = await self._workspace(principal, workspace_id, "read")
        selected_ids = await planning_service._selected_tool_ids(workspace)
        levels = await resolve_tool_access(user_id=principal.user_id, is_admin=principal.is_admin, surface="workspace", tool_config_ids=selected_ids)
        allowed_tool_ids = [tool_id for tool_id in selected_ids if levels.get(tool_id, "deny") in {"read", "read_write"}]
        tools: list[dict[str, Any]] = []
        for tool_id in allowed_tool_ids:
            tool_config = await repository.get_tool_config(tool_id)
            if tool_config is None or not bool(getattr(tool_config, "enabled", False)):
                continue
            enriched = await self._enrich_authorized_tool(tool_config, levels[tool_id])
            if enriched is not None:
                tools.append(enriched)
        db = await get_db()
        grants = await db.workspaceindexgrant.find_many(where={"workspaceId": workspace_id}, order={"indexName": "asc"})
        indexes = [{"name": "workspace_code", "source_type": "workspace_code"}]
        # Resolve only named grants; never enumerate global metadata for a
        # workspace caller. Stale grants are omitted until an owner repairs it.
        granted_descriptors = []
        for grant in grants:
            descriptor = await self._resolve_general_index_descriptor(grant.indexName)
            if descriptor is not None:
                granted_descriptors.append(descriptor)
        backing_tool_ids = [descriptor["backing_tool_id"] for descriptor in granted_descriptors if descriptor.get("backing_tool_id")]
        index_levels = await resolve_tool_access(
            user_id=principal.user_id,
            is_admin=principal.is_admin,
            surface="workspace",
            tool_config_ids=backing_tool_ids,
        )
        indexes.extend(
            descriptor
            for descriptor in granted_descriptors
            if not descriptor.get("backing_tool_id") or index_levels.get(descriptor["backing_tool_id"], "deny") in {"read", "read_write"}
        )
        return {
            "tools": tools,
            "indexes": indexes,
            "catalog_revision": self._hash(json.dumps({"tools": tools, "indexes": indexes}, default=str, sort_keys=True))[:16],
        }

    async def _resolve_general_index_descriptor(self, index_name: str) -> dict[str, Any] | None:
        """Resolve exactly one index through its authoritative source model.

        Document indexes live in IndexMetadata. Schema and PDM indexes are
        discovered from their durable index jobs, whose ``indexName`` is the
        canonical backend identity. This lookup never returns a global catalog
        to callers.
        """
        if not index_name or index_name.startswith("userspace_"):
            return None
        metadata = await repository.get_index_metadata(index_name)
        if metadata is not None:
            if not bool(getattr(metadata, "enabled", False)) or str(getattr(metadata, "sourceType", "")) == "userspace_code":
                return None
            return {
                "name": index_name,
                "source_type": getattr(metadata, "sourceType", "upload"),
                "description": getattr(metadata, "description", "") or "",
                "document_count": getattr(metadata, "documentCount", 0),
                "chunk_count": getattr(metadata, "chunkCount", 0),
                "enabled": True,
            }
        db = await get_db()
        for source_type, job_model, valid_tool_types in (
            ("schema", db.schemaindexjob, SCHEMA_INDEXER_CAPABLE_TOOL_TYPES),
            ("pdm", db.pdmindexjob, {ToolType.SOLIDWORKS_PDM}),
        ):
            job = await job_model.find_first(where={"indexName": index_name}, order={"createdAt": "desc"})
            if job is None:
                continue
            tool = await repository.get_tool_config(str(job.toolConfigId))
            if tool is not None and bool(getattr(tool, "enabled", False)) and getattr(tool, "tool_type", None) in valid_tool_types:
                return {
                    "name": index_name,
                    "source_type": source_type,
                    "description": str(getattr(tool, "description", "") or ""),
                    "enabled": True,
                    "backing_tool_id": str(job.toolConfigId),
                }
        return None

    async def _exec(self, principal: DevelopmentPrincipal, workspace_id: str, action: str, args: dict[str, Any]) -> Any:
        from ragtime.userspace.runtime_service import userspace_runtime_service

        session = await userspace_runtime_service.get_runtime_session(workspace_id, principal.user_id)
        active = session.session
        if not active:
            raise HTTPException(status_code=404, detail="No active runtime session")
        provider = active.provider_session_id
        if action == "exec_start":
            from ragtime.core.app_settings import get_app_settings
            from ragtime.core.userspace_limits import resolve_userspace_exec_timeout

            try:
                timeout_seconds = resolve_userspace_exec_timeout(await get_app_settings(), args.get("timeout_seconds"))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return await userspace_runtime_service._runtime_manager_request(
                "POST",
                f"/sessions/{provider}/exec-jobs",
                json_payload={
                    "command": args["command"],
                    "timeout_seconds": timeout_seconds,
                    "cwd": args.get("cwd"),
                    "user_id": principal.user_id,
                    "credential_id": principal.credential_id,
                    "operation": "development_exec",
                },
                retry_safe=False,
            )
        suffix = "exec-jobs" if action == "exec_list" else f"exec-jobs/{args['job_id']}"
        if action == "exec_cancel":
            return await userspace_runtime_service._runtime_manager_request("POST", f"/sessions/{provider}/{suffix}/cancel", retry_safe=False)
        if action == "exec_get":
            suffix += f"?cursor={int(args.get('cursor', 0))}&limit={int(args.get('limit', 16384))}"
        return await userspace_runtime_service._runtime_manager_request("GET", f"/sessions/{provider}/{suffix}", allow_list_response=action == "exec_list")

    async def execute(self, principal: DevelopmentPrincipal, workspace_id: str, operation: str, arguments: dict[str, Any] | None = None) -> Any:
        """Guard development inputs before side effects and outputs before release."""
        candidate = {"workspace_id": workspace_id, "operation": operation, "arguments": arguments or {}}
        try:
            await authorize_external_content(
                candidate,
                direction="inbound",
                principal=principal,
                surface="development",
                tool_id=operation,
                resource_id=workspace_id,
                operation=operation,
            )
            result = await self._execute_unprotected(principal, workspace_id, operation, arguments)
            await authorize_external_content(
                result,
                direction="outbound",
                principal=principal,
                surface="development",
                tool_id=operation,
                resource_id=workspace_id,
                operation=operation,
                execution_completed=True,
            )
            return result
        except Exception as exc:
            # Existing operational errors remain available only after their
            # body has passed the same release gate. Protection failures get a
            # fixed transport-safe detail instead of classifier diagnostics.
            if hasattr(exc, "public_detail"):
                raise HTTPException(status_code=403, detail=public_error_detail(exc)) from exc
            try:
                await authorize_external_content(
                    {"error": getattr(exc, "detail", str(exc))},
                    direction="outbound",
                    principal=principal,
                    surface="development",
                    tool_id=operation,
                    resource_id=workspace_id,
                    operation=operation,
                )
            except Exception as release_exc:
                raise HTTPException(status_code=403, detail=public_error_detail(release_exc)) from release_exc
            raise

    async def _execute_unprotected(self, principal: DevelopmentPrincipal, workspace_id: str, operation: str, arguments: dict[str, Any] | None = None) -> Any:
        args = arguments if arguments is not None else {}
        op = next((item for item in _OPERATIONS if item[0] == operation), None)
        if op is None:
            raise HTTPException(status_code=404, detail="Unknown development operation")
        args = self._validate_arguments(op[3], args)
        scope = op[2]
        await self._workspace(principal, workspace_id, scope)
        # Reuse the existing workspace runtime audit sink for minimal external
        # operation attribution. Arguments are intentionally omitted because
        # they can contain file contents, queries, or commands.
        from ragtime.userspace.runtime_service import userspace_runtime_service

        await userspace_runtime_service._audit(
            workspace_id,
            "development_operation",
            user_id=principal.user_id,
            session_id=None,
            payload={"operation": operation, "credential_id": principal.credential_id, "scope": scope},
        )
        if operation == "context":
            return await self._context(principal, workspace_id)
        if operation == "operation_describe":
            contract = next((item for item in self.list_operations(principal.scopes) if item["name"] == args["operation"]), None)
            if contract is None:
                raise HTTPException(status_code=404, detail="Operation is not authorized")
            return contract
        if operation == "files_list":
            return await planning_service.list_files(
                workspace_id,
                principal.user_id,
                prefix=str(args.get("prefix", "")),
                offset=int(args.get("offset", 0)),
                limit=int(args.get("limit", 200)),
                developer_full_inventory=True,
            )
        if operation == "file_read":
            return await self._read_with_hash(principal, workspace_id, args["path"], start_line=args.get("start_line", 1), max_lines=args.get("max_lines", 400))
        if operation in {"file_write", "file_patch", "file_delete"}:
            normalized_path = userspace_service._normalize_workspace_relative_path(args["path"])
            lock = await userspace_service._get_workspace_file_mutation_lock(workspace_id, normalized_path)
            async with lock:
                current_content: str | None = None
                existing_file = None
                try:
                    current_file = await userspace_service.get_workspace_file(workspace_id, args["path"], principal.user_id, is_admin=principal.is_admin)
                    current_content = current_file.content
                    existing_file = current_file
                except HTTPException as exc:
                    if exc.status_code != 404:
                        raise
                expected = args["expected_hash"]
                actual = self._hash(current_content) if current_content is not None else None
                if expected != actual:
                    raise HTTPException(status_code=409, detail={"code": "content_hash_conflict", "expected_hash": expected, "actual_hash": actual})
                if operation == "file_delete":
                    await userspace_service.delete_workspace_file(
                        workspace_id, args["path"], principal.user_id, expected_content_hash=expected, require_content_hash=True, mutation_lock_held=True
                    )
                    return {"path": args["path"], "deleted": True}
                if operation == "file_write":
                    content = args["content"]
                else:
                    if current_content is None:
                        raise HTTPException(status_code=409, detail="Patch target not found")
                    if args["old"] not in current_content:
                        raise HTTPException(status_code=409, detail="Patch target not found")
                    content = current_content.replace(args["old"], args["new"], 1)
                request = self._build_file_write_request(
                    content=content,
                    arguments=args,
                    existing_file=existing_file if operation == "file_patch" else None,
                )
                result = await userspace_service.upsert_workspace_file(
                    workspace_id, args["path"], request, principal.user_id, expected_content_hash=expected, require_content_hash=True, mutation_lock_held=True
                )
                return {"path": result.path, "content_hash": self._hash(content), "updated_at": result.updated_at}
        if operation == "code_search":
            return await agent_read_service.search_code(
                workspace_id,
                principal.user_id,
                is_admin=principal.is_admin,
                query=str(args["query"]),
                mode=args.get("mode", "hybrid"),
                max_results=int(args.get("max_results", 8)),
                max_chars_per_result=1200,
            )
        if operation == "snapshot_create":
            return (await userspace_service.create_snapshot(workspace_id, principal.user_id, str(args.get("message") or "Snapshot"))).model_dump()
        if operation == "snapshots_list":
            return {"snapshots": [item.model_dump() for item in await userspace_service.list_snapshots(workspace_id, principal.user_id)]}
        if operation == "snapshot_restore":
            return (await userspace_service.restore_snapshot(workspace_id, str(args["snapshot_id"]), principal.user_id)).model_dump()
        if operation == "validate":
            from ragtime.rag.components import validate_userspace_source_content

            status = await userspace_service.get_workspace_entrypoint_status_authoritative(workspace_id)
            files = await userspace_service.list_workspace_files(workspace_id, principal.user_id, is_admin=principal.is_admin)
            diagnostics = []
            for item in files:
                if not item.path.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".html", ".htm")):
                    continue
                try:
                    source = await userspace_service.get_workspace_file(workspace_id, item.path, principal.user_id, is_admin=principal.is_admin)
                except HTTPException:
                    continue
                diagnostics.append({"path": item.path, **await validate_userspace_source_content(source.content, item.path)})
            return {
                "entrypoint": {"state": status.state, "framework": status.framework, "command": status.command, "cwd": status.cwd, "error": status.error},
                "valid": status.state == "valid" and all(not item.get("errors") for item in diagnostics),
                "diagnostics": diagnostics,
            }
        if operation in {"runtime_status", "runtime_start", "runtime_stop", "preview_launch"}:
            from ragtime.userspace.runtime_service import userspace_runtime_service

            if operation == "runtime_status":
                # Refresh provider-backed readiness first; this is read-only and
                # deliberately does not start or reconcile a runtime session.
                await userspace_runtime_service.get_devserver_status(workspace_id, principal.user_id)
                return (await userspace_runtime_service.get_runtime_session(workspace_id, principal.user_id)).model_dump()
            if operation == "runtime_start":
                return (await userspace_runtime_service.start_runtime_session(workspace_id, principal.user_id)).model_dump()
            if operation == "runtime_stop":
                return (await userspace_runtime_service.stop_runtime_session(workspace_id, principal.user_id)).model_dump()
            origin = str(args.get("control_plane_origin") or userspace_runtime_service._default_public_control_plane_origin())
            return (
                await userspace_runtime_service.issue_workspace_preview_launch(workspace_id, principal.user_id, control_plane_origin=origin, auto_start=True)
            ).model_dump()
        if operation == "resources":
            return await self._resources(principal, workspace_id)
        if operation == "http_api_catalog_search":
            catalog = await self._resources(principal, workspace_id)
            tool = next((item for item in catalog["tools"] if item["component_id"] == args["component_id"]), None)
            if tool is None or tool.get("tool_type") != "http_api":
                raise HTTPException(status_code=403, detail="HTTP API tool is not authorized for this caller")
            config = await repository.get_tool_config(args["component_id"])
            if config is None:
                raise HTTPException(status_code=404, detail="HTTP API tool is unavailable")
            try:
                connection = HttpApiConnectionConfig(**(getattr(config, "connection_config", {}) or {}))
            except Exception as exc:
                raise HTTPException(status_code=409, detail="HTTP API catalog is unavailable") from exc
            if not connection.openapi_catalog:
                return {"component_id": args["component_id"], "query": args["query"], "total_results": 0, "results": []}
            limit = max(1, min(int(args.get("limit", 10)), 50))
            matches = search_openapi_catalog(connection.openapi_catalog, args["query"], limit=limit)
            return {
                "component_id": args["component_id"],
                "query": args["query"],
                "total_results": len(matches),
                "results": [
                    {
                        "operation_id": item.operation_id,
                        "method": getattr(item.method, "value", item.method),
                        "path": item.path,
                        "summary": item.summary,
                        "description": item.description,
                        "tags": list(item.tags),
                    }
                    for item in matches
                ],
            }
        if operation == "execute_component":
            catalog = await self._resources(principal, workspace_id)
            tool = next((item for item in catalog["tools"] if item["component_id"] == args["component_id"]), None)
            if tool is None:
                raise HTTPException(status_code=403, detail="Tool is not authorized for this caller")
            component_result = await userspace_service.execute_component(
                workspace_id, ExecuteComponentRequest(component_id=args["component_id"], request=args["request"]), principal.user_id
            )
            return component_result.model_dump()
        if operation == "index_search":
            if args["index_name"] != "workspace_code":
                resources = await self._resources(principal, workspace_id)
                granted = next((item for item in resources["indexes"] if item["name"] == args["index_name"]), None)
                if granted is None:
                    raise HTTPException(status_code=403, detail="Index is not granted to this workspace")
                source_type = str(granted.get("source_type") or "upload")
                if source_type == "schema":
                    index_result = await search_schema_index(query=args["query"], index_name=args["index_name"], max_results=8)
                elif source_type == "pdm":
                    index_result = await search_pdm_index(query=args["query"], index_name=args["index_name"], max_results=8)
                else:
                    index_result = await search_filesystem_index(
                        query=args["query"],
                        index_name=args["index_name"],
                        max_results=8,
                        max_chars_per_result=1200,
                    )
                return {"index_name": args["index_name"], "source_type": source_type, "result": index_result}
            return await agent_read_service.search_code(
                workspace_id, principal.user_id, is_admin=principal.is_admin, query=str(args["query"]), mode="hybrid", max_results=8, max_chars_per_result=1200
            )
        if operation in {"index_grants_list", "index_grant_create", "index_grant_delete"}:
            await userspace_service.enforce_workspace_role(workspace_id, principal.user_id, "owner", is_admin=principal.is_admin)
            db = await get_db()
            if operation == "index_grants_list":
                resources = await self._resources(principal, workspace_id)
                return {"grants": [item["name"] for item in resources["indexes"] if item["name"] != "workspace_code"]}
            if operation == "index_grant_create":
                if await self._resolve_general_index_descriptor(args["index_name"]) is None:
                    raise HTTPException(status_code=404, detail="Eligible index not found")
                await db.workspaceindexgrant.upsert(
                    where={"workspaceId_indexName": {"workspaceId": workspace_id, "indexName": str(args["index_name"])}},
                    data={"create": {"workspaceId": workspace_id, "indexName": str(args["index_name"]), "createdById": principal.user_id}, "update": {}},
                )
                return {"index_name": args["index_name"], "granted": True}
            await db.workspaceindexgrant.delete_many(where={"workspaceId": workspace_id, "indexName": str(args["index_name"])})
            return {"index_name": args["index_name"], "granted": False}
        if operation.startswith("exec_"):
            return await self._exec(principal, workspace_id, operation, args)
        raise HTTPException(status_code=400, detail="Invalid development operation state")


development_service = DevelopmentService()
