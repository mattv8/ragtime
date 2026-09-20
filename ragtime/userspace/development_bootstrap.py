"""Generated, bounded external-development setup and MCP delivery helpers.

The helpers here intentionally operate on supplied context.  Authentication and
workspace ACL checks remain in the HTTP/MCP adapters and in ``development_service``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException

from ragtime.userspace.instruction_content import build_external_guidance_documents

MAX_MCP_RESPONSE_BYTES = 24 * 1024
# Native clients load a skill file as a single artifact, so keep its frontmatter
# and complete inline guidance within this delivery budget.
MAX_NATIVE_SKILL_BYTES = 24 * 1024
# Leave room for the JSON envelope so an ordinary page remains below 8 KiB.
MAX_PAGE_TEXT_BYTES = 8 * 1024
MAX_INLINE_DESCRIPTION_BYTES = 2048
MAX_INLINE_DESCRIPTIONS_BYTES = 8192


def content_hash(value: Any) -> str:
    """Return a stable content identity, never a runtime-status identity."""
    encoded = value.encode("utf-8") if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _guidance_documents() -> dict[str, str]:
    return build_external_guidance_documents()


def _skill_frontmatter(topic: str) -> str:
    tasks = {
        "runtime": "building, starting, debugging, or previewing workspace runtimes",
        "live-data": "wiring selected live data tools into a dashboard",
        "identity": "adding authentication or identity-aware UI",
        "persistence": "adding SQLite persistence without replacing live data",
        "ui": "building themed User Space UI",
        "storage": "using authorized mounts or object storage",
        "workspace": "editing, validating, and snapshotting a User Space workspace",
    }
    description = f"Use when {tasks.get(topic, 'working in User Space')}."
    return f"---\nname: ragtime-{topic}\ndescription: {description}\n---\n\n"


def _skill(topic: str, text: str) -> tuple[str, list[str]]:
    """Return a complete native skill or explicit bounded references.

    Inline delivery avoids a second client-side reference-loading hop.  When a
    future canonical topic cannot fit, its exact text remains available in
    ordered, concrete reference files rather than being summarized.
    """
    frontmatter = _skill_frontmatter(topic)
    inline = f"{frontmatter}{text}"
    if len(inline.encode("utf-8")) <= MAX_NATIVE_SKILL_BYTES:
        return inline, []

    chunks = _split_text(text)
    links = "\n".join(f"{index}. [Reference {index:03d}](references/{index:03d}.md)" for index in range(1, len(chunks) + 1))
    skill = f"{frontmatter}This guidance is too large to load inline. Read every required reference below in listed order before acting:\n\n{links}\n"
    if len(skill.encode("utf-8")) > MAX_NATIVE_SKILL_BYTES:
        raise ValueError("Native skill reference index exceeds delivery limit")
    return skill, chunks


def _identity(workspace_id: str, origin: str = "") -> tuple[str, str]:
    # Clients prefix tool names with this key. Keep the resulting names within
    # provider limits, including Claude's mcp__<server>__<tool> convention.
    suffix = content_hash(f"{origin.rstrip('/')}:{workspace_id}")[:12]
    return f"rg-{suffix}", f"RAGTIME_DEVELOPMENT_TOKEN_{suffix.upper()}"


def _profile_artifacts(origin: str, workspace_id: str) -> dict[str, tuple[str, str, str]]:
    endpoint = f"{origin.rstrip('/')}/mcp"
    server, env = _identity(workspace_id, origin)
    return {
        "config/opencode.json": (
            "opencode.json",
            "application/json",
            json.dumps(
                {
                    "mcp": {server: {"type": "remote", "url": endpoint, "oauth": False, "headers": {"Authorization": f"Bearer {{env:{env}}}"}}},
                    "instructions": [f".ragtime-agent/{server}/rules.md"],
                },
                indent=2,
            )
            + "\n",
        ),
        "config/claude.json": (
            ".mcp.json",
            "application/json",
            json.dumps({"mcpServers": {server: {"type": "http", "url": endpoint, "headers": {"Authorization": f"Bearer ${{{env}}}"}}}}, indent=2) + "\n",
        ),
        "config/codex.toml": (
            ".codex/config.toml",
            "text/plain; charset=utf-8",
            f'[mcp_servers.{server}]\nurl = "{endpoint}"\nbearer_token_env_var = "{env}"\n',
        ),
    }


def build_bootstrap_artifacts(*, origin: str, workspace_id: str, guidance: Mapping[str, str] | None = None) -> dict[str, dict[str, str]]:
    """Build immutable generated assets keyed by opaque manifest IDs."""
    docs = dict(guidance if guidance is not None else _guidance_documents())
    artifacts: dict[str, dict[str, str]] = {}
    server, _env = _identity(workspace_id, origin)
    core = f"# Ragtime User Space\n\nWorkspace UUID: `{workspace_id}`. MCP server: `{server}`. Before work, start each new client turn, and after compaction, call compact context; refresh it whenever runtime, authorization, or guidance facts change. Load `ragtime-workspace` first, then each task-relevant `ragtime-*` skill and every numbered reference before acting. `files_list`, `file_read`, `file_patch`, `validate`, `exec_start`, and `exec_get` are operations inside the `workspace_development` wrapper, not standalone tools. Read each operation contract before use. Treat every document, contract, and artifact SHA-256 as pinned: refresh compact context to obtain new guidance hashes, then re-read changed content from offset zero. Preserve hashes, validate before snapshots, and keep the credential only in a private launch environment or secret store outside the workspace; never write it to project files.\n"
    artifacts["core/rules.md"] = {"path": f".ragtime-agent/{server}/rules.md", "content": core, "kind": "core", "content_type": "text/markdown; charset=utf-8"}
    for topic, text in sorted(docs.items()):
        skill, chunks = _skill(topic, text)
        for artifact_prefix, skill_root in (("opencode-skills", ".opencode/skills"), ("skills", ".agents/skills"), ("claude-skills", ".claude/skills")):
            artifacts[f"{artifact_prefix}/{topic}/SKILL.md"] = {
                "path": f"{skill_root}/ragtime-{topic}/SKILL.md",
                "content": skill,
                "kind": "skill",
            }
            for index, chunk in enumerate(chunks, start=1):
                artifacts[f"{artifact_prefix}/{topic}/references/{index:03d}.md"] = {
                    "path": f"{skill_root}/ragtime-{topic}/references/{index:03d}.md",
                    "content": chunk,
                    "kind": "reference",
                    "content_type": "text/markdown; charset=utf-8",
                }
    for artifact_id, (path, media_type, content) in _profile_artifacts(origin, workspace_id).items():
        artifacts[artifact_id] = {"path": path, "content": content, "kind": "config", "content_type": media_type}
    return artifacts


def build_bootstrap_manifest(*, origin: str, workspace_id: str, scopes: frozenset[str] | set[str], guidance: Mapping[str, str] | None = None) -> dict[str, Any]:
    artifacts = build_bootstrap_artifacts(origin=origin, workspace_id=workspace_id, guidance=guidance)
    server, _env = _identity(workspace_id, origin)
    revision = content_hash({key: item["content"] for key, item in sorted(artifacts.items())})
    base = f"{origin.rstrip('/')}/indexes/userspace/development/workspaces/{workspace_id}"
    entries = [
        {
            "id": key,
            "path": item["path"],
            "kind": item["kind"],
            "content_type": item.get("content_type", "text/plain; charset=utf-8"),
            "bytes": len(item["content"].encode("utf-8")),
            "sha256": content_hash(item["content"]),
            "download_url": f"{base}/bootstrap/files/{key}?revision={revision}",
        }
        for key, item in sorted(artifacts.items())
    ]

    def profile_assets(config_id: str, skill_prefix: str) -> list[str]:
        return ["core/rules.md", config_id, *sorted(key for key in artifacts if key.startswith(skill_prefix))]

    def profile_destinations(artifact_ids: list[str]) -> dict[str, str]:
        return {artifact_id: artifacts[artifact_id]["path"] for artifact_id in artifact_ids}

    opencode_assets = profile_assets("config/opencode.json", "opencode-skills/")
    claude_assets = profile_assets("config/claude.json", "claude-skills/")
    codex_assets = profile_assets("config/codex.toml", "skills/")
    return {
        "workspace_id": workspace_id,
        "guidance_revision": revision,
        "credential_env_var": _identity(workspace_id, origin)[1],
        "scopes": sorted(scopes),
        "artifacts": entries,
        "profiles": {
            "opencode": {
                "artifact_ids": opencode_assets,
                "destinations": profile_destinations(opencode_assets),
                "config_artifact": "config/opencode.json",
                "skill_root": ".opencode/skills",
                "core_include": f".ragtime-agent/{server}/rules.md",
                "restart_required": True,
                "merge": "merge only the workspace-specific mcp entry and the exact managed core include in opencode.json instructions; preserve OPENCODE_DISABLE_EXTERNAL_SKILLS and all other user configuration",
            },
            "claude-code": {
                "artifact_ids": claude_assets,
                "destinations": profile_destinations(claude_assets),
                "config_artifact": "config/claude.json",
                "skill_root": ".claude/skills",
                "instruction_file": "CLAUDE.md",
                "managed_core_markers": {
                    "start": "<!-- RAGTIME managed core start -->",
                    "end": "<!-- RAGTIME managed core end -->",
                    "include": f"@.ragtime-agent/{server}/rules.md",
                },
                "restart_required": True,
                "merge": "replace exactly one managed-core marker block (deduplicate duplicate blocks) and merge only the workspace-specific mcpServers entry; preserve all other user content",
            },
            "codex": {
                "artifact_ids": codex_assets,
                "destinations": profile_destinations(codex_assets),
                "config_artifact": "config/codex.toml",
                "skill_root": ".agents/skills",
                "instruction_file": "AGENTS.md",
                "managed_core_markers": {
                    "start": "<!-- RAGTIME managed core start -->",
                    "end": "<!-- RAGTIME managed core end -->",
                    "include": f"@.ragtime-agent/{server}/rules.md",
                },
                "restart_required": True,
                "merge": "replace exactly one managed-core marker block (deduplicate duplicate blocks) and merge only the workspace-specific table into .codex/config.toml; preserve all other user content",
            },
        },
        "context_url": f"{base}/context",
        "mcp_url": f"{origin.rstrip('/')}/mcp",
        "reader_protocol": "Call compact context, then page each referenced document or contract using its sha256 revision until next_offset is null. Every call is reauthorized.",
        "install": "Keep the credential only in a private launch environment or secret store outside the workspace, and set credential_env_var when launching the client. Fetch only the chosen profile's artifact_ids with Authorization: Bearer token; verify bytes and sha256; write only their declared destinations; merge only the named workspace-specific config entry; replace and deduplicate only the declared RAGTIME managed-core marker block; preserve every other user rule, provider, config, and discovery-security flag; start a new client session.",
    }


def get_bootstrap_artifact(
    *, artifact_id: str, revision: str | None, origin: str, workspace_id: str, guidance: Mapping[str, str] | None = None
) -> dict[str, Any]:
    manifest = build_bootstrap_manifest(origin=origin, workspace_id=workspace_id, scopes=frozenset(), guidance=guidance)
    if revision is not None and revision != manifest["guidance_revision"]:
        raise HTTPException(status_code=409, detail={"code": "stale_revision", "revision": manifest["guidance_revision"]})
    artifacts = build_bootstrap_artifacts(origin=origin, workspace_id=workspace_id, guidance=guidance)
    item = artifacts.get(artifact_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Bootstrap artifact not found")
    return {
        "id": artifact_id,
        "path": item["path"],
        "kind": item["kind"],
        "content_type": item.get("content_type", "text/plain; charset=utf-8"),
        "sha256": content_hash(item["content"]),
        "content": item["content"],
        "guidance_revision": manifest["guidance_revision"],
    }


def _split_text(text: str) -> list[str]:
    encoded = text.encode("utf-8")
    chunks: list[str] = []
    offset = 0
    while offset < len(encoded):
        raw = encoded[offset : offset + MAX_PAGE_TEXT_BYTES]
        try:
            chunk = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            chunk = raw[: exc.start].decode("utf-8")
        if not chunk:
            raise ValueError("Guidance has an invalid UTF-8 boundary")
        chunks.append(chunk)
        offset += len(chunk.encode("utf-8"))
    return chunks or [""]


def serialize_mcp_payload(value: Any, limit: int | None = MAX_MCP_RESPONSE_BYTES) -> str:
    # The exact JSON text emitted by MCP uses this serializer.
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    if limit is not None and len(text.encode("utf-8")) > limit:
        raise HTTPException(status_code=413, detail="MCP response exceeds bounded delivery limit")
    return text


def _bounded(value: Any, limit: int = MAX_MCP_RESPONSE_BYTES) -> dict[str, Any]:
    serialize_mcp_payload(value, limit)
    return value


def _catalog(full_context: Mapping[str, Any]) -> dict[str, Any]:
    bundle = full_context.get("instruction_bundle", full_context)
    if not isinstance(bundle, Mapping):
        bundle = {}
    capabilities = bundle.get("capabilities", {})
    return dict(capabilities) if isinstance(capabilities, Mapping) else {}


def build_compact_context(
    full_context: Mapping[str, Any],
    *,
    guidance: Mapping[str, str] | None = None,
    operations: list[dict[str, Any]] | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Return the MCP-safe context index; use readers for all substantial text."""
    docs = dict(guidance if guidance is not None else _guidance_documents())
    catalog = _catalog(full_context)
    contracts = operations or []
    bundle = full_context.get("instruction_bundle", full_context)
    bundle = bundle if isinstance(bundle, Mapping) else {}
    facts, fresh_status = _facts_payload(full_context)
    workspace = facts.get("workspace", {})
    resolved_workspace_id = workspace_id or (workspace.get("id") if isinstance(workspace, Mapping) else None) or full_context.get("workspace_id")
    document_refs = [
        {
            "id": key,
            "sha256": content_hash(value),
            "bytes": len(value.encode("utf-8")),
            "read_arguments": {"workspace_id": resolved_workspace_id, "document_id": key, "revision": content_hash(value), "offset": 0},
        }
        for key, value in sorted(docs.items())
    ]
    facts_json = serialize_mcp_payload(facts, limit=None)
    facts_ref: dict[str, Any] | None = None
    if len(facts_json.encode("utf-8")) > 6 * 1024:
        facts_ref = {"reader": "workspace_development_facts", "sha256": content_hash(facts_json), "bytes": len(facts_json.encode("utf-8"))}
        facts = {"workspace": bundle.get("workspace", {}), "user": bundle.get("user", {}), "facts": {}}
    result = {
        "context_revision": full_context.get("context_revision"),
        "guidance_documents": document_refs,
        "operation_contracts": [
            {
                "name": item.get("name"),
                "sha256": content_hash(item),
                "scope": item.get("scope"),
                "read_arguments": {"workspace_id": resolved_workspace_id, "operation": item.get("name"), "revision": content_hash(item), "offset": 0},
            }
            for item in contracts
        ],
        **facts,
        "facts_reference": facts_ref,
        "fresh_status": _bounded_fresh_status(fresh_status),
        "next_steps": [
            "Read required skills and their references.",
            "Use operation contracts before calling an operation.",
            "Refresh context after relevant live facts change.",
        ],
        "resources": {
            "total_tools": len(catalog.get("authorized_tools", [])),
            "total_indexes": len(catalog.get("authorized_indexes", [])),
            "reader": "workspace_development_resources",
        },
        "readers": {
            "document": "workspace_development_document",
            "operation_contract": "workspace_development_contract",
            "resources": "workspace_development_resources",
            "resource_description": "workspace_development_resource_description",
            "resource_contract": "workspace_development_resource_contract",
            "facts": "workspace_development_facts",
        },
    }
    return _bounded(result)


def _require_continuation_pin(revision: str | None, offset: int) -> None:
    if offset > 0 and not revision:
        raise HTTPException(status_code=422, detail="A target sha256 revision is required when offset is greater than zero")


def read_document(document_id: str, revision: str | None, *, guidance: Mapping[str, str] | None = None, offset: int = 0) -> dict[str, Any]:
    _require_continuation_pin(revision, offset)
    docs = dict(guidance if guidance is not None else _guidance_documents())
    content = docs.get(document_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Guidance document not found")
    digest = content_hash(content)
    if revision is not None and revision != digest:
        raise HTTPException(status_code=409, detail={"code": "stale_revision", "revision": digest})
    return _page_text({"id": document_id, "sha256": digest}, content, offset)


def _invalid_utf8_offset(encoded: bytes, offset: int) -> bool:
    try:
        encoded[:offset].decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


_VOLATILE_FACT_KEYS = frozenset({"runtime", "runtime_status", "diagnostics", "recent_failures"})


def _facts_payload(full_context: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = full_context.get("instruction_bundle", full_context)
    bundle = bundle if isinstance(bundle, Mapping) else {}
    raw_facts = bundle.get("facts", {})
    raw_facts = raw_facts if isinstance(raw_facts, Mapping) else {}
    static_facts = {key: value for key, value in raw_facts.items() if key not in _VOLATILE_FACT_KEYS}
    fresh = {key: raw_facts[key] for key in _VOLATILE_FACT_KEYS if key in raw_facts}
    for key in _VOLATILE_FACT_KEYS:
        if key in bundle:
            fresh[key] = bundle[key]
    return {"workspace": bundle.get("workspace", {}), "user": bundle.get("user", {}), "facts": static_facts}, fresh


def _bounded_fresh_status(status: Mapping[str, Any]) -> dict[str, Any]:
    candidate = dict(status)
    if len(serialize_mcp_payload(candidate, limit=None).encode("utf-8")) <= 2048:
        return candidate
    return {"available": bool(candidate), "truncated": True}


def read_context_facts(full_context: Mapping[str, Any], revision: str | None, *, offset: int = 0) -> dict[str, Any]:
    _require_continuation_pin(revision, offset)
    facts, _fresh = _facts_payload(full_context)
    text = serialize_mcp_payload(facts, limit=None)
    digest = content_hash(text)
    if revision is not None and revision != digest:
        raise HTTPException(status_code=409, detail={"code": "stale_revision", "revision": digest})
    return _page_text({"id": "facts", "sha256": digest}, text, offset)


def _page_text(metadata: dict[str, Any], text: str, offset: int) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    if offset < 0 or offset > len(encoded) or (offset and _invalid_utf8_offset(encoded, offset)):
        raise HTTPException(status_code=422, detail="Invalid UTF-8 page offset")
    raw = encoded[offset : offset + MAX_PAGE_TEXT_BYTES]
    try:
        maximum = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        maximum = raw[: exc.start].decode("utf-8")
    if offset < len(encoded) and not maximum:
        raise HTTPException(status_code=422, detail="Invalid UTF-8 page offset")
    low, high, best = 0, len(maximum), ""
    while low <= high:
        middle = (low + high) // 2
        candidate = maximum[:middle]
        next_offset = offset + len(candidate.encode("utf-8"))
        page = {
            **metadata,
            "offset": offset,
            "text": candidate,
            "next_offset": next_offset if next_offset < len(encoded) else None,
            "total_bytes": len(encoded),
        }
        if len(serialize_mcp_payload(page, limit=None).encode("utf-8")) <= MAX_MCP_RESPONSE_BYTES:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if offset < len(encoded) and not best:
        raise HTTPException(status_code=413, detail="MCP page metadata exceeds bounded delivery limit")
    next_offset = offset + len(best.encode("utf-8"))
    return _bounded(
        {**metadata, "offset": offset, "text": best, "next_offset": next_offset if next_offset < len(encoded) else None, "total_bytes": len(encoded)}
    )


def read_operation_contract(name: str, revision: str | None, *, operations: list[dict[str, Any]], offset: int = 0) -> dict[str, Any]:
    _require_continuation_pin(revision, offset)
    item = next((candidate for candidate in operations if candidate.get("name") == name), None)
    if item is None:
        raise HTTPException(status_code=404, detail="Operation contract not found")
    digest = content_hash(item)
    if revision is not None and revision != digest:
        raise HTTPException(status_code=409, detail={"code": "stale_revision", "revision": digest})
    return _page_text({"name": name, "sha256": digest}, serialize_mcp_payload(item, limit=None), offset)


def _resource_kind(resource: Any) -> str:
    return "tool" if isinstance(resource, Mapping) and (resource.get("component_id") is not None or resource.get("tool_type") is not None) else "index"


def _resource_id(resource: Any) -> str:
    identity = resource
    if isinstance(resource, Mapping):
        for key in ("component_id", "index_name", "id", "name"):
            if resource.get(key) is not None:
                identity = str(resource[key])
                break
    return f"{_resource_kind(resource)}:{content_hash(identity)}"


def _resource(full_context: Mapping[str, Any], resource_id: str) -> Any:
    catalog = _catalog(full_context)
    resource = next(
        (item for item in list(catalog.get("authorized_tools", [])) + list(catalog.get("authorized_indexes", [])) if _resource_id(item) == resource_id), None
    )
    if resource is None:
        raise HTTPException(status_code=404, detail="Resource not found")
    return resource


def _resource_summary(resource: Any) -> dict[str, Any]:
    value = dict(resource) if isinstance(resource, Mapping) else {}
    resource_id = _resource_id(value)
    summary: dict[str, Any] = {"resource_id": resource_id, "kind": _resource_kind(value)}
    for key in ("tool_type", "enabled"):
        if key in value and len(serialize_mcp_payload(value[key], limit=None).encode("utf-8")) <= 512:
            summary[key] = value[key]
    description = value.get("description")
    if isinstance(description, str):
        if len(description.encode("utf-8")) <= MAX_INLINE_DESCRIPTION_BYTES:
            summary["description"] = description
        else:
            summary["description"] = None
            summary["description_reference"] = {
                "reader": "workspace_development_resource_description",
                "resource_id": resource_id,
                "sha256": content_hash(description),
                "complete": True,
            }
    summary["contract_reference"] = {
        "reader": "workspace_development_resource_contract",
        "resource_id": resource_id,
        "sha256": content_hash(resource),
        "complete": True,
    }
    return summary


def read_resources(full_context: Mapping[str, Any], *, offset: int = 0, limit: int = 20) -> dict[str, Any]:
    catalog = _catalog(full_context)
    resources = list(catalog.get("authorized_tools", [])) + list(catalog.get("authorized_indexes", []))
    offset, limit = max(offset, 0), max(1, min(limit, 100))
    items: list[dict[str, Any]] = []
    for resource in resources[offset : offset + limit]:
        candidate = _resource_summary(resource)
        prospective = items + [candidate]
        next_offset = offset + len(prospective)
        page = {"items": prospective, "offset": offset, "next_offset": next_offset if next_offset < len(resources) else None, "total": len(resources)}
        if len(serialize_mcp_payload(page, limit=None).encode("utf-8")) > MAX_MCP_RESPONSE_BYTES:
            break
        items = prospective
    next_offset = offset + len(items)
    if not items and offset < len(resources):
        raise HTTPException(status_code=413, detail="A resource summary exceeds bounded delivery limit")
    return _bounded({"items": items, "offset": offset, "next_offset": next_offset if next_offset < len(resources) else None, "total": len(resources)})


def read_resource_description(full_context: Mapping[str, Any], *, resource_id: str, revision: str | None, offset: int = 0) -> dict[str, Any]:
    _require_continuation_pin(revision, offset)
    resource = _resource(full_context, resource_id)
    description = resource.get("description") if isinstance(resource, Mapping) else None
    if not isinstance(description, str):
        raise HTTPException(status_code=404, detail="Resource has no description")
    digest = content_hash(description)
    if revision is not None and revision != digest:
        raise HTTPException(status_code=409, detail={"code": "stale_revision", "revision": digest})
    return _page_text({"resource_id": resource_id, "sha256": digest}, description, offset)


def read_resource_contract(full_context: Mapping[str, Any], *, resource_id: str, revision: str | None, offset: int = 0) -> dict[str, Any]:
    _require_continuation_pin(revision, offset)
    resource = _resource(full_context, resource_id)
    digest = content_hash(resource)
    if revision is not None and revision != digest:
        raise HTTPException(status_code=409, detail={"code": "stale_revision", "revision": digest})
    return _page_text({"resource_id": resource_id, "sha256": digest}, serialize_mcp_payload(resource, limit=None), offset)
