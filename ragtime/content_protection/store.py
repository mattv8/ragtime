"""Persistence and authoritative identity reads for content protection."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Literal

from prisma import Json
from pydantic import BaseModel, ConfigDict, Field

from ragtime.content_protection.models import ContentProtectionConfig
from ragtime.core.database import get_db
from ragtime.core.logging import get_logger

logger = get_logger(__name__)


class _LegacyProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    level: int = Field(ge=0, le=2)
    scope: str = Field(min_length=1, max_length=4000)


class _LegacyGroupProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)


class _LegacyRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope_kind: Literal["group", "tool", "mcp_route", "surface"]
    scope_key: str = Field(min_length=1, max_length=256)
    mode: Literal["require", "inherit"]


class _LegacyUserOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1)
    mode: Literal["inherit", "always_classify", "never_classify"]


class _LegacyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(default=0, ge=0)
    enabled: bool = False
    classifier_model: str | None = None
    coverage_mode: Literal["all_supported_traffic", "selected_scopes"] = "all_supported_traffic"
    profiles: list[_LegacyProfile] = Field(default_factory=list)
    group_profiles: list[_LegacyGroupProfile] = Field(default_factory=list)
    requirements: list[_LegacyRequirement] = Field(default_factory=list)
    user_overrides: list[_LegacyUserOverride] = Field(default_factory=list)


async def load_config_record() -> ContentProtectionConfig:
    db = await get_db()
    record = await db.contentprotectionconfig.find_unique(where={"id": "default"})
    if record is None:
        return ContentProtectionConfig()
    payload = dict(record.config)
    if payload.get("schema_version") == 2:
        return ContentProtectionConfig.model_validate(_canonicalize_system_copy(payload))
    if not _is_recognizable_legacy_config(payload):
        raise ValueError("invalid_content_protection_config")
    return await _reset_legacy_config(db, record)


def _canonicalize_system_copy(payload: dict[str, Any]) -> dict[str, Any]:
    """Upgrade only server-owned rule wording while retaining policy structure."""
    categories = payload.get("categories")
    if not isinstance(categories, list):
        return payload
    canonical = ContentProtectionConfig().categories[-1].model_dump(mode="json")
    normalized = dict(payload)
    normalized_categories: list[Any] = []
    for category in categories:
        if isinstance(category, dict) and category.get("id") == "rule_override" and category.get("system") is True:
            normalized_categories.append(
                {
                    **category,
                    **{field: canonical[field] for field in ("name", "description", "includes", "excludes", "examples", "denial_message")},
                }
            )
        else:
            normalized_categories.append(category)
    normalized["categories"] = normalized_categories
    return normalized


def _is_recognizable_legacy_config(payload: dict[str, Any]) -> bool:
    """Only reset the previously persisted v1 document shape, never unknown data."""
    if "schema_version" in payload:
        return False
    try:
        _LegacyConfig.model_validate(payload)
    except ValueError:
        return False
    return True


async def _reset_legacy_config(db: Any, record: Any) -> ContentProtectionConfig:
    async with db.tx() as tx:
        await tx.execute_raw("SELECT pg_advisory_xact_lock(hashtext('content-protection-config'))")
        current = await tx.contentprotectionconfig.find_unique(where={"id": "default"})
        if current is None:
            return ContentProtectionConfig()
        current_payload = dict(current.config)
        if current_payload.get("schema_version") == 2:
            return ContentProtectionConfig.model_validate(current_payload)
        if not _is_recognizable_legacy_config(current_payload):
            raise ValueError("invalid_content_protection_config")
        reset = ContentProtectionConfig(legacy_reset=True, legacy_was_enabled=bool(current_payload.get("enabled", False)), revision=int(current.revision) + 1)
        updated = await tx.contentprotectionconfig.update_many(
            where={"id": "default", "revision": int(current.revision)},
            data={"revision": reset.revision, "config": Json(reset.model_dump(mode="json")), "updatedBy": None},
        )
        if int(getattr(updated, "count", updated)) != 1:
            raise ValueError("revision_conflict")
        logger.warning("content_protection_legacy_config_reset", extra={"content_protection": {"legacy_was_enabled": reset.legacy_was_enabled}})
        return reset


async def resolve_identities(user_ids: set[str]) -> tuple[set[str], dict[str, set[str]], dict[str, str | None]]:
    """Read users and unexpired group memberships from the authoritative store."""
    if not user_ids:
        return set(), {}, {}
    db = await get_db()
    users = await db.user.find_many(where={"id": {"in": sorted(user_ids)}})
    verified = {str(user.id) for user in users if not bool(getattr(user, "disabled", False))}
    memberships = await db.authgroupmembership.find_many(where={"userId": {"in": sorted(verified)}}) if verified else []
    now = datetime.now(UTC)
    groups: dict[str, set[str]] = {user_id: set() for user_id in verified}
    expiries: dict[str, str | None] = {}
    for membership in memberships:
        expires = getattr(membership, "expiresAt", None)
        if expires is not None:
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=UTC)
            if expires <= now:
                continue
        groups[str(membership.userId)].add(str(membership.groupId))
        current = expiries.get(str(membership.userId))
        encoded = expires.isoformat() if expires else None
        if current is None or (encoded is not None and encoded < current):
            expiries[str(membership.userId)] = encoded
    return verified, groups, expiries


async def validate_references(config: ContentProtectionConfig) -> None:
    """Reject dangling durable references before writing the policy document."""
    db = await get_db()
    group_ids = {item.group_id for item in config.group_access_levels} | {item.scope_key for item in config.requirements if item.scope_kind == "group"}
    user_ids = {item.user_id for item in config.user_overrides}
    tool_ids = {item.scope_key for item in config.requirements if item.scope_kind == "tool"}
    route_ids = {item.scope_key for item in config.requirements if item.scope_kind == "mcp_route" and item.scope_key != "default"}
    groups, users, tools, routes = await asyncio.gather(
        db.authgroup.find_many(where={"id": {"in": sorted(group_ids)}}) if group_ids else _empty(),
        db.user.find_many(where={"id": {"in": sorted(user_ids)}}) if user_ids else _empty(),
        db.toolconfig.find_many(where={"id": {"in": sorted(tool_ids)}}) if tool_ids else _empty(),
        db.mcprouteconfig.find_many(where={"id": {"in": sorted(route_ids)}}) if route_ids else _empty(),
    )
    # Runtime tools are durable DB identifiers. Static tools have canonical
    # registry IDs and are validated against the running registry instead.
    if tool_ids:
        from ragtime.tools.registry import get_all_tools

        tool_ids = {identifier for identifier in tool_ids if identifier not in get_all_tools()}
    if (
        group_ids != {str(row.id) for row in groups}
        or user_ids != {str(row.id) for row in users}
        or tool_ids != {str(row.id) for row in tools}
        or route_ids != {str(row.id) for row in routes}
    ):
        raise ValueError("unknown_policy_reference")


async def _empty() -> list[Any]:
    return []


async def save_config_record(config: ContentProtectionConfig, expected_revision: int, actor_id: str | None) -> ContentProtectionConfig:
    """Compare-and-swap the entire validated policy document."""
    await validate_references(config)
    db = await get_db()
    async with db.tx() as tx:
        # Serializes the missing-row case too, then update_many supplies an
        # explicit revision compare-and-swap for existing rows.
        await tx.execute_raw("SELECT pg_advisory_xact_lock(hashtext('content-protection-config'))")
        current = await tx.contentprotectionconfig.find_unique(where={"id": "default"})
        if current is not None and int(current.revision) != expected_revision:
            raise ValueError("revision_conflict")
        if current is None and expected_revision != 0:
            raise ValueError("revision_conflict")
        saved = config.model_copy(update={"revision": expected_revision + 1})
        payload = Json(saved.model_dump(mode="json"))
        if current is None:
            await tx.contentprotectionconfig.create(data={"id": "default", "revision": saved.revision, "config": payload, "updatedBy": actor_id})
            return saved
        updated = await tx.contentprotectionconfig.update_many(
            where={"id": "default", "revision": expected_revision},
            data={"revision": saved.revision, "config": payload, "updatedBy": actor_id},
        )
        if int(getattr(updated, "count", updated)) != 1:
            raise ValueError("revision_conflict")
    return saved
