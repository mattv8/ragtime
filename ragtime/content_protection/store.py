"""Persistence and authoritative identity reads for content protection."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from prisma import Json

from ragtime.content_protection.models import ContentProtectionConfig
from ragtime.core.database import get_db


async def load_config_record() -> ContentProtectionConfig:
    db = await get_db()
    record = await db.contentprotectionconfig.find_unique(where={"id": "default"})
    if record is None:
        return ContentProtectionConfig()
    return ContentProtectionConfig.model_validate(record.config)


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
    profile_ids = {profile.id for profile in config.profiles}
    if any(item.profile_id not in profile_ids for item in config.group_profiles):
        raise ValueError("unknown_profile_reference")
    group_ids = {item.group_id for item in config.group_profiles} | {item.scope_key for item in config.requirements if item.scope_kind == "group"}
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
