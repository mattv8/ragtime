"""Detection and transport-safe errors for contradictory group policies."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ragtime.content_protection.models import ContentProtectionConfig

_MAX_CONFLICT_ROWS = 50


class GroupPolicyConflictError(ValueError):
    """A candidate would leave users in both Skip and Require groups."""

    def __init__(self, detail: dict[str, Any]) -> None:
        self.code = "conflicting_group_policies"
        self.detail = detail
        super().__init__(self.code)


def group_policy_ids(config: ContentProtectionConfig) -> tuple[set[str], set[str]]:
    """Return the configured Skip and Require group identifiers."""
    skip = {item.scope_key for item in config.requirements if item.scope_kind == "group" and item.mode == "skip"}
    require = {item.scope_key for item in config.requirements if item.scope_kind == "group" and item.mode == "require"}
    return skip, require


def policy_conflict_detail(conflicts: list[dict[str, Any]], total: int, conflicted_group_ids: list[str] | None = None) -> dict[str, Any]:
    """Build the capped, stable admin API response payload."""
    return {
        "code": "conflicting_group_policies",
        "message": "Users cannot belong to both content-protection Skip and Require groups.",
        "conflicts": conflicts[:_MAX_CONFLICT_ROWS],
        "total": total,
        "truncated": total > _MAX_CONFLICT_ROWS,
        "conflicted_group_ids": conflicted_group_ids
        or sorted({group_id for row in conflicts for group_id in [*row["skip_group_ids"], *row["require_group_ids"]]}),
    }


def valid_membership_groups(memberships: list[Any], now: datetime | None = None) -> set[str]:
    """Return group IDs for memberships which have not expired."""
    current_time = now or datetime.now(UTC)
    group_ids: set[str] = set()
    for membership in memberships:
        expires_at = getattr(membership, "expiresAt", None)
        if expires_at is not None:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= current_time:
                continue
        group_ids.add(str(membership.groupId))
    return group_ids


def conflicts_for_user(config: ContentProtectionConfig, user: Any, group_ids: set[str]) -> list[dict[str, Any]]:
    """Return a single conflict row when an eligible user has both policy groups."""
    if bool(getattr(user, "disabled", False)):
        return []
    skip_ids, require_ids = group_policy_ids(config)
    matched_skip = sorted(group_ids & skip_ids)
    matched_require = sorted(group_ids & require_ids)
    if not matched_skip or not matched_require:
        return []
    return [
        {
            "user_id": str(user.id),
            "username": str(user.username),
            "display_name": getattr(user, "displayName", None),
            "skip_group_ids": matched_skip,
            "require_group_ids": matched_require,
        }
    ]


async def find_group_policy_conflicts(config: ContentProtectionConfig, db: Any, user_id: str | None = None) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Find all active membership conflicts, returning only a bounded preview."""
    skip_ids, require_ids = group_policy_ids(config)
    policy_group_ids = skip_ids | require_ids
    if not skip_ids or not require_ids:
        return [], 0, []
    users, memberships = await _load_conflict_population(db, policy_group_ids)
    memberships_by_user: dict[str, list[Any]] = {}
    for membership in memberships:
        memberships_by_user.setdefault(str(membership.userId), []).append(membership)
    rows: list[dict[str, Any]] = []
    for user in users:
        groups = valid_membership_groups(memberships_by_user.get(str(user.id), []))
        rows.extend(conflicts_for_user(config, user, groups))
    rows.sort(key=lambda row: (row["username"].lower(), row["username"], row["user_id"]))
    if user_id is not None:
        rows = [row for row in rows if row["user_id"] == user_id]
    conflicted_group_ids = sorted({group_id for row in rows for group_id in [*row["skip_group_ids"], *row["require_group_ids"]]})
    return rows[:_MAX_CONFLICT_ROWS], len(rows), conflicted_group_ids


async def _load_conflict_population(db: Any, policy_group_ids: set[str]) -> tuple[list[Any], list[Any]]:
    memberships = await db.authgroupmembership.find_many(where={"groupId": {"in": sorted(policy_group_ids)}})
    active_user_ids = sorted({str(membership.userId) for membership in memberships if valid_membership_groups([membership])})
    if not active_user_ids:
        return [], memberships
    users = await db.user.find_many(where={"id": {"in": active_user_ids}}, order={"username": "asc"})
    return users, memberships
