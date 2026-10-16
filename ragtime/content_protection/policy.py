"""Pure policy resolution helpers."""

from collections.abc import Iterable

from ragtime.content_protection.models import ContentProtectionConfig, ProtectionContext


def resolve_required(
    config: ContentProtectionConfig,
    context: ProtectionContext,
    group_ids_by_user: dict[str, set[str]] | None = None,
    verified_user_ids: Iterable[str] | None = None,
) -> tuple[bool, str]:
    """Resolve every audience slot without allowing one audience to bypass another."""
    if not config.enabled:
        return False, "master_off"
    overrides = {item.user_id: item.mode for item in config.user_overrides}
    primary_identity = (context.user_id,) if context.user_id is not None else ()
    identities = tuple(dict.fromkeys((*primary_identity, *context.audience_user_ids)))
    allowed_identities = set(verified_user_ids) if verified_user_ids is not None else {identity for identity in identities if identity}
    verified = tuple(identity for identity in identities if identity and identity in allowed_identities)
    group_ids_by_user = group_ids_by_user or {}
    scope_matches: list[str] = []
    for requirement in config.requirements:
        if requirement.mode != "require" or requirement.scope_kind == "group":
            continue
        if requirement.scope_kind == "surface" and requirement.scope_key == context.surface:
            scope_matches.append("surface")
        elif requirement.scope_kind == "mcp_route" and requirement.scope_key == (context.mcp_route or "default"):
            scope_matches.append("mcp_route")
        elif requirement.scope_kind == "tool" and requirement.scope_key == context.tool_id:
            scope_matches.append("tool")
    scoped = config.coverage_mode == "all_supported_traffic" or bool(scope_matches)
    inherited_slots = 0
    skipped_slots = 0
    for identity in verified:
        override = overrides.get(identity, "inherit")
        if override == "always_classify":
            return True, "user_override:always_classify"
        if override == "never_classify":
            continue
        groups = group_ids_by_user.get(identity, set())
        if any(item.scope_kind == "group" and item.mode == "require" and item.scope_key in groups for item in config.requirements):
            return True, "group_override:require"
        if any(item.scope_kind == "group" and item.mode == "skip" and item.scope_key in groups for item in config.requirements):
            skipped_slots += 1
            continue
        inherited_slots += 1
    has_unverified_identity = any(identity not in allowed_identities for identity in identities)
    has_baseline_slot = context.public or context.baseline in {"anonymous", "service", "public"}
    if not verified and not identities:
        has_baseline_slot = True
    if has_unverified_identity or has_baseline_slot:
        inherited_slots += 1
    if scoped and inherited_slots:
        return True, "scope_requirement"
    if skipped_slots and not inherited_slots:
        return False, "group_override:skip"
    if verified and not inherited_slots:
        return False, "user_override:never_classify"
    return False, "no_requirement"
