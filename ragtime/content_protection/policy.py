"""Pure policy resolution helpers."""

from collections.abc import Iterable

from ragtime.content_protection.models import ContentProtectionConfig, ProtectionContext


def resolve_required(
    config: ContentProtectionConfig,
    context: ProtectionContext,
    group_ids_by_user: dict[str, set[str]] | None = None,
    verified_user_ids: Iterable[str] | None = None,
) -> tuple[bool, str]:
    """Resolve additive scope coverage without treating a user bypass as shared."""
    if not config.enabled:
        return False, "master_off"
    overrides = {item.user_id: item.mode for item in config.user_overrides}
    identities = tuple(dict.fromkeys((context.user_id, *context.audience_user_ids)))
    allowed_identities = set(verified_user_ids) if verified_user_ids is not None else {identity for identity in identities if identity}
    verified = tuple(identity for identity in identities if identity and identity in allowed_identities)
    modes = [overrides.get(identity, "inherit") for identity in verified]
    if "always_classify" in modes:
        return True, "user_override:always_classify"
    group_ids_by_user = group_ids_by_user or {}
    scope_matches = []
    for requirement in config.requirements:
        if requirement.mode != "require":
            continue
        if requirement.scope_kind == "surface" and requirement.scope_key == context.surface:
            scope_matches.append("surface")
        elif requirement.scope_kind == "mcp_route" and requirement.scope_key == (context.mcp_route or "default"):
            scope_matches.append("mcp_route")
        elif requirement.scope_kind == "tool" and requirement.scope_key == context.tool_id:
            scope_matches.append("tool")
        elif requirement.scope_kind == "group" and any(requirement.scope_key in group_ids_by_user.get(identity, set()) for identity in verified):
            scope_matches.append("group")
    scoped = config.coverage_mode == "all_supported_traffic" or bool(scope_matches)
    # Never can only suppress its own requirement; anonymous/service never gain it.
    if scoped and any(mode != "never_classify" for mode in modes) or (scoped and not verified):
        return True, "scope_requirement"
    return False, "user_override:never_classify" if modes else "no_requirement"
