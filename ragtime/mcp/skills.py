"""MCP presentation and lookup of admin-provided instruction skills."""

from dataclasses import dataclass
from typing import Any

from ragtime.skills.contract import (
    MCP_INLINE_SKILL_TEXT_LIMIT,
    READ_SKILL_TOOL_NAME,
    SKILL_MODE_ALWAYS,
    SKILL_OWNER_ROUTE,
    SKILL_OWNER_TOOL,
    SkillRecord,
    build_skill_uri,
)

_LISTED_DESCRIPTION_MAX_LENGTH = 300
_READ_SKILL_DESCRIPTION_MAX_LENGTH = 8000
_ROUTE_OWNER_LABEL = "route"


@dataclass(frozen=True)
class RouteSkillSet:
    records: list[SkillRecord]
    inline_always: bool


def _short_description(description: str) -> str:
    single_line = " ".join(description.split())
    if len(single_line) <= _LISTED_DESCRIPTION_MAX_LENGTH:
        return single_line
    return single_line[: _LISTED_DESCRIPTION_MAX_LENGTH - 1] + "…"


def _fits_inline(skills: list[SkillRecord]) -> bool:
    return sum(len(skill["content"]) for skill in skills) <= MCP_INLINE_SKILL_TEXT_LIMIT


def _always_sections(skills: list[SkillRecord]) -> list[str]:
    return [f"### {skill['name']}\n{skill['content'].strip()}" for skill in skills]


def tool_skill_lines(config: dict[str, Any]) -> str:
    skills = config.get("skills") or []
    if not skills:
        return ""
    always = [skill for skill in skills if skill["mode"] == SKILL_MODE_ALWAYS]
    inline = _fits_inline(always)
    lines = ["Skills:"]
    for skill in skills:
        uri = build_skill_uri(SKILL_OWNER_TOOL, str(config["id"]), skill["name"])
        if skill["mode"] != SKILL_MODE_ALWAYS:
            lines.append(f"- {skill['name']}: {_short_description(skill['description'])} (load with {READ_SKILL_TOOL_NAME}, uri {uri})")
        elif inline:
            lines.append(f"- {skill['name']} (always applied; text below)")
        else:
            lines.append(f"- {skill['name']} (always applied): before using this tool, call {READ_SKILL_TOOL_NAME} with uri {uri}.")
    if always and inline:
        lines.extend(["", "Always follow these instructions when using this tool:", *_always_sections(always)])
    return "\n" + "\n".join(lines)


def visible_skills(definitions: list[Any], route_id: str | None, route_skills: list[SkillRecord]) -> dict[str, tuple[SkillRecord, str]]:
    visible: dict[str, tuple[SkillRecord, str]] = {}
    for definition in definitions:
        config = definition.tool_config
        if not config.get("id"):
            continue
        for skill in config.get("skills") or []:
            uri = build_skill_uri(SKILL_OWNER_TOOL, str(config["id"]), skill["name"])
            visible[uri] = (skill, f"tool {config.get('name', config['id'])}")
    if route_id:
        for skill in route_skills:
            uri = build_skill_uri(SKILL_OWNER_ROUTE, route_id, skill["name"])
            visible[uri] = (skill, _ROUTE_OWNER_LABEL)
    return visible


def _read_skill_description_line(uri: str, skill: SkillRecord, owner: str, inline_route_always: bool) -> str:
    line = f"- {skill['name']} ({owner}): {_short_description(skill['description'])} (uri {uri})"
    if skill["mode"] != SKILL_MODE_ALWAYS:
        return line
    if owner != _ROUTE_OWNER_LABEL:
        return line + " Always applies to that tool."
    if inline_route_always:
        return line + " Always applies; text above."
    return line + " Always applies; load it before using this server's tools."


def read_skill_description(visible: dict[str, tuple[SkillRecord, str]], *, inline_route_always: bool = True) -> str:
    route_always = [skill for skill, owner in visible.values() if owner == _ROUTE_OWNER_LABEL and skill["mode"] == SKILL_MODE_ALWAYS]
    inline_route_always = inline_route_always and _fits_inline(route_always)
    header = "Load an admin-provided instruction skill by URI."
    if route_always and inline_route_always:
        header += "\n\nAlways follow these instructions on this server:\n" + "\n".join(_always_sections(route_always)) + "\n\nSkills:"
    lines = [_read_skill_description_line(uri, skill, owner, inline_route_always) for uri, (skill, owner) in visible.items()]
    included: list[str] = []
    for index, line in enumerate(lines):
        remaining = len(lines) - index - 1
        suffix = f"\n… and {remaining} more; see tool descriptions or resources/list" if remaining else ""
        if len(header + "\n" + "\n".join(included + [line]) + suffix) > _READ_SKILL_DESCRIPTION_MAX_LENGTH:
            included.append(f"… and {len(lines) - index} more; see tool descriptions or resources/list")
            break
        included.append(line)
    return header + "\n" + "\n".join(included)


def route_instructions(route_id: str, skills: list[SkillRecord], *, inline_always: bool = True) -> str | None:
    if not skills:
        return None
    always = [skill for skill in skills if skill["mode"] == SKILL_MODE_ALWAYS]
    on_demand = [skill for skill in skills if skill["mode"] != SKILL_MODE_ALWAYS]
    parts = ["Admin-provided instructions for this MCP server."]
    if always and inline_always:
        parts.append("Always follow these instructions:\n" + "\n\n".join(_always_sections(always)))
    elif always:
        listing = []
        for skill in always:
            uri = build_skill_uri(SKILL_OWNER_ROUTE, route_id, skill["name"])
            listing.append(
                f"- {skill['name']} (always applies): {_short_description(skill['description'])} (uri {uri}). "
                "Load it with read_skill, or resources/read, before using this server's tools."
            )
        parts.append("\n".join(listing))
    if on_demand:
        listing = [f"Load these skills when relevant, with the {READ_SKILL_TOOL_NAME} tool when it is listed, otherwise resources/read:"]
        for skill in on_demand:
            uri = build_skill_uri(SKILL_OWNER_ROUTE, route_id, skill["name"])
            listing.append(f"- {skill['name']}: {_short_description(skill['description'])} (uri {uri})")
        parts.append("\n".join(listing))
    return "\n\n".join(parts)
