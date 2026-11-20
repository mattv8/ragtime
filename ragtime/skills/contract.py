"""Shared contract for instruction skills.

A skill is an admin-authored instruction file in the Agent Skills format
(``name``, ``description``, Markdown body) owned by exactly one tool config or
one custom MCP route. Chat prompts, the ``read_skill`` tool, MCP descriptions
and MCP resources all address skills through the URIs built here.
"""

from __future__ import annotations

import json
import re
from typing import Literal, TypedDict

SkillMode = Literal["always", "on_demand"]
SkillOwnerKind = Literal["tool", "route"]

SKILL_MODE_ALWAYS: SkillMode = "always"
SKILL_MODE_ON_DEMAND: SkillMode = "on_demand"
SKILL_OWNER_TOOL: SkillOwnerKind = "tool"
SKILL_OWNER_ROUTE: SkillOwnerKind = "route"

SKILL_NAME_MAX_LENGTH = 64
SKILL_DESCRIPTION_MAX_LENGTH = 1024
SKILL_CONTENT_MAX_LENGTH = 48_000
ALWAYS_INCLUDE_BUDGET_CHARS = 32_000
MAX_SKILLS_PER_OWNER = 50
MCP_INLINE_SKILL_TEXT_LIMIT = 1_500

READ_SKILL_TOOL_NAME = "read_skill"
SKILL_MARKDOWN_MIME_TYPE = "text/markdown"

SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

_URI_SEGMENT_BY_OWNER: dict[SkillOwnerKind, str] = {
    SKILL_OWNER_TOOL: "tools",
    SKILL_OWNER_ROUTE: "routes",
}
_OWNER_BY_URI_SEGMENT: dict[str, SkillOwnerKind] = {segment: owner for owner, segment in _URI_SEGMENT_BY_OWNER.items()}
_SKILL_URI_PATTERN = re.compile(r"^skill://(tools|routes)/([A-Za-z0-9_-]+)/([a-z0-9]+(?:-[a-z0-9]+)*)/SKILL\.md$")


class SkillRecord(TypedDict):
    """Enabled skill as carried on tool-config and route dictionaries at runtime."""

    id: str
    name: str
    description: str
    content: str
    mode: SkillMode
    sort_order: int


def build_skill_uri(owner_kind: SkillOwnerKind, owner_id: str, name: str) -> str:
    """Return the stable ``skill://`` URI of a skill's SKILL.md."""
    return f"skill://{_URI_SEGMENT_BY_OWNER[owner_kind]}/{owner_id}/{name}/SKILL.md"


def parse_skill_uri(uri: str) -> tuple[SkillOwnerKind, str, str] | None:
    """Split a skill URI into ``(owner_kind, owner_id, name)``; ``None`` when malformed."""
    match = _SKILL_URI_PATTERN.match((uri or "").strip())
    if match is None:
        return None
    segment, owner_id, name = match.groups()
    return _OWNER_BY_URI_SEGMENT[segment], owner_id, name


def render_skill_markdown(skill: SkillRecord) -> str:
    """Render a skill as a SKILL.md document with YAML frontmatter."""
    # JSON string literals are valid YAML scalars, which keeps quoting safe.
    frontmatter = f"---\nname: {skill['name']}\ndescription: {json.dumps(skill['description'], ensure_ascii=False)}\n---\n\n"
    return frontmatter + skill["content"].strip() + "\n"
