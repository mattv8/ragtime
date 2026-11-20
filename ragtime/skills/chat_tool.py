"""Request-scoped loader for tool-owned instruction skills."""

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ragtime.skills.contract import (
    READ_SKILL_TOOL_NAME,
    SKILL_MODE_ON_DEMAND,
    SKILL_OWNER_TOOL,
    SkillRecord,
    build_skill_uri,
    render_skill_markdown,
)


class _ReadInstructionSkillInput(BaseModel):
    uri: str = Field(description="The skill:// URI listed under SYSTEM TOOLS.")


def build_read_skill_tool(tool_configs: list[dict[str, Any]]) -> StructuredTool | None:
    """Build a loader restricted to instruction skills of visible tools."""
    instruction_skills: dict[str, SkillRecord] = {}
    has_on_demand = False
    for config in tool_configs:
        for instruction_skill in config.get("skills") or []:
            uri = build_skill_uri(SKILL_OWNER_TOOL, str(config.get("id") or ""), instruction_skill["name"])
            instruction_skills[uri] = instruction_skill
            has_on_demand |= instruction_skill["mode"] == SKILL_MODE_ON_DEMAND
    if not has_on_demand:
        return None

    async def read_instruction_skill(uri: str) -> str:
        instruction_skill = instruction_skills.get(uri)
        if instruction_skill is None:
            return json.dumps({"error": "Unknown instruction skill URI", "valid_uris": list(instruction_skills)}, ensure_ascii=False)
        return render_skill_markdown(instruction_skill)

    return StructuredTool.from_function(
        coroutine=read_instruction_skill,
        name=READ_SKILL_TOOL_NAME,
        description=(
            "Load admin-provided instruction skills listed under SYSTEM TOOLS by URI. "
            "This is unrelated to search_tool_skills and load_tool_skills, which load optional tools."
        ),
        args_schema=_ReadInstructionSkillInput,
        metadata={"instruction_skill_loader": True},
    )
