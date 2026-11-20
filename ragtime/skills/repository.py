"""Database access and shared validation for instruction skills."""

from typing import Any, cast

from prisma.models import InstructionSkill as PrismaInstructionSkill

from ragtime.core.database import get_db
from ragtime.skills.contract import ALWAYS_INCLUDE_BUDGET_CHARS, MAX_SKILLS_PER_OWNER, SkillOwnerKind, build_skill_uri
from ragtime.skills.models import CreateInstructionSkill, InstructionSkill


def instruction_skill_response(record: PrismaInstructionSkill, owner_kind: SkillOwnerKind, owner_id: str) -> InstructionSkill:
    return InstructionSkill(
        id=record.id,
        owner_kind=owner_kind,
        owner_id=owner_id,
        name=record.name,
        description=record.description,
        content=record.content,
        mode=cast(Any, record.mode),
        enabled=record.enabled,
        sort_order=record.sortOrder,
        uri=build_skill_uri(owner_kind, owner_id, record.name),
        created_at=record.createdAt,
        updated_at=record.updatedAt,
    )


def validate_instruction_skills(instruction_skills: list[CreateInstructionSkill], owner_kind: SkillOwnerKind) -> None:
    if len(instruction_skills) > MAX_SKILLS_PER_OWNER:
        raise ValueError(f"An owner may have at most {MAX_SKILLS_PER_OWNER} skills.")
    if len({item.name for item in instruction_skills}) != len(instruction_skills):
        raise ValueError("Duplicate skill name.")
    total = sum(len(item.content) for item in instruction_skills if item.enabled and item.mode == "always")
    if total > ALWAYS_INCLUDE_BUDGET_CHARS:
        raise ValueError(f"Always-include skills for this {owner_kind} total {total:,} characters; the limit is {ALWAYS_INCLUDE_BUDGET_CHARS:,}.")


def _owner_filter(owner_kind: SkillOwnerKind, owner_id: str) -> dict[str, str]:
    return {"toolConfigId" if owner_kind == "tool" else "mcpRouteId": owner_id}


async def owner_exists(owner_kind: SkillOwnerKind, owner_id: str) -> bool:
    db = await get_db()
    if owner_kind == "tool":
        return await db.toolconfig.find_unique(where={"id": owner_id}) is not None
    return await db.mcprouteconfig.find_unique(where={"id": owner_id}) is not None


async def list_instruction_skills(owner_kind: SkillOwnerKind, owner_id: str) -> list[InstructionSkill]:
    db = await get_db()
    records = await db.instructionskill.find_many(where=cast(Any, _owner_filter(owner_kind, owner_id)))
    return sorted(
        (instruction_skill_response(record, owner_kind, owner_id) for record in records),
        key=lambda item: (item.sort_order, item.name),
    )


def _instruction_skill_data(instruction_skill: CreateInstructionSkill) -> dict[str, Any]:
    data = instruction_skill.model_dump()
    data["sortOrder"] = data.pop("sort_order")
    return data


async def create_instruction_skills(owner_kind: SkillOwnerKind, owner_id: str, instruction_skills: list[CreateInstructionSkill]) -> None:
    if instruction_skills:
        db = await get_db()
        data = [{**_instruction_skill_data(item), **_owner_filter(owner_kind, owner_id)} for item in instruction_skills]
        await db.instructionskill.create_many(data=cast(Any, data))


async def create_instruction_skill(owner_kind: SkillOwnerKind, owner_id: str, instruction_skill: CreateInstructionSkill) -> InstructionSkill:
    db = await get_db()
    data = {**_instruction_skill_data(instruction_skill), **_owner_filter(owner_kind, owner_id)}
    record = await db.instructionskill.create(data=cast(Any, data))
    return instruction_skill_response(record, owner_kind, owner_id)


async def update_instruction_skill(owner_kind: SkillOwnerKind, owner_id: str, skill_id: str, instruction_skill: CreateInstructionSkill) -> InstructionSkill:
    db = await get_db()
    record = await db.instructionskill.update(where={"id": skill_id}, data=cast(Any, _instruction_skill_data(instruction_skill)))
    assert record is not None
    return instruction_skill_response(record, owner_kind, owner_id)


async def delete_instruction_skill(skill_id: str) -> None:
    db = await get_db()
    await db.instructionskill.delete(where={"id": skill_id})


async def reorder_instruction_skills(skill_ids: list[str]) -> None:
    db = await get_db()
    async with db.tx() as transaction:
        for index, skill_id in enumerate(skill_ids):
            await transaction.instructionskill.update(where={"id": skill_id}, data={"sortOrder": index})


async def instruction_skill_summary() -> dict[str, dict[str, dict[str, int]]]:
    db = await get_db()
    records = await db.query_raw(
        """
        SELECT tool_config_id, mcp_route_id,
               COUNT(*) AS total, COUNT(*) FILTER (WHERE enabled) AS enabled
        FROM instruction_skills
        GROUP BY tool_config_id, mcp_route_id
        """
    )
    summary: dict[str, dict[str, dict[str, int]]] = {"tools": {}, "routes": {}}
    for record in records:
        group = "tools" if record["tool_config_id"] is not None else "routes"
        owner_id = record["tool_config_id"] or record["mcp_route_id"]
        assert owner_id is not None
        summary[group][owner_id] = {"total": int(record["total"]), "enabled": int(record["enabled"])}
    return summary
