"""Administrative instruction skill endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from prisma.errors import UniqueViolationError
from pydantic import ValidationError

from ragtime.core.app_settings import invalidate_settings_cache
from ragtime.core.security import require_admin
from ragtime.indexer.repository import repository as tool_repository
from ragtime.mcp.server import notify_tools_changed
from ragtime.rag import rag
from ragtime.skills import repository
from ragtime.skills.builtin import list_builtin_skills
from ragtime.skills.contract import SkillOwnerKind
from ragtime.skills.models import CopyInstructionSkills, CreateInstructionSkill, InstructionSkill, ReorderInstructionSkills, UpdateInstructionSkill

router = APIRouter(prefix="/indexes", tags=["Instruction Skills"], dependencies=[Depends(require_admin)])


async def _list_owned(owner_kind: SkillOwnerKind, owner_id: str) -> list[InstructionSkill]:
    if not await repository.owner_exists(owner_kind, owner_id):
        raise HTTPException(status_code=404, detail=f"{owner_kind.capitalize()} not found")
    return await repository.list_instruction_skills(owner_kind, owner_id)


async def _refresh(owner_kind: SkillOwnerKind) -> None:
    if owner_kind == "tool":
        invalidate_settings_cache()
    notify_tools_changed()
    if owner_kind == "tool":
        await rag.initialize()


def _validate(instruction_skills: list[CreateInstructionSkill], owner_kind: SkillOwnerKind) -> None:
    try:
        repository.validate_instruction_skills(instruction_skills, owner_kind)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _check_names(existing: list[InstructionSkill], incoming: list[CreateInstructionSkill]) -> None:
    names = {item.name for item in existing}
    if any(item.name in names for item in incoming):
        raise HTTPException(status_code=409, detail="Instruction skill name already exists")


def _find_instruction_skill(existing: list[InstructionSkill], skill_id: str) -> InstructionSkill:
    for item in existing:
        if item.id == skill_id:
            return item
    raise HTTPException(status_code=404, detail="Instruction skill not found")


def _append_after(existing: list[InstructionSkill], incoming: list[CreateInstructionSkill]) -> list[CreateInstructionSkill]:
    """Place incoming skills after the owner's existing ones, keeping their relative order."""
    first_sort_order = max((item.sort_order for item in existing), default=-1) + 1
    return [item.model_copy(update={"sort_order": first_sort_order + offset}) for offset, item in enumerate(incoming)]


def _validation_detail(error: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors())


def _tool_owner_id(tool_id: str) -> str:
    return tool_id


def _route_owner_id(route_id: str) -> str:
    return route_id


def _register_owner_routes(path: str, owner_kind: SkillOwnerKind) -> None:
    owner_dependency = _tool_owner_id if owner_kind == "tool" else _route_owner_id

    async def list_instruction_skills(owner_id: str = Depends(owner_dependency)) -> list[InstructionSkill]:
        return await _list_owned(owner_kind, owner_id)

    async def create_instruction_skill(request: CreateInstructionSkill, owner_id: str = Depends(owner_dependency)) -> InstructionSkill:
        existing = await _list_owned(owner_kind, owner_id)
        _check_names(existing, [request])
        _validate([*existing, request], owner_kind)
        if "sort_order" not in request.model_fields_set:
            request = _append_after(existing, [request])[0]
        try:
            result = await repository.create_instruction_skill(owner_kind, owner_id, request)
        except UniqueViolationError as exc:
            raise HTTPException(status_code=409, detail="Instruction skill name already exists") from exc
        await _refresh(owner_kind)
        return result

    async def reorder_instruction_skills(request: ReorderInstructionSkills, owner_id: str = Depends(owner_dependency)) -> list[InstructionSkill]:
        existing = await _list_owned(owner_kind, owner_id)
        if len(request.skill_ids) != len(existing) or set(request.skill_ids) != {item.id for item in existing}:
            raise HTTPException(status_code=400, detail="Skill IDs must match the owner's skills")
        await repository.reorder_instruction_skills(request.skill_ids)
        await _refresh(owner_kind)
        return await repository.list_instruction_skills(owner_kind, owner_id)

    async def copy_instruction_skills(request: CopyInstructionSkills, owner_id: str = Depends(owner_dependency)) -> list[InstructionSkill]:
        existing = await _list_owned("tool", owner_id)
        source = await _list_owned("tool", request.source_tool_id)
        incoming = _append_after(existing, [CreateInstructionSkill.model_validate(item.model_dump()) for item in source])
        _check_names(existing, incoming)
        _validate([*existing, *incoming], "tool")
        try:
            await repository.create_instruction_skills("tool", owner_id, incoming)
        except UniqueViolationError as exc:
            raise HTTPException(status_code=409, detail="Instruction skill name already exists") from exc
        await _refresh("tool")
        return await repository.list_instruction_skills("tool", owner_id)

    async def update_instruction_skill(skill_id: str, request: UpdateInstructionSkill, owner_id: str = Depends(owner_dependency)) -> InstructionSkill:
        existing = await _list_owned(owner_kind, owner_id)
        current = _find_instruction_skill(existing, skill_id)
        try:
            updated = CreateInstructionSkill.model_validate({**current.model_dump(), **request.model_dump(exclude_unset=True)})
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=_validation_detail(exc)) from exc
        others = [item for item in existing if item.id != skill_id]
        _check_names(others, [updated])
        _validate([*others, updated], owner_kind)
        try:
            result = await repository.update_instruction_skill(owner_kind, owner_id, skill_id, updated)
        except UniqueViolationError as exc:
            raise HTTPException(status_code=409, detail="Instruction skill name already exists") from exc
        await _refresh(owner_kind)
        return result

    async def delete_instruction_skill(skill_id: str, owner_id: str = Depends(owner_dependency)) -> dict[str, str]:
        existing = await _list_owned(owner_kind, owner_id)
        _find_instruction_skill(existing, skill_id)
        await repository.delete_instruction_skill(skill_id)
        await _refresh(owner_kind)
        return {"message": "Instruction skill deleted"}

    router.add_api_route(path, list_instruction_skills, methods=["GET"], response_model=list[InstructionSkill], name=f"list_{owner_kind}_instruction_skills")
    router.add_api_route(
        path, create_instruction_skill, methods=["POST"], status_code=201, response_model=InstructionSkill, name=f"create_{owner_kind}_instruction_skill"
    )
    router.add_api_route(
        path + "/order", reorder_instruction_skills, methods=["PUT"], response_model=list[InstructionSkill], name=f"reorder_{owner_kind}_instruction_skills"
    )
    if owner_kind == "tool":
        router.add_api_route(
            path + "/copy", copy_instruction_skills, methods=["POST"], response_model=list[InstructionSkill], name="copy_tool_instruction_skills"
        )
    router.add_api_route(
        path + "/{skill_id}", update_instruction_skill, methods=["PUT"], response_model=InstructionSkill, name=f"update_{owner_kind}_instruction_skill"
    )
    router.add_api_route(path + "/{skill_id}", delete_instruction_skill, methods=["DELETE"], name=f"delete_{owner_kind}_instruction_skill")


@router.get("/tools/{tool_id}/skills/builtin")
async def get_builtin_skills(tool_id: str) -> list[dict[str, Any]]:
    tool = await tool_repository.get_tool_config(tool_id)
    if tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    skills = await repository.list_instruction_skills("tool", tool_id)
    return list_builtin_skills(tool.model_dump(), skills)


_register_owner_routes("/tools/{tool_id}/skills", "tool")
_register_owner_routes("/mcp-routes/{route_id}/skills", "route")


@router.get("/skills/summary")
async def get_instruction_skill_summary() -> dict[str, Any]:
    return await repository.instruction_skill_summary()
