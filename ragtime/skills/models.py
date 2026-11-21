"""Validation and API models for admin-authored instruction skills."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from ragtime.skills.contract import (
    SKILL_CONTENT_MAX_LENGTH,
    SKILL_DESCRIPTION_MAX_LENGTH,
    SKILL_NAME_MAX_LENGTH,
    SKILL_NAME_PATTERN,
    SkillMode,
    SkillOwnerKind,
)


class CreateInstructionSkill(BaseModel):
    name: str = Field(min_length=1, max_length=SKILL_NAME_MAX_LENGTH)
    description: str = Field(min_length=1, max_length=SKILL_DESCRIPTION_MAX_LENGTH)
    content: str = Field(max_length=SKILL_CONTENT_MAX_LENGTH)
    mode: SkillMode = "on_demand"
    enabled: bool = True
    sort_order: int = 0

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not SKILL_NAME_PATTERN.fullmatch(value):
            raise ValueError("Name must contain lowercase letters and numbers separated by single hyphens")
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Description must not be blank")
        return value


class UpdateInstructionSkill(BaseModel):
    name: str | None = None
    description: str | None = None
    content: str | None = None
    mode: SkillMode | None = None
    enabled: bool | None = None
    sort_order: int | None = None


class InstructionSkill(CreateInstructionSkill):
    id: str
    owner_kind: SkillOwnerKind
    owner_id: str
    uri: str
    created_at: datetime
    updated_at: datetime


class ReorderInstructionSkills(BaseModel):
    skill_ids: list[str]


class CopyInstructionSkills(BaseModel):
    source_tool_id: str
