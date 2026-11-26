"""Code-defined instruction skills for tool configurations."""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ragtime.skills.contract import SkillMode, SkillRecord
from ragtime.skills.models import InstructionSkill

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuiltinSkill:
    name: str
    description: str
    mode: SkillMode
    topic: str
    applies: Callable[[dict], bool]
    reason_when_off: str


def _always(config: dict) -> bool:
    return True


def _linked_pdm(config: dict) -> bool:
    return bool((config.get("connection_config") or {}).get("pdm_tool_id"))


def _scripts_enabled(config: dict) -> bool:
    return bool((config.get("connection_config") or {}).get("allow_scripts"))


BUILTIN_SKILLS: dict[str, tuple[BuiltinSkill, ...]] = {
    "solidworks": (
        BuiltinSkill("solidworks-essentials", "Load before using SolidWorks: review essential call and safety guidance.", "always", "essentials", _always, ""),
        BuiltinSkill(
            "solidworks-find-model", "Load before searching for a model by catalog, product, part number or file name.", "on_demand", "find-model", _always, ""
        ),
        BuiltinSkill("solidworks-repair-assembly", "Load before diagnosing or repairing an assembly.", "on_demand", "repair-assembly", _always, ""),
        BuiltinSkill("solidworks-assemblies", "Load before inspecting or working with assemblies and mates.", "on_demand", "assemblies", _always, ""),
        BuiltinSkill("solidworks-features-and-sketches", "Load before inspecting features or sketches.", "on_demand", "features-and-sketches", _always, ""),
        BuiltinSkill("solidworks-documents", "Load before opening, managing or exporting documents.", "on_demand", "documents", _always, ""),
        BuiltinSkill("solidworks-pdm", "Load before working with the linked PDM vault.", "on_demand", "pdm", _linked_pdm, "Needs a linked PDM vault"),
        BuiltinSkill("solidworks-studying-models", "Load before studying a model's structure or properties.", "on_demand", "studying-models", _always, ""),
        BuiltinSkill(
            "solidworks-scripts", "Load before writing or running SolidWorks scripts.", "on_demand", "scripts", _scripts_enabled, "Needs Allow scripts"
        ),
        BuiltinSkill("solidworks-previews", "Load before producing a model preview.", "on_demand", "previews", _always, ""),
    ),
}


def _read_topic(topic: str) -> str:
    from ragtime.solidworks.guidance import read_topic_markdown  # inline-import: keep

    return read_topic_markdown(topic)


def builtin_skills_for(config: dict, admin_skill_names: set[str]) -> list[SkillRecord]:
    records: list[SkillRecord] = []
    for index, skill in enumerate(BUILTIN_SKILLS.get(config.get("tool_type", ""), ())):
        if skill.name in admin_skill_names or not skill.applies(config):
            continue
        try:
            content = _read_topic(skill.topic)
        except Exception:
            logger.warning("Failed to load built-in skill %s", skill.name, exc_info=True)
            continue
        records.append(
            {
                "id": f"builtin:{skill.name}",
                "name": skill.name,
                "description": skill.description,
                "content": content,
                "mode": skill.mode,
                "sort_order": -100 + index,
            }
        )
    return records


def apply_builtin_skills(config: dict, admin_skill_names: set[str]) -> dict:
    return {**config, "skills": [*builtin_skills_for(config, admin_skill_names), *config.get("skills", [])]}


def list_builtin_skills(config: dict, all_admin_skills: Iterable[InstructionSkill]) -> list[dict[str, Any]]:
    names = {skill.name for skill in all_admin_skills}
    result = []
    for skill in BUILTIN_SKILLS.get(config.get("tool_type", ""), ()):
        applies = skill.applies(config)
        result.append(
            {
                "name": skill.name,
                "description": skill.description,
                "mode": skill.mode,
                "applies": applies,
                "reason": None if applies else skill.reason_when_off,
                "overridden": skill.name in names,
            }
        )
    return result
