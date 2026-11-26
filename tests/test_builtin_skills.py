"""Built-in tool instruction skills."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI

from ragtime.core.app_settings import SettingsCache
from ragtime.core.security import require_admin
from ragtime.skills.builtin import BUILTIN_SKILLS, apply_builtin_skills, builtin_skills_for, list_builtin_skills
from ragtime.skills.contract import SKILL_NAME_PATTERN
from ragtime.skills.routes import router
from ragtime.solidworks.guidance import GUIDE_TOPICS, read_topic_markdown


@pytest.fixture
def config():
    return {"tool_type": "solidworks", "connection_config": {}, "skills": []}


def test_registry_shape():
    skills = BUILTIN_SKILLS["solidworks"]
    assert len(skills) == 10
    assert len({skill.name for skill in skills}) == len(skills)
    assert all(SKILL_NAME_PATTERN.fullmatch(skill.name) and len(skill.description) <= 160 for skill in skills)


def test_all_builtin_topics_resolve_to_real_guidance(config):
    config["connection_config"] = {"pdm_tool_id": "vault", "allow_scripts": True}
    skills = BUILTIN_SKILLS["solidworks"]
    topic_ids = {topic["id"] for topic in GUIDE_TOPICS}
    records = {record["name"]: record for record in builtin_skills_for(config, set())}
    assert len(records) == len(skills)
    for skill in skills:
        assert skill.topic in topic_ids
        content = records[skill.name]["content"]
        assert content.strip()
        assert content == read_topic_markdown(skill.topic)
    assert len(records["solidworks-essentials"]["content"]) <= 1400


def test_conditions_and_listing(config):
    listing = list_builtin_skills(config, cast(Any, [SimpleNamespace(name="solidworks-essentials", enabled=False)]))
    by_name = {item["name"]: item for item in listing}
    assert by_name["solidworks-essentials"]["overridden"]
    assert by_name["solidworks-pdm"]["reason"] == "Needs a linked PDM vault"
    assert by_name["solidworks-scripts"]["reason"] == "Needs Allow scripts"
    config["connection_config"] = {"pdm_tool_id": "vault", "allow_scripts": True}
    assert all(item["applies"] and item["reason"] is None for item in list_builtin_skills(config, []))
    assert list_builtin_skills({"tool_type": "unknown"}, []) == []


def test_records_order_and_overrides(config):
    admin = {"id": "admin", "name": "admin", "sort_order": -1000}
    config["skills"] = [admin]
    with patch("ragtime.skills.builtin._read_topic", side_effect=lambda topic: f"body:{topic}"):
        result = apply_builtin_skills(config, {"solidworks-essentials", "solidworks-documents"})
        records = result["skills"]
        assert records[-1] == admin
        assert records[0] == {
            "id": "builtin:solidworks-find-model",
            "name": "solidworks-find-model",
            "description": BUILTIN_SKILLS["solidworks"][1].description,
            "content": "body:find-model",
            "mode": "on_demand",
            "sort_order": -99,
        }
        assert not {"solidworks-essentials", "solidworks-documents"} & {item["name"] for item in records}
        assert builtin_skills_for({"tool_type": "unknown"}, set()) == []


def test_reader_failure_skips_only_failed_skill(config, caplog):
    def read(topic):
        if topic == "essentials":
            raise ValueError("missing")
        return topic

    with patch("ragtime.skills.builtin._read_topic", side_effect=read):
        records = builtin_skills_for(config, set())
    assert records and all(item["name"] != "solidworks-essentials" for item in records)
    assert "Failed to load built-in skill solidworks-essentials" in caplog.text


def test_settings_merge_includes_disabled_overrides():
    asyncio.run(_settings_merge_includes_disabled_overrides())


async def _settings_merge_includes_disabled_overrides():
    def skill(name, enabled):
        return SimpleNamespace(id=name, name=name, description="d", content="body", mode="always", sortOrder=0, enabled=enabled)

    cfg = SimpleNamespace(
        id="t",
        name="Tool",
        toolType="solidworks",
        description="",
        connectionConfig={},
        maxResults=10,
        allowWrite=False,
        instructionSkills=[skill("solidworks-essentials", False), skill("solidworks-documents", True), skill("admin", True)],
    )
    db = SimpleNamespace(toolconfig=SimpleNamespace(find_many=AsyncMock(return_value=[cfg])))
    with patch("ragtime.core.app_settings.get_db", AsyncMock(return_value=db)), patch("ragtime.skills.builtin._read_topic", return_value="body"):
        result = await SettingsCache().get_enabled_tool_configs()
    names = [item["name"] for item in result[0]["skills"]]
    assert "solidworks-essentials" not in names
    assert names.count("solidworks-documents") == 1
    assert names[-2:] == ["admin", "solidworks-documents"]
    assert all(item["id"].startswith("builtin:") for item in result[0]["skills"][:-2])


def test_admin_route(config):
    asyncio.run(_admin_route(config))


async def _admin_route(config):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_admin] = lambda: None
    tool = SimpleNamespace(model_dump=lambda: config)
    with (
        patch("ragtime.skills.routes.tool_repository.get_tool_config", AsyncMock(side_effect=[tool, None])),
        patch("ragtime.skills.routes.repository.list_instruction_skills", AsyncMock(return_value=[])),
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/indexes/tools/t/skills/builtin")
            missing = await client.get("/indexes/tools/missing/skills/builtin")
    assert response.status_code == 200
    assert len(response.json()) == 10
    assert missing.status_code == 404
