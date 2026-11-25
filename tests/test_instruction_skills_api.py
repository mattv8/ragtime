import os
import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import httpx
from fastapi import FastAPI
from fastapi.routing import APIRoute
from prisma import Json, Prisma
from prisma.enums import ToolType as PrismaToolType
from pydantic import ValidationError

from ragtime.core.app_settings import SettingsCache
from ragtime.core.security import require_admin
from ragtime.skills.models import CreateInstructionSkill
from ragtime.skills.routes import router


class InstructionSkillValidationTests(unittest.TestCase):
    def test_rejects_invalid_names_and_blank_descriptions(self) -> None:
        for name, description in [("Bad Name", "description"), ("valid", "   ")]:
            with self.subTest(name=name), self.assertRaises(ValidationError):
                CreateInstructionSkill(name=name, description=description, content="", mode="on_demand")

    def test_budget_counts_only_enabled_always_skills(self) -> None:
        from ragtime.skills.repository import validate_instruction_skills

        instruction_skills = [
            CreateInstructionSkill(name="one", description="d", content="x" * 32000, mode="always"),
            CreateInstructionSkill(name="two", description="d", content="x", mode="always", enabled=False),
        ]
        validate_instruction_skills(instruction_skills, "tool")
        instruction_skills[1].enabled = True
        with self.assertRaisesRegex(ValueError, "this tool total 32,001 characters; the limit is 32,000"):
            validate_instruction_skills(instruction_skills, "tool")

    def test_maximum_skill_count(self) -> None:
        from ragtime.skills.repository import validate_instruction_skills

        instruction_skills = [CreateInstructionSkill(name=f"item-{index}", description="d", content="", mode="on_demand") for index in range(51)]
        with self.assertRaisesRegex(ValueError, "at most 50"):
            validate_instruction_skills(instruction_skills, "route")


class InstructionSkillRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.dependency_overrides[require_admin] = lambda: None
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_every_path_resolves_before_parameter_routes(self) -> None:
        from starlette.routing import Match

        cases = [
            ("GET", "/indexes/skills/summary", "get_instruction_skill_summary"),
            ("GET", "/indexes/tools/t/skills", "list_tool_instruction_skills"),
            ("POST", "/indexes/tools/t/skills", "create_tool_instruction_skill"),
            ("PUT", "/indexes/tools/t/skills/order", "reorder_tool_instruction_skills"),
            ("POST", "/indexes/tools/t/skills/copy", "copy_tool_instruction_skills"),
            ("PUT", "/indexes/tools/t/skills/s", "update_tool_instruction_skill"),
            ("DELETE", "/indexes/tools/t/skills/s", "delete_tool_instruction_skill"),
            ("GET", "/indexes/mcp-routes/r/skills", "list_route_instruction_skills"),
            ("POST", "/indexes/mcp-routes/r/skills", "create_route_instruction_skill"),
            ("PUT", "/indexes/mcp-routes/r/skills/order", "reorder_route_instruction_skills"),
            ("PUT", "/indexes/mcp-routes/r/skills/s", "update_route_instruction_skill"),
            ("DELETE", "/indexes/mcp-routes/r/skills/s", "delete_route_instruction_skill"),
        ]
        for method, path, name in cases:
            scope = {"type": "http", "path": path, "method": method, "root_path": ""}
            match = next(route for route in router.routes if route.matches(scope)[0] == Match.FULL)
            assert isinstance(match, APIRoute)
            self.assertEqual(match.name, name)

    async def test_create_validates_owner_names_budget_and_refresh(self) -> None:
        from ragtime.skills.models import InstructionSkill

        now = datetime.now(timezone.utc)
        result = InstructionSkill(
            id="s",
            owner_kind="tool",
            owner_id="t",
            name="guide",
            description="d",
            content="x",
            mode="always",
            enabled=True,
            sort_order=0,
            uri="skill://tools/t/guide/SKILL.md",
            created_at=now,
            updated_at=now,
        )
        payload = {"name": "guide", "description": "d", "content": "x", "mode": "always"}
        with (
            mock.patch("ragtime.skills.routes.repository.owner_exists", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.skills.routes.repository.list_instruction_skills", mock.AsyncMock(return_value=[])),
            mock.patch("ragtime.skills.routes.repository.create_instruction_skill", mock.AsyncMock(return_value=result)),
            mock.patch("ragtime.skills.routes.invalidate_settings_cache") as invalidate,
            mock.patch("ragtime.skills.routes.notify_tools_changed") as notify,
            mock.patch("ragtime.skills.routes.rag.initialize", mock.AsyncMock()) as initialize,
        ):
            response = await self.client.post("/indexes/tools/t/skills", json=payload)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["uri"], "skill://tools/t/guide/SKILL.md")
        invalidate.assert_called_once()
        notify.assert_called_once()
        initialize.assert_awaited_once()

        with (
            mock.patch("ragtime.skills.routes.repository.owner_exists", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.skills.routes.repository.list_instruction_skills", mock.AsyncMock(return_value=[result])),
        ):
            self.assertEqual((await self.client.post("/indexes/tools/t/skills", json=payload)).status_code, 409)
            self.assertEqual((await self.client.put("/indexes/tools/t/skills/order", json={"skill_ids": []})).status_code, 400)
            self.assertEqual((await self.client.put("/indexes/tools/t/skills/nope", json={"name": "new"})).status_code, 404)
            blank_description = await self.client.put("/indexes/tools/t/skills/s", json={"description": " "})
            self.assertEqual(blank_description.status_code, 422)
            self.assertEqual(blank_description.json()["detail"], "description: Value error, Description must not be blank")

        with (
            mock.patch("ragtime.skills.routes.repository.owner_exists", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.skills.routes.repository.list_instruction_skills", mock.AsyncMock(return_value=[result.model_copy(update={"sort_order": 3})])),
            mock.patch("ragtime.skills.routes.repository.create_instruction_skill", mock.AsyncMock(return_value=result)) as create,
            mock.patch("ragtime.skills.routes.invalidate_settings_cache"),
            mock.patch("ragtime.skills.routes.notify_tools_changed"),
            mock.patch("ragtime.skills.routes.rag.initialize", mock.AsyncMock()),
        ):
            await self.client.post("/indexes/tools/t/skills", json={**payload, "name": "appended"})
        call = create.await_args
        assert call is not None
        self.assertEqual(call.args[2].sort_order, 4)

        with mock.patch("ragtime.skills.routes.repository.owner_exists", mock.AsyncMock(return_value=False)):
            self.assertEqual((await self.client.get("/indexes/tools/missing/skills")).status_code, 404)

    async def test_create_rejects_over_budget_and_invalid_fields(self) -> None:
        payload = {"name": "guide", "description": "d", "content": "x" * 32001, "mode": "always"}
        with (
            mock.patch("ragtime.skills.routes.repository.owner_exists", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.skills.routes.repository.list_instruction_skills", mock.AsyncMock(return_value=[])),
            mock.patch("ragtime.skills.routes.repository.create_instruction_skill", mock.AsyncMock()) as create,
        ):
            response = await self.client.post("/indexes/tools/t/skills", json=payload)
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["detail"], "Always-include skills for this tool total 32,001 characters; the limit is 32,000.")
            invalid = await self.client.post("/indexes/tools/t/skills", json={**payload, "name": "Bad Name"})
            self.assertEqual(invalid.status_code, 422)
            create.assert_not_awaited()

    async def test_route_write_notifies_without_tool_refresh(self) -> None:
        with (
            mock.patch("ragtime.skills.routes.repository.owner_exists", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.skills.routes.repository.list_instruction_skills", mock.AsyncMock(return_value=[])),
            mock.patch("ragtime.skills.routes.repository.reorder_instruction_skills", mock.AsyncMock()),
            mock.patch("ragtime.skills.routes.invalidate_settings_cache") as invalidate,
            mock.patch("ragtime.skills.routes.notify_tools_changed") as notify,
            mock.patch("ragtime.skills.routes.rag.initialize", mock.AsyncMock()) as initialize,
        ):
            response = await self.client.put("/indexes/mcp-routes/r/skills/order", json={"skill_ids": []})
        self.assertEqual(response.status_code, 200)
        notify.assert_called_once()
        invalidate.assert_not_called()
        initialize.assert_not_awaited()

    async def test_copy_is_atomic_and_rejects_name_clashes(self) -> None:
        from ragtime.skills.models import InstructionSkill

        now = datetime.now(timezone.utc)
        source = InstructionSkill(
            id="s",
            owner_kind="tool",
            owner_id="source",
            name="guide",
            description="d",
            content="x",
            mode="on_demand",
            enabled=True,
            sort_order=2,
            uri="skill://tools/source/guide/SKILL.md",
            created_at=now,
            updated_at=now,
        )
        with (
            mock.patch("ragtime.skills.routes.repository.owner_exists", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.skills.routes.repository.list_instruction_skills", mock.AsyncMock(side_effect=[[], [source], []])),
            mock.patch("ragtime.skills.routes.repository.create_instruction_skills", mock.AsyncMock()) as create,
            mock.patch("ragtime.skills.routes.invalidate_settings_cache"),
            mock.patch("ragtime.skills.routes.notify_tools_changed"),
            mock.patch("ragtime.skills.routes.rag.initialize", mock.AsyncMock()),
        ):
            response = await self.client.post("/indexes/tools/target/skills/copy", json={"source_tool_id": "source"})
        self.assertEqual(response.status_code, 200)
        call = create.await_args
        assert call is not None
        self.assertEqual(call.args[0:2], ("tool", "target"))
        self.assertEqual(call.args[2][0].sort_order, 0)
        create.assert_awaited_once()

        existing = source.model_copy(update={"name": "existing", "sort_order": 4})
        with (
            mock.patch("ragtime.skills.routes.repository.owner_exists", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.skills.routes.repository.list_instruction_skills", mock.AsyncMock(side_effect=[[existing], [source], []])),
            mock.patch("ragtime.skills.routes.repository.create_instruction_skills", mock.AsyncMock()) as create,
            mock.patch("ragtime.skills.routes.invalidate_settings_cache"),
            mock.patch("ragtime.skills.routes.notify_tools_changed"),
            mock.patch("ragtime.skills.routes.rag.initialize", mock.AsyncMock()),
        ):
            await self.client.post("/indexes/tools/target/skills/copy", json={"source_tool_id": "source"})
        call = create.await_args
        assert call is not None
        self.assertEqual(call.args[2][0].sort_order, 5)

        with (
            mock.patch("ragtime.skills.routes.repository.owner_exists", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.skills.routes.repository.list_instruction_skills", mock.AsyncMock(side_effect=[[source], [source]])),
            mock.patch("ragtime.skills.routes.repository.create_instruction_skills", mock.AsyncMock()) as create,
        ):
            response = await self.client.post("/indexes/tools/target/skills/copy", json={"source_tool_id": "source"})
        self.assertEqual(response.status_code, 409)
        create.assert_not_awaited()

        source_large = source.model_copy(update={"name": "source-guide", "mode": "always", "content": "x" * 20000})
        target_large = source.model_copy(update={"name": "target-guide", "mode": "always", "content": "x" * 20000})
        with (
            mock.patch("ragtime.skills.routes.repository.owner_exists", mock.AsyncMock(return_value=True)),
            mock.patch("ragtime.skills.routes.repository.list_instruction_skills", mock.AsyncMock(side_effect=[[target_large], [source_large]])),
            mock.patch("ragtime.skills.routes.repository.create_instruction_skills", mock.AsyncMock()) as create,
        ):
            response = await self.client.post("/indexes/tools/target/skills/copy", json={"source_tool_id": "source"})
        self.assertEqual(response.status_code, 422)
        create.assert_not_awaited()

    async def test_summary_and_admin_requirement(self) -> None:
        summary = {"tools": {"t": {"total": 2, "enabled": 1}}, "routes": {}}
        with mock.patch("ragtime.skills.routes.repository.instruction_skill_summary", mock.AsyncMock(return_value=summary)):
            response = await self.client.get("/indexes/skills/summary")
        self.assertEqual(response.json(), summary)
        self.app.dependency_overrides.clear()
        response = await self.client.get("/indexes/skills/summary")
        self.assertIn(response.status_code, (401, 403))

    async def test_summary_is_not_shadowed_in_main_app(self) -> None:
        from ragtime.main import app

        summary: dict[str, dict[str, dict[str, int]]] = {"tools": {}, "routes": {}}
        app.dependency_overrides[require_admin] = lambda: None
        try:
            with mock.patch("ragtime.skills.routes.repository.instruction_skill_summary", mock.AsyncMock(return_value=summary)):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.get("/indexes/skills/summary")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), summary)
        finally:
            app.dependency_overrides.pop(require_admin, None)


class InstructionSkillRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_owner_and_runtime_records(self) -> None:
        from ragtime.skills.contract import is_default_route_owner
        from ragtime.skills.repository import _owner_filter, enabled_skill_records, owner_exists, to_skill_record

        self.assertTrue(is_default_route_owner("route", "default"))
        self.assertFalse(is_default_route_owner("tool", "default"))
        self.assertEqual(_owner_filter("route", "default"), {"defaultRoute": True})
        self.assertEqual(_owner_filter("route", "other"), {"mcpRouteId": "other"})
        self.assertEqual(_owner_filter("tool", "default"), {"toolConfigId": "default"})
        with mock.patch("ragtime.skills.repository.get_db", mock.AsyncMock()) as get_db:
            self.assertTrue(await owner_exists("route", "default"))
            get_db.assert_not_awaited()

        def item(name: str, order: int, enabled: bool) -> SimpleNamespace:
            return SimpleNamespace(id=name, name=name, description="d", content="text", mode="always", sortOrder=order, enabled=enabled)

        items = [item("z", 0, True), item("a", 0, True), item("off", -1, False)]
        self.assertEqual(
            to_skill_record(cast(Any, items[0])), {"id": "z", "name": "z", "description": "d", "content": "text", "mode": "always", "sort_order": 0}
        )
        self.assertEqual([record["name"] for record in enabled_skill_records(cast(Any, items))], ["a", "z"])

    async def test_summary_uses_aggregated_owner_counts(self) -> None:
        from ragtime.skills.repository import instruction_skill_summary

        rows = [
            {"tool_config_id": "tool", "mcp_route_id": None, "default_route": False, "total": 2, "enabled": 1},
            {"tool_config_id": None, "mcp_route_id": "route", "default_route": False, "total": 3, "enabled": 0},
            {"tool_config_id": None, "mcp_route_id": None, "default_route": True, "total": 1, "enabled": 1},
        ]
        db = SimpleNamespace(query_raw=mock.AsyncMock(return_value=rows))
        with mock.patch("ragtime.skills.repository.get_db", mock.AsyncMock(return_value=db)):
            result = await instruction_skill_summary()
        self.assertEqual(
            result, {"tools": {"tool": {"total": 2, "enabled": 1}}, "routes": {"route": {"total": 3, "enabled": 0}, "default": {"total": 1, "enabled": 1}}}
        )
        query = db.query_raw.await_args.args[0]
        self.assertNotIn("content", query.lower())
        self.assertIn("group by", query.lower())

    async def test_only_enabled_instruction_skills_are_sorted_and_attached(self) -> None:
        def item(name: str, order: int, enabled: bool) -> SimpleNamespace:
            return SimpleNamespace(id=name, name=name, description="d", content="text", mode="always", sortOrder=order, enabled=enabled)

        config = SimpleNamespace(
            id="t",
            name="tool",
            toolType="postgres",
            description="",
            connectionConfig={},
            maxResults=10,
            timeoutMaxSeconds=300,
            allowWrite=False,
            instructionSkills=[item("z", 0, True), item("a", 0, True), item("off", 0, False)],
        )
        db = SimpleNamespace(toolconfig=SimpleNamespace(find_many=mock.AsyncMock(return_value=[config])))
        with mock.patch("ragtime.core.app_settings.get_db", mock.AsyncMock(return_value=db)):
            result = await SettingsCache().get_enabled_tool_configs()
        self.assertEqual([item["name"] for item in result[0]["skills"]], ["a", "z"])
        self.assertEqual(set(result[0]["skills"][0]), {"id", "name", "description", "content", "mode", "sort_order"})


@unittest.skipUnless(os.environ.get("INSTRUCTION_SKILLS_TEST_DATABASE_URL"), "Dedicated instruction skills test database not configured")
class InstructionSkillDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_route_skills_with_database(self) -> None:
        from ragtime.skills.repository import list_enabled_skill_records

        database = Prisma(datasource={"url": os.environ["INSTRUCTION_SKILLS_TEST_DATABASE_URL"]})
        await database.connect()
        route = await database.mcprouteconfig.create(data={"name": "instruction-default-test", "routePath": str(uuid.uuid4())})
        tool = await database.toolconfig.create(data={"name": "instruction-default-test", "toolType": PrismaToolType.postgres, "connectionConfig": Json({})})
        prefix = f"test-{uuid.uuid4()}"
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[require_admin] = lambda: None
        path = "/indexes/mcp-routes/default/skills"
        data = {"name": prefix, "description": "A guide", "content": "test", "mode": "always"}
        try:
            with (
                mock.patch("ragtime.skills.repository.get_db", mock.AsyncMock(return_value=database)),
                mock.patch("ragtime.skills.routes.notify_tools_changed"),
            ):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    created = await client.post(path, json=data)
                    self.assertEqual(created.status_code, 201)
                    skill_id = created.json()["id"]
                    self.assertEqual((await client.post(path, json=data)).status_code, 409)
                    second = await client.post(path, json={**data, "name": f"{prefix}-second"})
                    disabled = await client.post(path, json={**data, "name": f"{prefix}-disabled", "enabled": False})
                    self.assertEqual(second.status_code, 201)
                    self.assertEqual(disabled.status_code, 201)
                    self.assertEqual((await client.put(f"{path}/{skill_id}", json={"content": "updated"})).json()["content"], "updated")
                    current = (await client.get(path)).json()
                    ids = [second.json()["id"], skill_id, disabled.json()["id"]]
                    ids.extend(item["id"] for item in current if item["id"] not in ids)
                    self.assertEqual((await client.put(f"{path}/order", json={"skill_ids": ids})).status_code, 200)
                    records = await list_enabled_skill_records("route", "default")
                    self.assertEqual([item["name"] for item in records if item["name"].startswith(prefix)], [f"{prefix}-second", prefix])
                    summary = (await client.get("/indexes/skills/summary")).json()
                    self.assertGreaterEqual(summary["routes"]["default"]["total"], 3)
                    custom = await client.post(f"/indexes/mcp-routes/{route.id}/skills", json=data)
                    self.assertEqual(custom.status_code, 201)
                    await database.mcprouteconfig.delete(where={"id": route.id})
                    self.assertEqual((await client.get(path)).status_code, 200)
                    self.assertTrue(any(item["id"] == skill_id for item in (await client.get(path)).json()))
                    with self.assertRaises(Exception):
                        await database.instructionskill.create(
                            data={"name": f"{prefix}-invalid", "description": "d", "content": "", "defaultRoute": True, "toolConfigId": tool.id}
                        )
                    self.assertEqual((await client.delete(f"{path}/{skill_id}")).status_code, 200)
        finally:
            await database.instructionskill.delete_many(where=cast(Any, {"defaultRoute": True, "name": {"startsWith": prefix}}))
            await database.mcprouteconfig.delete_many(where={"id": route.id})
            await database.toolconfig.delete(where={"id": tool.id})
            await database.disconnect()

    async def test_crud_copy_and_summary_with_database(self) -> None:
        database = Prisma(datasource={"url": os.environ["INSTRUCTION_SKILLS_TEST_DATABASE_URL"]})
        await database.connect()
        identifier = str(uuid.uuid4())
        source = await database.toolconfig.create(data={"name": "instruction-test-source", "toolType": PrismaToolType.postgres, "connectionConfig": Json({})})
        target = await database.toolconfig.create(data={"name": "instruction-test-target", "toolType": PrismaToolType.postgres, "connectionConfig": Json({})})
        route = await database.mcprouteconfig.create(data={"name": "instruction-test-route", "routePath": identifier})
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[require_admin] = lambda: None
        try:
            with (
                mock.patch("ragtime.skills.repository.get_db", mock.AsyncMock(return_value=database)),
                mock.patch("ragtime.skills.routes.invalidate_settings_cache"),
                mock.patch("ragtime.skills.routes.notify_tools_changed"),
                mock.patch("ragtime.skills.routes.rag.initialize", mock.AsyncMock()),
            ):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    path = f"/indexes/tools/{source.id}/skills"
                    data = {"name": "guide", "description": "A guide", "content": "test", "mode": "always"}
                    created = await client.post(path, json=data)
                    self.assertEqual(created.status_code, 201)
                    skill_id = created.json()["id"]
                    self.assertEqual((await client.post(path, json=data)).status_code, 409)
                    updated = await client.put(f"{path}/{skill_id}", json={"enabled": False})
                    self.assertFalse(updated.json()["enabled"])
                    self.assertEqual((await client.put(f"{path}/order", json={"skill_ids": [skill_id]})).status_code, 200)
                    second = await client.post(path, json={**data, "name": "added-later", "mode": "on_demand"})
                    self.assertEqual([item["name"] for item in (await client.get(path)).json()], ["guide", "added-later"])
                    copied = await client.post(f"/indexes/tools/{target.id}/skills/copy", json={"source_tool_id": source.id})
                    self.assertEqual([item["name"] for item in copied.json()], ["guide", "added-later"])
                    self.assertEqual((await client.delete(f"{path}/{second.json()['id']}")).status_code, 200)
                    route_created = await client.post(f"/indexes/mcp-routes/{route.id}/skills", json=data)
                    self.assertEqual(route_created.status_code, 201)
                    route_disabled = await client.post(f"/indexes/mcp-routes/{route.id}/skills", json={**data, "name": "disabled", "enabled": False})
                    self.assertEqual(route_disabled.status_code, 201)
                    summary = (await client.get("/indexes/skills/summary")).json()
                    self.assertEqual(summary["tools"][source.id], {"total": 1, "enabled": 0})
                    self.assertEqual(summary["routes"][route.id], {"total": 2, "enabled": 1})
                    await client.put(f"{path}/{skill_id}", json={"enabled": True})
                    with mock.patch("ragtime.core.app_settings.get_db", mock.AsyncMock(return_value=database)):
                        runtime_configs = await SettingsCache().get_enabled_tool_configs()
                    runtime_source = next(config for config in runtime_configs if config["id"] == source.id)
                    self.assertEqual([item["name"] for item in runtime_source["skills"]], ["guide"])
                    self.assertEqual((await client.delete(f"{path}/{skill_id}")).status_code, 200)
                    self.assertEqual((await client.get(path)).json(), [])
        finally:
            await database.toolconfig.delete(where={"id": source.id})
            await database.toolconfig.delete(where={"id": target.id})
            await database.mcprouteconfig.delete(where={"id": route.id})
            await database.disconnect()
