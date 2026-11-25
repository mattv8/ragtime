import asyncio
import re
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import httpx
from fastapi import FastAPI

import ragtime.mcp.routes as routes
import ragtime.mcp.server as server
from ragtime.mcp.guide import GENERAL_TOOLS_GUIDE, WORKSPACE_GUIDE, default_route_instructions
from ragtime.skills.contract import SkillRecord

MARKER = "unique-default-admin-skill-marker"
SKILL: SkillRecord = {"id": "1", "name": "rules", "description": "rules", "content": MARKER, "mode": "always", "sort_order": 0}


class DefaultRouteGuidanceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cache = mock.patch.object(server, "_default_route_skills", [SKILL])
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.protection = mock.patch.object(server, "classification_required", mock.AsyncMock(return_value=False))
        self.protection.start()
        self.addCleanup(self.protection.stop)

    def test_workspace_names_exist(self):
        names = set(re.findall(r"\bworkspace_development\w*", GENERAL_TOOLS_GUIDE + WORKSPACE_GUIDE))
        self.assertTrue(names)
        self.assertLessEqual(names, server._DEVELOPMENT_TOOL_NAMES)

    def test_composition_matrix(self):
        for general in (False, True):
            for workspace in (False, True):
                text = default_route_instructions(general_tools=general, workspace_tools=workspace, route_skills=[SKILL], inline_always=True)
                self.assertEqual("## Using Ragtime tools" in (text or ""), general)
                self.assertEqual("## Building in a Ragtime workspace" in (text or ""), workspace)
                self.assertEqual(MARKER in (text or ""), general)
                if not general and not workspace:
                    self.assertIsNone(text)

    async def test_initialization_is_request_scoped(self):
        instance = server._create_server("test")
        unbound = instance.create_initialization_options()
        self.assertIn("## Using Ragtime tools", unbound.instructions or "")
        self.assertIn("## Building in a Ragtime workspace", unbound.instructions or "")
        self.assertNotIn(MARKER, unbound.instructions or "")
        assert unbound.capabilities.tools is not None
        self.assertTrue(unbound.capabilities.tools.listChanged)
        for principal, general, workspace in (
            (None, True, False),
            (SimpleNamespace(general_tools_allowed=True), True, True),
            (SimpleNamespace(general_tools_allowed=False), False, True),
        ):
            with self.subTest(principal=principal), server.mcp_request_context(principal, "default"):
                async with server.default_route_skills_context():
                    options = instance.create_initialization_options()
                    text = options.instructions or ""
                    self.assertEqual("## Using Ragtime tools" in text, general)
                    self.assertEqual("## Building in a Ragtime workspace" in text, workspace)
                    self.assertEqual(MARKER in text, general)
                    assert options.capabilities.tools is not None
                    self.assertTrue(options.capabilities.tools.listChanged)

    async def test_inflight_load_does_not_restore_invalidated_cache(self):
        server._default_route_skills = None
        started, finish = asyncio.Event(), asyncio.Event()

        async def load(*args: Any) -> list[SkillRecord]:
            started.set()
            await finish.wait()
            return [SKILL]

        with mock.patch.object(server, "list_enabled_skill_records", mock.AsyncMock(side_effect=load)):
            task = asyncio.create_task(server.get_default_route_skills())
            await started.wait()
            server.notify_tools_changed()
            finish.set()
            self.assertEqual(await task, [SKILL])
            self.assertIsNone(server._default_route_skills)

    async def test_classification_is_skipped_without_records_or_general_access(self):
        check = cast(mock.AsyncMock, server.classification_required)
        with server.mcp_request_context(SimpleNamespace(general_tools_allowed=False)):
            async with server.default_route_skills_context():
                pass
        with mock.patch.object(server, "_default_route_skills", []), server.mcp_request_context(None):
            async with server.default_route_skills_context():
                pass
        check.assert_not_awaited()

    async def test_filtered_initialization_matches_lanes(self):
        row = SimpleNamespace(
            enabled=True,
            toolSelections=[],
            includeKnowledgeSearch=False,
            includeGitHistory=False,
            selectedDocumentIndexes=[],
            selectedFilesystemIndexes=[],
            selectedSchemaIndexes=[],
            name="test",
        )
        db = SimpleNamespace(mcpdefaultroutefilter=SimpleNamespace(find_unique=mock.AsyncMock(return_value=row)))
        with (
            mock.patch.object(server, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(server, "get_app_settings", mock.AsyncMock(return_value={})),
            mock.patch.object(server, "_default_filter_servers", {}),
            mock.patch.object(server, "_default_filter_adapters", {}),
        ):
            result = await server.get_default_route_filtered_server("filter")
            assert result is not None
            instance, _ = result
            for principal, general, workspace in (
                (None, True, False),
                (SimpleNamespace(general_tools_allowed=True), True, True),
                (SimpleNamespace(general_tools_allowed=False), False, True),
            ):
                with server.mcp_request_context(principal):
                    async with server.default_route_skills_context():
                        options = instance.create_initialization_options()
                        text = options.instructions or ""
                        self.assertEqual(MARKER in text, general)
                        self.assertEqual("## Building in a Ragtime workspace" in text, workspace)
                        assert options.capabilities.tools is not None
                        self.assertTrue(options.capabilities.tools.listChanged)

    async def test_cache_invalidation_and_failure(self):
        server._default_route_skills = None
        loader = mock.AsyncMock(side_effect=[RuntimeError("offline"), [SKILL], []])
        with mock.patch.object(server, "list_enabled_skill_records", loader):
            self.assertEqual(await server.get_default_route_skills(), [])
            self.assertIsNone(server._default_route_skills)
            self.assertEqual(await server.get_default_route_skills(), [SKILL])
            self.assertEqual(await server.get_default_route_skills(), [SKILL])
            server.notify_tools_changed()
            self.assertEqual(await server.get_default_route_skills(), [])
            self.assertEqual(loader.await_count, 3)

    async def test_http_initialize_respects_request_principal(self):
        app = FastAPI()
        app.router.routes.extend(cast(Any, routes.get_mcp_routes()))
        settings = {"mcp_enabled": True, "mcp_default_route_auth": False, "server_name": "guidance-test"}
        previous = routes._mcp_http_state["session_manager"]
        routes._mcp_http_state["session_manager"] = None
        resolve = mock.AsyncMock()
        try:
            with (
                mock.patch.object(routes, "get_app_settings", mock.AsyncMock(return_value=settings)),
                mock.patch.object(server, "get_app_settings", mock.AsyncMock(return_value=settings)),
                mock.patch.object(routes, "_resolve_default_development_principal", resolve),
                mock.patch.object(routes, "_log_mcp", mock.AsyncMock()),
            ):
                async with routes.mcp_lifespan_manager():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mcp.test") as client:
                        for principal, general in ((None, True), (SimpleNamespace(general_tools_allowed=False), False)):
                            resolve.return_value = principal
                            response = await client.post(
                                "/mcp",
                                headers={"Accept": "application/json, text/event-stream"},
                                json={
                                    "jsonrpc": "2.0",
                                    "id": 1,
                                    "method": "initialize",
                                    "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
                                },
                            )
                            self.assertEqual(response.status_code, 200, response.text)
                            result = response.json()["result"]
                            self.assertEqual(MARKER in result["instructions"], general)
                            self.assertEqual("## Using Ragtime tools" in result["instructions"], general)
                            self.assertEqual("## Building in a Ragtime workspace" in result["instructions"], not general)
                            self.assertTrue(result["capabilities"]["tools"]["listChanged"])
        finally:
            routes._mcp_http_state["session_manager"] = previous
