import os
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi import FastAPI, HTTPException
from prisma import Json

from ragtime.content_protection import routes
from ragtime.content_protection.conflicts import GroupPolicyConflictError
from ragtime.content_protection.models import ContentProtectionConfig
from ragtime.core.database import connect_db, disconnect_db


class ContentProtectionRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(routes.router)
        self.app.dependency_overrides[routes.require_admin] = lambda: SimpleNamespace(id="admin-1", role="admin")
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        self.app.dependency_overrides.clear()

    async def test_all_endpoints_require_admin(self) -> None:
        def reject() -> None:
            raise HTTPException(status_code=403, detail="Admin access required")

        self.app.dependency_overrides[routes.require_admin] = reject

        response = await self.client.get("/indexes/content-protection/config")

        self.assertEqual(response.status_code, 403)

        response = await self.client.get("/indexes/content-protection/policy-conflicts")

        self.assertEqual(response.status_code, 403)

    async def test_config_revision_conflict_returns_409(self) -> None:
        with mock.patch.object(
            routes,
            "save_config",
            new=mock.AsyncMock(side_effect=ValueError("revision_conflict")),
        ) as save_config:
            response = await self.client.put(
                "/indexes/content-protection/config",
                json={"expected_revision": 3, "config": {"revision": 3, "enabled": False}},
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json(), {"detail": "Content protection configuration changed"})
        await_args = save_config.await_args
        assert await_args is not None
        self.assertIsInstance(await_args.args[0], ContentProtectionConfig)

    async def test_config_group_policy_conflict_returns_structured_422(self) -> None:
        detail = {"code": "conflicting_group_policies", "message": "Conflict", "conflicts": [], "total": 0, "truncated": False}
        with mock.patch.object(routes, "save_config", new=mock.AsyncMock(side_effect=GroupPolicyConflictError(detail))):
            response = await self.client.put(
                "/indexes/content-protection/config",
                json={"expected_revision": 3, "config": {"revision": 3, "enabled": False}},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"detail": detail})

    async def test_policy_conflicts_returns_capped_contract(self) -> None:
        with (
            mock.patch.object(routes, "load_config", new=mock.AsyncMock(return_value=ContentProtectionConfig())),
            mock.patch.object(routes, "find_group_policy_conflicts", new=mock.AsyncMock(return_value=([{"user_id": "u"}], 51, ["require", "skip"]))),
            mock.patch.object(routes, "get_db", new=mock.AsyncMock(return_value=object())),
        ):
            response = await self.client.get("/indexes/content-protection/policy-conflicts")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"conflicts": [{"user_id": "u"}], "total": 51, "truncated": True, "conflicted_group_ids": ["require", "skip"]})

    async def test_policy_conflicts_passes_optional_user_filter(self) -> None:
        finder = mock.AsyncMock(return_value=([], 0, []))
        with (
            mock.patch.object(routes, "load_config", new=mock.AsyncMock(return_value=ContentProtectionConfig())),
            mock.patch.object(routes, "find_group_policy_conflicts", new=finder),
            mock.patch.object(routes, "get_db", new=mock.AsyncMock(return_value=object())),
        ):
            response = await self.client.get("/indexes/content-protection/policy-conflicts?user_id=user-51")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(finder.await_args.args[2], "user-51")

    async def test_policy_conflicts_invalid_config_returns_safe_409(self) -> None:
        with mock.patch.object(routes, "load_config", new=mock.AsyncMock(side_effect=ValueError("private parser detail"))):
            response = await self.client.get("/indexes/content-protection/policy-conflicts")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "invalid_content_protection_config")
        self.assertIn("must be repaired", response.json()["detail"]["message"])
        self.assertNotIn("private parser detail", response.text)

    async def test_config_uses_the_canonical_client_path(self) -> None:
        config = {"revision": 4, "enabled": False, "coverage_mode": "all_supported_traffic"}
        with mock.patch.object(routes, "load_config", new=mock.AsyncMock(return_value=config)):
            response = await self.client.get("/indexes/content-protection/config")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), config)

    async def test_sample_rejects_oversized_body_without_calling_service(self) -> None:
        with mock.patch.object(routes, "test_sample", new=mock.AsyncMock()) as test_sample:
            response = await self.client.post(
                "/indexes/content-protection/test",
                json={"config": {}, "sample": "x" * 1_048_577, "profile_ids": []},
            )

        self.assertEqual(response.status_code, 422)
        test_sample.assert_not_awaited()

    async def test_catalog_combines_static_and_runtime_tools_with_default_route(self) -> None:
        db = SimpleNamespace(
            user=SimpleNamespace(find_many=mock.AsyncMock(return_value=[SimpleNamespace(id="u-1", username="sam", displayName="Sam")])),
            authgroup=SimpleNamespace(find_many=mock.AsyncMock(return_value=[SimpleNamespace(id="g-1", displayName="Finance")])),
            toolconfig=SimpleNamespace(find_many=mock.AsyncMock(return_value=[SimpleNamespace(id="tool-1", name="Production DB")])),
            mcprouteconfig=SimpleNamespace(find_many=mock.AsyncMock(return_value=[SimpleNamespace(id="route-1", name="Reporting")])),
        )
        with (
            mock.patch.object(routes, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(routes, "get_all_tools", return_value={"search_knowledge": object()}),
            mock.patch.object(routes.app_settings, "get_app_settings", new=mock.AsyncMock(return_value={"typesafe_api_key": "set", "openrouter_api_key": ""})),
        ):
            response = await self.client.get("/indexes/content-protection/catalog")

        self.assertEqual(response.status_code, 200)
        catalog = response.json()
        self.assertEqual(catalog["users"], [{"id": "u-1", "name": "Sam"}])
        self.assertIn({"id": "search_knowledge", "name": "search_knowledge"}, catalog["tools"])
        self.assertIn({"id": "tool-1", "name": "Production DB"}, catalog["tools"])
        self.assertEqual(catalog["mcp_routes"][0], {"id": "default", "name": "Default MCP route"})
        self.assertIn({"id": "component", "name": "Component"}, catalog["surfaces"])
        self.assertEqual(catalog["classifier_status"], {"typesafe_key_configured": True, "openrouter_key_configured": False})


@unittest.skipUnless(
    os.environ.get("RAGTIME_CONTENT_PROTECTION_INTEGRATION") == "1",
    "requires isolated Prisma/Postgres opt-in",
)
class ContentProtectionConfigDatabaseRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db = await connect_db()
        self.original = await self.db.contentprotectionconfig.find_unique(where={"id": "default"})
        self.app = FastAPI()
        self.app.include_router(routes.router)
        self.app.dependency_overrides[routes.require_admin] = lambda: SimpleNamespace(id="admin-1", role="admin")
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self) -> None:
        try:
            await self.client.aclose()
            self.app.dependency_overrides.clear()
            if self.original is None:
                await self.db.contentprotectionconfig.delete_many(where={"id": "default"})
            else:
                await self.db.contentprotectionconfig.upsert(
                    where={"id": "default"},
                    data={
                        "create": {
                            "id": "default",
                            "revision": self.original.revision,
                            "config": Json(self.original.config),
                            "updatedBy": self.original.updatedBy,
                        },
                        "update": {
                            "revision": self.original.revision,
                            "config": Json(self.original.config),
                            "updatedBy": self.original.updatedBy,
                        },
                    },
                )
        finally:
            await disconnect_db()

    async def test_disabled_config_round_trips_through_the_real_database(self) -> None:
        current = await self.client.get("/indexes/content-protection/config")
        self.assertEqual(current.status_code, 200)
        config = current.json()
        config["enabled"] = False

        saved = await self.client.put(
            "/indexes/content-protection/config",
            json={"expected_revision": config["revision"], "config": config},
        )
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertFalse(saved.json()["enabled"])

        loaded = await self.client.get("/indexes/content-protection/config")
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(loaded.json()["revision"], saved.json()["revision"])
        self.assertFalse(loaded.json()["enabled"])
