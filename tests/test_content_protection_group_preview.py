import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi import FastAPI, HTTPException

from ragtime.content_protection import routes, service
from ragtime.content_protection.models import ContentProtectionConfig
from ragtime.rag.prompts import build_access_level_prompt_fragment


def preview_config() -> ContentProtectionConfig:
    payload = ContentProtectionConfig().model_dump(mode="json")
    payload["share_with_assistant"] = True
    payload["access_levels"][0]["guidance"] = "Use operational guidance."
    payload["access_levels"][1]["guidance"] = "Use finance guidance."
    return ContentProtectionConfig.model_validate(payload)


class GroupPreviewRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(routes.router)
        self.app.dependency_overrides[routes.require_admin] = lambda: SimpleNamespace(id="admin-1", role="admin")
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.config = preview_config().model_dump(mode="json")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        self.app.dependency_overrides.clear()

    async def test_preview_rejects_invalid_synthetic_selectors(self) -> None:
        invalid_requests = (
            ({"access_level_ids": ["standard", "standard"]}, "duplicate_access_level_ids"),
            ({"access_level_ids": ["missing"]}, "unknown_access_level_ids"),
            ({"access_level_ids": ["x" * 129]}, "invalid_access_level_ids"),
            ({"access_level_ids": [f"level-{index}" for index in range(25)]}, "invalid_access_level_ids"),
            ({"access_level_ids": ["standard"], "user_id": "user-1"}, "ambiguous_synthetic_access_levels"),
            ({"access_level_ids": ["standard"], "public": True}, "ambiguous_synthetic_access_levels"),
            ({"access_level_ids": ["standard"], "baseline": "service"}, "ambiguous_synthetic_access_levels"),
        )

        for request, code in invalid_requests:
            response = await self.client.post("/indexes/content-protection/preview", json={"config": self.config, **request})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["detail"]["code"], code)

    async def test_preview_synthetic_selection_is_admin_only(self) -> None:
        self.app.dependency_overrides[routes.require_admin] = lambda: (_ for _ in ()).throw(HTTPException(status_code=403, detail="Admin access required"))

        response = await self.client.post("/indexes/content-protection/preview", json={"config": self.config, "access_level_ids": []})

        self.assertEqual(response.status_code, 403)

    async def test_preview_empty_selector_returns_default_synthetic_snapshot(self) -> None:
        with mock.patch.object(service, "_classify", new=mock.AsyncMock()) as classify:
            response = await self.client.post("/indexes/content-protection/preview", json={"config": self.config, "access_level_ids": []})

        classify.assert_not_awaited()
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["required"])
        self.assertEqual(response.json()["provenance"], "synthetic_access_levels")
        self.assertIn("Use operational guidance.", response.json()["prompt_fragment"])

    async def test_explicit_null_selector_keeps_verified_user_resolution(self) -> None:
        identities = ({"user-1"}, {"user-1": set()}, {"user-1": None})
        with mock.patch.object(service, "resolve_identities", new=mock.AsyncMock(return_value=identities)) as resolve:
            response = await self.client.post(
                "/indexes/content-protection/preview",
                json={"config": self.config, "user_id": "user-1", "access_level_ids": None},
            )
        self.assertEqual(response.status_code, 200)
        resolve.assert_awaited_once_with({"user-1"})
        self.assertIsInstance(response.json()["required"], bool)
        self.assertNotEqual(response.json()["provenance"], "synthetic_access_levels")
        self.assertIn("Use operational guidance.", response.json()["prompt_fragment"])


class GroupPreviewServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_selector_uses_default_without_identity_lookup(self) -> None:
        config = preview_config()
        with mock.patch.object(service, "resolve_identities", new=mock.AsyncMock()) as resolve_identities:
            result = await service.preview_policy(service.ProtectionContext(), config, access_level_ids=[])

        resolve_identities.assert_not_awaited()
        self.assertIsNone(result["required"])
        self.assertEqual(result["provenance"], "synthetic_access_levels")
        self.assertEqual(result["access_levels"], [[config.access_levels[0].model_dump(mode="json")]])

    async def test_multiple_selector_uses_sorted_union_and_matches_real_snapshot_fragment(self) -> None:
        config = preview_config()
        with mock.patch.object(service, "resolve_identities", new=mock.AsyncMock()) as resolve_identities:
            synthetic = await service.preview_policy(service.ProtectionContext(), config, access_level_ids=["finance", "standard"])

        resolve_identities.assert_not_awaited()
        payload = config.model_dump(mode="json")
        payload["group_access_levels"] = [
            {"group_id": "finance-group", "access_level_id": "standard"},
            {"group_id": "finance-group", "access_level_id": "finance"},
        ]
        mapped_config = ContentProtectionConfig.model_validate(payload)
        identities = ({"user-1"}, {"user-1": {"finance-group"}}, {"user-1": None})
        with mock.patch.object(service, "resolve_identities", new=mock.AsyncMock(return_value=identities)):
            real = await service.preview_policy(service.ProtectionContext(user_id="user-1"), mapped_config)
        access_levels = synthetic["access_levels"]
        assert isinstance(access_levels, list) and isinstance(access_levels[0], list)
        selected_levels = access_levels[0]
        assert all(isinstance(level, dict) for level in selected_levels)
        self.assertEqual([level["id"] for level in selected_levels], ["finance", "standard"])
        self.assertEqual(synthetic["granted_category_ids"], ["company_finance", "operational"])
        self.assertEqual(build_access_level_prompt_fragment(synthetic), build_access_level_prompt_fragment(real))
        self.assertIn("Use finance guidance.", build_access_level_prompt_fragment(synthetic))
        self.assertEqual(synthetic["guidance_revision"], real["guidance_revision"])

    async def test_omitted_selector_retains_identity_resolution(self) -> None:
        config = preview_config()
        resolved = service._ResolvedPolicy(
            True, "group", {"user-1"}, {"user-1": {"group-1"}}, {"user-1": None}, [[config.access_levels[0].model_dump(mode="json")]], {"operational"}
        )
        with mock.patch.object(service, "_resolve", new=mock.AsyncMock(return_value=resolved)) as resolve:
            result = await service.preview_policy(service.ProtectionContext(user_id="user-1"), config)

        resolve.assert_awaited_once()
        self.assertTrue(result["required"])
        self.assertEqual(result["provenance"], "group")
