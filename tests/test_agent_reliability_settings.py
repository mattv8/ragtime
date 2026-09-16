import unittest
from types import SimpleNamespace
from typing import cast
from unittest import mock

from fastapi import HTTPException
from prisma import models as prisma_models

from ragtime.indexer import routes as indexer_routes
from ragtime.indexer.models import AppSettings, UpdateSettingsRequest


class AgentReliabilitySettingsTests(unittest.TestCase):
    def test_credit_settings_defaults_and_builder_model_are_valid(self) -> None:
        from ragtime.indexer.models import AppSettings

        settings = AppSettings()

        self.assertIsNone(settings.userspace_build_model)
        self.assertFalse(settings.openrouter_credit_monitor_enabled)
        self.assertEqual(settings.openrouter_low_credit_threshold_usd, 5.0)

    def test_credit_threshold_rejects_negative_values(self) -> None:
        from pydantic import ValidationError

        from ragtime.indexer.models import UpdateSettingsRequest

        with self.assertRaises(ValidationError):
            UpdateSettingsRequest(openrouter_low_credit_threshold_usd=-0.01)

    def test_chat_task_outcome_contract_is_optional_for_legacy_tasks(self) -> None:
        from ragtime.indexer.models import ChatTask

        task = ChatTask(id="task-1", conversation_id="conversation-1", user_message="hello")

        self.assertIsNone(task.execution_policy)
        self.assertIsNone(task.termination_reason)
        self.assertIsNone(task.outcome_summary)

    def test_management_key_is_encrypted_and_never_returned_after_storage(self) -> None:
        from ragtime.core.encryption import ENCRYPTED_PREFIX, decrypt_secret
        from ragtime.indexer.repository import IndexerRepository
        from tests.test_db_fixtures import FakeDb, make_settings_row

        repository = IndexerRepository()
        fake_db = FakeDb(make_settings_row())

        async def exercise() -> None:
            with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
                stored = await repository.update_settings({"openrouter_management_api_key": "management-secret"})
                self.assertIsNone(stored.openrouter_management_api_key)
                self.assertTrue(stored.has_openrouter_management_api_key)

                update_data = fake_db.appsettings.last_update_data
                assert update_data is not None
                encrypted = update_data["openrouterManagementApiKey"]
                self.assertTrue(encrypted.startswith(ENCRYPTED_PREFIX))
                self.assertNotEqual(encrypted, "management-secret")
                self.assertEqual(decrypt_secret(encrypted), "management-secret")

                cleared = await repository.update_settings({"openrouter_management_api_key": ""})
                cleared_key = fake_db.appsettings.last_update_data
                assert cleared_key is not None
                self.assertIsNone(cleared_key["openrouterManagementApiKey"])
                self.assertIsNone(cleared.openrouter_management_api_key)
                self.assertFalse(cleared.has_openrouter_management_api_key)

        import asyncio

        asyncio.run(exercise())


class BuilderModelSettingsRouteTests(unittest.IsolatedAsyncioTestCase):
    def _available_models(self) -> indexer_routes.AvailableModelsResponse:
        return indexer_routes.AvailableModelsResponse(
            models=[indexer_routes.AvailableModel(id="claude-builder", name="Claude Builder", provider="anthropic")],
            provider_states=[indexer_routes.ProviderModelState(provider="anthropic", configured=True, connected=True, available=True)],
        )

    async def test_builder_model_update_canonicalizes_an_available_model_before_persistence(self) -> None:
        current = AppSettings()
        updated = AppSettings(userspace_build_model="anthropic::claude-builder")

        with (
            mock.patch.object(indexer_routes, "get_available_chat_models", mock.AsyncMock(return_value=self._available_models())),
            mock.patch.object(indexer_routes.repository, "get_settings", mock.AsyncMock(return_value=current)),
            mock.patch.object(indexer_routes.repository, "update_settings", mock.AsyncMock(return_value=updated)) as update_settings,
            mock.patch.object(indexer_routes.rag, "initialize", mock.AsyncMock()),
            mock.patch.object(indexer_routes, "invalidate_settings_cache"),
            mock.patch.object(indexer_routes, "notify_tools_changed"),
        ):
            response = await indexer_routes.update_settings(
                UpdateSettingsRequest(userspace_build_model="claude-builder"),
                cast(prisma_models.User, SimpleNamespace(id="user-1", role="user")),
            )

        update_settings.assert_awaited_once()
        await_args = update_settings.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertEqual(await_args.args[0]["userspace_build_model"], "anthropic::claude-builder")
        self.assertEqual(response.settings.userspace_build_model, "anthropic::claude-builder")

    async def test_builder_model_update_rejects_an_unavailable_model_without_writing_settings(self) -> None:
        unavailable = indexer_routes.AvailableModelsResponse(
            models=[indexer_routes.AvailableModel(id="claude-available", name="Claude Available", provider="anthropic")],
            provider_states=[indexer_routes.ProviderModelState(provider="anthropic", configured=True, connected=True, available=True)],
        )

        with (
            mock.patch.object(indexer_routes, "get_available_chat_models", mock.AsyncMock(return_value=unavailable)),
            mock.patch.object(indexer_routes.repository, "update_settings", mock.AsyncMock()) as update_settings,
        ):
            with self.assertRaises(HTTPException) as raised:
                await indexer_routes.update_settings(
                    UpdateSettingsRequest(userspace_build_model="anthropic::missing"),
                    cast(prisma_models.User, SimpleNamespace(id="user-1", role="user")),
                )

        self.assertEqual(raised.exception.status_code, 400)
        update_settings.assert_not_awaited()
