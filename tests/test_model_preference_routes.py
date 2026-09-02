import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest import mock

from fastapi import HTTPException
from prisma.models import User

from ragtime.indexer import routes as indexer_routes
from ragtime.indexer.models import ModelPreferenceRequest


def _make_user(user_id: str = "user-1", role: str = "user") -> User:
    return cast(User, SimpleNamespace(id=user_id, role=role))


@dataclass(frozen=True)
class FakeModelAvailabilitySnapshot:
    available_model_ids: frozenset[str]
    authoritative_providers: frozenset[str]


class ModelPreferenceRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_workspace_preferences_returns_exact_and_inherited_values(self) -> None:
        user = _make_user()
        fake_module = SimpleNamespace(
            ModelAvailabilitySnapshot=FakeModelAvailabilitySnapshot,
            get_user_default_model=mock.AsyncMock(return_value="openai::gpt-5"),
            get_workspace_user_default_model=mock.AsyncMock(return_value="anthropic::claude-sonnet-4.5"),
            resolve_new_conversation_model=mock.AsyncMock(return_value="anthropic::claude-sonnet-4.5"),
        )

        with (
            mock.patch.object(indexer_routes, "_assert_workspace_access", mock.AsyncMock()) as access_mock,
            mock.patch.object(indexer_routes, "_get_model_preferences_module", return_value=fake_module),
            mock.patch.object(
                indexer_routes.repository,
                "get_settings",
                mock.AsyncMock(return_value=SimpleNamespace(default_chat_model="omlx::glm-4", llm_model="omlx::glm-4", allowed_chat_models=[])),
            ),
        ):
            response = await indexer_routes.get_chat_model_preferences("ws-1", user)

        access_mock.assert_awaited_once_with("ws-1", user, "viewer")
        fake_module.get_user_default_model.assert_awaited_once_with(user.id)
        fake_module.get_workspace_user_default_model.assert_awaited_once_with(user.id, "ws-1")
        fake_module.resolve_new_conversation_model.assert_awaited_once()
        self.assertEqual(response.user_default_chat_model, "openai::gpt-5")
        self.assertEqual(response.workspace_default_chat_model, "anthropic::claude-sonnet-4.5")
        self.assertEqual(response.global_default_chat_model, "omlx::glm-4")
        self.assertEqual(response.effective_default_chat_model, "anthropic::claude-sonnet-4.5")

    async def test_get_workspace_preferences_hides_inaccessible_workspace(self) -> None:
        user = _make_user()

        with mock.patch.object(
            indexer_routes,
            "_assert_workspace_access",
            mock.AsyncMock(side_effect=HTTPException(status_code=404, detail="Workspace not found")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await indexer_routes.get_chat_model_preferences("ws-404", user)

        self.assertEqual(raised.exception.status_code, 404)

    async def test_put_general_preference_canonicalizes_scoped_model_identifier(self) -> None:
        user = _make_user()
        request = ModelPreferenceRequest(default_chat_model="gpt-5")
        fake_module = SimpleNamespace(
            ModelAvailabilitySnapshot=FakeModelAvailabilitySnapshot,
            get_user_default_model=mock.AsyncMock(return_value="openai::gpt-5"),
            get_workspace_user_default_model=mock.AsyncMock(return_value=None),
            set_user_default_model=mock.AsyncMock(return_value="openai::gpt-5"),
            resolve_new_conversation_model=mock.AsyncMock(return_value="openai::gpt-5"),
        )
        available = indexer_routes.AvailableModelsResponse(
            models=[indexer_routes.AvailableModel(id="gpt-5", name="GPT-5", provider="openai")],
            provider_states=[indexer_routes.ProviderModelState(provider="openai", configured=True, connected=True, available=True)],
        )

        with (
            mock.patch.object(indexer_routes, "_get_model_preferences_module", return_value=fake_module),
            mock.patch.object(indexer_routes, "get_available_chat_models", mock.AsyncMock(return_value=available)),
            mock.patch.object(
                indexer_routes.repository,
                "get_settings",
                mock.AsyncMock(return_value=SimpleNamespace(default_chat_model="omlx::glm-4", llm_model="omlx::glm-4", allowed_chat_models=[])),
            ),
        ):
            response = await indexer_routes.put_chat_model_preferences(request, user)

        fake_module.set_user_default_model.assert_awaited_once_with(user.id, "openai::gpt-5")
        self.assertEqual(response.effective_default_chat_model, "openai::gpt-5")

    async def test_put_workspace_preference_allows_owner_member_and_admin(self) -> None:
        roles = [
            ("owner-1", "user"),
            ("member-1", "user"),
            ("admin-1", "admin"),
        ]
        available = indexer_routes.AvailableModelsResponse(
            models=[indexer_routes.AvailableModel(id="claude-sonnet-4.5", name="Claude", provider="anthropic")],
            provider_states=[indexer_routes.ProviderModelState(provider="anthropic", configured=True, connected=True, available=True)],
        )

        for user_id, role in roles:
            with self.subTest(user_id=user_id, role=role):
                user = _make_user(user_id, role)
                fake_module = SimpleNamespace(
                    ModelAvailabilitySnapshot=FakeModelAvailabilitySnapshot,
                    get_user_default_model=mock.AsyncMock(return_value=None),
                    get_workspace_user_default_model=mock.AsyncMock(return_value="anthropic::claude-sonnet-4.5"),
                    set_workspace_user_default_model=mock.AsyncMock(return_value="anthropic::claude-sonnet-4.5"),
                    resolve_new_conversation_model=mock.AsyncMock(return_value="anthropic::claude-sonnet-4.5"),
                )
                with (
                    mock.patch.object(indexer_routes, "_assert_workspace_access", mock.AsyncMock()) as access_mock,
                    mock.patch.object(indexer_routes, "_get_model_preferences_module", return_value=fake_module),
                    mock.patch.object(indexer_routes, "get_available_chat_models", mock.AsyncMock(return_value=available)),
                    mock.patch.object(
                        indexer_routes.repository,
                        "get_settings",
                        mock.AsyncMock(return_value=SimpleNamespace(default_chat_model="omlx::glm-4", llm_model="omlx::glm-4", allowed_chat_models=[])),
                    ),
                ):
                    response = await indexer_routes.put_chat_model_preferences(
                        ModelPreferenceRequest(workspace_id="ws-1", default_chat_model="anthropic::claude-sonnet-4.5"),
                        user,
                    )

                access_mock.assert_awaited_once_with("ws-1", user, "viewer")
                fake_module.set_workspace_user_default_model.assert_awaited_once_with(user.id, "ws-1", "anthropic::claude-sonnet-4.5")
                self.assertEqual(response.workspace_id, "ws-1")

    async def test_put_reset_is_idempotent(self) -> None:
        user = _make_user()
        fake_module = SimpleNamespace(
            ModelAvailabilitySnapshot=FakeModelAvailabilitySnapshot,
            get_user_default_model=mock.AsyncMock(return_value=None),
            get_workspace_user_default_model=mock.AsyncMock(return_value=None),
            set_user_default_model=mock.AsyncMock(return_value=None),
            resolve_new_conversation_model=mock.AsyncMock(return_value="omlx::glm-4"),
        )

        with (
            mock.patch.object(indexer_routes, "_get_model_preferences_module", return_value=fake_module),
            mock.patch.object(
                indexer_routes.repository,
                "get_settings",
                mock.AsyncMock(return_value=SimpleNamespace(default_chat_model="omlx::glm-4", llm_model="omlx::glm-4", allowed_chat_models=[])),
            ),
        ):
            response = await indexer_routes.put_chat_model_preferences(ModelPreferenceRequest(default_chat_model=None), user)

        fake_module.set_user_default_model.assert_awaited_once_with(user.id, None)
        self.assertIsNone(response.user_default_chat_model)
        self.assertEqual(response.effective_default_chat_model, "omlx::glm-4")

    async def test_put_rejects_unselectable_model_when_provider_state_is_authoritative(self) -> None:
        user = _make_user()
        fake_module = SimpleNamespace(ModelAvailabilitySnapshot=FakeModelAvailabilitySnapshot)
        available = indexer_routes.AvailableModelsResponse(
            models=[indexer_routes.AvailableModel(id="gpt-5", name="GPT-5", provider="openai")],
            provider_states=[indexer_routes.ProviderModelState(provider="openai", configured=True, connected=True, available=True)],
        )

        with (
            mock.patch.object(indexer_routes, "_get_model_preferences_module", return_value=fake_module),
            mock.patch.object(indexer_routes, "get_available_chat_models", mock.AsyncMock(return_value=available)),
        ):
            with self.assertRaises(HTTPException) as raised:
                await indexer_routes.put_chat_model_preferences(
                    ModelPreferenceRequest(default_chat_model="openai::missing"),
                    user,
                )

        self.assertEqual(raised.exception.status_code, 400)

    async def test_put_returns_503_when_discovery_loading_or_provider_not_authoritative(self) -> None:
        user = _make_user()
        fake_module = SimpleNamespace(ModelAvailabilitySnapshot=FakeModelAvailabilitySnapshot)

        loading = indexer_routes.AvailableModelsResponse(models_loading=True)
        with (
            mock.patch.object(indexer_routes, "_get_model_preferences_module", return_value=fake_module),
            mock.patch.object(indexer_routes, "get_available_chat_models", mock.AsyncMock(return_value=loading)),
        ):
            with self.assertRaises(HTTPException) as raised:
                await indexer_routes.put_chat_model_preferences(
                    ModelPreferenceRequest(default_chat_model="openai::gpt-5"),
                    user,
                )
        self.assertEqual(raised.exception.status_code, 503)

        provider_error = indexer_routes.AvailableModelsResponse(
            models=[indexer_routes.AvailableModel(id="gpt-5", name="GPT-5", provider="openai")],
            provider_states=[
                indexer_routes.ProviderModelState(
                    provider="openai",
                    configured=True,
                    connected=True,
                    error="provider refresh failed",
                )
            ],
        )
        with (
            mock.patch.object(indexer_routes, "_get_model_preferences_module", return_value=fake_module),
            mock.patch.object(indexer_routes, "get_available_chat_models", mock.AsyncMock(return_value=provider_error)),
        ):
            with self.assertRaises(HTTPException) as raised:
                await indexer_routes.put_chat_model_preferences(
                    ModelPreferenceRequest(default_chat_model="openai::gpt-5"),
                    user,
                )
        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
