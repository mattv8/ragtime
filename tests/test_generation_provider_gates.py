import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi import FastAPI, HTTPException

from ragtime.api import routes as api_routes
from ragtime.indexer import routes as indexer_routes
from ragtime.indexer import title_generation
from ragtime.rag import components
from ragtime.userspace.agent_routes import agent_management_router
from ragtime.userspace.development_credentials_routes import router as development_credentials_router


def _disabled() -> HTTPException:
    return HTTPException(status_code=403, detail={"code": "chat_generation_disabled"})


class GenerationProviderGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_verify_api_key_rejects_rtdev_bearer_when_api_key_is_unset(self) -> None:
        with mock.patch.object(api_routes, "settings", SimpleNamespace(api_key="")):
            with self.assertRaises(HTTPException) as raised:
                await api_routes.verify_api_key("bEaReR rtdev_selector_secret")
        self.assertEqual(raised.exception.status_code, 401)

    async def test_disabled_title_policy_never_invokes_provider(self) -> None:
        provider = SimpleNamespace(ainvoke=mock.AsyncMock())
        with (
            mock.patch.object(title_generation.rag, "llm", provider),
            mock.patch.object(type(title_generation.rag), "is_ready", new_callable=mock.PropertyMock, return_value=True),
            mock.patch.object(title_generation, "require_generation", mock.AsyncMock(side_effect=_disabled())),
        ):
            with self.assertRaises(HTTPException):
                await title_generation._generate_title("summarize this", user_id="disabled-user")
        provider.ainvoke.assert_not_awaited()

    async def test_disabled_title_does_not_use_deterministic_fallback(self) -> None:
        conversation = SimpleNamespace(id="conversation", title="Untitled Chat", user_id="disabled-user")
        with (
            mock.patch.object(title_generation.repository, "get_conversation", mock.AsyncMock(return_value=conversation)),
            mock.patch.object(title_generation, "_generate_title", mock.AsyncMock(side_effect=_disabled())),
            mock.patch.object(title_generation.repository, "update_conversation_title", mock.AsyncMock()) as update_title,
        ):
            await title_generation.update_conversation_title_from_question("conversation", "A deterministic fallback must not run")
        update_title.assert_not_awaited()

    async def test_stream_callback_rechecks_policy_after_first_model_invocation(self) -> None:
        gate = mock.AsyncMock(side_effect=[None, _disabled()])
        with mock.patch.object(components, "require_generation", gate):
            callback = components._GenerationPolicyGateCallback("chat", "caller", "owner")
            await callback.on_chat_model_start()
            with self.assertRaises(HTTPException):
                await callback.on_chat_model_start()
        self.assertEqual(gate.await_args_list[0].args, ("caller", "owner"))
        self.assertEqual(gate.await_args_list[0].kwargs, {"surface": "chat"})

    async def test_rtdev_bearer_is_rejected_by_session_only_surfaces(self) -> None:
        app = FastAPI()
        app.include_router(indexer_routes.router)
        app.include_router(agent_management_router)
        app.include_router(development_credentials_router)
        headers = {"Authorization": "Bearer rtdev_0123456789abcdef0123456789abcdef_secret"}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.gather(
                client.post("/indexes/conversations/conversation/messages", headers=headers, json={"message": "hello"}),
                client.get("/indexes/userspace/workspaces/workspace/agent-access", headers=headers),
                client.get("/indexes/userspace/development/workspaces/workspace/credentials", headers=headers),
            )

        self.assertEqual([response.status_code for response in responses], [401, 401, 401])
