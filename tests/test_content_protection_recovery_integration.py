import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

from ragtime.content_protection import service as protection_service
from ragtime.content_protection.models import ContentProtectionConfig
from ragtime.rag import components as components_module
from ragtime.rag.components import RAGComponents


class HostedRecoveryActualFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_executor_tool_result_deny_runs_once_then_public_entrypoint_recovers(self) -> None:
        recovery_prompts: list[str] = []
        tool_executions: list[str] = []
        components = object.__new__(RAGComponents)

        async def lookup(query: str) -> str:
            tool_executions.append(query)
            return "DENIED-CANARY"

        tool = StructuredTool.from_function(coroutine=lookup, name="lookup", description="Lookup a record")

        class ToolCallingModel(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "test"

            def bind_tools(self, _tools, **_kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **_kwargs):
                if any(isinstance(message, ToolMessage) for message in messages):
                    message = AIMessage(content="ordinary answer")
                else:
                    message = AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"query": "safe"}, "id": "call-1"}])
                return ChatResult(generations=[ChatGeneration(message=message)])

        async def unprotected(_self, *_args):
            executor = _self._build_runtime_executor([tool], "system", llm=ToolCallingModel())
            result = await executor.ainvoke({"input": "safe", "user_input": [], "chat_history": []})
            return result["output"]

        class RecoveryLLM:
            async def ainvoke(self, messages, **_kwargs):
                recovery_prompts.append("\n".join(str(message.content) for message in messages))
                return AIMessage(content="approved alternative")

        async def get_llm(_self, _model):
            return SimpleNamespace(llm=RecoveryLLM(), model="ordinary", provider="fake")

        components._tool_configs = []
        components._app_settings = {}
        config = ContentProtectionConfig(enabled=True, classifier_model="openai::classifier")

        async def classify(_config, envelope, **_kwargs):
            if envelope["direction"] == "tool_result":
                return {"verdict": "deny", "reason": "restricted", "reason_code": "restricted"}
            return {"verdict": "allow", "reason_code": ""}

        with (
            mock.patch.object(protection_service, "load_config", new=mock.AsyncMock(return_value=config)),
            mock.patch.object(protection_service, "resolve_identities", new=mock.AsyncMock(return_value=({"caller", "owner"}, {}, {}))),
            mock.patch.object(protection_service, "_provider_settings_identity", new=mock.AsyncMock(return_value={})),
            mock.patch.object(protection_service, "_classify", new=classify),
            mock.patch.object(protection_service, "_audit", new=mock.AsyncMock()),
            mock.patch.object(components, "_process_query_unprotected", new=MethodType(unprotected, components)),
            mock.patch.object(components, "_get_request_scoped_llm", new=MethodType(get_llm, components)),
            mock.patch("ragtime.rag.components.require_generation", new=mock.AsyncMock()),
        ):
            answer = await components.process_query("safe", user_id="caller", owner_user_id="owner")

        self.assertEqual(answer, "approved alternative")
        self.assertEqual(tool_executions, ["safe"])
        self.assertEqual(len(recovery_prompts), 1)
        self.assertNotIn("DENIED-CANARY", recovery_prompts[0])

    async def test_public_turn_restores_recovery_prompt_scope_after_completion(self) -> None:
        components = object.__new__(RAGComponents)

        async def unprotected(_self, *_args):
            components_module._recovery_request_scope.set(components_module._RecoveryRequestScope("system", "turn", ()))
            return "approved"

        with (
            mock.patch.object(components, "_process_query_unprotected", new=MethodType(unprotected, components)),
            mock.patch("ragtime.rag.components.require_generation", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_history", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_inbound", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_assistant", new=mock.AsyncMock()),
        ):
            self.assertEqual(await components.process_query("safe", user_id="caller"), "approved")

        self.assertIsNone(components_module._recovery_request_scope.get())


if __name__ == "__main__":
    unittest.main()
