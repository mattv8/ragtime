import asyncio
import unittest
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Awaitable, Callable, cast
from unittest import mock

from ragtime.content_protection import hosted
from ragtime.content_protection import service as protection_service
from ragtime.content_protection.models import ContentProtectionConfig, ContentProtectionError, ProtectionContext, Requirement
from ragtime.content_protection.policy import resolve_required
from ragtime.indexer import routes as indexer_routes
from ragtime.rag.components import RAGComponents

if TYPE_CHECKING:
    from prisma.models import User

    from ragtime.indexer.models import Conversation


@dataclass
class _Context:
    tool_id: str | None = None


class _Service:
    def __init__(self) -> None:
        self.context = _Context()
        self.calls: list[tuple[str, object, str | None]] = []
        self.classification_required: Callable[..., Awaitable[bool]] = self._classification_required
        self.authorize_content: Callable[..., Awaitable[None]] = self._authorize_content

    def current_context(self):
        return self.context

    async def _classification_required(self, _context=None):
        return True

    def protection_context(self, _context):
        return nullcontext()

    async def load_config(self):
        return SimpleNamespace(enabled=False, requirements=())

    async def _authorize_content(self, candidate, *, direction, context=None, tool_id=None, **_kwargs):
        self.calls.append((direction, candidate, tool_id or getattr(context, "tool_id", None)))


class _ProtectionError(Exception):
    def public_detail(self):
        return {
            "code": "content_denied",
            "message": "This request conflicts with the access policy. Try rephrasing your question.",
            "reason": "This request conflicts with the access policy.",
            "next_step": "Try rephrasing your question.",
            "request_id": "ref-1",
            "reason_code": "profile_mismatch",
        }


class _Tool:
    name = "canonical_tool"

    def __init__(self, coroutine, func=None, name="canonical_tool"):
        self.name = name
        self.coroutine = coroutine
        self.func = func


class HostedProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_arguments_are_checked_before_execution_and_result_before_return(self) -> None:
        service = _Service()
        observed: list[str] = []

        async def original(**kwargs):
            observed.append("executed")
            self.assertEqual(service.calls[0][0], "proposed_operation")
            return {"result": kwargs["query"]}

        def clone(_tool, **overrides):
            return _Tool(overrides["coroutine"])

        with mock.patch.object(hosted, "_service", return_value=service):
            guarded = hosted.wrap_tools([_Tool(original)], clone)[0]
            result = await guarded.coroutine(query="safe")

        self.assertEqual(result, {"result": "safe"})
        self.assertEqual(observed, ["executed"])
        self.assertEqual(
            service.calls,
            [
                ("proposed_operation", {"query": "safe"}, "canonical_tool"),
                ("tool_result", {"result": "safe"}, "canonical_tool"),
            ],
        )

    async def test_buffered_stream_withholds_text_until_complete_segment_is_authorized(self) -> None:
        service = _Service()

        async def stream():
            yield "first "
            yield "segment"
            yield {"type": "tool_start", "tool": "canonical_tool", "input": {"q": "x"}}

        with mock.patch.object(hosted, "_service", return_value=service):
            released = [event async for event in hosted.buffered_stream(stream(), context=service.context)]

        self.assertEqual(released, ["first segment", {"type": "tool_start", "tool": "canonical_tool", "input": {"q": "x"}}])
        self.assertEqual(service.calls[0][:2], ("assistant_response", "first segment"))
        self.assertEqual(service.calls[1], ("proposed_operation", {"q": "x"}, "canonical_tool"))

    async def test_synchronous_tool_is_guarded_without_bypassing_the_executor_boundary(self) -> None:
        service = _Service()
        observed: list[str] = []

        def original(query):
            observed.append("executed")
            self.assertEqual(service.calls[0][0], "proposed_operation")
            return {"result": query}

        def clone(_tool, **overrides):
            return _Tool(overrides["coroutine"], overrides["func"])

        with mock.patch.object(hosted, "_service", return_value=service):
            guarded = hosted.wrap_tools([_Tool(None, original)], clone)[0]
            result = guarded.func(query="safe")

        self.assertEqual(result, {"result": "safe"})
        self.assertEqual(observed, ["executed"])
        self.assertEqual([call[0] for call in service.calls], ["proposed_operation", "tool_result"])

    def test_public_error_event_never_uses_the_raw_exception_text(self) -> None:
        error = _ProtectionError("secret source and denied body")
        event = hosted.public_error_event(error)
        self.assertEqual(event["code"], "content_denied")
        self.assertEqual(event["reason"], "This request conflicts with the access policy.")
        self.assertEqual(event["next_step"], "Try rephrasing your question.")
        self.assertNotIn("secret source", str(event))

    async def test_configured_tool_uses_durable_id_and_child_task_keeps_turn_scope(self) -> None:
        policy = ContentProtectionConfig(
            enabled=True,
            classifier_model="openai::classifier",
            coverage_mode="selected_scopes",
            requirements=[Requirement(scope_kind="tool", scope_key="tool-config-42", mode="require")],
        )
        self.assertFalse(resolve_required(policy, ProtectionContext(user_id="caller", tool_id="query_finance"))[0])
        self.assertTrue(resolve_required(policy, ProtectionContext(user_id="caller", tool_id="tool-config-42"))[0])

        service = _Service()
        service.classification_required = mock.AsyncMock(side_effect=lambda context: context.tool_id == "tool-config-42")

        async def protected_tool(**_kwargs):
            return {"ok": True}

        async def unrelated_tool(**_kwargs):
            return {"ok": True}

        def clone(tool, **overrides):
            return _Tool(overrides["coroutine"], overrides.get("func"))

        protected = _Tool(protected_tool, name="query_finance")
        unrelated = _Tool(unrelated_tool, name="search_builtin")
        with mock.patch.object(hosted, "_service", return_value=service), hosted.bind_context(service.context):
            wrapped = hosted.wrap_tools(
                [protected, unrelated],
                clone,
                tool_ids_by_name={"query_finance": "tool-config-42"},
            )
            await asyncio.create_task(wrapped[0].coroutine(query="safe"))
            await wrapped[1].coroutine(query="safe")
            await hosted.authorize_assistant("downstream answer", context=service.context)

        self.assertEqual(
            service.calls,
            [
                ("proposed_operation", {"query": "safe"}, "tool-config-42"),
                ("tool_result", {"ok": True}, "tool-config-42"),
                ("proposed_operation", {"query": "safe"}, "search_builtin"),
                ("tool_result", {"ok": True}, "search_builtin"),
                ("assistant_response", "downstream answer", "tool-config-42"),
            ],
        )

    async def test_buffered_stream_does_not_release_raw_tool_error_when_unauthorized(self) -> None:
        service = _Service()
        service.classification_required = mock.AsyncMock(return_value=True)

        async def deny_raw_error(candidate, **kwargs):
            if isinstance(candidate, dict) and candidate.get("type") == "tool_end":
                raise _ProtectionError()
            service.calls.append((kwargs["direction"], candidate, kwargs.get("tool_id") or getattr(kwargs.get("context"), "tool_id", None)))

        service.authorize_content = deny_raw_error

        async def stream():
            yield {"type": "tool_end", "tool": "query_finance", "output": "secret raw error"}

        with mock.patch.object(hosted, "_service", return_value=service):
            with self.assertRaises(_ProtectionError):
                _ = [event async for event in hosted.buffered_stream(stream(), context=service.context)]

    async def test_disabled_or_never_policy_passes_stream_events_through_without_buffering(self) -> None:
        service = _Service()
        service.classification_required = mock.AsyncMock(return_value=False)

        async def stream():
            yield "first "
            yield "second"

        with mock.patch.object(hosted, "_service", return_value=service):
            released = [event async for event in hosted.buffered_stream(stream(), context=service.context)]

        self.assertEqual(released, ["first ", "second"])
        self.assertEqual(service.calls, [])

    async def test_buffered_stream_rejects_oversize_content_before_releasing_it(self) -> None:
        service = _Service()
        service.classification_required = mock.AsyncMock(return_value=True)

        async def stream():
            yield "x" * (1024 * 1024 + 1)

        with mock.patch.object(hosted, "_service", return_value=service):
            with self.assertRaises(ContentProtectionError) as raised:
                _ = [event async for event in hosted.buffered_stream(stream(), context=service.context)]

        self.assertEqual(raised.exception.code, "content_unclassifiable")

    def test_hosted_context_preserves_an_already_bound_workspace_surface_and_audience(self) -> None:
        outer = ProtectionContext(
            user_id="caller",
            audience_user_ids=("caller", "owner", "collaborator"),
            surface="workspace_chat",
            baseline="user",
        )
        with protection_service.protection_context(outer):
            context = hosted.hosted_context(user_id="caller", owner_user_id="owner")

        self.assertEqual(context.surface, "workspace_chat")
        self.assertEqual(context.audience_user_ids, ("caller", "owner", "collaborator"))

    def test_runtime_tool_aliases_map_to_durable_tool_config_ids(self) -> None:
        components = object.__new__(RAGComponents)
        components._tool_configs = [
            {"id": "pdm-id", "name": "Vault", "tool_type": "solidworks_pdm"},
            {"id": "influx-id", "name": "Data--Flux", "tool_type": "influxdb"},
        ]
        self.assertEqual(
            components._content_protection_tool_ids(),
            {
                "search_vault": "pdm-id",
                "lookup_vault": "pdm-id",
                "query_data__flux": "influx-id",
            },
        )

    def test_influx_runtime_alias_preserves_factory_whitespace_transform(self) -> None:
        components = object.__new__(RAGComponents)
        components._tool_configs = [
            {"id": "influx-id", "name": " Data Flux ", "tool_type": "influxdb"},
        ]

        self.assertEqual(components._content_protection_tool_ids(), {"query__data_flux_": "influx-id"})

    def test_route_context_uses_explicit_workspace_and_shared_surfaces(self) -> None:
        workspace = indexer_routes._conversation_protection_context(user_id="caller", owner_user_id="owner", workspace_id="workspace-1")
        shared = indexer_routes._conversation_protection_context(user_id="caller", owner_user_id="owner", shared=True)
        self.assertEqual(workspace.surface, "workspace_chat")
        self.assertEqual(shared.surface, "shared_chat")

    async def test_nonstreaming_send_authorizes_once_before_persisting(self) -> None:
        class StopAfterInbound(Exception):
            pass

        async def stop_after_inbound(*_args, **_kwargs):
            raise StopAfterInbound()

        request = indexer_routes.SendMessageRequest(message="safe")
        conversation = SimpleNamespace(id="conversation-1", user_id="owner-1")
        user = SimpleNamespace(id="caller-1")
        with mock.patch.object(indexer_routes, "authorize_inbound", side_effect=stop_after_inbound) as authorize:
            with self.assertRaises(StopAfterInbound):
                await indexer_routes._send_message_to_loaded_conversation(
                    cast("Conversation", conversation),
                    request,
                    cast("User", user),
                    workspace_id="workspace-1",
                    blocked_tool_names=set(),
                )

        authorize.assert_awaited_once()
        await_args = authorize.await_args
        assert await_args is not None
        self.assertEqual(await_args.kwargs["context"].surface, "workspace_chat")

    async def test_process_query_binds_caller_owner_and_authorizes_before_and_after_model(self) -> None:
        calls: list[tuple[str, object]] = []

        class FakeComponents:
            async def _process_query_unprotected(self, *_args):
                calls.append(("model", None))
                return "approved answer"

        context = _Context()

        async def authorize_history(candidate, **_kwargs):
            calls.append(("history", candidate))

        async def authorize_inbound(candidate, **_kwargs):
            calls.append(("inbound", candidate))

        async def authorize_assistant(candidate, **_kwargs):
            calls.append(("assistant", candidate))

        with (
            mock.patch("ragtime.rag.components.require_generation", new=mock.AsyncMock()) as generation_gate,
            mock.patch("ragtime.rag.components.content_protection_context", return_value=context) as make_context,
            mock.patch("ragtime.rag.components.bind_content_protection_context", return_value=nullcontext()),
            mock.patch("ragtime.rag.components.authorize_history", side_effect=authorize_history),
            mock.patch("ragtime.rag.components.authorize_inbound", side_effect=authorize_inbound),
            mock.patch("ragtime.rag.components.authorize_assistant", side_effect=authorize_assistant),
        ):
            answer = await RAGComponents.process_query(
                cast(RAGComponents, FakeComponents()),
                "request",
                ["prior"],
                user_id="caller",
                owner_user_id="owner",
            )

        self.assertEqual(answer, "approved answer")
        generation_gate.assert_awaited_once_with("caller", "owner")
        make_context.assert_called_once_with(user_id="caller", owner_user_id="owner", surface="chat")
        self.assertEqual(calls, [("history", ["prior"]), ("inbound", "request"), ("model", None), ("assistant", "approved answer")])

    async def test_process_query_stream_checks_generation_gate_before_classification(self) -> None:
        class FakeComponents:
            def _content_protection_tool_ids(self):
                return {}

            async def _process_query_stream_unprotected(self, *_args):
                yield "answer"

        context = _Context()

        async def passthrough(stream, **_kwargs):
            async for event in stream:
                yield event

        with (
            mock.patch("ragtime.rag.components.require_generation", new=mock.AsyncMock()) as generation_gate,
            mock.patch("ragtime.rag.components.content_protection_context", return_value=context),
            mock.patch("ragtime.rag.components.bind_content_protection_context", return_value=nullcontext()),
            mock.patch("ragtime.rag.components.authorize_history", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.authorize_inbound", new=mock.AsyncMock()),
            mock.patch("ragtime.rag.components.content_protection_buffered_stream", side_effect=passthrough),
        ):
            events = [
                event
                async for event in RAGComponents.process_query_stream(
                    cast(RAGComponents, FakeComponents()),
                    "request",
                    ["prior"],
                    user_id="caller",
                    owner_user_id="owner",
                )
            ]

        self.assertEqual(events, ["answer"])
        generation_gate.assert_awaited_once_with("caller", "owner")
