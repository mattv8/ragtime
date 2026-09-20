import unittest
from typing import cast
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from mcp.types import CallToolResult, TextContent

from ragtime.content_protection import service
from ragtime.content_protection.models import ContentProtectionConfig, ContentProtectionError, ProtectionContext
from ragtime.content_protection.provider import parse_verdict


class ContentProtectionCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_policy_does_not_invoke_provider_or_mutate_candidate(self) -> None:
        candidate = {"text": "unchanged"}
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(return_value=ContentProtectionConfig(enabled=False))),
            mock.patch.object(service, "classify", mock.AsyncMock()) as classify,
        ):
            await service.authorize_content(candidate, direction="inbound", context=ProtectionContext(user_id="u"))
        self.assertEqual(candidate, {"text": "unchanged"})
        classify.assert_not_awaited()

    async def test_required_binary_candidate_is_rejected_without_provider_call(self) -> None:
        config = ContentProtectionConfig(enabled=True)
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(return_value=config)),
            mock.patch.object(service, "classify", mock.AsyncMock()) as classify,
        ):
            with self.assertRaises(ContentProtectionError) as raised:
                await service.authorize_content(b"opaque", direction="inbound", context=ProtectionContext(user_id="u"))
        self.assertEqual(raised.exception.code, "content_unclassifiable")
        classify.assert_not_awaited()

    def test_public_error_never_contains_model_reason(self) -> None:
        detail = ContentProtectionError("content_denied", "request-1").public_detail()
        self.assertEqual(detail["code"], "content_denied")
        self.assertNotIn("reason", detail)

    def test_truncated_or_inconsistent_provider_verdict_is_rejected(self) -> None:
        with self.assertRaises(ContentProtectionError):
            parse_verdict('{"verdict":"allow"')
        with self.assertRaises(ContentProtectionError):
            parse_verdict('{"verdict":"allow","reason_code":"uncertain"}')

    async def test_real_langchain_history_is_losslessly_normalized_for_openai_transport(self) -> None:
        config = ContentProtectionConfig(enabled=True, classifier_model="openai::classifier")
        policy = service._ResolvedPolicy(True, "all_supported_traffic", {"u"}, {"u": set()}, {"u": None}, [[{"id": "standard", "scope": "ordinary"}]])
        provider_envelopes: list[dict[str, object]] = []

        async def stub_provider(_config, envelope, **_kwargs):
            provider_envelopes.append(envelope)
            return {"verdict": "allow", "reason_code": "permitted"}

        history = [
            HumanMessage(content="earlier request", additional_kwargs={"trace": "one"}),
            AIMessage(content="calling tool", tool_calls=[{"name": "lookup", "args": {"account": "A-1"}, "id": "call-1"}]),
            ToolMessage(content="tool result", tool_call_id="call-1", additional_kwargs={"source": "database"}),
        ]
        context = ProtectionContext(user_id="u", audience_user_ids=("u",), surface="openai_api")
        with (
            mock.patch.object(service, "load_config", new=mock.AsyncMock(return_value=config)),
            mock.patch.object(service, "_resolve", new=mock.AsyncMock(return_value=policy)),
            mock.patch.object(service, "_provider_settings_identity", new=mock.AsyncMock(return_value="provider")),
            mock.patch.object(service, "classify", new=stub_provider),
        ):
            await service.authorize_content(history, direction="stored_readback", context=context)
            await service.authorize_content("follow-up", direction="inbound", context=context, supporting_context=history)

        self.assertEqual(len(provider_envelopes), 2)
        normalized_history = cast(list[dict[str, object]], provider_envelopes[0]["candidate"])
        self.assertIsInstance(normalized_history, list)
        self.assertEqual([message["type"] for message in normalized_history], ["human", "ai", "tool"])
        self.assertEqual(cast(list[dict[str, object]], normalized_history[1]["tool_calls"])[0]["args"], {"account": "A-1"})
        self.assertEqual(provider_envelopes[1]["supporting_context"], normalized_history)

    async def test_real_langchain_image_history_remains_unclassifiable(self) -> None:
        config = ContentProtectionConfig(enabled=True, classifier_model="openai::classifier")
        with (
            mock.patch.object(service, "load_config", new=mock.AsyncMock(return_value=config)),
            mock.patch.object(service, "classify", new=mock.AsyncMock()) as classify,
        ):
            with self.assertRaises(ContentProtectionError) as raised:
                await service.authorize_content(
                    [HumanMessage(content=[{"type": "image_url", "image_url": {"url": "https://example.test/private.png"}}])],
                    direction="stored_readback",
                    context=ProtectionContext(user_id="u", surface="openai_api"),
                )
        self.assertEqual(raised.exception.code, "content_unclassifiable")
        classify.assert_not_awaited()

    def test_mcp_transport_wrapper_is_normalized_without_string_coercion(self) -> None:
        wrapper = CallToolResult(content=[TextContent(type="text", text="approved tool response")])
        normalized = service.normalize_transport_value(wrapper)
        self.assertEqual(normalized["content"], [{"type": "text", "text": "approved tool response", "annotations": None, "meta": None}])
