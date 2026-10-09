import unittest
from typing import cast
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from mcp.types import CallToolResult, TextContent

from ragtime.content_protection import service
from ragtime.content_protection.models import AccessLevel, ContentProtectionConfig, ContentProtectionError, ProtectionContext


class ContentProtectionCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_policy_does_not_invoke_provider_or_mutate_candidate(self) -> None:
        candidate = {"text": "unchanged"}
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(return_value=ContentProtectionConfig(enabled=False))),
            mock.patch.object(service, "detect", mock.AsyncMock()) as detect,
        ):
            await service.authorize_content(candidate, direction="inbound", context=ProtectionContext(user_id="u"))
        self.assertEqual(candidate, {"text": "unchanged"})
        detect.assert_not_awaited()

    async def test_required_binary_candidate_is_rejected_without_provider_call(self) -> None:
        config = ContentProtectionConfig(enabled=True)
        with (
            mock.patch.object(service, "load_config", mock.AsyncMock(return_value=config)),
            mock.patch.object(service, "detect", mock.AsyncMock()) as detect,
        ):
            with self.assertRaises(ContentProtectionError) as raised:
                await service.authorize_content(b"opaque", direction="inbound", context=ProtectionContext(user_id="u"))
        self.assertEqual(raised.exception.code, "content_unclassifiable")
        detect.assert_not_awaited()

    def test_public_error_contains_reason_and_next_step(self) -> None:
        detail = ContentProtectionError("content_denied", "request-1", reason="Policy excludes this request.", reason_code="restricted_content").public_detail()
        self.assertEqual(detail["code"], "content_denied")
        self.assertEqual(detail["reason"], "Policy excludes this request.")
        self.assertIn(detail["reason"], detail["message"])
        self.assertIn(detail["next_step"], detail["message"])
        self.assertEqual(detail["reason_code"], "restricted_content")

    def test_invalid_probability_contract_is_rejected(self) -> None:
        config = ContentProtectionConfig(enabled=True)
        with self.assertRaises(ContentProtectionError):
            service._authorize_probabilities(config, {"operational": "unknown"}, set())

    async def test_readiness_uses_fixed_default_fixture_before_custom_capacity_check(self) -> None:
        defaults = ContentProtectionConfig()
        custom = ContentProtectionConfig(
            categories=[defaults.categories[0], defaults.categories[-1]],
            access_levels=[AccessLevel(id="standard", name="Standard", granted_category_ids=["operational"])],
        )
        calls: list[ContentProtectionConfig] = []

        async def detect(config, envelope):
            calls.append(config)
            probabilities = {category.id: 0.0 for category in config.categories}
            candidate = str(envelope["candidate"])
            if "forecast" in candidate:
                probabilities["company_finance"] = 1.0
            elif "TYPESAFE_API_KEY" in candidate:
                probabilities["credentials"] = 1.0
            elif "ignore all rules" in candidate:
                probabilities["rule_override"] = 1.0
            return {"probabilities": probabilities, "model": "test", "usage": {"input_tokens": 1, "output_tokens": 1}, "transport": "test", "cache_hit": False}

        with mock.patch.object(service, "detect", new=detect):
            result = await service.probe_readiness(custom)

        self.assertEqual(result["verdict"], "allow")
        self.assertEqual(len(calls), 5)
        self.assertEqual(
            {category.id for category in calls[0].categories}, {"operational", "company_finance", "personnel", "strategic", "credentials", "rule_override"}
        )
        self.assertEqual([category.id for category in calls[-1].categories], ["operational", "rule_override"])

    async def test_real_langchain_history_is_losslessly_normalized_for_openai_transport(self) -> None:
        config = ContentProtectionConfig(enabled=True, classifier={"backend": "llm", "llm_model": "openai::classifier"})
        policy = service._ResolvedPolicy(
            True,
            "all_supported_traffic",
            {"u"},
            {"u": set()},
            {"u": None},
            [[{"id": "standard", "granted_category_ids": ["operational"], "guidance": ""}]],
            {"operational"},
        )
        provider_envelopes: list[dict[str, object]] = []

        async def stub_provider(_config, envelope):
            provider_envelopes.append(envelope)
            return {
                "probabilities": {
                    category.id: 0
                    for category in _config.categories
                    if envelope["direction"] in {"inbound", "proposed_operation"} or category.id != "rule_override"
                },
                "model": "test",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "transport": "test",
                "cache_hit": False,
            }

        history = [
            HumanMessage(content="earlier request", additional_kwargs={"trace": "one"}),
            AIMessage(content="calling tool", tool_calls=[{"name": "lookup", "args": {"account": "A-1"}, "id": "call-1"}]),
            ToolMessage(content="tool result", tool_call_id="call-1", additional_kwargs={"source": "database"}),
        ]
        context = ProtectionContext(user_id="u", audience_user_ids=("u",), surface="openai_api")
        with (
            mock.patch.object(service, "load_config", new=mock.AsyncMock(return_value=config)),
            mock.patch.object(service, "_resolve", new=mock.AsyncMock(return_value=policy)),
            mock.patch.object(service, "_provider_settings_identity", new=mock.AsyncMock(return_value="settings")),
            mock.patch.object(service, "detect", new=stub_provider),
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
        config = ContentProtectionConfig(enabled=True, classifier={"backend": "llm", "llm_model": "openai::classifier"})
        with (
            mock.patch.object(service, "load_config", new=mock.AsyncMock(return_value=config)),
            mock.patch.object(service, "detect", new=mock.AsyncMock()) as detect,
        ):
            with self.assertRaises(ContentProtectionError) as raised:
                await service.authorize_content(
                    [HumanMessage(content=[{"type": "image_url", "image_url": {"url": "https://example.test/private.png"}}])],
                    direction="stored_readback",
                    context=ProtectionContext(user_id="u", surface="openai_api"),
                )
        self.assertEqual(raised.exception.code, "content_unclassifiable")
        detect.assert_not_awaited()

    def test_mcp_transport_wrapper_is_normalized_without_string_coercion(self) -> None:
        wrapper = CallToolResult(content=[TextContent(type="text", text="approved tool response")])
        normalized = service.normalize_transport_value(wrapper)
        self.assertEqual(normalized["content"], [{"type": "text", "text": "approved tool response", "annotations": None, "meta": None}])
