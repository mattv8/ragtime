import json
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.content_protection import provider
from ragtime.content_protection.models import ContentProtectionConfig, ContentProtectionError, Profile


def _config(model: str = "openai::classifier") -> ContentProtectionConfig:
    return ContentProtectionConfig(
        classifier_model=model,
        profiles=[Profile(id="standard", name="Standard", level=0, scope="ordinary operational information")],
    )


class ContentProtectionProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        provider._clients.clear()
        provider._local_context_limits.clear()

    def test_parse_verdict_rejects_duplicates_extraneous_and_inconsistent_values(self) -> None:
        for payload in (
            '{"verdict":"allow","verdict":"deny","reason_code":"permitted"}',
            '{"verdict":"allow","reason_code":"permitted","extra":true}',
            '{"verdict":"deny","reason_code":"permitted"}',
        ):
            with self.subTest(payload=payload), self.assertRaises(ContentProtectionError) as raised:
                provider.parse_verdict(payload)
            self.assertEqual(raised.exception.code, "classifier_invalid_response")

    def test_parse_verdict_keeps_reason_only_for_sample_calls(self) -> None:
        payload = '{"verdict":"deny","reason_code":"restricted_content","reason":"Scope does not permit this."}'
        self.assertNotIn("reason", provider.parse_verdict(payload))
        self.assertEqual(provider.parse_verdict(payload, include_reason=True)["reason"], "Scope does not permit this.")

    def test_parse_verdict_rejects_sample_reason_over_maximum_length(self) -> None:
        payload = '{"verdict":"deny","reason_code":"restricted_content","reason":"' + "x" * 121 + '"}'
        with self.assertRaises(ContentProtectionError) as raised:
            provider.parse_verdict(payload, include_reason=True)
        self.assertEqual(raised.exception.code, "classifier_invalid_response")

    def test_supported_provider_client_options_are_bounded(self) -> None:
        openai_class = mock.Mock(return_value=object())
        anthropic_class = mock.Mock(return_value=object())
        ollama_class = mock.Mock(return_value=object())
        modules = {
            "langchain_openai": SimpleNamespace(ChatOpenAI=openai_class),
            "langchain_anthropic": SimpleNamespace(ChatAnthropic=anthropic_class),
            "langchain_ollama": SimpleNamespace(ChatOllama=ollama_class),
        }
        settings = {
            "openai_api_key": "openai-secret",
            "anthropic_api_key": "anthropic-secret",
            "openrouter_api_key": "router-secret",
            "omlx_api_key": "omlx-secret",
            "llm_ollama_base_url": "http://ollama:11434",
            "llm_llama_cpp_base_url": "http://llama:8080",
            "llm_lmstudio_base_url": "http://studio:1234",
            "llm_omlx_base_url": "http://omlx:8000",
        }
        with mock.patch.dict(sys.modules, modules):
            for configured in ("openai::x", "openrouter::x", "llama.cpp::x", "lmstudio::x", "omlx::x", "anthropic::x", "ollama::x"):
                with self.subTest(configured=configured):
                    provider._build_client(*provider._selected_model(_config(configured)), settings)
        for call in openai_class.call_args_list:
            options = call.kwargs
            self.assertEqual(options["max_tokens"], 256)
            self.assertEqual(options["max_retries"], 0)
            self.assertFalse(options["streaming"])
            self.assertNotIn("tools", options)
        self.assertEqual(anthropic_class.call_args.kwargs["max_tokens_to_sample"], 256)
        self.assertEqual(anthropic_class.call_args.kwargs["max_retries"], 0)
        self.assertFalse(ollama_class.call_args.kwargs["reasoning"])
        self.assertEqual(ollama_class.call_args.kwargs["num_predict"], 256)

    def test_omlx_client_disables_template_thinking_without_changing_local_options(self) -> None:
        openai_class = mock.Mock(return_value=object())
        settings = {
            "omlx_api_key": "omlx-secret",
            "llm_omlx_base_url": "http://omlx:8000",
            "llm_lmstudio_base_url": "http://studio:1234",
        }
        with mock.patch.dict(sys.modules, {"langchain_openai": SimpleNamespace(ChatOpenAI=openai_class)}):
            provider._build_client("omlx", "Qwen3.5-9B", settings)
            provider._build_client("lmstudio", "local-model", settings)

        omlx_options, lmstudio_options = (call.kwargs for call in openai_class.call_args_list)
        self.assertEqual(omlx_options["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertNotIn("enable_thinking", omlx_options)
        self.assertNotIn("extra_body", lmstudio_options)

    async def test_omlx_classify_binds_strict_native_response_format_per_request(self) -> None:
        bound_client = SimpleNamespace(
            ainvoke=mock.AsyncMock(return_value=SimpleNamespace(content='{"verdict":"allow","reason_code":"permitted"}', response_metadata={}))
        )
        client = SimpleNamespace(bind=mock.Mock(return_value=bound_client))
        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value={"omlx_api_key": "secret"})),
            mock.patch.object(provider, "_client_for", return_value=client),
            mock.patch.object(provider, "_preflight_context", mock.AsyncMock(return_value=(8192, False))),
        ):
            with provider.security_classification_context():
                self.assertEqual(
                    (
                        await provider.classify(
                            _config("omlx::Qwen3.5-9B"),
                            {"direction": "inbound", "candidate": "ordinary", "audience_constraints": [{"scope": "ordinary"}]},
                        )
                    )["verdict"],
                    "allow",
                )
                self.assertEqual(
                    (
                        await provider.classify(
                            _config("omlx::Qwen3.5-9B"),
                            {"direction": "sample", "candidate": "ordinary", "audience_constraints": [{"scope": "ordinary"}]},
                            include_reason=True,
                        )
                    )["verdict"],
                    "allow",
                )

        production_schema = client.bind.call_args_list[0].kwargs["response_format"]["json_schema"]["schema"]
        sample_schema = client.bind.call_args_list[1].kwargs["response_format"]["json_schema"]["schema"]
        self.assertEqual(production_schema["required"], ["verdict", "reason_code"])
        self.assertFalse(production_schema["additionalProperties"])
        self.assertEqual(production_schema["properties"]["verdict"]["enum"], ["allow", "deny"])
        self.assertEqual(production_schema["properties"]["reason_code"]["enum"], ["permitted", "restricted_content", "uncertain"])
        self.assertNotIn("reason", production_schema["properties"])
        self.assertEqual(sample_schema["properties"]["reason"]["maxLength"], 120)
        self.assertNotIn("reason", sample_schema["required"])
        self.assertEqual(bound_client.ainvoke.await_args.kwargs["config"], {"callbacks": []})

    def test_omlx_response_format_leaves_other_provider_requests_unchanged(self) -> None:
        response_format = provider._response_format("omlx", include_reason=False)
        assert response_format is not None
        self.assertEqual(response_format["type"], "json_schema")
        self.assertIsNone(provider._response_format("openai", include_reason=False))

    async def test_inbound_opaque_path_policy_preserves_the_exact_candidate(self) -> None:
        client = SimpleNamespace(
            ainvoke=mock.AsyncMock(return_value=SimpleNamespace(content='{"verdict":"allow","reason_code":"permitted"}', response_metadata={}))
        )
        opaque_path = "a1b2c3d4e5f6.txt"
        envelope: dict[str, object] = {
            "direction": "inbound",
            "surface": "development",
            "tool_id": "file_read",
            "operation": "file_read",
            "candidate": {"workspace_id": "workspace-1", "operation": "file_read", "arguments": {"path": opaque_path}},
            "audience_constraints": [{"scope": "ordinary"}],
        }
        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value={"openai_api_key": "secret"})),
            mock.patch.object(provider, "_client_for", return_value=client),
            mock.patch.object(provider, "_preflight_context", mock.AsyncMock(return_value=(8192, False))),
        ):
            with provider.security_classification_context():
                await provider.classify(_config(), envelope)

        messages = client.ainvoke.await_args.args[0]
        self.assertEqual(json.loads(messages[1].content)["data_envelope"], envelope)
        policy = messages[0].content.lower()
        self.assertIn("opaque filename/path tokens are not credentials", policy)
        self.assertIn("do not infer sensitivity from a filename's spelling, entropy, or token pattern", policy)
        self.assertIn("inbound/proposed-operation boundaries", policy)
        self.assertIn("explicit submitted credentials or restricted content remain evidence", policy)
        self.assertNotIn("ignore candidate", policy)

    async def test_classify_uses_messages_disables_callbacks_and_reuses_client(self) -> None:
        client = SimpleNamespace(
            ainvoke=mock.AsyncMock(return_value=SimpleNamespace(content='{"verdict":"allow","reason_code":"permitted"}', response_metadata={}))
        )
        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value={"openai_api_key": "secret"})),
            mock.patch.object(provider, "_build_client", return_value=client) as build_client,
            mock.patch.object(provider, "_preflight_context", mock.AsyncMock(return_value=(8192, False))),
        ):
            envelope: dict[str, object] = {"direction": "inbound", "candidate": "ordinary", "audience_constraints": [{"scope": "ordinary"}]}
            with provider.security_classification_context():
                self.assertEqual((await provider.classify(_config(), envelope))["verdict"], "allow")
                self.assertEqual((await provider.classify(_config(), envelope))["verdict"], "allow")
        self.assertEqual(build_client.call_count, 1)
        messages = client.ainvoke.await_args.args[0]
        self.assertEqual([type(message).__name__ for message in messages], ["SystemMessage", "HumanMessage"])
        self.assertIn("audience constraints", messages[0].content.lower())
        self.assertEqual(client.ainvoke.await_args.kwargs["config"], {"callbacks": []})

    async def test_truncated_or_refused_response_fails_closed_without_provider_detail(self) -> None:
        for response in (
            SimpleNamespace(content='{"verdict":"allow","reason_code":"permitted"}', response_metadata={"finish_reason": "length"}),
            SimpleNamespace(content="", additional_kwargs={"refusal": "candidate text"}, response_metadata={}),
            SimpleNamespace(content='{"verdict":"allow","reason_code":"permitted"}', response_metadata={"done_reason": "length"}),
            SimpleNamespace(content='{"verdict":"allow","reason_code":"permitted"}', response_metadata={}, tool_calls=[{"name": "tool"}]),
        ):
            client = SimpleNamespace(ainvoke=mock.AsyncMock(return_value=response))
            with (
                mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value={"openai_api_key": "secret"})),
                mock.patch.object(provider, "_client_for", return_value=client),
                mock.patch.object(provider, "_preflight_context", mock.AsyncMock(return_value=(8192, False))),
                self.assertRaises(ContentProtectionError) as raised,
            ):
                with provider.security_classification_context():
                    await provider.classify(_config(), {"direction": "inbound", "candidate": "hidden", "audience_constraints": [{"scope": "ordinary"}]})
            self.assertEqual(raised.exception.code, "classifier_invalid_response")
            self.assertNotIn("hidden", raised.exception.request_id)

    async def test_unsupported_provider_and_empty_production_policy_fail_closed(self) -> None:
        with self.assertRaises(ContentProtectionError) as unsupported:
            with provider.security_classification_context():
                await provider.classify(
                    _config("github_copilot::x"), {"direction": "inbound", "candidate": "value", "audience_constraints": [{"scope": "ordinary"}]}
                )
        self.assertEqual(unsupported.exception.code, "classifier_unavailable")
        with self.assertRaises(ContentProtectionError) as policy:
            with provider.security_classification_context():
                await provider.classify(_config(), {"direction": "inbound", "candidate": "value"})
        self.assertEqual(policy.exception.code, "classifier_unavailable")

    async def test_direct_provider_call_requires_private_security_purpose(self) -> None:
        with self.assertRaises(ContentProtectionError) as raised:
            await provider.classify(_config(), {"direction": "inbound", "candidate": "value", "audience_constraints": [{"scope": "ordinary"}]})
        self.assertEqual(raised.exception.code, "classifier_unavailable")

    async def test_oversized_context_is_rejected_before_a_local_client_is_created(self) -> None:
        settings = {"llm_ollama_base_url": "http://ollama:11434"}
        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value=settings)),
            mock.patch.object(provider, "_local_context_limit", mock.AsyncMock(return_value=300)),
            mock.patch.object(provider, "_client_for") as client_for,
            self.assertRaises(ContentProtectionError) as raised,
        ):
            with provider.security_classification_context():
                await provider.classify(
                    _config("ollama::small"),
                    {"direction": "inbound", "candidate": "x" * 1000, "audience_constraints": [{"scope": "ordinary"}]},
                )
        self.assertEqual(raised.exception.code, "content_unclassifiable")
        client_for.assert_not_called()

    async def test_unknown_local_capacity_fails_closed_before_provider_call(self) -> None:
        with (
            mock.patch("ragtime.core.app_settings.get_app_settings", mock.AsyncMock(return_value={"llm_ollama_base_url": "http://ollama:11434"})),
            mock.patch.object(provider, "_local_context_limit", mock.AsyncMock(return_value=None)),
            mock.patch.object(provider, "_client_for") as client_for,
            self.assertRaises(ContentProtectionError) as raised,
        ):
            with provider.security_classification_context():
                await provider.classify(
                    _config("ollama::unknown"),
                    {"direction": "inbound", "candidate": "ordinary", "audience_constraints": [{"scope": "ordinary"}]},
                )
        self.assertEqual(raised.exception.code, "content_unclassifiable")
        client_for.assert_not_called()
