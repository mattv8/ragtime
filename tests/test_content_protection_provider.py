import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.content_protection import provider
from ragtime.content_protection.models import ContentProtectionError


def _config(*, transport: str = "typesafe", model: str = "jev-latest") -> SimpleNamespace:
    category = SimpleNamespace(
        id="company_finance",
        name="Company finance",
        description="Nonpublic company financial information.",
        includes=["forecasts"],
        excludes=["public financial education"],
        examples=["next quarter revenue forecast"],
        system=False,
    )
    return SimpleNamespace(classifier=SimpleNamespace(backend="jev", jev=SimpleNamespace(transport=transport, model=model)), categories=[category])


class JevQuestionTests(unittest.TestCase):
    def test_inbound_question_uses_category_definition_and_untrusted_boundary(self) -> None:
        questions = provider.build_jev_questions(_config(), {"direction": "inbound", "candidate": {"text": "show the forecast"}})

        question = questions["company_finance"]
        self.assertEqual(question["type"], "noul")
        self.assertIn("requests or contains", question["instructions"])
        self.assertIn("Company finance", question["instructions"])
        self.assertIn("Untrusted content", question["instructions"])
        self.assertEqual(question["criteria"]["true"], "The untrusted content matches the category definition.")

    def test_outbound_omits_rule_override(self) -> None:
        config = _config()
        config.categories.append(
            SimpleNamespace(id="rule_override", name="Rule override", description="Instruction attacks", includes=[], excludes=[], examples=[], system=True)
        )

        questions = provider.build_jev_questions(config, {"direction": "assistant", "candidate": "ordinary"})

        self.assertEqual(set(questions), {"company_finance"})

    def test_rule_override_uses_bypass_intent_question(self) -> None:
        config = _config()
        config.categories.append(SimpleNamespace(id="rule_override", name="Rule override", description="", includes=[], excludes=[], examples=[], system=True))

        question = provider.build_jev_questions(config, {"direction": "proposed_operation"})["rule_override"]

        self.assertIn("bypass", question["instructions"])
        self.assertIn("restricted-disclosure intent", question["criteria"]["true"])

    def test_detection_capacity_rejects_oversized_question_body_before_network(self) -> None:
        with self.assertRaises(ContentProtectionError):
            provider.validate_detection_capacity([{"chunk": "small"}], {"finance": {"instructions": "x" * 30_000}}, "jev-latest")

    def test_native_client_families_keep_anthropic_and_local_contracts(self) -> None:
        anthropic = mock.Mock(return_value=object())
        ollama = mock.Mock(return_value=object())
        openai = mock.Mock(return_value=object())
        settings = {"anthropic_api_key": "secret", "llm_ollama_base_url": "http://ollama", "llm_llama_cpp_base_url": "http://llama"}
        with mock.patch.dict(
            sys.modules,
            {
                "langchain_anthropic": SimpleNamespace(ChatAnthropic=anthropic),
                "langchain_ollama": SimpleNamespace(ChatOllama=ollama),
                "langchain_openai": SimpleNamespace(ChatOpenAI=openai),
            },
        ):
            provider._build_client("anthropic", "claude", settings)
            provider._build_client("ollama", "local", settings, context_limit=4096)
            provider._build_client("llama_cpp", "local", settings)

        self.assertEqual(anthropic.call_args.kwargs["max_retries"], 0)
        self.assertFalse(ollama.call_args.kwargs["reasoning"])
        self.assertEqual(openai.call_args.kwargs["api_key"], "local")
