import unittest

from ragtime.core.usage_accounting import normalize_provider_model_rows, normalize_usage_provider_model


class UsageAccountingNormalizationTests(unittest.TestCase):
    def test_scoped_model_splits_provider_and_model(self) -> None:
        self.assertEqual(
            normalize_usage_provider_model("openai_codex::gpt-5.5", "openai_codex::gpt-5.5"),
            ("openai_codex", "gpt-5.5"),
        )

    def test_scoped_model_infers_provider_when_provider_missing(self) -> None:
        self.assertEqual(
            normalize_usage_provider_model("", "github_copilot::claude-sonnet-4.5"),
            ("github_copilot", "claude-sonnet-4.5"),
        )

    def test_openrouter_model_keeps_publisher_prefix_as_model_id(self) -> None:
        self.assertEqual(
            normalize_usage_provider_model(
                "openrouter::deepseek/deepseek-v4-pro",
                "openrouter::deepseek/deepseek-v4-pro",
            ),
            ("openrouter", "deepseek/deepseek-v4-pro"),
        )

    def test_provider_rows_are_normalized_and_coalesced(self) -> None:
        rows = [
            {
                "provider": "openai_codex::gpt-5.5",
                "model": "openai_codex::gpt-5.5",
                "request_source": "ui",
                "total_requests": 2,
                "total_input_tokens": 10,
                "total_output_tokens": 20,
                "total_tokens": 30,
            },
            {
                "provider": "openai_codex",
                "model": "gpt-5.5",
                "request_source": "ui",
                "total_requests": 3,
                "total_input_tokens": 40,
                "total_output_tokens": 50,
                "total_tokens": 90,
            },
        ]

        self.assertEqual(
            normalize_provider_model_rows(rows),
            [
                {
                    "provider": "openai_codex",
                    "model": "gpt-5.5",
                    "request_source": "ui",
                    "total_requests": 5,
                    "total_input_tokens": 50,
                    "total_output_tokens": 70,
                    "total_tokens": 120,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
