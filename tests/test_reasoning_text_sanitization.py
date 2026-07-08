import unittest

from ragtime.indexer.chat_events import append_reasoning_event
from ragtime.rag.components import RAGComponents


class ReasoningTextSanitizationTests(unittest.TestCase):
    def test_empty_html_comment_markers_are_removed_from_reasoning_summary(self) -> None:
        payload = {
            "summary": [
                {
                    "type": "summary_text",
                    "text": "**Planning schema comparison**\n\n<!-- -->**Inspecting migrations**\n\n<!-- -->",
                }
            ]
        }

        text = RAGComponents._extract_reasoning_text_from_payload(payload)

        self.assertIn("**Planning schema comparison**", text)
        self.assertIn("**Inspecting migrations**", text)
        self.assertNotIn("<!-- -->", text)

    def test_empty_html_comment_markers_inside_fenced_code_are_preserved(self) -> None:
        payload = {
            "reasoning_summary_text": "Before\n```html\n<!-- -->\n```\nAfter",
        }

        text = RAGComponents._extract_reasoning_text_from_payload(payload)

        self.assertIn("```html\n<!-- -->\n```", text)

    def test_split_empty_html_comment_marker_is_removed_after_reasoning_append(self) -> None:
        events: list[dict[str, object]] = []
        started_at = append_reasoning_event(events, "**First**\n\n<!--", None)

        append_reasoning_event(events, " -->**Second**", started_at)

        self.assertEqual(len(events), 1)
        self.assertIn("**First**", str(events[0]["content"]))
        self.assertIn("**Second**", str(events[0]["content"]))
        self.assertNotIn("<!-- -->", str(events[0]["content"]))


if __name__ == "__main__":
    unittest.main()
