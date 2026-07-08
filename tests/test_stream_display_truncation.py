import json
import unittest

from ragtime.rag.components import (
    STREAM_DISPLAY_UNTRUNCATED_TOOL_NAMES,
    compact_stream_display_output,
)


class StreamDisplayTruncationTests(unittest.IsolatedAsyncioTestCase):
    def _spawn_payload(self, final_output_lengths: list[int], suffix: str = "") -> str:
        subagents = [
            {
                "conversation_id": f"conv-{i:03d}",
                "task_id": f"task-{i:03d}",
                "status": "completed" if i % 2 == 0 else "failed",
                "name": f"Agent {i}",
                "role": "worker" if i % 2 == 0 else "reviewer",
                "file_scope": [f"file{i}.py"],
                "final_output": ("x" * length) + suffix,
            }
            for i, length in enumerate(final_output_lengths)
        ]
        return json.dumps({"subagents": subagents}, indent=2)

    def test_spawn_subagents_long_output_truncates_each_final_output(self):
        payload = self._spawn_payload([1500, 900, 1200])
        self.assertGreater(len(payload), 2000)

        result = compact_stream_display_output("spawn_subagents", payload)

        self.assertIsInstance(result, str)
        parsed = json.loads(result)
        self.assertIn("subagents", parsed)
        self.assertEqual(len(parsed["subagents"]), 3)
        self.assertTrue(parsed.get("_ragtime_output_truncated"))

        for original, truncated in zip(json.loads(payload)["subagents"], parsed["subagents"]):
            self.assertEqual(truncated["conversation_id"], original["conversation_id"])
            self.assertEqual(truncated["task_id"], original["task_id"])
            self.assertEqual(truncated["status"], original["status"])
            self.assertEqual(truncated["name"], original["name"])
            self.assertEqual(truncated["role"], original["role"])
            self.assertEqual(truncated["file_scope"], original["file_scope"])
            self.assertLessEqual(len(truncated["final_output"]), 800 + 30)
            self.assertIn("characters omitted", truncated["final_output"])

    def test_spawn_subagents_output_under_2000_unchanged(self):
        payload = self._spawn_payload([10, 20, 30])
        self.assertLessEqual(len(payload), 2000)

        result = compact_stream_display_output("spawn_subagents", payload)

        self.assertEqual(result, payload)
        parsed = json.loads(result)
        # No flag is added when nothing is shortened.
        self.assertNotIn("_ragtime_output_truncated", parsed)

    def test_generic_json_dict_uses_structured_truncation(self):
        long_value = "y" * 3000
        payload = json.dumps({"tool": "some_tool", "result": long_value}, indent=2)
        self.assertGreater(len(payload), 2000)

        result = compact_stream_display_output("some_tool", payload)

        parsed = json.loads(result)
        self.assertLessEqual(len(result), 2000)
        self.assertTrue(parsed.get("_ragtime_output_truncated"))
        self.assertIn("tool", parsed)

    def test_non_json_string_blind_slice(self):
        text = "z" * 4000
        result = compact_stream_display_output("some_tool", text)

        self.assertTrue(result.endswith("... (truncated)"))
        self.assertEqual(len(result), 2015)

    def test_exempt_tool_unchanged(self):
        text = "a" * 3000
        for exempt in STREAM_DISPLAY_UNTRUNCATED_TOOL_NAMES:
            result = compact_stream_display_output(exempt, text)
            self.assertEqual(
                result,
                text,
                f"exempt tool {exempt} should not be truncated",
            )


if __name__ == "__main__":
    unittest.main()
