from __future__ import annotations

import json
import unittest
from typing import cast
from unittest import mock

from ragtime.api import routes as api_routes


def _stream_events(events: list[object]):
    async def _generator(*_args: object, **_kwargs: object):
        for event in events:
            yield event

    return _generator


async def _collect_stream_chunks(events: list[object]) -> list[dict[str, object] | str]:
    chunks: list[dict[str, object] | str] = []
    with mock.patch.object(api_routes.rag, "process_query_stream", _stream_events(events)):
        async for line in api_routes._stream_response_tokens(
            user_message=object(),
            chat_history=[],
            model="test-model",
        ):
            self_contained_lines = [part for part in line.split("\n") if part.startswith("data: ")]
            for data_line in self_contained_lines:
                payload = data_line.removeprefix("data: ")
                if payload == "[DONE]":
                    chunks.append(payload)
                else:
                    chunks.append(json.loads(payload))
    return chunks


def _deltas(chunks: list[dict[str, object] | str]) -> list[dict[str, object]]:
    deltas: list[dict[str, object]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict):
            deltas.append(delta)
    return deltas


class OpenAIStreamingShapeTests(unittest.IsolatedAsyncioTestCase):
    async def test_reasoning_streams_as_reasoning_content_not_html_content(self) -> None:
        chunks = await _collect_stream_chunks(
            [
                {"type": "reasoning", "content": "without"},
                {"type": "reasoning", "content": " getting"},
                " too complex",
            ]
        )

        deltas = _deltas(chunks)
        self.assertEqual(deltas[0], {"reasoning_content": "without"})
        self.assertEqual(deltas[1], {"reasoning_content": " getting"})
        self.assertEqual(deltas[2], {"content": " too complex"})
        self.assertNotIn('<details type="reasoning">', json.dumps(deltas))
        self.assertEqual(chunks[-1], "[DONE]")

    async def test_tool_blocks_remain_ordered_between_reasoning_chunks(self) -> None:
        chunks = await _collect_stream_chunks(
            [
                {"type": "reasoning", "content": "I should inspect the database."},
                {"type": "tool_start", "tool": "postgres_query", "input": {"sql": "select 1"}},
                {"type": "tool_end", "tool": "postgres_query", "output": "1"},
                {"type": "reasoning", "content": "The query succeeded."},
                "Result is 1.",
            ]
        )

        deltas = _deltas(chunks)
        self.assertEqual(deltas[0], {"reasoning_content": "I should inspect the database."})
        tool_call_content = deltas[1].get("content")
        tool_result_content = deltas[2].get("content")
        self.assertIsInstance(tool_call_content, str)
        self.assertIsInstance(tool_result_content, str)
        tool_call_text = cast(str, tool_call_content)
        tool_result_text = cast(str, tool_result_content)
        self.assertIn('<details type="tool_call">', tool_call_text)
        self.assertIn('<details type="tool_result">', tool_result_text)
        self.assertEqual(deltas[3], {"reasoning_content": "The query succeeded."})
        self.assertEqual(deltas[4], {"content": "Result is 1."})
        self.assertNotIn('<details type="reasoning">', json.dumps(deltas))


if __name__ == "__main__":
    unittest.main()
