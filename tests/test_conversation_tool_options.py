import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.indexer.conversation_tool_options import (
    load_conversation_tool_options,
    normalize_conversation_tool_options,
    resolve_effective_allow_write,
)
from ragtime.rag import components as rag_components


class ConversationToolOptionTests(unittest.TestCase):
    def test_normalizes_boolean_options_and_drops_empty_rows(self) -> None:
        options = normalize_conversation_tool_options(
            {
                "tool-readonly": {"write_access_enabled": True},
                "tool-global-write": {"read_only_enabled": True, "ignored": True},
                "tool-empty": {"write_access_enabled": False, "read_only_enabled": False},
                "": {"write_access_enabled": True},
                "tool-invalid": "yes",
            }
        )

        self.assertEqual(
            options,
            {
                "tool-readonly": {"write_access_enabled": True},
                "tool-global-write": {"read_only_enabled": True},
            },
        )

    def test_resolves_effective_write_from_global_and_conversation_override(self) -> None:
        self.assertFalse(resolve_effective_allow_write(False, None))
        self.assertTrue(resolve_effective_allow_write(True, None))
        self.assertTrue(resolve_effective_allow_write(False, {"write_access_enabled": True}))
        self.assertFalse(resolve_effective_allow_write(True, {"read_only_enabled": True}))
        self.assertFalse(
            resolve_effective_allow_write(
                True,
                {"write_access_enabled": True, "read_only_enabled": True},
            )
        )

    def test_loads_persisted_options_for_one_tool_row(self) -> None:
        self.assertEqual(
            load_conversation_tool_options({"write_access_enabled": True, "read_only_enabled": False, "ignored": True}),
            {"write_access_enabled": True},
        )


class ConversationToolRuntimeOverrideTests(unittest.IsolatedAsyncioTestCase):
    async def test_override_rebuild_does_not_add_unselected_sibling_tool(self) -> None:
        rag = rag_components.RAGComponents.__new__(rag_components.RAGComponents)
        rag._tool_configs = [
            {
                "id": "tool-1",
                "name": "Demo",
                "tool_type": "postgres",
                "allow_write": False,
            }
        ]
        rag._app_settings = {"max_tool_output_chars": 0}

        rebuilt_tools = [
            SimpleNamespace(name="query_demo"),
            SimpleNamespace(name="search_demo_schema"),
        ]
        option_rows = [SimpleNamespace(toolConfigId="tool-1", options={"write_access_enabled": True})]
        fake_db = SimpleNamespace(
            conversationtooloption=SimpleNamespace(
                find_many=mock.AsyncMock(return_value=option_rows),
            )
        )

        with (
            mock.patch.object(
                rag,
                "build_tools_from_runtime_config",
                new=mock.AsyncMock(return_value=rebuilt_tools),
            ),
            mock.patch.object(rag_components, "get_db", mock.AsyncMock(return_value=fake_db)),
        ):
            runtime_tools = await rag._apply_conversation_tool_overrides(
                "conversation-1",
                [SimpleNamespace(name="query_demo")],
            )

        self.assertEqual([tool.name for tool in runtime_tools], ["query_demo"])


if __name__ == "__main__":
    unittest.main()
