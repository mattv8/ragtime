from __future__ import annotations

import unittest

from ragtime.core.visualization_tools import (
    CHART_TOOL_NAME,
    DATATABLE_TOOL_NAME,
    HTML_COMPONENT_TOOL_NAME,
    VISUALIZATION_MARKER_BY_TYPE,
    VISUALIZATION_TOOL_NAME_BY_TYPE,
    VISUALIZATION_TOOL_NAMES,
    is_visualization_tool_name,
    visualization_tool_name_for_type,
)


class VisualizationToolsRegistryTests(unittest.TestCase):
    def test_tool_names(self) -> None:
        self.assertEqual(CHART_TOOL_NAME, "create_chart")
        self.assertEqual(DATATABLE_TOOL_NAME, "create_datatable")
        self.assertEqual(HTML_COMPONENT_TOOL_NAME, "create_html_component")
        self.assertEqual(VISUALIZATION_TOOL_NAMES, frozenset({"create_chart", "create_datatable", "create_html_component"}))

    def test_type_to_name_mapping(self) -> None:
        self.assertEqual(
            VISUALIZATION_TOOL_NAME_BY_TYPE,
            {"chart": CHART_TOOL_NAME, "datatable": DATATABLE_TOOL_NAME, "html_component": HTML_COMPONENT_TOOL_NAME},
        )
        self.assertEqual(set(VISUALIZATION_TOOL_NAME_BY_TYPE.values()), set(VISUALIZATION_TOOL_NAMES))

    def test_type_to_marker_mapping(self) -> None:
        self.assertEqual(
            VISUALIZATION_MARKER_BY_TYPE,
            {"chart": "__chart__", "datatable": "__datatable__", "html_component": "__html_component__"},
        )
        self.assertEqual(set(VISUALIZATION_MARKER_BY_TYPE), set(VISUALIZATION_TOOL_NAME_BY_TYPE))

    def test_visualization_tool_name_for_type(self) -> None:
        self.assertEqual(visualization_tool_name_for_type("chart"), "create_chart")
        self.assertEqual(visualization_tool_name_for_type("datatable"), "create_datatable")
        self.assertEqual(visualization_tool_name_for_type("html_component"), "create_html_component")

    def test_visualization_tool_name_for_type_rejects_unknown(self) -> None:
        for tool_type in ("map", "", "create_chart"):
            with self.subTest(tool_type=tool_type):
                with self.assertRaises(ValueError) as raised:
                    visualization_tool_name_for_type(tool_type)
                self.assertIn(repr(tool_type), str(raised.exception))

    def test_is_visualization_tool_name(self) -> None:
        for name in VISUALIZATION_TOOL_NAMES:
            self.assertTrue(is_visualization_tool_name(name))
        for candidate in (None, "", "create_download_link", "chart"):
            with self.subTest(name=candidate):
                self.assertFalse(is_visualization_tool_name(candidate))


if __name__ == "__main__":
    unittest.main()
