from __future__ import annotations

import json
import unittest
from typing import Any

from pydantic import ValidationError

from ragtime.core.visualization_tools import HTML_COMPONENT_TOOL_NAME
from ragtime.tools.html_component import (
    CHAT_HTML_COMPONENT_DESCRIPTION_SUFFIX,
    HTML_COMPONENT_EXPECTED_INPUT_SHAPE,
    HTML_COMPONENT_MAX_DATA_CHARS,
    HTML_COMPONENT_MAX_HEIGHT,
    HTML_COMPONENT_MAX_HTML_CHARS,
    HTML_COMPONENT_MIN_HEIGHT,
    USERSPACE_HTML_COMPONENT_DESCRIPTION_SUFFIX,
    CreateHtmlComponentInput,
    create_html_component,
    create_html_component_tool,
)

FRAGMENT = '<div id="root"></div><script>window.ragtime.onData(function (d) {});</script>'
FULL_DOCUMENT = "<!DOCTYPE html><HTML lang=\"en\"><head><title>t</title></head><body><div id='root'></div></body></HTML>"
SOURCE_DATA: dict[str, Any] = {"columns": ["lat", "lng", "shipments"], "rows": [[31.9, -99.9, 412], [40.7, -74.0, 288]]}
DATA_CONNECTION: dict[str, Any] = {
    "component_kind": "tool_config",
    "component_id": "tool-config-1",
    "request": {"query": "SELECT lat, lng, COUNT(*) AS shipments FROM shipments GROUP BY 1, 2"},
}
ENVELOPE_KEYS = ["__html_component__", "title", "html", "data", "description", "height", "data_connection"]


async def _create(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"title": "Component", "html": FRAGMENT}
    kwargs.update(overrides)
    output = await create_html_component(**kwargs)
    assert not output.startswith("Error:"), output
    return json.loads(output)


class HtmlDocumentWrappingTests(unittest.IsolatedAsyncioTestCase):
    async def test_fragment_is_wrapped_into_full_document(self) -> None:
        payload = await _create()
        html = payload["html"]
        self.assertTrue(html.startswith("<!doctype html><html><head>"))
        self.assertIn('<meta charset="utf-8">', html)
        self.assertIn('<meta name="viewport" content="width=device-width, initial-scale=1">', html)
        self.assertIn(f"<body>{FRAGMENT}</body></html>", html)

    async def test_full_document_passes_through_unchanged(self) -> None:
        payload = await _create(html=FULL_DOCUMENT)
        self.assertEqual(payload["html"], FULL_DOCUMENT)

    async def test_html_prefixed_tag_name_is_not_mistaken_for_document(self) -> None:
        fragment = "<htmlx-widget></htmlx-widget>"
        payload = await _create(html=fragment)
        self.assertIn(f"<body>{fragment}</body>", payload["html"])


class HtmlComponentDataCanonicalizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_data_array_rows_become_object_rows_with_row_count(self) -> None:
        payload = await _create(source_data=SOURCE_DATA)
        self.assertEqual(
            payload["data"],
            {
                "columns": ["lat", "lng", "shipments"],
                "rows": [
                    {"lat": 31.9, "lng": -99.9, "shipments": 412},
                    {"lat": 40.7, "lng": -74.0, "shipments": 288},
                ],
                "row_count": 2,
            },
        )

    async def test_source_data_object_rows_are_preserved(self) -> None:
        payload = await _create(source_data={"columns": ["a"], "rows": [{"a": 1}]})
        self.assertEqual(payload["data"], {"columns": ["a"], "rows": [{"a": 1}], "row_count": 1})

    async def test_source_data_takes_precedence_over_data(self) -> None:
        payload = await _create(source_data={"columns": ["a"], "rows": [[1]]}, data={"ignored": True})
        self.assertEqual(payload["data"]["columns"], ["a"])
        self.assertNotIn("ignored", payload["data"])

    async def test_tabular_data_dict_is_canonicalized(self) -> None:
        payload = await _create(data={"columns": ["state", "count"], "rows": [["TX", 5], ["NY", 3]]})
        self.assertEqual(
            payload["data"],
            {"columns": ["state", "count"], "rows": [{"state": "TX", "count": 5}, {"state": "NY", "count": 3}], "row_count": 2},
        )

    async def test_non_tabular_dict_data_passes_through(self) -> None:
        data = {"thresholds": {"warn": 10, "crit": 20}, "labels": ["a", "b"]}
        payload = await _create(data=data)
        self.assertEqual(payload["data"], data)

    async def test_non_tabular_list_data_passes_through(self) -> None:
        data = [{"id": 1}, {"id": 2}]
        payload = await _create(data=data)
        self.assertEqual(payload["data"], data)

    async def test_missing_data_is_null(self) -> None:
        payload = await _create()
        self.assertIsNone(payload["data"])

    async def test_invalid_source_data_returns_error_string(self) -> None:
        output = await create_html_component(title="Bad", html=FRAGMENT, source_data={"columns": [], "rows": [[1]]})
        self.assertTrue(output.startswith("Error: Cannot create component 'Bad' - "))
        self.assertIn("source_data.columns", output)


class HtmlComponentDataConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_data_connection_with_tabular_data_is_normalized(self) -> None:
        payload = await _create(source_data=SOURCE_DATA, data_connection={**DATA_CONNECTION, "source_tool": "Postgres"})
        self.assertEqual(payload["data_connection"]["component_kind"], "tool_config")
        self.assertEqual(payload["data_connection"]["component_id"], "tool-config-1")
        self.assertEqual(payload["data_connection"]["request"], DATA_CONNECTION["request"])
        self.assertEqual(payload["data_connection"]["component_name"], "Postgres")

    async def test_data_connection_without_tabular_data_returns_error(self) -> None:
        non_tabular_data: tuple[dict[str, Any] | list[Any] | None, ...] = (None, {"static": True}, [1, 2, 3])
        for data in non_tabular_data:
            with self.subTest(data=data):
                output = await create_html_component(title="Map", html=FRAGMENT, data=data, data_connection=DATA_CONNECTION)
                self.assertEqual(
                    output,
                    "Error: Cannot create component 'Map' - data_connection requires tabular source_data (columns/rows) so refresh can replace window.ragtime.data",
                )

    async def test_invalid_data_connection_returns_error(self) -> None:
        output = await create_html_component(title="Map", html=FRAGMENT, source_data=SOURCE_DATA, data_connection={"component_kind": "web_research"})
        self.assertTrue(output.startswith("Error: Cannot create component 'Map' - "))
        self.assertIn("data_connection.component_id is required", output)
        self.assertIn("data_connection.request is required", output)

    async def test_missing_data_connection_is_null(self) -> None:
        payload = await _create(source_data=SOURCE_DATA)
        self.assertIsNone(payload["data_connection"])


class HtmlComponentLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_html_size_limit(self) -> None:
        output = await create_html_component(title="Big", html="x" * (HTML_COMPONENT_MAX_HTML_CHARS + 1))
        self.assertEqual(output, "Error: Cannot create component 'Big' - html exceeds 200,000 characters")
        with self.assertRaises(ValidationError):
            CreateHtmlComponentInput.model_validate({"title": "Big", "html": "x" * (HTML_COMPONENT_MAX_HTML_CHARS + 1)})

    async def test_html_at_limit_is_accepted(self) -> None:
        payload = await _create(html="x" * HTML_COMPONENT_MAX_HTML_CHARS)
        self.assertIn("x" * HTML_COMPONENT_MAX_HTML_CHARS, payload["html"])

    async def test_data_size_limit(self) -> None:
        output = await create_html_component(title="Big", html=FRAGMENT, data={"blob": "x" * HTML_COMPONENT_MAX_DATA_CHARS})
        self.assertEqual(output, "Error: Cannot create component 'Big' - data exceeds 1,000,000 characters")

    async def test_tabular_data_size_limit_is_measured_after_canonicalization(self) -> None:
        source = {"columns": ["c"], "rows": [["x" * 1000]] * 1000}
        output = await create_html_component(title="Big", html=FRAGMENT, source_data=source)
        self.assertEqual(output, "Error: Cannot create component 'Big' - data exceeds 1,000,000 characters")

    async def test_empty_html_is_rejected(self) -> None:
        output = await create_html_component(title="Empty", html="   ")
        self.assertEqual(output, "Error: Cannot create component 'Empty' - html must be a non-empty string")
        with self.assertRaises(ValidationError):
            CreateHtmlComponentInput.model_validate({"title": "Empty", "html": ""})

    async def test_height_bounds(self) -> None:
        for height in (HTML_COMPONENT_MIN_HEIGHT - 1, HTML_COMPONENT_MAX_HEIGHT + 1):
            with self.subTest(height=height):
                with self.assertRaises(ValidationError):
                    CreateHtmlComponentInput.model_validate({"title": "H", "html": FRAGMENT, "height": height})
                output = await create_html_component(title="H", html=FRAGMENT, height=height)
                self.assertEqual(output, "Error: Cannot create component 'H' - height must be between 200 and 1200")
        for height in (HTML_COMPONENT_MIN_HEIGHT, HTML_COMPONENT_MAX_HEIGHT):
            with self.subTest(height=height):
                self.assertEqual(CreateHtmlComponentInput.model_validate({"title": "H", "html": FRAGMENT, "height": height}).height, height)
                payload = await _create(height=height)
                self.assertEqual(payload["height"], height)

    async def test_height_defaults_to_null(self) -> None:
        payload = await _create()
        self.assertIsNone(payload["height"])


class HtmlComponentSchemaTests(unittest.TestCase):
    def test_http_asset_urls_are_rejected(self) -> None:
        for markup in (
            '<script src="http://cdn.example.com/lib.js"></script>',
            "<link rel='stylesheet' href='http://cdn.example.com/lib.css'>",
            '<img SRC = "http://example.com/a.png">',
        ):
            with self.subTest(markup=markup):
                with self.assertRaises(ValidationError) as raised:
                    CreateHtmlComponentInput.model_validate({"title": "T", "html": markup})
                self.assertIn("http://", str(raised.exception))

    def test_https_and_inline_http_text_are_accepted(self) -> None:
        markup = '<script src="https://cdn.example.com/lib.js"></script><p>See http://example.com for docs</p>'
        self.assertEqual(CreateHtmlComponentInput.model_validate({"title": "T", "html": markup}).html, markup)

    def test_aliases_map_to_html(self) -> None:
        for alias in ("code", "markup", "content"):
            with self.subTest(alias=alias):
                parsed = CreateHtmlComponentInput.model_validate({"title": "T", alias: FRAGMENT})
                self.assertEqual(parsed.html, FRAGMENT)

    def test_explicit_html_wins_over_alias(self) -> None:
        parsed = CreateHtmlComponentInput.model_validate({"title": "T", "html": FRAGMENT, "code": "<b>other</b>"})
        self.assertEqual(parsed.html, FRAGMENT)

    def test_top_level_columns_and_rows_become_source_data(self) -> None:
        parsed = CreateHtmlComponentInput.model_validate({"title": "T", "html": FRAGMENT, "columns": ["a"], "rows": [[1]]})
        self.assertEqual(parsed.source_data, {"columns": ["a"], "rows": [[1]]})

    def test_schema_exposes_optional_data_connection(self) -> None:
        schema = CreateHtmlComponentInput.model_json_schema()
        self.assertEqual(set(schema["required"]), {"title", "html"})
        for key in ("data", "source_data", "description", "height", "data_connection"):
            self.assertIn(key, schema["properties"])


class HtmlComponentToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_name_and_description_constants(self) -> None:
        self.assertEqual(create_html_component_tool.name, HTML_COMPONENT_TOOL_NAME)
        self.assertIs(create_html_component_tool.args_schema, CreateHtmlComponentInput)
        description = create_html_component_tool.description
        self.assertIn("ONLY when create_chart or create_datatable cannot express the visual", description)
        self.assertIn("window.ragtime.onData", description)
        self.assertIn("leaflet", description.lower())
        self.assertIn("source_data", description)
        self.assertIn("data_connection", description)
        self.assertIn("200 KB", description)
        self.assertIn("1 MB", description)
        self.assertIn("200-1200", description)
        self.assertIn("source_data", CHAT_HTML_COMPONENT_DESCRIPTION_SUFFIX)
        self.assertIn("omit `data_connection`", CHAT_HTML_COMPONENT_DESCRIPTION_SUFFIX)
        self.assertIn("Prefer create_chart or create_datatable", CHAT_HTML_COMPONENT_DESCRIPTION_SUFFIX)
        self.assertIn("not available for persistent dashboards", USERSPACE_HTML_COMPONENT_DESCRIPTION_SUFFIX)

    async def test_envelope_keys_and_order(self) -> None:
        output = await create_html_component_tool.ainvoke(
            {
                "title": "Shipments by origin",
                "html": FRAGMENT,
                "source_data": SOURCE_DATA,
                "description": "Shipments by origin state",
                "height": 480,
                "data_connection": DATA_CONNECTION,
            }
        )
        payload = json.loads(output)
        self.assertEqual(list(payload.keys()), ENVELOPE_KEYS)
        self.assertIs(payload["__html_component__"], True)
        self.assertEqual(payload["title"], "Shipments by origin")
        self.assertEqual(payload["description"], "Shipments by origin state")
        self.assertEqual(payload["height"], 480)
        self.assertEqual(payload["data"]["row_count"], 2)
        self.assertEqual(payload["data_connection"]["component_id"], "tool-config-1")

    async def test_alias_code_reaches_coroutine_through_tool(self) -> None:
        payload = json.loads(await create_html_component_tool.ainvoke({"title": "T", "code": FRAGMENT}))
        self.assertIn(f"<body>{FRAGMENT}</body>", payload["html"])

    async def test_top_level_columns_rows_reach_coroutine_through_tool(self) -> None:
        payload = json.loads(await create_html_component_tool.ainvoke({"title": "T", "html": FRAGMENT, "columns": ["a"], "rows": [[1]]}))
        self.assertEqual(payload["data"], {"columns": ["a"], "rows": [{"a": 1}], "row_count": 1})

    async def test_http_src_validation_error_feedback_names_tool_and_shape(self) -> None:
        feedback = await create_html_component_tool.ainvoke({"title": "T", "html": '<script src="http://cdn.example.com/lib.js"></script>'})
        self.assertIsInstance(feedback, str)
        self.assertIn(f"Tool input validation error for {HTML_COMPONENT_TOOL_NAME}", feedback)
        self.assertIn("http://", feedback)
        self.assertIn("Expected input shape:", feedback)
        self.assertIn(HTML_COMPONENT_EXPECTED_INPUT_SHAPE, feedback)
        self.assertIn("How to fix:", feedback)

    async def test_height_validation_error_feedback(self) -> None:
        feedback = await create_html_component_tool.ainvoke({"title": "T", "html": FRAGMENT, "height": 50})
        self.assertIn(f"Tool input validation error for {HTML_COMPONENT_TOOL_NAME}", feedback)
        self.assertIn("height", feedback)


if __name__ == "__main__":
    unittest.main()
