import unittest
from unittest import mock

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError

from ragtime.solidworks.tool import (
    SOLIDWORKS_TOOL_INPUT_SCHEMA,
    build_solidworks_tool_description,
    build_solidworks_tool_name,
    create_solidworks_tool,
)


def _args_model(tool: StructuredTool) -> type[BaseModel]:
    """Return the Pydantic model behind a tool's arguments; the tool types it as a union."""
    model = tool.args_schema
    assert isinstance(model, type) and issubclass(model, BaseModel)
    return model


class SolidworksToolTests(unittest.IsolatedAsyncioTestCase):
    def test_name_uses_supplied_safe_name(self) -> None:
        self.assertEqual(build_solidworks_tool_name("design_shop_2024"), "solidworks_design_shop_2024")

    def test_description_matches_pinned_contract(self) -> None:
        description = build_solidworks_tool_description(
            {"name": "Design Shop", "description": "Engineering models", "connection_config": {"solidworks_version": 2024}}
        )
        self.assertEqual(
            description,
            "Read SolidWorks model data (features, sketches, dimensions, mates, components, properties, mass properties) "
            "from 'Design Shop' as JSON. Engineering models. Read-only. Pinned to SolidWorks 2024. "
            'Call with operation "guide" first, then "describe" for the operation list.',
        )

    def test_description_defaults_to_any_installed_version(self) -> None:
        for connection in ({}, {"solidworks_version": None}, None):
            with self.subTest(connection=connection):
                description = build_solidworks_tool_description({"name": "Shop", "connection_config": connection})
                self.assertIn("from 'Shop' as JSON. Read-only. Pinned to SolidWorks any installed version.", description)
                self.assertTrue(description.endswith('Call with operation "guide" first, then "describe" for the operation list.'))

    def test_description_cap_truncates_only_admin_description(self) -> None:
        config = {"name": "Shop", "description": "x" * 3000, "connection_config": {"solidworks_version": 2025}}
        description = build_solidworks_tool_description(config)
        without_admin = build_solidworks_tool_description({**config, "description": ""})
        prefix, suffix = without_admin.split(" Read-only.")
        self.assertLess(len(description), 1500)
        self.assertTrue(description.startswith(prefix + " xxx"))
        self.assertTrue(description.endswith(". Read-only." + suffix))
        self.assertEqual(config["description"], "x" * 3000)

    def test_input_schema_and_pydantic_arguments(self) -> None:
        self.assertEqual(SOLIDWORKS_TOOL_INPUT_SCHEMA["type"], "object")
        self.assertEqual(SOLIDWORKS_TOOL_INPUT_SCHEMA["required"], ["operation"])
        properties = SOLIDWORKS_TOOL_INPUT_SCHEMA["properties"]
        self.assertEqual({key: value["type"] for key, value in properties.items()}, {"operation": "string", "arguments": "object", "timeout": "integer"})
        tool = create_solidworks_tool({"name": "Shop"}, "shop")
        self.assertIsInstance(tool, StructuredTool)
        self.assertEqual(tool.name, "solidworks_shop")
        self.assertEqual(tool.description, build_solidworks_tool_description({"name": "Shop"}))
        model = _args_model(tool)
        self.assertEqual(model.model_json_schema()["required"], ["operation"])
        self.assertEqual(model(operation="guide").model_dump(), {"operation": "guide", "arguments": None, "timeout": None})
        for invalid in ({}, {"operation": 123}, {"operation": "guide", "arguments": []}, {"operation": "guide", "timeout": "invalid"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                model(**invalid)

    def test_timeout_is_bounded_between_one_and_3600_seconds(self) -> None:
        timeout_schema = SOLIDWORKS_TOOL_INPUT_SCHEMA["properties"]["timeout"]
        self.assertEqual((timeout_schema["minimum"], timeout_schema["maximum"]), (1, 3600))
        model = _args_model(create_solidworks_tool({"name": "Shop"}, "shop"))
        self.assertEqual(model(operation="guide", timeout=3600).model_dump()["timeout"], 3600)
        for timeout in (0, -1, 3601):
            with self.subTest(timeout=timeout), self.assertRaises(ValidationError):
                model(operation="guide", timeout=timeout)

    async def test_coroutine_forwards_arguments_and_preserves_response(self) -> None:
        config = {"id": "sw-1", "name": "Shop", "connection_config": {}, "timeout_max_seconds": 100}
        response = '{"ok":true,"data":"' + "x" * 20000 + '"}'
        with mock.patch("ragtime.solidworks.tool.solidworks_service.execute", new=mock.AsyncMock(return_value=response)) as execute:
            tool = create_solidworks_tool(config, "shop")
            result = await tool.ainvoke({"operation": "features.list", "arguments": {"document": "doc-1"}, "timeout": 12})
        self.assertEqual(result, response)
        execute.assert_awaited_once_with(config, "features.list", {"document": "doc-1"}, 12)

    async def test_coroutine_forwards_optional_defaults(self) -> None:
        config = {"name": "Shop"}
        with mock.patch("ragtime.solidworks.tool.solidworks_service.execute", new=mock.AsyncMock(return_value="{}")) as execute:
            tool = create_solidworks_tool(config, "shop")
            self.assertEqual(await tool.ainvoke({"operation": "guide"}), "{}")
        execute.assert_awaited_once_with(config, "guide", None, None)
