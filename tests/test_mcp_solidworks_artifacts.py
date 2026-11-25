import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mcp.server import Server
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, ImageContent, ResourceLink, TextContent
from starlette.requests import Request

from ragtime.mcp import artifact_content
from ragtime.mcp import server as mcp_server
from ragtime.mcp.tools import MCPToolAdapter, MCPToolDefinition
from ragtime.solidworks.artifacts import ArtifactNotFound

PRIMARY = "a" * 32
PREVIEW = "b" * 32
PNG = b"\x89PNG\r\n\x1a\nimage"


def result(operation="model.export", *, ok=True):
    preview = PRIMARY if operation == "model.preview" else PREVIEW
    artifact = {
        "id": PRIMARY,
        "file_name": "model.png" if operation == "model.preview" else "model.step",
        "format": "png" if operation == "model.preview" else "step",
        "size_bytes": 123,
        "page_url": f"/?artifact={PRIMARY}",
        "preview_image_url": f"/indexes/solidworks/artifacts/{preview}/content",
        "path": "/should/not/be/read",
    }
    return json.dumps({"ok": ok, "operation": operation, "result": {"artifact": artifact}})


async def invoke(text, *, tool_type="solidworks", deny=False):
    server = Server("artifact-test")
    adapter = MCPToolAdapter()
    definition = MCPToolDefinition(
        name="solidworks_test",
        description="test",
        input_schema={},
        tool_config={"tool_type": tool_type},
        execute_fn=mock.AsyncMock(return_value=text),
    )
    adapter._tool_definitions[definition.name] = definition
    mcp_server._register_handlers(server, adapter)

    async def authorize(_content, **kwargs):
        if deny and kwargs["direction"] == "outbound":
            raise ValueError("denied")

    with (
        mock.patch.object(adapter, "resolve_canonical_tool_id", new=mock.AsyncMock(return_value="id")),
        mock.patch.object(adapter, "execute_tool", new=mock.AsyncMock(return_value=text)),
        mock.patch("ragtime.mcp.server.authorize_external_content", side_effect=authorize),
        mock.patch.object(artifact_content.settings, "external_base_url", "https://ragtime.example"),
    ):
        response = await server.request_handlers[CallToolRequest](CallToolRequest(params=CallToolRequestParams(name=definition.name, arguments={})))
    assert isinstance(response.root, CallToolResult)
    return response.root


class SolidworksMcpArtifactTests(unittest.IsolatedAsyncioTestCase):
    async def test_preview_and_export_return_text_image_and_primary_link(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "content"
            image.write_bytes(PNG)
            for operation in ("model.preview", "model.export"):
                with self.subTest(operation=operation):
                    with mock.patch.object(
                        artifact_content, "load_artifact", new=mock.AsyncMock(return_value=({"format": "png", "expired": False}, image))
                    ) as load:
                        text = result(operation)
                        response = await invoke(text)
                    self.assertEqual([type(item) for item in response.content], [TextContent, ImageContent, ResourceLink])
                    text_content = response.content[0]
                    link = response.content[2]
                    assert isinstance(text_content, TextContent)
                    assert isinstance(link, ResourceLink)
                    assert link.description is not None
                    self.assertEqual(text_content.text, text)
                    self.assertEqual(str(link.uri), f"https://ragtime.example/?artifact={PRIMARY}")
                    self.assertEqual(link.mimeType, "text/html")
                    self.assertIsNone(link.size)
                    self.assertIn("png" if operation == "model.preview" else "step", link.description)
                    load.assert_awaited_once_with(PRIMARY if operation == "model.preview" else PREVIEW)

    async def test_unavailable_preview_keeps_text_and_link(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "content"
            for condition in ("missing", "expired", "oversized", "non-png"):
                with self.subTest(condition=condition):
                    image.write_bytes(PNG + (b"x" * artifact_content.MAX_IMAGE_BYTES if condition == "oversized" else b""))
                    meta = {"format": "step" if condition == "non-png" else "png", "expired": condition == "expired"}
                    loader = mock.AsyncMock(side_effect=ArtifactNotFound()) if condition == "missing" else mock.AsyncMock(return_value=(meta, image))
                    with mock.patch.object(artifact_content, "load_artifact", new=loader):
                        response = await invoke(result())
                    self.assertEqual([type(item) for item in response.content], [TextContent, ResourceLink])

    async def test_other_results_are_unchanged(self):
        for tool_type, ok in (("postgres", True), ("solidworks", False)):
            with self.subTest(tool_type=tool_type, ok=ok):
                with mock.patch.object(artifact_content, "load_artifact", new=mock.AsyncMock()) as loader:
                    text = result(ok=ok)
                    response = await invoke(text, tool_type=tool_type)
                self.assertEqual(len(response.content), 1)
                text_content = response.content[0]
                assert isinstance(text_content, TextContent)
                self.assertEqual(text_content.text, text)
                loader.assert_not_awaited()

    async def test_outbound_denial_does_not_load_image(self):
        with mock.patch.object(artifact_content, "load_artifact", new=mock.AsyncMock()) as loader:
            response = await invoke(result(), deny=True)
        self.assertTrue(response.isError)
        self.assertEqual(len(response.content), 1)
        loader.assert_not_awaited()

    async def test_untrusted_preview_path_is_not_used(self):
        payload = json.loads(result())
        payload["result"]["artifact"]["preview_image_url"] = "/tmp/secret"
        with mock.patch.object(artifact_content, "load_artifact", new=mock.AsyncMock()) as loader:
            response = await invoke(json.dumps(payload))
        self.assertEqual([type(item) for item in response.content], [TextContent, ResourceLink])
        loader.assert_not_awaited()

    async def test_missing_origin_omits_link(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "content"
            image.write_bytes(PNG)
            with (
                mock.patch.object(artifact_content.settings, "external_base_url", ""),
                mock.patch.object(artifact_content, "load_artifact", new=mock.AsyncMock(return_value=({"format": "png", "expired": False}, image))),
            ):
                content = await artifact_content.build_artifact_content(result())
        self.assertEqual([type(item) for item in content], [ImageContent])

    async def test_hostile_request_host_does_not_create_link(self):
        request = Request({"type": "http", "scheme": "https", "server": ("attacker.example", 443), "headers": [(b"host", b"attacker.example")]})
        server = Server("hostile-host")
        adapter = MCPToolAdapter()
        adapter._tool_definitions["solidworks_test"] = MCPToolDefinition(
            name="solidworks_test", description="test", input_schema={}, tool_config={"tool_type": "solidworks"}, execute_fn=mock.AsyncMock()
        )
        mcp_server._register_handlers(server, adapter)
        with (
            mock.patch.object(artifact_content.settings, "external_base_url", ""),
            mock.patch.object(adapter, "resolve_canonical_tool_id", new=mock.AsyncMock(return_value="id")),
            mock.patch.object(adapter, "execute_tool", new=mock.AsyncMock(return_value=result())),
            mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()),
            mock.patch.object(artifact_content, "load_artifact", new=mock.AsyncMock(side_effect=ArtifactNotFound())),
            mock.patch.object(type(server), "request_context", new_callable=mock.PropertyMock, return_value=mock.Mock(request=request)),
        ):
            response = await server.request_handlers[CallToolRequest](CallToolRequest(params=CallToolRequestParams(name="solidworks_test", arguments={})))
        assert isinstance(response.root, CallToolResult)
        self.assertEqual([type(item) for item in response.root.content], [TextContent])

    async def test_invalid_configured_origins_omit_link(self):
        for origin in ("https://example.com/path", "https://user@example.com", "https://example.com?query=1", "https://example.com:invalid"):
            with self.subTest(origin=origin), mock.patch.object(artifact_content.settings, "external_base_url", origin):
                self.assertIsNone(artifact_content._public_origin())

    async def test_builder_error_keeps_text(self):
        with mock.patch("ragtime.mcp.server.build_artifact_content", new=mock.AsyncMock(side_effect=RuntimeError("private content"))):
            text = result()
            response = await invoke(text)
        self.assertEqual(len(response.content), 1)
        text_content = response.content[0]
        assert isinstance(text_content, TextContent)
        self.assertEqual(text_content.text, text)
