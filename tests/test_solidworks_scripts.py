import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server import Server
from mcp.types import ListToolsRequest, ListToolsResult

from ragtime.content_protection.models import ContentProtectionError
from ragtime.indexer.models import SolidworksConnectionConfig
from ragtime.mcp.server import _register_handlers
from ragtime.mcp.tools import MCPToolAdapter, MCPToolDefinition
from ragtime.solidworks.service import SolidworksService, _invalid_script_field
from ragtime.solidworks.tool import build_solidworks_tool_description


def test_scripts_off_refusal_names_the_setting():
    transport = AsyncMock()
    denied = json.loads(asyncio.run(SolidworksService(transport_factory=transport).execute({}, "script.run", {})))
    error = denied["error"]
    assert error["code"] == "scripts_not_allowed"
    assert error["message"] == "Scripts are off for this tool. An admin can turn on Allow scripts in the tool's SolidWorks settings."
    assert error["details"]["setting"] == "Allow scripts"
    assert error["details"]["where"] == "Tool settings > SolidWorks"
    assert error["details"]["still_available"] == [
        "Read models (document.open, mates.list, components.list)",
        "Export the result (model.export)",
    ]
    transport.assert_not_called()


def test_gate_and_validation_precede_review_and_transport():
    transport = AsyncMock()
    service = SolidworksService(transport_factory=transport)
    with patch("ragtime.solidworks.service.authorize_content", new_callable=AsyncMock) as review:
        denied = json.loads(asyncio.run(service.execute({}, "script.run", {})))
        assert denied["error"]["code"] == "scripts_not_allowed"
        invalid = json.loads(asyncio.run(service.execute({"connection_config": {"allow_scripts": True}}, "script.run", {})))
        assert invalid["error"]["details"]["field"] == "source"
        review.assert_not_awaited()
        transport.assert_not_called()


@pytest.mark.parametrize("instance", ["other", "i0", "i1x", "a", ""])
def test_invalid_instance_precedes_review_and_transport(instance):
    transport = AsyncMock()
    service = SolidworksService(transport_factory=transport)
    with patch("ragtime.solidworks.service.authorize_content", new_callable=AsyncMock) as review:
        result = json.loads(
            asyncio.run(
                service.execute(
                    {"connection_config": {"allow_scripts": True}},
                    "script.run",
                    {"source": "code", "purpose": "build", "instance": instance},
                )
            )
        )
    assert result["error"]["code"] == "invalid_arguments"
    assert result["error"]["details"]["field"] == "instance"
    review.assert_not_awaited()
    transport.assert_not_called()


@pytest.mark.parametrize("instance", ["i1", "i123", "a1", "a123"])
def test_valid_instance(instance):
    assert _invalid_script_field({"source": "code", "purpose": "build", "instance": instance}, None) is None


def test_review_arguments_and_failure_propagation():
    service = SolidworksService(transport_factory=lambda _: pytest.fail("transport reached"))
    failure = ContentProtectionError("content_denied", "request")
    config = {"id": "tool", "connection_config": {"allow_scripts": True}}
    arguments = {"source": "code", "purpose": "build", "arguments": {"width": 1}}
    with (
        patch("ragtime.solidworks.service.current_context", return_value="principal"),
        patch("ragtime.solidworks.service.authorize_content", new_callable=AsyncMock, side_effect=failure) as review,
    ):
        with pytest.raises(ContentProtectionError) as raised:
            asyncio.run(service.execute(config, "script.run", arguments))
        assert raised.value is failure
        review.assert_awaited_once_with(
            "code",
            direction="inbound",
            context="principal",
            tool_id="tool",
            operation="solidworks.script.run",
            supporting_context={"purpose": "build", "arguments": {"width": 1}},
        )


@pytest.mark.parametrize(
    ("arguments", "timeout", "field"),
    [
        ({"source": "", "purpose": "ok"}, None, "source"),
        ({"source": "é" * 8193, "purpose": "ok"}, None, "source"),
        ({"source": "x", "purpose": "a" * 501}, None, "purpose"),
        ({"source": "x", "purpose": "a\n"}, None, "purpose"),
        ({"source": "x", "purpose": "a", "arguments": {"x": "a" * 4096}}, None, "arguments"),
        ({"source": "x", "purpose": "a"}, 601, "timeout"),
        ({"source": "x" * 16384, "purpose": "a" * 500, "arguments": {"x": "a" * 4088}}, 600, None),
    ],
)
def test_script_limits(arguments, timeout, field):
    assert _invalid_script_field(arguments, timeout) == field


def test_context_forwards_stored_scripts_flag():
    assert SolidworksService._context({"allow_scripts": True}, False, 120)["allow_scripts"] is True
    assert SolidworksService._context({}, False, 120)["allow_scripts"] is False


def test_encoded_request_limit_refuses_before_review_or_transport():
    transport = AsyncMock()
    service = SolidworksService(transport_factory=transport)
    with patch("ragtime.solidworks.service.authorize_content", new_callable=AsyncMock) as review:
        result = json.loads(
            asyncio.run(
                service.execute(
                    {"connection_config": {"allow_scripts": True}},
                    "script.run",
                    {"source": '"' * 16000, "purpose": "build"},
                )
            )
        )
    assert result["error"]["code"] == "invalid_arguments"
    assert result["error"]["details"]["field"] == "source"
    review.assert_not_awaited()
    transport.assert_not_called()


def test_script_config_default_and_round_trip():
    connection = {"host": "example", "user": "alice"}
    assert SolidworksConnectionConfig.model_validate(connection).allow_scripts is False
    config = SolidworksConnectionConfig.model_validate({**connection, "allow_scripts": True})
    assert SolidworksConnectionConfig.model_validate(config.model_dump()).allow_scripts is True
    with pytest.raises(ValueError):
        SolidworksConnectionConfig.model_validate({**connection, "allow_scripts": "invalid"})


def test_description_does_not_claim_read_only_with_scripts_enabled():
    description = build_solidworks_tool_description({"connection_config": {"allow_scripts": True}})
    assert "Scripts: on." in description
    assert "Read-only." not in description


@pytest.mark.parametrize(("cap", "timeout", "expected"), [(None, None, 100), (900, None, 120), (900, 700, None), (300, 250, 250)])
def test_disabled_protection_forwards_script_with_context_and_timeout(cap, timeout, expected):
    requests = []

    async def call(request, *, timeout_seconds):
        requests.append((request, timeout_seconds))
        result = {"host_version": "1", "operations": [{"name": "script.run", "access": "script"}]} if request["operation"] == "describe" else {"result": "ok"}
        return {"id": request["id"], "operation": request["operation"], "ok": True, "result": result, "host": {"host_version": "1"}}

    transport = MagicMock()
    transport.call = AsyncMock(side_effect=call)
    service = SolidworksService(transport_factory=lambda _: transport, clock=lambda: 0)
    config = {"id": "tool", "connection_config": {"allow_scripts": True}}
    if cap is not None:
        config["timeout_max_seconds"] = cap
    with (
        patch("ragtime.content_protection.service.load_config", new_callable=AsyncMock, return_value=SimpleNamespace(enabled=False)),
        patch("ragtime.solidworks.pdm_link.resolve_pdm_link", new_callable=AsyncMock, return_value=None),
    ):
        result = json.loads(asyncio.run(service.execute(config, "script.run", {"source": "x" * 16384, "purpose": "build"}, timeout)))
    if expected is None:
        assert result["error"]["details"]["field"] == "timeout"
        assert requests == []
        return
    assert result["ok"]
    assert [request["operation"] for request, _ in requests] == ["describe", "script.run"]
    assert requests[-1][0]["context"]["allow_scripts"] is True
    assert requests[-1][1] == expected


@pytest.mark.parametrize(("write", "scripts"), [(False, False), (False, True), (True, False), (True, True)])
def test_mcp_annotations(write, scripts):
    adapter = MCPToolAdapter()
    definition = MCPToolDefinition(
        name="solidworks",
        description="test",
        input_schema={"type": "object", "properties": {}},
        tool_config={
            "tool_type": "solidworks",
            "allow_write": write,
            "connection_config": {"allow_scripts": scripts, "pdm_approved_folders": [r"\Design"]},
        },
        execute_fn=AsyncMock(),
    )
    server = Server("test")
    _register_handlers(server, adapter)

    async def list_tools():
        return await server.request_handlers[ListToolsRequest](ListToolsRequest())

    with patch.object(adapter, "get_available_tools", new=AsyncMock(return_value=[definition])):
        response = asyncio.run(list_tools())
    assert isinstance(response.root, ListToolsResult)
    annotation = response.root.tools[0].annotations
    assert annotation is not None
    assert annotation.readOnlyHint is not (write or scripts)
    assert annotation.destructiveHint is (write or scripts)
