import asyncio
import io
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.messages import AIMessage, ToolMessage
from PIL import Image

from ragtime.rag import components
from ragtime.rag.components import RAGComponents, settings
from ragtime.solidworks import artifacts


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "index_data_path", str(tmp_path))
    monkeypatch.setattr(artifacts, "ARTIFACT_DIR", tmp_path / "_solidworks_artifacts")
    component = RAGComponents.__new__(RAGComponents)
    monkeypatch.setattr(component, "_downsample_image_data_url", lambda value: value)
    return component, tmp_path


def artifact(root, artifact_id="a" * 32, *, expired=False, format="png"):
    directory = root / "_solidworks_artifacts" / artifact_id
    directory.mkdir(parents=True)
    meta = {
        "id": artifact_id,
        "kind": "preview",
        "format": format,
        "file_name": "part.png",
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=-1 if expired else 1)).isoformat(),
    }
    (directory / "meta.json").write_text(json.dumps(meta))
    image = io.BytesIO()
    Image.new("RGB", (1, 1), "red").save(image, format="PNG")
    (directory / "content").write_bytes(image.getvalue())
    return artifact_id


def step(name, call_id, payload):
    action = ToolAgentAction(
        tool=name,
        tool_input={},
        log="",
        message_log=[AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": call_id}])],
        tool_call_id=call_id,
    )
    return action, json.dumps(payload)


def payload(artifact_id, operation="model.preview", mode="primary"):
    preview = {"id": artifact_id, "kind": "preview"}
    primary = preview if mode == "primary" else {"kind": "model"}
    if mode == "url":
        primary["preview_image_url"] = f"/indexes/solidworks/artifacts/{artifact_id}/content"
    return {"ok": True, "operation": operation, "result": {"artifact": primary, "artifacts": [preview] if mode == "list" else []}}


def images(component, steps, **kwargs):
    messages = component._format_intermediate_steps_for_agent(steps, **kwargs)
    return {
        msg.tool_call_id: msg.content
        for msg in messages
        if isinstance(msg, ToolMessage)
        and isinstance(msg.content, list)
        and any(isinstance(part, dict) and part.get("type") == "image_url" for part in msg.content)
    }


@pytest.mark.parametrize("mode,operation", [("primary", "model.preview"), ("list", "model.export"), ("url", "model.export")])
def test_preview_attached(context, mode, operation):
    component, root = context
    artifact_id = artifact(root)
    result = images(component, [step("solidworks_machine", "one", payload(artifact_id, operation, mode))])
    assert list(result) == ["one"]
    text_part, image_part = result["one"][1:3]
    assert isinstance(text_part, dict) and "latest SolidWorks preview (part.png)" in text_part["text"]
    assert isinstance(image_part, dict) and image_part["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("preview_last", [True, False])
def test_latest_image_only(context, preview_last):
    component, root = context
    artifact_id = artifact(root)
    screenshot = root / "screenshot.png"
    screenshot.write_bytes(b"screenshot")
    steps = [step("capture_userspace_screenshot", "screen", {"screenshot_path": str(screenshot)}), step("solidworks_machine", "preview", payload(artifact_id))]
    if not preview_last:
        steps.reverse()
    assert list(images(component, steps)) == (["preview"] if preview_last else ["screen"])
    assert not images(component, steps, include_latest_screenshot_image=False)


def test_invalid_latest_step_keeps_previous_image(context):
    component, root = context
    artifact_id = artifact(root)
    assert list(images(component, [step("solidworks_machine", "valid", payload(artifact_id)), step("solidworks_machine", "missing", payload("b" * 32))])) == [
        "valid"
    ]


@pytest.mark.parametrize("contents", [b"not png", b"\x89PNG\r\n\x1a\ncorrupt"])
def test_invalid_png_is_ignored(context, contents):
    component, root = context
    artifact_id = artifact(root)
    (root / "_solidworks_artifacts" / artifact_id / "content").write_bytes(contents)
    assert not images(component, [step("solidworks_machine", "one", payload(artifact_id))])


@pytest.mark.parametrize(
    "model,provider,answer,expected",
    [("vision", "openai", True, True), ("text", "openai", False, False), ("vision", "openai_compatible", True, False), (None, "openai", False, True)],
)
def test_request_image_capability(context, monkeypatch, model, provider, answer, expected):
    component, _ = context
    calls = []

    async def supports(model_id):
        calls.append(model_id)
        return answer

    monkeypatch.setattr(components, "supports_image_input", supports)
    assert asyncio.run(component._request_supports_reference_images(model, provider)) is expected
    assert calls == ([model] if model and provider != "openai_compatible" else [])


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("tool", ["solidworks_machine", "capture_userspace_screenshot"])
def test_runtime_formatter_respects_capability(context, monkeypatch, enabled, tool):
    component, root = context
    artifact_id = artifact(root)
    screenshot = root / "screenshot.png"
    screenshot.write_bytes(b"screenshot")
    observation = payload(artifact_id) if tool.startswith("solidworks_") else {"screenshot_path": str(screenshot)}
    captured = {}
    component._app_settings = None
    monkeypatch.setattr(components, "wrap_tools_with_content_protection", lambda tools, *args, **kwargs: tools)
    monkeypatch.setattr(component, "_content_protection_tool_ids", lambda: {})
    monkeypatch.setattr(component, "_create_thinking_aware_agent", lambda *args, message_formatter: captured.setdefault("formatter", message_formatter))
    monkeypatch.setattr(components, "AgentExecutor", lambda **kwargs: kwargs)
    component._build_runtime_executor([object()], "system", llm=object(), include_reference_images=enabled)
    messages = captured["formatter"]([step(tool, "one", observation)])
    result = next(message for message in messages if isinstance(message, ToolMessage))
    assert isinstance(result.content, list) is enabled
    assert json.dumps(observation) in str(result.content)


def test_list_content_and_multi_tool_group_preserved(context, monkeypatch):
    component, _ = context
    assistant = AIMessage(content="", tool_calls=[{"name": "first", "args": {}, "id": "a"}, {"name": "second", "args": {}, "id": "b"}])
    content: list[str | dict[str, Any]] = [{"type": "text", "text": "structured tool content"}]
    original = [assistant, ToolMessage(content=content, tool_call_id="a"), ToolMessage(content="other", tool_call_id="b")]
    monkeypatch.setattr(components, "format_to_tool_messages", lambda steps: original)
    output = component._format_intermediate_steps_for_agent([(object(), "one"), (object(), "two")])
    assert output == original
    assert output[1].content == content


@pytest.mark.parametrize("state", ["expired", "missing", "non-png", "malicious", "other-tool", "failed", "wrong-operation"])
def test_invalid_preview_is_ignored(context, state):
    component, root = context
    artifact_id = "a" * 32
    if state not in {"missing", "malicious"}:
        artifact(root, artifact_id, expired=state == "expired", format="step" if state == "non-png" else "png")
    data = payload(artifact_id)
    if state == "malicious":
        data["result"]["artifact"]["id"] = "../../outside"
        data["result"]["artifact"]["path"] = str(root / "outside.png")
        (root / "outside.png").write_bytes(b"secret")
    if state == "failed":
        data["ok"] = False
    if state == "wrong-operation":
        data["operation"] = "document.open"
    assert not images(component, [step("other_tool" if state == "other-tool" else "solidworks_machine", "one", data)])
