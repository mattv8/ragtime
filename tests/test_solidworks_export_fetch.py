import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from ragtime.core.ssh import SSHResult
from ragtime.indexer.background_tasks import _truncate_tool_output
from ragtime.rag.components import compact_stream_display_output
from ragtime.solidworks.artifacts import resolve_artifact_content
from ragtime.solidworks.host_activity import HostActivity, HostBusyError, host_key
from ragtime.solidworks.service import SolidworksService, _artifact_remote_path

ROOT = r"C:\Users\alice\AppData\Local\RagtimeSolidWorks\work"
DATA = b"example"


def file_info(role, name):
    return {
        "role": role,
        "path": ROOT + "\\" + name,
        "file_name": name,
        "format": "png" if role == "preview" else "step",
        "size_bytes": len(DATA),
        "sha256": hashlib.sha256(DATA).hexdigest(),
    }


class Transport:
    def __init__(self):
        self.deleted = []
        self.downloaded = []

    async def call(self, request: dict, *, timeout_seconds: float) -> dict:
        raise AssertionError("call is not used")

    async def run_powershell(self, script: str, *, timeout_seconds: float) -> SSHResult:
        raise AssertionError("run_powershell is not used")

    async def upload_file(self, local_path: Path, remote_path: str, *, timeout_seconds: float) -> None:
        raise AssertionError("upload_file is not used")

    async def download_file(self, remote_path: str, local_path: Path, *, max_bytes: int, timeout_seconds: float) -> int:
        assert timeout_seconds > 0
        self.downloaded.append(remote_path)
        local_path.write_bytes(DATA)
        return len(DATA)

    async def remove_file(self, remote_path: str, *, timeout_seconds: float) -> None:
        self.deleted.append(remote_path)


async def store(local, **kwargs):
    assert local.read_bytes() == DATA
    ident = "a" * 32 if kwargs["kind"] == "preview" else "b" * 32
    return {
        "id": ident,
        "kind": kwargs["kind"],
        "format": kwargs["format"],
        "file_name": kwargs["file_name"],
        "size_bytes": len(DATA),
        "sha256": kwargs["sha256"],
        "expires_at": "2026-12-01",
        "content_url": "/content/" + ident,
        "preview_image_url": "/content/" + (kwargs["preview_artifact_id"] or ident),
        "page_url": "/?artifact=" + ident,
    }


async def _export_and_preview_link_and_bound():
    transport = Transport()
    response = {
        "protocol": 1,
        "operation": "model.export",
        "ok": True,
        "host": {},
        "result": {"files": [file_info("model", "x.step"), file_info("preview", "x.png")], "document": {"title": "x" * 1000}, "warnings": ["w" * 500] * 3},
    }
    with mock.patch("ragtime.solidworks.service.store_artifact", side_effect=store) as stored:
        output = await SolidworksService()._fetch_artifacts(response, transport, {"id": "tool"}, time.monotonic() + 10)
    parsed = json.loads(output)
    assert parsed["result"]["artifact"]["kind"] == "model"
    assert stored.call_args_list[1].kwargs["preview_artifact_id"] == "a" * 32
    assert len(output) <= 1800
    assert ROOT not in output and "C:/Users" not in output
    assert len(transport.deleted) == 2
    assert transport.downloaded == [
        "C:/Users/alice/AppData/Local/RagtimeSolidWorks/work/x.step",
        "C:/Users/alice/AppData/Local/RagtimeSolidWorks/work/x.png",
    ]
    assert transport.deleted == transport.downloaded
    assert json.loads(compact_stream_display_output("solidworks_test", output)) == parsed
    assert json.loads(_truncate_tool_output(output)) == parsed


async def _preview_and_failure_cleanup():
    transport = Transport()
    item = file_info("preview", "x.png")
    response = {"protocol": 1, "operation": "model.preview", "ok": True, "result": {"files": [item]}}
    with mock.patch("ragtime.solidworks.service.store_artifact", side_effect=store):
        output = json.loads(await SolidworksService()._fetch_artifacts(response, transport, {}, time.monotonic() + 10))
        assert output["result"]["artifact"]["kind"] == "preview"
        item["sha256"] = "0" * 64
        output = json.loads(await SolidworksService()._fetch_artifacts(response, transport, {}, time.monotonic() + 10))
        assert output["error"]["code"] == "artifact_transfer_failed"
        item["size_bytes"] = 100 * 1024 * 1024 + 1
        output = json.loads(await SolidworksService()._fetch_artifacts(response, transport, {}, time.monotonic() + 10))
        assert output["error"]["code"] == "artifact_too_large"
        item["size_bytes"] = len(DATA)
        with mock.patch.object(transport, "download_file", side_effect=OSError("private path")):
            output = json.loads(await SolidworksService()._fetch_artifacts(response, transport, {}, time.monotonic() + 10))
            assert output["error"]["code"] == "artifact_transfer_failed"
            assert "private path" not in json.dumps(output)
    assert len(transport.deleted) == 4


@pytest.mark.parametrize(
    "path",
    [
        r"\\server\share\AppData\Local\RagtimeSolidWorks\work\a.png",
        ROOT.replace("\\", "/") + "/x.png",
        ROOT + r"\..\x.png",
        ROOT + r"\x.png:stream",
        ROOT.replace("work", "other") + r"\x.png",
        ROOT + r"\.hidden",
    ],
)
def test_invalid_paths(path):
    with pytest.raises(ValueError):
        _artifact_remote_path(path)


async def _gate_is_held_during_download():
    activity = HostActivity()
    started, release = asyncio.Event(), asyncio.Event()
    config = {"id": "tool", "connection_config": {"host": "render", "user": "alice"}}

    class BlockingTransport(Transport):
        async def call(self, request: dict, *, timeout_seconds: float) -> dict:
            if request["operation"] == "describe":
                result = {"operations": [{"name": "model.preview", "access": "read"}]}
            else:
                result = {"files": [file_info("preview", "x.png")]}
            return {"protocol": 1, "id": request["id"], "operation": request["operation"], "ok": True, "result": result, "host": {}}

        async def download_file(self, remote_path: str, local_path: Path, *, max_bytes: int, timeout_seconds: float) -> int:
            started.set()
            await release.wait()
            return await super().download_file(remote_path, local_path, max_bytes=max_bytes, timeout_seconds=timeout_seconds)

    service = SolidworksService(lambda config: BlockingTransport())
    with mock.patch("ragtime.solidworks.service.host_activity", activity), mock.patch("ragtime.solidworks.service.store_artifact", side_effect=store):
        task = asyncio.create_task(service.execute(config, "model.preview"))
        try:
            await asyncio.wait_for(started.wait(), 5)
            with pytest.raises(HostBusyError):
                await activity.run_exclusive(host_key(config), 0, lambda: asyncio.sleep(0))
        finally:
            release.set()
            await task


def test_export_and_preview_link_and_bound():
    asyncio.run(_export_and_preview_link_and_bound())


def test_preview_and_failure_cleanup():
    asyncio.run(_preview_and_failure_cleanup())


def test_gate_is_held_during_download():
    asyncio.run(_gate_is_held_during_download())


@pytest.mark.parametrize(
    "files",
    [
        [file_info("preview", "one.png"), {"path": r"\\server\bad"}, file_info("preview", "two.png")],
        [file_info("preview", f"{index}.png") for index in range(4)],
    ],
)
def test_invalid_collection_cleans_all_valid_paths(files):
    async def run():
        transport = Transport()
        response = {"operation": "model.preview", "result": {"files": files}}
        output = json.loads(await SolidworksService()._fetch_artifacts(response, transport, {}, time.monotonic() + 10))
        assert output["error"]["code"] == "artifact_transfer_failed"
        assert transport.deleted == [item["path"].replace("\\", "/") for item in files if item["path"].startswith("C:")]
        assert not transport.downloaded

    asyncio.run(run())


def test_real_store_fetch_and_owner_resolution(tmp_path):
    async def run():
        transport = Transport()
        response = {"protocol": 1, "operation": "model.preview", "result": {"files": [file_info("preview", "x.png")]}}
        context = SimpleNamespace(user_id="owner", surface="chat", mcp_route=None)
        with (
            mock.patch("ragtime.solidworks.artifacts.ARTIFACT_DIR", tmp_path / "artifacts"),
            mock.patch("ragtime.solidworks.service.current_context", return_value=context),
        ):
            output = json.loads(await SolidworksService()._fetch_artifacts(response, transport, {"id": "tool"}, time.monotonic() + 10))
            meta, path = await resolve_artifact_content(output["result"]["artifact"]["id"], SimpleNamespace(id="owner", role="user"))
            assert meta["owner_user_id"] == "owner"
            assert path.read_bytes() == DATA

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["download", "verify", "store", "delete"])
def test_failure_logging_is_sanitized(phase, caplog):
    async def run():
        transport = Transport()
        item = file_info("preview", "x.png")
        if phase == "verify":
            item["sha256"] = "0" * 64
        response = {"protocol": 1, "operation": "model.preview", "result": {"files": [item]}}
        target = {"download": "download_file", "delete": "remove_file"}.get(phase)
        failure = (
            mock.patch.object(transport, target, side_effect=OSError("secret /private/path"))
            if target
            else mock.patch("ragtime.solidworks.service.store_artifact", side_effect=OSError("secret /private/path") if phase == "store" else store)
        )
        with mock.patch("ragtime.solidworks.service.store_artifact", side_effect=store), failure:
            await SolidworksService()._fetch_artifacts(response, transport, {"id": "tool"}, time.monotonic() + 10)
        assert f"phase={phase}" in caplog.text
        assert "operation=model.preview" in caplog.text and "tool_id=tool" in caplog.text
        assert "secret" not in caplog.text and "/private/path" not in caplog.text and ROOT not in caplog.text

    asyncio.run(run())


MISSING = object()


def _agent_failure_output(error, operation="model.preview"):
    class FailingTransport(Transport):
        async def call(self, request: dict, *, timeout_seconds: float) -> dict:
            if request["operation"] == "describe":
                result = {"operations": [{"name": operation, "access": "read"}]}
                return {"protocol": 1, "id": request["id"], "operation": "describe", "ok": True, "result": result, "host": {}}
            response = {"protocol": 1, "id": request["id"], "operation": request["operation"], "ok": False, "host": {}}
            if error is not MISSING:
                response["error"] = error
            return response

    async def run():
        config = {"id": "tool", "connection_config": {"host": "render", "user": "alice"}}
        return json.loads(await SolidworksService(lambda config: FailingTransport()).execute(config, operation))

    return asyncio.run(run())


def test_agent_invalid_arguments_message_and_details_pass_through():
    error = {"code": "invalid_arguments", "message": "Drawings support native only.", "details": {"errors": ["native"], "warnings": []}}
    output = _agent_failure_output(error)
    assert output["ok"] is False
    assert output["error"] == error


def test_agent_export_failure_details_pass_through():
    error = {"code": "export_failed", "message": "SolidWorks could not save the file in that format.", "details": {"errors": [1], "warnings": []}}
    output = _agent_failure_output(error, "model.export")
    assert output["error"] == error


@pytest.mark.parametrize(
    "message",
    ["", "x" * 501, 123, None],
)
def test_agent_message_falls_back_to_generic(message):
    output = _agent_failure_output({"code": "export_failed", "message": message, "details": {}})
    assert output["error"]["code"] == "export_failed"
    assert output["error"]["message"] == "The SolidWorks agent could not complete the requested operation."


def test_agent_message_at_limit_passes_through():
    output = _agent_failure_output({"code": "export_failed", "message": "x" * 500})
    assert output["error"]["message"] == "x" * 500


@pytest.mark.parametrize("details", [["not", "a", "dict"], "text", None])
def test_agent_non_dict_details_are_dropped(details):
    output = _agent_failure_output({"code": "export_failed", "message": "Failed.", "details": details})
    assert output["error"]["message"] == "Failed."
    assert output["error"]["details"] == {}


def test_agent_oversized_details_are_dropped():
    output = _agent_failure_output({"code": "export_failed", "message": "Failed.", "details": {"errors": ["e" * 2100]}})
    assert output["error"]["details"] == {}


@pytest.mark.parametrize("code", ["Bad-Code", "x" * 65, "", None])
def test_agent_invalid_code_becomes_internal_error(code):
    output = _agent_failure_output({"code": code, "message": "Failed."})
    assert output["error"]["code"] == "internal_error"
    assert output["error"]["message"] == "Failed."


@pytest.mark.parametrize("error", [MISSING, "export failed", ["export_failed"], 42])
def test_agent_failure_without_error_dict_is_internal_error(error):
    output = _agent_failure_output(error)
    assert output["ok"] is False
    assert output["error"]["code"] == "internal_error"
    assert output["error"]["message"] == "The SolidWorks agent could not complete the requested operation."
    assert output["error"]["details"] == {}
