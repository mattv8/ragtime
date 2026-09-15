import hashlib
import json
import os
import sys
import tarfile
import tempfile
import types
from unittest.mock import patch

os.environ.setdefault("INDEX_DATA_PATH", os.path.join(tempfile.gettempdir(), "ragtime-test-index-data"))

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace.service import (
    _RUNTIME_BOOTSTRAP_CONFIG_PATH,
    _RUNTIME_BOOTSTRAP_TEMPLATE_VERSION,
    UserSpaceService,
)
from runtime.worker.service import WorkerService


def test_sync_runtime_bootstrap_config_accepts_floatlike_template_version(tmp_path) -> None:
    service = UserSpaceService()
    files_dir = tmp_path / "workspace-files"
    config_path = files_dir / _RUNTIME_BOOTSTRAP_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "managed_by": "ragtime",
                "auto_update": True,
                "template_version": "1.0",
                "watch_paths": [],
                "commands": [],
            }
        ),
        encoding="utf-8",
    )

    with patch.object(service, "_workspace_files_dir", return_value=files_dir):
        service._sync_runtime_bootstrap_config("workspace-1")

    updated = json.loads(config_path.read_text(encoding="utf-8"))
    assert updated["managed_by"] == "ragtime"
    assert updated["template_version"] == _RUNTIME_BOOTSTRAP_TEMPLATE_VERSION


def test_sync_runtime_bootstrap_config_respects_string_false_auto_update(tmp_path) -> None:
    service = UserSpaceService()
    files_dir = tmp_path / "workspace-files"
    config_path = files_dir / _RUNTIME_BOOTSTRAP_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    original = {
        "version": 1,
        "managed_by": "ragtime",
        "auto_update": "false",
        "template_version": "1.0",
        "watch_paths": [],
        "commands": [{"name": "custom_setup", "run": "true"}],
    }
    config_path.write_text(json.dumps(original), encoding="utf-8")

    with patch.object(service, "_workspace_files_dir", return_value=files_dir):
        service._sync_runtime_bootstrap_config("workspace-1")

    assert json.loads(config_path.read_text(encoding="utf-8")) == original


def test_legacy_default_bootstrap_detection_accepts_floatlike_version() -> None:
    payload = {
        "version": "1.0",
        "commands": [
            {"name": "npm_ci"},
            {"name": "npm_install"},
            {"name": "pip_requirements"},
        ],
    }

    assert UserSpaceService._is_legacy_default_bootstrap(payload) is True


def test_workspace_archive_manifest_accepts_floatlike_version(tmp_path) -> None:
    archive_path = tmp_path / "workspace.tar.gz"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "manifest.json").write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source_dir / "manifest.json", arcname="manifest.json")

    manifest = UserSpaceService()._extract_workspace_archive_sync(
        archive_path,
        tmp_path / "extract",
        max_entries=10,
        max_bytes=1024,
    )

    assert manifest == {"version": "1.0"}


def test_worker_bootstrap_digest_streams_watched_file_bytes_without_changing_digest(tmp_path) -> None:
    config = {
        "watch_paths": ["package-lock.json"],
        "commands": [{"name": "npm_ci", "run": "npm ci"}],
    }
    payload = json.dumps(config, separators=(",", ":")).encode("utf-8")
    config_path = tmp_path / ".ragtime" / "runtime-bootstrap.json"
    config_path.parent.mkdir()
    config_path.write_bytes(payload)
    watched_path = tmp_path / "package-lock.json"
    watched_bytes = b'{"lockfileVersion":3}\n'
    watched_path.write_bytes(watched_bytes)
    expected = hashlib.sha256(payload + b"package-lock.json::file" + watched_bytes).hexdigest()

    original_read_bytes = type(watched_path).read_bytes

    def reject_watched_read_bytes(path):
        if path == watched_path:
            raise AssertionError("watched file must be streamed")
        return original_read_bytes(path)

    with patch.object(type(watched_path), "read_bytes", new=reject_watched_read_bytes):
        digest = WorkerService()._runtime_bootstrap_config_digest_sync(tmp_path)

    assert digest == expected
