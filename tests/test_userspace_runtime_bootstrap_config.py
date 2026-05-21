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

    manifest = UserSpaceService()._extract_workspace_archive_sync(archive_path, tmp_path / "extract")

    assert manifest == {"version": "1.0"}
