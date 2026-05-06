from pathlib import Path

from ragtime.indexer.filesystem_service import FilesystemIndexerService
from ragtime.indexer.file_utils import (
    collect_files_recursive,
    has_binary_content,
    should_index_file_type,
)
from ragtime.indexer.models import FilesystemConnectionConfig, OcrMode


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_collect_files_includes_non_binary_files_outside_extension_patterns(tmp_path):
    _write(tmp_path / "README.md", b"# Notes\n")
    _write(tmp_path / "LaunchAgent.plist", b"<?xml version='1.0'?><plist></plist>\n")
    _write(tmp_path / "script", b"#!/bin/sh\necho hello\n")
    _write(tmp_path / "data.bin", b"\x00\x01\x02\x03")
    _write(tmp_path / "photo.jpg", b"\xff\xd8\xff\xe0binary-jpeg")

    config = FilesystemConnectionConfig(
        base_path=str(tmp_path),
        index_name="test",
        file_patterns=["**/*.md"],
        exclude_patterns=[],
        ocr_mode=OcrMode.DISABLED,
    )

    collected = FilesystemIndexerService()._collect_files(config)

    assert {path.name for path in collected} == {
        "README.md",
        "LaunchAgent.plist",
        "script",
    }


def test_file_type_policy_detects_text_and_binary_samples(tmp_path):
    text_file = tmp_path / "config.plist"
    binary_file = tmp_path / "archive.bin"
    image_file = tmp_path / "photo.jpg"
    pdf_file = tmp_path / "report.pdf"
    text_file.write_text("<?xml version='1.0'?><plist></plist>\n", encoding="utf-8")
    binary_file.write_bytes(b"\x00\x01\x02\x03")
    image_file.write_bytes(b"\xff\xd8\xff\xe0binary-jpeg")
    pdf_file.write_bytes(b"%PDF-1.4\n")

    assert has_binary_content(text_file) is False
    assert has_binary_content(binary_file) is True
    assert should_index_file_type(
        text_file,
        matches_include_pattern=False,
        ocr_enabled=False,
    ) is True
    assert should_index_file_type(
        binary_file,
        matches_include_pattern=True,
        ocr_enabled=False,
    ) is False
    assert should_index_file_type(
        image_file,
        matches_include_pattern=True,
        ocr_enabled=False,
    ) is False
    assert should_index_file_type(
        image_file,
        matches_include_pattern=False,
        ocr_enabled=True,
    ) is True
    assert should_index_file_type(
        pdf_file,
        matches_include_pattern=False,
        ocr_enabled=False,
    ) is False
    assert should_index_file_type(
        pdf_file,
        matches_include_pattern=True,
        ocr_enabled=False,
    ) is True


def test_shared_collector_includes_non_binary_files_outside_patterns(tmp_path):
    _write(tmp_path / "README.md", b"# Notes\n")
    _write(tmp_path / "LaunchAgent.plist", b"<?xml version='1.0'?><plist></plist>\n")
    _write(tmp_path / "script", b"#!/bin/sh\necho hello\n")
    _write(tmp_path / "node_modules" / "package" / "index.js", b"console.log('skip')\n")
    _write(tmp_path / "data.bin", b"\x00\x01\x02\x03")
    _write(tmp_path / "photo.jpg", b"\xff\xd8\xff\xe0binary-jpeg")

    collected = collect_files_recursive(
        tmp_path,
        include_patterns=["**/*.md"],
        exclude_patterns=[],
        ocr_enabled=False,
    )

    assert {path.name for path, _size in collected} == {
        "README.md",
        "LaunchAgent.plist",
        "script",
    }
