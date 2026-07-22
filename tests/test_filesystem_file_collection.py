from pathlib import Path
from unittest import IsolatedAsyncioTestCase, mock

from ragtime.indexer.file_utils import (
    collect_files_recursive,
    has_binary_content,
    should_index_file_type,
)
from ragtime.indexer.filesystem_service import FilesystemIndexerService
from ragtime.indexer.models import (
    FilesystemAnalysisJob,
    FilesystemAnalysisStatus,
    FilesystemConnectionConfig,
    OcrMode,
)


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
    assert (
        should_index_file_type(
            text_file,
            matches_include_pattern=False,
            ocr_enabled=False,
        )
        is True
    )
    assert (
        should_index_file_type(
            binary_file,
            matches_include_pattern=True,
            ocr_enabled=False,
        )
        is False
    )
    assert (
        should_index_file_type(
            image_file,
            matches_include_pattern=True,
            ocr_enabled=False,
        )
        is False
    )
    assert (
        should_index_file_type(
            image_file,
            matches_include_pattern=False,
            ocr_enabled=True,
        )
        is True
    )
    assert (
        should_index_file_type(
            pdf_file,
            matches_include_pattern=False,
            ocr_enabled=False,
        )
        is False
    )
    assert (
        should_index_file_type(
            pdf_file,
            matches_include_pattern=True,
            ocr_enabled=False,
        )
        is True
    )


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


class FilesystemAnalysisTests(IsolatedAsyncioTestCase):
    async def test_analysis_uses_single_walk_without_preliminary_rglob(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            _write(base_path / "keep.txt", b"hello world\n")
            _write(base_path / "src" / "nested.txt", b"nested\n")
            _write(base_path / "src" / "deeper" / "deep.txt", b"deep\n")
            _write(base_path / "node_modules" / "skip.js", b"console.log('skip')\n")
            (base_path / "keep-link.txt").symlink_to(base_path / "keep.txt")
            (base_path / "linked-dir").symlink_to(base_path / "src", target_is_directory=True)

            config = FilesystemConnectionConfig(
                base_path=str(base_path),
                index_name="analysis-test",
                file_patterns=["**/*"],
                exclude_patterns=["**/node_modules/**"],
                ocr_mode=OcrMode.DISABLED,
            )
            job = FilesystemAnalysisJob(id="job-1", tool_config_id="tool-1")
            service = FilesystemIndexerService()
            service._append_embedding_dimension_warning = mock.AsyncMock()  # type: ignore[method-assign]

            with (
                mock.patch(
                    "ragtime.indexer.llm_exclusions.get_smart_exclusion_suggestions",
                    new=mock.AsyncMock(return_value=([], False)),
                ),
                mock.patch.object(Path, "rglob", side_effect=AssertionError("analysis should not use rglob")),
            ):
                await service._run_analysis(job, config)

            self.assertEqual(job.status, FilesystemAnalysisStatus.COMPLETED)
            self.assertGreaterEqual(job.total_dirs_to_scan, job.dirs_scanned)
            self.assertGreaterEqual(job.total_dirs_to_scan, 3)
            result = service._analysis_results[job.id]
            self.assertEqual(result.total_files, 3)
            self.assertEqual(result.directories_scanned, job.dirs_scanned)
