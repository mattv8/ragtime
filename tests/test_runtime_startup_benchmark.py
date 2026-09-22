"""Unit coverage for the disposable runtime-startup benchmark CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_userspace_runtime.py"
SPEC = importlib.util.spec_from_file_location("runtime_startup_benchmark", SCRIPT)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_parser_selects_pinned_baseline_session_and_worker_runner(tmp_path) -> None:
    args = benchmark.parse_args(
        [
            "--baseline-ref",
            "abc123",
            "--session-dir",
            str(tmp_path),
            "--runner",
            "worker",
            "--smoke",
        ]
    )

    assert args.baseline_ref == "abc123"
    assert args.session_dir == tmp_path
    assert args.runner == "worker"
    assert args.output == tmp_path / "D-benchmark-raw.jsonl"


def test_defaults_preserve_legacy_baseline_and_session_location() -> None:
    args = benchmark.parse_args(["--smoke"])

    assert args.baseline_ref == "d9ca2e2f7e003dd8c1a8ca079643f5913038fa3f"
    assert args.session_dir == benchmark.ROOT / ".opencode/sessions/userspace-spinup"


def test_baseline_ref_is_resolved_before_safe_cache_naming(tmp_path) -> None:
    with mock.patch.object(benchmark, "run", return_value="a" * 40 + "\n") as run:
        revision = benchmark.resolve_baseline_revision("refs/heads/feature/startup")

    assert revision == "a" * 40
    assert run.call_args.args[0][-1] == "refs/heads/feature/startup^{commit}"
    assert benchmark.baseline_archive_path(tmp_path, revision) == tmp_path / f"baseline-{revision}.tar.gz"


def test_bootstrap_counter_reports_repeated_execution_despite_valid_stamp(tmp_path) -> None:
    ragtime = tmp_path / ".ragtime"
    ragtime.mkdir()
    (ragtime / ".runtime-bootstrap.done").write_text("valid stamp", encoding="utf-8")
    (ragtime / "benchmark-bootstrap-count").write_text("2", encoding="utf-8")

    assert benchmark.read_bootstrap_execution_count(tmp_path) == 2


def test_snapshot_source_excludes_local_state_and_dependency_cache(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "keep.py").write_text("kept\n", encoding="utf-8")
    for ignored in (".data", ".env", ".opencode", "node_modules", ".pytest_cache"):
        directory = source / ignored
        directory.mkdir()
        (directory / "secret").write_text("excluded\n", encoding="utf-8")

    frozen = benchmark.snapshot_source(source, tmp_path / "frozen")

    assert (frozen / "keep.py").is_file()
    assert not any((frozen / ignored).exists() for ignored in (".data", ".env", ".opencode", "node_modules", ".pytest_cache"))


def test_worker_runner_uses_baseline_worker_startup_apis() -> None:
    compile(benchmark._WORKER_RUNNER, "worker-benchmark-runner", "exec")
    assert "WorkerService" in benchmark._WORKER_RUNNER
    assert "WorkerStartSessionRequest" in benchmark._WORKER_RUNNER
    assert ".start_session(" in benchmark._WORKER_RUNNER
    assert ".get_session(" in benchmark._WORKER_RUNNER
    assert ".stop_session(" in benchmark._WORKER_RUNNER
    assert ".shutdown(" in benchmark._WORKER_RUNNER
    assert "runtime_operation_phase" in benchmark._WORKER_RUNNER
    assert "launch_port" in benchmark._WORKER_RUNNER
    assert "service = WorkerService()" in benchmark._WORKER_RUNNER
    assert "services = [WorkerService()" not in benchmark._WORKER_RUNNER
    assert "def seed_workspace" in benchmark._WORKER_RUNNER
    assert "copy_fixture(files, fixture)" not in benchmark._WORKER_RUNNER.split("async def launch", 1)[1]
    assert "bootstrap_execution_count" in benchmark._WORKER_RUNNER
    assert "benchmark-bootstrap-count" in benchmark._WORKER_RUNNER
    assert ".runtime-bootstrap.done" not in benchmark._WORKER_RUNNER
