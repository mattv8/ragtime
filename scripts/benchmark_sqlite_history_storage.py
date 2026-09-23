#!/usr/bin/env python3
"""Benchmark incremental SQLite history storage with real capture images.

Each scenario ingests a baseline image into its own Restic repository, then
captures and ingests a changed (or unchanged) second image. Repository storage
is measured from the repository filesystem before and after the second ingest;
it is not inferred from Restic's retained-total statistics.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import resource
import sqlite3
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.core.sqlite_recovery import capture_database
from runtime.worker.sqlite_history.models import ResticArtifact
from runtime.worker.sqlite_history.repository import ResticRepository
from scripts.sqlite_history_benchmark_fixtures import SQLiteBenchmarkFixture

DEFAULT_RESTIC_BINARIES = (
    Path("/var/folders/v2/jhfs1xs518s0fv3tlh30zm400000gn/T/opencode/restic"),
    Path("/opt/ragtime-backup/bin/restic"),
)
SCENARIOS = ("unchanged", "sparse", "append", "wal", "vacuum")


@dataclass(frozen=True)
class TimedImage:
    capture_seconds: float
    logical_image_size_bytes: int
    sha256: str


@dataclass(frozen=True)
class TimedRestore:
    materialize_seconds: float
    sha256_verified: bool


@dataclass(frozen=True)
class RepositoryMeasurements:
    initial_ingest_seconds: float
    second_ingest_seconds: float
    repository_bytes_before_second_ingest: int
    repository_bytes_after_second_ingest: int
    repository_stored_delta_bytes: int
    retained_repository_size_bytes: int


@dataclass(frozen=True)
class BenchmarkResult:
    scenario: str
    size_mib: int
    initial_image: TimedImage
    second_image: TimedImage
    initial_restore: TimedRestore
    second_restore: TimedRestore
    repository: RepositoryMeasurements
    resource_usage: dict[str, int | None]
    run_directory: str
    error: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size_bytes(path: Path) -> int:
    """Return actual regular-file bytes currently retained by a repository."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _resource_usage() -> dict[str, int | None]:
    """Report only counters supplied by the platform; do not invent I/O values."""
    try:
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        max_rss_bytes = max_rss if sys.platform == "darwin" else max_rss * 1024
    except (AttributeError, OSError):
        max_rss_bytes = None
    return {
        "process_peak_rss_bytes": max_rss_bytes,
        "process_read_bytes": None,
        "process_write_bytes": None,
    }


def _capture(source: Path, destination: Path, *, source_connection: sqlite3.Connection | None = None) -> TimedImage:
    started = time.perf_counter()
    result = capture_database(
        source,
        destination,
        include_fingerprint=False,
        source_connection=source_connection,
    )
    return TimedImage(
        capture_seconds=time.perf_counter() - started,
        logical_image_size_bytes=int(result["size_bytes"]),
        sha256=str(result["sha256"]),
    )


async def _ingest(repository: ResticRepository, image: Path, operation_id: str) -> tuple[ResticArtifact, float]:
    started = time.perf_counter()
    artifact = await repository.ingest(
        image,
        workspace_id="benchmark-workspace",
        operation_id=operation_id,
        sha256=_sha256_file(image),
        size_bytes=image.stat().st_size,
    )
    return artifact, time.perf_counter() - started


async def _materialize_and_verify(repository: ResticRepository, artifact: ResticArtifact, destination: Path) -> TimedRestore:
    started = time.perf_counter()
    await repository.materialize(artifact, destination)
    elapsed = time.perf_counter() - started
    return TimedRestore(elapsed, _sha256_file(destination) == artifact.sha256)


def _mutate(fixture: SQLiteBenchmarkFixture, source: Path, scenario: str) -> sqlite3.Connection | None:
    if scenario == "unchanged":
        return None
    if scenario == "sparse":
        fixture.mutate_sparse(source)
    elif scenario == "append":
        fixture.mutate_append(source)
    elif scenario == "wal":
        return fixture.mutate_wal_only(source)
    elif scenario == "vacuum":
        fixture.mutate_vacuum(source)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return None


async def _run_single_benchmark(size_mib: int, scenario: str, run_directory: Path, restic_binary: Path) -> BenchmarkResult:
    run_directory.mkdir(parents=False, exist_ok=False)
    source = run_directory / "source.sqlite3"
    fixture = SQLiteBenchmarkFixture(size_mib=size_mib, seed=42)
    fixture.create_fixture(source)

    initial_image = _capture(source, run_directory / "initial-image.sqlite3")
    repository_path = run_directory / "restic-repository"
    repository = ResticRepository(
        repository_path=repository_path,
        cache_path=run_directory / "restic-cache",
        password_path=run_directory / "restic-secrets" / "password",
        scratch_path=run_directory / "restic-scratch",
        binary=restic_binary,
    )
    await repository.initialize()
    initial_artifact, initial_ingest_seconds = await _ingest(repository, run_directory / "initial-image.sqlite3", "initial")

    wal_connection = _mutate(fixture, source, scenario)
    try:
        second_image = _capture(source, run_directory / "second-image.sqlite3", source_connection=wal_connection)
    finally:
        if wal_connection is not None:
            wal_connection.close()

    repository_bytes_before_second_ingest = _directory_size_bytes(repository_path)
    second_artifact, second_ingest_seconds = await _ingest(repository, run_directory / "second-image.sqlite3", "second")
    repository_bytes_after_second_ingest = _directory_size_bytes(repository_path)
    initial_restore = await _materialize_and_verify(repository, initial_artifact, run_directory / "restored-initial.sqlite3")
    second_restore = await _materialize_and_verify(repository, second_artifact, run_directory / "restored-second.sqlite3")
    if not initial_restore.sha256_verified or not second_restore.sha256_verified:
        raise RuntimeError("Materialized image SHA-256 did not match its captured image")

    return BenchmarkResult(
        scenario=scenario,
        size_mib=size_mib,
        initial_image=initial_image,
        second_image=second_image,
        initial_restore=initial_restore,
        second_restore=second_restore,
        repository=RepositoryMeasurements(
            initial_ingest_seconds=initial_ingest_seconds,
            second_ingest_seconds=second_ingest_seconds,
            repository_bytes_before_second_ingest=repository_bytes_before_second_ingest,
            repository_bytes_after_second_ingest=repository_bytes_after_second_ingest,
            repository_stored_delta_bytes=repository_bytes_after_second_ingest - repository_bytes_before_second_ingest,
            retained_repository_size_bytes=repository_bytes_after_second_ingest,
        ),
        resource_usage=_resource_usage(),
        run_directory=str(run_directory),
    )


def _empty_result(size_mib: int, scenario: str, run_directory: Path, error: Exception) -> dict[str, Any]:
    return {"scenario": scenario, "size_mib": size_mib, "run_directory": str(run_directory), "error": str(error)}


def _resolve_restic_binary(requested: Path | None) -> Path:
    if requested is not None:
        if requested.is_file():
            return requested
        raise FileNotFoundError(f"Restic binary not found at {requested}")
    for candidate in DEFAULT_RESTIC_BINARIES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Restic binary not found; set --restic-binary")


async def run_benchmark(sizes_mib: list[int], scenarios: list[str], workdir: Path | None, restic_binary: Path | None) -> tuple[list[dict[str, Any]], Path]:
    binary = _resolve_restic_binary(restic_binary)
    if workdir is None:
        parent = Path(tempfile.mkdtemp(prefix="sqlite-history-benchmark-"))
    else:
        workdir = workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        parent = Path(tempfile.mkdtemp(prefix="sqlite-history-benchmark-", dir=workdir))

    results: list[dict[str, Any]] = []
    for size_mib in sizes_mib:
        for scenario in scenarios:
            scenario_directory = parent / f"{size_mib}mib-{scenario}"
            print(f"Running benchmark: {size_mib}MiB {scenario}", file=sys.stderr)
            try:
                results.append(asdict(await _run_single_benchmark(size_mib, scenario, scenario_directory, binary)))
            except Exception as error:
                results.append(_empty_result(size_mib, scenario, scenario_directory, error))
    return results, parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[1, 64, 256])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--restic-binary", type=Path)
    args = parser.parse_args()
    try:
        results, run_directory = asyncio.run(run_benchmark(args.sizes_mib, args.scenarios, args.workdir, args.restic_binary))
        report = {
            "version": 2,
            "timestamp": time.time(),
            "run_directory": str(run_directory),
            "parameters": {"sizes_mib": args.sizes_mib, "scenarios": args.scenarios},
            "results": results,
        }
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 1 if any(result.get("error") is not None for result in results) else 0
    except Exception as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
