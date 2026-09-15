#!/usr/bin/env python3
"""Reproducible, disposable User Space sandbox startup benchmark.

The candidate and pinned baseline are frozen into filtered snapshots before
either run.  A warm run primes each lane root once, then measures restarts on
that same root; cold runs intentionally allocate one root per sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT / ".opencode/sessions/userspace-spinup"
BASELINE_REVISION = "d9ca2e2f7e003dd8c1a8ca079643f5913038fa3f"
BASELINE_ARCHIVE = SESSION / f"baseline-{BASELINE_REVISION}.tar.gz"
FIXTURES = ROOT / "tests/fixtures/userspace_runtime_benchmark"


_RUNNER = r"""
import asyncio, contextlib, json, os, resource, socket, subprocess, sys, time, urllib.request
from pathlib import Path
from runtime.worker import sandbox

def io_values():
    values = {}
    with contextlib.suppress(OSError):
        for line in Path('/proc/self/io').read_text().splitlines():
            key, value = line.split(':', 1); values[key.strip()] = int(value.strip())
    return values

def usage_delta(before, after):
    return {'user_ms': (after.ru_utime-before.ru_utime)*1000, 'system_ms': (after.ru_stime-before.ru_stime)*1000,
            'minor_faults': after.ru_minflt-before.ru_minflt, 'major_faults': after.ru_majflt-before.ru_majflt}

def app_command(fixture, port):
    if fixture == 'static': return ('python3', '-m', 'http.server', str(port), '--bind', '127.0.0.1', '--directory', '/workspace')
    if fixture == 'node': return ('node', '/workspace/server.js')
    return ('python3', '/workspace/app.py')

async def fetch(url):
    return await asyncio.to_thread(lambda: urllib.request.urlopen(url, timeout=1).read())

async def ready(port):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            main, asset = await asyncio.gather(fetch('http://127.0.0.1:%d/' % port), fetch('http://127.0.0.1:%d/asset.js' % port))
            if b'runtime-benchmark-ready' in main and b'runtimeBenchmarkAssetLoaded' in asset: return
        except Exception: pass
        await asyncio.sleep(.025)
    raise RuntimeError('HTTP app-ready marker or asset unavailable')

async def drain(stream, sink):
    if stream is None: return
    while True:
        chunk = await stream.read(8192)
        if not chunk: return
        sink.extend(chunk[-65536:])

async def stop(process, spec):
    started = time.perf_counter_ns()
    if process.returncode is None:
        process.terminate()
        try: await asyncio.wait_for(process.wait(), 5)
        except TimeoutError:
            process.kill()
            await asyncio.wait_for(process.wait(), 5)
    await asyncio.to_thread(sandbox.cleanup_sandbox, spec)
    return (time.perf_counter_ns() - started) / 1e6

def copy_fixture(files, fixture):
    for item in Path('/fixture').iterdir():
        if item.is_file(): (files / item.name).write_bytes(item.read_bytes())
    if fixture == 'node':
        dependencies = files / 'node_modules' / 'benchmark-synthetic-dependencies'
        for index in range(2048):
            dependency = dependencies / ('pkg-%04d' % index) / 'index.js'
            dependency.parent.mkdir(parents=True, exist_ok=True)
            dependency.write_text('module.exports=%d;\n' % index)

async def launch(spec, fixture):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
    stdout, stderr = bytearray(), bytearray()
    before_self, before_children, before_io = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN), io_values()
    started = time.perf_counter_ns()
    process = None
    drains = []
    failure = None
    try:
        process = await sandbox.spawn_sandboxed(spec, app_command(fixture, port), cwd='/workspace', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env={'PORT': str(port)})
        drains = [asyncio.create_task(drain(process.stdout, stdout)), asyncio.create_task(drain(process.stderr, stderr))]
        await ready(port)
        startup_ms = (time.perf_counter_ns() - started) / 1e6
    except Exception as exc:
        startup_ms = (time.perf_counter_ns() - started) / 1e6
        failure = str(exc)
    finally:
        stop_ms = await stop(process, spec) if process is not None else 0.0
        if drains: await asyncio.gather(*drains, return_exceptions=True)
    after_self, after_children, after_io = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN), io_values()
    if failure:
        raise RuntimeError('%s; stdout_tail=%r; stderr_tail=%r' % (failure, bytes(stdout[-4096:]), bytes(stderr[-4096:])))
    return {'startup_to_app_ready_ms': startup_ms, 'stop_cleanup_ms': stop_ms, 'app_ready_http': True, 'asset_fetch': True,
            'cpu_scope': {'self': usage_delta(before_self, after_self), 'children_reaped_after_stop': usage_delta(before_children, after_children)},
            'io_scope': {'runner_process': {'read_bytes': after_io.get('read_bytes',0)-before_io.get('read_bytes',0), 'write_bytes': after_io.get('write_bytes',0)-before_io.get('write_bytes',0)}}}

def mount(path):
    return subprocess.check_output(['stat', '-fc', '%T %d', path], text=True).strip()

async def main():
    base, mode, fixture, profile, count, lanes, storage = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
    specs, primes = [], []
    if profile == 'warm':
        for lane in range(lanes):
            root = base / ('lane-%d' % lane); files = root/'files'; files.mkdir(parents=True, exist_ok=True); copy_fixture(files, fixture)
            spec = sandbox.SandboxSpec(workspace_id='bench-%s-%d' % (fixture, lane), workspace_files_path=files, rootfs_path=root/'rootfs', mode=mode)
            prime = await launch(spec, fixture) # exactly one prime per lane root
            specs.append(spec)
            primes.append(prime)
    async def one(index):
        if profile == 'cold':
            root = base / ('cold-%03d' % index); files = root/'files'; files.mkdir(parents=True, exist_ok=True); copy_fixture(files, fixture)
            spec = sandbox.SandboxSpec(workspace_id='bench-%s-cold-%d' % (fixture,index), workspace_files_path=files, rootfs_path=root/'rootfs', mode=mode)
        else: spec = specs[index % lanes]
        result = await launch(spec, fixture)
        if profile == 'warm':
            # This is intentionally duplicated on the lane's samples so each
            # raw restart can be interpreted with its one-time rootfs cost.
            result['warm_prime_once_per_lane'] = primes[index % lanes]
        if lanes > 1:
            # getrusage(RUSAGE_CHILDREN) belongs to this runner process, not
            # an asyncio task.  Concurrent samples would overlap, so do not
            # publish misleading per-workload CPU/fault deltas.
            result.pop('cpu_scope', None)
            result.pop('io_scope', None)
            result['resource_metrics_scope'] = 'omitted: concurrent lanes share runner RUSAGE_CHILDREN and /proc/self/io counters'
        else:
            result['resource_metrics_scope'] = 'per-sample runner self plus children, sampled after child exit and sandbox cleanup'
        result.update({'sample': index, 'mode': mode, 'fixture': fixture, 'fixture_dependency_label': '2048 synthetic Node dependency files' if fixture == 'node' else None,
                       'profile': profile, 'lanes': lanes, 'storage_requested': storage, 'capabilities': sandbox.sandbox_diagnostics(),
                       'mount_metadata': {'bench': mount('/bench'), 'usr': mount('/usr')}})
        print(json.dumps(result), flush=True)
    for offset in range(0, count, lanes): await asyncio.gather(*(one(index) for index in range(offset, min(count, offset+lanes))))
asyncio.run(main())
"""


def docker_binary() -> str:
    preferred = Path("/opt/homebrew/bin/docker")
    if preferred.is_file() and os.access(preferred, os.X_OK):
        return str(preferred)
    found = shutil.which("docker")
    if found:
        return found
    raise RuntimeError("Docker executable unavailable (tried /opt/homebrew/bin/docker then PATH)")


def run(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {completed.stderr.strip()}")
    return completed.stdout


def ensure_baseline_archive() -> str:
    if not BASELINE_ARCHIVE.exists():
        BASELINE_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        with BASELINE_ARCHIVE.open("wb") as output:
            archive = subprocess.Popen(["git", "archive", "--format=tar", BASELINE_REVISION], cwd=ROOT, stdout=subprocess.PIPE)
            assert archive.stdout is not None
            import gzip

            with gzip.GzipFile(fileobj=output, mode="wb") as compressed:
                shutil.copyfileobj(archive.stdout, compressed)
            if archive.wait() != 0:
                raise RuntimeError("git archive failed")
    return hashlib.sha256(BASELINE_ARCHIVE.read_bytes()).hexdigest()


def safe_extract(archive: tarfile.TarFile, target: Path) -> None:
    target.mkdir()
    destination = target.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if destination not in (member_path, *member_path.parents) or member.issym() or member.islnk():
            raise RuntimeError(f"unsafe baseline archive member: {member.name}")
    # ``filter=`` is only available in newer Python releases.  Members above
    # have already been constrained to regular, in-tree archive paths.
    archive.extractall(target)


def snapshot_source(source: Path, destination: Path) -> Path:
    """Freeze only benchmark-safe bytes; never mount checkout .git/.data/.env."""
    ignored = shutil.ignore_patterns(".git", ".data", ".env", ".env.*", ".opencode", "__pycache__", "*.pyc")
    shutil.copytree(source, destination, symlinks=True, ignore=ignored)
    return destination


def frozen_sources(directory: Path) -> tuple[Path, Path, str]:
    archive_sha = ensure_baseline_archive()
    extracted = directory / "baseline-archive"
    with tarfile.open(BASELINE_ARCHIVE, "r:gz") as archive:
        safe_extract(archive, extracted)
    baseline = snapshot_source(extracted, directory / "baseline")
    candidate = snapshot_source(ROOT, directory / "candidate")
    return baseline, candidate, archive_sha


def json_lines(command: list[str], output: Path, common: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Persist each sample as it arrives, retaining stderr for failed containers."""
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    with output.open("a") as raw:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row.update(common)
            raw.write(json.dumps(row) + "\n")
            raw.flush()
            yield row
        stderr = process.stderr.read()
        if process.wait() != 0:
            raise RuntimeError(f"benchmark container failed ({process.returncode}): {stderr[-8000:]}")


def docker_run(
    source: Path, source_label: str, args: argparse.Namespace, docker: str, image_id: str, archive_sha: str, output: Path, host_root: Path | None
) -> list[dict[str, Any]]:
    command = [
        docker,
        "run",
        "--rm",
        "--entrypoint",
        "python",
        "-e",
        "PYTHONPATH=/repo",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-v",
        f"{source}:/repo:ro",
        "-v",
        f"{FIXTURES / args.fixture}:/fixture:ro",
        "-w",
        "/repo",
    ]
    if args.storage == "bind":
        assert host_root is not None
        host_root.mkdir(parents=True, exist_ok=True)
        command.extend(["-v", f"{host_root}:/bench"])
    # No /bench mount is native container writable-layer storage (typically overlayfs).
    if args.capability_profile == "sys-admin":
        command.extend(["--cap-add=SYS_ADMIN", "--security-opt", "apparmor=unconfined", "--security-opt", "seccomp=unconfined"])
    command.extend([args.image, "-c", _RUNNER, "/bench", args.mode, args.fixture, args.profile, str(args.samples), str(args.concurrency), args.storage])
    common = {
        "source": source_label,
        "baseline_revision": BASELINE_REVISION,
        "baseline_archive_sha256": archive_sha,
        "runtime_image": args.image,
        "runtime_image_id": image_id,
    }
    return list(json_lines(command, output, common))


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(row["startup_to_app_ready_ms"] for row in rows)
    percentile = lambda p: values[round((len(values) - 1) * p)] if values else 0.0
    return {
        "count": len(rows),
        "startup_to_app_ready_ms": {"p50": percentile(0.5), "p95": percentile(0.95), "mean": statistics.fmean(values) if values else 0.0},
        "p95_note": "p95 is screening-only at n=20.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--concurrency", choices=(1, 4), type=int, default=1, help="One warm root per lane; cold roots remain fresh.")
    parser.add_argument("--mode", choices=("chroot", "pivot_root"), default="chroot")
    parser.add_argument("--capability-profile", choices=("no-mount", "sys-admin"), default="no-mount")
    parser.add_argument("--fixture", choices=("static", "node", "python_sqlite"), default="static")
    parser.add_argument("--profile", choices=("cold", "warm"), default="cold")
    parser.add_argument(
        "--storage", choices=("container", "bind"), default="container", help="container is native Linux writable-layer storage; bind measures host sharing."
    )
    parser.add_argument("--image", default="docker-runtime", help="Disposable runtime image to inspect and execute.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=SESSION / "E-benchmark-raw.jsonl")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.samples > 1 and os.environ.get("RAGTIME_BENCHMARK_QUIET_WINDOW") != "1":
        parser.error("set RAGTIME_BENCHMARK_QUIET_WINDOW=1 before a multi-sample run")
    if args.samples == 1 and not args.smoke:
        parser.error("single sample is smoke-only; pass --smoke")
    if args.mode == "pivot_root" and args.capability_profile != "sys-admin":
        parser.error("pivot_root requires --capability-profile sys-admin")
    docker = docker_binary()
    image_id = run([docker, "image", "inspect", "--format", "{{.Id}}", args.image]).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Docker Desktop can mount checkout-local scratch while macOS /var temp
    # paths are not necessarily shared.  Excluding .opencode above prevents
    # a candidate snapshot from recursively copying this destination.
    with tempfile.TemporaryDirectory(prefix="userspace-runtime-benchmark-", dir=SESSION) as directory:
        temporary = Path(directory)
        baseline, candidate, archive_sha = frozen_sources(temporary)
        host_runs = temporary / "bind-runs" if args.storage == "bind" else None
        rows: list[dict[str, Any]] = []
        for label, source in (("baseline", baseline), ("candidate", candidate)):
            rows.extend(docker_run(source, label, args, docker, image_id, archive_sha, args.output, host_runs / label if host_runs else None))
    measured = {label: [row for row in rows if row["source"] == label] for label in ("baseline", "candidate")}
    print(
        json.dumps(
            {
                "summary": {label: summary(source_rows) for label, source_rows in measured.items()},
                "output": str(args.output),
                "docker": docker,
                "storage_requested": args.storage,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
