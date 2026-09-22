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
                        'effective_startup_limit': 'not applicable: direct sandbox runner',
                        'mount_metadata': {'bench': mount('/bench'), 'usr': mount('/usr')}})
        print(json.dumps(result), flush=True)
    for offset in range(0, count, lanes): await asyncio.gather(*(one(index) for index in range(offset, min(count, offset+lanes))))
asyncio.run(main())
"""


_WORKER_RUNNER = r"""
import asyncio, json, os, sys, time, urllib.request
from pathlib import Path
from runtime.manager.models import WorkerStartSessionRequest
from runtime.worker.sandbox import sandbox_diagnostics
from runtime.worker.service import WorkerService

def app_config(fixture):
    if fixture == 'static': return {'command': 'python3 -m http.server $PORT --bind 127.0.0.1 --directory /workspace', 'cwd': '.', 'framework': 'static'}
    if fixture == 'node': return {'command': 'node server.js', 'cwd': '.', 'framework': 'node'}
    return {'command': 'python3 app.py', 'cwd': '.', 'framework': 'custom'}

def seed_workspace(files, fixture):
    for item in Path('/fixture').iterdir():
        if item.is_file(): (files / item.name).write_bytes(item.read_bytes())
    if fixture == 'node':
        dependencies = files / 'node_modules' / 'benchmark-synthetic-dependencies'
        for index in range(2048):
            dependency = dependencies / ('pkg-%04d' % index) / 'index.js'
            dependency.parent.mkdir(parents=True, exist_ok=True)
            dependency.write_text('module.exports=%d;\n' % index)
    (files / 'benchmark-watch.txt').write_text('stable benchmark input\n', encoding='utf-8')
    ragtime = files / '.ragtime'; ragtime.mkdir(exist_ok=True)
    (ragtime / 'runtime-entrypoint.json').write_text(json.dumps(app_config(fixture)), encoding='utf-8')
    counter_command = "python3 -c \"from pathlib import Path; p=Path('.ragtime/benchmark-bootstrap-count'); p.write_text(str(int(p.read_text() if p.exists() else '0') + 1), encoding='utf-8')\""
    (ragtime / 'runtime-bootstrap.json').write_text(json.dumps({'commands': [{'name': 'benchmark_counter', 'run': counter_command}], 'watch_paths': ['benchmark-watch.txt']}), encoding='utf-8')

async def fetch(url):
    return await asyncio.to_thread(lambda: urllib.request.urlopen(url, timeout=1).read())

async def app_ready(port):
    main, asset = await asyncio.gather(fetch('http://127.0.0.1:%d/' % port), fetch('http://127.0.0.1:%d/asset.js' % port))
    return b'runtime-benchmark-ready' in main and b'runtimeBenchmarkAssetLoaded' in asset

async def launch(service, workspace_id, fixture):
    started = time.perf_counter_ns()
    response = await service.start_session(WorkerStartSessionRequest(workspace_id=workspace_id, provider_session_id='provider-' + workspace_id, pty_access_token='benchmark-token'))
    phases, deadline = {}, time.monotonic() + 30
    while time.monotonic() < deadline:
        response = await service.get_session(response.worker_session_id)
        if response.runtime_operation_phase and response.runtime_operation_phase not in phases:
            phases[response.runtime_operation_phase] = (time.perf_counter_ns() - started) / 1e6
        if response.devserver_running and response.launch_port and await app_ready(response.launch_port):
            return response.worker_session_id, {'startup_to_app_ready_ms': (time.perf_counter_ns() - started) / 1e6, 'app_ready_http': True, 'asset_fetch': True, 'observed_phase_ms': phases}
        if response.runtime_operation_phase == 'failed':
            raise RuntimeError(response.last_error or 'worker startup failed')
        await asyncio.sleep(.025)
    raise RuntimeError('worker did not reach app-ready HTTP marker')

def bootstrap_execution_count(base, workspace_id):
    count_path = base / 'workspaces' / workspace_id / 'files' / '.ragtime' / 'benchmark-bootstrap-count'
    return int(count_path.read_text(encoding='utf-8').strip()) if count_path.is_file() else 0

async def main():
    base, fixture, profile, count, lanes, storage, requested_mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
    os.environ['RUNTIME_WORKSPACE_ROOT'] = str(base)
    caps = sandbox_diagnostics(); actual_mode = caps.get('sandbox_mode')
    if actual_mode != requested_mode: raise RuntimeError('requested sandbox mode %s, runtime reports %s' % (requested_mode, actual_mode))
    service = WorkerService()
    effective_startup_limit = getattr(service._startup_semaphore, '_value', 'unavailable')
    primes = []
    try:
        if profile == 'warm':
            for lane in range(lanes):
                workspace_id = 'bench-%s-%d' % (fixture, lane)
                files = base / 'workspaces' / workspace_id / 'files'; files.mkdir(parents=True, exist_ok=True); seed_workspace(files, fixture)
                prime_session_id, prime = await launch(service, workspace_id, fixture)
                await service.stop_session(prime_session_id)
                if bootstrap_execution_count(base, workspace_id) != 1: raise RuntimeError('warm bootstrap did not complete exactly once per lane root')
                primes.append(prime)
        async def one(index):
            lane = index % lanes
            workspace_id = 'bench-%s-%d' % (fixture, lane) if profile == 'warm' else 'bench-%s-cold-%d' % (fixture, index)
            if profile == 'cold':
                files = base / 'workspaces' / workspace_id / 'files'; files.mkdir(parents=True, exist_ok=True); seed_workspace(files, fixture)
            session_id, result = await launch(service, workspace_id, fixture)
            stopped = time.perf_counter_ns()
            await service.stop_session(session_id)
            result['stop_cleanup_ms'] = (time.perf_counter_ns() - stopped) / 1e6
            if profile == 'warm':
                if bootstrap_execution_count(base, workspace_id) != 1: raise RuntimeError('warm restart unexpectedly invalidated bootstrap stamp')
                result['warm_prime_once_per_lane'] = primes[lane]
                result['bootstrap_execution_count'] = 1
            result.update({'sample': index, 'runner': 'worker', 'fixture': fixture, 'fixture_dependency_label': '2048 synthetic Node dependency files' if fixture == 'node' else None, 'profile': profile, 'lanes': lanes, 'storage_requested': storage, 'capabilities': caps, 'requested_sandbox_mode': requested_mode, 'actual_sandbox_mode': actual_mode, 'effective_startup_limit': effective_startup_limit, 'stage_timing_note': 'operation phases observed through baseline runtime_operation_phase; no API latency is measured'})
            print(json.dumps(result), flush=True)
        for offset in range(0, count, lanes): await asyncio.gather(*(one(index) for index in range(offset, min(count, offset + lanes))))
    finally:
        await service.shutdown()
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


def baseline_archive_path(session_dir: Path, baseline_ref: str) -> Path:
    """Return the session-local cache path for a resolved immutable revision."""
    if len(baseline_ref) != 40 or any(character not in "0123456789abcdef" for character in baseline_ref.lower()):
        raise ValueError("baseline revision must be a resolved 40-character commit hash")
    return session_dir / f"baseline-{baseline_ref}.tar.gz"


def resolve_baseline_revision(baseline_ref: str) -> str:
    """Resolve a user ref once so cache names cannot follow a moving ref."""
    revision = run(["git", "rev-parse", "--verify", f"{baseline_ref}^{{commit}}"]).strip()
    baseline_archive_path(Path("."), revision)
    return revision


def ensure_baseline_archive(session_dir: Path, baseline_ref: str) -> tuple[Path, str]:
    archive_path = baseline_archive_path(session_dir, baseline_ref)
    if not archive_path.exists():
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{archive_path.name}-", dir=archive_path.parent)
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with temporary_path.open("wb") as output:
                archive = subprocess.Popen(["git", "archive", "--format=tar", baseline_ref], cwd=ROOT, stdout=subprocess.PIPE)
                assert archive.stdout is not None
                import gzip

                with gzip.GzipFile(fileobj=output, mode="wb") as compressed:
                    shutil.copyfileobj(archive.stdout, compressed)
                if archive.wait() != 0:
                    raise RuntimeError("git archive failed")
            os.replace(temporary_path, archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return archive_path, hashlib.sha256(archive_path.read_bytes()).hexdigest()


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
    ignored = shutil.ignore_patterns(
        ".git", ".data", ".env", ".env.*", ".opencode", "node_modules", "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache", ".mypy_cache"
    )
    shutil.copytree(source, destination, symlinks=True, ignore=ignored)
    return destination


def tree_sha256(source: Path) -> str:
    """Hash frozen source bytes and names without traversing symlink targets."""
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix().encode()
        if path.is_symlink():
            digest.update(b"link\0" + relative + b"\0" + os.readlink(path).encode())
        elif path.is_file():
            digest.update(b"file\0" + relative + b"\0")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def read_bootstrap_execution_count(workspace_files: Path) -> int:
    """Read the worker-only bootstrap counter from canonical workspace files."""
    count_path = workspace_files / ".ragtime" / "benchmark-bootstrap-count"
    return int(count_path.read_text(encoding="utf-8").strip()) if count_path.is_file() else 0


def frozen_sources(directory: Path, session_dir: Path, baseline_ref: str) -> tuple[Path, Path, Path, dict[str, str]]:
    archive_path, archive_sha = ensure_baseline_archive(session_dir, baseline_ref)
    extracted = directory / "baseline-archive"
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, extracted)
    baseline = snapshot_source(extracted, directory / "baseline")
    candidate = snapshot_source(ROOT, directory / "candidate")
    fixtures = snapshot_source(FIXTURES, directory / "fixtures")
    return (
        baseline,
        candidate,
        fixtures,
        {
            "baseline_archive_sha256": archive_sha,
            "baseline_source_sha256": tree_sha256(baseline),
            "candidate_source_sha256": tree_sha256(candidate),
            "fixture_source_sha256": tree_sha256(fixtures),
        },
    )


def json_lines(command: list[str], output: Path, common: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Persist samples while redirecting unbounded container stderr to scratch."""
    with tempfile.TemporaryDirectory(prefix="userspace-runtime-stderr-", dir=output.parent) as scratch:
        stderr_path = Path(scratch) / "container.stderr"
        with stderr_path.open("w", encoding="utf-8") as stderr_stream:
            process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=stderr_stream)
            assert process.stdout is not None
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
            returncode = process.wait()
        if returncode != 0:
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"benchmark container failed ({returncode}): {stderr[-8000:]}")


def docker_run(
    source: Path,
    fixtures: Path,
    source_label: str,
    args: argparse.Namespace,
    docker: str,
    image_id: str,
    source_hashes: dict[str, str],
    output: Path,
    host_root: Path | None,
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
        f"{fixtures / args.fixture}:/fixture:ro",
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
    runner = _WORKER_RUNNER if args.runner == "worker" else _RUNNER
    if args.runner == "worker":
        command.extend([image_id, "-c", runner, "/bench", args.fixture, args.profile, str(args.samples), str(args.concurrency), args.storage, args.mode])
    else:
        command.extend([image_id, "-c", runner, "/bench", args.mode, args.fixture, args.profile, str(args.samples), str(args.concurrency), args.storage])
    common = {
        "source": source_label,
        "baseline_revision": args.baseline_revision,
        "baseline_ref_requested": args.baseline_ref,
        "runtime_image_requested": args.image,
        "runtime_image_id": image_id,
        "runner": args.runner,
        **source_hashes,
    }
    return list(json_lines(command, output, common))


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(row["startup_to_app_ready_ms"] for row in rows)
    percentile = lambda p: values[round((len(values) - 1) * p)] if values else 0.0
    return {
        "count": len(rows),
        "startup_to_app_ready_ms": {"p50": percentile(0.5), "p95": percentile(0.95), "mean": statistics.fmean(values) if values else 0.0},
        "p95_note": "p95 is screening-only; interpret it only alongside the recorded sample count.",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--baseline-ref", default=BASELINE_REVISION, help="Pinned git revision used for the baseline arm.")
    parser.add_argument("--session-dir", type=Path, default=SESSION, help="Scratch and default-output directory; never mounted as source.")
    parser.add_argument(
        "--runner", choices=("sandbox", "worker"), default="sandbox", help="sandbox measures spawn_sandboxed; worker measures WorkerService startup pipeline."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, help="JSONL sample output; defaults inside --session-dir.")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = args.session_dir / "D-benchmark-raw.jsonl"
    return args


def main() -> int:
    args = parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    if args.samples > 1 and os.environ.get("RAGTIME_BENCHMARK_QUIET_WINDOW") != "1":
        raise SystemExit("set RAGTIME_BENCHMARK_QUIET_WINDOW=1 before a multi-sample run")
    if args.samples == 1 and not args.smoke:
        raise SystemExit("single sample is smoke-only; pass --smoke")
    if args.mode == "pivot_root" and args.capability_profile != "sys-admin":
        raise SystemExit("pivot_root requires --capability-profile sys-admin")
    args.baseline_revision = resolve_baseline_revision(args.baseline_ref)
    docker = docker_binary()
    image_id = run([docker, "image", "inspect", "--format", "{{.Id}}", args.image]).strip()
    if not image_id.startswith("sha256:"):
        raise RuntimeError(f"docker image inspect did not return an immutable image ID: {image_id!r}")
    args.session_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Docker Desktop can mount checkout-local scratch while macOS /var temp
    # paths are not necessarily shared.  Excluding .opencode above prevents
    # a candidate snapshot from recursively copying this destination.
    with tempfile.TemporaryDirectory(prefix="userspace-runtime-benchmark-", dir=args.session_dir) as directory:
        temporary = Path(directory)
        baseline, candidate, fixtures, source_hashes = frozen_sources(temporary, args.session_dir, args.baseline_revision)
        host_runs = temporary / "bind-runs" if args.storage == "bind" else None
        rows: list[dict[str, Any]] = []
        for label, source in (("baseline", baseline), ("candidate", candidate)):
            rows.extend(docker_run(source, fixtures, label, args, docker, image_id, source_hashes, args.output, host_runs / label if host_runs else None))
    measured = {label: [row for row in rows if row["source"] == label] for label in ("baseline", "candidate")}
    print(
        json.dumps(
            {
                "summary": {label: summary(source_rows) for label, source_rows in measured.items()},
                "output": str(args.output),
                "docker": docker,
                "storage_requested": args.storage,
                "runner": args.runner,
                "runtime_image_id": image_id,
                "source_hashes": source_hashes,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
