#!/usr/bin/env python3
"""Switch the local development Compose code mounts between registered worktrees.

This deliberately owns only host orchestration. Migration reconciliation is
performed by worktree_migrations.py in an ephemeral target application image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

CODE_SERVICES = ("ragtime", "runtime", "runtime-s3")
CONTAINERS = {"ragtime": "ragtime-dev", "runtime": "runtime-dev", "runtime-s3": "runtime-s3-dev"}


class SwitchRefusal(RuntimeError):
    """A safe precondition refusal (rather than a failed mutation)."""


class Runner:
    def run(self, args: Sequence[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(list(args), text=True, capture_output=capture, check=check)


@dataclass(frozen=True)
class Worktree:
    path: Path
    head: str
    branch: str
    dirty: bool
    active: bool


def worktree_labels(item: Worktree) -> tuple[str, str]:
    return "dirty" if item.dirty else "clean", "active" if item.active else "inactive"


def git(runner: Runner, root: Path, *args: str) -> str:
    return runner.run(["git", "-C", str(root), *args], capture=True).stdout.strip()


def repository_root(runner: Runner, script: Path) -> Path:
    """Return the primary checkout, even when this launcher lives in a linked tree."""
    common = Path(git(runner, script.parent, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    if common.name != ".git":
        raise SwitchRefusal(f"unexpected Git common directory: {common}")
    return common.parent.resolve()


def worktrees(runner: Runner, root: Path, active_source: str = "") -> list[Worktree]:
    raw = git(runner, root, "worktree", "list", "--porcelain")
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in raw.splitlines() + [""]:
        if not line:
            if current:
                records.append(current)
            current = {}
        elif " " in line:
            key, value = line.split(" ", 1)
            current[key] = value
        else:
            current[line] = ""
    result = []
    for record in records:
        path = Path(record["worktree"]).resolve()
        if not (path / "docker/docker-compose.dev.yml").is_file() or not (path / "prisma/schema.prisma").is_file():
            continue
        branch = record.get("branch", "detached")
        if branch.startswith("refs/heads/"):
            branch = branch.removeprefix("refs/heads/")
        dirty = bool(git(runner, path, "status", "--porcelain"))
        result.append(Worktree(path, record.get("HEAD", "unknown"), branch, dirty, active_source == str(path / "ragtime")))
    return result


def fingerprint_target(target: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((target / "prisma").rglob("*")):
        if path.is_file() and path.name in {"schema.prisma", "migration.sql", "down.sql", "migration_lock.toml"}:
            digest.update(str(path.relative_to(target)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def yaml_quote(value: str) -> str:
    return json.dumps(value)


def write_override(path: Path, primary: Path, target: Path, *, storage: bool) -> None:
    """Replace only bind lists that must stay primary-anchored.

    Target Compose remains the project directory so all code/build relative paths
    resolve from the selected checkout. Named volumes are preserved verbatim.

    The ragtime volume list must mirror docker/docker-compose.dev.yml; verify()
    re-checks only the code and /data mounts afterwards.
    """
    data = primary / ".data"
    env = primary / ".env"
    storage_key_mount = "      - object-storage-key:/run/ragtime-storage-key\n" if storage else ""
    content = (
        "services:\n"
        "  ragtime:\n"
        "    env_file: !override\n      - " + yaml_quote(str(env)) + "\n"
        "    volumes: !override\n"
        f"      - {yaml_quote(str(target / 'ragtime') + ':/ragtime/ragtime')}\n"
        f"      - {yaml_quote(str(target / 'tests') + ':/ragtime/tests:ro')}\n"
        f"      - {yaml_quote(str(target / 'scripts') + ':/ragtime/scripts:ro')}\n"
        "      - ragtime-node-modules:/ragtime/ragtime/frontend/node_modules\n"
        f"      - {yaml_quote(str(target / 'prisma') + ':/ragtime/prisma:ro')}\n"
        f"      - {yaml_quote(str(target / 'runtime') + ':/ragtime/runtime')}\n"
        f"      - {yaml_quote(str(data) + ':/data')}\n"
        f"{storage_key_mount}"
        "      - ${DOCKER_SOCKET_PATH:-/var/run/docker.sock}:/var/run/docker.sock\n"
        "      - ~/.ssh:/root/.ssh:ro\n"
        "      - ./scripts:/docker-scripts:ro\n"
        "  runtime:\n    volumes: !override\n"
        f"      - {yaml_quote(str(target / 'runtime') + ':/runtime/runtime')}\n"
        f"      - {yaml_quote(str(data) + ':/data')}\n"
    )
    if storage:
        content += (
            f"  runtime-s3:\n    volumes: !override\n      - {yaml_quote(str(data) + ':/data')}\n      - object-storage-key:/run/ragtime-storage-key:ro\n"
        )
    path.write_text(content, encoding="utf-8")


class Switcher:
    def __init__(self, runner: Runner, primary: Path, target: Worktree, dry_run: bool) -> None:
        self.runner, self.primary, self.target, self.dry_run = runner, primary, target, dry_run
        self.state_root = primary / ".data/worktree-switch"
        self.run_root = self.state_root / "runs" / str(uuid.uuid4())
        self.override = self.run_root / "compose.override.yml"
        self.lock = self.state_root / "switch.lock"
        self.stopped: list[tuple[str, str]] = []
        self.apply_started = False
        self.project = ""
        self.services: list[str] = []
        self.stop_services: list[str] = []
        self.sources: list[Path] = []
        self.initial_fingerprint = ""
        self.helper_name = f"worktree-switch-{self.run_root.name}"
        self.db_volume = ""
        self.target_config: dict[str, Any] = {}
        self.image_ids: dict[str, str] = {}
        self.api_url = "http://localhost:8000"
        self.vite_url = "http://localhost:8001"

    def compose(self, *, primary: bool = False, override: bool = True) -> list[str]:
        root = self.primary if primary else self.target.path
        compose_file = root / "docker/docker-compose.dev.yml"
        command = [
            "docker",
            "compose",
            "--project-name",
            self.project,
            "--project-directory",
            str(root / "docker"),
            "--env-file",
            str(self.primary / ".env"),
            "--file",
            str(compose_file),
        ]
        if not primary and override:
            command += ["--file", str(self.override)]
        return command

    def output(self, args: Sequence[str]) -> str:
        return self.runner.run(args, capture=True).stdout.strip()

    def inspect(self, container: str) -> dict[str, Any]:
        payload = json.loads(self.output(["docker", "inspect", container]))
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise SwitchRefusal(f"invalid Docker inspect payload for {container}")
        return payload[0]

    def require_local_docker(self) -> None:
        host = os.environ.get("DOCKER_HOST", "")
        if host and not host.startswith("unix://"):
            raise SwitchRefusal("refusing remote DOCKER_HOST; switcher is local-development only")
        context = self.output(["docker", "context", "show"])
        if context not in {"default", "colima"}:
            raise SwitchRefusal(f"refusing non-local Docker context: {context}")

    def validate(self) -> None:
        self.require_local_docker()
        if not (self.primary / ".env").is_file() or not (self.primary / ".data").is_dir():
            raise SwitchRefusal("primary .env and .data must exist")
        common = git(self.runner, self.primary, "rev-parse", "--path-format=absolute", "--git-common-dir")
        target_common = git(self.runner, self.target.path, "rev-parse", "--path-format=absolute", "--git-common-dir")
        if common != target_common:
            raise SwitchRefusal("target is not a registered worktree of this repository")
        if self.target.path not in {item.path for item in worktrees(self.runner, self.primary)}:
            raise SwitchRefusal("target is not currently registered as a Git worktree")
        labels = self.output(["docker", "inspect", "--format", '{{index .Config.Labels "com.docker.compose.project"}}', "ragtime-db-dev"])
        if not labels:
            raise SwitchRefusal("development database is not running; start the existing dev stack first")
        self.project = labels
        if self.output(["docker", "inspect", "--format", "{{.State.Running}}", "ragtime-db-dev"]) != "true":
            raise SwitchRefusal("development database is not running; start the existing dev stack first")
        db_volume = self.output(
            ["docker", "inspect", "--format", '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql"}}{{.Name}}{{end}}{{end}}', "ragtime-db-dev"]
        )
        if db_volume != f"{self.project}_ragtime-db-data":
            raise SwitchRefusal(f"refusing non-development DB volume: {db_volume!r}")
        self.db_volume = db_volume
        app_project = self.output(["docker", "inspect", "--format", '{{index .Config.Labels "com.docker.compose.project"}}', "ragtime-dev"])
        if app_project != self.project:
            raise SwitchRefusal("ragtime-dev is not owned by the existing development Compose project")
        for container in ("ragtime-dev", "runtime-dev"):
            details = self.inspect(container)
            labels = details.get("Config", {}).get("Labels", {})
            networks = details.get("NetworkSettings", {}).get("Networks", {})
            if labels.get("com.docker.compose.project") != self.project or f"{self.project}_default" not in networks:
                raise SwitchRefusal(f"{container} is not on the expected local development Compose network")
        db_url = self.output(["docker", "inspect", "--format", "{{range .Config.Env}}{{println .}}{{end}}", "ragtime-dev"])
        if not any(re.match(r"DATABASE_URL=postgresql(?:\+[^:]+)?://[^@]+@ragtime-db:5432/ragtime(?:\?|$)", line) for line in db_url.splitlines()):
            raise SwitchRefusal("ragtime-dev DATABASE_URL does not point at the existing development database")
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True)
        try:
            base_services = set(self.output([*self.compose(override=False), "config", "--services"]).splitlines())
        except subprocess.CalledProcessError as error:
            raise SwitchRefusal("target Compose file is invalid") from error
        write_override(self.override, self.primary, self.target.path, storage="runtime-s3" in base_services)
        config = json.loads(self.output([*self.compose(), "config", "--format", "json"]))
        available = config.get("services", {})
        if not {"ragtime", "runtime"}.issubset(available):
            raise SwitchRefusal("target Compose must contain ragtime and runtime services")
        target_environment = available["ragtime"].get("environment", [])
        if isinstance(target_environment, dict):
            target_database_url = target_environment.get("DATABASE_URL", "")
        else:
            target_database_url = next((item.removeprefix("DATABASE_URL=") for item in target_environment if item.startswith("DATABASE_URL=")), "")
        if not re.fullmatch(r"postgresql(?:\+[^:]+)?://[^@]+@ragtime-db:5432/ragtime(?:\?[^\s]*)?", str(target_database_url)):
            raise SwitchRefusal("target Compose DATABASE_URL does not point at the existing development database")
        self.services = [name for name in CODE_SERVICES if name in available]
        self.target_config = config
        # Storage is optional in older target Compose files, but an outgoing
        # writer must still be stopped before migration reconciliation.
        self.stop_services = list(self.services)
        if "runtime-s3" not in self.stop_services:
            try:
                if self.output(["docker", "inspect", "--format", "{{.State.Running}}", "runtime-s3-dev"]) == "true":
                    self.stop_services.append("runtime-s3")
            except subprocess.CalledProcessError:
                pass
        self.sources = [item.path for item in worktrees(self.runner, self.primary) if item.path not in {self.primary, self.target.path}]
        self.initial_fingerprint = fingerprint_target(self.target.path)

    def _lock_owner_alive(self, pid_text: str) -> bool:
        if not pid_text.isdigit():
            return True  # An unknown owner is treated as alive; refuse conservatively.
        try:
            os.kill(int(pid_text), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        return True

    def acquire_lock(self) -> None:
        for reclaim_attempted in (False, True):
            try:
                self.lock.mkdir()
                (self.lock / "pid").write_text(str(os.getpid()), encoding="ascii")
                return
            except FileExistsError as error:
                try:
                    owner = (self.lock / "pid").read_text().strip()
                except OSError:
                    owner = ""
                if not reclaim_attempted and owner and not self._lock_owner_alive(owner):
                    print(f"Reclaiming stale switch lock left by exited pid {owner}.", file=sys.stderr)
                    try:
                        # Atomic rename: exactly one concurrent reclaimer wins;
                        # the loser retries mkdir and refuses if beaten to it.
                        self.lock.rename(self.run_root / "reclaimed-switch.lock")
                    except OSError:
                        pass
                    continue
                raise SwitchRefusal(f"another switch owns {self.lock} (pid {owner or 'unknown'})") from error

    def release_lock(self) -> None:
        try:
            owner = (self.lock / "pid").read_text().strip()
        except OSError:
            return
        if owner == str(os.getpid()):
            (self.lock / "pid").unlink(missing_ok=True)
            self.lock.rmdir()

    def helper(self, command: str, plan: Path) -> str:
        mounts = [
            "-v",
            f"{self.primary / 'scripts'}:/switch-tools:ro",
            "-v",
            f"{self.target.path}:/switch-target:ro",
            "-v",
            f"{self.primary}:/switch-primary:ro",
            "-v",
            f"{self.state_root}:/switch-state",
        ]
        for index, source in enumerate(self.sources):
            mounts += ["-v", f"{source / 'prisma'}:/switch-sources/{index}:ro"]
        args = [
            *self.compose(),
            "run",
            "-T",
            "--rm",
            "--name",
            self.helper_name,
            "--no-deps",
            "--entrypoint",
            "python",
            *mounts,
            "ragtime",
            "/switch-tools/worktree_migrations.py",
            command,
            "--plan",
            f"/switch-state/runs/{self.run_root.name}/plan.json",
        ]
        if command == "check":
            args += [
                "--target",
                "/switch-target/prisma",
                "--primary",
                "/switch-primary/prisma",
                "--state-root",
                "/switch-state",
                "--registry",
                "/switch-tools/worktree_down_migrations.json",
            ]
            for index in range(len(self.sources)):
                args += ["--source", f"/switch-sources/{index}"]
        try:
            return self.output(args)
        except subprocess.CalledProcessError as error:
            # `check` deliberately uses exit 2 for a safe migration refusal,
            # while still returning its machine-readable diagnostics on stdout.
            if error.stderr:
                print(error.stderr.strip(), file=sys.stderr)
            if command == "check" and error.stdout:
                return error.stdout.strip()
            raise

    def assert_unchanged(self) -> None:
        if fingerprint_target(self.target.path) != self.initial_fingerprint:
            raise SwitchRefusal("target Prisma files changed during switch; rerun the full command")

    def build(self) -> None:
        self.runner.run([*self.compose(), "build", *self.services])
        image_ids: dict[str, str] = {}
        for service in self.services:
            # Compose's `images` command describes existing containers, which
            # still use the outgoing image until activation recreates them.
            references = {line.strip() for line in self.output([*self.compose(), "config", "--images", service]).splitlines() if line.strip()}
            # `config --images SERVICE` includes dependencies. Select this
            # service's explicit ref or Compose default, never the first line.
            # Compatibility mode uses underscores for implicit image names.
            explicit_image = self.target_config["services"][service].get("image")
            candidates = {str(explicit_image)} if explicit_image else {f"{self.project}-{service}", f"{self.project}_{service}"}
            matching = references & candidates
            if len(matching) != 1:
                raise SwitchRefusal(f"unable to uniquely resolve built image for {service} from {sorted(references)!r}")
            image_id = self.output(["docker", "image", "inspect", "--format", "{{.Id}}", matching.pop()])
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
                raise SwitchRefusal(f"unable to resolve canonical image ID for {service}: {image_id!r}")
            image_ids[service] = image_id
        self.image_ids = image_ids

    def stop_writers(self) -> None:
        for service in self.stop_services:
            container = CONTAINERS[service]
            try:
                if self.output(["docker", "inspect", "--format", "{{.State.Running}}", container]) == "true":
                    container_id = self.output(["docker", "inspect", "--format", "{{.Id}}", container])
                    if not container_id:
                        raise SwitchRefusal(f"unable to identify outgoing {service} container")
                    self.stopped.append((service, container_id))
            except subprocess.CalledProcessError:
                continue
        if self.stopped:
            self.runner.run(["docker", "stop", *[container_id for _, container_id in self.stopped]])

    def activate(self) -> None:
        self.runner.run([*self.compose(), "up", "-d", "--force-recreate", "--no-deps", *self.services])

    def verify(self, plan: Path) -> None:
        self.helper("verify", plan)
        for service in self.services:
            container = CONTAINERS[service]
            inspected = self.inspect(container)
            if not inspected.get("State", {}).get("Running"):
                raise RuntimeError(f"{service} is not running after activation")
            mounts = {item.get("Destination"): item for item in inspected.get("Mounts", [])}
            data_mount = mounts.get("/data")
            if not data_mount or data_mount.get("Source") != str(self.primary / ".data") or not data_mount.get("RW"):
                raise RuntimeError(f"{service} does not have the expected writable primary /data mount")
            if service in {"ragtime", "runtime"}:
                destination = "/ragtime/ragtime" if service == "ragtime" else "/runtime/runtime"
                source = self.target.path / ("ragtime" if service == "ragtime" else "runtime")
                code_mount = mounts.get(destination)
                if not code_mount or code_mount.get("Source") != str(source) or code_mount.get("RW") is not True:
                    raise RuntimeError(f"{service} does not have the expected writable target code mount")
            else:
                image = inspected.get("Image")
                if image != self.image_ids.get(service):
                    raise RuntimeError("runtime-s3 image differs from resolved target image")
        db = self.inspect("ragtime-db-dev")
        if not any(item.get("Name") == self.db_volume and item.get("Destination") == "/var/lib/postgresql" for item in db.get("Mounts", [])):
            raise RuntimeError("development DB volume changed during activation")
        # API needs the app's real healthy status, not merely an HTTP connection.
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self._services_ready():
                return
            time.sleep(1)
        raise RuntimeError("Ragtime API/Vite readiness timed out")

    def _services_ready(self) -> bool:
        try:
            ragtime = self.inspect("ragtime-dev")
            runtime = self.inspect("runtime-dev")
            if any(item.get("State", {}).get("Health", {}).get("Status") != "healthy" for item in (ragtime, runtime)):
                return False
            env = dict(value.split("=", 1) for value in ragtime.get("Config", {}).get("Env", []) if "=" in value)
            scheme = "https" if env.get("ENABLE_HTTPS", "").lower() in {"1", "true", "yes"} else "http"
            self.api_url = f"{scheme}://localhost:{env.get('PORT', '8000')}"
            self.vite_url = f"{scheme}://localhost:{env.get('API_PORT', '8001')}"
            health = json.loads(
                self.output(
                    [
                        "docker",
                        "exec",
                        "ragtime-dev",
                        "python",
                        "-c",
                        f"import urllib.request; print(urllib.request.urlopen('{self.api_url}/health',timeout=2).read().decode())",
                    ]
                )
            )
            if health.get("status") != "healthy":
                return False
            self.runner.run(
                [
                    "docker",
                    "exec",
                    "ragtime-dev",
                    "python",
                    "-c",
                    f"import urllib.request; urllib.request.urlopen('{self.vite_url}',timeout=2)",
                ]
            )
            return True
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return False

    def record_state(self) -> None:
        state = self.state_root / "state.json"
        temporary = state.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": 1,
                    "target": str(self.target.path),
                    "branch": self.target.branch,
                    "project": self.project,
                    "compose": str(self.target.path / "docker/docker-compose.dev.yml"),
                    "override": str(self.override),
                },
                indent=2,
            )
            + "\n"
        )
        temporary.replace(state)

    def run(self) -> None:
        self.validate()
        self.acquire_lock()
        try:
            if not self.dry_run:
                self.build()
            plan = self.run_root / "plan.json"
            check = json.loads(self.helper("check", plan))
            for key in ("reverse", "forward", "warnings"):
                values = check.get(key, [])
                if values:
                    print(f"Migration {key} ({len(values)}): {', '.join(values)}", file=sys.stderr)
            if check.get("status") != "ready":
                raise SwitchRefusal("migration check refused: " + "; ".join(check.get("reasons", [])))
            self.assert_unchanged()
            if self.dry_run:
                print(
                    json.dumps(
                        {"target": str(self.target.path), "branch": self.target.branch, "project": self.project, "plan": check.get("plan"), "dry_run": True}
                    )
                )
                return
            self.stop_writers()
            self.assert_unchanged()
            self.apply_started = True
            self.helper("apply", plan)
            self.assert_unchanged()
            self.activate()
            self.verify(plan)
            self.record_state()
            self.stopped.clear()
            print(f"Switched to {self.target.branch} ({self.target.path}); API {self.api_url}, Vite {self.vite_url}")
        except BaseException:
            self.runner.run(["docker", "rm", "-f", self.helper_name], check=False)
            if self.apply_started:
                try:
                    self.runner.run(["docker", "stop", *[CONTAINERS[service] for service in self.stop_services]], check=False)
                except Exception:
                    pass
                print("Switch crossed migration boundary; code writers remain stopped. Retry the full switch command.", file=sys.stderr)
            elif self.stopped:
                self.runner.run(["docker", "start", *[container_id for _, container_id in self.stopped]], check=False)
            raise
        finally:
            self.release_lock()


def choose(items: list[Worktree]) -> Worktree | None:
    number_width = len(str(len(items)))
    branch_width = max((len(item.branch) for item in items), default=0)
    for number, item in enumerate(items, 1):
        dirty, active = worktree_labels(item)
        print(f"{number:>{number_width}}) {item.branch:<{branch_width}} | {dirty:<5} | {active:<8} | {item.path}")
    try:
        answer = input("Selection (0/q/Esc to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if answer in {"", "0", "q", "Q", "\x1b"}:
        return None
    if not answer.isdigit() or not 1 <= int(answer) <= len(items):
        raise SwitchRefusal(f"invalid selection: {answer}")
    return items[int(answer) - 1]


def select_worktree(candidates: list[Worktree], selection: str) -> Worktree:
    """Match a branch name or absolute worktree path, tolerating ~ and trailing slashes."""
    expanded = os.path.expanduser(selection)
    resolved = str(Path(expanded).resolve()) if os.path.isabs(expanded) else ""
    matches = [item for item in candidates if selection in {item.branch, str(item.path)} or (resolved and resolved == str(item.path))]
    if not matches:
        raise SwitchRefusal(f"unknown selection: {selection}")
    if len(matches) > 1:
        raise SwitchRefusal(f"ambiguous selection: {selection}")
    return matches[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Switch the local Ragtime dev stack to a registered Git worktree.")
    parser.add_argument("selection", nargs="?", help="branch name or absolute registered worktree path")
    parser.add_argument("--list", action="store_true", help="list registered eligible worktrees")
    parser.add_argument("--dry-run", action="store_true", help="check and render a non-disruptive switch")
    args = parser.parse_args(argv)
    runner = Runner()
    primary = repository_root(runner, Path(__file__).resolve())
    active = ""
    try:
        active = runner.run(
            ["docker", "inspect", "--format", '{{range .Mounts}}{{if eq .Destination "/ragtime/ragtime"}}{{.Source}}{{end}}{{end}}', "ragtime-dev"],
            capture=True,
            check=False,
        ).stdout.strip()
    except Exception:
        pass
    candidates = worktrees(runner, primary, active)
    recorded_target = ""
    state_path = primary / ".data/worktree-switch/state.json"
    try:
        recorded_target = str(json.loads(state_path.read_text(encoding="utf-8")).get("target", ""))
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    if args.list:
        for item in candidates:
            state = "state-match" if recorded_target == str(item.path) and item.active else "state-mismatch" if recorded_target == str(item.path) else ""
            dirty, active = worktree_labels(item)
            print(f"{item.branch}\t{item.head[:12]}\t{dirty}\t{active}\t{state}\t{item.path}")
        return 0
    target = None
    if args.selection:
        target = select_worktree(candidates, args.selection)
    else:
        target = choose(candidates)
        if target is None:
            print("Cancelled.")
            return 0
    if recorded_target and not any(str(item.path) == recorded_target and item.active for item in candidates):
        print(f"Warning: state.json records {recorded_target}, but observed Ragtime mounts do not match it.", file=sys.stderr)
    Switcher(runner, primary, target, args.dry_run).run()
    return 0


def _raise_keyboard_interrupt(*_: object) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
        signal.signal(signal.SIGHUP, _raise_keyboard_interrupt)
        raise SystemExit(main())
    except SwitchRefusal as error:
        print(f"Refused: {error}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Switch failed: {error}", file=sys.stderr)
        raise SystemExit(1)
