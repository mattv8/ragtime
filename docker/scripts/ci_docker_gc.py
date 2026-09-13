#!/usr/bin/env python3
"""Conservative garbage collection for Docker resources created by Ragtime CI.

This intentionally knows only the names and labels emitted by managed-buildx.
It never invokes a Docker prune command and treats missing metadata as a reason
to leave a resource for a later run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

STALE_AFTER = dt.timedelta(hours=1)
GIB = 1024**3
BUILDER_RE = re.compile(r"^ragtime-ci-([0-9a-f]{8})-([0-9]+)-([0-9]+)-([0-9a-f]{8})$")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_time(value: object) -> Optional[dt.datetime]:
    """Parse the ISO forms returned by Docker and GitHub, or return None."""
    if not isinstance(value, str) or not value:
        return None
    normalized = re.sub(r"\.(\d{6})\d+(?=(?:Z|[+-]\d\d:\d\d)$)", r".\1", value)
    try:
        parsed = dt.datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(dt.timezone.utc) if parsed.tzinfo is not None else None


def repo_hash(repository: str) -> str:
    return hashlib.sha256(repository.encode("utf-8")).hexdigest()[:8]


def scope_hash(scope: str) -> str:
    return hashlib.sha256(scope.encode("utf-8")).hexdigest()[:8]


def builder_name(repository: str, run_id: str, attempt: str, scope: str) -> str:
    return "ragtime-ci-{}-{}-{}-{}".format(repo_hash(repository), run_id, attempt, scope_hash(scope))


def image_tag(repository: str, run_id: str, attempt: str, scope: str) -> str:
    return "ragtime-ci-{}:{}-{}-{}".format(repo_hash(repository), run_id, attempt, scope_hash(scope))


@dataclass(frozen=True)
class Context:
    repository: str
    run_id: str
    attempt: str
    api_url: str
    token: Optional[str]
    may_mutate: bool

    @classmethod
    def from_env(cls, environ: Optional[dict[str, str]] = None) -> "Context":
        env = os.environ if environ is None else environ
        repository = env.get("GITHUB_REPOSITORY", "")
        run_id = env.get("GITHUB_RUN_ID", "")
        attempt = env.get("GITHUB_RUN_ATTEMPT", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("GITHUB_REPOSITORY must be owner/repository")
        if not run_id.isdecimal() or not attempt.isdecimal():
            raise ValueError("GITHUB_RUN_ID and GITHUB_RUN_ATTEMPT must be numeric")
        api_url = env.get("GITHUB_API_URL", "https://api.github.com")
        parsed = urllib.parse.urlparse(api_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("GITHUB_API_URL must be a clean HTTPS base URL")
        event = env.get("GITHUB_EVENT_NAME", "")
        head_repository = env.get("GITHUB_HEAD_REPOSITORY")
        if event == "pull_request" and not head_repository:
            try:
                with open(env.get("GITHUB_EVENT_PATH", ""), encoding="utf-8") as event_file:
                    payload = json.load(event_file)
                head_repository = payload["pull_request"]["head"]["repo"]["full_name"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                head_repository = None
        same_repo_pr = event == "pull_request" and head_repository == repository
        may_mutate = env.get("GITHUB_ACTIONS") == "true" and (event in {"push", "workflow_dispatch"} or same_repo_pr)
        return cls(repository, run_id, attempt, api_url.rstrip("/"), env.get("GH_TOKEN") or env.get("GITHUB_TOKEN"), may_mutate)


@dataclass(frozen=True)
class Container:
    id: str
    name: str
    image: str
    image_id: str
    created: Optional[dt.datetime]
    mounts: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Volume:
    name: str
    created: Optional[dt.datetime]


@dataclass(frozen=True)
class Image:
    id: str
    tags: tuple[str, ...]
    labels: dict[str, str]
    created: Optional[dt.datetime]


class Docker:
    """Small Docker CLI boundary; tests replace this class with a fake."""

    def __init__(self, repository: str) -> None:
        self.managed_prefix = "buildx_buildkit_ragtime-ci-{}-".format(repo_hash(repository))
        self.repository = repository

    def _run(self, args: Sequence[str]) -> str:
        return subprocess.run(["docker", *args], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout

    @staticmethod
    def _not_found(error: subprocess.CalledProcessError) -> bool:
        return (
            "No such object" in (error.stderr or "")
            or "No such container" in (error.stderr or "")
            or "No such volume" in (error.stderr or "")
            or "No such image" in (error.stderr or "")
        )

    def containers(self) -> list[Container]:
        ids = self._run(["ps", "-aq"]).split()
        if not ids:
            return []
        try:
            items = json.loads(self._run(["inspect", *ids]))
            return [self._container_from_item(item) for item in items]
        except subprocess.CalledProcessError:
            result: list[Container] = []
            for container_id in ids:
                try:
                    result.append(self.inspect_container(container_id))
                except subprocess.CalledProcessError as error:
                    if not self._not_found(error):
                        raise
            return result

    def inspect_container(self, container_id: str) -> Container:
        item = json.loads(self._run(["inspect", container_id]))[0]
        return self._container_from_item(item)

    @staticmethod
    def _container_from_item(item: dict[str, Any]) -> Container:
        mounts = tuple((mount.get("Name", ""), mount.get("Destination", "")) for mount in item.get("Mounts", []))
        return Container(
            item["Id"], item["Name"].lstrip("/"), item.get("Config", {}).get("Image", ""), item.get("Image", ""), parse_time(item.get("Created")), mounts
        )

    def container_present(self, container_id: str) -> bool:
        return bool(self._run(["ps", "-aq", "--no-trunc", "--filter", "id=" + container_id]).strip())

    def volumes(self) -> list[Volume]:
        names = self._run(["volume", "ls", "--filter", "name=" + self.managed_prefix, "-q"]).split()
        result: list[Volume] = []
        for name in names:
            try:
                item = json.loads(self._run(["volume", "inspect", name]))[0]
            except subprocess.CalledProcessError as error:
                if not self._not_found(error):
                    raise
                continue
            result.append(Volume(name, parse_time(item.get("CreatedAt"))))
        return result

    def inspect_volume(self, name: str) -> Volume:
        item = json.loads(self._run(["volume", "inspect", name]))[0]
        return Volume(name, parse_time(item.get("CreatedAt")))

    def images(self) -> list[Image]:
        result: list[Image] = []
        args = ["image", "ls", "--filter", "label=org.ragtime.ci.repository=" + self.repository, "--format", "{{.ID}}"]
        for line in set(self._run(args).splitlines()):
            try:
                item = json.loads(self._run(["image", "inspect", line]))[0]
            except subprocess.CalledProcessError as error:
                if not self._not_found(error):
                    raise
                continue
            result.append(Image(item["Id"], tuple(item.get("RepoTags") or ()), item.get("Config", {}).get("Labels") or {}, parse_time(item.get("Created"))))
        return result

    def inspect_image_tag(self, tag: str) -> Image:
        item = json.loads(self._run(["image", "inspect", tag]))[0]
        return Image(item["Id"], tuple(item.get("RepoTags") or ()), item.get("Config", {}).get("Labels") or {}, parse_time(item.get("Created")))

    def remove_container(self, container_id: str) -> None:
        self._run(["rm", "-f", container_id])

    def remove_volume(self, name: str) -> None:
        self._run(["volume", "rm", name])

    def remove_image_tag(self, tag: str) -> None:
        self._run(["image", "rm", tag])

    def root_dir(self) -> Path:
        return Path(json.loads(self._run(["info", "--format", "{{json .}}"])).get("DockerRootDir", "/var/lib/docker"))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


class GitHubAttempts:
    def __init__(self, context: Context) -> None:
        self.context = context
        self.cache: dict[tuple[str, str], Optional[bool]] = {}

    def terminal_stale(self, run_id: str, attempt: str, now: dt.datetime) -> bool:
        key = (run_id, attempt)
        if key in self.cache:
            return bool(self.cache[key])
        if not self.context.token:
            self.cache[key] = None
            return False
        url = "{}/repos/{}/actions/runs/{}/attempts/{}".format(self.context.api_url, self.context.repository, run_id, attempt)
        request = urllib.request.Request(url, headers={"Authorization": "Bearer " + self.context.token, "Accept": "application/vnd.github+json"})
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=10) as response:  # noqa: S310 - redirects are rejected
                payload: Any = json.load(response)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            self.cache[key] = None
            return False
        updated = parse_time(payload.get("updated_at")) if isinstance(payload, dict) else None
        complete = (
            isinstance(payload, dict)
            and str(payload.get("id")) == run_id
            and str(payload.get("run_attempt")) == attempt
            and payload.get("status") == "completed"
        )
        self.cache[key] = bool(complete and updated and now - updated >= STALE_AFTER)
        return bool(self.cache[key])


class Collector:
    def __init__(self, context: Context, docker: Docker, github: GitHubAttempts, now: Optional[dt.datetime] = None) -> None:
        self.context, self.docker, self.github, self.now = context, docker, github, now or _now()
        self.messages: list[str] = []

    def _stale_attempt(self, run_id: str, attempt: str, created: Optional[dt.datetime]) -> bool:
        return bool(created and self.now - created >= STALE_AFTER and self.github.terminal_stale(run_id, attempt, self.now))

    def _owned_container(self, container: Container) -> Optional[tuple[str, str, str]]:
        match = re.fullmatch(r"buildx_buildkit_(ragtime-ci-[a-z0-9-]+)0", container.name)
        if not match or not re.fullmatch(r"(?:docker\.io/)?moby/buildkit(?:[@:].+)?", container.image):
            return None
        builder = match.group(1)
        parsed = BUILDER_RE.fullmatch(builder)
        if not parsed or parsed.group(1) != repo_hash(self.context.repository):
            return None
        volume = container.name + "_state"
        if (volume, "/var/lib/buildkit") not in container.mounts:
            return None
        return parsed.group(2), parsed.group(3), volume

    def collect(self, apply: bool = False) -> list[str]:
        mutable = apply and self.context.may_mutate
        if apply and not mutable:
            self.messages.append("mutations disabled: untrusted or non-GitHub Actions environment")
        containers = self.docker.containers()
        referenced_images = {container.image_id for container in containers}
        referenced_volumes = {mount[0] for container in containers for mount in container.mounts if mount[0]}
        for container in containers:
            owned = self._owned_container(container)
            if not owned:
                continue
            run_id, attempt, volume = owned
            if (run_id, attempt) == (self.context.run_id, self.context.attempt):
                self.messages.append("keep current builder {}".format(container.name))
            elif not self._stale_attempt(run_id, attempt, container.created):
                self.messages.append("keep unverified builder {}".format(container.name))
            elif mutable:
                try:
                    fresh = self.docker.inspect_container(container.id)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError, KeyError, json.JSONDecodeError):
                    self.messages.append("keep builder {}: reinspection failed".format(container.name))
                    continue
                if fresh != container or not self._owned_container(fresh):
                    self.messages.append("keep builder {}: changed during collection".format(container.name))
                    continue
                self.docker.remove_container(container.id)
                try:
                    if self.docker.container_present(container.id):
                        self.messages.append("keep volume {}: builder still exists".format(volume))
                    elif volume not in {mount[0] for item in self.docker.containers() for mount in item.mounts}:
                        self.docker.remove_volume(volume)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError, KeyError, json.JSONDecodeError):
                    self.messages.append("keep volume {}: builder absence uncertain".format(volume))
                self.messages.append("removed stale builder {}".format(container.name))
            else:
                self.messages.append("would remove stale builder {}".format(container.name))

        for candidate_volume in self.docker.volumes():
            match = re.fullmatch(r"buildx_buildkit_(ragtime-ci-[a-z0-9-]+)0_state", candidate_volume.name)
            if not match or candidate_volume.name in referenced_volumes:
                continue
            parsed = BUILDER_RE.fullmatch(match.group(1))
            if not parsed or parsed.group(1) != repo_hash(self.context.repository):
                continue
            run_id, attempt = parsed.group(2), parsed.group(3)
            if (run_id, attempt) == (self.context.run_id, self.context.attempt) or not self._stale_attempt(run_id, attempt, candidate_volume.created):
                continue
            if mutable:
                try:
                    fresh_volume = self.docker.inspect_volume(candidate_volume.name)
                    attached = {mount[0] for item in self.docker.containers() for mount in item.mounts}
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError, KeyError, json.JSONDecodeError):
                    self.messages.append("keep volume {}: reinspection failed".format(candidate_volume.name))
                    continue
                if fresh_volume == candidate_volume and candidate_volume.name not in attached:
                    self.docker.remove_volume(candidate_volume.name)
                    self.messages.append("removed stale volume {}".format(candidate_volume.name))
                else:
                    self.messages.append("keep volume {}: changed or attached during collection".format(candidate_volume.name))
            else:
                self.messages.append("would remove stale volume {}".format(candidate_volume.name))

        for image in self.docker.images():
            for tag in image.tags:
                match = re.fullmatch(r"ragtime-ci-([0-9a-f]{8}):([0-9]+)-([0-9]+)-([0-9a-f]{8})", tag)
                if not match or match.group(1) != repo_hash(self.context.repository):
                    continue
                run_id, attempt = match.group(2), match.group(3)
                labels = image.labels
                if (
                    labels.get("org.ragtime.ci.repository") != self.context.repository
                    or labels.get("org.ragtime.ci.run-id") != run_id
                    or labels.get("org.ragtime.ci.run-attempt") != attempt
                ):
                    continue
                if (
                    image.id in referenced_images
                    or (run_id, attempt) == (self.context.run_id, self.context.attempt)
                    or not self._stale_attempt(run_id, attempt, image.created)
                ):
                    continue
                if mutable:
                    try:
                        fresh_image = self.docker.inspect_image_tag(tag)
                        fresh_references = {container.image_id for container in self.docker.containers()}
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError, KeyError, json.JSONDecodeError):
                        self.messages.append("keep image tag {}: reinspection failed".format(tag))
                        continue
                    if fresh_image == image and image.id not in fresh_references:
                        self.docker.remove_image_tag(tag)
                        self.messages.append("removed stale image tag {}".format(tag))
                    else:
                        self.messages.append("keep image tag {}: changed or in use during collection".format(tag))
                else:
                    self.messages.append("would remove stale image tag {}".format(tag))
        return self.messages


class Lock:
    def __init__(self, repository: str) -> None:
        directory = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "ragtime-ci-gc"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        self.file = (directory / (repo_hash(repository) + ".lock")).open("a+")

    def acquire(self) -> bool:
        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def close(self) -> None:
        self.file.close()


def disk_space(docker: Docker) -> tuple[int, int]:
    usage = shutil.disk_usage(docker.root_dir())
    return usage.free, usage.total


def ensure_headroom(docker: Docker, minimum_gib: int) -> tuple[int, int]:
    if minimum_gib < 1:
        raise ValueError("minimum free space must be at least 1 GiB")
    free, total = disk_space(docker)
    if free < minimum_gib * GIB:
        raise RuntimeError("Docker filesystem has {:.1f} GiB free; requires {} GiB".format(free / GIB, minimum_gib))
    return free, total


def headroom_report(docker: Docker, free: int, total: int) -> str:
    try:
        inodes = os.statvfs(docker.root_dir()).f_favail
        inode_text = "; {} inodes available".format(inodes)
    except OSError:
        inode_text = ""
    return "Docker filesystem: {:.1f}/{:.1f} GiB free (25 GiB recommended){}".format(free / GIB, total / GIB, inode_text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--scope", required=True)
    for name in ("collect", "report"):
        command = sub.add_parser(name)
        if name == "collect":
            command.add_argument("--apply", action="store_true")
    for command in (prepare, sub.choices["collect"], sub.choices["report"]):
        command.add_argument("--min-free-gib", type=int, default=15)
    args = parser.parse_args(argv)
    try:
        if args.min_free_gib < 1:
            raise ValueError("minimum free space must be at least 1 GiB")
        context = Context.from_env()
        docker = Docker(context.repository)
        if args.command == "prepare":
            if not args.scope or "\n" in args.scope:
                raise ValueError("scope must be a non-empty single line")
            collector = Collector(context, docker, GitHubAttempts(context))
            before = disk_space(docker)
            lock = Lock(context.repository)
            if lock.acquire():
                try:
                    collector.collect(apply=True)
                finally:
                    lock.close()
            else:
                collector.messages.append("collector lock busy; skipped collection")
                lock.close()
            after = disk_space(docker)
            for message in collector.messages:
                print(message, file=sys.stderr)
            print("before collection: " + headroom_report(docker, *before), file=sys.stderr)
            print("after collection: " + headroom_report(docker, *after), file=sys.stderr)
            if after[0] < args.min_free_gib * GIB:
                raise RuntimeError("Docker filesystem has {:.1f} GiB free; requires {} GiB".format(after[0] / GIB, args.min_free_gib))
            print("builder_name=" + builder_name(context.repository, context.run_id, context.attempt, args.scope))
            print("image_tag=" + image_tag(context.repository, context.run_id, context.attempt, args.scope))
            return 0
        collector = Collector(context, docker, GitHubAttempts(context))
        apply = getattr(args, "apply", False)
        if apply:
            lock = Lock(context.repository)
            if lock.acquire():
                try:
                    collector.collect(apply=True)
                finally:
                    lock.close()
            else:
                collector.messages.append("collector lock busy; skipped collection")
                lock.close()
        else:
            collector.collect()
        free, total = disk_space(docker)
        print(headroom_report(docker, free, total), file=sys.stderr)
        for message in collector.messages:
            print(message, file=sys.stderr)
        if args.command == "collect" and free < args.min_free_gib * GIB:
            raise RuntimeError("Docker filesystem has {:.1f} GiB free; requires {} GiB".format(free / GIB, args.min_free_gib))
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print("ci docker gc: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
