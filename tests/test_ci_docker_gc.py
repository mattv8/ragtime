import datetime as dt
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from email.message import Message
from unittest import mock

MODULE_PATH = pathlib.Path(__file__).parents[1] / "docker/scripts/ci_docker_gc.py"
SPEC = importlib.util.spec_from_file_location("ci_docker_gc", MODULE_PATH)
assert SPEC and SPEC.loader
gc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gc
SPEC.loader.exec_module(gc)

NOW = dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc)


class FakeGitHub:
    def __init__(self, result=True):
        self.result = result
        self.calls = []

    def terminal_stale(self, run_id, attempt, now):
        self.calls.append((run_id, attempt))
        return self.result


class FakeDocker:
    def __init__(self, containers=(), volumes=(), images=()):
        self._containers = list(containers)
        self._volumes = list(volumes)
        self._images = list(images)
        self.removed = []

    def containers(self):
        return list(self._containers)

    def volumes(self):
        return list(self._volumes)

    def inspect_volume(self, name):
        for item in self._volumes:
            if item.name == name:
                return item
        raise subprocess.CalledProcessError(1, "volume inspect")

    def images(self):
        return list(self._images)

    def inspect_image_tag(self, tag):
        for item in self._images:
            if tag in item.tags:
                return item
        raise subprocess.CalledProcessError(1, "image inspect")

    def inspect_container(self, container_id):
        for item in self._containers:
            if item.id == container_id:
                return item
        raise subprocess.CalledProcessError(1, "inspect")

    def container_present(self, container_id):
        return any(item.id == container_id for item in self._containers)

    def remove_container(self, container_id):
        self.removed.append(("container", container_id))
        self._containers = [item for item in self._containers if item.id != container_id]

    def remove_volume(self, name):
        self.removed.append(("volume", name))
        self._volumes = [item for item in self._volumes if item.name != name]

    def remove_image_tag(self, tag):
        self.removed.append(("image", tag))


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, *_args):
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


class FakeOpener:
    def __init__(self, result):
        self.result = result

    def open(self, *_args, **_kwargs):
        if isinstance(self.result, BaseException):
            raise self.result
        return FakeResponse(self.result)


def context(**extra):
    env = {"GITHUB_REPOSITORY": "owner/repo", "GITHUB_RUN_ID": "9", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "push"}
    env.update(extra)
    return gc.Context.from_env(env)


def old():
    return NOW - dt.timedelta(hours=2)


def builder(run_id="8", attempt="1"):
    return "ragtime-ci-{}-{}-{}-deadbeef".format(gc.repo_hash("owner/repo"), run_id, attempt)


def container(run_id="8", attempt="1"):
    name = "buildx_buildkit_{}0".format(builder(run_id, attempt))
    return gc.Container("cid", name, "moby/buildkit@sha256:x", "builder-image", old(), ((name + "_state", "/var/lib/buildkit"),))


class CollectorTests(unittest.TestCase):
    def test_apply_removes_only_verified_stale_owned_builder_and_volume(self):
        item = container()
        docker = FakeDocker([item], [gc.Volume(item.name + "_state", old())])
        collector = gc.Collector(context(), docker, FakeGitHub(), NOW)
        collector.collect(apply=True)
        self.assertEqual(docker.removed, [("container", "cid"), ("volume", item.name + "_state")])

    def test_dry_run_has_no_mutations(self):
        docker = FakeDocker([container()])
        messages = gc.Collector(context(), docker, FakeGitHub(), NOW).collect()
        self.assertEqual(docker.removed, [])
        self.assertIn("would remove stale builder", " ".join(messages))

    def test_current_builder_and_foreign_or_bad_mount_are_kept(self):
        current = container("9", "1")
        foreign = gc.Container(
            "bad", "buildx_buildkit_ragtime-ci-ffffffff-8-1-deadbeef0", "moby/buildkit", "builder-image", old(), (("x", "/var/lib/buildkit"),)
        )
        docker = FakeDocker([current, foreign])
        gc.Collector(context(), docker, FakeGitHub(), NOW).collect(apply=True)
        self.assertEqual(docker.removed, [])

    def test_fresh_or_unknown_attempt_is_kept(self):
        fresh = gc.Container("cid", container().name, "moby/buildkit", "builder-image", NOW - dt.timedelta(minutes=30), container().mounts)
        docker = FakeDocker([fresh])
        gc.Collector(context(), docker, FakeGitHub(False), NOW).collect(apply=True)
        self.assertEqual(docker.removed, [])

    def test_race_reinspection_prevents_delete(self):
        item = container()
        docker = FakeDocker([item])
        docker.inspect_container = mock.Mock(return_value=gc.Container(item.id, item.name, item.image, item.image_id, item.created, ()))
        gc.Collector(context(), docker, FakeGitHub(), NOW).collect(apply=True)
        self.assertEqual(docker.removed, [])

    def test_volume_and_image_races_prevent_delete(self):
        volume = gc.Volume(container().name + "_state", old())
        tag = gc.image_tag("owner/repo", "8", "1", "scope")
        labels = {"org.ragtime.ci.repository": "owner/repo", "org.ragtime.ci.run-id": "8", "org.ragtime.ci.run-attempt": "1"}
        image = gc.Image("img", (tag,), labels, old())
        docker = FakeDocker(volumes=[volume], images=[image])
        docker.inspect_volume = mock.Mock(return_value=gc.Volume(volume.name, NOW))
        docker.inspect_image_tag = mock.Mock(return_value=gc.Image("new", (tag,), labels, old()))
        gc.Collector(context(), docker, FakeGitHub(), NOW).collect(apply=True)
        self.assertEqual(docker.removed, [])

    def test_buildkit_prefix_impersonator_is_not_owned(self):
        item = container()
        evil = gc.Container(item.id, item.name, "moby/buildkit-evil:latest", item.image_id, item.created, item.mounts)
        docker = FakeDocker([evil])
        gc.Collector(context(), docker, FakeGitHub(), NOW).collect(apply=True)
        self.assertEqual(docker.removed, [])

    def test_orphan_volume_requires_correct_node_suffix_and_age(self):
        good = gc.Volume(container().name + "_state", old())
        bad = gc.Volume("buildx_buildkit_{}_state".format(builder()), old())
        docker = FakeDocker(volumes=[good, bad])
        gc.Collector(context(), docker, FakeGitHub(), NOW).collect(apply=True)
        self.assertEqual(docker.removed, [("volume", good.name)])

    def test_referenced_or_mislabeled_image_is_kept(self):
        tag = gc.image_tag("owner/repo", "8", "1", "scope")
        image = gc.Image("img", (tag,), {"org.ragtime.ci.repository": "owner/repo", "org.ragtime.ci.run-id": "8"}, old())
        docker = FakeDocker([gc.Container("use", "app", "application", "img", old(), ())], images=[image])
        gc.Collector(context(), docker, FakeGitHub(), NOW).collect(apply=True)
        self.assertEqual(docker.removed, [])

    def test_verified_unreferenced_managed_image_tag_is_removed(self):
        tag = gc.image_tag("owner/repo", "8", "1", "scope")
        labels = {"org.ragtime.ci.repository": "owner/repo", "org.ragtime.ci.run-id": "8", "org.ragtime.ci.run-attempt": "1"}
        docker = FakeDocker(images=[gc.Image("img", (tag,), labels, old())])
        gc.Collector(context(), docker, FakeGitHub(), NOW).collect(apply=True)
        self.assertEqual(docker.removed, [("image", tag)])


class ParsingAndGuardTests(unittest.TestCase):
    def test_parse_time_handles_z_fractional_and_invalid(self):
        self.assertEqual(gc.parse_time("2026-01-02T00:00:00.123Z"), dt.datetime(2026, 1, 2, 0, 0, 0, 123000, tzinfo=dt.timezone.utc))
        self.assertEqual(gc.parse_time("2026-01-02T00:00:00.123456789+02:00"), dt.datetime(2026, 1, 1, 22, 0, 0, 123456, tzinfo=dt.timezone.utc))
        self.assertIsNone(gc.parse_time("2026-01-02T00:00:00"))
        self.assertIsNone(gc.parse_time("not-a-date"))

    def test_fork_events_cannot_mutate(self):
        self.assertFalse(context(GITHUB_EVENT_NAME="pull_request", GITHUB_HEAD_REPOSITORY="fork/repo").may_mutate)
        self.assertFalse(context(GITHUB_ACTIONS="false").may_mutate)

    def test_low_disk_refuses_preflight(self):
        class DiskDocker:
            def root_dir(self):
                return pathlib.Path("/")

        with mock.patch.object(gc.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(10 * gc.GIB, 9 * gc.GIB, gc.GIB // 2)):
            with self.assertRaisesRegex(RuntimeError, "requires 1 GiB"):
                gc.ensure_headroom(DiskDocker(), 1)

    def test_lock_contention_is_reported_as_unavailable(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {"XDG_CACHE_HOME": directory}, clear=False):
            lock = gc.Lock("owner/repo")
            with mock.patch.object(gc.fcntl, "flock", side_effect=BlockingIOError):
                self.assertFalse(lock.acquire())
            lock.close()

    def test_github_failures_and_mismatched_payloads_keep_resources(self):
        attempt = gc.GitHubAttempts(context(GITHUB_TOKEN="token"))
        hdrs_404 = Message()
        hdrs_403 = Message()
        hdrs_429 = Message()
        hdrs_302 = Message()
        failures = [
            urllib.error.HTTPError("https://api.github.com", 404, "missing", hdrs_404, None),
            urllib.error.HTTPError("https://api.github.com", 403, "forbidden", hdrs_403, None),
            urllib.error.HTTPError("https://api.github.com", 429, "limited", hdrs_429, None),
            urllib.error.HTTPError("https://api.github.com", 302, "redirect", hdrs_302, None),
            TimeoutError(),
            b"{not json",
            {"status": "completed", "updated_at": "broken"},
            {"id": 7, "run_attempt": 1, "status": "completed", "updated_at": "2026-01-01T00:00:00Z"},
            {"id": 8, "run_attempt": 2, "status": "completed", "updated_at": "2026-01-01T00:00:00Z"},
            {"id": 8, "run_attempt": 1, "status": "in_progress", "updated_at": "2026-01-01T00:00:00Z"},
            {"id": 8, "run_attempt": 1, "status": "completed", "updated_at": "2026-01-02T00:30:00Z"},
        ]
        for result in failures:
            with self.subTest(result=result), mock.patch.object(gc.urllib.request, "build_opener", return_value=FakeOpener(result)):
                attempt.cache.clear()
                self.assertFalse(attempt.terminal_stale("8", "1", NOW))

    def test_github_matching_terminal_payload_is_accepted(self):
        payload = {"id": 8, "run_attempt": 1, "status": "completed", "updated_at": "2026-01-01T00:00:00Z"}
        attempt = gc.GitHubAttempts(context(GITHUB_TOKEN="token"))
        with mock.patch.object(gc.urllib.request, "build_opener", return_value=FakeOpener(payload)):
            self.assertTrue(attempt.terminal_stale("8", "1", NOW))

    def test_docker_image_removed_between_ls_and_inspect(self):
        """Regression test for Finding 1: image removed after ls but before inspect.

        This tests the race condition where an image is listed by docker image ls,
        but is removed before docker image inspect. The _not_found() predicate must
        catch "No such image" to avoid propagating the error and failing the job.
        """
        docker = gc.Docker("test-repo")

        def failing_run(args: tuple) -> str:
            # Simulate: ls returns an image ID, but inspect fails with "No such image"
            if "image" in args and "inspect" in args:
                error = subprocess.CalledProcessError(1, args)
                error.stderr = "Error: No such image: sha256:deadbeef"
                raise error
            # ls succeeds and returns the image ID
            if "image" in args and "ls" in args:
                return "sha256:deadbeef\n"
            return ""

        with mock.patch.object(docker, "_run", side_effect=failing_run):
            result = docker.images()
            # Should return empty list (image was removed), not raise
            self.assertEqual(result, [])

    def test_docker_not_found_matches_all_resource_types(self):
        """Regression test: _not_found must match all four resource error strings."""
        test_cases = [
            ("No such object: fake", True),
            ("No such container: fake", True),
            ("No such volume: fake", True),
            ("No such image: fake", True),
            ("Error: unknown error", False),
            ("", False),
            (None, False),
        ]
        for stderr, should_match in test_cases:
            error = subprocess.CalledProcessError(1, ["docker"])
            error.stderr = stderr
            with self.subTest(stderr=stderr):
                self.assertEqual(gc.Docker._not_found(error), should_match)


if __name__ == "__main__":
    unittest.main()
