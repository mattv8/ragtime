import io
import json
import os
import subprocess
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

from scripts.switch_dev_worktree import (
    Runner,
    Switcher,
    SwitchRefusal,
    Worktree,
    choose,
    fingerprint_target,
    main,
    print_error,
    repository_root,
    select_worktree,
    write_override,
)


class FakeRunner(Runner):
    def __init__(self, primary: Path, target: Path, *, storage: bool = True):
        self.calls: list[list[str]] = []
        self.primary, self.target, self.storage = primary, target, storage
        self.image_refs = {"ragtime": "docker-ragtime", "runtime": "docker-runtime", "runtime-s3": "docker-runtime-s3"}
        self.explicit_images: dict[str, str] = {}
        self.image_ids = {
            "ragtime": "sha256:" + "a" * 64,
            "runtime": "sha256:" + "b" * 64,
            "runtime-s3": "sha256:" + "c" * 64,
        }
        self.old_container_image_ids = {
            "ragtime": "sha256:" + "d" * 64,
            "runtime": "sha256:" + "e" * 64,
            "runtime-s3": "sha256:" + "f" * 64,
        }
        self.actual_container_image_ids = {"runtime-s3": self.image_ids["runtime-s3"]}

    def inspect_payload(self, container: str) -> dict[str, object]:
        mounts = [{"Source": str(self.primary / ".data"), "Destination": "/data", "RW": True}]
        if container == "ragtime-dev":
            mounts.append({"Source": str(self.target / "ragtime"), "Destination": "/ragtime/ragtime", "RW": True})
            return {
                "State": {"Running": True, "Health": {"Status": "healthy"}},
                "Config": {
                    "Env": ["PORT=8123", "API_PORT=8124", "DATABASE_URL=postgresql://ragtime:ragtime_dev@ragtime-db:5432/ragtime"],
                    "Labels": {"com.docker.compose.project": "docker"},
                },
                "NetworkSettings": {"Networks": {"docker_default": {}}},
                "Mounts": mounts,
                "Image": self.old_container_image_ids["ragtime"],
            }
        if container == "runtime-dev":
            mounts.append({"Source": str(self.target / "runtime"), "Destination": "/runtime/runtime", "RW": True})
            return {
                "State": {"Running": True, "Health": {"Status": "healthy"}},
                "Config": {"Env": [], "Labels": {"com.docker.compose.project": "docker"}},
                "NetworkSettings": {"Networks": {"docker_default": {}}},
                "Mounts": mounts,
                "Image": self.old_container_image_ids["runtime"],
            }
        if container == "runtime-s3-dev":
            return {"State": {"Running": True}, "Mounts": mounts, "Image": self.actual_container_image_ids["runtime-s3"]}
        return {"State": {"Running": True}, "Mounts": [{"Name": "docker_ragtime-db-data", "Destination": "/var/lib/postgresql", "RW": True}]}

    def run(self, args: Sequence[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        args_list = list(args)
        self.calls.append(args_list)
        joined = " ".join(args_list)
        text = ""
        if args_list[:3] == ["docker", "context", "show"]:
            text = "default"
        elif args_list[:2] == ["git", "-C"]:
            if "--git-common-dir" in args_list:
                text = str(self.primary / ".git")
            elif "worktree list" in joined:
                text = ""
        elif " config --format json" in joined:
            services: dict[str, dict[str, object]] = {
                "ragtime": {"environment": ["DATABASE_URL=postgresql://ragtime:ragtime_dev@ragtime-db:5432/ragtime"]},
                "runtime": {},
            }
            if self.storage:
                services["runtime-s3"] = {}
            for service, reference in self.explicit_images.items():
                services[service]["image"] = reference
            text = json.dumps({"services": services})
        elif " config --services" in joined:
            text = "ragtime\nruntime" + ("\nruntime-s3" if self.storage else "")
        elif " config --images " in f" {joined} ":
            text = self.image_refs[args_list[-1]]
            if args_list[-1] == "ragtime":
                text += "\n" + self.image_refs["runtime"] + "\npgvector/pgvector:pg18\nghcr.io/searxng/searxng:latest"
        elif args_list[:3] == ["docker", "image", "inspect"]:
            ref = args_list[-1]
            service = next(service for service, image_ref in self.image_refs.items() if image_ref == ref)
            text = self.image_ids[service]
        elif " images -q " in f" {joined} ":
            # Compose reads the outgoing containers, not the newly built refs,
            # and emits the ID without Docker inspect's sha256 prefix.
            text = self.old_container_image_ids[args_list[-1]].removeprefix("sha256:")
        elif args_list[:2] == ["docker", "inspect"] and len(args_list) == 3:
            text = json.dumps([self.inspect_payload(args_list[-1])])
        elif " inspect " in f" {joined} ":
            container = args_list[-1]
            if "com.docker.compose.project" in joined:
                text = "docker"
            elif ".Config.Env" in joined:
                text = "DATABASE_URL=postgresql://ragtime:ragtime_dev@ragtime-db:5432/ragtime\n"
            elif "/var/lib/postgresql" in joined:
                text = "docker_ragtime-db-data"
            elif "{{.Id}}" in joined:
                text = f"id-{container}"
            elif "{{.State.Running}}" in joined:
                text = "true"
        elif "worktree_migrations.py check" in joined:
            text = json.dumps(
                {
                    "status": "ready",
                    "reasons": [],
                    "warnings": ["forward has down"],
                    "reverse": ["old"],
                    "forward": ["new"],
                    "plan": "/switch-state/runs/x/plan.json",
                }
            )
        elif "worktree_migrations.py" in joined:
            text = "{}"
        elif "urllib.request" in joined:
            text = json.dumps({"status": "healthy"})
        return subprocess.CompletedProcess(args_list, returncode=0, stdout=text, stderr="")


class SwitchTests(unittest.TestCase):
    def make_tree(self, *, storage: bool = True):
        temp = tempfile.TemporaryDirectory()
        primary, target = Path(temp.name) / "primary", Path(temp.name) / "target space"
        for item in (primary, target):
            (item / "docker").mkdir(parents=True)
            (item / "prisma/migrations/one").mkdir(parents=True)
            (item / "docker/docker-compose.dev.yml").write_text("services:\n  ragtime:\n  runtime:\n" + ("  runtime-s3:\n" if storage else ""))
            (item / "prisma/schema.prisma").write_text("generator client {}")
            (item / "prisma/migrations/one/migration.sql").write_text("SELECT 1;")
            (item / "ragtime").mkdir()
            (item / "runtime").mkdir()
            (item / "tests").mkdir()
            (item / "scripts").mkdir()
        (primary / ".data").mkdir()
        (primary / ".env").write_text("X=y")
        return temp, primary, target

    def switch(self, primary, target, *, dry_run=False, storage=True):
        runner = FakeRunner(primary, target, storage=storage)
        item = Worktree(target, "h", "branch", False, False)
        patcher = mock.patch("scripts.switch_dev_worktree.worktrees", return_value=[item])
        return runner, Switcher(runner, primary, item, dry_run), patcher

    def test_success_orders_build_check_stop_apply_recreate_and_verify(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target)
        with patcher:
            switch.run()
        calls = [" ".join(call) for call in runner.calls]
        positions = [
            next(i for i, call in enumerate(calls) if token in call)
            for token in (
                " build ",
                "worktree_migrations.py check",
                "docker stop",
                "worktree_migrations.py apply",
                " up -d --force-recreate",
                "worktree_migrations.py verify",
            )
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue((primary / ".data/worktree-switch/state.json").is_file())
        self.assertTrue(any("localhost:8123/health" in call for call in calls))
        self.assertTrue(any("localhost:8124" in call for call in calls))
        self.assertEqual(switch.api_url, "http://localhost:8123")
        self.assertEqual(switch.vite_url, "http://localhost:8124")

    def test_build_resolves_recreated_image_not_outgoing_container_image(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target)
        with patcher:
            switch.run()
        self.assertEqual(switch.image_ids["runtime-s3"], runner.image_ids["runtime-s3"])
        self.assertNotEqual(switch.image_ids["runtime-s3"], runner.old_container_image_ids["runtime-s3"])
        calls = [" ".join(call) for call in runner.calls]
        self.assertTrue(any(" config --images runtime-s3" in call for call in calls))
        self.assertFalse(any(" images -q " in f" {call} " for call in calls))

    def test_verify_rejects_actual_recreated_image_mismatch(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target)
        runner.actual_container_image_ids["runtime-s3"] = "sha256:" + "f" * 64
        with patcher, self.assertRaisesRegex(RuntimeError, "runtime-s3 image differs"):
            switch.run()

    def test_build_accepts_explicit_image_reference(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target)
        runner.image_refs["runtime-s3"] = "registry.example/runtime-s3:stable@sha256:" + "a" * 64
        runner.explicit_images["runtime-s3"] = runner.image_refs["runtime-s3"]
        with patcher:
            switch.run()
        self.assertEqual(switch.image_ids["runtime-s3"], runner.image_ids["runtime-s3"])

    def test_build_resolves_requested_service_among_dependency_images(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target)
        with patcher:
            switch.run()
        self.assertEqual(switch.image_ids, runner.image_ids)

    def test_build_resolves_compose_compatibility_image_names(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target)
        runner.image_refs = {service: f"docker_{service}" for service in runner.image_refs}
        with patcher:
            switch.run()
        self.assertEqual(switch.image_ids, runner.image_ids)

    def test_unresolved_target_image_refuses_before_migrations_or_stop(self):
        for reference, image_id in (("", "sha256:" + "a" * 64), ("first\nsecond", "sha256:" + "a" * 64), ("docker-runtime-s3", "abc123")):
            with self.subTest(reference=reference, image_id=image_id):
                temp, primary, target = self.make_tree()
                self.addCleanup(temp.cleanup)
                runner, switch, patcher = self.switch(primary, target)
                runner.image_refs["runtime-s3"] = reference
                runner.image_ids["runtime-s3"] = image_id
                with patcher, self.assertRaises(SwitchRefusal):
                    switch.run()
                calls = [" ".join(call) for call in runner.calls]
                self.assertFalse(any("docker stop" in call or "worktree_migrations.py apply" in call or " up " in call for call in calls))

    def test_optional_storage_is_not_emitted_or_activated(self):
        temp, primary, target = self.make_tree(storage=False)
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target, dry_run=True, storage=False)
        with patcher:
            switch.run()
        override = switch.override.read_text()
        self.assertNotIn("runtime-s3:", override)
        self.assertNotIn("object-storage-key", override)
        self.assertNotIn("runtime-s3", switch.services)

    def test_override_mounts_target_runtime_and_shared_storage_key(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        override = primary / "compose.override.yml"

        write_override(override, primary, target, storage=True)

        content = override.read_text()
        self.assertIn(json.dumps(f"{target / 'runtime'}:/ragtime/runtime"), content)
        self.assertIn("      - object-storage-key:/run/ragtime-storage-key\n", content)
        self.assertIn("      - object-storage-key:/run/ragtime-storage-key:ro\n", content)

    def test_dry_run_never_builds_stops_or_applies(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target, dry_run=True)
        with patcher:
            switch.run()
        calls = [" ".join(call) for call in runner.calls]
        self.assertTrue(any("worktree_migrations.py check" in call for call in calls))
        self.assertFalse(any(token in call for call in calls for token in (" build ", "docker stop", " apply")))

    def test_apply_failure_stops_writers_and_removes_unique_helper(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target)
        original = switch.helper
        switch.helper = lambda command, plan: (_ for _ in ()).throw(RuntimeError("failed")) if command == "apply" else original(command, plan)
        with patcher, self.assertRaisesRegex(RuntimeError, "failed"):
            switch.run()
        calls = [" ".join(call) for call in runner.calls]
        self.assertTrue(any("docker rm -f worktree-switch-" in call for call in calls))
        self.assertTrue(any("docker stop ragtime-dev runtime-dev runtime-s3-dev" in call for call in calls))

    def test_stale_down_file_refuses_before_stop(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target)
        original = switch.helper

        def stale(command, plan):
            result = original(command, plan)
            if command == "check":
                (target / "prisma/migrations/one/down.sql").write_text("DROP TABLE x;")
            return result

        switch.helper = stale
        with patcher, self.assertRaises(SwitchRefusal):
            switch.run()
        self.assertFalse(any("docker stop" in " ".join(call) for call in runner.calls))

    def test_env_guard_refuses_non_development_url(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target)
        original = runner.run

        def wrong_url(args, **kwargs):
            result = original(args, **kwargs)
            if ".Config.Env" in " ".join(args):
                result.stdout = "DATABASE_URL=postgresql://x@production:5432/ragtime\n"
            return result

        runner.run = wrong_url
        with patcher, self.assertRaises(SwitchRefusal):
            switch.validate()

    def test_cancelled_selector_and_down_fingerprint(self):
        item = Worktree(Path("/tmp/a"), "h", "a", False, False)
        with mock.patch("builtins.input", return_value="q"):
            self.assertIsNone(choose([item]))
        temp, _, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        before = fingerprint_target(target)
        (target / "prisma/migrations/one/down.sql").write_text("DROP TABLE x;")
        self.assertNotEqual(before, fingerprint_target(target))

    def test_selector_columns_align_and_selection_maps_to_choice(self):
        items = [
            Worktree(Path("/tmp/one"), "h", "a", False, True),
            Worktree(Path("/tmp/two"), "h", "long-branch-name", True, False),
            *[Worktree(Path(f"/tmp/{number}"), "h", "mid", False, False) for number in range(3, 11)],
        ]
        with mock.patch("builtins.input", return_value="10"), mock.patch("builtins.print") as printed:
            self.assertIs(choose(items), items[9])
        rows = [call.args[0] for call in printed.call_args_list]
        separators = [[index for index, character in enumerate(row) if character == "|"] for row in rows]
        self.assertEqual(len(rows), len(items))
        self.assertEqual(len(separators[0]), 3)
        self.assertTrue(all(offsets == separators[0] for offsets in separators))
        for row, item in zip(rows, items):
            self.assertIn(item.branch, row)
            self.assertTrue(row.endswith(str(item.path)))

    def test_choose_empty_choices_cancels_without_crashing(self):
        with mock.patch("builtins.input", return_value="q"):
            self.assertIsNone(choose([]))

    def test_stale_lock_from_dead_process_is_reclaimed(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target, dry_run=True)
        switch.lock.mkdir(parents=True)
        (switch.lock / "pid").write_text("54321")
        with patcher, mock.patch("scripts.switch_dev_worktree.os.kill", side_effect=ProcessLookupError):
            switch.run()
        self.assertFalse(switch.lock.exists())
        self.assertTrue((switch.run_root / "reclaimed-switch.lock" / "pid").is_file())

    def test_lock_owned_by_live_process_refuses(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target, dry_run=True)
        switch.lock.mkdir(parents=True)
        (switch.lock / "pid").write_text("54321")
        with patcher, mock.patch("scripts.switch_dev_worktree.os.kill", return_value=None):
            with self.assertRaisesRegex(SwitchRefusal, "another switch owns"):
                switch.run()
        self.assertTrue(switch.lock.exists())

    def test_select_worktree_distinguishes_unknown_ambiguous_and_normalizes_paths(self):
        one = Worktree(Path("/x/one"), "h", "alpha", False, False)
        two = Worktree(Path("/x/two"), "h", "beta", False, False)
        duplicate = Worktree(Path("/x/three"), "h", "alpha", False, False)
        self.assertIs(select_worktree([one, two], "/x/one/"), one)
        self.assertIs(select_worktree([one, two], "beta"), two)
        with mock.patch.dict("os.environ", {"HOME": "/x"}):
            self.assertIs(select_worktree([one, two], "~/one"), one)
        with self.assertRaisesRegex(SwitchRefusal, "unknown selection"):
            select_worktree([one, two], "missing")
        with self.assertRaisesRegex(SwitchRefusal, "ambiguous selection"):
            select_worktree([one, duplicate], "alpha")

    def test_invalid_target_compose_is_a_clean_refusal(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, patcher = self.switch(primary, target, dry_run=True)
        original = runner.run

        def broken_compose(args, **kwargs):
            if " config --services" in " ".join(args):
                raise subprocess.CalledProcessError(15, args)
            return original(args, **kwargs)

        runner.run = broken_compose
        with patcher, self.assertRaisesRegex(SwitchRefusal, "target Compose file is invalid"):
            switch.run()

    def test_verify_retries_starting_health_until_healthy(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, _ = self.switch(primary, target)
        original_inspect = switch.inspect
        calls = {"ragtime-dev": 0, "runtime-dev": 0}

        def inspect(container):
            result = original_inspect(container)
            if container in {"ragtime-dev", "runtime-dev"}:
                calls[container] += 1
                result["State"]["Health"]["Status"] = "starting" if calls[container] == 2 else "healthy"
            return result

        switch.inspect = inspect
        switch.services = ["ragtime", "runtime"]
        switch.image_ids = runner.image_ids
        switch.db_volume = "docker_ragtime-db-data"
        with mock.patch("scripts.switch_dev_worktree.time.sleep") as sleep:
            switch.verify(Path("plan.json"))
        sleep.assert_called_once_with(1)

    def test_verify_times_out_when_health_never_ready(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner, switch, _ = self.switch(primary, target)
        original_inspect = switch.inspect

        def inspect(container):
            result = original_inspect(container)
            if container in {"ragtime-dev", "runtime-dev"}:
                result["State"]["Health"]["Status"] = "starting"
            return result

        switch.inspect = inspect
        switch.services = ["ragtime", "runtime"]
        switch.db_volume = "docker_ragtime-db-data"
        ticks = iter([0, 1, 91])
        with (
            mock.patch("scripts.switch_dev_worktree.time.monotonic", side_effect=lambda: next(ticks)),
            mock.patch("scripts.switch_dev_worktree.time.sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "readiness timed out"),
        ):
            switch.verify(Path("plan.json"))
        sleep.assert_called_once_with(1)

    def test_repository_root_uses_common_dir(self):
        runner = mock.Mock()
        runner.run.return_value.stdout = "/tmp/primary/.git\n"
        self.assertEqual(repository_root(runner, Path("/tmp/linked/scripts/tool.py")), Path("/tmp/primary").resolve())

    def test_migration_refusal_labels_active_and_target_code(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        runner = FakeRunner(primary, target)
        outgoing = Worktree(primary, "outgoing-head", "outgoing-branch", False, True)
        selected = Worktree(target, "target-head", "target-branch", False, False)
        switch = Switcher(runner, primary, selected, True, outgoing=outgoing)
        switch.validate = mock.Mock()
        switch.acquire_lock = mock.Mock()
        switch.release_lock = mock.Mock()
        switch.helper = mock.Mock(return_value=json.dumps({"status": "refused", "reasons": ["missing required down migration"], "warnings": []}))

        with self.assertRaisesRegex(
            SwitchRefusal,
            r"active code: outgoing-branch; target code: target-branch.*missing required down migration",
        ):
            switch.run()

    def test_main_passes_observed_active_worktree_to_switcher(self):
        primary = Path("/tmp/primary")
        active = Worktree(primary, "active-head", "active", False, True)
        target = Worktree(Path("/tmp/target"), "target-head", "target", False, False)
        runner = mock.Mock()
        runner.run.return_value.stdout = str(primary / "ragtime")
        switcher = mock.Mock()
        with (
            mock.patch("scripts.switch_dev_worktree.Runner", return_value=runner),
            mock.patch("scripts.switch_dev_worktree.repository_root", return_value=primary),
            mock.patch("scripts.switch_dev_worktree.worktrees", return_value=[active, target]),
            mock.patch("scripts.switch_dev_worktree.Switcher", return_value=switcher) as switcher_class,
        ):
            self.assertEqual(main(["target"]), 0)
        switcher_class.assert_called_once_with(runner, primary, target, False, outgoing=active)
        switcher.run.assert_called_once_with()

    def test_migration_refusal_uses_detached_and_unknown_code_labels(self):
        temp, primary, target = self.make_tree()
        self.addCleanup(temp.cleanup)
        selected = Worktree(target, "target-head", "target", False, False)
        switch = Switcher(FakeRunner(primary, target), primary, selected, True, outgoing=Worktree(primary, "abcdef1234567890", "detached", False, True))
        switch.validate = mock.Mock()
        switch.acquire_lock = mock.Mock()
        switch.release_lock = mock.Mock()
        switch.helper = mock.Mock(return_value=json.dumps({"status": "refused", "reasons": [], "warnings": []}))
        with self.assertRaisesRegex(SwitchRefusal, r"active code: detached@abcdef123456; target code: target"):
            switch.run()

        unknown = Switcher(FakeRunner(primary, target), primary, selected, True)
        unknown.validate = mock.Mock()
        unknown.acquire_lock = mock.Mock()
        unknown.release_lock = mock.Mock()
        unknown.helper = mock.Mock(return_value=json.dumps({"status": "refused", "reasons": [], "warnings": []}))
        with self.assertRaisesRegex(SwitchRefusal, r"active code: unknown; target code: target"):
            unknown.run()

    def test_print_error_uses_color_only_for_supported_terminal_stderr(self):
        class Stderr(io.StringIO):
            def __init__(self, tty: bool):
                super().__init__()
                self.tty = tty

            def isatty(self):
                return self.tty

        for tty, environment, expected in (
            (True, {"TERM": "xterm"}, "\033[1;31merror\033[0m\n"),
            (False, {"TERM": "xterm"}, "error\n"),
            (True, {"TERM": "xterm", "NO_COLOR": "1"}, "error\n"),
            (True, {"TERM": "dumb"}, "error\n"),
        ):
            with self.subTest(tty=tty, environment=environment):
                stderr = Stderr(tty)
                with mock.patch("sys.stderr", stderr), mock.patch.dict(os.environ, environment, clear=True):
                    print_error("error")
                self.assertEqual(stderr.getvalue(), expected)
