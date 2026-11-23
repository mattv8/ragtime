import asyncio
import threading
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from ragtime.solidworks.host_activity import HostActivity, HostActivitySnapshot, HostBusyError, HostUpdatingError
from ragtime.solidworks.host_updater import SolidworksHostUpdater, parse_host_version


class Activity:
    def __init__(self):
        self.value = HostActivitySnapshot(0, None, False)
        self.busy = False

    def snapshot(self, key):
        return self.value

    async def run_exclusive(self, key, wait_seconds, work):
        if self.busy:
            raise HostBusyError()
        return await work()


class UpdaterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {"id": "one", "tool_type": "solidworks", "enabled": True, "connection_config": {"pairing_id": "pair"}}
        self.tools = [self.config]
        self.version = "1.2.3"
        self.tick = 1000.0
        self.activity = Activity()
        self.status = {
            "reachable": True,
            "installed": True,
            "running": True,
            "instances_reported": True,
            "host_version": "1.0.0",
            "host_features": ["stop-if-idle"],
            "solidworks": {"instances": []},
            "activity": None,
        }
        self.result = {"running": True, "host_version": "1.2.3"}
        self.checks = 0
        self.gates = []
        self.updater = SolidworksHostUpdater(
            status_service_factory=lambda: self,
            installer_factory=lambda: self,
            list_tools=self.list_tools,
            activity=self.activity,  # type: ignore[arg-type]
            clock=lambda: self.tick,
            now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
            artifact=lambda: SimpleNamespace(host_version=self.version) if self.version else None,
        )

    async def list_tools(self):
        return self.tools

    async def host_status(self, config, *, include_pdm):
        self.checks += 1
        self.assertFalse(include_pdm)
        return self.status

    async def install_host(self, config, *, idle_gate_seconds):
        self.gates.append(idle_gate_seconds)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def observe(self, version="1.0.0"):
        self.updater.observe(self.config, version)

    def state(self):
        state = self.updater.state_for("one")
        assert state is not None
        return state

    async def test_observe_and_shape(self):
        self.assertIsNone(parse_host_version("1.2.3-beta"))
        self.assertEqual(parse_host_version("1.2.3"), (1, 2, 3))
        self.observe("bad")
        self.assertIsNone(self.updater.state_for("one"))
        self.observe()
        self.assertEqual(
            set(self.state()), {"enabled", "state", "installed_version", "target_version", "reason", "last_attempt_at", "next_attempt_at", "message"}
        )
        self.assertEqual(self.state()["state"], "pending")
        self.observe("1.3.0")
        self.assertEqual(self.state()["state"], "newer")
        self.observe("1.2.3")
        self.assertEqual(self.state()["state"], "current")
        self.config["connection_config"]["auto_update_host"] = False
        self.observe()
        self.assertEqual(self.state()["state"], "off")
        self.assertFalse(self.state()["enabled"])

    async def test_quiet_and_urgent_gate(self):
        self.observe()
        self.activity.value = HostActivitySnapshot(0, 10, False)
        await self.updater.run_once()
        self.assertEqual(self.state()["reason"], "in_use")
        self.assertEqual(self.checks, 0)
        self.updater.request_now(self.config)
        await self.updater.run_once()
        self.assertEqual(self.gates, [0])
        self.assertEqual(self.state()["state"], "current")

    async def test_status_rules(self):
        cases = [
            ({"running": False}, "not_running"),
            ({"running": False, "session": {"interactive": False}}, "logged_off"),
            ({"instances_reported": False}, "in_use"),
            ({"solidworks": {"instances": [{"busy": True}]}}, "in_use"),
            ({"solidworks": {"instances": [None]}}, "in_use"),
            ({"solidworks": {"instances": [{"busy": False, "mode": "background", "owned": True, "documents": 1}]}}, "documents_open"),
            ({"activity": {"in_flight": 1, "idle_seconds": 500, "pdm_busy": False}}, "in_use"),
            ({"activity": {"in_flight": 0, "idle_seconds": 500, "pdm_busy": True}}, "in_use"),
            ({"activity": {"in_flight": 0, "idle_seconds": 20, "pdm_busy": False}}, "in_use"),
        ]
        original = self.status.copy()
        for change, reason in cases:
            with self.subTest(reason=reason, change=change):
                self.tick += 301
                self.status = {**original, **change}
                self.observe()
                await self.updater.run_once()
                self.assertEqual(self.state()["reason"], reason)
                self.assertEqual(self.gates, [])

    async def test_attach_and_no_feature(self):
        self.observe()
        self.status["solidworks"]["instances"] = [{"busy": False, "mode": "attach", "owned": False, "documents": 2}]
        self.status["host_features"] = []
        await self.updater.run_once()
        self.assertEqual(self.gates, [None])

    async def test_spacing_and_current(self):
        self.observe()
        self.status["running"] = False
        await self.updater.run_once()
        await self.updater.run_once()
        self.assertEqual(self.checks, 1)
        self.tick += 300
        self.status["running"] = True
        self.status["host_version"] = "1.2.3"
        await self.updater.run_once()
        self.assertEqual(self.state()["state"], "current")
        self.assertEqual(self.gates, [])

    async def test_busy_and_backoff(self):
        self.observe()
        self.activity.busy = True
        await self.updater.run_once()
        self.assertEqual(self.state()["reason"], "in_use")
        self.activity.busy = False
        self.tick += 300
        self.result = {"error": {"code": "host_busy"}}
        await self.updater.run_once()
        self.assertEqual(self.state()["reason"], "busy")
        self.assertIsNone(self.state()["next_attempt_at"])
        self.result = {"error": {"code": "install_failed", "message": "safe"}}
        for delay in (1800, 7200, 28800, 86400, 86400):
            self.tick += 300
            await self.updater.run_once()
            self.assertEqual(self.state()["state"], "failed")
            self.assertEqual(self.updater._states["pairing:pair"].next_attempt, self.tick + delay)
            self.assertEqual(self.state()["next_attempt_at"], (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=delay)).isoformat())
            self.tick += delay

    async def test_exception_and_new_target(self):
        self.observe()
        self.result = RuntimeError("private")
        await self.updater.run_once()
        self.assertEqual(self.state()["state"], "failed")
        self.version = "1.2.4"
        self.observe()
        self.assertEqual(self.state()["state"], "pending")
        self.assertIsNone(self.state()["next_attempt_at"])
        self.updater._states["pairing:pair"].state = "updating"
        self.observe("1.2.4")
        self.assertEqual(self.state()["state"], "current")
        self.updater.note_manual_action(self.config, {"installed": False})
        self.assertIsNone(self.updater.state_for("one"))

    async def test_shared_key_and_off(self):
        other = {**self.config, "id": "two", "connection_config": {"pairing_id": "pair", "auto_update_host": False}}
        self.tools.append(other)
        self.observe()
        await self.updater.run_once()
        self.assertEqual(self.state()["state"], "off")
        other["enabled"] = False
        self.observe()
        await self.updater.run_once()
        self.assertEqual(self.gates, [300])

    async def test_deduplication_and_missing_artifact(self):
        self.version = ""
        self.observe()
        self.assertIsNone(self.updater.state_for("one"))
        self.version = "1.2.3"
        self.updater.observe({**self.config, "id": None}, "1.0.0")
        self.assertIsNone(self.updater.state_for("one"))
        self.tools.append({**self.config, "id": "two"})
        self.observe()
        await self.updater.run_once()
        self.assertEqual(self.gates, [300])
        other_state = self.updater.state_for("two")
        assert other_state is not None
        self.assertEqual(other_state["state"], "current")

    async def test_urgent_check_bypass_only_once(self):
        self.observe()
        self.status["running"] = False
        await self.updater.run_once()
        self.updater.request_now(self.config)
        await self.updater.run_once()
        await self.updater.run_once()
        self.assertEqual(self.checks, 2)

    async def test_cancellation_during_install_preserves_urgency(self):
        self.observe()
        self.updater.request_now(self.config)
        started = asyncio.Event()

        async def install(config, *, idle_gate_seconds):
            started.set()
            await asyncio.Event().wait()
            return {}

        self.install_host = install
        task = asyncio.create_task(self.updater.run_once())
        await started.wait()
        self.assertEqual(self.state()["state"], "updating")
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.state()["state"], "pending")
        self.assertEqual(self.state()["reason"], "busy")
        self.assertIsNone(self.state()["next_attempt_at"])
        self.assertTrue(self.updater._states["pairing:pair"].urgent)

    async def test_manual_current_clears_failure_and_urgency(self):
        self.observe()
        self.updater.request_now(self.config)
        self.result = RuntimeError("private")
        await self.updater.run_once()
        self.assertEqual(self.state()["state"], "failed")
        self.updater.note_manual_action(self.config, {"installed": True, "host_version": self.version})
        state = self.updater._states["pairing:pair"]
        self.assertEqual(state.state, "current")
        self.assertEqual(state.failures, 0)
        self.assertFalse(state.urgent)
        self.assertIsNone(state.message)
        self.assertIsNone(state.next_attempt_at)

    async def test_never_downgrade_discovered_newer_version(self):
        self.observe()
        self.status["host_version"] = "2.0.0"
        await self.updater.run_once()
        self.assertEqual(self.state()["state"], "newer")
        self.assertEqual(self.gates, [])

    async def test_missing_instances_reported_is_unsafe(self):
        self.observe()
        del self.status["instances_reported"]
        await self.updater.run_once()
        self.assertEqual(self.state()["reason"], "in_use")
        self.assertEqual(self.gates, [])
        self.tick += 300
        self.status["instances_reported"] = 1
        await self.updater.run_once()
        self.assertEqual(self.state()["reason"], "in_use")
        self.assertEqual(self.gates, [])

    async def test_stop_cancels_install_without_backoff(self):
        self.observe()
        started = asyncio.Event()

        async def install(config, *, idle_gate_seconds):
            started.set()
            await asyncio.Event().wait()
            return {}

        self.install_host = install
        self.updater.start()
        await started.wait()
        task = self.updater._task
        assert task is not None
        await self.updater.stop()
        self.assertTrue(task.cancelled())
        self.assertEqual(self.state()["state"], "pending")
        self.assertEqual(self.state()["reason"], "busy")
        self.assertIsNone(self.state()["next_attempt_at"])

    async def test_observe_ignores_missing_key_and_bad_target(self):
        self.updater.observe({**self.config, "connection_config": {}}, "1.0.0")
        self.assertIsNone(self.updater.state_for("one"))
        self.version = "not-a-version"
        self.observe()
        self.assertIsNone(self.updater.state_for("one"))

    async def test_observe_preserves_updating_and_failure_backoff(self):
        self.observe()
        state = self.updater._states["pairing:pair"]
        state.state = "updating"
        self.observe()
        self.assertEqual(state.state, "updating")
        state.state = "pending"
        self.result = {"error": {"code": "install_failed", "message": "safe"}}
        await self.updater.run_once()
        self.observe()
        self.assertEqual(state.state, "failed")
        self.updater.request_now(self.config)
        self.tick += 300
        await self.updater.run_once()
        self.assertEqual(self.checks, 1)
        self.assertEqual(len(self.gates), 1)

    async def test_loop_retries_after_exception(self):
        calls = 0
        retried = asyncio.Event()

        async def run():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("private")
            retried.set()

        self.updater.run_once = run
        real_sleep = asyncio.sleep

        async def yield_once(_seconds):
            await real_sleep(0)

        with patch("ragtime.solidworks.host_updater.asyncio.sleep", yield_once):
            self.updater.start()
            await retried.wait()
            await self.updater.stop()
        self.assertGreaterEqual(calls, 2)

    async def test_call_during_gate_acquisition_rechecks_documents(self):
        await self._test_call_draining(urgent=True)
        self.assertEqual(self.state()["reason"], "documents_open")
        self.assertEqual(self.checks, 1)
        self.assertEqual(self.gates, [])

    async def test_call_during_gate_acquisition_rechecks_quietness(self):
        await self._test_call_draining(urgent=False)
        self.assertEqual(self.state()["reason"], "in_use")
        self.assertEqual(self.checks, 0)
        self.assertEqual(self.gates, [])

    async def _test_call_draining(self, *, urgent):
        updater_test = self

        class RacingActivity(HostActivity):
            async def run_exclusive(self, key, wait_seconds, work):
                started = asyncio.Event()
                finish = asyncio.Event()

                async def open_document(ticket):
                    started.set()
                    await finish.wait()
                    updater_test.status["solidworks"]["instances"] = [{"busy": False, "mode": "background", "owned": True, "documents": 1}]

                caller = asyncio.create_task(self.run_call(key, 1, open_document))
                await started.wait()

                async def release():
                    while not self.snapshot(key).exclusive:
                        await asyncio.sleep(0.001)
                    finish.set()

                releaser = asyncio.create_task(release())
                try:
                    return await super().run_exclusive(key, wait_seconds, work)
                finally:
                    await caller
                    await releaser

        activity = RacingActivity(poll_seconds=0.001)
        self.updater.activity = activity
        self.observe()
        if urgent:
            self.updater.request_now(self.config)
        await self.updater.run_once()

    async def test_cancelled_caller_keeps_install_gate_until_thread_finishes(self):
        activity = HostActivity(poll_seconds=0.001)
        self.updater.activity = activity
        self.observe()
        started, finish = threading.Event(), threading.Event()

        def blocking():
            started.set()
            finish.wait(2)

        async def install(config, *, idle_gate_seconds):
            await asyncio.to_thread(blocking)
            return {"running": True, "host_version": "1.2.3"}

        self.install_host = install
        task = asyncio.create_task(self.updater.run_once())
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(activity.snapshot("pairing:pair").exclusive)
            self.assertEqual(self.state()["state"], "updating")
            with self.assertRaises(HostUpdatingError):
                await activity.run_call("pairing:pair", 0, lambda ticket: asyncio.sleep(0))
        finally:
            finish.set()

        async def wait_for_release():
            while activity.snapshot("pairing:pair").exclusive:
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_for_release(), 1)
        self.assertEqual(self.state()["state"], "current")

    async def test_loop_start_stop(self):
        calls = 0

        async def run():
            nonlocal calls
            calls += 1
            raise ValueError("private")

        self.updater.run_once = run
        self.updater.start()
        task = self.updater._task
        assert task is not None
        self.updater.start()
        self.assertIs(self.updater._task, task)
        await asyncio.sleep(0)
        self.assertEqual(calls, 1)
        self.assertFalse(task.done())
        await self.updater.stop()
        await self.updater.stop()
