import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from ragtime.indexer.tool_health import ToolHealthMonitor, ToolHeartbeatStatus
from ragtime.rag.prompts import build_tool_system_prompt


class ToolHealthMonitorTests(unittest.TestCase):
    def test_missing_heartbeat_is_not_healthy(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30)

        self.assertFalse(monitor.is_tool_healthy("tool-a"))
        self.assertEqual(monitor.healthy_tool_ids_for_configs([SimpleNamespace(id="tool-a")]), [])
        self.assertEqual(monitor.filter_healthy_tool_config_dicts([{"id": "tool-a"}]), [])

    def test_read_path_does_not_seed_from_persisted_results(self) -> None:
        # Reads must be pure O(1) lookups: a persisted result alone, without an
        # explicit seed, must NOT make a tool healthy.
        monitor = ToolHealthMonitor(stale_after_seconds=30, cold_start_grace_seconds=600)
        now = datetime.now(timezone.utc)
        config = SimpleNamespace(id="tool-a", last_test_result=True, last_test_at=now, last_test_error=None)

        self.assertEqual(monitor.healthy_tool_ids_for_configs([config]), [])
        self.assertEqual(monitor.filter_healthy_tool_config_dicts([{"id": "tool-a"}]), [])
        self.assertFalse(monitor.is_tool_healthy("tool-a"))

    def test_cold_start_seed_warm_starts_health_after_restart(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30, cold_start_grace_seconds=600)
        now = datetime.now(timezone.utc)
        # last_test_at older than the normal stale window but within the
        # cold-start grace window — the exact restart scenario.
        last_test_at = now - timedelta(seconds=120)
        config = SimpleNamespace(id="tool-a", last_test_result=True, last_test_at=last_test_at, last_test_error=None)

        seeded = asyncio.run(monitor.seed_from_persisted_results([config]))

        self.assertEqual(set(seeded), {"tool-a"})
        self.assertTrue(monitor.is_tool_healthy("tool-a"))
        self.assertEqual(monitor.healthy_tool_ids_for_configs([config]), ["tool-a"])

    def test_cold_start_seed_rejects_results_older_than_grace(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30, cold_start_grace_seconds=600)
        too_old = datetime.now(timezone.utc) - timedelta(seconds=601)
        config = SimpleNamespace(id="tool-a", last_test_result=True, last_test_at=too_old, last_test_error=None)

        seeded = asyncio.run(monitor.seed_from_persisted_results([config]))

        self.assertEqual(seeded, {})
        self.assertFalse(monitor.is_tool_healthy("tool-a"))

    def test_cold_start_seed_does_not_warm_start_failed_results(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30, cold_start_grace_seconds=600)
        now = datetime.now(timezone.utc)
        config = SimpleNamespace(id="tool-a", last_test_result=False, last_test_at=now, last_test_error="Connection refused")

        asyncio.run(monitor.seed_from_persisted_results([config]))

        self.assertFalse(monitor.is_tool_healthy("tool-a"))
        self.assertEqual(monitor.get_unavailable_reason("tool-a"), "Connection refused")

    def test_cold_start_seed_is_idempotent(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30, cold_start_grace_seconds=600)
        now = datetime.now(timezone.utc)
        config = SimpleNamespace(id="tool-a", last_test_result=True, last_test_at=now, last_test_error=None)

        first = asyncio.run(monitor.seed_from_persisted_results([config]))
        second = asyncio.run(monitor.seed_from_persisted_results([config]))

        self.assertEqual(set(first), {"tool-a"})
        self.assertEqual(second, {})

    def test_cold_start_seed_does_not_overwrite_live_status(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30, cold_start_grace_seconds=600)
        now = datetime.now(timezone.utc)
        # A live heartbeat already marked the tool offline before seeding runs.
        monitor._statuses = {
            "tool-a": ToolHeartbeatStatus(tool_id="tool-a", alive=False, error="Connection refused", checked_at=now),
        }
        config = SimpleNamespace(id="tool-a", last_test_result=True, last_test_at=now, last_test_error=None)

        seeded = asyncio.run(monitor.seed_from_persisted_results([config]))

        self.assertEqual(seeded, {})
        self.assertFalse(monitor.is_tool_healthy("tool-a"))

    def test_cold_start_seed_clamps_future_dated_results(self) -> None:
        # A backward host-clock shift can leave last_test_at in the future.
        # The seed must be clamped to <= now so a later live heartbeat still
        # out-ranks it in the newest-wins guard rather than being dropped.
        monitor = ToolHealthMonitor(stale_after_seconds=30, cold_start_grace_seconds=600)
        seed_call_time = datetime.now(timezone.utc)
        future = seed_call_time + timedelta(seconds=300)
        config = SimpleNamespace(id="tool-a", last_test_result=True, last_test_at=future, last_test_error=None)

        asyncio.run(monitor.seed_from_persisted_results([config]))

        seeded = monitor.get_status("tool-a")
        assert seeded is not None
        assert seeded.checked_at is not None
        self.assertLessEqual(seeded.checked_at, datetime.now(timezone.utc))
        self.assertTrue(monitor.is_tool_healthy("tool-a"))

        persisted: list[dict[str, ToolHeartbeatStatus]] = []

        async def persist_noop(statuses: dict[str, ToolHeartbeatStatus]) -> None:
            persisted.append(statuses)

        monitor._persist_statuses = persist_noop  # type: ignore[method-assign]

        live_time = datetime.now(timezone.utc)
        asyncio.run(monitor._store_statuses({"tool-a": ToolHeartbeatStatus(tool_id="tool-a", alive=False, error="Connection refused", checked_at=live_time)}))

        status = monitor.get_status("tool-a")
        assert status is not None
        self.assertFalse(status.provisional)
        self.assertFalse(status.alive)

    def test_live_heartbeat_overwrites_provisional_seed(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30, cold_start_grace_seconds=600)
        seed_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        config = SimpleNamespace(id="tool-a", last_test_result=True, last_test_at=seed_time, last_test_error=None)
        asyncio.run(monitor.seed_from_persisted_results([config]))

        persisted: list[dict[str, ToolHeartbeatStatus]] = []

        async def persist_noop(statuses: dict[str, ToolHeartbeatStatus]) -> None:
            persisted.append(statuses)

        monitor._persist_statuses = persist_noop  # type: ignore[method-assign]

        live_time = datetime.now(timezone.utc)
        asyncio.run(monitor._store_statuses({"tool-a": ToolHeartbeatStatus(tool_id="tool-a", alive=False, error="Connection timeout", checked_at=live_time)}))

        status = monitor.get_status("tool-a")
        assert status is not None
        self.assertFalse(status.provisional)
        self.assertFalse(status.alive)

    def test_only_recent_successful_heartbeats_are_healthy(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30)
        now = datetime.now(timezone.utc)
        monitor._statuses = {
            "healthy": ToolHeartbeatStatus(
                tool_id="healthy",
                alive=True,
                checked_at=now,
            ),
            "offline": ToolHeartbeatStatus(
                tool_id="offline",
                alive=False,
                error="Connection refused",
                checked_at=now,
            ),
            "stale": ToolHeartbeatStatus(
                tool_id="stale",
                alive=True,
                checked_at=now - timedelta(seconds=60),
            ),
        }

        configs = [
            SimpleNamespace(id="healthy"),
            SimpleNamespace(id="offline"),
            SimpleNamespace(id="stale"),
        ]
        self.assertEqual(monitor.healthy_tool_ids_for_configs(configs), ["healthy"])
        self.assertEqual(
            monitor.filter_healthy_tool_config_dicts([{"id": "healthy"}, {"id": "offline"}, {"id": "stale"}]),
            [{"id": "healthy"}],
        )
        self.assertEqual(monitor.get_unavailable_reason("offline"), "Connection refused")
        self.assertEqual(monitor.get_unavailable_reason("stale"), "Heartbeat stale")

    def test_unhealthy_tools_are_excluded_from_prompt_context(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30)
        now = datetime.now(timezone.utc)
        monitor._statuses = {
            "healthy": ToolHeartbeatStatus(
                tool_id="healthy",
                alive=True,
                checked_at=now,
            ),
            "offline": ToolHeartbeatStatus(
                tool_id="offline",
                alive=False,
                error="Connection refused",
                checked_at=now,
            ),
            "missing-heartbeat": ToolHeartbeatStatus(
                tool_id="missing-heartbeat",
                alive=True,
                checked_at=now - timedelta(seconds=60),
            ),
        }

        prompt = build_tool_system_prompt(
            monitor.filter_healthy_tool_config_dicts(
                [
                    {
                        "id": "healthy",
                        "name": "Healthy Docker",
                        "tool_type": "ssh_shell",
                    },
                    {
                        "id": "offline",
                        "name": "Offline Docker",
                        "tool_type": "ssh_shell",
                    },
                    {
                        "id": "missing-heartbeat",
                        "name": "Stale Docker",
                        "tool_type": "ssh_shell",
                    },
                ]
            )
        )

        self.assertIn("Healthy Docker", prompt)
        self.assertNotIn("Offline Docker", prompt)
        self.assertNotIn("Stale Docker", prompt)

    def test_manual_success_refreshes_stale_tool_and_reports_change(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30)
        now = datetime.now(timezone.utc)
        monitor._statuses = {
            "docker-1": ToolHeartbeatStatus(
                tool_id="docker-1",
                alive=True,
                checked_at=now - timedelta(seconds=60),
            )
        }

        persisted_statuses = []

        async def persist_noop(statuses: dict[str, ToolHeartbeatStatus]) -> None:
            persisted_statuses.append(statuses)

        monitor._persist_statuses = persist_noop  # type: ignore[method-assign]

        result = asyncio.run(
            monitor.record_tool_test_result(
                "docker-1",
                success=True,
                message="OK",
                checked_at=now,
            )
        )

        self.assertEqual(result.changed_tool_ids, {"docker-1"})
        self.assertTrue(monitor.is_tool_healthy("docker-1"))
        self.assertIsNone(monitor.get_unavailable_reason("docker-1"))
        self.assertEqual(list(persisted_statuses[0]), ["docker-1"])

    def test_older_heartbeat_does_not_overwrite_newer_manual_test(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30)
        now = datetime.now(timezone.utc)
        monitor._statuses = {
            "ssh-1": ToolHeartbeatStatus(
                tool_id="ssh-1",
                alive=True,
                checked_at=now,
            )
        }

        persisted_statuses = []

        async def persist_noop(statuses: dict[str, ToolHeartbeatStatus]) -> None:
            persisted_statuses.append(statuses)

        monitor._persist_statuses = persist_noop  # type: ignore[method-assign]

        result = asyncio.run(
            monitor._store_statuses(
                {
                    "ssh-1": ToolHeartbeatStatus(
                        tool_id="ssh-1",
                        alive=False,
                        error="Connection timeout",
                        checked_at=now - timedelta(seconds=1),
                    )
                }
            )
        )

        self.assertEqual(result.statuses, {})
        self.assertEqual(result.changed_tool_ids, set())
        self.assertTrue(monitor.is_tool_healthy("ssh-1"))
        self.assertEqual(persisted_statuses, [])

    def test_single_transient_failure_does_not_disable_healthy_tool(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30, failures_before_offline=2)
        now = datetime.now(timezone.utc)
        monitor._statuses = {
            "ssh-1": ToolHeartbeatStatus(
                tool_id="ssh-1",
                alive=True,
                checked_at=now,
            )
        }

        persisted_statuses = []

        async def persist_noop(statuses: dict[str, ToolHeartbeatStatus]) -> None:
            persisted_statuses.append(statuses)

        monitor._persist_statuses = persist_noop  # type: ignore[method-assign]

        result = asyncio.run(
            monitor._store_statuses(
                {
                    "ssh-1": ToolHeartbeatStatus(
                        tool_id="ssh-1",
                        alive=False,
                        error="Connection timeout",
                        checked_at=now + timedelta(seconds=1),
                    )
                }
            )
        )

        self.assertEqual(result.statuses, {})
        self.assertEqual(result.changed_tool_ids, set())
        self.assertTrue(monitor.is_tool_healthy("ssh-1"))
        self.assertEqual(persisted_statuses, [])

    def test_consecutive_failures_disable_healthy_tool(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30, failures_before_offline=2)
        now = datetime.now(timezone.utc)
        monitor._statuses = {
            "ssh-1": ToolHeartbeatStatus(
                tool_id="ssh-1",
                alive=True,
                checked_at=now,
            )
        }

        async def persist_noop(statuses: dict[str, ToolHeartbeatStatus]) -> None:
            del statuses

        monitor._persist_statuses = persist_noop  # type: ignore[method-assign]

        asyncio.run(
            monitor._store_statuses(
                {
                    "ssh-1": ToolHeartbeatStatus(
                        tool_id="ssh-1",
                        alive=False,
                        error="Timeout 1",
                        checked_at=now + timedelta(seconds=1),
                    )
                }
            )
        )
        result = asyncio.run(
            monitor._store_statuses(
                {
                    "ssh-1": ToolHeartbeatStatus(
                        tool_id="ssh-1",
                        alive=False,
                        error="Timeout 2",
                        checked_at=now + timedelta(seconds=2),
                    )
                }
            )
        )

        self.assertEqual(result.changed_tool_ids, {"ssh-1"})
        self.assertFalse(monitor.is_tool_healthy("ssh-1"))
        self.assertEqual(monitor.get_unavailable_reason("ssh-1"), "Timeout 2")


if __name__ == "__main__":
    unittest.main()
