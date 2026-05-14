import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest

from ragtime.indexer.tool_health import ToolHeartbeatStatus, ToolHealthMonitor
from ragtime.rag.prompts import build_tool_system_prompt


class ToolHealthMonitorTests(unittest.TestCase):
    def test_missing_heartbeat_is_not_healthy(self) -> None:
        monitor = ToolHealthMonitor(stale_after_seconds=30)

        self.assertFalse(monitor.is_tool_healthy("tool-a"))
        self.assertEqual(monitor.healthy_tool_ids_for_configs([SimpleNamespace(id="tool-a")]), [])
        self.assertEqual(monitor.filter_healthy_tool_config_dicts([{"id": "tool-a"}]), [])

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
            monitor.filter_healthy_tool_config_dicts(
                [{"id": "healthy"}, {"id": "offline"}, {"id": "stale"}]
            ),
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


if __name__ == "__main__":
    unittest.main()
