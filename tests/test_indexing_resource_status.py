"""Cached indexing resource status response contract tests."""

import unittest
from datetime import datetime, timezone
from unittest import mock

from fastapi.routing import APIRoute

from ragtime.indexer.models import IndexResourceStatus
from ragtime.indexer.resource_governor import resource_governor
from ragtime.indexer.routes import get_index_resource_status, router


class IndexingResourceStatusTests(unittest.TestCase):
    def test_status_contract_includes_event_loop_lag_and_safe_job_fields(self) -> None:
        status = IndexResourceStatus.model_validate(
            {
                "sampled_at": datetime.now(timezone.utc).isoformat(),
                "stale": False,
                "memory_source": "cgroup_v2",
                "system_available_bytes": 10,
                "container_limit_bytes": 20,
                "container_usage_bytes": 5,
                "application_rss_bytes": 4,
                "effective_budget_bytes": 8,
                "committed_bytes": 2,
                "effective_cpu_capacity": 2,
                "event_loop_lag_ms": 1.5,
                "worker_limit": 4,
                "worker_target": 2,
                "workers_active": 1,
                "workers_live": 1,
                "active_jobs": 1,
                "waiting_jobs": 0,
                "limiting_reason": "none",
                "jobs": [
                    {
                        "job_id": "job-1",
                        "stage": "chunking",
                        "state": "running",
                        "reason": "none",
                        "committed_bytes": 2,
                    }
                ],
            }
        )

        self.assertEqual(status.event_loop_lag_ms, 1.5)
        self.assertEqual(status.jobs[0].job_id, "job-1")

    def test_resource_endpoint_is_registered_before_parameterized_index_routes(self) -> None:
        paths = [route.path for route in router.routes if isinstance(route, APIRoute)]

        self.assertLess(paths.index("/indexes/resources"), paths.index("/indexes/{name}/webhook"))


class IndexingResourceStatusEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_returns_cached_governor_snapshot(self) -> None:
        snapshot = {
            "sampled_at": datetime.now(timezone.utc).isoformat(),
            "stale": True,
            "memory_source": "unavailable",
            "system_available_bytes": None,
            "container_limit_bytes": None,
            "container_usage_bytes": None,
            "application_rss_bytes": None,
            "effective_budget_bytes": 0,
            "committed_bytes": 0,
            "effective_cpu_capacity": 1.0,
            "event_loop_lag_ms": None,
            "worker_limit": 1,
            "worker_target": 1,
            "workers_active": 0,
            "workers_live": 0,
            "active_jobs": 0,
            "waiting_jobs": 0,
            "limiting_reason": "metrics_unavailable",
            "jobs": [],
        }

        with mock.patch.object(resource_governor, "snapshot", return_value=snapshot) as snapshot_method:
            response = await get_index_resource_status(_user=mock.Mock())

        self.assertEqual(response.stale, snapshot["stale"])
        self.assertEqual(response.memory_source, snapshot["memory_source"])
        self.assertEqual(response.event_loop_lag_ms, snapshot["event_loop_lag_ms"])
        self.assertEqual(response.jobs, [])
        snapshot_method.assert_called_once_with()
