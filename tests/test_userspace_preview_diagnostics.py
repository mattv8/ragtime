from __future__ import annotations

import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest import mock

inserted_fake_rag_prompts = "ragtime.rag.prompts" not in sys.modules
if inserted_fake_rag_prompts:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace import service as userspace_service_module
from ragtime.userspace.models import UserSpacePreviewDiagnosticEvent
from ragtime.userspace.service import UserSpaceService

if inserted_fake_rag_prompts:
    sys.modules.pop("ragtime.rag", None)
    sys.modules.pop("ragtime.rag.prompts", None)

from ragtime.rag.prompts import build_userspace_diagnostics_turn_reminder_line


class _FakeDiagnosticModel:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], SimpleNamespace] = {}
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []

    async def find_unique(self, *, where):  # type: ignore[no-untyped-def]
        key = where["workspaceId_diagnosticKey"]
        return self.rows.get((key["workspaceId"], key["diagnosticKey"]))

    async def create(self, *, data):  # type: ignore[no-untyped-def]
        row = SimpleNamespace(**data)
        self.rows[(data["workspaceId"], data["diagnosticKey"])] = row
        self.created.append(data)
        return row

    async def update(self, *, where, data):  # type: ignore[no-untyped-def]
        key = where["workspaceId_diagnosticKey"]
        row = self.rows[(key["workspaceId"], key["diagnosticKey"])]
        for field, value in data.items():
            setattr(row, field, value)
        self.updated.append(data)
        return row

    async def find_many(self, *, where, order, take):  # type: ignore[no-untyped-def]
        rows = [row for row in self.rows.values() if row.workspaceId == where["workspaceId"]]
        updated_filter = where.get("updatedAt", {})
        if "gte" in updated_filter:
            rows = [row for row in rows if row.updatedAt >= updated_filter["gte"]]
        rows.sort(key=lambda row: row.updatedAt, reverse=order.get("updatedAt") == "desc")
        return rows[:take]

    async def delete_many(self, *, where):  # type: ignore[no-untyped-def]
        workspace_id = where["workspaceId"]
        keep_ids = set(where["id"]["notIn"])
        self.rows = {key: row for key, row in self.rows.items() if row.workspaceId != workspace_id or getattr(row, "id", None) in keep_ids}


class UserSpacePreviewDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_records_aggregated_preview_fetch_without_query_payloads(self) -> None:
        service = UserSpaceService()
        model = _FakeDiagnosticModel()
        fake_db = SimpleNamespace(workspacepreviewdiagnostic=model)

        event = UserSpacePreviewDiagnosticEvent(
            kind="preview_fetch",
            method="GET",
            target="https://example.com/api/orders/123?secret=token#frag",
            elapsed_ms=250.4,
            status_code=200,
        )

        with mock.patch.object(userspace_service_module, "get_db", mock.AsyncMock(return_value=fake_db)):
            await service.record_workspace_preview_diagnostic_events("workspace-1", [event])

        self.assertEqual(len(model.created), 1)
        created = model.created[0]
        self.assertEqual(created["workspaceId"], "workspace-1")
        self.assertEqual(created["kind"], "preview_fetch")
        self.assertEqual(created["method"], "GET")
        self.assertEqual(created["targetLabel"], "GET example.com/api/orders/:id")
        self.assertNotIn("secret", created["diagnosticKey"])
        self.assertEqual(created["count"], 1)
        self.assertEqual(created["lastMs"], 250)
        self.assertEqual(created["avgMs"], 250)
        self.assertEqual(created["maxMs"], 250)
        self.assertEqual(created["lastStatusCode"], 200)

    async def test_url_credentials_are_not_stored_in_target_label(self) -> None:
        service = UserSpaceService()
        model = _FakeDiagnosticModel()
        fake_db = SimpleNamespace(workspacepreviewdiagnostic=model)
        event = UserSpacePreviewDiagnosticEvent(
            kind="preview_fetch",
            method="GET",
            target="https://user:secret@example.com:8443/api/orders/123?token=hidden",
            elapsed_ms=10,
        )

        with mock.patch.object(userspace_service_module, "get_db", mock.AsyncMock(return_value=fake_db)):
            await service.record_workspace_preview_diagnostic_events("workspace-1", [event])

        created = model.created[0]
        self.assertEqual(created["targetLabel"], "GET example.com:8443/api/orders/:id")
        self.assertNotIn("user", created["targetLabel"])
        self.assertNotIn("secret", created["diagnosticKey"])

    async def test_updates_running_average_and_last_error(self) -> None:
        service = UserSpaceService()
        model = _FakeDiagnosticModel()
        now = datetime.now(timezone.utc)
        model.rows[("workspace-1", "preview_fetch:GET:example.com/api/orders/:id")] = SimpleNamespace(
            id="diag-1",
            workspaceId="workspace-1",
            kind="preview_fetch",
            diagnosticKey="preview_fetch:GET:example.com/api/orders/:id",
            targetLabel="GET example.com/api/orders/:id",
            method="GET",
            componentId=None,
            count=2,
            errorCount=0,
            lastMs=100,
            avgMs=150,
            maxMs=200,
            lastError=None,
            lastStatusCode=200,
            lastRowCount=None,
            createdAt=now,
            updatedAt=now,
        )
        fake_db = SimpleNamespace(workspacepreviewdiagnostic=model)

        event = UserSpacePreviewDiagnosticEvent(
            kind="preview_fetch",
            method="GET",
            target="https://example.com/api/orders/456",
            elapsed_ms=900,
            status_code=500,
            error="HTTP 500 Internal Server Error with lots of detail" * 20,
        )

        with mock.patch.object(userspace_service_module, "get_db", mock.AsyncMock(return_value=fake_db)):
            await service.record_workspace_preview_diagnostic_events("workspace-1", [event])

        self.assertEqual(len(model.updated), 1)
        updated = model.updated[0]
        self.assertEqual(updated["count"], 3)
        self.assertEqual(updated["errorCount"], 1)
        self.assertEqual(updated["lastMs"], 900)
        self.assertEqual(updated["avgMs"], 400)
        self.assertEqual(updated["maxMs"], 900)
        self.assertLessEqual(len(updated["lastError"]), 240)

    async def test_http_error_status_records_last_error_without_payload(self) -> None:
        service = UserSpaceService()
        model = _FakeDiagnosticModel()
        fake_db = SimpleNamespace(workspacepreviewdiagnostic=model)

        event = UserSpacePreviewDiagnosticEvent(
            kind="preview_xhr",
            method="POST",
            target="/api/save",
            elapsed_ms=100,
            status_code=500,
        )

        with mock.patch.object(userspace_service_module, "get_db", mock.AsyncMock(return_value=fake_db)):
            await service.record_workspace_preview_diagnostic_events("workspace-1", [event])

        created = model.created[0]
        self.assertEqual(created["errorCount"], 1)
        self.assertEqual(created["lastError"], "HTTP 500")

    async def test_lists_prompt_summary_sorted_by_signal(self) -> None:
        service = UserSpaceService()
        model = _FakeDiagnosticModel()
        now = datetime.now(timezone.utc)
        stale = now - timedelta(days=2)
        model.rows[("workspace-1", "component_execute:tool-1")] = SimpleNamespace(
            id="slow",
            workspaceId="workspace-1",
            kind="component_execute",
            diagnosticKey="component_execute:tool-1",
            targetLabel="component tool-1",
            method=None,
            componentId="tool-1",
            count=5,
            errorCount=0,
            lastMs=1200,
            avgMs=2400,
            maxMs=9100,
            lastError=None,
            lastStatusCode=None,
            lastRowCount=184,
            createdAt=now,
            updatedAt=now,
        )
        model.rows[("workspace-1", "preview_xhr:POST:/api/save")] = SimpleNamespace(
            id="error",
            workspaceId="workspace-1",
            kind="preview_xhr",
            diagnosticKey="preview_xhr:POST:/api/save",
            targetLabel="POST /api/save",
            method="POST",
            componentId=None,
            count=2,
            errorCount=1,
            lastMs=820,
            avgMs=700,
            maxMs=820,
            lastError="HTTP 500",
            lastStatusCode=500,
            lastRowCount=None,
            createdAt=now,
            updatedAt=now,
        )
        model.rows[("workspace-1", "preview_fetch:GET:/old")] = SimpleNamespace(
            id="stale",
            workspaceId="workspace-1",
            kind="preview_fetch",
            diagnosticKey="preview_fetch:GET:/old",
            targetLabel="GET /old",
            method="GET",
            componentId=None,
            count=99,
            errorCount=99,
            lastMs=9999,
            avgMs=9999,
            maxMs=9999,
            lastError="old",
            lastStatusCode=500,
            lastRowCount=None,
            createdAt=stale,
            updatedAt=stale,
        )
        fake_db = SimpleNamespace(workspacepreviewdiagnostic=model)

        with mock.patch.object(userspace_service_module, "get_db", mock.AsyncMock(return_value=fake_db)):
            summary = await service.list_workspace_preview_diagnostic_summary("workspace-1")

        self.assertEqual([item.diagnostic_key for item in summary], ["component_execute:tool-1", "preview_xhr:POST:/api/save"])

    def test_turn_reminder_formatter_shows_only_longest_target_and_error_count(self) -> None:
        now = datetime.now(timezone.utc)
        line = build_userspace_diagnostics_turn_reminder_line(
            [
                {
                    "kind": "component_execute",
                    "target_label": "component sales_query",
                    "count": 8,
                    "last_ms": 1200,
                    "avg_ms": 2400,
                    "max_ms": 9100,
                    "last_error": None,
                    "last_status_code": None,
                    "last_row_count": 184,
                    "updated_at": now,
                },
                {
                    "kind": "preview_fetch",
                    "target_label": "GET /api/orders/:id",
                    "count": 5,
                    "last_ms": 820,
                    "avg_ms": 650,
                    "max_ms": 1700,
                    "last_error": "HTTP 500",
                    "last_status_code": 500,
                    "last_row_count": None,
                    "updated_at": now,
                },
            ]
        )

        self.assertIn("Preview diagnostics are advisory aggregates", line)
        self.assertIn("Use the userspace_diagnostics tool for full execution times and errors", line)
        self.assertIn("component sales_query count=8 avg=2.4s max=9.1s last=1.2s rows=184", line)
        self.assertIn("1 target with recent errors", line)
        self.assertNotIn("GET /api/orders/:id count=5", line)
        self.assertLess(len(line), 360)
