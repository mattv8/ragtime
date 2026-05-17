from __future__ import annotations

import asyncio
import json
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from pydantic import ValidationError

inserted_fake_rag_prompts = "ragtime.rag.prompts" not in sys.modules
if inserted_fake_rag_prompts:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    fake_prompts_module.build_workspace_scm_setup_prompt = lambda *args, **kwargs: ""
    fake_rag_package.prompts = fake_prompts_module
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.core.sql_utils import format_psql_csv_output, format_query_result
from ragtime.indexer.live_visualizations import (
    LiveVisualizationRefreshError,
    build_component_request_from_visualization,
    render_refreshed_visualization,
)
from ragtime.tools.chart import CreateLiveChartInput, create_chart
from ragtime.tools.datatable import CreateLiveDataTableInput, create_datatable
from ragtime.userspace.models import ExecuteComponentRequest, ExecuteComponentResponse
from ragtime.userspace.models import UserSpaceWorkspace
from ragtime.userspace.service import UserSpaceService

if inserted_fake_rag_prompts:
    sys.modules.pop("ragtime.rag", None)
    sys.modules.pop("ragtime.rag.prompts", None)


class _CaptureComponentExecutionService(UserSpaceService):
    def __init__(self) -> None:
        super().__init__()
        self.component_execution_calls: list[dict[str, object]] = []

    async def _execute_component_for_selected_tool_ids(self, **kwargs):  # type: ignore[no-untyped-def]
        self.component_execution_calls.append(dict(kwargs))
        return (
            ExecuteComponentResponse(
                component_id=str(kwargs.get("component_id") or "tool-1"),
                columns=[],
                rows=[],
                row_count=0,
                error="captured",
            ),
            "select * from accounts",
        )


class ChatLiveVisualizationRefreshTests(unittest.TestCase):
    def test_live_datatable_schema_rejects_static_payload(self) -> None:
        with self.assertRaises(ValidationError):
            CreateLiveDataTableInput.model_validate(
                {
                    "title": "Accounts",
                    "columns": ["name"],
                    "data": [["A"]],
                    "data_connection": {
                        "component_id": "tool-1",
                        "request": {"query": "select name from accounts limit 10"},
                    },
                }
            )

    def test_live_datatable_schema_accepts_refresh_metadata(self) -> None:
        payload = CreateLiveDataTableInput.model_validate(
            {
                "title": "Accounts",
                "source_data": {"columns": ["name"], "rows": [["A"]]},
                "data_connection": {
                    "component_id": "tool-1",
                    "request": {"query": "select name from accounts limit 10"},
                },
            }
        )

        self.assertEqual(payload.data_connection["component_id"], "tool-1")

    def test_live_chart_schema_requires_result_mapping(self) -> None:
        with self.assertRaises(ValidationError):
            CreateLiveChartInput.model_validate(
                {
                    "chart_type": "bar",
                    "title": "Revenue",
                    "source_data": {
                        "columns": ["month", "revenue"],
                        "rows": [["Jan", 12]],
                    },
                    "data_connection": {
                        "component_id": "tool-1",
                        "request": {"query": "select month, revenue from sales"},
                    },
                }
            )

    def test_live_chart_schema_accepts_refresh_metadata(self) -> None:
        payload = CreateLiveChartInput.model_validate(
            {
                "chart_type": "bar",
                "title": "Revenue",
                "source_data": {
                    "columns": ["month", "revenue"],
                    "rows": [["Jan", 12]],
                },
                "data_connection": {
                    "component_id": "tool-1",
                    "request": {"query": "select month, revenue from sales"},
                    "result_mapping": {
                        "label_field": "month",
                        "datasets": [{"label": "Revenue", "data_field": "revenue"}],
                    },
                },
            }
        )

        self.assertEqual(payload.data_connection["component_id"], "tool-1")

    def test_live_chart_schema_accepts_dataset_field_alias(self) -> None:
        payload = CreateLiveChartInput.model_validate(
            {
                "chart_type": "bar",
                "title": "Revenue",
                "source_data": {
                    "columns": ["GL Account", "Total Amount Rounded"],
                    "rows": [["Packaging", 42.5]],
                },
                "data_connection": {
                    "component_id": "tool-1",
                    "request": {"query": "select gl_account, total_amount from spend limit 10"},
                    "result_mapping": {
                        "label": "GL Account",
                        "dataset": "Total Amount Rounded",
                    },
                },
            }
        )

        self.assertEqual(payload.data_connection["result_mapping"]["dataset"], "Total Amount Rounded")

    def test_create_datatable_derives_rows_from_source_data(self) -> None:
        output = asyncio.run(
            create_datatable(
                title="Accounts",
                source_data={"columns": ["name", "total"], "rows": [["A", 10]]},
                data_connection={
                    "component_id": "tool-1",
                    "request": {"query": "select name, total from accounts limit 10"},
                },
            )
        )

        payload = json.loads(output)
        self.assertEqual(payload["config"]["columns"], [{"title": "name"}, {"title": "total"}])
        self.assertEqual(payload["config"]["data"], [["A", 10]])

    def test_create_chart_derives_datasets_from_source_data(self) -> None:
        output = asyncio.run(
            create_chart(
                chart_type="bar",
                title="Revenue",
                source_data={
                    "columns": ["month", "revenue"],
                    "rows": [["Jan", "12"], ["Feb", "20.5"]],
                },
                data_connection={
                    "component_id": "tool-1",
                    "request": {"query": "select month, revenue from sales limit 10"},
                    "result_mapping": {
                        "label_field": "month",
                        "datasets": [{"label": "Revenue", "data_field": "revenue"}],
                    },
                },
            )
        )

        payload = json.loads(output)
        self.assertEqual(payload["config"]["data"]["labels"], ["Jan", "Feb"])
        self.assertEqual(payload["config"]["data"]["datasets"][0]["data"], [12, 20.5])

    def test_create_chart_derives_datasets_from_dataset_alias(self) -> None:
        output = asyncio.run(
            create_chart(
                chart_type="bar",
                title="Spend",
                source_data={
                    "columns": ["GL Account", "Total Amount Rounded"],
                    "rows": [["Packaging", "42.5"], ["Freight", 12]],
                },
                data_connection={
                    "component_id": "tool-1",
                    "request": {"query": "select gl_account, total_amount from spend limit 10"},
                    "result_mapping": {
                        "label": "GL Account",
                        "dataset": "Total Amount Rounded",
                    },
                },
            )
        )

        payload = json.loads(output)
        self.assertEqual(payload["config"]["data"]["labels"], ["Packaging", "Freight"])
        self.assertEqual(payload["config"]["data"]["datasets"][0]["data"], [42.5, 12])

    def test_build_component_request_uses_persisted_live_metadata(self) -> None:
        output = json.dumps(
            {
                "__datatable__": True,
                "title": "Accounts",
                "config": {"columns": [], "data": []},
                "data_connection": {
                    "component_id": "tool-1",
                    "request": {"query": "select name from accounts"},
                },
            }
        )

        request = build_component_request_from_visualization(output, "datatable")

        self.assertEqual(request.component_id, "tool-1")
        self.assertEqual(request.request, {"query": "select name from accounts"})

    def test_build_component_request_supports_legacy_source_fields(self) -> None:
        output = json.dumps(
            {
                "__datatable__": True,
                "title": "Accounts",
                "config": {"columns": [], "data": []},
                "data_connection": {
                    "source_tool_config_id": "tool-2",
                    "source_input": {"query": "select id from accounts"},
                },
            }
        )

        request = build_component_request_from_visualization(output, "datatable")

        self.assertEqual(request.component_id, "tool-2")
        self.assertEqual(request.request, {"query": "select id from accounts"})

    def test_chat_component_execution_rejects_unbounded_postgres_request(self) -> None:
        result = asyncio.run(
            UserSpaceService()._execute_postgres_query(
                {},
                "select name from accounts",
                30,
                100,
                require_result_limit=True,
            )
        )

        self.assertEqual(result, "Error: SELECT queries must include a LIMIT clause")

    def test_preview_component_execution_can_skip_limit_requirement(self) -> None:
        result = asyncio.run(
            UserSpaceService()._execute_postgres_query(
                {},
                "select name from accounts",
                30,
                100,
                require_result_limit=False,
                enforce_result_limit=False,
            )
        )

        self.assertEqual(result, "Error: No connection configured")

    def test_postgres_live_component_execution_skips_limit_enforcement(self) -> None:
        with patch(
            "ragtime.userspace.service.enforce_max_results",
            side_effect=AssertionError("live component queries should not be capped"),
        ) as enforce_mock:
            result = asyncio.run(
                UserSpaceService()._execute_postgres_query(
                    {},
                    "select name from accounts limit 100000",
                    30,
                    100,
                    require_result_limit=False,
                    enforce_result_limit=False,
                )
            )

        self.assertEqual(result, "Error: No connection configured")
        enforce_mock.assert_not_called()

    def test_workspace_component_execution_defaults_to_unbounded_live_data(self) -> None:
        service = _CaptureComponentExecutionService()
        now = datetime.now(timezone.utc)
        workspace = UserSpaceWorkspace(
            id="workspace-1",
            name="Workspace",
            owner_user_id="user-1",
            selected_tool_ids=["tool-1"],
            created_at=now,
            updated_at=now,
        )

        asyncio.run(
            service._execute_component_for_workspace(
                workspace,
                ExecuteComponentRequest(
                    component_id="tool-1",
                    request={"query": "select * from accounts"},
                ),
                error_log_prefix="test",
            )
        )

        call = service.component_execution_calls[0]
        self.assertIs(call["require_result_limit"], False)
        self.assertIs(call["enforce_result_limit"], False)

    def test_chat_live_component_execution_defaults_to_unbounded_data(self) -> None:
        service = _CaptureComponentExecutionService()

        asyncio.run(
            service.execute_component_for_selected_tools(
                ["tool-1"],
                ExecuteComponentRequest(
                    component_id="tool-1",
                    request={"query": "select * from accounts"},
                ),
            )
        )

        call = service.component_execution_calls[0]
        self.assertIs(call["require_result_limit"], False)
        self.assertIs(call["enforce_result_limit"], False)

    def test_postgres_csv_formatter_keeps_multiline_description_in_one_row(self) -> None:
        output = format_psql_csv_output(
            'id,description,part_number\n'
            '1,"Widget body\nwith wrapped notes",ABC-100\n'
            '2,Plain description,XYZ-200\n'
        )

        rows, columns = UserSpaceService._parse_query_output(output)

        self.assertEqual(columns, ["id", "description", "part_number"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], 1)
        self.assertEqual(rows[0]["description"], "Widget body\nwith wrapped notes")
        self.assertEqual(rows[0]["part_number"], "ABC-100")
        self.assertEqual(rows[1]["description"], "Plain description")
        self.assertEqual(rows[1]["part_number"], "XYZ-200")

    def test_postgres_json_transport_preserves_nested_jsonb_values(self) -> None:
        output = UserSpaceService._format_postgres_json_transport_output(
            json.dumps(
                {
                    "columns": ["id", "payload"],
                    "rows": [
                        {
                            "id": 1,
                            "payload": {
                                "active": True,
                                "items": ["A", "B"],
                                "nested": {"amount": 12.5},
                            },
                        }
                    ],
                }
            )
        )

        self.assertIsNotNone(output)
        rows, columns = UserSpaceService._parse_query_output(output or "")

        self.assertEqual(columns, ["id", "payload"])
        self.assertEqual(rows[0]["id"], 1)
        self.assertEqual(rows[0]["payload"]["active"], True)
        self.assertEqual(rows[0]["payload"]["items"], ["A", "B"])
        self.assertEqual(rows[0]["payload"]["nested"], {"amount": 12.5})

    def test_postgres_json_transport_wrapper_removes_trailing_semicolon(self) -> None:
        wrapped_query = UserSpaceService._wrap_postgres_query_for_json_transport(
            "select id, payload from accounts;"
        )

        self.assertIn("row_to_json(__ragtime_component_query)", wrapped_query)
        self.assertIn("select id, payload from accounts", wrapped_query)
        self.assertNotIn("accounts;", wrapped_query)

    def test_component_formatter_preserves_large_result_metadata(self) -> None:
        source_rows = [
            {"id": index, "amount": index * 2, "name": f"Customer {index}"}
            for index in range(2500)
        ]
        output = format_query_result(
            source_rows,
            ["id", "amount", "name"],
            max_output_length=None,
            metadata_max_length=None,
            include_ascii=False,
        )

        rows, columns = UserSpaceService._parse_query_output(output)

        self.assertEqual(columns, ["id", "amount", "name"])
        self.assertEqual(len(rows), len(source_rows))
        self.assertEqual(rows[-1]["id"], 2499)
        self.assertEqual(rows[-1]["amount"], 4998)

    def test_default_formatter_still_omits_oversized_metadata_for_tool_display(self) -> None:
        source_rows = [{"id": index, "name": f"Customer {index}"} for index in range(2500)]
        output = format_query_result(source_rows, ["id", "name"])

        self.assertFalse(output.startswith("<!--TABLEDATA:"))

    def test_datatable_refresh_preserves_options_and_replaces_rows(self) -> None:
        output = json.dumps(
            {
                "__datatable__": True,
                "title": "Accounts",
                "description": "Current accounts",
                "config": {
                    "columns": [{"title": "Old"}],
                    "data": [["stale"]],
                    "pageLength": 25,
                    "searching": False,
                    "ordering": False,
                    "paging": False,
                    "info": False,
                },
                "data_connection": {
                    "component_id": "tool-1",
                    "request": {"query": "select name, total from accounts"},
                },
            }
        )
        component_response = ExecuteComponentResponse(
            component_id="tool-1",
            columns=["name", "total"],
            rows=[{"name": "A", "total": 10}, {"name": "B", "total": 20}],
            row_count=2,
        )

        refreshed = json.loads(
            render_refreshed_visualization(
                original_output=output,
                tool_type="datatable",
                component_response=component_response,
            )
        )

        self.assertEqual(refreshed["title"], "Accounts")
        self.assertEqual(refreshed["description"], "Current accounts")
        self.assertEqual(refreshed["config"]["columns"], [{"title": "name"}, {"title": "total"}])
        self.assertEqual(refreshed["config"]["data"], [["A", 10], ["B", 20]])
        self.assertEqual(refreshed["config"]["pageLength"], 25)
        self.assertFalse(refreshed["config"]["searching"])
        self.assertFalse(refreshed["config"]["ordering"])

    def test_datatable_refresh_preserves_column_metadata_when_shape_matches(self) -> None:
        output = json.dumps(
            {
                "__datatable__": True,
                "title": "Accounts",
                "config": {
                    "columns": [
                        {"title": "Customer", "className": "customer-col"},
                        {"title": "Invoice Total", "className": "numeric"},
                    ],
                    "data": [["stale", 0]],
                },
                "data_connection": {
                    "component_id": "tool-1",
                    "request": {"query": "select name, total from accounts"},
                },
            }
        )
        component_response = ExecuteComponentResponse(
            component_id="tool-1",
            columns=["name", "total"],
            rows=[{"name": "A", "total": 10}],
            row_count=1,
        )

        refreshed = json.loads(
            render_refreshed_visualization(
                original_output=output,
                tool_type="datatable",
                component_response=component_response,
            )
        )

        self.assertEqual(
            refreshed["config"]["columns"],
            [
                {"title": "Customer", "className": "customer-col"},
                {"title": "Invoice Total", "className": "numeric"},
            ],
        )
        self.assertEqual(refreshed["config"]["data"], [["A", 10]])

    def test_chart_refresh_uses_result_mapping_and_preserves_dataset_style(self) -> None:
        output = json.dumps(
            {
                "__chart__": True,
                "config": {
                    "type": "bar",
                    "data": {
                        "labels": ["Old"],
                        "datasets": [
                            {
                                "label": "Old Revenue",
                                "data": [1],
                                "backgroundColor": "#2563eb",
                            }
                        ],
                    },
                    "options": {"responsive": True},
                },
                "data_connection": {
                    "component_id": "tool-1",
                    "request": {"query": "select month, revenue from sales"},
                    "result_mapping": {
                        "label_field": "month",
                        "datasets": [{"label": "Revenue", "data_field": "revenue"}],
                    },
                },
            }
        )
        component_response = ExecuteComponentResponse(
            component_id="tool-1",
            columns=["month", "revenue"],
            rows=[{"month": "Jan", "revenue": "12"}, {"month": "Feb", "revenue": "20.5"}],
            row_count=2,
        )

        refreshed = json.loads(
            render_refreshed_visualization(
                original_output=output,
                tool_type="chart",
                component_response=component_response,
            )
        )

        self.assertEqual(refreshed["config"]["data"]["labels"], ["Jan", "Feb"])
        dataset = refreshed["config"]["data"]["datasets"][0]
        self.assertEqual(dataset["label"], "Revenue")
        self.assertEqual(dataset["data"], [12, 20.5])
        self.assertEqual(dataset["backgroundColor"], "#2563eb")
        self.assertEqual(refreshed["config"]["options"], {"responsive": True})

    def test_chart_refresh_requires_result_mapping(self) -> None:
        output = json.dumps(
            {
                "__chart__": True,
                "config": {"type": "line", "data": {"labels": [], "datasets": []}},
                "data_connection": {
                    "component_id": "tool-1",
                    "request": {"query": "select month, revenue from sales"},
                },
            }
        )
        component_response = ExecuteComponentResponse(
            component_id="tool-1",
            columns=["month", "revenue"],
            rows=[{"month": "Jan", "revenue": 12}],
            row_count=1,
        )

        with self.assertRaises(LiveVisualizationRefreshError):
            render_refreshed_visualization(
                original_output=output,
                tool_type="chart",
                component_response=component_response,
            )


if __name__ == "__main__":
    unittest.main()
