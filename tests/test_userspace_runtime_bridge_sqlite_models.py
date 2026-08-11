from __future__ import annotations

import unittest

from pydantic import ValidationError

from ragtime.userspace.models import (
    RuntimeBridgeSqliteMutationOperation,
    RuntimeBridgeSqliteMutationRequest,
    RuntimeBridgeSqliteQueryRequest,
)


class RuntimeBridgeSqliteModelTests(unittest.TestCase):
    def test_query_request_forbids_runtime_identity_and_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            RuntimeBridgeSqliteQueryRequest.model_validate(
                {
                    "target_workspace_id": "target-ws",
                    "sql": "select 1",
                    "workspace_id": "source-ws",
                    "session_id": "sess-1",
                    "leased_by_user_id": "user-1",
                    "unexpected": True,
                }
            )

    def test_query_request_rejects_unsupported_database_name(self) -> None:
        with self.assertRaises(ValidationError):
            RuntimeBridgeSqliteQueryRequest.model_validate(
                {
                    "target_workspace_id": "target-ws",
                    "database_name": "other.sqlite3",
                    "sql": "select 1",
                }
            )

    def test_query_request_validates_max_rows_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            RuntimeBridgeSqliteQueryRequest.model_validate(
                {
                    "target_workspace_id": "target-ws",
                    "sql": "select 1",
                    "max_rows": 0,
                }
            )
        with self.assertRaises(ValidationError):
            RuntimeBridgeSqliteQueryRequest.model_validate(
                {
                    "target_workspace_id": "target-ws",
                    "sql": "select 1",
                    "max_rows": 501,
                }
            )

    def test_query_request_rejects_scalar_and_string_parameters(self) -> None:
        for parameters in ("bad", 7, 3.5, True):
            with self.assertRaises(ValidationError):
                RuntimeBridgeSqliteQueryRequest.model_validate(
                    {
                        "target_workspace_id": "target-ws",
                        "sql": "select 1",
                        "parameters": parameters,
                    }
                )

    def test_query_request_accepts_list_and_dict_parameters(self) -> None:
        list_request = RuntimeBridgeSqliteQueryRequest.model_validate(
            {
                "target_workspace_id": "target-ws",
                "sql": "select ?",
                "parameters": [1, "two", None],
            }
        )
        dict_request = RuntimeBridgeSqliteQueryRequest.model_validate(
            {
                "target_workspace_id": "target-ws",
                "sql": "select :value",
                "parameters": {"value": 1},
            }
        )

        self.assertEqual(list_request.parameters, [1, "two", None])
        self.assertEqual(dict_request.parameters, {"value": 1})

    def test_query_request_defaults_database_name_and_max_rows(self) -> None:
        request = RuntimeBridgeSqliteQueryRequest.model_validate(
            {
                "target_workspace_id": "target-ws",
                "sql": "select 1",
            }
        )

        self.assertEqual(request.database_name, "app.sqlite3")
        self.assertEqual(request.max_rows, 200)

    def test_mutation_request_forbids_runtime_identity_and_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            RuntimeBridgeSqliteMutationRequest.model_validate(
                {
                    "target_workspace_id": "target-ws",
                    "operations": [{"kind": "delete", "table": "widgets", "where": {"id": 1}}],
                    "workspace_id": "source-ws",
                    "session_id": "sess-1",
                    "leased_by_user_id": "user-1",
                    "unexpected": True,
                }
            )

    def test_mutation_request_validates_operation_count_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            RuntimeBridgeSqliteMutationRequest.model_validate(
                {
                    "target_workspace_id": "target-ws",
                    "operations": [],
                }
            )
        with self.assertRaises(ValidationError):
            RuntimeBridgeSqliteMutationRequest.model_validate(
                {
                    "target_workspace_id": "target-ws",
                    "operations": [{"kind": "delete", "table": "widgets", "where": {"id": index}} for index in range(501)],
                }
            )

    def test_mutation_operation_accepts_valid_insert_upsert_update_and_delete_shapes(self) -> None:
        insert_operation = RuntimeBridgeSqliteMutationOperation.model_validate({"kind": "insert", "table": "widgets", "values": {"name": "A"}})
        upsert_operation = RuntimeBridgeSqliteMutationOperation.model_validate(
            {
                "kind": "upsert",
                "table": "widgets",
                "values": {"id": 1, "name": "A"},
                "conflict_columns": ["id"],
            }
        )
        update_operation = RuntimeBridgeSqliteMutationOperation.model_validate(
            {
                "kind": "update",
                "table": "widgets",
                "values": {"name": "B"},
                "where": {"id": 1},
            }
        )
        delete_operation = RuntimeBridgeSqliteMutationOperation.model_validate({"kind": "delete", "table": "widgets", "where": {"id": 1}})

        self.assertEqual(insert_operation.kind, "insert")
        self.assertEqual(upsert_operation.conflict_columns, ["id"])
        self.assertEqual(update_operation.where, {"id": 1})
        self.assertEqual(delete_operation.kind, "delete")

    def test_mutation_operation_rejects_malformed_payloads(self) -> None:
        with self.assertRaises(ValidationError):
            RuntimeBridgeSqliteMutationOperation.model_validate(
                {
                    "kind": "insert",
                    "table": "widgets",
                    "where": {"id": 1},
                }
            )
        with self.assertRaises(ValidationError):
            RuntimeBridgeSqliteMutationOperation.model_validate(
                {
                    "kind": "delete",
                    "table": "widgets",
                    "values": {"name": "bad"},
                }
            )

    def test_mutation_operation_rejects_upsert_without_conflict_columns(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            RuntimeBridgeSqliteMutationOperation.model_validate(
                {
                    "kind": "upsert",
                    "table": "widgets",
                    "values": {"id": 1, "name": "A"},
                }
            )

        self.assertIn("conflict_columns", str(raised.exception))

    def test_mutation_operation_rejects_update_without_where(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            RuntimeBridgeSqliteMutationOperation.model_validate(
                {
                    "kind": "update",
                    "table": "widgets",
                    "values": {"name": "A"},
                }
            )

        self.assertIn("where", str(raised.exception))

    def test_mutation_request_rejects_unsupported_database_name(self) -> None:
        with self.assertRaises(ValidationError):
            RuntimeBridgeSqliteMutationRequest.model_validate(
                {
                    "target_workspace_id": "target-ws",
                    "database_name": "custom.sqlite3",
                    "operations": [{"kind": "delete", "table": "widgets", "where": {"id": 1}}],
                }
            )


if __name__ == "__main__":
    unittest.main()
