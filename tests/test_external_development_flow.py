import unittest

from fastapi import HTTPException

from ragtime.userspace.development_service import development_service


class ExternalDevelopmentOperationContractTests(unittest.TestCase):
    def test_registry_exposes_complete_fixed_operation_set(self) -> None:
        operations = {item["name"]: item for item in development_service.list_operations()}
        self.assertEqual(
            set(operations),
            {
                "context",
                "operation_describe",
                "files_list",
                "file_read",
                "file_write",
                "file_patch",
                "file_delete",
                "code_search",
                "snapshot_create",
                "snapshots_list",
                "snapshot_restore",
                "validate",
                "runtime_status",
                "runtime_start",
                "runtime_stop",
                "preview_launch",
                "resources",
                "http_api_catalog_search",
                "execute_component",
                "index_search",
                "index_grants_list",
                "index_grant_create",
                "index_grant_delete",
                "exec_start",
                "exec_list",
                "exec_get",
                "exec_cancel",
            },
        )
        for operation in operations.values():
            self.assertIn("description", operation)
            self.assertIn("scope", operation)
            self.assertEqual(operation["input_schema"]["type"], "object")

    def test_content_hash_is_deterministic(self) -> None:
        self.assertEqual(development_service._hash("hello"), development_service._hash("hello"))
        self.assertNotEqual(development_service._hash("hello"), development_service._hash("goodbye"))

    def test_file_contracts_advertise_the_validated_artifact_type(self) -> None:
        for operation in development_service.list_operations():
            if operation["name"] not in {"file_write", "file_patch"}:
                continue
            with self.subTest(operation=operation["name"]):
                schema = operation["input_schema"]
                artifact_schema = schema["properties"]["artifact_type"]
                allowed: set[str | None] = set()
                for alternative in artifact_schema.get("anyOf", [artifact_schema]):
                    if alternative.get("type") == "null":
                        allowed.add(None)
                    elif "const" in alternative:
                        allowed.add(alternative["const"])
                    else:
                        self.assertIn("enum", alternative, "Artifact type must not advertise arbitrary strings")
                        allowed.update(alternative["enum"])
                self.assertEqual(allowed, {"module_ts", None})
                self.assertIn("$defs", schema)
                self.assertIn("UserSpaceLiveDataConnection", schema["$defs"])

    def test_required_hash_and_types_are_rejected_before_dispatch(self) -> None:
        write = next(item for item in development_service.list_operations() if item["name"] == "file_write")
        with self.assertRaises(HTTPException) as missing:
            development_service._validate_arguments(write["input_schema"], {"path": "a.ts", "content": "x"})
        self.assertEqual(missing.exception.status_code, 422)
        with self.assertRaises(HTTPException) as invalid:
            development_service._validate_arguments(write["input_schema"], {"path": "a.ts", "content": "x", "expected_hash": 1})
        self.assertEqual(invalid.exception.status_code, 422)

    def test_registry_does_not_advertise_placeholder_operations(self) -> None:
        source = open("ragtime/userspace/development_service.py", encoding="utf-8").read()
        self.assertNotIn("General index search is not available", source)

    def test_registry_filters_operations_by_credential_scope(self) -> None:
        operations = development_service.list_operations(frozenset({"read"}))
        self.assertTrue(operations)
        self.assertEqual({operation["scope"] for operation in operations}, {"read"})

    def test_file_write_request_forwards_typed_live_data_contract(self) -> None:
        request = development_service._build_file_write_request(
            content="export const dashboard = true",
            arguments={
                "live_data_requested": True,
                "live_data_connections": [{"component_id": "tool-1", "request": {"query": "select 1"}}],
                "live_data_checks": [{"component_id": "tool-1", "connection_check_passed": True, "transformation_check_passed": True}],
            },
        )
        self.assertTrue(request.live_data_requested)
        assert request.live_data_connections is not None
        assert request.live_data_checks is not None
        self.assertEqual(request.live_data_connections[0].component_id, "tool-1")
        self.assertTrue(request.live_data_checks[0].connection_check_passed)

    def test_file_write_request_rejects_malformed_live_data_contract(self) -> None:
        with self.assertRaises(HTTPException) as invalid:
            development_service._build_file_write_request(
                content="x",
                arguments={"live_data_connections": [{"component_id": "tool-1"}]},
            )
        self.assertEqual(invalid.exception.status_code, 422)
