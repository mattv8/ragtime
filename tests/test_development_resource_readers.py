import asyncio
import json
import unittest
from contextlib import nullcontext
from typing import Any, Mapping, cast
from unittest.mock import AsyncMock, call, patch

from fastapi import HTTPException

from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_bootstrap import content_hash, read_resource_contract, read_resources
from ragtime.userspace.development_service import DevelopmentService


def _principal() -> DevelopmentPrincipal:
    return DevelopmentPrincipal(user_id="user-1", is_admin=False, auth_source="user")


def _read_resources_with_dynamic_arguments(context: Mapping[str, object], arguments: Mapping[str, object]) -> None:
    read_resources(context, **cast(Any, arguments))


class DevelopmentResourceReaderTests(unittest.TestCase):
    def test_resource_summaries_are_searchable_before_paging_without_changing_pins(self) -> None:
        resource = {
            "name": "Accounting replica",
            "component_id": "tool-accounting",
            "description": "Configured reporting source",
            "tool_type": "http_api",
        }
        context = {
            "capabilities": {
                "authorized_tools": [{"name": "Other", "component_id": "tool-other"}, resource],
                "authorized_indexes": [],
            }
        }

        page = read_resources(context, query="ACCOUNTING", limit=1)

        item = page["items"][0]
        self.assertEqual(item["name"], resource["name"])
        self.assertEqual(item["component_id"], resource["component_id"])
        self.assertEqual(item["contract_reference"]["sha256"], content_hash(resource))
        contract = read_resource_contract(context, resource_id=item["resource_id"], revision=item["contract_reference"]["sha256"])
        self.assertEqual(contract["sha256"], content_hash(resource))

    def test_oversized_identity_values_keep_hash_pinned_contract_references(self) -> None:
        resource = {"name": "n" * 600, "component_id": "tool-" + "x" * 600, "description": "description"}
        page = read_resources({"capabilities": {"authorized_tools": [resource], "authorized_indexes": []}})

        item = page["items"][0]
        self.assertNotIn("name", item)
        self.assertNotIn("component_id", item)
        for key in ("name_reference", "component_id_reference"):
            self.assertEqual(item[key]["reader"], "workspace_development_resource_contract")
            self.assertEqual(item[key]["sha256"], content_hash(resource))

    def test_service_shaped_indexes_expose_index_name_without_changing_identity(self) -> None:
        index = {"name": "workspace_code", "source_type": "workspace_code"}
        context = {"capabilities": {"authorized_tools": [], "authorized_indexes": [index]}}

        item = read_resources(context)["items"][0]

        self.assertEqual(item["index_name"], "workspace_code")
        self.assertEqual(item["resource_id"], "index:" + content_hash("workspace_code"))
        self.assertEqual(item["contract_reference"]["sha256"], content_hash(index))

    def test_identity_summary_uses_serialized_multibyte_and_control_character_budget(self) -> None:
        inline_name = ("λ" * 200) + ("\x00" * 18)
        oversized_name = f"{inline_name}\x00"
        self.assertLessEqual(len(json.dumps(inline_name, ensure_ascii=False, separators=(",", ":")).encode("utf-8")), 512)
        self.assertGreater(len(json.dumps(oversized_name, ensure_ascii=False, separators=(",", ":")).encode("utf-8")), 512)
        context = {
            "capabilities": {
                "authorized_tools": [
                    {"name": inline_name, "tool_type": "http_api"},
                    {"name": oversized_name, "tool_type": "http_api"},
                ],
                "authorized_indexes": [],
            }
        }

        inline, oversized = read_resources(context)["items"]
        self.assertEqual(inline["name"], inline_name)
        self.assertNotIn("name", oversized)
        self.assertTrue(oversized["name_reference"]["complete"])

    def test_invalid_page_arguments_are_rejected_without_coercion(self) -> None:
        context: dict[str, object] = {"capabilities": {"authorized_tools": [], "authorized_indexes": []}}
        for arguments in ({"offset": -1}, {"offset": True}, {"limit": 0}, {"limit": 101}, {"limit": "20"}, {"query": 1}):
            with self.subTest(arguments=arguments), self.assertRaises(HTTPException) as raised:
                _read_resources_with_dynamic_arguments(context, arguments)
            self.assertEqual(raised.exception.status_code, 422)

    def test_catalog_accessor_uses_protection_and_audit_without_context_loading(self) -> None:
        service = DevelopmentService()
        principal = _principal()
        resources = {"tools": [{"component_id": "tool-1"}], "indexes": [{"name": "workspace_code"}]}

        with (
            patch("ragtime.userspace.development_service.content_protection_service.current_context", return_value=object()),
            patch("ragtime.userspace.development_service.authorize_external_content", new_callable=AsyncMock) as authorize,
            patch.object(service, "_workspace", new_callable=AsyncMock) as workspace,
            patch.object(service, "_resources", new_callable=AsyncMock, return_value=resources) as catalog,
            patch.object(service, "_audit_operation", new_callable=AsyncMock) as audit,
            patch("ragtime.userspace.development_service.planning_service.get_workspace_context", new_callable=AsyncMock) as full_context,
        ):
            result = asyncio.run(service.get_resource_catalog(principal, "workspace-1"))

        self.assertEqual(result, {"authorized_tools": resources["tools"], "authorized_indexes": resources["indexes"]})
        workspace.assert_awaited_once_with(principal, "workspace-1", "read")
        catalog.assert_awaited_once_with(principal, "workspace-1")
        audit.assert_awaited_once_with(principal, "workspace-1", "resources_catalog", "read")
        self.assertEqual(authorize.await_count, 2)
        self.assertEqual(
            authorize.await_args_list,
            [
                call(
                    {"workspace_id": "workspace-1", "operation": "resources_catalog", "arguments": {}},
                    direction="inbound",
                    principal=principal,
                    surface="development",
                    tool_id="resources_catalog",
                    resource_id="workspace-1",
                    operation="resources_catalog",
                ),
                call(
                    result,
                    direction="outbound",
                    principal=principal,
                    surface="development",
                    tool_id="resources_catalog",
                    resource_id="workspace-1",
                    operation="resources_catalog",
                    execution_completed=True,
                ),
            ],
        )
        full_context.assert_not_awaited()

    def test_catalog_accessor_binds_protection_context_when_none_is_prebound(self) -> None:
        service = DevelopmentService()
        principal = _principal()
        resources: dict[str, list[dict[str, str]]] = {"tools": [], "indexes": []}

        with (
            patch("ragtime.userspace.development_service.content_protection_service.current_context", return_value=None),
            patch("ragtime.userspace.development_service.external_protection_context", return_value=nullcontext()),
            patch("ragtime.userspace.development_service.authorize_external_content", new_callable=AsyncMock),
            patch.object(service, "_workspace", new_callable=AsyncMock),
            patch.object(service, "_resources", new_callable=AsyncMock, return_value=resources),
            patch.object(service, "_audit_operation", new_callable=AsyncMock),
        ):
            result = asyncio.run(service.get_resource_catalog(principal, "workspace-1"))

        self.assertEqual(result, {"authorized_tools": [], "authorized_indexes": []})

    def test_resources_catalog_cannot_be_called_or_discovered_publicly(self) -> None:
        service = DevelopmentService()
        principal = _principal()

        self.assertNotIn("resources_catalog", {item["name"] for item in service.list_operations()})
        with (
            patch("ragtime.userspace.development_service.content_protection_service.current_context", return_value=object()),
            patch("ragtime.userspace.development_service.authorize_external_content", new_callable=AsyncMock),
            self.assertRaises(HTTPException) as raised,
        ):
            asyncio.run(service.execute(principal, "workspace-1", "resources_catalog"))
        self.assertEqual(raised.exception.status_code, 404)

    def test_full_instruction_bundle_and_targeted_catalog_keep_resource_pins(self) -> None:
        resource = {"component_id": "tool-1", "name": "Accounting", "description": "\u03bb\x00" * 9000}
        catalog = {"authorized_tools": [resource], "authorized_indexes": []}

        full_item = read_resources({"instruction_bundle": {"capabilities": catalog}})["items"][0]
        targeted_item = read_resources({"capabilities": catalog})["items"][0]

        self.assertEqual(full_item["resource_id"], targeted_item["resource_id"])
        self.assertEqual(full_item["contract_reference"]["sha256"], targeted_item["contract_reference"]["sha256"])
        self.assertEqual(full_item["description_reference"]["sha256"], targeted_item["description_reference"]["sha256"])


if __name__ == "__main__":
    unittest.main()
