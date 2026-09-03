from __future__ import annotations

import json
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi import FastAPI, HTTPException

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace import external_api as external_api_module  # noqa: E402
from ragtime.userspace import external_api_routes  # noqa: E402

_NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def _write_manifest(files_root: Path, payload: dict) -> None:
    manifest_path = files_root / ".ragtime" / "external-api.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


class _FakePublishedEndpointTable:
    def __init__(self, *rows: SimpleNamespace) -> None:
        self.rows = {row.id: row for row in rows}
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self.deleted: list[dict] = []

    async def find_many(self, *, where: dict | None = None, order: dict | None = None) -> list[SimpleNamespace]:
        _ = order
        workspace_id = (where or {}).get("workspaceId")
        rows = list(self.rows.values())
        if workspace_id is not None:
            rows = [row for row in rows if row.workspaceId == workspace_id]
        return rows

    async def find_first(self, *, where: dict, include: dict | None = None) -> SimpleNamespace | None:
        _ = include
        for row in self.rows.values():
            if all(getattr(row, key, None) == value for key, value in where.items()):
                return row
        return None

    async def find_unique(self, *, where: dict, include: dict | None = None) -> SimpleNamespace | None:
        _ = include
        row_id = where.get("id")
        if row_id is not None:
            return self.rows.get(row_id)
        for row in self.rows.values():
            if all(getattr(row, key, None) == value for key, value in where.items()):
                return row
        return None

    async def create(self, *, data: dict) -> SimpleNamespace:
        self.created.append(data)
        row = SimpleNamespace(
            id=data["id"],
            workspaceId=data["workspaceId"],
            key=data["key"],
            label=data["label"],
            description=data["description"],
            method=data["method"],
            path=data["path"],
            definitionHash=data["definitionHash"],
            enabled=data["enabled"],
            approvedByUserId=data.get("approvedByUserId"),
            approvedAt=data["approvedAt"],
            createdAt=data["createdAt"],
            updatedAt=data["updatedAt"],
        )
        self.rows[row.id] = row
        return row

    async def update(self, *, where: dict, data: dict) -> SimpleNamespace:
        self.updated.append({"where": where, "data": data})
        row = self.rows[where["id"]]
        for key, value in data.items():
            if isinstance(value, dict) and "increment" in value:
                setattr(row, key, int(getattr(row, key, 0) or 0) + int(value["increment"]))
                continue
            setattr(row, key, value)
        return row

    async def delete(self, *, where: dict) -> SimpleNamespace:
        self.deleted.append(where)
        return self.rows.pop(where["id"])


class _FakeCredentialEndpointTable:
    def __init__(self) -> None:
        self.rows: list[SimpleNamespace] = []
        self.created: list[dict] = []
        self.deleted: list[dict] = []

    async def create(self, *, data: dict) -> SimpleNamespace:
        self.created.append(data)
        row = SimpleNamespace(id=f"grant-{len(self.rows) + 1}", credentialId=data["credentialId"], endpointId=data["endpointId"])
        self.rows.append(row)
        return row

    async def delete_many(self, *, where: dict) -> SimpleNamespace:
        self.deleted.append(where)
        if "credentialId" in where:
            self.rows = [row for row in self.rows if row.credentialId != where["credentialId"]]
        return SimpleNamespace(count=0)


class _FakeCredentialTable:
    def __init__(self, *rows: SimpleNamespace) -> None:
        self.rows = {row.id: row for row in rows}
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self.deleted: list[dict] = []
        self.request_log_table: _FakeRequestLogTable | None = None
        self.grant_table: _FakeCredentialEndpointTable | None = None

    async def find_first(self, *, where: dict, include: dict | None = None) -> SimpleNamespace | None:
        _ = include
        for row in self.rows.values():
            if all(getattr(row, key, None) == value for key, value in where.items()):
                return row
        return None

    async def find_unique(self, *, where: dict, include: dict | None = None) -> SimpleNamespace | None:
        _ = include
        row_id = where.get("id")
        if row_id is not None:
            return self.rows.get(row_id)
        return None

    async def find_many(self, *, where: dict | None = None, include: dict | None = None, order: dict | None = None) -> list[SimpleNamespace]:
        _ = include, order
        workspace_id = (where or {}).get("workspaceId")
        rows = list(self.rows.values())
        if workspace_id is not None:
            rows = [row for row in rows if row.workspaceId == workspace_id]
        return rows

    async def create(self, *, data: dict) -> SimpleNamespace:
        self.created.append(data)
        row = SimpleNamespace(
            id=data["id"],
            workspaceId=data["workspaceId"],
            label=data["label"],
            tokenPrefix=data["tokenPrefix"],
            tokenHash=data["tokenHash"],
            enabled=data["enabled"],
            expiresAt=data.get("expiresAt"),
            lastUsedAt=None,
            requestCount=0,
            createdAt=data["createdAt"],
            updatedAt=data["updatedAt"],
            revokedAt=None,
            createdByUserId=data.get("createdByUserId"),
            endpointGrants=[],
        )
        self.rows[row.id] = row
        return row

    async def update(self, *, where: dict, data: dict) -> SimpleNamespace:
        self.updated.append({"where": where, "data": data})
        row = self.rows[where["id"]]
        for key, value in data.items():
            if isinstance(value, dict) and "increment" in value:
                setattr(row, key, int(getattr(row, key, 0) or 0) + int(value["increment"]))
                continue
            setattr(row, key, value)
        return row

    async def delete(self, *, where: dict) -> SimpleNamespace:
        self.deleted.append(where)
        row = self.rows.pop(where["id"])
        if self.grant_table is not None:
            await self.grant_table.delete_many(where={"credentialId": where["id"]})
        if self.request_log_table is not None:
            for request_row in self.request_log_table.rows:
                if getattr(request_row, "credentialId", None) == where["id"]:
                    request_row.credentialId = None
        return row


class _FakeRequestLogTable:
    def __init__(self, *rows: SimpleNamespace) -> None:
        self.rows = list(rows)
        self.created: list[dict] = []
        self.deleted: list[dict] = []
        self.find_many_calls: list[dict] = []
        self.find_unique_calls: list[dict] = []

    async def create(self, *, data: dict) -> SimpleNamespace:
        self.created.append(data)
        row = SimpleNamespace(**data)
        self.rows.append(row)
        return row

    async def find_many(self, *, where: dict | None = None, take: int | None = None, order: list[dict] | dict | None = None) -> list[SimpleNamespace]:
        self.find_many_calls.append({"where": where, "take": take, "order": order})
        workspace_id = (where or {}).get("workspaceId")
        rows = list(self.rows)
        if workspace_id is not None:
            rows = [row for row in rows if row.workspaceId == workspace_id]
        rows.sort(key=lambda row: (row.createdAt, row.id), reverse=True)
        cursor_filter = (where or {}).get("OR") or []
        if cursor_filter:
            filtered: list[SimpleNamespace] = []
            for row in rows:
                created_before = any(
                    row.createdAt < clause.get("createdAt", {}).get("lt")
                    for clause in cursor_filter
                    if isinstance(clause, dict) and "createdAt" in clause and isinstance(clause.get("createdAt"), dict)
                )
                same_timestamp_lower_id = any(
                    row.createdAt == clause.get("AND", [{}, {}])[0].get("createdAt") and row.id < clause.get("AND", [{}, {}])[1].get("id", {}).get("lt", "")
                    for clause in cursor_filter
                    if isinstance(clause, dict) and isinstance(clause.get("AND"), list) and len(clause["AND"]) == 2
                )
                if created_before or same_timestamp_lower_id:
                    filtered.append(row)
            rows = filtered
        return rows[:take] if take is not None else rows

    async def find_unique(self, *, where: dict) -> SimpleNamespace | None:
        self.find_unique_calls.append({"where": where})
        row_id = where.get("id")
        for row in self.rows:
            if row.id == row_id:
                return row
        return None

    async def delete_many(self, *, where: dict) -> SimpleNamespace:
        self.deleted.append(where)
        return SimpleNamespace(count=0)


class ManifestParsingTests(unittest.TestCase):
    def test_parse_manifest_accepts_canonical_get_and_head_entries(self) -> None:
        with TemporaryDirectory() as temp_dir:
            files_root = Path(temp_dir)
            _write_manifest(
                files_root,
                {
                    "version": 1,
                    "endpoints": [
                        {
                            "key": "accounting-periods",
                            "label": "Accounting periods",
                            "description": "Lists periods.",
                            "method": "GET",
                            "path": "/backend/periods",
                        },
                        {
                            "key": "accounting-periods-head",
                            "label": "Accounting periods HEAD",
                            "description": "Checks periods.",
                            "method": "HEAD",
                            "path": "/backend/periods/head",
                        },
                    ],
                },
            )

            manifest = external_api_module.parse_external_api_manifest(files_root)

        self.assertTrue(manifest.valid)
        self.assertEqual(manifest.version, 1)
        self.assertEqual(manifest.errors, [])
        self.assertEqual([candidate.key for candidate in manifest.candidates], ["accounting-periods", "accounting-periods-head"])
        self.assertTrue(all(candidate.valid for candidate in manifest.candidates))
        self.assertTrue(manifest.candidates[0].definition_hash)

    def test_parse_manifest_rejects_reserved_paths_duplicate_keys_and_noncanonical_paths(self) -> None:
        with TemporaryDirectory() as temp_dir:
            files_root = Path(temp_dir)
            _write_manifest(
                files_root,
                {
                    "version": 1,
                    "endpoints": [
                        {
                            "key": "dup",
                            "label": "One",
                            "description": "desc",
                            "method": "GET",
                            "path": "/auth/me",
                        },
                        {
                            "key": "dup",
                            "label": "Two",
                            "description": "desc",
                            "method": "GET",
                            "path": "/backend//periods",
                        },
                        {
                            "key": "encoded",
                            "label": "Encoded",
                            "description": "desc",
                            "method": "GET",
                            "path": "/backend/%2Fsecret",
                        },
                    ],
                },
            )

            manifest = external_api_module.parse_external_api_manifest(files_root)

        self.assertFalse(manifest.valid)
        self.assertGreaterEqual(len(manifest.errors), 3)
        self.assertIn("duplicate endpoint key: dup", manifest.errors)
        self.assertTrue(any("reserved" in error for error in manifest.errors))
        self.assertTrue(any("canonical" in error or "normalized" in error for error in manifest.errors))


class AuthenticationAndAccountingTests(unittest.IsolatedAsyncioTestCase):
    async def test_authenticate_validates_token_hash_and_live_definition_hash(self) -> None:
        with TemporaryDirectory() as temp_dir:
            files_root = Path(temp_dir)
            _write_manifest(
                files_root,
                {
                    "version": 1,
                    "endpoints": [
                        {
                            "key": "accounting-periods",
                            "label": "Accounting periods",
                            "description": "Lists periods.",
                            "method": "GET",
                            "path": "/backend/periods",
                        }
                    ],
                },
            )
            manifest = external_api_module.parse_external_api_manifest(files_root)
            token = "rtws_aabbccddeeff00112233445566778899_super-secret"
            credential = SimpleNamespace(
                id="cred-1",
                workspaceId="ws-1",
                label="Power Query",
                tokenPrefix="aabbccddeeff00112233445566778899",
                tokenHash=external_api_module._hash_service_token(token),
                enabled=True,
                expiresAt=None,
                revokedAt=None,
                endpointGrants=[
                    SimpleNamespace(
                        endpoint=SimpleNamespace(
                            id="ep-1",
                            key="accounting-periods",
                            label="Accounting periods",
                            description="Lists periods.",
                            method="GET",
                            path="/backend/periods",
                            enabled=True,
                            definitionHash=manifest.candidates[0].definition_hash,
                        )
                    )
                ],
            )
            db = SimpleNamespace(workspaceservicecredential=_FakeCredentialTable(credential))

            with (
                mock.patch.object(external_api_module, "get_db", mock.AsyncMock(return_value=db)),
                mock.patch.object(external_api_module, "_workspace_files_root", return_value=files_root),
            ):
                principal = await external_api_module.authenticate_workspace_service_request(
                    workspace_id="ws-1",
                    method="GET",
                    path="/backend/periods",
                    bearer_token=token,
                )

            self.assertEqual(principal.credential_id, "cred-1")
            self.assertEqual(principal.endpoint_key, "accounting-periods")
            self.assertEqual(principal.path_template, "/backend/periods")

    async def test_authenticate_rejects_stale_definition_and_revoked_or_expired_credentials(self) -> None:
        with TemporaryDirectory() as temp_dir:
            files_root = Path(temp_dir)
            _write_manifest(
                files_root,
                {
                    "version": 1,
                    "endpoints": [
                        {
                            "key": "accounting-periods",
                            "label": "Accounting periods",
                            "description": "Lists periods.",
                            "method": "GET",
                            "path": "/backend/periods-v2",
                        }
                    ],
                },
            )
            token = "rtws_aabbccddeeff00112233445566778899_super-secret"
            base_endpoint = SimpleNamespace(
                id="ep-1",
                key="accounting-periods",
                label="Accounting periods",
                description="Lists periods.",
                method="GET",
                path="/backend/periods",
                enabled=True,
                definitionHash="stale-hash",
            )
            stale = SimpleNamespace(
                id="cred-stale",
                workspaceId="ws-1",
                label="Stale",
                tokenPrefix="aabbccddeeff00112233445566778899",
                tokenHash=external_api_module._hash_service_token(token),
                enabled=True,
                expiresAt=None,
                revokedAt=None,
                endpointGrants=[SimpleNamespace(endpoint=base_endpoint)],
            )
            revoked = SimpleNamespace(
                id="cred-revoked",
                workspaceId="ws-1",
                label="Revoked",
                tokenPrefix="11223344556677889900aabbccddeeff",
                tokenHash=external_api_module._hash_service_token("rtws_11223344556677889900aabbccddeeff_secret"),
                enabled=False,
                expiresAt=None,
                revokedAt=_NOW,
                endpointGrants=[SimpleNamespace(endpoint=base_endpoint)],
            )
            expired = SimpleNamespace(
                id="cred-expired",
                workspaceId="ws-1",
                label="Expired",
                tokenPrefix="99aabbccddeeff001122334455667788",
                tokenHash=external_api_module._hash_service_token("rtws_99aabbccddeeff001122334455667788_secret"),
                enabled=True,
                expiresAt=_NOW - timedelta(minutes=1),
                revokedAt=None,
                endpointGrants=[SimpleNamespace(endpoint=base_endpoint)],
            )
            db = SimpleNamespace(workspaceservicecredential=_FakeCredentialTable(stale, revoked, expired))

            with (
                mock.patch.object(external_api_module, "get_db", mock.AsyncMock(return_value=db)),
                mock.patch.object(external_api_module, "_workspace_files_root", return_value=files_root),
                mock.patch.object(external_api_module, "utc_now", return_value=_NOW),
            ):
                with self.assertRaises(HTTPException) as stale_error:
                    await external_api_module.authenticate_workspace_service_request(
                        workspace_id="ws-1",
                        method="GET",
                        path="/backend/periods",
                        bearer_token=token,
                    )
                with self.assertRaises(HTTPException) as revoked_error:
                    await external_api_module.authenticate_workspace_service_request(
                        workspace_id="ws-1",
                        method="GET",
                        path="/backend/periods",
                        bearer_token="rtws_11223344556677889900aabbccddeeff_secret",
                    )
                with self.assertRaises(HTTPException) as expired_error:
                    await external_api_module.authenticate_workspace_service_request(
                        workspace_id="ws-1",
                        method="GET",
                        path="/backend/periods",
                        bearer_token="rtws_99aabbccddeeff001122334455667788_secret",
                    )

        self.assertEqual(stale_error.exception.status_code, 403)
        self.assertEqual(revoked_error.exception.status_code, 401)
        self.assertEqual(expired_error.exception.status_code, 401)

    async def test_record_workspace_api_request_updates_usage_and_runs_best_effort_retention(self) -> None:
        credential_table = _FakeCredentialTable(
            SimpleNamespace(
                id="cred-1",
                workspaceId="ws-1",
                label="Power Query",
                tokenPrefix="aabbccddeeff0011",
                tokenHash="hash",
                enabled=True,
                expiresAt=None,
                lastUsedAt=None,
                requestCount=2,
                createdAt=_NOW,
                updatedAt=_NOW,
                revokedAt=None,
                createdByUserId="user-1",
            )
        )
        request_log_table = _FakeRequestLogTable()
        db = SimpleNamespace(
            workspaceservicecredential=credential_table,
            workspaceapirequestlog=request_log_table,
        )
        principal = external_api_module.WorkspaceServicePrincipal(
            credential_id="cred-1",
            credential_label="Power Query",
            workspace_id="ws-1",
            endpoint_id="ep-1",
            endpoint_key="accounting-periods",
            endpoint_label="Accounting periods",
            method="GET",
            path_template="/backend/periods",
        )

        with (
            mock.patch.object(external_api_module, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(external_api_module, "utc_now", side_effect=[_NOW, _NOW, _NOW]),
            mock.patch.object(external_api_module, "_REQUEST_LOG_CLEANUP_LAST_RUN_ON", None),
        ):
            await external_api_module.record_workspace_api_request(
                principal,
                status_code=200,
                duration_ms=45,
                client_fingerprint="fp-1",
            )

        self.assertEqual(credential_table.rows["cred-1"].requestCount, 3)
        self.assertEqual(credential_table.updated[-1]["data"]["lastUsedAt"], _NOW)
        self.assertEqual(request_log_table.created[-1]["pathTemplate"], "/backend/periods")
        self.assertEqual(request_log_table.created[-1]["clientFingerprint"], "fp-1")
        self.assertEqual(request_log_table.deleted[-1]["createdAt"]["lt"], _NOW - timedelta(days=90))

    async def test_record_workspace_api_denied_request_omits_secret_fields(self) -> None:
        request_log_table = _FakeRequestLogTable()
        db = SimpleNamespace(workspaceapirequestlog=request_log_table)

        with (
            mock.patch.object(external_api_module, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(external_api_module, "utc_now", return_value=_NOW),
        ):
            await external_api_module.record_workspace_api_denied_request(
                workspace_id="ws-1",
                method="GET",
                path="/backend/periods?company=1",
                status_code=401,
                reason="invalid_token",
                token_selector="aabbccddeeff0011",
                client_fingerprint="fp-1",
            )

        created = request_log_table.created[-1]
        self.assertEqual(created["workspaceId"], "ws-1")
        self.assertEqual(created["statusCode"], 401)
        self.assertEqual(created["pathTemplate"], "/backend/periods")
        serialized = json.dumps(created, default=str).lower()
        self.assertNotIn("authorization", serialized)
        self.assertNotIn("company=1", serialized)

    async def test_list_workspace_api_requests_payload_uses_bounded_cursor_pagination(self) -> None:
        request_log_table = _FakeRequestLogTable(
            SimpleNamespace(
                id="req-z",
                workspaceId="ws-1",
                credentialId="cred-z",
                credentialLabel="Z",
                endpointKey="periods",
                endpointLabel="Periods",
                method="GET",
                pathTemplate="/api/z",
                statusCode=200,
                durationMs=11,
                createdAt=_NOW,
            ),
            SimpleNamespace(
                id="req-y",
                workspaceId="ws-1",
                credentialId="cred-y",
                credentialLabel="Y",
                endpointKey="periods",
                endpointLabel="Periods",
                method="GET",
                pathTemplate="/api/y",
                statusCode=200,
                durationMs=12,
                createdAt=_NOW,
            ),
            SimpleNamespace(
                id="req-x",
                workspaceId="ws-1",
                credentialId="cred-x",
                credentialLabel="X",
                endpointKey="periods",
                endpointLabel="Periods",
                method="GET",
                pathTemplate="/api/x",
                statusCode=200,
                durationMs=13,
                createdAt=_NOW - timedelta(minutes=1),
            ),
            SimpleNamespace(
                id="req-w",
                workspaceId="ws-1",
                credentialId="cred-w",
                credentialLabel="W",
                endpointKey="periods",
                endpointLabel="Periods",
                method="GET",
                pathTemplate="/api/w",
                statusCode=200,
                durationMs=14,
                createdAt=_NOW - timedelta(minutes=2),
            ),
        )
        db = SimpleNamespace(workspaceapirequestlog=request_log_table)

        with mock.patch.object(external_api_module, "get_db", mock.AsyncMock(return_value=db)):
            first_page = await external_api_module.list_workspace_api_requests_payload(
                workspace_id="ws-1",
                cursor=None,
                limit=2,
            )
            second_page = await external_api_module.list_workspace_api_requests_payload(
                workspace_id="ws-1",
                cursor="req-y",
                limit=2,
            )

        self.assertEqual(first_page["limit"], 2)
        self.assertEqual([item["id"] for item in first_page["items"]], ["req-z", "req-y"])
        self.assertEqual(first_page["cursor"], "req-y")
        self.assertEqual(request_log_table.find_many_calls[0]["take"], 3)
        self.assertEqual(
            request_log_table.find_many_calls[0]["order"],
            [{"createdAt": "desc"}, {"id": "desc"}],
        )
        self.assertEqual(request_log_table.find_unique_calls[0]["where"], {"id": "req-y"})
        self.assertEqual([item["id"] for item in second_page["items"]], ["req-x", "req-w"])
        self.assertIsNone(second_page["cursor"])
        self.assertEqual(
            request_log_table.find_many_calls[1]["where"],
            {
                "workspaceId": "ws-1",
                "OR": [
                    {"createdAt": {"lt": _NOW}},
                    {"AND": [{"createdAt": _NOW}, {"id": {"lt": "req-y"}}]},
                ],
            },
        )


class ManagementRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_delete_revoked_credential_service_removes_record_and_preserves_history(self) -> None:
        endpoint = SimpleNamespace(
            id="ep-1",
            workspaceId="ws-1",
            key="accounting-periods",
            label="Accounting periods",
            description="Lists periods.",
            method="GET",
            path="/backend/periods",
            enabled=True,
            definitionHash="hash-1",
            approvedAt=_NOW,
            createdAt=_NOW,
            updatedAt=_NOW,
        )
        credential = SimpleNamespace(
            id="cred-1",
            workspaceId="ws-1",
            label="Revoked key",
            tokenPrefix="selector",
            tokenHash="hash",
            enabled=False,
            expiresAt=None,
            lastUsedAt=None,
            requestCount=2,
            createdAt=_NOW,
            updatedAt=_NOW,
            revokedAt=_NOW,
            createdByUserId="owner-1",
            endpointGrants=[SimpleNamespace(endpoint=endpoint)],
        )
        credential_table = _FakeCredentialTable(credential)
        grant_table = _FakeCredentialEndpointTable()
        grant_table.rows = [SimpleNamespace(id="grant-1", credentialId="cred-1", endpointId="ep-1")]
        request_log_table = _FakeRequestLogTable(
            SimpleNamespace(
                id="req-1",
                workspaceId="ws-1",
                credentialId="cred-1",
                credentialLabel="Revoked key",
                endpointKey="accounting-periods",
                endpointLabel="Accounting periods",
                method="GET",
                pathTemplate="/backend/periods",
                statusCode=200,
                durationMs=20,
                createdAt=_NOW,
            )
        )
        credential_table.grant_table = grant_table
        credential_table.request_log_table = request_log_table
        db = SimpleNamespace(
            workspaceservicecredential=credential_table,
            workspaceservicecredentialendpoint=grant_table,
            workspaceapirequestlog=request_log_table,
        )
        fake_userspace_service = SimpleNamespace(_record_runtime_audit_event=mock.AsyncMock(return_value=True))

        with (
            mock.patch.object(external_api_module, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(external_api_module, "userspace_service", fake_userspace_service),
        ):
            await external_api_module.delete_revoked_workspace_service_credential(
                workspace_id="ws-1",
                credential_id="cred-1",
                user_id="owner-1",
            )

        self.assertNotIn("cred-1", credential_table.rows)
        self.assertEqual(credential_table.deleted, [{"id": "cred-1"}])
        self.assertEqual(grant_table.deleted, [{"credentialId": "cred-1"}])
        self.assertIsNone(request_log_table.rows[0].credentialId)
        fake_userspace_service._record_runtime_audit_event.assert_awaited_once_with(
            "ws-1",
            "owner-1",
            "external_api.credential_deleted",
            {"credential_id": "cred-1", "credential_label": "Revoked key"},
        )

    async def test_delete_revoked_credential_service_rejects_active_and_wrong_workspace_records(self) -> None:
        active = SimpleNamespace(
            id="cred-active",
            workspaceId="ws-1",
            label="Active key",
            tokenPrefix="selector-a",
            tokenHash="hash-a",
            enabled=True,
            expiresAt=None,
            lastUsedAt=None,
            requestCount=0,
            createdAt=_NOW,
            updatedAt=_NOW,
            revokedAt=None,
            createdByUserId="owner-1",
            endpointGrants=[],
        )
        other_workspace = SimpleNamespace(
            id="cred-other",
            workspaceId="ws-2",
            label="Other workspace key",
            tokenPrefix="selector-b",
            tokenHash="hash-b",
            enabled=False,
            expiresAt=None,
            lastUsedAt=None,
            requestCount=0,
            createdAt=_NOW,
            updatedAt=_NOW,
            revokedAt=_NOW,
            createdByUserId="owner-2",
            endpointGrants=[],
        )
        db = SimpleNamespace(workspaceservicecredential=_FakeCredentialTable(active, other_workspace))

        with mock.patch.object(external_api_module, "get_db", mock.AsyncMock(return_value=db)):
            with self.assertRaises(HTTPException) as active_error:
                await external_api_module.delete_revoked_workspace_service_credential(
                    workspace_id="ws-1",
                    credential_id="cred-active",
                    user_id="owner-1",
                )
            with self.assertRaises(HTTPException) as missing_error:
                await external_api_module.delete_revoked_workspace_service_credential(
                    workspace_id="ws-1",
                    credential_id="missing",
                    user_id="owner-1",
                )
            with self.assertRaises(HTTPException) as wrong_workspace_error:
                await external_api_module.delete_revoked_workspace_service_credential(
                    workspace_id="ws-1",
                    credential_id="cred-other",
                    user_id="owner-1",
                )

        self.assertEqual(active_error.exception.status_code, 400)
        self.assertEqual(active_error.exception.detail, "Only revoked credentials can be deleted")
        self.assertEqual(missing_error.exception.status_code, 404)
        self.assertEqual(missing_error.exception.detail, "Service credential not found")
        self.assertEqual(wrong_workspace_error.exception.status_code, 404)
        self.assertEqual(wrong_workspace_error.exception.detail, "Service credential not found")

    async def test_delete_revoked_credential_service_rejects_enabled_rows_even_if_revoked_at_is_present(self) -> None:
        inconsistent = SimpleNamespace(
            id="cred-inconsistent",
            workspaceId="ws-1",
            label="Inconsistent key",
            tokenPrefix="selector-c",
            tokenHash="hash-c",
            enabled=True,
            expiresAt=None,
            lastUsedAt=None,
            requestCount=0,
            createdAt=_NOW,
            updatedAt=_NOW,
            revokedAt=_NOW,
            createdByUserId="owner-1",
            endpointGrants=[],
        )
        db = SimpleNamespace(workspaceservicecredential=_FakeCredentialTable(inconsistent))

        with mock.patch.object(external_api_module, "get_db", mock.AsyncMock(return_value=db)):
            with self.assertRaises(HTTPException) as inconsistent_error:
                await external_api_module.delete_revoked_workspace_service_credential(
                    workspace_id="ws-1",
                    credential_id="cred-inconsistent",
                    user_id="owner-1",
                )

        self.assertEqual(inconsistent_error.exception.status_code, 400)
        self.assertEqual(inconsistent_error.exception.detail, "Only revoked credentials can be deleted")

    async def test_delete_revoked_credential_route_uses_owner_admin_auth_and_returns_no_content(self) -> None:
        app = FastAPI()
        app.include_router(external_api_routes.router)
        transport = httpx.ASGITransport(app=app)

        service_mock = mock.AsyncMock(return_value=None)

        async def enforce_workspace_role(workspace_id: str, user_id: str, role: str, *, is_admin: bool = False) -> None:
            _ = workspace_id, role
            if user_id in {"owner-1", "admin-1"} and (user_id != "admin-1" or is_admin):
                return
            raise HTTPException(status_code=403, detail="Forbidden")

        fake_userspace_service = SimpleNamespace(enforce_workspace_role=mock.AsyncMock(side_effect=enforce_workspace_role))

        try:
            with (
                mock.patch.object(external_api_routes, "userspace_service", fake_userspace_service),
                mock.patch.object(external_api_routes, "delete_revoked_workspace_service_credential", service_mock),
            ):
                app.dependency_overrides[external_api_routes.get_current_user] = lambda: SimpleNamespace(id="owner-1", role="user")
                async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
                    owner_response = await client.delete("/indexes/userspace/workspaces/ws-1/external-api/credentials/cred-1/record")
                self.assertEqual(owner_response.status_code, 204)
                self.assertEqual(owner_response.text, "")

                app.dependency_overrides[external_api_routes.get_current_user] = lambda: SimpleNamespace(id="admin-1", role="admin")
                async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
                    admin_response = await client.delete("/indexes/userspace/workspaces/ws-1/external-api/credentials/cred-2/record")
                self.assertEqual(admin_response.status_code, 204)

                app.dependency_overrides[external_api_routes.get_current_user] = lambda: SimpleNamespace(id="viewer-1", role="user")
                async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
                    forbidden_response = await client.delete("/indexes/userspace/workspaces/ws-1/external-api/credentials/cred-3/record")
                self.assertEqual(forbidden_response.status_code, 403)
        finally:
            app.dependency_overrides.clear()

        self.assertEqual(service_mock.await_count, 2)
        service_mock.assert_any_await(workspace_id="ws-1", credential_id="cred-1", user_id="owner-1")
        service_mock.assert_any_await(workspace_id="ws-1", credential_id="cred-2", user_id="admin-1")

    async def test_routes_require_authenticated_owner_or_admin(self) -> None:
        app = FastAPI()
        app.include_router(external_api_routes.router)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
            response = await client.get("/indexes/userspace/workspaces/ws-1/external-api/manifest")

        self.assertEqual(response.status_code, 401)

    async def test_manifest_and_credentials_routes_return_fixed_shapes(self) -> None:
        app = FastAPI()
        app.include_router(external_api_routes.router)
        app.dependency_overrides[external_api_routes.get_current_user] = lambda: SimpleNamespace(id="owner-1", role="user")

        endpoint_table = _FakePublishedEndpointTable(
            SimpleNamespace(
                id="ep-1",
                workspaceId="ws-1",
                key="accounting-periods",
                label="Accounting periods",
                description="Lists periods.",
                method="GET",
                path="/backend/periods",
                enabled=True,
                definitionHash="hash-1",
                approvedAt=_NOW,
                createdAt=_NOW,
                updatedAt=_NOW,
            )
        )
        credential_table = _FakeCredentialTable()
        grant_table = _FakeCredentialEndpointTable()
        request_log_table = _FakeRequestLogTable(
            SimpleNamespace(
                id="req-1",
                workspaceId="ws-1",
                credentialId="cred-1",
                credentialLabel="Power Query",
                endpointKey="accounting-periods",
                endpointLabel="Accounting periods",
                method="GET",
                pathTemplate="/backend/periods",
                statusCode=200,
                durationMs=34,
                createdAt=_NOW,
            )
        )
        db = SimpleNamespace(
            workspacepublishedendpoint=endpoint_table,
            workspaceservicecredential=credential_table,
            workspaceservicecredentialendpoint=grant_table,
            workspaceapirequestlog=request_log_table,
        )

        with TemporaryDirectory() as temp_dir:
            files_root = Path(temp_dir)
            _write_manifest(
                files_root,
                {
                    "version": 1,
                    "endpoints": [
                        {
                            "key": "accounting-periods",
                            "label": "Accounting periods",
                            "description": "Lists periods.",
                            "method": "GET",
                            "path": "/backend/periods",
                        }
                    ],
                },
            )
            fake_userspace_service = SimpleNamespace(
                enforce_workspace_role=mock.AsyncMock(),
                _record_runtime_audit_event=mock.AsyncMock(return_value=True),
            )
            fake_runtime_service = SimpleNamespace(get_preview_origin=lambda workspace_id, control_plane_origin=None: f"https://{workspace_id}.preview.example")

            try:
                with (
                    mock.patch.object(external_api_module, "get_db", mock.AsyncMock(return_value=db)),
                    mock.patch.object(external_api_module, "_workspace_files_root", return_value=files_root),
                    mock.patch.object(external_api_module, "userspace_service", fake_userspace_service),
                    mock.patch.object(external_api_routes, "userspace_service", fake_userspace_service),
                    mock.patch.object(external_api_routes, "userspace_runtime_service", fake_runtime_service),
                    mock.patch.object(external_api_module, "utc_now", return_value=_NOW),
                ):
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
                        manifest_response = await client.get("/indexes/userspace/workspaces/ws-1/external-api/manifest")
                        self.assertEqual(manifest_response.status_code, 200)
                        self.assertEqual(
                            manifest_response.json(),
                            {
                                "preview_origin": "https://ws-1.preview.example",
                                "version": 1,
                                "valid": True,
                                "errors": [],
                                "candidates": [
                                    {
                                        "key": "accounting-periods",
                                        "label": "Accounting periods",
                                        "description": "Lists periods.",
                                        "method": "GET",
                                        "path": "/backend/periods",
                                        "valid": True,
                                        "errors": [],
                                    }
                                ],
                            },
                        )

                        publish_response = await client.post("/indexes/userspace/workspaces/ws-1/external-api/endpoints/accounting-periods/publish")
                        self.assertEqual(publish_response.status_code, 200)
                        self.assertEqual(publish_response.json()["key"], "accounting-periods")
                        published_payload = publish_response.json()

                        endpoints_response = await client.get("/indexes/userspace/workspaces/ws-1/external-api/endpoints")
                        self.assertEqual(endpoints_response.status_code, 200)
                        self.assertEqual(
                            endpoints_response.json(),
                            {
                                "preview_origin": "https://ws-1.preview.example",
                                "items": [published_payload],
                            },
                        )

                        create_response = await client.post(
                            "/indexes/userspace/workspaces/ws-1/external-api/credentials",
                            json={
                                "label": "Power Query",
                                "endpoint_keys": ["accounting-periods"],
                                "expires_at": None,
                            },
                        )
                        self.assertEqual(create_response.status_code, 200)
                        create_payload = create_response.json()
                        self.assertEqual(create_payload["label"], "Power Query")
                        self.assertTrue(create_payload["token"].startswith("rtws_"))
                        self.assertEqual(create_payload["endpoint_keys"], ["accounting-periods"])

                        credentials_response = await client.get("/indexes/userspace/workspaces/ws-1/external-api/credentials")
                        self.assertEqual(credentials_response.status_code, 200)
                        self.assertEqual(credentials_response.json()["items"][0]["label"], "Power Query")
                        self.assertNotIn(create_payload["token"], credentials_response.text)

                        rotate_response = await client.post(f"/indexes/userspace/workspaces/ws-1/external-api/credentials/{create_payload['id']}/rotate")
                        self.assertEqual(rotate_response.status_code, 200)
                        self.assertNotEqual(rotate_response.json()["token_prefix"], create_payload["token_prefix"])

                        requests_response = await client.get("/indexes/userspace/workspaces/ws-1/external-api/requests")
                        self.assertEqual(requests_response.status_code, 200)
                        self.assertEqual(requests_response.json()["items"][0]["endpoint_key"], "accounting-periods")

                        unpublish_response = await client.delete("/indexes/userspace/workspaces/ws-1/external-api/endpoints/ep-1")
                        self.assertEqual(unpublish_response.status_code, 200)
                        self.assertEqual(
                            unpublish_response.json(),
                            published_payload,
                        )

                        create_after_unpublish = await client.post(
                            "/indexes/userspace/workspaces/ws-1/external-api/credentials",
                            json={
                                "label": "Denied after unpublish",
                                "endpoint_keys": ["accounting-periods"],
                                "expires_at": None,
                            },
                        )
                        self.assertEqual(create_after_unpublish.status_code, 400)
                        self.assertIn("published endpoints", create_after_unpublish.text.lower())

                        revoke_response = await client.delete(f"/indexes/userspace/workspaces/ws-1/external-api/credentials/{create_payload['id']}")
                        self.assertEqual(revoke_response.status_code, 200)
                        self.assertEqual(revoke_response.json()["enabled"], False)
            finally:
                app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
