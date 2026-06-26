import io
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from fastapi import HTTPException, UploadFile
from openpyxl import Workbook
from starlette.datastructures import Headers
from starlette.requests import Request

from ragtime.userspace import preview_host, runtime_routes


def _make_upload_file(filename: str, content: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


def _build_xlsx_bytes() -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    assert worksheet is not None
    worksheet.title = "Support Files"
    worksheet.append(["Vendor", "Amount"])
    worksheet.append(["ACME", 123.45])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _build_request(host: str = "31eccb83-4529-4e71-b575-3e510826fcc6.ragtime.hammerton.com") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/__ragtime/files/uploads/payload.bin",
            "raw_path": b"/__ragtime/files/uploads/payload.bin",
            "query_string": b"",
            "headers": [(b"host", host.encode("utf-8"))],
            "scheme": "https",
            "server": (host, 443),
            "client": ("127.0.0.1", 12345),
        }
    )


def _build_body_request(
    body: bytes,
    *,
    content_type: str = "application/octet-stream",
    host: str = "31eccb83-4529-4e71-b575-3e510826fcc6.ragtime.hammerton.com",
) -> Request:
    sent = False

    async def receive() -> dict[str, object]:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/__ragtime/archives/extract",
            "raw_path": b"/__ragtime/archives/extract",
            "query_string": b"",
            "headers": [(b"host", host.encode("utf-8")), (b"content-type", content_type.encode("utf-8"))],
            "scheme": "https",
            "server": (host, 443),
            "client": ("127.0.0.1", 12345),
        },
        receive,
    )


def _build_zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class _FakeRuntimeService:
    @staticmethod
    def _normalize_file_path(file_path: str) -> str:
        from ragtime.core.workspace_ops import normalize_runtime_file_path
        from ragtime.userspace.service import UserSpaceService

        svc = UserSpaceService.__new__(UserSpaceService)
        return normalize_runtime_file_path(file_path, is_reserved_path=svc.is_reserved_internal_path)

    @staticmethod
    def get_preview_origin(workspace_id: str) -> str:
        return f"https://{workspace_id}.ragtime.hammerton.com"

    @staticmethod
    def get_preview_base_domains() -> list[str]:
        return ["ragtime.hammerton.com"]


class _FakeUserSpaceService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.touched: list[str] = []

    def is_reserved_internal_path(self, relative_path: str) -> bool:
        from ragtime.userspace.service import UserSpaceService

        return UserSpaceService.is_reserved_internal_path(self, relative_path)

    async def enforce_workspace_role(self, workspace_id: str, user_id: str, required_role: str):
        return None

    async def ensure_workspace_path_not_in_disabled_mount(self, workspace_id: str, relative_path: str) -> str:
        return relative_path

    async def _is_workspace_mount_owned_path(self, workspace_id: str, relative_path: str) -> bool:
        return False

    def _workspace_files_dir(self, workspace_id: str) -> Path:
        return self.root / "files"

    async def clear_workspace_changed_file_acknowledgements_for_paths_for_all_users(self, workspace_id: str, paths: list[str]) -> None:
        return None

    async def touch_workspace(self, workspace_id: str) -> None:
        self.touched.append(workspace_id)

    @staticmethod
    def _normalize_object_storage_bucket_name(name: str) -> str:
        return name

    def _ensure_object_storage_config(self, workspace_id: str) -> dict[str, object]:
        return {"buckets": [{"name": "default-bucket"}]}

    def _workspace_object_storage_buckets_dir(self, workspace_id: str) -> Path:
        return self.root / "s3" / "buckets"


class UserSpaceDocumentParseTests(unittest.IsolatedAsyncioTestCase):
    def test_primitive_file_path_allows_managed_sqlite_database(self) -> None:
        from ragtime.userspace.service import UserSpaceService

        svc = UserSpaceService.__new__(UserSpaceService)
        self.assertFalse(svc.is_reserved_internal_path(".ragtime/db/app.sqlite3"))
        self.assertFalse(svc.is_reserved_internal_path(".ragtime/db/something.sqlite"))

    def test_primitive_file_path_keeps_other_internal_paths_reserved(self) -> None:
        from ragtime.userspace.service import UserSpaceService

        svc = UserSpaceService.__new__(UserSpaceService)
        self.assertTrue(svc.is_reserved_internal_path(".ragtime/audit-identity.json"))
        self.assertTrue(svc.is_reserved_internal_path(".ragtime/db/something.txt"))

    async def test_extracts_text_from_xlsx_upload(self) -> None:
        upload = _make_upload_file(
            "support.xlsx",
            _build_xlsx_bytes(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        result = await runtime_routes._parse_uploaded_document_upload(upload)

        self.assertEqual(result["filename"], "support.xlsx")
        self.assertIn("Vendor", result["text"])
        self.assertIn("ACME", result["text"])

    async def test_rejects_oversized_upload(self) -> None:
        upload = _make_upload_file(
            "support.csv",
            b"12345",
            "text/csv",
        )

        with mock.patch.object(runtime_routes, "_DOCUMENT_PARSE_MAX_BYTES", 4):
            with self.assertRaises(HTTPException) as exc:
                await runtime_routes._parse_uploaded_document_upload(upload)

        self.assertEqual(exc.exception.status_code, 413)
        detail = str(exc.exception.detail)
        self.assertIn("platform document parsing upload size limit", detail)
        self.assertIn("4 bytes", detail)

    async def test_extracts_structured_spreadsheet_rows(self) -> None:
        upload = _make_upload_file(
            "support.xlsx",
            _build_xlsx_bytes(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        result = await runtime_routes._parse_uploaded_spreadsheet_upload(upload)

        self.assertEqual(result["filename"], "support.xlsx")
        self.assertEqual(result["sheets"][0]["name"], "Support Files")
        self.assertEqual(result["sheets"][0]["rows"][1], ["ACME", 123.45])

    async def test_normalizes_spreadsheet_tables_to_records(self) -> None:
        upload = _make_upload_file(
            "support.xlsx",
            _build_xlsx_bytes(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        result = await runtime_routes._normalize_uploaded_table_upload(upload)

        table = result["tables"][0]
        self.assertTrue(table["has_header_row"])
        self.assertEqual(table["columns"][0]["key"], "vendor")
        self.assertEqual(table["records"][0], {"vendor": "ACME", "amount": 123.45})

    async def test_renders_document_preview_html(self) -> None:
        upload = _make_upload_file("notes.txt", b"Hello\nWorld", "text/plain")

        result = await runtime_routes._render_uploaded_document_preview_upload(upload)

        self.assertEqual(result["format"], "html")
        self.assertIn("Hello", result["html"])
        self.assertIn("<br>", result["html"])

    async def test_workspace_file_primitive_round_trips_binary_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _FakeUserSpaceService(Path(temp_dir))
            upload = _make_upload_file("payload.bin", b"\x00hello", "application/octet-stream")

            with (
                mock.patch.object(runtime_routes, "_userspace_service", return_value=service),
                mock.patch.object(runtime_routes, "_runtime_service", return_value=_FakeRuntimeService()),
                mock.patch.object(runtime_routes, "_get_primitive_upload_max_bytes", mock.AsyncMock(return_value=1024 * 1024)),
            ):
                write_result = await runtime_routes._primitive_workspace_file_write(
                    "workspace-1",
                    "uploads/payload.bin",
                    upload,
                    "user-1",
                )
                response = await runtime_routes._primitive_workspace_file_response(
                    "workspace-1",
                    "uploads/payload.bin",
                    "user-1",
                )

        self.assertEqual(write_result["path"], "uploads/payload.bin")
        self.assertEqual(response.body, b"\x00hello")

    async def test_workspace_file_primitive_round_trips_managed_sqlite_database_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _FakeUserSpaceService(Path(temp_dir))
            request = _build_body_request(b"sqlite-bytes")

            with (
                mock.patch.object(runtime_routes, "_userspace_service", return_value=service),
                mock.patch.object(runtime_routes, "_runtime_service", return_value=_FakeRuntimeService()),
                mock.patch.object(runtime_routes, "_get_primitive_upload_max_bytes", mock.AsyncMock(return_value=1024 * 1024)),
            ):
                write_result = await runtime_routes._primitive_workspace_file_write_request(
                    "workspace-1",
                    ".ragtime/db/app.sqlite3",
                    request,
                    "user-1",
                )
                response = await runtime_routes._primitive_workspace_file_response(
                    "workspace-1",
                    ".ragtime/db/app.sqlite3",
                    "user-1",
                )

        self.assertEqual(write_result["path"], ".ragtime/db/app.sqlite3")
        self.assertEqual(response.body, b"sqlite-bytes")

    async def test_workspace_file_primitive_limit_error_names_configured_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _FakeUserSpaceService(Path(temp_dir))
            upload = _make_upload_file("payload.bin", b"12345", "application/octet-stream")

            with (
                mock.patch.object(runtime_routes, "_userspace_service", return_value=service),
                mock.patch.object(runtime_routes, "_runtime_service", return_value=_FakeRuntimeService()),
                mock.patch.object(runtime_routes, "_get_primitive_upload_max_bytes", mock.AsyncMock(return_value=4)),
            ):
                with self.assertRaises(HTTPException) as exc:
                    await runtime_routes._primitive_workspace_file_write(
                        "workspace-1",
                        "uploads/payload.bin",
                        upload,
                        "user-1",
                    )

        self.assertEqual(exc.exception.status_code, 413)
        detail = str(exc.exception.detail)
        self.assertIn("configured User Space primitive upload size limit", detail)
        self.assertIn("4 bytes", detail)

    async def test_object_primitive_writes_to_bucket_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _FakeUserSpaceService(Path(temp_dir))
            upload = _make_upload_file("image.png", b"png-bytes", "image/png")

            with (
                mock.patch.object(runtime_routes, "_userspace_service", return_value=service),
                mock.patch.object(runtime_routes, "_get_primitive_upload_max_bytes", mock.AsyncMock(return_value=1024 * 1024)),
            ):
                write_result = await runtime_routes._primitive_object_write(
                    "workspace-1",
                    "default-bucket",
                    "images/image.png",
                    upload,
                    "user-1",
                )
                response = await runtime_routes._primitive_object_response(
                    "workspace-1",
                    "default-bucket",
                    "images/image.png",
                    "user-1",
                )

        self.assertEqual(write_result["url"], "/__ragtime/objects/default-bucket/images/image.png")
        self.assertEqual(response.body, b"png-bytes")

    async def test_progress_primitive_updates_and_reads_state(self) -> None:
        service = _FakeUserSpaceService(Path("/tmp/unused"))
        runtime_routes._PRIMITIVE_PROGRESS_STATES.clear()

        with mock.patch.object(runtime_routes, "_userspace_service", return_value=service):
            updated = await runtime_routes._primitive_progress_put(
                "workspace-1",
                "import-1",
                {"phase": "parsing", "progress": 0.5, "message": "Halfway"},
                "user-1",
            )
            fetched = await runtime_routes._primitive_progress_get("workspace-1", "import-1", "user-1")

        self.assertEqual(updated["phase"], "parsing")
        self.assertEqual(fetched["progress"], 0.5)

    async def test_capabilities_describe_workspace_permissions_and_limits(self) -> None:
        service = _FakeUserSpaceService(Path("/tmp/unused"))

        with (
            mock.patch.object(runtime_routes, "_userspace_service", return_value=service),
            mock.patch.object(runtime_routes, "_get_primitive_upload_max_bytes", mock.AsyncMock(return_value=42)),
            mock.patch.object(runtime_routes, "_get_primitive_archive_max_entries", mock.AsyncMock(return_value=7)),
        ):
            result = await runtime_routes._primitive_capabilities("workspace-1", "user-1")

        self.assertTrue(result["can_write_files"])
        self.assertEqual(result["max_upload_bytes"], 42)
        self.assertEqual(result["max_archive_entries"], 7)
        self.assertEqual(result["object_buckets"], ["default-bucket"])

    async def test_upload_target_returns_same_origin_put_contract(self) -> None:
        service = _FakeUserSpaceService(Path("/tmp/unused"))

        with (
            mock.patch.object(runtime_routes, "_userspace_service", return_value=service),
            mock.patch.object(runtime_routes, "_runtime_service", return_value=_FakeRuntimeService()),
            mock.patch.object(runtime_routes, "_get_primitive_upload_max_bytes", mock.AsyncMock(return_value=123)),
        ):
            result = await runtime_routes._primitive_upload_target(
                "workspace-1",
                {"target": "files", "path": "uploads/payload.bin"},
                "user-1",
                preview_origin=True,
            )

        self.assertEqual(result["method"], "PUT")
        self.assertEqual(result["url"], "/__ragtime/files/uploads/payload.bin")
        self.assertEqual(result["max_bytes"], 123)

    async def test_archive_extract_writes_safe_zip_entries_to_workspace_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _FakeUserSpaceService(Path(temp_dir))
            request = _build_body_request(_build_zip_bytes({"bundle/readme.txt": b"hello"}), content_type="application/zip")

            with (
                mock.patch.object(runtime_routes, "_userspace_service", return_value=service),
                mock.patch.object(runtime_routes, "_runtime_service", return_value=_FakeRuntimeService()),
                mock.patch.object(runtime_routes, "_get_primitive_upload_max_bytes", mock.AsyncMock(return_value=1024 * 1024)),
                mock.patch.object(runtime_routes, "_get_primitive_archive_max_entries", mock.AsyncMock(return_value=10)),
            ):
                result = await runtime_routes._primitive_archive_extract(
                    "workspace-1",
                    request,
                    "user-1",
                    destination_path="uploads",
                )

            written = Path(temp_dir) / "files" / "uploads" / "bundle" / "readme.txt"
            self.assertEqual(result["extracted_count"], 1)
            self.assertEqual(written.read_bytes(), b"hello")

    async def test_archive_extract_rejects_traversal_entries(self) -> None:
        service = _FakeUserSpaceService(Path("/tmp/unused"))
        request = _build_body_request(_build_zip_bytes({"../escape.txt": b"nope"}), content_type="application/zip")

        with (
            mock.patch.object(runtime_routes, "_userspace_service", return_value=service),
            mock.patch.object(runtime_routes, "_get_primitive_upload_max_bytes", mock.AsyncMock(return_value=1024 * 1024)),
            mock.patch.object(runtime_routes, "_get_primitive_archive_max_entries", mock.AsyncMock(return_value=10)),
        ):
            with self.assertRaises(HTTPException) as exc:
                await runtime_routes._primitive_archive_extract("workspace-1", request, "user-1")

        self.assertEqual(exc.exception.status_code, 400)

    async def test_archive_extract_limit_error_names_configured_file_limit(self) -> None:
        service = _FakeUserSpaceService(Path("/tmp/unused"))
        request = _build_body_request(
            _build_zip_bytes({"one.txt": b"1", "two.txt": b"2"}),
            content_type="application/zip",
        )

        with (
            mock.patch.object(runtime_routes, "_userspace_service", return_value=service),
            mock.patch.object(runtime_routes, "_get_primitive_upload_max_bytes", mock.AsyncMock(return_value=1024 * 1024)),
            mock.patch.object(runtime_routes, "_get_primitive_archive_max_entries", mock.AsyncMock(return_value=1)),
        ):
            with self.assertRaises(HTTPException) as exc:
                await runtime_routes._primitive_archive_extract("workspace-1", request, "user-1")

        self.assertEqual(exc.exception.status_code, 413)
        detail = str(exc.exception.detail)
        self.assertIn("configured User Space primitive archive file limit", detail)
        self.assertIn("1", detail)

    async def test_job_primitive_updates_and_reads_state(self) -> None:
        service = _FakeUserSpaceService(Path("/tmp/unused"))
        runtime_routes._PRIMITIVE_JOBS.clear()

        with mock.patch.object(runtime_routes, "_userspace_service", return_value=service):
            created = await runtime_routes._primitive_job_create(
                "workspace-1",
                {"job_id": "import-1", "kind": "archive", "status": "running"},
                "user-1",
            )
            updated = await runtime_routes._primitive_job_put(
                "workspace-1",
                "import-1",
                {"status": "completed", "result": {"count": 2}},
                "user-1",
            )
            fetched = await runtime_routes._primitive_job_get("workspace-1", "import-1", "user-1")

        self.assertEqual(created["job_id"], "import-1")
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(fetched["result"], {"count": 2})

    async def test_direct_subdomain_preview_session_resolves_from_registry(self) -> None:
        workspace_id = "31eccb83-4529-4e71-b575-3e510826fcc6"
        host = f"{workspace_id}.ragtime.hammerton.com"
        claims = {
            "workspace_id": workspace_id,
            "sub": "user-1",
            "preview_mode": "workspace",
            "preview_host": host,
        }
        preview_host._preview_host_sessions.clear()
        preview_host._preview_host_session_order.clear()
        preview_host._preview_host_expiry_heap.clear()

        with mock.patch.object(preview_host, "_runtime_service", return_value=_FakeRuntimeService()):
            await preview_host._register_preview_session(
                host,
                claims,
                datetime.now(timezone.utc) + timedelta(minutes=5),
            )
            resolved = await preview_host._resolve_preview_session(host, None)

        self.assertEqual(resolved["workspace_id"], workspace_id)
        self.assertEqual(resolved["preview_mode"], "workspace")

    async def test_preview_file_primitive_uses_workspace_subdomain_session(self) -> None:
        claims = {
            "workspace_id": "workspace-1",
            "sub": "user-1",
            "preview_mode": "workspace",
        }

        async def fake_verify(_request: Request) -> dict[str, str]:
            return claims

        with (
            mock.patch.object(preview_host, "_verify_preview_session_cookie", fake_verify),
            mock.patch.object(
                preview_host,
                "_primitive_workspace_file_write_request",
                mock.AsyncMock(return_value={"path": "uploads/payload.bin"}),
            ) as file_write,
        ):
            result = await preview_host.preview_file_write(
                _build_request(),
                "uploads/payload.bin",
            )

        self.assertEqual(result["path"], "uploads/payload.bin")
        file_write.assert_awaited_once()

    async def test_preview_write_primitives_reject_shared_public_sessions(self) -> None:
        async def fake_verify(_request: Request) -> dict[str, str]:
            return {
                "workspace_id": "workspace-1",
                "preview_mode": "shared_public_host",
            }

        with mock.patch.object(preview_host, "_verify_preview_session_cookie", fake_verify):
            with self.assertRaises(HTTPException) as exc:
                await preview_host.preview_file_write(
                    _build_request(),
                    "uploads/payload.bin",
                )

        self.assertEqual(exc.exception.status_code, 403)

    async def test_preview_archive_extract_rejects_shared_public_sessions(self) -> None:
        async def fake_verify(_request: Request) -> dict[str, str]:
            return {
                "workspace_id": "workspace-1",
                "preview_mode": "shared_public_host",
            }

        with mock.patch.object(preview_host, "_verify_preview_session_cookie", fake_verify):
            with self.assertRaises(HTTPException) as exc:
                await preview_host.preview_archive_extract(_build_body_request(b"payload"))

        self.assertEqual(exc.exception.status_code, 403)

    async def test_preview_parse_primitive_allows_authorized_shared_sessions(self) -> None:
        upload = _make_upload_file("support.xlsx", b"payload", "application/octet-stream")

        async def fake_verify(_request: Request) -> dict[str, str]:
            return {
                "workspace_id": "workspace-1",
                "preview_mode": "shared_public_host",
            }

        with (
            mock.patch.object(preview_host, "_verify_preview_session_cookie", fake_verify),
            mock.patch.object(
                preview_host,
                "_parse_uploaded_document_upload",
                mock.AsyncMock(return_value={"text": "parsed"}),
            ) as parse_upload,
        ):
            result = await preview_host.preview_parse_document(_build_request(), upload)

        self.assertEqual(result["text"], "parsed")
        parse_upload.assert_awaited_once()
