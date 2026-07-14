import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict, cast

from ragtime.userspace.cloud_mounts import (
    GOOGLE_DRIVE_SCOPE,
    CloudMountProvider,
    google_drive_api_disabled_error_message,
    google_drive_scope_error_message,
    microsoft_graph_permission_error_message,
)


class _StubResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _StubCloudMountProvider(CloudMountProvider):
    def __init__(self, provider: str, responses: list[dict[str, Any]]) -> None:
        super().__init__(provider, {"access_token": "token"})  # type: ignore[arg-type]
        self.responses = responses
        self.requests: list[tuple[str, dict[str, Any]]] = []

    async def _request_json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        self.requests.append((url, kwargs))
        return self.responses.pop(0)


class _CloudListCase(TypedDict):
    label: str
    provider: str
    responses: list[dict[str, Any]]
    path: str
    expected_names: list[str]
    expected_paths: list[str]
    expected_url: str
    expected_params: dict[str, str] | None


class CloudMountProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_cloud_listing_happy_path_cases(self) -> None:
        cases: list[_CloudListCase] = [
            {
                "label": "microsoft root",
                "provider": "microsoft_drive",
                "responses": [
                    {
                        "value": [
                            {"id": "drive-b", "name": "Shared Docs"},
                            {"id": "drive-a", "name": "OneDrive"},
                        ]
                    }
                ],
                "path": ".",
                "expected_names": ["OneDrive", "Shared Docs"],
                "expected_paths": ["drives/drive-a", "drives/drive-b"],
                "expected_url": "https://graph.microsoft.com/v1.0/me/drives",
                "expected_params": None,
            },
            {
                "label": "microsoft drive children",
                "provider": "microsoft_drive",
                "responses": [{"value": [{"name": "Projects", "folder": {}, "size": 0}]}],
                "path": "drives/drive-a",
                "expected_names": ["Projects"],
                "expected_paths": ["drives/drive-a/Projects"],
                "expected_url": "https://graph.microsoft.com/v1.0/drives/drive-a/root/children",
                "expected_params": None,
            },
            {
                "label": "google root",
                "provider": "google_drive",
                "responses": [
                    {
                        "drives": [
                            {"id": "shared-b", "name": "Team B"},
                            {"id": "shared-a", "name": "Team A"},
                        ]
                    }
                ],
                "path": ".",
                "expected_names": ["My Drive", "Team A", "Team B"],
                "expected_paths": ["my-drive", "drives/shared-a", "drives/shared-b"],
                "expected_url": "https://www.googleapis.com/drive/v3/drives",
                "expected_params": None,
            },
            {
                "label": "google shared drive children",
                "provider": "google_drive",
                "responses": [
                    {
                        "files": [
                            {
                                "id": "folder-1",
                                "name": "Projects",
                                "mimeType": "application/vnd.google-apps.folder",
                            }
                        ]
                    }
                ],
                "path": "drives/shared-a",
                "expected_names": ["Projects"],
                "expected_paths": ["drives/shared-a/Projects"],
                "expected_url": "https://www.googleapis.com/drive/v3/files",
                "expected_params": {"corpora": "drive", "driveId": "shared-a"},
            },
        ]

        for case in cases:
            with self.subTest(label=case["label"]):
                provider = _StubCloudMountProvider(case["provider"], case["responses"])
                entries = await provider.list_dir(case["path"])

                self.assertEqual([entry.name for entry in entries], case["expected_names"])
                self.assertEqual([entry.path for entry in entries], case["expected_paths"])
                self.assertEqual(provider.requests[0][0], case["expected_url"])
                if case["expected_params"] is not None:
                    request_params = provider.requests[0][1]["params"]
                    for key, expected in case["expected_params"].items():
                        self.assertEqual(request_params[key], expected)

    async def test_google_missing_drive_scope_fails_before_provider_request(self) -> None:
        provider = CloudMountProvider(
            "google_drive",
            {
                "access_token": "token",
                "scopes": ["openid", "https://www.googleapis.com/auth/userinfo.email"],
            },
        )

        with self.assertRaisesRegex(RuntimeError, "missing Drive read/write permission"):
            await provider.list_dir(".")

        self.assertIn(GOOGLE_DRIVE_SCOPE, google_drive_scope_error_message())

    async def test_google_create_folder_in_my_drive_uses_drive_root_parent(self) -> None:
        provider = _StubCloudMountProvider(
            "google_drive",
            [
                {"id": "created-folder", "name": "New Project"},
            ],
        )

        await provider.create_dir("my-drive/New Project")

        url, kwargs = provider.requests[0]
        self.assertEqual(url, "https://www.googleapis.com/drive/v3/files")
        self.assertEqual(kwargs["params"]["supportsAllDrives"], "true")
        self.assertEqual(
            kwargs["json"],
            {
                "name": "New Project",
                "mimeType": "application/vnd.google-apps.folder",
                "parents": ["root"],
            },
        )

    async def test_google_create_folder_rejects_virtual_provider_root(self) -> None:
        provider = _StubCloudMountProvider("google_drive", [])

        with self.assertRaisesRegex(IsADirectoryError, "Select My Drive or a shared drive"):
            await provider.create_dir("New Project")

        self.assertEqual(provider.requests, [])

    async def test_google_create_folder_in_shared_drive_preserves_drive_context(self) -> None:
        provider = _StubCloudMountProvider(
            "google_drive",
            [
                {"id": "created-folder", "name": "Team Project"},
            ],
        )

        await provider.create_dir("drives/shared-a/Team Project")

        _url, kwargs = provider.requests[0]
        self.assertNotIn("driveId", kwargs["params"])
        self.assertEqual(kwargs["json"]["parents"], ["shared-a"])

    async def test_google_drive_api_disabled_error_is_actionable(self) -> None:
        provider = CloudMountProvider("google_drive", {"access_token": "token"})
        error = provider._provider_request_error(  # type: ignore[arg-type]
            cast(
                Any,
                _StubResponse(
                    403,
                    "Google Drive API has not been used in project 742727405992 before or it is disabled. "
                    "Enable it by visiting https://console.developers.google.com/apis/api/drive.googleapis.com/overview?project=742727405992",
                ),
            )
        )

        self.assertEqual(error, google_drive_api_disabled_error_message())
        self.assertIn("Enable the Google Drive API", error)

    async def test_microsoft_graph_permission_error_is_actionable(self) -> None:
        provider = CloudMountProvider("microsoft_drive", {"access_token": "token"})
        error = provider._provider_request_error(  # type: ignore[arg-type]
            cast(
                Any,
                _StubResponse(
                    403,
                    '{"error":{"code":"accessDenied","message":"Access denied"}}',
                ),
            )
        )

        self.assertEqual(error, microsoft_graph_permission_error_message())
        self.assertIn("admin consent", error)

    async def test_mock_cloud_merge_uploads_local_only_files(self) -> None:
        provider = CloudMountProvider(
            "google_drive",
            {
                "mock_tree": [
                    {"path": "my-drive", "is_dir": True},
                ]
            },
        )
        with TemporaryDirectory() as temp_dir:
            local_root = Path(temp_dir)
            local_file = local_root / "notes.txt"
            local_file.write_text("hello from workspace\n", encoding="utf-8")
            os.utime(local_file, (1_700_000_000, 1_700_000_000))

            result = await provider.sync_tree("my-drive", local_root, sync_mode="merge")

            self.assertTrue(result.success, result.errors)
            self.assertEqual(result.files_synced, 1)
            self.assertEqual(await provider.read_file("my-drive/notes.txt"), b"hello from workspace\n")

    async def test_mock_cloud_target_authoritative_deletes_remote_only_files(self) -> None:
        provider = CloudMountProvider(
            "google_drive",
            {
                "mock_tree": [
                    {"path": "my-drive", "is_dir": True},
                    {"path": "my-drive/remove.txt", "content": "delete me", "modified_at": "2024-01-01T00:00:00Z"},
                ]
            },
        )
        with TemporaryDirectory() as temp_dir:
            local_root = Path(temp_dir)

            preview = await provider.preview_sync_tree("my-drive", local_root, sync_mode="target_authoritative")
            result = await provider.sync_tree("my-drive", local_root, sync_mode="target_authoritative")

            self.assertTrue(preview.success, preview.errors)
            self.assertEqual(preview.delete_from_source_paths, ["remove.txt"])
            self.assertTrue(result.success, result.errors)
            with self.assertRaises(FileNotFoundError):
                await provider.read_file("my-drive/remove.txt")


if __name__ == "__main__":
    unittest.main()
