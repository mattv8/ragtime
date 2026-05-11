from __future__ import annotations

import hashlib
import asyncio
import json
import mimetypes
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import urlencode

import httpx
import posixpath

from ragtime.config import settings
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

CloudProviderName = Literal["microsoft_drive", "google_drive"]
VIRTUAL_DRIVES_SEGMENT = "drives"
GOOGLE_MY_DRIVE_SEGMENT = "my-drive"
GOOGLE_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
GOOGLE_DRIVE_READ_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
GOOGLE_DRIVE_COMPATIBLE_SCOPES = {
    GOOGLE_DRIVE_SCOPE,
    GOOGLE_DRIVE_READ_SCOPE,
}


@dataclass(frozen=True)
class CloudMountEntry:
    name: str
    path: str
    is_dir: bool
    size: int | None = None
    modified_at: datetime | None = None


CloudSyncMode = Literal["merge", "source_authoritative", "target_authoritative"]
CloudSyncProgressCallback = Callable[[int, int | None, str | None], Awaitable[None]]


@dataclass
class CloudSyncResult:
    files_synced: int
    errors: list[str]
    success: bool
    backend_used: str = "cloud_virtual"
    notice: str | None = None


@dataclass
class CloudSyncPreviewResult:
    sync_mode: CloudSyncMode
    delete_from_source_count: int
    delete_from_target_count: int
    delete_from_source_paths: list[str]
    delete_from_target_paths: list[str]
    state_fingerprint: str
    errors: list[str]
    success: bool
    backend_used: str = "cloud_virtual"
    notice: str | None = None
    sample_limit: int = 200


def normalize_cloud_path(value: str | None) -> str:
    raw = str(value or ".").strip().replace("\\", "/")
    if raw in {"", "/", "."}:
        return "."
    normalized = posixpath.normpath(raw.lstrip("/"))
    if normalized in {"", "."}:
        return "."
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError("Cloud mount paths must stay within the selected drive root")
    return normalized


def join_cloud_path(parent: str, name: str) -> str:
    parent = normalize_cloud_path(parent)
    safe_name = str(name or "").strip().strip("/")
    if not safe_name:
        return parent
    if parent == ".":
        return normalize_cloud_path(safe_name)
    return normalize_cloud_path(f"{parent}/{safe_name}")


def browser_path_for_cloud(path: str) -> str:
    normalized = normalize_cloud_path(path)
    return "/" if normalized == "." else f"/{normalized}"


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_size(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_sync_mode(value: str | None) -> CloudSyncMode:
    if value == "source_authoritative":
        return "source_authoritative"
    if value == "target_authoritative":
        return "target_authoritative"
    return "merge"


def _datetime_to_epoch_seconds(value: datetime | None) -> int:
    if value is None:
        return 0
    parsed = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _epoch_seconds_to_rfc3339(value: int | float) -> str:
    return datetime.fromtimestamp(max(0, float(value)), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _set_file_mtime(path: Path, modified_at: datetime | None) -> None:
    if modified_at is None:
        return
    timestamp = float(_datetime_to_epoch_seconds(modified_at))
    try:
        os.utime(path, (timestamp, timestamp))
    except OSError:
        pass


def _scan_local_tree(local_root: Path) -> tuple[dict[str, tuple[int, int]], set[str], list[str]]:
    errors: list[str] = []
    files: dict[str, tuple[int, int]] = {}
    directories: set[str] = {""}
    if not local_root.exists():
        return files, directories, errors
    for root, _dirnames, filenames in os.walk(local_root):
        root_path = Path(root)
        try:
            relative_dir = root_path.relative_to(local_root)
        except ValueError:
            continue
        relative_dir_text = "" if str(relative_dir) == "." else relative_dir.as_posix()
        directories.add(relative_dir_text)
        for filename in filenames:
            path = root_path / filename
            try:
                stat_result = path.stat()
            except OSError as exc:
                errors.append(f"stat {path}: {exc}")
                continue
            relative_file = filename if not relative_dir_text else f"{relative_dir_text}/{filename}"
            files[relative_file] = (int(stat_result.st_size), int(stat_result.st_mtime))
    return files, directories, errors


def _leaf_delete_directories(directories: set[str], files: set[str]) -> list[str]:
    leaf_dirs: list[str] = []
    for directory in sorted(path for path in directories if path):
        prefix = f"{directory}/"
        if any(other != directory and other.startswith(prefix) for other in directories):
            continue
        if any(path.startswith(prefix) for path in files):
            continue
        leaf_dirs.append(f"{directory}/")
    return leaf_dirs


def _collect_delete_paths(files: set[str], directories: set[str]) -> list[str]:
    paths = sorted(files)
    paths.extend(_leaf_delete_directories(directories, files))
    return paths


def _build_tree_state_fingerprint(
    remote_files: dict[str, tuple[int, int]],
    remote_dirs: set[str],
    local_files: dict[str, tuple[int, int]],
    local_dirs: set[str],
) -> str:
    digest = hashlib.sha256()
    for label, files, directories in (
        ("remote", remote_files, remote_dirs),
        ("local", local_files, local_dirs),
    ):
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        for directory in sorted(directories):
            digest.update(b"d\0")
            digest.update(directory.encode("utf-8", errors="ignore"))
            digest.update(b"\0")
        for path, (size_bytes, mtime_seconds) in sorted(files.items()):
            digest.update(b"f\0")
            digest.update(path.encode("utf-8", errors="ignore"))
            digest.update(b"\0")
            digest.update(f"{size_bytes}:{mtime_seconds}".encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _preview_cloud_sync_from_metadata(
    remote_files: dict[str, tuple[int, int]],
    remote_dirs: set[str],
    local_files: dict[str, tuple[int, int]],
    local_dirs: set[str],
    *,
    sync_mode: CloudSyncMode,
    sample_limit: int,
    errors: list[str],
) -> CloudSyncPreviewResult:
    delete_from_source_paths: list[str] = []
    delete_from_target_paths: list[str] = []
    if sync_mode == "source_authoritative":
        delete_from_target_paths = _collect_delete_paths(set(local_files) - set(remote_files), local_dirs - remote_dirs)
    elif sync_mode == "target_authoritative":
        delete_from_source_paths = _collect_delete_paths(set(remote_files) - set(local_files), remote_dirs - local_dirs)
    return CloudSyncPreviewResult(
        sync_mode=sync_mode,
        delete_from_source_count=len(delete_from_source_paths),
        delete_from_target_count=len(delete_from_target_paths),
        delete_from_source_paths=delete_from_source_paths[:sample_limit],
        delete_from_target_paths=delete_from_target_paths[:sample_limit],
        state_fingerprint=_build_tree_state_fingerprint(remote_files, remote_dirs, local_files, local_dirs),
        errors=errors,
        success=len(errors) == 0,
        sample_limit=sample_limit,
    )


def _google_query_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def has_google_drive_scope(scopes: Any) -> bool:
    if isinstance(scopes, str):
        values = scopes.split()
    elif isinstance(scopes, (list, tuple, set)):
        values = [str(scope) for scope in scopes]
    else:
        values = []
    return any(scope in GOOGLE_DRIVE_COMPATIBLE_SCOPES for scope in values)


def has_google_drive_write_scope(scopes: Any) -> bool:
    if isinstance(scopes, str):
        values = {scope for scope in scopes.split() if scope}
    elif isinstance(scopes, list):
        values = {str(scope) for scope in scopes if scope}
    else:
        values = set()
    return GOOGLE_DRIVE_SCOPE in values


def google_drive_scope_error_message() -> str:
    return (
        "Google Drive account is missing Drive read/write permission. Remove and reconnect "
        f"the OAuth account after adding {GOOGLE_DRIVE_SCOPE} to the Google OAuth consent screen."
    )


def google_drive_write_scope_error_message() -> str:
    return (
        "Google Drive account is missing Drive read/write permission. Remove and reconnect "
        f"the OAuth account after adding {GOOGLE_DRIVE_SCOPE} to the Google OAuth consent screen."
    )


def google_drive_api_disabled_error_message() -> str:
    return (
        "Google Drive API is disabled for the Google Cloud project that owns this OAuth client. "
        "Enable the Google Drive API in Google Cloud Console for that project, then retry. "
        "If you enabled it recently, wait a few minutes before reconnecting."
    )


def microsoft_graph_permission_error_message() -> str:
    return (
        "OneDrive/SharePoint account is missing Microsoft Graph permission or tenant consent. "
        "Add delegated permissions offline_access, User.Read, Files.ReadWrite.All, and Sites.ReadWrite.All, "
        "grant admin consent if required by the tenant, then reconnect the OAuth account."
    )


def microsoft_tenant_endpoint_error_message() -> str:
    return (
        "OneDrive/SharePoint OAuth needs a tenant-specific Microsoft endpoint for single-tenant app registrations. "
        "Set CLOUD_MOUNT_MICROSOFT_TENANT_ID to the Azure Directory (tenant) ID or primary tenant domain, "
        "then restart Ragtime. Use 'common' or 'organizations' only if the Azure app registration is configured as multi-tenant."
    )


def _microsoft_oauth_tenant() -> str:
    tenant = str(settings.cloud_mount_microsoft_tenant_id or "").strip()
    if not tenant:
        raise RuntimeError(microsoft_tenant_endpoint_error_message())
    return tenant


def _mock_tree_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tree = config.get("mock_tree")
    if isinstance(raw_tree, str):
        try:
            raw_tree = json.loads(raw_tree)
        except json.JSONDecodeError:
            raw_tree = []
    if isinstance(raw_tree, list) and raw_tree:
        return [item for item in raw_tree if isinstance(item, dict)]
    return [
        {"path": "Documents", "is_dir": True},
        {"path": "Documents/Project Plan.md", "content": "# Project Plan\n\nMock cloud mount file.\n"},
        {"path": "Reports", "is_dir": True},
        {"path": "Reports/Q1.txt", "content": "Mock report from cloud provider.\n"},
    ]


class CloudMountProvider:
    def __init__(self, provider: CloudProviderName, config: dict[str, Any]):
        self.provider: CloudProviderName = provider
        self.config = dict(config)

    @property
    def is_mock(self) -> bool:
        mode = str(self.config.get("auth_mode") or self.config.get("mode") or "").lower()
        token = str(self.config.get("access_token") or "")
        return mode == "mock" or token.startswith("mock-") or bool(self.config.get("mock_tree"))

    async def list_dir(self, source_path: str) -> list[CloudMountEntry]:
        if self.is_mock:
            return self._list_mock_dir(source_path)
        if self.provider == "microsoft_drive":
            return await self._list_microsoft_dir(source_path)
        return await self._list_google_dir(source_path)

    async def download_tree(
        self,
        source_path: str,
        destination: Path,
        *,
        progress_callback: CloudSyncProgressCallback | None = None,
        progress_total: int | None = None,
        progress_done_offset: int = 0,
    ) -> int:
        await asyncio.to_thread(destination.mkdir, parents=True, exist_ok=True)
        if self.is_mock:
            return self._download_mock_tree(source_path, destination)

        written = 0
        entries = await self.list_dir(source_path)
        for entry in entries:
            relative = normalize_cloud_path(entry.path)
            root = normalize_cloud_path(source_path)
            if root != "." and relative.startswith(root + "/"):
                relative = relative[len(root) + 1 :]
            target = destination / ("." if relative == "." else relative)
            if entry.is_dir:
                await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)
                written += await self.download_tree(
                    entry.path,
                    target,
                    progress_callback=progress_callback,
                    progress_total=progress_total,
                    progress_done_offset=progress_done_offset + written,
                )
            else:
                await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
                content = await self.read_file(entry.path)
                await asyncio.to_thread(target.write_bytes, content)
                _set_file_mtime(target, entry.modified_at)
                written += 1
                if progress_callback is not None:
                    await progress_callback(
                        progress_done_offset + written,
                        progress_total,
                        f"Downloaded {relative}",
                    )
        return written

    async def read_file(self, source_path: str) -> bytes:
        if self.is_mock:
            return self._read_mock_file(source_path)
        if self.provider == "microsoft_drive":
            return await self._read_microsoft_file(source_path)
        return await self._read_google_file(source_path)

    async def create_dir(self, source_path: str) -> None:
        if self.is_mock:
            self._create_mock_dir(source_path)
            return
        if self.provider == "microsoft_drive":
            await self._create_microsoft_dir(source_path)
            return
        await self._create_google_dir(source_path)

    async def ensure_dir(self, source_path: str) -> None:
        if self.is_mock:
            self._create_mock_dir(source_path)
            return
        if self.provider == "microsoft_drive":
            await self._ensure_microsoft_dir(source_path)
            return
        await self._ensure_google_dir(source_path)

    async def upload_file(self, source_path: str, local_path: Path) -> None:
        if self.is_mock:
            content, mtime = await asyncio.to_thread(
                lambda: (local_path.read_bytes(), int(local_path.stat().st_mtime))
            )
            self._write_mock_file(source_path, content, mtime)
            return
        if self.provider == "microsoft_drive":
            await self._upload_microsoft_file(source_path, local_path)
            return
        await self._upload_google_file(source_path, local_path)

    async def delete_path(self, source_path: str) -> None:
        if self.is_mock:
            self._delete_mock_path(source_path)
            return
        if self.provider == "microsoft_drive":
            await self._delete_microsoft_path(source_path)
            return
        await self._delete_google_path(source_path)

    async def scan_tree(self, source_path: str) -> tuple[dict[str, tuple[int, int]], set[str], list[str]]:
        errors: list[str] = []
        files: dict[str, tuple[int, int]] = {}
        directories: set[str] = {""}

        async def _scan(path: str, relative_dir: str) -> None:
            try:
                entries = await self.list_dir(path)
            except Exception as exc:
                errors.append(f"list {path}: {exc}")
                return
            for entry in entries:
                relative_path = entry.name if not relative_dir else f"{relative_dir}/{entry.name}"
                if entry.is_dir:
                    directories.add(relative_path)
                    await _scan(entry.path, relative_path)
                else:
                    files[relative_path] = (
                        int(entry.size or 0),
                        _datetime_to_epoch_seconds(entry.modified_at),
                    )

        await _scan(source_path, "")
        return files, directories, errors

    async def preview_sync_tree(
        self,
        source_path: str,
        local_path: Path,
        *,
        sync_mode: str | None = "merge",
        sample_limit: int = 200,
    ) -> CloudSyncPreviewResult:
        normalized_mode = _normalize_sync_mode(sync_mode)
        try:
            remote_files, remote_dirs, remote_errors = await self.scan_tree(source_path)
            local_files, local_dirs, local_errors = await asyncio.to_thread(
                _scan_local_tree,
                local_path,
            )
            return _preview_cloud_sync_from_metadata(
                remote_files,
                remote_dirs,
                local_files,
                local_dirs,
                sync_mode=normalized_mode,
                sample_limit=max(1, sample_limit),
                errors=[*remote_errors, *local_errors],
            )
        except Exception as exc:
            return CloudSyncPreviewResult(
                sync_mode=normalized_mode,
                delete_from_source_count=0,
                delete_from_target_count=0,
                delete_from_source_paths=[],
                delete_from_target_paths=[],
                state_fingerprint="",
                errors=[f"Cloud sync preview error: {exc}"],
                success=False,
                sample_limit=max(1, sample_limit),
            )

    async def sync_tree(
        self,
        source_path: str,
        local_path: Path,
        *,
        sync_mode: str | None = "merge",
        progress_callback: CloudSyncProgressCallback | None = None,
    ) -> CloudSyncResult:
        normalized_mode = _normalize_sync_mode(sync_mode)
        if normalized_mode == "source_authoritative":
            return await self._sync_tree_source_authoritative(
                source_path,
                local_path,
                progress_callback=progress_callback,
            )
        if normalized_mode == "target_authoritative":
            return await self._sync_tree_target_authoritative(
                source_path,
                local_path,
                progress_callback=progress_callback,
            )
        return await self._sync_tree_merge(
            source_path,
            local_path,
            progress_callback=progress_callback,
        )

    async def _sync_tree_source_authoritative(
        self,
        source_path: str,
        local_path: Path,
        *,
        progress_callback: CloudSyncProgressCallback | None = None,
    ) -> CloudSyncResult:
        temp_path = local_path.parent / f".{local_path.name}.cloud-sync"
        errors: list[str] = []
        files_synced = 0
        try:
            remote_files, _remote_dirs, remote_errors = await self.scan_tree(source_path)
            errors.extend(remote_errors)
            files_total = len(remote_files)
            if progress_callback is not None:
                await progress_callback(0, files_total, "Downloading cloud files")
            if temp_path.exists():
                await asyncio.to_thread(shutil.rmtree, temp_path)
            await asyncio.to_thread(temp_path.mkdir, parents=True, exist_ok=True)
            files_synced = await self.download_tree(
                source_path,
                temp_path,
                progress_callback=progress_callback,
                progress_total=files_total,
            )
            if local_path.is_symlink() or local_path.is_file():
                await asyncio.to_thread(local_path.unlink, missing_ok=True)
            elif local_path.exists():
                await asyncio.to_thread(shutil.rmtree, local_path)
            await asyncio.to_thread(temp_path.replace, local_path)
        except Exception as exc:
            errors.append(f"cloud pull sync error: {exc}")
        finally:
            if temp_path.exists():
                await asyncio.to_thread(shutil.rmtree, temp_path, ignore_errors=True)
        return CloudSyncResult(files_synced=files_synced, errors=errors, success=len(errors) == 0)

    async def _sync_tree_merge(
        self,
        source_path: str,
        local_path: Path,
        *,
        progress_callback: CloudSyncProgressCallback | None = None,
    ) -> CloudSyncResult:
        errors: list[str] = []
        files_synced = 0
        remote_files, remote_dirs, remote_errors = await self.scan_tree(source_path)
        local_files, local_dirs, local_errors = await asyncio.to_thread(
            _scan_local_tree,
            local_path,
        )
        errors.extend(remote_errors)
        errors.extend(local_errors)
        await asyncio.to_thread(local_path.mkdir, parents=True, exist_ok=True)
        files_total = 0
        for relative_path in sorted(set(remote_files) | set(local_files)):
            remote_meta = remote_files.get(relative_path)
            local_meta = local_files.get(relative_path)
            if remote_meta is None or local_meta is None:
                files_total += 1
                continue
            remote_size, remote_mtime = remote_meta
            local_size, local_mtime = local_meta
            if remote_mtime != local_mtime or remote_size != local_size:
                files_total += 1
        if progress_callback is not None:
            await progress_callback(0, files_total, "Syncing cloud files")

        for relative_dir in sorted(remote_dirs - local_dirs):
            if relative_dir:
                await asyncio.to_thread((local_path / relative_dir).mkdir, parents=True, exist_ok=True)
        for relative_dir in sorted(local_dirs - remote_dirs):
            if not relative_dir:
                continue
            try:
                await self.ensure_dir(join_cloud_path(source_path, relative_dir))
            except Exception as exc:
                errors.append(f"mkdir {relative_dir}: {exc}")

        for relative_path in sorted(set(remote_files) | set(local_files)):
            remote_meta = remote_files.get(relative_path)
            local_meta = local_files.get(relative_path)
            try:
                if remote_meta is None and local_meta is not None:
                    await self.upload_file(join_cloud_path(source_path, relative_path), local_path / relative_path)
                    files_synced += 1
                    if progress_callback is not None:
                        await progress_callback(files_synced, files_total, f"Uploaded {relative_path}")
                    continue
                if local_meta is None and remote_meta is not None:
                    target = local_path / relative_path
                    await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
                    remote_entry_path = join_cloud_path(source_path, relative_path)
                    content = await self.read_file(remote_entry_path)
                    await asyncio.to_thread(target.write_bytes, content)
                    os.utime(target, (remote_meta[1], remote_meta[1]))
                    files_synced += 1
                    if progress_callback is not None:
                        await progress_callback(files_synced, files_total, f"Downloaded {relative_path}")
                    continue
                if remote_meta is None or local_meta is None:
                    continue
                remote_size, remote_mtime = remote_meta
                local_size, local_mtime = local_meta
                if remote_mtime > local_mtime:
                    target = local_path / relative_path
                    await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
                    content = await self.read_file(join_cloud_path(source_path, relative_path))
                    await asyncio.to_thread(target.write_bytes, content)
                    os.utime(target, (remote_mtime, remote_mtime))
                    files_synced += 1
                elif local_mtime > remote_mtime:
                    await self.upload_file(join_cloud_path(source_path, relative_path), local_path / relative_path)
                    files_synced += 1
                elif remote_size != local_size:
                    target = local_path / relative_path
                    await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
                    content = await self.read_file(join_cloud_path(source_path, relative_path))
                    await asyncio.to_thread(target.write_bytes, content)
                    os.utime(target, (remote_mtime, remote_mtime))
                    files_synced += 1
                if progress_callback is not None:
                    await progress_callback(files_synced, files_total, f"Synced {relative_path}")
            except Exception as exc:
                errors.append(f"sync {relative_path}: {exc}")
        return CloudSyncResult(files_synced=files_synced, errors=errors, success=len(errors) == 0)

    async def _sync_tree_target_authoritative(
        self,
        source_path: str,
        local_path: Path,
        *,
        progress_callback: CloudSyncProgressCallback | None = None,
    ) -> CloudSyncResult:
        errors: list[str] = []
        files_synced = 0
        remote_files, remote_dirs, remote_errors = await self.scan_tree(source_path)
        local_files, local_dirs, local_errors = await asyncio.to_thread(
            _scan_local_tree,
            local_path,
        )
        errors.extend(remote_errors)
        errors.extend(local_errors)
        await asyncio.to_thread(local_path.mkdir, parents=True, exist_ok=True)
        files_total = sum(
            1
            for relative_path in local_files
            if remote_files.get(relative_path) != local_files[relative_path]
        ) + len(set(remote_files) - set(local_files)) + len([path for path in remote_dirs - local_dirs if path])
        if progress_callback is not None:
            await progress_callback(0, files_total, "Uploading cloud files")

        for relative_dir in sorted(local_dirs - remote_dirs):
            if not relative_dir:
                continue
            try:
                await self.ensure_dir(join_cloud_path(source_path, relative_dir))
            except Exception as exc:
                errors.append(f"mkdir {relative_dir}: {exc}")
        for relative_path in sorted(local_files):
            try:
                if remote_files.get(relative_path) != local_files[relative_path]:
                    await self.upload_file(join_cloud_path(source_path, relative_path), local_path / relative_path)
                    files_synced += 1
                    if progress_callback is not None:
                        await progress_callback(files_synced, files_total, f"Uploaded {relative_path}")
            except Exception as exc:
                errors.append(f"sync {relative_path}: {exc}")
        for relative_path in sorted(set(remote_files) - set(local_files)):
            try:
                await self.delete_path(join_cloud_path(source_path, relative_path))
                files_synced += 1
                if progress_callback is not None:
                    await progress_callback(files_synced, files_total, f"Deleted {relative_path}")
            except Exception as exc:
                errors.append(f"delete {relative_path}: {exc}")
        for relative_dir in sorted((path for path in remote_dirs - local_dirs if path), key=lambda path: (path.count("/"), path), reverse=True):
            try:
                await self.delete_path(join_cloud_path(source_path, relative_dir))
                files_synced += 1
                if progress_callback is not None:
                    await progress_callback(files_synced, files_total, f"Deleted {relative_dir}")
            except Exception as exc:
                errors.append(f"rmdir {relative_dir}: {exc}")
        return CloudSyncResult(files_synced=files_synced, errors=errors, success=len(errors) == 0)

    def _list_mock_dir(self, source_path: str) -> list[CloudMountEntry]:
        source = normalize_cloud_path(source_path)
        children: dict[str, CloudMountEntry] = {}
        for item in _mock_tree_entries(self.config):
            path = normalize_cloud_path(str(item.get("path") or ""))
            if path == ".":
                continue
            parts = path.split("/")
            parent = "." if len(parts) == 1 else "/".join(parts[:-1])
            if parent != source:
                continue
            name = parts[-1]
            is_dir = bool(item.get("is_dir")) or "content" not in item
            children[name] = CloudMountEntry(
                name=name,
                path=path,
                is_dir=is_dir,
                size=int(item.get("size") or len(str(item.get("content") or ""))) if not is_dir else None,
                modified_at=_parse_datetime(item.get("modified_at")),
            )
        return sorted(children.values(), key=lambda entry: (not entry.is_dir, entry.name.lower()))

    def _read_mock_file(self, source_path: str) -> bytes:
        source = normalize_cloud_path(source_path)
        for item in _mock_tree_entries(self.config):
            if normalize_cloud_path(str(item.get("path") or "")) == source:
                content = item.get("content", "")
                return content if isinstance(content, bytes) else str(content).encode("utf-8")
        raise FileNotFoundError(source)

    def _write_mock_file(self, source_path: str, content: bytes, mtime_seconds: int) -> None:
        source = normalize_cloud_path(source_path)
        tree = self.config.setdefault("mock_tree", [])
        if not isinstance(tree, list):
            tree = []
            self.config["mock_tree"] = tree
        parent = "." if "/" not in source else source.rsplit("/", 1)[0]
        if parent != ".":
            self._create_mock_dir(parent)
        for item in tree:
            if isinstance(item, dict) and normalize_cloud_path(str(item.get("path") or "")) == source:
                item["content"] = content.decode("utf-8", errors="replace")
                item["size"] = len(content)
                item["modified_at"] = _epoch_seconds_to_rfc3339(mtime_seconds)
                item.pop("is_dir", None)
                return
        tree.append({"path": source, "content": content.decode("utf-8", errors="replace"), "size": len(content), "modified_at": _epoch_seconds_to_rfc3339(mtime_seconds)})

    def _create_mock_dir(self, source_path: str) -> None:
        source = normalize_cloud_path(source_path)
        if source == ".":
            raise IsADirectoryError("Cannot create the cloud provider root")
        tree = self.config.setdefault("mock_tree", [])
        if not isinstance(tree, list):
            tree = []
            self.config["mock_tree"] = tree
        if not any(normalize_cloud_path(str(item.get("path") or "")) == source for item in tree if isinstance(item, dict)):
            tree.append({"path": source, "is_dir": True})

    def _delete_mock_path(self, source_path: str) -> None:
        source = normalize_cloud_path(source_path)
        tree = self.config.get("mock_tree")
        if not isinstance(tree, list):
            return
        prefix = f"{source}/"
        self.config["mock_tree"] = [
            item for item in tree
            if not (
                isinstance(item, dict)
                and (
                    normalize_cloud_path(str(item.get("path") or "")) == source
                    or normalize_cloud_path(str(item.get("path") or "")).startswith(prefix)
                )
            )
        ]

    def _download_mock_tree(self, source_path: str, destination: Path) -> int:
        source = normalize_cloud_path(source_path)
        written = 0
        for item in _mock_tree_entries(self.config):
            path = normalize_cloud_path(str(item.get("path") or ""))
            if path == ".":
                continue
            if source != "." and path != source and not path.startswith(source + "/"):
                continue
            relative = path if source == "." else path[len(source) :].lstrip("/")
            if not relative:
                continue
            target = destination / relative
            is_dir = bool(item.get("is_dir")) or "content" not in item
            if is_dir:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self._read_mock_file(path))
                _set_file_mtime(target, _parse_datetime(item.get("modified_at")))
                written += 1
        return written

    def _access_token(self) -> str:
        token = str(self.config.get("access_token") or self.config.get("oauth_token") or "").strip()
        if not token:
            raise RuntimeError(f"{self.provider} mount source is not connected")
        return token

    def _ensure_provider_scopes(self) -> None:
        scopes = self.config.get("scopes")
        if self.provider == "google_drive" and scopes and not has_google_drive_scope(scopes):
            raise RuntimeError(google_drive_scope_error_message())

    def _ensure_google_write_scope(self) -> None:
        scopes = self.config.get("scopes")
        if self.provider == "google_drive" and scopes and not has_google_drive_write_scope(scopes):
            raise RuntimeError(google_drive_write_scope_error_message())

    def _provider_request_error(self, response: httpx.Response) -> str:
        text = response.text[:300]
        if self.provider == "google_drive" and response.status_code == 403:
            lower_text = response.text.lower()
            if "insufficient" in lower_text:
                return google_drive_write_scope_error_message()
            if "drive.googleapis.com" in lower_text and ("disabled" in lower_text or "has not been used" in lower_text):
                return google_drive_api_disabled_error_message()
        if self.provider == "microsoft_drive" and response.status_code in {401, 403}:
            lower_text = response.text.lower()
            if any(marker in lower_text for marker in ("accessdenied", "access denied", "insufficient", "authorization_requestdenied")):
                return microsoft_graph_permission_error_message()
        return f"Cloud provider request failed ({response.status_code}): {text}"

    async def _request_json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        self._ensure_provider_scopes()
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._access_token()}"
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.request(method, url, headers=headers, **kwargs)
            if response.status_code == 401 and await self._refresh_access_token():
                headers["Authorization"] = f"Bearer {self._access_token()}"
                response = await client.request(method, url, headers=headers, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(self._provider_request_error(response))
        return response.json()

    async def _request_bytes(self, method: str, url: str, **kwargs: Any) -> bytes:
        self._ensure_provider_scopes()
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._access_token()}"
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.request(method, url, headers=headers, **kwargs)
            if response.status_code == 401 and await self._refresh_access_token():
                headers["Authorization"] = f"Bearer {self._access_token()}"
                response = await client.request(method, url, headers=headers, **kwargs)
        if response.status_code >= 400:
            if self.provider == "google_drive" and response.status_code == 403:
                provider_error = self._provider_request_error(response)
                if provider_error != f"Cloud provider request failed ({response.status_code}): {response.text[:300]}":
                    raise RuntimeError(provider_error)
            raise RuntimeError(f"Cloud provider download failed ({response.status_code}): {response.text[:300]}")
        return response.content

    async def _request_no_content(self, method: str, url: str, **kwargs: Any) -> None:
        self._ensure_provider_scopes()
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._access_token()}"
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.request(method, url, headers=headers, **kwargs)
            if response.status_code == 401 and await self._refresh_access_token():
                headers["Authorization"] = f"Bearer {self._access_token()}"
                response = await client.request(method, url, headers=headers, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(self._provider_request_error(response))

    async def _refresh_access_token(self) -> bool:
        refresh_token = str(self.config.get("refresh_token") or self.config.get("oauth_refresh_token") or "").strip()
        if not refresh_token or self.is_mock:
            return False
        try:
            payload = await refresh_cloud_oauth_token(self.provider, refresh_token=refresh_token)
        except Exception as exc:
            logger.warning("Failed to refresh %s cloud mount token: %s", self.provider, exc)
            return False
        access_token = str(payload.get("access_token") or "").strip()
        if not access_token:
            return False
        self.config["access_token"] = access_token
        if payload.get("refresh_token"):
            self.config["refresh_token"] = str(payload.get("refresh_token"))
        return True

    def _microsoft_drive_prefix(self) -> str:
        graph_base = str(self.config.get("graph_base_url") or "https://graph.microsoft.com/v1.0").rstrip("/")
        drive_id = str(self.config.get("drive_id") or "").strip()
        if drive_id:
            return f"{graph_base}/drives/{drive_id}"
        return f"{graph_base}/me/drive"

    def _microsoft_graph_base(self) -> str:
        return str(self.config.get("graph_base_url") or "https://graph.microsoft.com/v1.0").rstrip("/")

    def _uses_microsoft_virtual_drive_root(self) -> bool:
        return not str(self.config.get("drive_id") or "").strip()

    async def _list_microsoft_drives(self) -> list[CloudMountEntry]:
        graph_base = self._microsoft_graph_base()
        site_id = str(self.config.get("site_id") or "").strip()
        urls = [f"{graph_base}/sites/{site_id}/drives"] if site_id else [f"{graph_base}/me/drives"]
        drives: dict[str, CloudMountEntry] = {}

        for initial_url in urls:
            url: str | None = initial_url
            while url:
                data = await self._request_json("GET", url)
                for item in data.get("value", []):
                    if not isinstance(item, dict):
                        continue
                    drive_id = str(item.get("id") or "").strip()
                    if not drive_id:
                        continue
                    name = str(item.get("name") or item.get("driveType") or "Drive").strip() or "Drive"
                    drives[drive_id] = CloudMountEntry(
                        name=name,
                        path=join_cloud_path(VIRTUAL_DRIVES_SEGMENT, drive_id),
                        is_dir=True,
                        modified_at=_parse_datetime(item.get("lastModifiedDateTime")),
                    )
                next_link = data.get("@odata.nextLink")
                url = str(next_link) if next_link else None
        return sorted(drives.values(), key=lambda entry: entry.name.lower())

    def _split_virtual_drive_path(self, source_path: str) -> tuple[str | None, str]:
        normalized = normalize_cloud_path(source_path)
        if normalized == ".":
            return None, "."
        parts = normalized.split("/", 2)
        if len(parts) >= 2 and parts[0] == VIRTUAL_DRIVES_SEGMENT:
            return parts[1], parts[2] if len(parts) == 3 else "."
        return None, normalized

    def _microsoft_drive_prefix_for_path(self, source_path: str) -> tuple[str | None, str]:
        if not self._uses_microsoft_virtual_drive_root():
            return self._microsoft_drive_prefix(), normalize_cloud_path(source_path)
        drive_id, relative_path = self._split_virtual_drive_path(source_path)
        if drive_id:
            return f"{self._microsoft_graph_base()}/drives/{drive_id}", relative_path
        normalized = normalize_cloud_path(source_path)
        if normalized == ".":
            return None, "."
        return self._microsoft_drive_prefix(), normalized

    async def _list_microsoft_dir(self, source_path: str) -> list[CloudMountEntry]:
        prefix, normalized = self._microsoft_drive_prefix_for_path(source_path)
        if prefix is None:
            return await self._list_microsoft_drives()
        if normalized == ".":
            url = f"{prefix}/root/children"
        else:
            url = f"{prefix}/root:/{normalized}:/children"
        data = await self._request_json("GET", url)
        entries: list[CloudMountEntry] = []
        for item in data.get("value", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            path = join_cloud_path(source_path, name) if self._uses_microsoft_virtual_drive_root() else join_cloud_path(normalized, name)
            entries.append(
                CloudMountEntry(
                    name=name,
                    path=path,
                    is_dir=isinstance(item.get("folder"), dict),
                    size=_parse_size(item.get("size")),
                    modified_at=_parse_datetime(item.get("lastModifiedDateTime")),
                )
            )
        return sorted(entries, key=lambda entry: (not entry.is_dir, entry.name.lower()))

    async def _read_microsoft_file(self, source_path: str) -> bytes:
        prefix, normalized = self._microsoft_drive_prefix_for_path(source_path)
        if prefix is None:
            raise IsADirectoryError("Cannot read drive root as a file")
        if normalized == ".":
            raise IsADirectoryError("Cannot read drive root as a file")
        return await self._request_bytes("GET", f"{prefix}/root:/{normalized}:/content")

    async def _create_microsoft_dir(self, source_path: str) -> None:
        prefix, normalized = self._microsoft_drive_prefix_for_path(source_path)
        if prefix is None:
            raise IsADirectoryError("Select a drive before creating a folder")
        if normalized == ".":
            raise IsADirectoryError("Cannot create the drive root")
        parent_path = "." if "/" not in normalized else normalized.rsplit("/", 1)[0]
        name = normalized if parent_path == "." else normalized.rsplit("/", 1)[1]
        url = f"{prefix}/root/children" if parent_path == "." else f"{prefix}/root:/{parent_path}:/children"
        await self._request_json(
            "POST",
            url,
            json={"name": name, "folder": {}, "@microsoft.graph.conflictBehavior": "fail"},
        )

    async def _ensure_microsoft_dir(self, source_path: str) -> None:
        prefix, normalized = self._microsoft_drive_prefix_for_path(source_path)
        if prefix is None or normalized == ".":
            return
        current_relative = ""
        for part in normalized.split("/"):
            parent_relative = current_relative or "."
            current_relative = part if not current_relative else f"{current_relative}/{part}"
            try:
                await self._request_json("GET", f"{prefix}/root:/{current_relative}")
                continue
            except RuntimeError:
                url = f"{prefix}/root/children" if parent_relative == "." else f"{prefix}/root:/{parent_relative}:/children"
                await self._request_json(
                    "POST",
                    url,
                    json={"name": part, "folder": {}, "@microsoft.graph.conflictBehavior": "fail"},
                )

    async def _upload_microsoft_file(self, source_path: str, local_path: Path) -> None:
        prefix, normalized = self._microsoft_drive_prefix_for_path(source_path)
        if prefix is None or normalized == ".":
            raise IsADirectoryError("Select a file path inside a drive")
        parent_source_path = "." if "/" not in normalize_cloud_path(source_path) else normalize_cloud_path(source_path).rsplit("/", 1)[0]
        if parent_source_path != ".":
            await self._ensure_microsoft_dir(parent_source_path)
        content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        await self._request_json(
            "PUT",
            f"{prefix}/root:/{normalized}:/content",
            headers={"Content-Type": content_type},
            content=local_path.read_bytes(),
        )

    async def _delete_microsoft_path(self, source_path: str) -> None:
        prefix, normalized = self._microsoft_drive_prefix_for_path(source_path)
        if prefix is None or normalized == ".":
            raise IsADirectoryError("Cannot delete the drive root")
        await self._request_no_content("DELETE", f"{prefix}/root:/{normalized}")

    def _explicit_google_root_id(self) -> str:
        return str(self.config.get("root_id") or self.config.get("drive_id") or "").strip()

    def _uses_google_virtual_drive_root(self) -> bool:
        return not self._explicit_google_root_id()

    async def _list_google_drives(self) -> list[CloudMountEntry]:
        entries = [
            CloudMountEntry(
                name="My Drive",
                path=GOOGLE_MY_DRIVE_SEGMENT,
                is_dir=True,
            )
        ]
        seen_ids: set[str] = set()
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "pageSize": 100,
                "fields": "nextPageToken,drives(id,name)",
            }
            if page_token:
                params["pageToken"] = page_token
            data = await self._request_json(
                "GET",
                "https://www.googleapis.com/drive/v3/drives",
                params=params,
            )
            for item in data.get("drives", []):
                if not isinstance(item, dict):
                    continue
                drive_id = str(item.get("id") or "").strip()
                if not drive_id or drive_id in seen_ids:
                    continue
                seen_ids.add(drive_id)
                name = str(item.get("name") or "Shared drive").strip() or "Shared drive"
                entries.append(
                    CloudMountEntry(
                        name=name,
                        path=join_cloud_path(VIRTUAL_DRIVES_SEGMENT, drive_id),
                        is_dir=True,
                    )
                )
            page_token = str(data.get("nextPageToken") or "").strip() or None
            if not page_token:
                break
        return sorted(entries, key=lambda entry: entry.name.lower())

    def _split_google_virtual_path(self, source_path: str) -> tuple[str, str | None, str]:
        normalized = normalize_cloud_path(source_path)
        if not self._uses_google_virtual_drive_root():
            return "explicit", None, normalized
        if normalized == ".":
            return "virtual_root", None, "."
        if normalized == GOOGLE_MY_DRIVE_SEGMENT or normalized.startswith(GOOGLE_MY_DRIVE_SEGMENT + "/"):
            relative = normalized[len(GOOGLE_MY_DRIVE_SEGMENT) :].lstrip("/") or "."
            return "my_drive", None, relative
        parts = normalized.split("/", 2)
        if len(parts) >= 2 and parts[0] == VIRTUAL_DRIVES_SEGMENT:
            return "shared_drive", parts[1], parts[2] if len(parts) == 3 else "."
        return "legacy_root", None, normalized

    async def _google_folder_id_for_path(self, source_path: str) -> str:
        mode, drive_id, normalized = self._split_google_virtual_path(source_path)
        if mode == "virtual_root":
            raise IsADirectoryError("Cannot resolve all drives as a single folder")
        if mode == "shared_drive" and drive_id:
            folder_id = drive_id
        elif mode == "explicit":
            folder_id = self._explicit_google_root_id()
        else:
            folder_id = "root"
        if normalized == ".":
            return folder_id
        for part in normalized.split("/"):
            query = (
                f"'{folder_id}' in parents and name = '{_google_query_string(part)}' "
                "and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            )
            data = await self._request_json(
                "GET",
                "https://www.googleapis.com/drive/v3/files",
                params={"q": query, "fields": "files(id,name)", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"},
            )
            files = data.get("files") or []
            if not files:
                raise FileNotFoundError(normalized)
            folder_id = str(files[0].get("id") or "")
        return folder_id

    async def _ensure_google_dir(self, source_path: str) -> None:
        self._ensure_google_write_scope()
        normalized = normalize_cloud_path(source_path)
        if normalized == ".":
            return
        mode, drive_id, relative = self._split_google_virtual_path(source_path)
        if mode == "virtual_root":
            raise IsADirectoryError("Select My Drive or a shared drive before creating a folder")
        if mode == "shared_drive" and drive_id:
            folder_id = drive_id
        elif mode == "explicit":
            folder_id = self._explicit_google_root_id()
        else:
            folder_id = "root"
        if relative == ".":
            return
        for part in relative.split("/"):
            query = (
                f"'{folder_id}' in parents and name = '{_google_query_string(part)}' "
                "and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            )
            data = await self._request_json(
                "GET",
                "https://www.googleapis.com/drive/v3/files",
                params={"q": query, "fields": "files(id,name)", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"},
            )
            files = data.get("files") or []
            if files:
                folder_id = str(files[0].get("id") or "")
                continue
            created = await self._request_json(
                "POST",
                "https://www.googleapis.com/drive/v3/files",
                params={"supportsAllDrives": "true", "fields": "id,name"},
                json={"name": part, "mimeType": "application/vnd.google-apps.folder", "parents": [folder_id]},
            )
            folder_id = str(created.get("id") or "")

    async def _google_file_id_for_path(self, source_path: str) -> str:
        normalized = normalize_cloud_path(source_path)
        if normalized == ".":
            raise IsADirectoryError("Cannot read drive root as a file")
        parent = "." if "/" not in normalized else normalized.rsplit("/", 1)[0]
        name = normalized if parent == "." else normalized.rsplit("/", 1)[1]
        parent_id = await self._google_folder_id_for_path(parent)
        query = f"'{parent_id}' in parents and name = '{_google_query_string(name)}' and trashed = false"
        data = await self._request_json(
            "GET",
            "https://www.googleapis.com/drive/v3/files",
            params={"q": query, "fields": "files(id,name,mimeType)", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"},
        )
        files = data.get("files") or []
        if not files:
            raise FileNotFoundError(normalized)
        return str(files[0].get("id") or "")

    async def _list_google_dir(self, source_path: str) -> list[CloudMountEntry]:
        mode, drive_id, normalized = self._split_google_virtual_path(source_path)
        if mode == "virtual_root":
            return await self._list_google_drives()
        folder_id = await self._google_folder_id_for_path(source_path)
        query = f"'{folder_id}' in parents and trashed = false"
        params: dict[str, Any] = {
            "q": query,
            "fields": "files(id,name,mimeType,size,modifiedTime)",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "orderBy": "folder,name",
        }
        if mode == "shared_drive" and drive_id:
            params["corpora"] = "drive"
            params["driveId"] = drive_id
        data = await self._request_json(
            "GET",
            "https://www.googleapis.com/drive/v3/files",
            params=params,
        )
        entries: list[CloudMountEntry] = []
        for item in data.get("files", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            entries.append(
                CloudMountEntry(
                    name=name,
                    path=join_cloud_path(source_path, name),
                    is_dir=item.get("mimeType") == "application/vnd.google-apps.folder",
                    size=_parse_size(item.get("size")),
                    modified_at=_parse_datetime(item.get("modifiedTime")),
                )
            )
        return sorted(entries, key=lambda entry: (not entry.is_dir, entry.name.lower()))

    async def _read_google_file(self, source_path: str) -> bytes:
        file_id = await self._google_file_id_for_path(source_path)
        return await self._request_bytes(
            "GET",
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            params={"alt": "media", "supportsAllDrives": "true"},
        )

    async def _upload_google_file(self, source_path: str, local_path: Path) -> None:
        self._ensure_google_write_scope()
        normalized = normalize_cloud_path(source_path)
        if normalized == ".":
            raise IsADirectoryError("Select a file path inside a drive")
        parent_path = "." if "/" not in normalized else normalized.rsplit("/", 1)[0]
        name = normalized if parent_path == "." else normalized.rsplit("/", 1)[1]
        await self._ensure_google_dir(parent_path)
        parent_id = await self._google_folder_id_for_path(parent_path)
        existing_file_id: str | None = None
        try:
            existing_file_id = await self._google_file_id_for_path(source_path)
        except FileNotFoundError:
            existing_file_id = None
        content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        metadata: dict[str, Any] = {
            "name": name,
            "modifiedTime": _epoch_seconds_to_rfc3339(int(local_path.stat().st_mtime)),
        }
        if existing_file_id is None:
            metadata["parents"] = [parent_id]
        boundary = f"ragtime_{hashlib.sha256(f'{source_path}:{local_path.stat().st_mtime}'.encode()).hexdigest()[:24]}"
        body = b"\r\n".join([
            f"--{boundary}".encode("ascii"),
            b"Content-Type: application/json; charset=UTF-8",
            b"",
            json.dumps(metadata).encode("utf-8"),
            f"--{boundary}".encode("ascii"),
            f"Content-Type: {content_type}".encode("ascii"),
            b"",
            local_path.read_bytes(),
            f"--{boundary}--".encode("ascii"),
            b"",
        ])
        if existing_file_id:
            url = f"https://www.googleapis.com/upload/drive/v3/files/{existing_file_id}"
            method = "PATCH"
        else:
            url = "https://www.googleapis.com/upload/drive/v3/files"
            method = "POST"
        await self._request_json(
            method,
            url,
            params={"uploadType": "multipart", "supportsAllDrives": "true", "fields": "id,name,modifiedTime"},
            headers={"Content-Type": f"multipart/related; boundary={boundary}"},
            content=body,
        )

    async def _delete_google_path(self, source_path: str) -> None:
        file_id = await self._google_file_id_for_path(source_path)
        await self._request_no_content(
            "DELETE",
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            params={"supportsAllDrives": "true"},
        )

    async def _create_google_dir(self, source_path: str) -> None:
        self._ensure_google_write_scope()
        normalized = normalize_cloud_path(source_path)
        if normalized == ".":
            raise IsADirectoryError("Select My Drive or a shared drive before creating a folder")
        parent_path = "." if "/" not in normalized else normalized.rsplit("/", 1)[0]
        name = normalized if parent_path == "." else normalized.rsplit("/", 1)[1]
        mode, _drive_id, _parent_relative_path = self._split_google_virtual_path(parent_path)
        if mode == "virtual_root":
            raise IsADirectoryError("Select My Drive or a shared drive before creating a folder")
        parent_id = await self._google_folder_id_for_path(parent_path)
        metadata: dict[str, Any] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        params: dict[str, Any] = {
            "supportsAllDrives": "true",
            "fields": "id,name",
        }
        await self._request_json(
            "POST",
            "https://www.googleapis.com/drive/v3/files",
            params=params,
            json=metadata,
        )


async def exchange_cloud_oauth_code(
    provider: CloudProviderName,
    *,
    code: str,
    redirect_uri: str,
) -> dict[str, Any]:
    if provider == "microsoft_drive":
        client_id = settings.cloud_mount_microsoft_client_id
        client_secret = settings.cloud_mount_microsoft_client_secret
        tenant = _microsoft_oauth_tenant()
        token_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
    else:
        client_id = settings.cloud_mount_google_client_id
        client_secret = settings.cloud_mount_google_client_secret
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }

    if not client_id or not client_secret:
        raise RuntimeError(f"OAuth client credentials for {provider} are not configured")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(token_url, data=data)
    if response.status_code >= 400:
        raise RuntimeError(f"OAuth token exchange failed ({response.status_code}): {response.text[:300]}")
    payload = response.json()
    expires_in = int(payload.get("expires_in") or 3600)
    payload["expires_at"] = datetime.now(timezone.utc) + timedelta(seconds=max(60, expires_in - 60))
    return payload


async def refresh_cloud_oauth_token(
    provider: CloudProviderName,
    *,
    refresh_token: str,
) -> dict[str, Any]:
    if provider == "microsoft_drive":
        client_id = settings.cloud_mount_microsoft_client_id
        client_secret = settings.cloud_mount_microsoft_client_secret
        tenant = _microsoft_oauth_tenant()
        token_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    else:
        client_id = settings.cloud_mount_google_client_id
        client_secret = settings.cloud_mount_google_client_secret
        token_url = "https://oauth2.googleapis.com/token"

    if not client_id or not client_secret:
        raise RuntimeError(f"OAuth client credentials for {provider} are not configured")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            token_url,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
    if response.status_code >= 400:
        raise RuntimeError(f"OAuth token refresh failed ({response.status_code}): {response.text[:300]}")
    payload = response.json()
    expires_in = int(payload.get("expires_in") or 3600)
    payload["expires_at"] = datetime.now(timezone.utc) + timedelta(seconds=max(60, expires_in - 60))
    return payload


async def get_cloud_account_profile(provider: CloudProviderName, access_token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        if provider == "microsoft_drive":
            response = await client.get("https://graph.microsoft.com/v1.0/me", headers=headers)
        else:
            response = await client.get("https://www.googleapis.com/oauth2/v2/userinfo", headers=headers)
    if response.status_code >= 400:
        logger.warning("Failed to load cloud account profile for %s: %s", provider, response.text[:300])
        return {}
    return response.json()


def build_cloud_oauth_url(
    provider: CloudProviderName,
    *,
    state: str,
    redirect_uri: str,
) -> str:
    if provider == "microsoft_drive":
        client_id = settings.cloud_mount_microsoft_client_id
        tenant = _microsoft_oauth_tenant()
        scopes = "offline_access Files.ReadWrite.All Sites.ReadWrite.All User.Read"
        base = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
        params = {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "response_mode": "query",
            "scope": scopes,
            "state": state,
        }
    else:
        client_id = settings.cloud_mount_google_client_id
        scopes = f"{GOOGLE_DRIVE_SCOPE} https://www.googleapis.com/auth/userinfo.email"
        base = "https://accounts.google.com/o/oauth2/v2/auth"
        params = {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": scopes,
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
    if not client_id:
        raise RuntimeError(f"OAuth client id for {provider} is not configured")
    return f"{base}?{urlencode(params)}"


def cloud_mount_provider(source_type: str, config: dict[str, Any]) -> CloudMountProvider:
    if source_type not in {"microsoft_drive", "google_drive"}:
        raise ValueError(f"Unsupported cloud source type '{source_type}'")
    return CloudMountProvider(source_type, config)  # type: ignore[arg-type]
