"""Private, short-lived SolidWorks export artifacts."""

import asyncio
import errno
import json
import os
import re
import secrets
import shutil
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ragtime.config import settings
from ragtime.core.tool_access import filter_tool_ids_by_access
from ragtime.indexer.repository import repository

MAX_ARTIFACT_BYTES = 100 * 1024 * 1024
ARTIFACT_TTL = timedelta(days=7)
_STAGING_TTL = timedelta(hours=1)
ARTIFACT_URL_PREFIX = "/indexes/solidworks/artifacts"
ARTIFACT_DIR = Path(settings.index_data_path) / "_solidworks_artifacts"
_ID = re.compile(r"^[0-9a-f]{32}$")


class ArtifactError(Exception):
    pass


class ArtifactNotFound(ArtifactError):
    pass


class ArtifactExpired(ArtifactError):
    pass


class ArtifactForbidden(ArtifactError):
    pass


class ArtifactTooLarge(ArtifactError):
    pass


def sanitize_artifact_filename(file_name: str, format: str) -> str:
    name = "".join(c for c in str(file_name or "") if c.isprintable() and c not in "\"'/\\").strip()[:120]
    extension = re.sub(r"[^a-z0-9]", "", str(format).lower()) or "bin"
    return name if name and name not in {".", ".."} else f"model.{extension}"


def artifact_wire(meta: dict[str, Any]) -> dict[str, Any]:
    artifact_id = meta["id"]
    content_url = f"{ARTIFACT_URL_PREFIX}/{artifact_id}/content"
    wire = {key: meta[key] for key in ("id", "kind", "format", "file_name", "size_bytes", "sha256", "expires_at")}
    wire["content_url"] = content_url
    preview_id = meta.get("preview_artifact_id")
    if meta["kind"] == "preview" and meta["format"] == "png":
        wire["preview_image_url"] = content_url
    elif preview_id and _ID.fullmatch(str(preview_id)):
        wire["preview_image_url"] = f"{ARTIFACT_URL_PREFIX}/{preview_id}/content"
    wire["page_url"] = f"/?artifact={artifact_id}"
    return wire


def _store(source: Path, meta: dict[str, Any]) -> dict[str, Any]:
    size = source.stat().st_size
    if size > MAX_ARTIFACT_BYTES:
        source.unlink(missing_ok=True)
        raise ArtifactTooLarge()
    artifact_dir = ARTIFACT_DIR / f".{meta['id']}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    try:
        target = artifact_dir / "content"
        try:
            os.replace(source, target)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, target)
            source.unlink()
        size = target.stat().st_size
        if size > MAX_ARTIFACT_BYTES:
            raise ArtifactTooLarge()
        meta["size_bytes"] = size
        temporary = artifact_dir / "meta.tmp"
        temporary.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(temporary, artifact_dir / "meta.json")
        os.replace(artifact_dir, ARTIFACT_DIR / meta["id"])
        return artifact_wire(meta)
    except BaseException:
        shutil.rmtree(artifact_dir, ignore_errors=True)
        raise


async def store_artifact(
    source_path: Path,
    *,
    file_name: str,
    kind: str,
    format: str,
    sha256: str,
    owner_user_id: str | None,
    tool_config_id: str,
    surface: str | None,
    mcp_route: str | None,
    preview_artifact_id: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    meta = {
        "id": secrets.token_hex(16),
        "file_name": sanitize_artifact_filename(file_name, format),
        "kind": kind,
        "format": format,
        "sha256": sha256,
        "owner_user_id": owner_user_id,
        "tool_config_id": tool_config_id,
        "surface": surface,
        "mcp_route": mcp_route,
        "preview_artifact_id": preview_artifact_id,
        "created_at": now.isoformat(),
        "expires_at": (now + ARTIFACT_TTL).isoformat(),
    }
    return await asyncio.to_thread(_store, Path(source_path), meta)


def load_artifact_sync(artifact_id: str) -> tuple[dict[str, Any], Path]:
    if not isinstance(artifact_id, str) or not _ID.fullmatch(artifact_id):
        raise ArtifactNotFound()
    directory = ARTIFACT_DIR / artifact_id
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        expires_at = datetime.fromisoformat(meta["expires_at"])
        if expires_at.tzinfo is None or meta["id"] != artifact_id or not (directory / "content").is_file():
            raise ValueError("Invalid artifact")
        meta["expired"] = expires_at <= datetime.now(timezone.utc)
        return meta, directory / "content"
    except (OSError, ValueError, TypeError, KeyError):
        raise ArtifactNotFound() from None


async def load_artifact(artifact_id: str) -> tuple[dict[str, Any], Path]:
    return await asyncio.to_thread(load_artifact_sync, artifact_id)


async def authorize_artifact(meta: dict[str, Any], user: Any) -> None:
    user_id = str(user.id)
    if user.role == "admin" or (meta.get("owner_user_id") and str(meta["owner_user_id"]) == user_id):
        return
    tool_id = meta.get("tool_config_id")
    if tool_id and await repository.get_tool_config(tool_id):
        allowed = await filter_tool_ids_by_access(user_id=user_id, is_admin=False, surface="chat", tool_config_ids=[tool_id])
        if tool_id in allowed:
            return
    raise ArtifactForbidden()


async def resolve_artifact_content(artifact_id: str, user: Any) -> tuple[dict[str, Any], Path]:
    meta, path = await load_artifact(artifact_id)
    await authorize_artifact(meta, user)
    if meta["expired"]:
        raise ArtifactExpired()
    return meta, path


def _cleanup() -> int:
    if not ARTIFACT_DIR.exists():
        return 0
    removed = 0
    stale_before = (datetime.now(timezone.utc) - _STAGING_TTL).timestamp()
    for directory in ARTIFACT_DIR.iterdir():
        try:
            info = directory.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(info.st_mode):
            continue
        if directory.name.startswith(".") and _ID.fullmatch(directory.name[1:]):
            if info.st_mtime < stale_before:
                shutil.rmtree(directory, ignore_errors=True)
                removed += 1
            continue
        if not _ID.fullmatch(directory.name):
            continue
        try:
            meta, _ = load_artifact_sync(directory.name)
            expired = meta["expired"]
        except ArtifactNotFound:
            expired = True
        if expired:
            shutil.rmtree(directory, ignore_errors=True)
            removed += 1
    return removed


async def cleanup_expired_artifacts() -> int:
    return await asyncio.to_thread(_cleanup)
