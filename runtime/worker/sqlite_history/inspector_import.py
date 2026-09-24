"""Runtime-owned publication of uploaded SQLite inspector databases."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from runtime.core.secure_files import SecureFileError, open_directory, sha256_regular_file, stat_regular_file
from runtime.core.sqlite_history_scratch import SCRATCH_PREFIX, scratch_owner_lock

from .service import _PREVIEW_TTL, _catalog_lock, _history_subdirectory, _now, run_admitted_subprocess, run_sqlite_blocking, sqlite_workspace_access

_DATABASE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.(?:sqlite|sqlite3|db|db3)$", re.IGNORECASE)


class RuntimeInspectorImport:
    """Prepare an uploaded candidate, then delegate publication to apply()."""

    def __init__(self, history: Any) -> None:
        self._history = history

    @staticmethod
    def _validate_name(name: str) -> str:
        if not _DATABASE_NAME.fullmatch(name):
            raise HTTPException(status_code=400, detail="Invalid SQLite database filename")
        return name

    @staticmethod
    def _copy_confined(upload_dir: Path, upload_name: str, root: Path, candidate: str) -> None:
        """Validate and copy upload bytes in the existing confined SQLite child."""
        try:
            with open_directory(upload_dir.parent, upload_dir.name) as source_fd, open_directory(root.parent, root.name) as destination_fd:
                completed = run_admitted_subprocess(
                    [
                        sys.executable,
                        "-m",
                        "runtime.worker.sqlite_history.capture_child",
                        "--source-fd",
                        str(source_fd),
                        "--destination-fd",
                        str(destination_fd),
                        "--source-name",
                        upload_name,
                        "--destination-name",
                        candidate,
                    ],
                    pass_fds=(source_fd, destination_fd),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
        except (OSError, SecureFileError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(status_code=503, detail="Secure SQLite import confinement is unavailable") from exc
        if completed.returncode:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid SQLite database")

    async def import_database(self, workspace_id: str, database_name: str, creator_id: str, upload: Path) -> dict[str, Any]:
        async with self._history._installed():
            return await self._import_database(workspace_id, database_name, creator_id, upload)

    async def _import_database(self, workspace_id: str, database_name: str, creator_id: str, upload: Path) -> dict[str, Any]:
        database_name = self._validate_name(database_name)
        try:
            details = upload.lstat()
        except OSError as exc:
            raise HTTPException(status_code=400, detail="Uploaded database file was not readable") from exc
        if not upload.is_file() or upload.is_symlink() or details.st_size <= 0:
            raise HTTPException(status_code=400, detail="Uploaded database file was not readable")

        root = self._history._root(workspace_id)
        # The confinement child writes into a private import staging directory,
        # not the reaper-scanned candidate namespace.  It is atomically moved
        # under the catalog lock immediately before the preview publishes its
        # durable ownership reference.
        staging_name = f"{SCRATCH_PREFIX}import-{uuid4()}"
        staged = f"imports/{staging_name}/image.sqlite3"
        candidate = f"candidates/import-{uuid4()}.sqlite3"

        # Establish only runtime-private history directories before passing
        # descriptor roots to the confinement child.
        def ensure_candidate_dir() -> None:
            with _catalog_lock(root):
                self._history._enforce_quota(root, self._history._load(root, workspace_id), details.st_size)
                imports = _history_subdirectory(root, "imports", create=True)
                (imports / staging_name).mkdir(mode=0o700)

        await run_sqlite_blocking(ensure_candidate_dir)
        staging = root / "imports" / staging_name
        owner = scratch_owner_lock(staging)
        await run_sqlite_blocking(owner.__enter__)
        try:
            await run_sqlite_blocking(self._copy_confined, upload.parent, upload.name, root, staged)
            candidate_sha256 = await run_sqlite_blocking(sha256_regular_file, root, staged)
            candidate_size = (await run_sqlite_blocking(stat_regular_file, root, staged)).st_size
        except BaseException:
            await run_sqlite_blocking(owner.__exit__, None, None, None)
            await run_sqlite_blocking(shutil.rmtree, staging, ignore_errors=True)
            raise
        # Stable source ID deliberately identifies an upload rather than
        # pretending it is a catalog backup row.
        source_id = f"import-{candidate_sha256}"
        try:
            async with sqlite_workspace_access(workspace_id) as files_dir:
                drift = await run_sqlite_blocking(self._history._drift_confined, files_dir, root, database_name)

                def persist_preview() -> str:
                    with _catalog_lock(root):
                        manifest = self._history._load(root, workspace_id)
                        _history_subdirectory(root, "candidates", create=True)
                        os.replace(root / staged, root / candidate)
                        preview_id = str(uuid4())
                        manifest["previews"][preview_id] = {
                            "backup_id": source_id,
                            "database_name": database_name,
                            "candidate": candidate,
                            "candidate_sha256": candidate_sha256,
                            "current_fingerprint": drift["current_fingerprint"],
                            "migration_fingerprint": drift["migration_fingerprint"],
                            "mode": "overwrite",
                            "conflict_policy": "keep_current",
                            "table_policies": {},
                            "user_id": creator_id,
                            "expires_at": (_now() + _PREVIEW_TTL).isoformat(),
                            "import_source_id": source_id,
                        }
                        self._history._save(root, manifest)
                        return preview_id

                preview_id = await run_sqlite_blocking(persist_preview)
            # apply owns the one exclusive maintenance lease: drift recheck,
            # mandatory safety capture, no-follow publication and recovery.
            receipt = await self._history.apply(workspace_id, preview_id, user_id=creator_id)
            return {"receipt": receipt, "source_id": source_id, "size_bytes": candidate_size}
        except BaseException:
            # Keep the candidate only if apply has journaled a publication. The
            # existing recovery flow needs those exact bytes.
            def remove_unpublished() -> None:
                with _catalog_lock(root):
                    manifest = self._history._load(root, workspace_id)
                    if not any(row.get("candidate") == candidate for row in manifest.get("previews", {}).values()):
                        try:
                            os.unlink(root / candidate)
                        except FileNotFoundError:
                            try:
                                os.unlink(root / staged)
                            except FileNotFoundError:
                                pass

            await run_sqlite_blocking(remove_unpublished)
            raise
        finally:
            await run_sqlite_blocking(owner.__exit__, None, None, None)
            await run_sqlite_blocking(shutil.rmtree, staging, ignore_errors=True)
