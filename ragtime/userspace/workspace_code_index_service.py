from __future__ import annotations

import asyncio
import mimetypes
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, cast

from ragtime.core.app_settings import get_app_settings
from ragtime.core.database import get_db
from ragtime.core.datetimes import utc_now
from ragtime.core.logging import get_logger
from ragtime.indexer.code_extraction import extract_metadata
from ragtime.indexer.file_utils import compute_file_hash, is_excluded_by_patterns, is_excluded_directory, should_index_file_type
from ragtime.indexer.filesystem_service import FilesystemIndexerService
from ragtime.indexer.models import FilesystemConnectionConfig, OcrMode, VectorStoreType, WorkspaceCodeIndexJobPhase
from ragtime.indexer.repository import IndexerRepository
from ragtime.indexer.vector_backends import get_pgvector_backend
from ragtime.indexer.vector_utils import embed_documents_subbatched, get_embeddings_model, search_pgvector_embeddings

logger = get_logger(__name__)

DirtyOperation = Literal["upsert", "delete", "reindex"]

_DEFAULT_DEBOUNCE_SECONDS = 2
_DEFAULT_RECONCILE_INTERVAL_SECONDS = 300
_DEFAULT_MAX_DIRTY_ATTEMPTS = 3
_MAX_SEARCH_RESULTS = 25
USERSPACE_WORKSPACE_CODE_INDEX_PREFIX = "userspace_workspace_"
_UNSET_CURRENT_FILE = object()
_INDEXABLE_EXCLUDE_PATTERNS = [
    ".ragtime/**",
    ".git/**",
    "node_modules/**",
    "dist/**",
    "build/**",
    ".venv/**",
    "venv/**",
]


def _sql(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _normalize_path(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").lstrip("/")


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _extract_symbol_name(signature: str) -> str:
    match = re.search(r"\b(?:class|def|function|interface|type|struct|enum|trait)\s+([A-Za-z_$][\w$]*)", signature)
    if match:
        return match.group(1)
    method_match = re.search(r"\b([A-Za-z_$][\w$]*)\s*\(", signature)
    if method_match:
        return method_match.group(1)
    return signature[:120]


async def _load_code_index_setting_int(
    key: str,
    default: int,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Read a code-index tuning setting from app settings, falling back to defaults."""
    try:
        settings = await get_app_settings()
        value = int(settings.get(key, default))
    except Exception:
        return default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


class WorkspaceCodeIndexService:
    """Hidden per-workspace code index backed by filesystem_embeddings/pgvector."""

    def __init__(
        self,
        *,
        userspace_service: Any | None = None,
        debounce_seconds: float | None = None,
        reconcile_interval_seconds: float | None = None,
        max_attempts: int | None = None,
    ) -> None:
        self._userspace_service = userspace_service
        self._debounce_seconds_override = debounce_seconds
        self._reconcile_interval_seconds_override = reconcile_interval_seconds
        self._max_attempts_override = max_attempts
        self._filesystem_indexer = FilesystemIndexerService()
        self._worker_task: asyncio.Task[None] | None = None
        self._workspace_tasks: dict[str, asyncio.Task[None]] = {}
        self._shutdown = False

    async def _debounce_seconds(self) -> float:
        if self._debounce_seconds_override is not None:
            return self._debounce_seconds_override
        return await _load_code_index_setting_int(
            "userspace_code_index_debounce_seconds",
            _DEFAULT_DEBOUNCE_SECONDS,
            min_value=0,
            max_value=3600,
        )

    async def _reconcile_interval_seconds(self) -> float:
        if self._reconcile_interval_seconds_override is not None:
            return self._reconcile_interval_seconds_override
        return await _load_code_index_setting_int(
            "userspace_code_index_reconcile_interval_seconds",
            _DEFAULT_RECONCILE_INTERVAL_SECONDS,
            min_value=10,
            max_value=86400,
        )

    async def _max_dirty_attempts(self) -> int:
        if self._max_attempts_override is not None:
            return self._max_attempts_override
        return await _load_code_index_setting_int(
            "userspace_code_index_max_attempts",
            _DEFAULT_MAX_DIRTY_ATTEMPTS,
            min_value=1,
            max_value=20,
        )

    async def _enabled(self) -> bool:
        """Return whether User Space code indexing is enabled, defaulting to true on failure."""
        try:
            settings = await get_app_settings()
            return bool(settings.get("userspace_code_index_enabled", True))
        except Exception:
            return True

    async def set_enabled(self, enabled: bool) -> None:
        """Persist the global User Space code indexing toggle."""
        await IndexerRepository().update_settings({"userspace_code_index_enabled": enabled})

    @property
    def userspace_service(self) -> Any:
        if self._userspace_service is None:
            from ragtime.userspace.service import userspace_service

            self._userspace_service = userspace_service
        return self._userspace_service

    def index_name_for_workspace(self, workspace_id: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(workspace_id or "")).strip("_").lower()
        return f"{USERSPACE_WORKSPACE_CODE_INDEX_PREFIX}{safe or 'unknown'}"

    async def _workspace_mount_prefixes(self, workspace_id: str) -> list[str]:
        provider = self.userspace_service
        list_mount_paths = getattr(provider, "_list_workspace_mount_target_repo_paths", None)
        if not callable(list_mount_paths):
            return []
        list_mount_paths = cast(Callable[[str], Awaitable[list[str]]], list_mount_paths)
        try:
            return [prefix for prefix in await list_mount_paths(workspace_id) if prefix]
        except Exception:
            logger.debug("Failed to load workspace mount prefixes for code index workspace=%s", workspace_id, exc_info=True)
            return []

    def _workspace_path_matches_mount_prefix(self, path: str, prefix: str) -> bool:
        matcher = getattr(self.userspace_service, "_workspace_path_matches_mount_prefix", None)
        if callable(matcher):
            return bool(matcher(path, prefix))
        clean_path = path.strip("/")
        clean_prefix = prefix.strip("/")
        return clean_path == clean_prefix or clean_path.startswith(f"{clean_prefix}/")

    def coalesce_dirty_operation(self, existing: str | None, new: str) -> str:
        existing_op = str(existing or "").strip().lower()
        new_op = str(new or "").strip().lower()
        if existing_op == "reindex" or new_op == "reindex":
            return "reindex"
        if new_op == "delete":
            return "delete"
        if new_op == "upsert":
            return "upsert"
        return existing_op or new_op or "upsert"

    async def collect_indexable_files(self, workspace_id: str) -> list[Path]:
        files_root = self.userspace_service._workspace_files_dir(workspace_id)
        if not files_root.exists() or not files_root.is_dir():
            return []

        mount_prefixes = await self._workspace_mount_prefixes(workspace_id)
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(files_root, followlinks=False):
            current_dir = Path(dirpath)
            pruned: list[str] = []
            for dirname in dirnames:
                candidate_dir = current_dir / dirname
                try:
                    rel_dir = candidate_dir.relative_to(files_root).as_posix()
                    if candidate_dir.is_symlink() or is_excluded_directory(candidate_dir, files_root, _INDEXABLE_EXCLUDE_PATTERNS):
                        continue
                    if any(self._workspace_path_matches_mount_prefix(rel_dir, prefix) for prefix in mount_prefixes):
                        continue
                except (OSError, ValueError):
                    continue
                pruned.append(dirname)
            dirnames[:] = pruned

            for filename in filenames:
                path = current_dir / filename
                try:
                    rel_path = path.relative_to(files_root).as_posix()
                    if path.is_symlink() or not path.is_file():
                        continue
                    if is_excluded_by_patterns(path, files_root, _INDEXABLE_EXCLUDE_PATTERNS):
                        continue
                    if any(self._workspace_path_matches_mount_prefix(rel_path, prefix) for prefix in mount_prefixes):
                        continue
                    if not should_index_file_type(path, matches_include_pattern=False, ocr_enabled=False):
                        continue
                except (OSError, ValueError):
                    continue
                files.append(path)

        return sorted(files, key=lambda file_path: file_path.relative_to(files_root).as_posix())

    async def ensure_state(self, workspace_id: str) -> str:
        index_name = self.index_name_for_workspace(workspace_id)
        db = await get_db()
        await db.execute_raw(
            f"""
            INSERT INTO workspace_code_index_states (id, workspace_id, index_name, status, created_at, updated_at)
            VALUES ({_sql(str(uuid.uuid4()))}, {_sql(workspace_id)}, {_sql(index_name)}, 'pending', NOW(), NOW())
            ON CONFLICT (workspace_id) DO UPDATE SET
                index_name = EXCLUDED.index_name,
                updated_at = NOW()
            """
        )
        return index_name

    async def mark_dirty(self, workspace_id: str, path: str = "", operation: DirtyOperation = "upsert") -> None:
        if not await self._enabled():
            return
        normalized_path = _normalize_path(path)
        if operation != "reindex" and not normalized_path:
            return
        if operation == "reindex":
            normalized_path = ""

        await self.ensure_state(workspace_id)
        db = await get_db()
        existing = await db.query_raw(
            f"""
            SELECT operation FROM workspace_code_index_dirty_paths
            WHERE workspace_id = {_sql(workspace_id)} AND path = {_sql(normalized_path)}
            LIMIT 1
            """
        )
        next_operation = self.coalesce_dirty_operation(existing[0].get("operation") if existing else None, operation)
        await db.execute_raw(
            f"""
            INSERT INTO workspace_code_index_dirty_paths (id, workspace_id, path, operation, dirty_at, created_at, updated_at)
            VALUES ({_sql(str(uuid.uuid4()))}, {_sql(workspace_id)}, {_sql(normalized_path)}, {_sql(next_operation)}, NOW(), NOW(), NOW())
            ON CONFLICT (workspace_id, path) DO UPDATE SET
                operation = EXCLUDED.operation,
                dirty_at = NOW(),
                attempt_count = 0,
                last_error = NULL,
                updated_at = NOW()
            """
        )
        await db.execute_raw(
            f"""
            UPDATE workspace_code_index_states
            SET status = 'stale', updated_at = NOW()
            WHERE workspace_id = {_sql(workspace_id)}
            """
        )
        self.schedule_workspace(workspace_id)

    def schedule_workspace(self, workspace_id: str) -> bool:
        if self._shutdown:
            return False
        existing = self._workspace_tasks.get(workspace_id)
        if existing and not existing.done():
            return False
        task = asyncio.create_task(self._debounced_process(workspace_id))
        self._workspace_tasks[workspace_id] = task
        task.add_done_callback(lambda done_task: self._prune_workspace_task(workspace_id, done_task))
        return True

    def _prune_workspace_task(self, workspace_id: str, done_task: asyncio.Task[None]) -> None:
        if self._workspace_tasks.get(workspace_id) is done_task:
            self._workspace_tasks.pop(workspace_id, None)

    async def start(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return
        self._shutdown = False
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        self._shutdown = True
        tasks = [task for task in self._workspace_tasks.values() if not task.done()]
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            tasks.append(self._worker_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workspace_tasks.clear()
        self._worker_task = None

    async def _debounced_process(self, workspace_id: str) -> None:
        try:
            await asyncio.sleep(await self._debounce_seconds())
            while True:
                await self.process_dirty_workspace(workspace_id)
                max_attempts = await self._max_dirty_attempts()
                db = await get_db()
                remaining = await db.query_raw(
                    f"""
                    SELECT id FROM workspace_code_index_dirty_paths
                    WHERE workspace_id = {_sql(workspace_id)}
                      AND attempt_count < {max_attempts}
                    LIMIT 1
                    """
                )
                if not remaining:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Workspace code index processing failed for %s: %s", workspace_id, exc)

    async def _worker_loop(self) -> None:
        while not self._shutdown:
            try:
                await self.reconcile_stale_workspaces()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Workspace code index reconciliation failed: %s", exc)
            await asyncio.sleep(await self._reconcile_interval_seconds())

    async def reconcile_stale_workspaces(self, include_missing: bool = False) -> list[str]:
        if not await self._enabled():
            return []
        db = await get_db()
        max_attempts = await self._max_dirty_attempts()
        rows = await db.query_raw(
            f"""
            SELECT DISTINCT workspace_id FROM workspace_code_index_dirty_paths
            WHERE attempt_count < {max_attempts}
            ORDER BY workspace_id
            LIMIT 50
            """
        )
        scheduled: set[str] = set()
        for row in rows:
            workspace_id = str(row.get("workspace_id") or "")
            if workspace_id:
                self.schedule_workspace(workspace_id)
                scheduled.add(workspace_id)

        if include_missing:
            missing_rows = await db.query_raw(
                """
                SELECT w.id AS workspace_id
                FROM workspaces w
                LEFT JOIN workspace_code_index_states s ON s.workspace_id = w.id
                WHERE s.workspace_id IS NULL
                ORDER BY w.id
                """
            )
            for row in missing_rows:
                workspace_id = str(row.get("workspace_id") or "")
                if not workspace_id:
                    continue
                await self.mark_dirty(workspace_id, operation="reindex")
                self.schedule_workspace(workspace_id)
                scheduled.add(workspace_id)

        return list(scheduled)

    async def process_dirty_workspace(self, workspace_id: str) -> None:
        await self.ensure_state(workspace_id)
        db = await get_db()
        process_started_at = utc_now()
        max_attempts = await self._max_dirty_attempts()
        dirty_rows = await db.query_raw(
            f"""
            SELECT id, path, operation FROM workspace_code_index_dirty_paths
            WHERE workspace_id = {_sql(workspace_id)}
              AND attempt_count < {max_attempts}
            ORDER BY dirty_at ASC
            LIMIT 250
            """
        )
        if not dirty_rows:
            return

        await db.execute_raw(
            f"""
            UPDATE workspace_code_index_states
            SET status = 'indexing', last_error = NULL, updated_at = NOW()
            WHERE workspace_id = {_sql(workspace_id)}
            """
        )
        try:
            if any(row.get("operation") == "reindex" for row in dirty_rows):
                dirty_ids = [str(row.get("id") or "") for row in dirty_rows if row.get("id")]
                await self.reindex_workspace(workspace_id, processed_dirty_ids=dirty_ids, process_started_at=process_started_at)
                return
            index_name = self.index_name_for_workspace(workspace_id)
            job_id = await self._start_job(workspace_id, index_name, total_files=len(dirty_rows))
            try:
                embedding_dimension: int | None = None
                processed_chunks = 0
                for idx, row in enumerate(dirty_rows, start=1):
                    path = _normalize_path(str(row.get("path") or ""))
                    operation = str(row.get("operation") or "upsert")
                    if operation == "delete":
                        await self.delete_file_from_index(workspace_id, path)
                    else:
                        chunk_count, file_embedding_dimension = await self.index_file(
                            workspace_id,
                            path,
                            job_id=job_id,
                            processed_chunks_so_far=processed_chunks,
                        )
                        processed_chunks += chunk_count
                        if file_embedding_dimension is not None:
                            embedding_dimension = file_embedding_dimension
                    await db.execute_raw(
                        f"""
                        DELETE FROM workspace_code_index_dirty_paths
                        WHERE workspace_id = {_sql(workspace_id)} AND path = {_sql(path)}
                        """
                    )
                    await self._update_job_progress(job_id, processed_files=idx, current_file=path)
                repo_settings = await IndexerRepository().get_settings()
                await self._refresh_state_counts(
                    workspace_id,
                    status="ready",
                    embedding_dimension=embedding_dimension,
                    embedding_config_hash=repo_settings.get_embedding_config_hash(),
                )
                await self._complete_job(job_id)
            except Exception as exc:
                if job_id:
                    await self._fail_job(job_id, str(exc))
                raise
        except Exception as exc:
            dirty_ids = [str(row.get("id") or "") for row in dirty_rows if row.get("id")]
            if dirty_ids:
                quoted_ids = ", ".join(_sql(item) for item in dirty_ids)
                await db.execute_raw(
                    f"""
                    UPDATE workspace_code_index_dirty_paths
                    SET attempt_count = attempt_count + 1,
                        last_error = {_sql(str(exc)[:2000])},
                        updated_at = NOW()
                    WHERE id IN ({quoted_ids})
                    """
                )
            await db.execute_raw(
                f"""
                UPDATE workspace_code_index_states
                SET status = 'failed', last_error = {_sql(str(exc)[:2000])}, updated_at = NOW()
                WHERE workspace_id = {_sql(workspace_id)}
                """
            )
            raise

    async def reindex_workspace(
        self,
        workspace_id: str,
        *,
        processed_dirty_ids: list[str] | None = None,
        process_started_at: datetime | None = None,
    ) -> None:
        index_name = await self.ensure_state(workspace_id)
        backend = get_pgvector_backend()
        await backend.delete_index(index_name)
        db = await get_db()
        await db.execute_raw(f"DELETE FROM filesystem_file_metadata WHERE index_name = {_sql(index_name)}")
        await db.execute_raw(f"DELETE FROM workspace_code_symbols WHERE workspace_id = {_sql(workspace_id)}")

        files = await self.collect_indexable_files(workspace_id)
        total_files = len(files)
        job_id = await self._start_job(workspace_id, index_name, total_files=total_files, phase=WorkspaceCodeIndexJobPhase.LOADING_FILES)
        try:
            processed_files = 0
            embedding_dimension: int | None = None
            processed_chunks = 0
            for file_path in files:
                rel_path = file_path.relative_to(self.userspace_service._workspace_files_dir(workspace_id)).as_posix()
                chunk_count, file_embedding_dimension = await self.index_file(
                    workspace_id,
                    rel_path,
                    job_id=job_id,
                    processed_chunks_so_far=processed_chunks,
                )
                processed_chunks += chunk_count
                if file_embedding_dimension is not None:
                    embedding_dimension = file_embedding_dimension
                processed_files += 1
                if job_id:
                    await self._update_job_progress(job_id, processed_files=processed_files, current_file=rel_path)
            if job_id:
                await self._update_job_progress(job_id, phase=WorkspaceCodeIndexJobPhase.FINALIZING)
            if processed_dirty_ids:
                quoted_ids = ", ".join(_sql(item) for item in processed_dirty_ids)
                started_at_sql = _sql((process_started_at or utc_now()).isoformat())
                await db.execute_raw(
                    f"""
                    DELETE FROM workspace_code_index_dirty_paths
                    WHERE workspace_id = {_sql(workspace_id)}
                      AND id IN ({quoted_ids})
                      AND dirty_at <= {started_at_sql}::timestamptz
                    """
                )
            else:
                await db.execute_raw(f"DELETE FROM workspace_code_index_dirty_paths WHERE workspace_id = {_sql(workspace_id)}")
            repo_settings = await IndexerRepository().get_settings()
            await self._refresh_state_counts(
                workspace_id,
                status="ready",
                reconciled=True,
                embedding_dimension=embedding_dimension,
                embedding_config_hash=repo_settings.get_embedding_config_hash(),
            )
            if job_id:
                await self._complete_job(job_id)
        except Exception as exc:
            if job_id:
                await self._fail_job(job_id, str(exc))
            raise

    async def index_file(
        self,
        workspace_id: str,
        path: str,
        *,
        job_id: str | None = None,
        processed_chunks_so_far: int = 0,
    ) -> tuple[int, int | None]:
        normalized_path = _normalize_path(path)
        if not normalized_path:
            return 0, None
        files_root = self.userspace_service._workspace_files_dir(workspace_id)
        file_path = (files_root / normalized_path).resolve()
        try:
            file_path.relative_to(files_root.resolve())
        except ValueError:
            return 0, None
        if not file_path.exists() or not file_path.is_file():
            await self.delete_file_from_index(workspace_id, normalized_path)
            return 0, None
        mount_prefixes = await self._workspace_mount_prefixes(workspace_id)
        if any(self._workspace_path_matches_mount_prefix(normalized_path, prefix) for prefix in mount_prefixes):
            await self.delete_file_from_index(workspace_id, normalized_path)
            return 0, None
        if is_excluded_by_patterns(file_path, files_root, _INDEXABLE_EXCLUDE_PATTERNS):
            await self.delete_file_from_index(workspace_id, normalized_path)
            return 0, None
        if not should_index_file_type(file_path, matches_include_pattern=False, ocr_enabled=False):
            await self.delete_file_from_index(workspace_id, normalized_path)
            return 0, None

        index_name = await self.ensure_state(workspace_id)
        app_settings = await get_app_settings()
        if job_id:
            await self._update_job_progress(job_id, current_file=normalized_path, phase=WorkspaceCodeIndexJobPhase.CHUNKING)
        chunks = await self._filesystem_indexer._load_and_chunk_file(
            file_path,
            FilesystemConnectionConfig(
                base_path=str(files_root),
                index_name=index_name,
                recursive=True,
                chunk_size=int(app_settings.get("chunk_size") or 1000),
                chunk_overlap=int(app_settings.get("chunk_overlap") or 200),
                max_file_size_mb=5,
                ocr_mode=OcrMode.DISABLED,
                vector_store_type=VectorStoreType.PGVECTOR,
            ),
            bool(app_settings.get("chunking_use_tokens", True)),
        )
        await self.delete_file_from_index(workspace_id, normalized_path)
        if not chunks:
            return 0, None

        total_chunks = processed_chunks_so_far + len(chunks)
        if job_id:
            await self._update_job_progress(
                job_id,
                total_chunks=total_chunks,
                processed_chunks=processed_chunks_so_far,
                current_file=normalized_path,
                phase=WorkspaceCodeIndexJobPhase.EMBEDDING,
            )
        embeddings_model = await get_embeddings_model(app_settings, logger_override=logger)
        embeddings = await embed_documents_subbatched(embeddings_model, chunks, logger_override=logger)
        backend = get_pgvector_backend()
        await backend.store_embeddings(
            index_name,
            normalized_path,
            chunks,
            embeddings,
            {
                "source": normalized_path,
                "workspace_id": workspace_id,
                "index_kind": "userspace_workspace_code",
            },
        )
        await self._upsert_file_metadata(index_name, normalized_path, file_path, len(chunks))
        if job_id:
            await self._update_job_progress(
                job_id,
                total_chunks=total_chunks,
                processed_chunks=total_chunks,
                current_file=normalized_path,
                phase=WorkspaceCodeIndexJobPhase.INDEXING_SYMBOLS,
            )
        await self._replace_symbols(workspace_id, normalized_path, file_path)

        return len(chunks), len(embeddings[0]) if embeddings else None

    async def delete_file_from_index(self, workspace_id: str, path: str) -> None:
        normalized_path = _normalize_path(path)
        if not normalized_path:
            return
        index_name = await self.ensure_state(workspace_id)
        await get_pgvector_backend().delete_file_embeddings(index_name, normalized_path)
        db = await get_db()
        await db.execute_raw(f"DELETE FROM filesystem_file_metadata WHERE index_name = {_sql(index_name)} AND file_path = {_sql(normalized_path)}")
        await db.execute_raw(f"DELETE FROM workspace_code_symbols WHERE workspace_id = {_sql(workspace_id)} AND path = {_sql(normalized_path)}")

    async def delete_workspace_index(self, workspace_id: str) -> None:
        task = self._workspace_tasks.pop(workspace_id, None)
        if task and not task.done():
            task.cancel()
        index_name = self.index_name_for_workspace(workspace_id)
        await get_pgvector_backend().delete_index(index_name)
        db = await get_db()
        await db.execute_raw(f"DELETE FROM filesystem_file_metadata WHERE index_name = {_sql(index_name)}")
        await db.execute_raw(f"DELETE FROM workspace_code_symbols WHERE workspace_id = {_sql(workspace_id)}")
        await db.execute_raw(f"DELETE FROM workspace_code_index_dirty_paths WHERE workspace_id = {_sql(workspace_id)}")
        await db.execute_raw(f"DELETE FROM workspace_code_index_jobs WHERE workspace_id = {_sql(workspace_id)}")
        await db.execute_raw(f"DELETE FROM workspace_code_index_states WHERE workspace_id = {_sql(workspace_id)}")

    async def _upsert_file_metadata(self, index_name: str, rel_path: str, file_path: Path, chunk_count: int) -> None:
        db = await get_db()
        stat_result = file_path.stat()
        file_hash = await asyncio.to_thread(compute_file_hash, file_path)
        mime_type = mimetypes.guess_type(str(file_path))[0]
        await db.execute_raw(
            f"""
            INSERT INTO filesystem_file_metadata
                (id, index_name, file_path, file_hash, file_size, mime_type, chunk_count, last_indexed)
            VALUES
                ({_sql(str(uuid.uuid4()))}, {_sql(index_name)}, {_sql(rel_path)}, {_sql(file_hash)}, {stat_result.st_size}, {_sql(mime_type)}, {chunk_count}, NOW())
            ON CONFLICT (index_name, file_path) DO UPDATE SET
                file_hash = EXCLUDED.file_hash,
                file_size = EXCLUDED.file_size,
                mime_type = EXCLUDED.mime_type,
                chunk_count = EXCLUDED.chunk_count,
                last_indexed = EXCLUDED.last_indexed
            """
        )

    async def _replace_symbols(self, workspace_id: str, rel_path: str, file_path: Path) -> None:
        db = await get_db()
        await db.execute_raw(f"DELETE FROM workspace_code_symbols WHERE workspace_id = {_sql(workspace_id)} AND path = {_sql(rel_path)}")
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        _imports, definitions = await asyncio.to_thread(extract_metadata, text, file_path.suffix.lower())
        if not definitions:
            return
        values = []
        for signature in definitions[:200]:
            values.append(
                "("
                + ", ".join(
                    [
                        _sql(str(uuid.uuid4())),
                        _sql(workspace_id),
                        _sql(rel_path),
                        _sql("definition"),
                        _sql(_extract_symbol_name(signature)),
                        _sql(signature),
                        "NOW()",
                    ]
                )
                + ")"
            )
        await db.execute_raw(
            f"""
            INSERT INTO workspace_code_symbols (id, workspace_id, path, kind, name, signature, created_at)
            VALUES {", ".join(values)}
            """
        )

    async def _start_job(
        self,
        workspace_id: str,
        index_name: str,
        *,
        total_files: int | None = None,
        phase: WorkspaceCodeIndexJobPhase = WorkspaceCodeIndexJobPhase.COLLECTING,
    ) -> str:
        """Create a canonical workspace code index job row and return its id."""
        db = await get_db()
        rows = await db.query_raw(
            f"""
            INSERT INTO workspace_code_index_jobs
                (id, workspace_id, index_name, status, phase, total_files, processed_files, total_chunks, processed_chunks, current_file, created_at, started_at)
            VALUES
                ({_sql(str(uuid.uuid4()))}, {_sql(workspace_id)}, {_sql(index_name)}, 'indexing', {_sql(phase.value)}, {total_files or 0}, 0, 0, 0, NULL, NOW(), NOW())
            RETURNING id
            """
        )
        return str(rows[0].get("id") or "") if rows else ""

    async def _update_job_progress(
        self,
        job_id: str,
        *,
        processed_files: int | None = None,
        total_chunks: int | None = None,
        processed_chunks: int | None = None,
        current_file: str | None | object = _UNSET_CURRENT_FILE,
        phase: WorkspaceCodeIndexJobPhase | None = None,
    ) -> None:
        """Update progress columns on a job row."""
        db = await get_db()
        set_clauses = []
        if processed_files is not None:
            set_clauses.append(f"processed_files = {processed_files}")
        if total_chunks is not None:
            set_clauses.append(f"total_chunks = {total_chunks}")
        if processed_chunks is not None:
            set_clauses.append(f"processed_chunks = {processed_chunks}")
        if phase is not None:
            set_clauses.append(f"phase = {_sql(phase.value)}")
        if current_file is not _UNSET_CURRENT_FILE:
            if current_file is None:
                set_clauses.append("current_file = NULL")
            else:
                set_clauses.append(f"current_file = {_sql(str(current_file))}")
        if not set_clauses:
            return
        await db.execute_raw(
            f"""
            UPDATE workspace_code_index_jobs
            SET {", ".join(set_clauses)}
            WHERE id = {_sql(job_id)}
            """
        )

    async def _complete_job(self, job_id: str) -> None:
        db = await get_db()
        await db.execute_raw(
            f"""
            UPDATE workspace_code_index_jobs
            SET status = 'completed', current_file = NULL, completed_at = NOW()
            WHERE id = {_sql(job_id)}
            """
        )

    async def _fail_job(self, job_id: str, error_message: str) -> None:
        db = await get_db()
        await db.execute_raw(
            f"""
            UPDATE workspace_code_index_jobs
            SET status = 'failed', error_message = {_sql(str(error_message)[:2000])}, current_file = NULL, completed_at = NOW()
            WHERE id = {_sql(job_id)}
            """
        )

    async def _refresh_state_counts(
        self,
        workspace_id: str,
        *,
        status: str,
        reconciled: bool = False,
        embedding_dimension: int | None = None,
        embedding_config_hash: str | None = None,
    ) -> None:
        index_name = await self.ensure_state(workspace_id)
        db = await get_db()
        stats = await db.query_raw(
            f"""
            SELECT COUNT(*)::int AS chunk_count, COUNT(DISTINCT file_path)::int AS file_count
            FROM filesystem_embeddings
            WHERE index_name = {_sql(index_name)}
            """
        )
        symbol_stats = await db.query_raw(f"SELECT COUNT(*)::int AS symbol_count FROM workspace_code_symbols WHERE workspace_id = {_sql(workspace_id)}")
        chunk_count = int(stats[0].get("chunk_count") or 0) if stats else 0
        file_count = int(stats[0].get("file_count") or 0) if stats else 0
        symbol_count = int(symbol_stats[0].get("symbol_count") or 0) if symbol_stats else 0
        reconciled_sql = ", last_reconciled_at = NOW()" if reconciled else ""
        dimension_sql = f", embedding_dimension = {embedding_dimension}" if embedding_dimension is not None else ""
        hash_sql = f", embedding_config_hash = {_sql(embedding_config_hash)}" if embedding_config_hash else ""
        await db.execute_raw(
            f"""
            UPDATE workspace_code_index_states
            SET status = {_sql(status)},
                last_indexed_at = NOW(),
                file_count = {file_count},
                chunk_count = {chunk_count},
                symbol_count = {symbol_count},
                last_error = NULL,
                updated_at = NOW()
                {reconciled_sql}
                {dimension_sql}
                {hash_sql}
            WHERE workspace_id = {_sql(workspace_id)}
            """
        )

    async def search_workspace_code(
        self,
        *,
        workspace_id: str,
        query: str,
        mode: Literal["semantic", "symbols", "hybrid"] = "hybrid",
        max_results: int = 8,
        max_chars_per_result: int = 1200,
    ) -> dict[str, Any]:
        query = str(query or "").strip()
        max_results = max(1, min(int(max_results or 8), _MAX_SEARCH_RESULTS))
        max_chars_per_result = max(200, min(int(max_chars_per_result or 1200), 6000))
        if not query:
            return {"status": "rejected", "results": [], "error": "query is required"}

        if not await self._enabled():
            return {
                "status": "disabled",
                "workspace_id": workspace_id,
                "mode": mode,
                "query": query,
                "results": [],
                "result_count": 0,
                "error": "User Space code indexing is disabled",
            }

        index_name = await self.ensure_state(workspace_id)
        db = await get_db()
        state_rows = await db.query_raw(f"SELECT status, last_error FROM workspace_code_index_states WHERE workspace_id = {_sql(workspace_id)} LIMIT 1")
        status = str(state_rows[0].get("status") if state_rows else "pending")
        results: list[dict[str, Any]] = []

        if mode in {"symbols", "hybrid"}:
            escaped_query = _escape_like(query)
            symbol_rows = await db.query_raw(
                f"""
                SELECT path, kind, name, signature FROM workspace_code_symbols
                WHERE workspace_id = {_sql(workspace_id)}
                  AND (name ILIKE {_sql("%" + escaped_query + "%")} ESCAPE '\\' OR signature ILIKE {_sql("%" + escaped_query + "%")} ESCAPE '\\')
                ORDER BY path ASC, name ASC
                LIMIT {max_results}
                """
            )
            for row in symbol_rows:
                results.append(
                    {
                        "path": row.get("path"),
                        "kind": row.get("kind"),
                        "symbol": row.get("name"),
                        "snippet": row.get("signature"),
                        "score": 1.0,
                        "source": "symbol",
                    }
                )

        if mode in {"semantic", "hybrid"} and len(results) < max_results:
            app_settings = await get_app_settings()
            embeddings_model = await get_embeddings_model(app_settings, logger_override=logger)
            if embeddings_model is None:
                raise ValueError("Embeddings model is not configured")
            query_embedding = await asyncio.to_thread(embeddings_model.embed_query, query)
            semantic_rows = await search_pgvector_embeddings(
                "filesystem_embeddings",
                query_embedding,
                index_name=index_name,
                max_results=max_results,
                logger_override=logger,
            )
            seen = {(str(item.get("path")), str(item.get("snippet"))) for item in results}
            for row in semantic_rows:
                snippet = str(row.get("content") or "")[:max_chars_per_result]
                key = (str(row.get("file_path") or ""), snippet)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "path": row.get("file_path"),
                        "chunk_index": row.get("chunk_index"),
                        "snippet": snippet,
                        "score": row.get("similarity"),
                        "source": "semantic",
                    }
                )
                if len(results) >= max_results:
                    break

        return {
            "status": status,
            "workspace_id": workspace_id,
            "mode": mode,
            "query": query,
            "results": results[:max_results],
            "result_count": min(len(results), max_results),
        }


workspace_code_index_service = WorkspaceCodeIndexService()
