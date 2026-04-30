"""
Database repository for indexer jobs and metadata.

Provides Prisma-backed persistence for IndexJob and IndexMetadata,
replacing the previous JSON file-based storage.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, cast

from prisma import Json, Prisma
from prisma.enums import ChatTaskStatus as PrismaChatTaskStatus
from prisma.enums import IndexStatus as PrismaIndexStatus
from prisma.enums import ToolType as PrismaToolType
from prisma.enums import VectorStoreType as PrismaVectorStoreType
from prisma.models import IndexJob as PrismaIndexJob
from prisma.models import IndexMetadata as PrismaIndexMetadata

from ragtime.core.database import get_db
from ragtime.core.encryption import (
    CONNECTION_CONFIG_PASSWORD_FIELDS,
    decrypt_json_passwords,
    decrypt_secret,
    encrypt_json_passwords,
    encrypt_secret,
)
from ragtime.core.logging import get_logger
from ragtime.core.tokenization import count_tokens
from ragtime.core.userspace_preview_sandbox import (
    USERSPACE_PREVIEW_SANDBOX_DEFAULT_FLAGS,
    normalize_userspace_preview_sandbox_flags,
)
from ragtime.indexer.models import (
    SCHEMA_INDEXER_CAPABLE_TOOL_TYPES,
    AppSettings,
    ChatMessage,
    ChatTask,
    ChatTaskStatus,
    ChatTaskStreamingState,
    Conversation,
    ConversationBranch,
    ConversationBranchKind,
    ConversationBranchSummary,
    IndexConfig,
    IndexJob,
    IndexStatus,
    MessageSnapshotRestore,
    ProviderPromptDebugRecord,
    ToolCallRecord,
    ToolConfig,
    ToolGroup,
    ToolOutputMode,
    ToolType,
    VectorStoreType,
)
from ragtime.indexer.tool_health import get_heartbeat_timeout_seconds
from ragtime.indexer.tool_presentation import normalize_tool_presentation
from ragtime.indexer.utils import safe_tool_name
from ragtime.indexer.vector_backends import FAISS_INDEX_BASE_PATH

logger = get_logger(__name__)


def _sanitize_for_postgres(text: str) -> str:
    """
    Sanitize text for PostgreSQL storage.

    PostgreSQL text columns cannot store null bytes (\\u0000).
    This function removes them to prevent storage errors.

    Args:
        text: The text to sanitize.

    Returns:
        Sanitized text safe for PostgreSQL storage.
    """
    return text.replace("\x00", "") if text else text


def _sanitize_json_for_postgres(value: Any) -> Any:
    """Recursively sanitize nested JSON-like payloads for PostgreSQL storage."""
    if isinstance(value, str):
        return _sanitize_for_postgres(value)
    if isinstance(value, list):
        return [_sanitize_json_for_postgres(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json_for_postgres(item) for item in value]
    if isinstance(value, dict):
        sanitized: dict[Any, Any] = {}
        for key, item in value.items():
            sanitized_key = _sanitize_for_postgres(key) if isinstance(key, str) else key
            sanitized[sanitized_key] = _sanitize_json_for_postgres(item)
        return sanitized
    return value


def _safe_serialize(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value) if value is not None else ""


def _sql_quote_literal(value: Any) -> str:
    """Quote a scalar value for trusted raw SQL construction."""
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _identifier_in_allowed_models(
    identifier: str,
    allowed_models: list[str],
) -> bool:
    normalized_identifier = identifier.strip()
    return bool(normalized_identifier) and normalized_identifier in {
        value.strip() for value in allowed_models if value.strip()
    }


def _resolve_default_conversation_model(app_settings: Optional[AppSettings]) -> str:
    """Resolve the configured default model for a new conversation."""
    if app_settings is None:
        return "gpt-4-turbo"

    manual_default = str(getattr(app_settings, "default_chat_model", "") or "").strip()
    configured_model = str(app_settings.llm_model or "").strip()
    allowed_models = [
        str(value).strip()
        for value in (app_settings.allowed_chat_models or [])
        if str(value).strip()
    ]

    if allowed_models:
        if manual_default and _identifier_in_allowed_models(
            manual_default, allowed_models
        ):
            return manual_default
        if configured_model and _identifier_in_allowed_models(
            configured_model, allowed_models
        ):
            return configured_model
        return allowed_models[0]

    if manual_default:
        return manual_default
    if configured_model:
        return configured_model
    return "gpt-4-turbo"


def _extract_tool_calls_from_events(
    events: list[dict[str, Any]] | None,
) -> list[ToolCallRecord] | None:
    """Build compatibility tool-call records from chronological message events."""
    if not events:
        return None

    tool_calls: list[ToolCallRecord] = []
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "tool":
            continue
        tool_calls.append(
            ToolCallRecord(
                tool=str(event.get("tool", "")),
                input=event.get("input"),
                output=event.get("output"),
                connection=event.get("connection"),
                presentation=normalize_tool_presentation(
                    str(event.get("tool", "")),
                    cast(Optional[dict[str, Any]], event.get("connection")),
                    cast(Optional[dict[str, Any]], event.get("presentation")),
                ),
            )
        )

    return tool_calls or None


def _normalize_message_events(
    events: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Normalize additive tool event metadata for backwards compatibility."""
    if not events:
        return events

    normalized_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "tool":
            normalized_events.append(event)
            continue

        normalized_event = dict(event)
        presentation = normalize_tool_presentation(
            str(event.get("tool", "")),
            cast(Optional[dict[str, Any]], event.get("connection")),
            cast(Optional[dict[str, Any]], event.get("presentation")),
        )
        if presentation:
            normalized_event["presentation"] = presentation
        normalized_events.append(normalized_event)

    return normalized_events


def _synthesize_events_from_legacy_message(
    message: Any,
) -> list[dict[str, Any]] | None:
    """Build chronological events for legacy assistant messages that only stored tool_calls."""
    if not isinstance(message, dict):
        return None

    legacy_tool_calls = message.get("tool_calls")
    if not isinstance(legacy_tool_calls, list) or not legacy_tool_calls:
        return None

    events: list[dict[str, Any]] = []
    for tool_call in legacy_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        event = {
            "type": "tool",
            "tool": tool_call.get("tool", ""),
            "input": tool_call.get("input"),
            "output": tool_call.get("output"),
            "connection": tool_call.get("connection"),
        }
        presentation = normalize_tool_presentation(
            str(tool_call.get("tool", "")),
            cast(Optional[dict[str, Any]], tool_call.get("connection")),
            cast(Optional[dict[str, Any]], tool_call.get("presentation")),
        )
        if presentation:
            event["presentation"] = presentation
        events.append(event)

    content = str(message.get("content", "") or "")
    if content:
        events.append({"type": "content", "content": content})

    return events or None


def _coerce_legacy_message_entry(message: Any) -> dict[str, Any] | None:
    """Normalize a persisted message entry into a dictionary payload."""
    if isinstance(message, dict):
        return message

    if isinstance(message, str):
        stripped = message.strip()
        if not stripped:
            return None

        try:
            parsed = json.loads(stripped)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            return parsed

        if isinstance(parsed, str):
            parsed = parsed.strip()
            if parsed:
                return {"role": "assistant", "content": parsed}
            return None

        return {"role": "assistant", "content": stripped}

    return None


def _normalize_message_payloads(messages_data: Any) -> list[dict[str, Any]]:
    """Normalize persisted conversation messages into a list of dict entries."""
    raw_entries: list[Any]
    if messages_data is None:
        raw_entries = []
    elif isinstance(messages_data, list):
        raw_entries = messages_data
    elif isinstance(messages_data, tuple):
        raw_entries = list(messages_data)
    else:
        raw_entries = [messages_data]

    normalized: list[dict[str, Any]] = []
    for entry in raw_entries:
        message = _coerce_legacy_message_entry(entry)
        if message is not None:
            normalized.append(message)
    return normalized


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return count_tokens(text)


def _estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimate tokens for a message.

    If events exist, they contain the full picture (content + tool calls).
    Otherwise fall back to content + legacy tool_calls to avoid double-counting.
    """
    events = message.get("events")
    if isinstance(events, list) and events:
        tokens = 0
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("type") == "content":
                tokens += _estimate_text_tokens(str(event.get("content", "")))
            elif event.get("type") == "tool":
                tokens += _estimate_text_tokens(_safe_serialize(event.get("input")))
                tokens += _estimate_text_tokens(str(event.get("output", "")))
        return tokens

    # Fallback: use content + legacy tool_calls
    tokens = _estimate_text_tokens(str(message.get("content", "")))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            tokens += _estimate_text_tokens(_safe_serialize(call.get("input")))
            tokens += _estimate_text_tokens(str(call.get("output", "")))

    return tokens


def _estimate_conversation_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(_estimate_message_tokens(message) for message in messages)


def _to_prisma_index_status(status: IndexStatus) -> PrismaIndexStatus:
    """Convert model IndexStatus to Prisma IndexStatus."""
    return PrismaIndexStatus(status.value)


def _to_prisma_tool_type(tool_type: ToolType) -> PrismaToolType:
    """Convert model ToolType to Prisma ToolType."""
    return PrismaToolType(tool_type.value)


def _to_prisma_vector_store_type(
    store_type: VectorStoreType,
) -> PrismaVectorStoreType:
    """Convert model VectorStoreType to Prisma enum."""
    return PrismaVectorStoreType(store_type.value)


def _from_prisma_vector_store_type(
    store_type: PrismaVectorStoreType,
) -> VectorStoreType:
    """Convert Prisma vector store enum to model VectorStoreType."""
    return VectorStoreType(store_type.value)


def _to_prisma_task_status(status: ChatTaskStatus) -> PrismaChatTaskStatus:
    """Convert model ChatTaskStatus to Prisma ChatTaskStatus."""
    return PrismaChatTaskStatus(status.value)


class IndexerRepository:
    """Repository for indexer data persistence via Prisma."""

    def __init__(self) -> None:
        self._conversation_branch_locks: dict[str, asyncio.Lock] = {}

    def _get_conversation_branch_lock(self, conversation_id: str) -> asyncio.Lock:
        lock = self._conversation_branch_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._conversation_branch_locks[conversation_id] = lock
        return lock

    async def _get_db(self) -> Prisma:
        """Get database connection."""
        return await get_db()

    # -------------------------------------------------------------------------
    # Job Operations
    # -------------------------------------------------------------------------

    async def create_job(self, job: IndexJob) -> IndexJob:
        """Create a new indexing job in the database."""
        db = await self._get_db()

        # First create the config
        config_data = {
            "name": job.config.name,
            "filePatterns": job.config.file_patterns,
            "excludePatterns": job.config.exclude_patterns,
            "chunkSize": job.config.chunk_size,
            "chunkOverlap": job.config.chunk_overlap,
            "embeddingModel": job.config.embedding_model,
        }

        # Create job with nested config
        job_data: dict[str, Any] = {
            "id": job.id,
            "name": job.name,
            "status": _to_prisma_index_status(job.status),
            "sourceType": job.source_type,
            "sourcePath": job.source_path,
            "gitUrl": job.git_url,
            "gitBranch": job.git_branch,
            "gitToken": encrypt_secret(job.git_token) if job.git_token else None,
            "totalFiles": job.total_files,
            "processedFiles": job.processed_files,
            "totalChunks": job.total_chunks,
            "processedChunks": job.processed_chunks,
            "errorMessage": job.error_message,
            "createdAt": job.created_at,
            "startedAt": job.started_at,
            "completedAt": job.completed_at,
            "config": {"create": config_data},
        }

        await db.indexjob.create(data=cast(Any, job_data), include={"config": True})

        logger.debug(f"Created job {job.id} in database")
        return job

    async def get_job(self, job_id: str) -> Optional[IndexJob]:
        """Get a job by ID."""
        db = await self._get_db()

        prisma_job = await db.indexjob.find_unique(
            where={"id": job_id}, include={"config": True}
        )

        if prisma_job is None:
            return None

        return self._prisma_job_to_model(prisma_job)

    async def list_jobs(self) -> list[IndexJob]:
        """List all jobs."""
        db = await self._get_db()

        prisma_jobs = await db.indexjob.find_many(
            include={"config": True}, order={"createdAt": "desc"}
        )

        return [self._prisma_job_to_model(j) for j in prisma_jobs]

    async def update_job(self, job: IndexJob) -> IndexJob:
        """Update an existing job."""
        db = await self._get_db()

        await db.indexjob.update(
            where={"id": job.id},
            data={  # type: ignore[arg-type]
                "status": _to_prisma_index_status(job.status),
                "totalFiles": job.total_files,
                "processedFiles": job.processed_files,
                "totalChunks": job.total_chunks,
                "processedChunks": job.processed_chunks,
                "errorMessage": job.error_message,
                "startedAt": job.started_at,
                "completedAt": job.completed_at,
            },
        )

        logger.debug(f"Updated job {job.id} in database")
        return job

    async def get_active_job_for_index(self, name: str) -> Optional[IndexJob]:
        """Get an active (pending/processing) job for the given index name.

        Returns the first active job found, or None if no active jobs exist.
        """
        db = await self._get_db()

        # Find any pending or processing job for this index
        prisma_job = await db.indexjob.find_first(
            where={  # type: ignore[arg-type]
                "name": name,
                "OR": [
                    {"status": PrismaIndexStatus.pending},
                    {"status": PrismaIndexStatus.processing},
                ],
            },
            include={"config": True},
            order={"createdAt": "desc"},
        )

        if prisma_job:
            return self._prisma_job_to_model(prisma_job)
        return None

    async def count_completed_jobs(self, name: str) -> int:
        """Count how many successfully completed jobs exist for an index name."""
        db = await self._get_db()
        return await db.indexjob.count(
            where={  # type: ignore[arg-type]
                "name": name,
                "status": PrismaIndexStatus.completed,
            }
        )

    def _prisma_job_to_model(self, prisma_job: PrismaIndexJob) -> IndexJob:
        """Convert Prisma job to Pydantic model."""
        if prisma_job.config is None:
            raise ValueError(f"Job {prisma_job.id} has no config")

        config = IndexConfig(
            name=prisma_job.config.name,
            file_patterns=prisma_job.config.filePatterns,
            exclude_patterns=prisma_job.config.excludePatterns,
            chunk_size=prisma_job.config.chunkSize,
            chunk_overlap=prisma_job.config.chunkOverlap,
            embedding_model=prisma_job.config.embeddingModel,
        )

        return IndexJob(
            id=prisma_job.id,
            name=prisma_job.name,
            status=IndexStatus(prisma_job.status),
            config=config,
            source_type=prisma_job.sourceType,
            source_path=prisma_job.sourcePath,
            git_url=prisma_job.gitUrl,
            git_branch=prisma_job.gitBranch,
            git_token=(
                decrypt_secret(prisma_job.gitToken)
                if getattr(prisma_job, "gitToken", None)
                else None
            ),
            total_files=prisma_job.totalFiles,
            processed_files=prisma_job.processedFiles,
            total_chunks=prisma_job.totalChunks,
            processed_chunks=prisma_job.processedChunks,
            error_message=prisma_job.errorMessage,
            created_at=prisma_job.createdAt,
            started_at=prisma_job.startedAt,
            completed_at=prisma_job.completedAt,
        )

    # -------------------------------------------------------------------------
    # Index Metadata Operations
    # -------------------------------------------------------------------------

    async def upsert_index_metadata(
        self,
        name: str,
        path: str,
        document_count: int,
        chunk_count: int,
        size_bytes: int,
        source_type: str,
        source: Optional[str],
        config_snapshot: Optional[dict],
        description: str = "",
        git_branch: Optional[str] = None,
        git_token: Optional[str] = None,
        display_name: Optional[str] = None,
        vector_store_type: VectorStoreType = VectorStoreType.FAISS,
    ) -> None:
        """Create or update index metadata."""
        db = await self._get_db()

        # Build create/update data - only include configSnapshot if it has a value
        # Encrypt gitToken for secure storage
        encrypted_git_token = encrypt_secret(git_token) if git_token else None
        prisma_vector_store_type = _to_prisma_vector_store_type(vector_store_type)

        create_data: dict = {
            "name": name,
            "displayName": display_name,
            "description": description,
            "path": path,
            "documentCount": document_count,
            "chunkCount": chunk_count,
            "sizeBytes": size_bytes,
            "sourceType": source_type,
            "source": source,
            "gitBranch": git_branch,
            "gitToken": encrypted_git_token,
            "vectorStoreType": prisma_vector_store_type,
            "createdAt": datetime.utcnow(),
            "lastModified": datetime.utcnow(),
        }
        update_data: dict = {
            "description": description,
            "path": path,
            "documentCount": document_count,
            "chunkCount": chunk_count,
            "sizeBytes": size_bytes,
            "sourceType": source_type,
            "source": source,
            "gitBranch": git_branch,
            "gitToken": encrypted_git_token,
            "vectorStoreType": prisma_vector_store_type,
            "lastModified": datetime.utcnow(),
        }

        # Only update displayName if provided (don't overwrite with None)
        if display_name is not None:
            update_data["displayName"] = display_name

        # Only include configSnapshot if we have actual data
        if config_snapshot is not None:
            create_data["configSnapshot"] = Json(config_snapshot)
            update_data["configSnapshot"] = Json(config_snapshot)

        await db.indexmetadata.upsert(
            where={"name": name},
            data=cast(
                Any,
                {
                    "create": create_data,
                    "update": update_data,
                },
            ),
        )

        logger.debug(f"Upserted metadata for index {name}")

    async def get_index_metadata(self, name: str) -> Optional[PrismaIndexMetadata]:
        """Get metadata for a specific index."""
        db = await self._get_db()
        return await db.indexmetadata.find_unique(where={"name": name})

    async def update_index_metadata_counts(
        self,
        name: str,
        document_count: int,
        chunk_count: int,
        size_bytes: int,
    ) -> None:
        """Update only the document/chunk/size counts for an index.

        Used to restore counts after a failed re-index without overwriting
        other metadata fields like description, config_snapshot, etc.
        """
        db = await self._get_db()
        await db.indexmetadata.update(
            where={"name": name},
            data={
                "documentCount": document_count,
                "chunkCount": chunk_count,
                "sizeBytes": size_bytes,
                "lastModified": datetime.utcnow(),
            },
        )
        logger.debug(
            f"Updated counts for index {name}: {document_count} docs, {chunk_count} chunks"
        )

    async def list_index_metadata(self) -> list[PrismaIndexMetadata]:
        """List all index metadata."""
        db = await self._get_db()
        return await db.indexmetadata.find_many(order={"createdAt": "desc"})

    async def delete_index_metadata(self, name: str) -> bool:
        """Delete metadata for an index."""
        db = await self._get_db()

        try:
            await db.indexmetadata.delete(where={"name": name})
            logger.debug(f"Deleted metadata for index {name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete metadata for index {name}: {e}")
            return False

    async def set_index_enabled(self, name: str, enabled: bool) -> bool:
        """Set the enabled status for an index."""
        db = await self._get_db()

        try:
            await db.indexmetadata.update(
                where={"name": name}, data={"enabled": enabled}
            )
            logger.debug(f"Set index {name} enabled={enabled}")
            return True
        except Exception as e:
            logger.warning(f"Failed to update enabled status for index {name}: {e}")
            return False

    async def update_index_description(self, name: str, description: str) -> bool:
        """Update the description for an index."""
        db = await self._get_db()

        try:
            await db.indexmetadata.update(
                where={"name": name}, data={"description": description}
            )
            logger.debug(f"Updated description for index {name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to update description for index {name}: {e}")
            return False

    async def update_index_weight(self, name: str, weight: float) -> bool:
        """Update the search weight for an index."""
        db = await self._get_db()

        # Clamp weight to valid range
        weight = max(0.0, min(10.0, weight))

        try:
            await db.indexmetadata.update(
                where={"name": name}, data=cast(Any, {"searchWeight": weight})
            )
            logger.debug(f"Updated search weight for index {name} to {weight}")
            return True
        except Exception as e:
            logger.warning(f"Failed to update search weight for index {name}: {e}")
            return False

    async def update_index_memory_stats(self, name: str, stats: dict[str, Any]) -> bool:
        """Update the observed memory statistics for an index.

        Args:
            name: Index name
            stats: Dictionary with keys:
                - embedding_dimension: int (dimension of vectors)
                - steady_memory_bytes: int (RAM after loading)
                - peak_memory_bytes: int (peak RAM during loading)
                - load_time_seconds: float (time to load)

        Returns:
            True if update succeeded, False otherwise
        """
        db = await self._get_db()

        update_data: dict[str, Any] = {}
        if "embedding_dimension" in stats and stats["embedding_dimension"] is not None:
            update_data["embeddingDimension"] = stats["embedding_dimension"]
        if "steady_memory_bytes" in stats and stats["steady_memory_bytes"] is not None:
            update_data["steadyMemoryBytes"] = stats["steady_memory_bytes"]
        if "peak_memory_bytes" in stats and stats["peak_memory_bytes"] is not None:
            update_data["peakMemoryBytes"] = stats["peak_memory_bytes"]
        if "load_time_seconds" in stats and stats["load_time_seconds"] is not None:
            update_data["loadTimeSeconds"] = stats["load_time_seconds"]

        if not update_data:
            return True  # Nothing to update

        try:
            await db.indexmetadata.update(
                where={"name": name}, data=cast(Any, update_data)
            )
            logger.debug(f"Updated memory stats for index {name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to update memory stats for index {name}: {e}")
            return False

    async def update_index_config(
        self,
        name: str,
        git_branch: str | None = None,
        config_snapshot: dict | None = None,
    ) -> bool:
        """Update the git branch and/or config snapshot for an index."""
        db = await self._get_db()

        update_data = {}
        if git_branch is not None:
            update_data["gitBranch"] = git_branch
        if config_snapshot is not None:
            update_data["configSnapshot"] = Json(config_snapshot)

        if not update_data:
            return True  # Nothing to update

        try:
            await db.indexmetadata.update(
                where={"name": name}, data=cast(Any, update_data)
            )
            logger.debug(f"Updated config for index {name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to update config for index {name}: {e}")
            return False

    async def rename_index(
        self, old_name: str, new_name: str, display_name: Optional[str] = None
    ) -> bool:
        """
        Rename an index in the database.

        Updates the index_metadata name (primary key) and associated index_jobs.
        The caller is responsible for renaming the FAISS index directory on disk.

        Args:
            old_name: Current index name (safe tool name)
            new_name: New index name (safe tool name)
            display_name: Human-readable display name (original user input)

        Returns:
            True if rename succeeded, False otherwise
        """
        db = await self._get_db()

        try:
            # Check if new name already exists
            existing = await db.indexmetadata.find_unique(where={"name": new_name})
            if existing:
                logger.warning(f"Cannot rename index: name '{new_name}' already exists")
                return False

            # Get the existing metadata
            metadata = await db.indexmetadata.find_unique(where={"name": old_name})
            if not metadata:
                logger.warning(f"Cannot rename index: '{old_name}' not found")
                return False

            # Update the path to reflect the new name
            # Only replace the directory name at the end, not other occurrences
            base_dir = os.path.dirname(metadata.path)
            new_path = os.path.join(base_dir, new_name)

            # Build create data - only include configSnapshot if it has a value
            # Use new display_name if provided, otherwise preserve existing
            metadata_any = cast(Any, metadata)
            create_data: dict[str, Any] = {
                "name": new_name,
                "displayName": (
                    display_name
                    if display_name is not None
                    else getattr(metadata_any, "displayName", None)
                ),
                "description": metadata.description,
                "path": new_path,
                "documentCount": metadata.documentCount,
                "chunkCount": metadata.chunkCount,
                "sizeBytes": metadata.sizeBytes,
                "enabled": metadata.enabled,
                "searchWeight": getattr(metadata_any, "searchWeight", 1.0),
                "sourceType": metadata.sourceType,
                "source": metadata.source,
                "gitBranch": getattr(metadata_any, "gitBranch", None),
                # gitToken is already encrypted in the database, just copy it
                "gitToken": getattr(metadata_any, "gitToken", None),
                "createdAt": metadata.createdAt,
                "lastModified": metadata.lastModified,
            }

            # Only include configSnapshot if we have actual data
            if metadata.configSnapshot is not None:
                create_data["configSnapshot"] = Json(metadata.configSnapshot)

            # Use a transaction to ensure atomicity - delete and create together
            # Also update index_jobs that reference this index name
            async with db.tx() as tx:
                # Update index_jobs.name to new name
                await tx.indexjob.update_many(
                    where={"name": old_name}, data={"name": new_name}
                )
                await tx.indexmetadata.delete(where={"name": old_name})
                await tx.indexmetadata.create(data=cast(Any, create_data))

            logger.info(f"Renamed index '{old_name}' to '{new_name}' in database")
            return True
        except Exception as e:
            logger.exception(
                f"Failed to rename index '{old_name}' to '{new_name}': {e}"
            )
            return False

    # -------------------------------------------------------------------------
    # Application Settings Operations
    # -------------------------------------------------------------------------

    async def get_settings(self) -> AppSettings:
        """Get application settings, creating defaults if needed."""
        db = await self._get_db()

        prisma_settings = cast(
            Any, await db.appsettings.find_unique(where={"id": "default"})
        )

        if prisma_settings is None:
            # Create default settings
            prisma_settings = cast(
                Any, await db.appsettings.create(data={"id": "default"})
            )
            logger.info("Created default application settings")

        settings: Any = prisma_settings

        # Decrypt secrets for API response
        openai_key = settings.openaiApiKey or ""
        anthropic_key = settings.anthropicApiKey or ""
        github_models_api_token = getattr(settings, "githubModelsApiToken", "") or ""
        github_copilot_access_token = (
            getattr(settings, "githubCopilotAccessToken", "") or ""
        )
        github_copilot_refresh_token = (
            getattr(settings, "githubCopilotRefreshToken", "") or ""
        )
        github_copilot_oauth_refresh_token = (
            getattr(settings, "githubCopilotOauthRefreshToken", "") or ""
        )
        mcp_password = getattr(settings, "mcpDefaultRoutePassword", None)

        # Decrypt if encrypted (starts with 'enc::')
        if openai_key:
            openai_key = decrypt_secret(openai_key)
        if anthropic_key:
            anthropic_key = decrypt_secret(anthropic_key)
        if github_models_api_token:
            github_models_api_token = decrypt_secret(github_models_api_token)
        if github_copilot_access_token:
            github_copilot_access_token = decrypt_secret(github_copilot_access_token)
        if github_copilot_refresh_token:
            github_copilot_refresh_token = decrypt_secret(github_copilot_refresh_token)
        if github_copilot_oauth_refresh_token:
            github_copilot_oauth_refresh_token = decrypt_secret(
                github_copilot_oauth_refresh_token
            )
        if mcp_password:
            mcp_password = decrypt_secret(mcp_password)

        try:
            userspace_preview_sandbox_flags = normalize_userspace_preview_sandbox_flags(
                getattr(settings, "userspacePreviewSandboxFlags", None)
            )
        except ValueError as exc:
            logger.warning(
                "Invalid userspace preview sandbox flags in app settings; "
                "falling back to defaults: %s",
                exc,
            )
            userspace_preview_sandbox_flags = list(
                USERSPACE_PREVIEW_SANDBOX_DEFAULT_FLAGS
            )

        return AppSettings(
            id=settings.id,
            # Server branding
            server_name=getattr(settings, "serverName", "Ragtime"),
            # Embedding settings
            embedding_provider=settings.embeddingProvider,
            embedding_model=settings.embeddingModel,
            embedding_dimensions=getattr(settings, "embeddingDimensions", None),
            ollama_protocol=settings.ollamaProtocol,
            ollama_host=settings.ollamaHost,
            ollama_port=settings.ollamaPort,
            ollama_base_url=settings.ollamaBaseUrl,
            llama_cpp_protocol=getattr(settings, "llamaCppProtocol", "http"),
            llama_cpp_host=getattr(settings, "llamaCppHost", "host.docker.internal"),
            llama_cpp_port=getattr(settings, "llamaCppPort", 8081),
            llama_cpp_base_url=getattr(
                settings,
                "llamaCppBaseUrl",
                "http://host.docker.internal:8081",
            ),
            lmstudio_protocol=getattr(settings, "lmstudioProtocol", "http"),
            lmstudio_host=getattr(settings, "lmstudioHost", "host.docker.internal"),
            lmstudio_port=getattr(settings, "lmstudioPort", 1234),
            lmstudio_base_url=getattr(
                settings,
                "lmstudioBaseUrl",
                "http://host.docker.internal:1234",
            ),
            lmstudio_api_key=decrypt_secret(
                getattr(settings, "lmstudioApiKey", None) or ""
            ),
            # LLM settings
            llm_provider=settings.llmProvider,
            llm_model=settings.llmModel,
            llm_max_tokens=getattr(settings, "llmMaxTokens", 4096),
            image_payload_max_width=getattr(settings, "imagePayloadMaxWidth", 1024),
            image_payload_max_height=getattr(settings, "imagePayloadMaxHeight", 1024),
            image_payload_max_pixels=getattr(settings, "imagePayloadMaxPixels", 786432),
            image_payload_max_bytes=getattr(settings, "imagePayloadMaxBytes", 350000),
            llm_ollama_protocol=getattr(settings, "llmOllamaProtocol", "http"),
            llm_ollama_host=getattr(settings, "llmOllamaHost", "localhost"),
            llm_ollama_port=getattr(settings, "llmOllamaPort", 11434),
            llm_ollama_base_url=getattr(
                settings,
                "llmOllamaBaseUrl",
                "http://localhost:11434",
            ),
            llm_llama_cpp_protocol=getattr(settings, "llmLlamaCppProtocol", "http"),
            llm_llama_cpp_host=getattr(
                settings, "llmLlamaCppHost", "host.docker.internal"
            ),
            llm_llama_cpp_port=getattr(settings, "llmLlamaCppPort", 8080),
            llm_llama_cpp_base_url=getattr(
                settings,
                "llmLlamaCppBaseUrl",
                "http://host.docker.internal:8080",
            ),
            llm_lmstudio_protocol=getattr(settings, "llmLmstudioProtocol", "http"),
            llm_lmstudio_host=getattr(
                settings, "llmLmstudioHost", "host.docker.internal"
            ),
            llm_lmstudio_port=getattr(settings, "llmLmstudioPort", 1234),
            llm_lmstudio_base_url=getattr(
                settings,
                "llmLmstudioBaseUrl",
                "http://host.docker.internal:1234",
            ),
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            github_models_api_token=github_models_api_token,
            github_copilot_access_token=github_copilot_access_token,
            github_copilot_refresh_token=github_copilot_refresh_token,
            github_copilot_oauth_refresh_token=github_copilot_oauth_refresh_token,
            github_copilot_token_expires_at=getattr(
                settings, "githubCopilotTokenExpiresAt", None
            ),
            github_copilot_enterprise_url=getattr(
                settings, "githubCopilotEnterpriseUrl", None
            ),
            github_copilot_base_url=getattr(
                settings,
                "githubCopilotBaseUrl",
                "https://api.githubcopilot.com",
            ),
            include_copilot_third_party_models=getattr(
                settings, "includeCopilotThirdPartyModels", False
            ),
            has_github_copilot_auth=bool(github_copilot_access_token),
            allowed_chat_models=settings.allowedChatModels or [],
            default_chat_model=getattr(settings, "defaultChatModel", None),
            # OpenAPI model configuration
            allowed_openapi_models=getattr(settings, "allowedOpenapiModels", None)
            or [],
            openapi_sync_chat_models=getattr(settings, "openapiSyncChatModels", True),
            max_iterations=settings.maxIterations,
            # Tool settings
            enabled_tools=settings.enabledTools,
            odoo_container=settings.odooContainer,
            postgres_container=settings.postgresContainer,
            postgres_host=settings.postgresHost,
            postgres_port=settings.postgresPort,
            postgres_user=settings.postgresUser,
            postgres_password=(
                decrypt_secret(settings.postgresPassword)
                if settings.postgresPassword
                else ""
            ),
            postgres_database=settings.postgresDb,
            max_query_results=settings.maxQueryResults,
            query_timeout=settings.queryTimeout,
            enable_write_ops=settings.enableWriteOps,
            # Token optimization
            max_tool_output_chars=getattr(settings, "maxToolOutputChars", 5000),
            scratchpad_window_size=getattr(settings, "scratchpadWindowSize", 6),
            # Search configuration
            search_results_k=getattr(settings, "searchResultsK", 5),
            aggregate_search=getattr(settings, "aggregateSearch", True),
            # Retrieval optimization
            search_use_mmr=getattr(settings, "searchUseMmr", True),
            search_mmr_lambda=getattr(settings, "searchMmrLambda", 0.5),
            context_token_budget=getattr(settings, "contextTokenBudget", 4000),
            chunking_use_tokens=getattr(settings, "chunkingUseTokens", True),
            ivfflat_lists=getattr(settings, "ivfflatLists", 100),
            # Performance configuration
            sequential_index_loading=getattr(settings, "sequentialIndexLoading", False),
            # API Tool Output configuration
            tool_output_mode=getattr(settings, "toolOutputMode", "default"),
            # MCP configuration
            mcp_enabled=getattr(settings, "mcpEnabled", False),
            mcp_default_route_auth=getattr(settings, "mcpDefaultRouteAuth", False),
            mcp_default_route_auth_method=getattr(
                settings, "mcpDefaultRouteAuthMethod", "password"
            ),
            mcp_default_route_password=mcp_password,
            mcp_default_route_client_id=getattr(
                settings, "mcpDefaultRouteClientId", None
            ),
            mcp_default_route_allowed_group=getattr(
                settings, "mcpDefaultRouteAllowedGroup", None
            ),
            has_mcp_default_password=getattr(settings, "mcpDefaultRoutePassword", None)
            is not None,
            # Embedding dimension tracking
            embedding_dimension=getattr(settings, "embeddingDimension", None),
            embedding_config_hash=getattr(settings, "embeddingConfigHash", None),
            # OCR configuration
            default_ocr_mode=getattr(settings, "defaultOcrMode", "disabled"),
            default_ocr_vision_model=getattr(settings, "defaultOcrVisionModel", None),
            ocr_concurrency_limit=getattr(settings, "ocrConcurrencyLimit", 1),
            ollama_embedding_timeout_seconds=getattr(
                settings, "ollamaEmbeddingTimeoutSeconds", 180
            ),
            # User Space configuration
            snapshot_retention_days=getattr(settings, "snapshotRetentionDays", 0),
            snapshot_stale_branch_threshold=getattr(
                settings, "snapshotStaleBranchThreshold", 50
            ),
            userspace_preview_sandbox_flags=userspace_preview_sandbox_flags,
            userspace_duplicate_copy_files_default=getattr(
                settings,
                "userspaceDuplicateCopyFilesDefault",
                True,
            ),
            userspace_duplicate_copy_metadata_default=getattr(
                settings,
                "userspaceDuplicateCopyMetadataDefault",
                True,
            ),
            userspace_duplicate_copy_chats_default=getattr(
                settings,
                "userspaceDuplicateCopyChatsDefault",
                False,
            ),
            userspace_duplicate_copy_mounts_default=getattr(
                settings,
                "userspaceDuplicateCopyMountsDefault",
                False,
            ),
            updated_at=settings.updatedAt,
        )

    async def update_settings(self, updates: dict) -> AppSettings:
        """Update application settings with provided fields."""
        db = await self._get_db()

        # Map snake_case to camelCase for Prisma
        field_mapping = {
            # Server branding
            "server_name": "serverName",
            # Embedding settings
            "embedding_provider": "embeddingProvider",
            "embedding_model": "embeddingModel",
            "embedding_dimensions": "embeddingDimensions",
            "ollama_protocol": "ollamaProtocol",
            "ollama_host": "ollamaHost",
            "ollama_port": "ollamaPort",
            "ollama_base_url": "ollamaBaseUrl",
            "llama_cpp_protocol": "llamaCppProtocol",
            "llama_cpp_host": "llamaCppHost",
            "llama_cpp_port": "llamaCppPort",
            "llama_cpp_base_url": "llamaCppBaseUrl",
            "lmstudio_protocol": "lmstudioProtocol",
            "lmstudio_host": "lmstudioHost",
            "lmstudio_port": "lmstudioPort",
            "lmstudio_base_url": "lmstudioBaseUrl",
            "lmstudio_api_key": "lmstudioApiKey",
            # LLM settings
            "llm_provider": "llmProvider",
            "llm_model": "llmModel",
            "llm_max_tokens": "llmMaxTokens",
            "image_payload_max_width": "imagePayloadMaxWidth",
            "image_payload_max_height": "imagePayloadMaxHeight",
            "image_payload_max_pixels": "imagePayloadMaxPixels",
            "image_payload_max_bytes": "imagePayloadMaxBytes",
            "llm_ollama_protocol": "llmOllamaProtocol",
            "llm_ollama_host": "llmOllamaHost",
            "llm_ollama_port": "llmOllamaPort",
            "llm_ollama_base_url": "llmOllamaBaseUrl",
            "llm_llama_cpp_protocol": "llmLlamaCppProtocol",
            "llm_llama_cpp_host": "llmLlamaCppHost",
            "llm_llama_cpp_port": "llmLlamaCppPort",
            "llm_llama_cpp_base_url": "llmLlamaCppBaseUrl",
            "llm_lmstudio_protocol": "llmLmstudioProtocol",
            "llm_lmstudio_host": "llmLmstudioHost",
            "llm_lmstudio_port": "llmLmstudioPort",
            "llm_lmstudio_base_url": "llmLmstudioBaseUrl",
            "openai_api_key": "openaiApiKey",
            "anthropic_api_key": "anthropicApiKey",
            "github_models_api_token": "githubModelsApiToken",
            "github_copilot_access_token": "githubCopilotAccessToken",
            "github_copilot_refresh_token": "githubCopilotRefreshToken",
            "github_copilot_oauth_refresh_token": "githubCopilotOauthRefreshToken",
            "github_copilot_token_expires_at": "githubCopilotTokenExpiresAt",
            "github_copilot_enterprise_url": "githubCopilotEnterpriseUrl",
            "github_copilot_base_url": "githubCopilotBaseUrl",
            "include_copilot_third_party_models": "includeCopilotThirdPartyModels",
            "allowed_chat_models": "allowedChatModels",
            "default_chat_model": "defaultChatModel",
            "allowed_openapi_models": "allowedOpenapiModels",
            "openapi_sync_chat_models": "openapiSyncChatModels",
            "max_iterations": "maxIterations",
            # Token optimization settings
            "max_tool_output_chars": "maxToolOutputChars",
            "scratchpad_window_size": "scratchpadWindowSize",
            # Tool settings
            "enabled_tools": "enabledTools",
            "odoo_container": "odooContainer",
            "postgres_container": "postgresContainer",
            "postgres_host": "postgresHost",
            "postgres_port": "postgresPort",
            "postgres_user": "postgresUser",
            "postgres_password": "postgresPassword",
            "postgres_database": "postgresDb",
            "max_query_results": "maxQueryResults",
            "query_timeout": "queryTimeout",
            "enable_write_ops": "enableWriteOps",
            # Search configuration
            "search_results_k": "searchResultsK",
            "aggregate_search": "aggregateSearch",
            # Retrieval optimization
            "search_use_mmr": "searchUseMmr",
            "search_mmr_lambda": "searchMmrLambda",
            "context_token_budget": "contextTokenBudget",
            "chunking_use_tokens": "chunkingUseTokens",
            "ivfflat_lists": "ivfflatLists",
            # Performance configuration
            "sequential_index_loading": "sequentialIndexLoading",
            # API Tool Output configuration
            "tool_output_mode": "toolOutputMode",
            # MCP configuration
            "mcp_enabled": "mcpEnabled",
            "mcp_default_route_auth": "mcpDefaultRouteAuth",
            "mcp_default_route_auth_method": "mcpDefaultRouteAuthMethod",
            "mcp_default_route_password": "mcpDefaultRoutePassword",
            "mcp_default_route_client_id": "mcpDefaultRouteClientId",
            "mcp_default_route_allowed_group": "mcpDefaultRouteAllowedGroup",
            # Embedding dimension tracking (internal use)
            "embedding_dimension": "embeddingDimension",
            "embedding_config_hash": "embeddingConfigHash",
            # OCR configuration
            "default_ocr_mode": "defaultOcrMode",
            "default_ocr_vision_model": "defaultOcrVisionModel",
            "ocr_concurrency_limit": "ocrConcurrencyLimit",
            "ollama_embedding_timeout_seconds": "ollamaEmbeddingTimeoutSeconds",
            # User Space configuration
            "snapshot_retention_days": "snapshotRetentionDays",
            "snapshot_stale_branch_threshold": "snapshotStaleBranchThreshold",
            "userspace_preview_sandbox_flags": "userspacePreviewSandboxFlags",
            "userspace_duplicate_copy_files_default": "userspaceDuplicateCopyFilesDefault",
            "userspace_duplicate_copy_metadata_default": "userspaceDuplicateCopyMetadataDefault",
            "userspace_duplicate_copy_chats_default": "userspaceDuplicateCopyChatsDefault",
            "userspace_duplicate_copy_mounts_default": "userspaceDuplicateCopyMountsDefault",
        }

        # Build update data with only provided fields
        update_data = {}
        for snake_key, camel_key in field_mapping.items():
            if snake_key in updates and updates[snake_key] is not None:
                update_data[camel_key] = updates[snake_key]

        # Encrypt secret fields before storage
        # These need to be reversibly encrypted so we can show them in the UI
        secret_fields = [
            "openai_api_key",
            "anthropic_api_key",
            "github_models_api_token",
            "github_copilot_access_token",
            "github_copilot_refresh_token",
            "github_copilot_oauth_refresh_token",
            "postgres_password",
            "lmstudio_api_key",
        ]
        for field in secret_fields:
            if field in updates and updates[field]:
                camel_key = field_mapping[field]
                update_data[camel_key] = encrypt_secret(updates[field])

        # Special handling for mcp_default_route_password:
        # - Empty string clears the password (set to None)
        # - Non-empty string gets encrypted before storage
        if "mcp_default_route_password" in updates:
            pwd_value = updates["mcp_default_route_password"]
            if pwd_value == "":
                # Clear password
                update_data["mcpDefaultRoutePassword"] = None
            elif pwd_value is not None:
                # Encrypt and store
                update_data["mcpDefaultRoutePassword"] = encrypt_secret(pwd_value)

        # Special handling for mcp_default_route_client_id:
        # - Empty string clears the client_id (set to None)
        if "mcp_default_route_client_id" in updates:
            cid_value = updates["mcp_default_route_client_id"]
            if cid_value == "":
                update_data["mcpDefaultRouteClientId"] = None
            elif cid_value is not None:
                update_data["mcpDefaultRouteClientId"] = cid_value

        # Special handling for mcp_default_route_allowed_group:
        # - Empty string clears the group (set to None/empty)
        if "mcp_default_route_allowed_group" in updates:
            group_value = updates["mcp_default_route_allowed_group"]
            if group_value in {"", None}:
                update_data["mcpDefaultRouteAllowedGroup"] = None
            elif group_value is not None:
                update_data["mcpDefaultRouteAllowedGroup"] = group_value

        # Optional manual default chat model override:
        # - None or empty string clears override (automatic selection)
        if "default_chat_model" in updates:
            default_chat_model_value = updates["default_chat_model"]
            if default_chat_model_value is None:
                update_data["defaultChatModel"] = None
            else:
                normalized_default_chat_model = str(default_chat_model_value).strip()
                update_data["defaultChatModel"] = normalized_default_chat_model or None

        # Fields that the UI may intentionally clear with an explicit null.
        if (
            "embedding_dimensions" in updates
            and updates["embedding_dimensions"] is None
        ):
            update_data["embeddingDimensions"] = None

        if (
            "default_ocr_vision_model" in updates
            and updates["default_ocr_vision_model"] is None
        ):
            update_data["defaultOcrVisionModel"] = None

        # Special handling for GitHub Copilot nullable fields.
        # These may need explicit clearing to NULL.
        if "github_copilot_token_expires_at" in updates:
            update_data["githubCopilotTokenExpiresAt"] = updates[
                "github_copilot_token_expires_at"
            ]

        if "github_copilot_enterprise_url" in updates:
            enterprise_value = updates["github_copilot_enterprise_url"]
            if enterprise_value == "":
                update_data["githubCopilotEnterpriseUrl"] = None
            else:
                update_data["githubCopilotEnterpriseUrl"] = enterprise_value

        if not update_data:
            return await self.get_settings()

        # Ensure default settings exist
        await self.get_settings()

        await db.appsettings.update(
            where={"id": "default"}, data=update_data  # type: ignore[arg-type]
        )

        logger.info(f"Updated settings: {list(update_data.keys())}")
        return await self.get_settings()

    async def update_embedding_tracking(
        self, dimension: int, config_hash: str
    ) -> AppSettings:
        """
        Update embedding dimension and config hash after successful indexing.
        Called by filesystem_service after first batch of embeddings is inserted.
        """
        db = await self._get_db()
        await self.get_settings()  # Ensure exists

        await db.appsettings.update(
            where={"id": "default"},
            data=cast(
                Any,
                {
                    "embeddingDimension": dimension,
                    "embeddingConfigHash": config_hash,
                },
            ),
        )  # type: ignore[arg-type]
        logger.info(
            f"Updated embedding tracking: dimension={dimension}, hash={config_hash}"
        )
        return await self.get_settings()

    async def clear_embedding_tracking(self) -> AppSettings:
        """
        Clear embedding tracking when all filesystem indexes are deleted.
        This allows a fresh start with a new embedding provider.
        """
        db = await self._get_db()
        await self.get_settings()  # Ensure exists

        await db.appsettings.update(
            where={"id": "default"},
            data=cast(
                Any,
                {
                    "embeddingDimension": None,
                    "embeddingConfigHash": None,
                },
            ),
        )  # type: ignore[arg-type]
        logger.info("Cleared embedding tracking")
        return await self.get_settings()

    # -------------------------------------------------------------------------
    # Tool Configuration Operations
    # -------------------------------------------------------------------------

    async def create_tool_config(self, config: ToolConfig) -> ToolConfig:
        """Create a new tool configuration."""
        db = await self._get_db()

        # Encrypt password fields in connection_config
        encrypted_config = encrypt_json_passwords(
            config.connection_config, CONNECTION_CONFIG_PASSWORD_FIELDS
        )

        create_data: dict[str, Any] = {
            "name": config.name,
            "toolType": _to_prisma_tool_type(config.tool_type),
            "enabled": config.enabled,
            "description": config.description,
            "connectionConfig": Json(encrypted_config),
            "maxResults": config.max_results,
            "timeout": config.timeout,
            "timeoutMaxSeconds": config.timeout_max_seconds,
            "allowWrite": config.allow_write,
        }
        if config.group_id:
            create_data["groupId"] = config.group_id

        prisma_config = await db.toolconfig.create(
            data=create_data,  # type: ignore[arg-type]
            include={"group": True},
        )

        logger.info(f"Created tool config: {config.name} ({config.tool_type.value})")
        return self._prisma_tool_config_to_model(prisma_config)

    async def get_tool_config(self, config_id: str) -> Optional[ToolConfig]:
        """Get a tool configuration by ID."""
        db = await self._get_db()

        prisma_config = await db.toolconfig.find_unique(where={"id": config_id})
        if prisma_config is None:
            return None

        return self._prisma_tool_config_to_model(prisma_config)

    async def list_tool_configs(self, enabled_only: bool = False) -> list[ToolConfig]:
        """List all tool configurations."""
        db = await self._get_db()

        where = {"enabled": True} if enabled_only else {}
        prisma_configs = await db.toolconfig.find_many(
            where=where,  # type: ignore[arg-type]
            order={"createdAt": "desc"},
            include={"group": True},
        )

        configs = [self._prisma_tool_config_to_model(c) for c in prisma_configs]

        # Auto-disable FAISS filesystem indexes if file is missing
        # This handles cases where indexing failed or was interrupted, preventing
        # the UI from showing a "green" toggle for a broken index.
        for config in configs:
            if config.tool_type == ToolType.FILESYSTEM_INDEXER:
                conn_config = config.connection_config or {}
                # Check for explicit 'faiss' type (defaults to pgvector if unset)
                if conn_config.get("vector_store_type") == "faiss":
                    index_name = conn_config.get("index_name")
                    if index_name:
                        index_dir = FAISS_INDEX_BASE_PATH / index_name
                        faiss_file = index_dir / "index.faiss"

                        if not await asyncio.to_thread(faiss_file.exists):
                            reason = f"FAISS file missing at {faiss_file}"
                            config.disabled_reason = reason

                            if config.enabled:
                                logger.warning(
                                    f"Auto-disabling filesystem index '{config.name}': {reason}"
                                )
                                # Update DB
                                await db.toolconfig.update(
                                    where={"id": config.id},
                                    data={"enabled": False},  # type: ignore[arg-type]
                                )
                                # Update local object so UI reflects change immediately
                                config.enabled = False

        if enabled_only:
            return [c for c in configs if c.enabled]

        return configs

    async def update_tool_config(
        self, config_id: str, updates: dict[str, Any]
    ) -> Optional[ToolConfig]:
        """Update a tool configuration."""
        db = await self._get_db()

        # Map snake_case to camelCase
        field_mapping = {
            "name": "name",
            "enabled": "enabled",
            "description": "description",
            "connection_config": "connectionConfig",
            "max_results": "maxResults",
            "timeout": "timeout",
            "timeout_max_seconds": "timeoutMaxSeconds",
            "allow_write": "allowWrite",
            "last_test_at": "lastTestAt",
            "last_test_result": "lastTestResult",
            "last_test_error": "lastTestError",
        }

        update_data = {}
        for snake_key, camel_key in field_mapping.items():
            if snake_key in updates and updates[snake_key] is not None:
                value = updates[snake_key]
                if snake_key == "connection_config":
                    # Preserve non-password fields from existing config
                    # This ensures metadata like last_schema_indexed_at is not lost
                    existing_config = await db.toolconfig.find_unique(
                        where={"id": config_id}
                    )
                    if existing_config and existing_config.connectionConfig:
                        merged_config = dict(existing_config.connectionConfig)
                        # Update only the fields provided in the request
                        merged_config.update(value)
                        value = merged_config

                    # Encrypt password fields
                    value = encrypt_json_passwords(
                        value, CONNECTION_CONFIG_PASSWORD_FIELDS
                    )
                    value = Json(value)
                update_data[camel_key] = value

        # Handle group_id: empty string means ungroup, non-empty means set
        if "group_id" in updates:
            gid = updates["group_id"]
            update_data["groupId"] = None if gid in (None, "") else gid

        if not update_data:
            return await self.get_tool_config(config_id)

        try:
            await db.toolconfig.update(
                where={"id": config_id}, data=update_data  # type: ignore[arg-type]
            )
            logger.info(f"Updated tool config {config_id}: {list(update_data.keys())}")
            return await self.get_tool_config(config_id)
        except Exception as e:
            logger.error(f"Failed to update tool config {config_id}: {e}")
            return None

    async def delete_tool_config(self, config_id: str) -> bool:
        """Delete a tool configuration."""
        db = await self._get_db()

        try:
            await db.toolconfig.delete(where={"id": config_id})
            logger.info(f"Deleted tool config {config_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete tool config {config_id}: {e}")
            return False

    async def rename_tool_config(
        self, config_id: str, new_name: str
    ) -> tuple[Optional[ToolConfig], dict[str, int]]:
        """
        Rename a tool configuration and update all associated index names.

        This method ensures consistency across:
        - tool_configs.name
        - schema_embeddings.index_name (for postgres/mssql tools)
        - pdm_embeddings.index_name (for solidworks_pdm tools)
        - pdm_document_metadata.index_name (for solidworks_pdm tools)
        - filesystem_embeddings.index_name (for filesystem_indexer tools)
        - filesystem_file_metadata.index_name (for filesystem_indexer tools)
        - schema_index_jobs.index_name (for postgres/mssql tools)
        - pdm_index_jobs.index_name (for solidworks_pdm tools)
        - connection_config.index_name (for filesystem_indexer tools)

        Args:
            config_id: The tool configuration ID to rename
            new_name: The new display name for the tool

        Returns:
            Tuple of (updated ToolConfig or None, dict of update counts)
        """
        db = await self._get_db()
        update_counts: dict[str, int] = {
            "schema_embeddings": 0,
            "schema_index_jobs": 0,
            "pdm_embeddings": 0,
            "pdm_document_metadata": 0,
            "pdm_index_jobs": 0,
            "filesystem_embeddings": 0,
            "filesystem_file_metadata": 0,
        }

        # Get current tool config
        tool = await self.get_tool_config(config_id)
        if not tool:
            logger.warning(f"Cannot rename tool: config {config_id} not found")
            return None, update_counts

        old_safe_name = safe_tool_name(tool.name)
        new_safe_name = safe_tool_name(new_name)

        # If safe names are the same, just update the display name
        if old_safe_name == new_safe_name:
            updated = await self.update_tool_config(config_id, {"name": new_name})
            return updated, update_counts

        try:
            # Update based on tool type
            if tool.tool_type in SCHEMA_INDEXER_CAPABLE_TOOL_TYPES:
                # Schema embeddings: schema_{old_safe_name} -> schema_{new_safe_name}
                old_index_name = f"schema_{old_safe_name}"
                new_index_name = f"schema_{new_safe_name}"

                result = await db.execute_raw(
                    f"UPDATE schema_embeddings SET index_name = '{new_index_name}' "
                    f"WHERE index_name = '{old_index_name}'"
                )
                update_counts["schema_embeddings"] = (
                    result if isinstance(result, int) else 0
                )

                # Also update schema_index_jobs.index_name
                result = await db.execute_raw(
                    f"UPDATE schema_index_jobs SET index_name = '{new_index_name}' "
                    f"WHERE tool_config_id = '{config_id}'"
                )
                update_counts["schema_index_jobs"] = (
                    result if isinstance(result, int) else 0
                )

            elif tool.tool_type == ToolType.SOLIDWORKS_PDM:
                # PDM embeddings: pdm_{old_safe_name} -> pdm_{new_safe_name}
                old_index_name = f"pdm_{old_safe_name}"
                new_index_name = f"pdm_{new_safe_name}"

                result = await db.execute_raw(
                    f"UPDATE pdm_embeddings SET index_name = '{new_index_name}' "
                    f"WHERE index_name = '{old_index_name}'"
                )
                update_counts["pdm_embeddings"] = (
                    result if isinstance(result, int) else 0
                )

                result = await db.execute_raw(
                    f"UPDATE pdm_document_metadata SET index_name = '{new_index_name}' "
                    f"WHERE index_name = '{old_index_name}'"
                )
                update_counts["pdm_document_metadata"] = (
                    result if isinstance(result, int) else 0
                )

                # Also update pdm_index_jobs.index_name
                result = await db.execute_raw(
                    f"UPDATE pdm_index_jobs SET index_name = '{new_index_name}' "
                    f"WHERE tool_config_id = '{config_id}'"
                )
                update_counts["pdm_index_jobs"] = (
                    result if isinstance(result, int) else 0
                )

            elif tool.tool_type == ToolType.FILESYSTEM_INDEXER:
                # Filesystem embeddings use connection_config.index_name
                fs_old_index_name = (
                    tool.connection_config.get("index_name")
                    if tool.connection_config
                    else None
                )
                fs_new_index_name = new_safe_name

                if fs_old_index_name and fs_old_index_name != fs_new_index_name:
                    # Update embeddings table
                    result = await db.execute_raw(
                        f"UPDATE filesystem_embeddings SET index_name = '{fs_new_index_name}' "
                        f"WHERE index_name = '{fs_old_index_name}'"
                    )
                    update_counts["filesystem_embeddings"] = (
                        result if isinstance(result, int) else 0
                    )

                    # Update file metadata table
                    result = await db.execute_raw(
                        f"UPDATE filesystem_file_metadata SET index_name = '{fs_new_index_name}' "
                        f"WHERE index_name = '{fs_old_index_name}'"
                    )
                    update_counts["filesystem_file_metadata"] = (
                        result if isinstance(result, int) else 0
                    )

                    # Update connection_config.index_name
                    updated_config = dict(tool.connection_config)
                    updated_config["index_name"] = fs_new_index_name
                    await self.update_tool_config(
                        config_id, {"connection_config": updated_config}
                    )

            # Update the tool name
            updated = await self.update_tool_config(config_id, {"name": new_name})

            total_updates = sum(update_counts.values())
            if total_updates > 0:
                logger.info(
                    f"Renamed tool '{tool.name}' to '{new_name}' "
                    f"(safe: '{old_safe_name}' -> '{new_safe_name}'): {update_counts}"
                )
            else:
                logger.info(
                    f"Renamed tool '{tool.name}' to '{new_name}' (no embeddings)"
                )

            return updated, update_counts

        except Exception as e:
            logger.error(f"Failed to rename tool config {config_id}: {e}")
            return None, update_counts

    async def update_tool_test_result(
        self, config_id: str, success: bool, error: Optional[str] = None
    ) -> None:
        """Update the test result for a tool configuration."""
        db = await self._get_db()

        try:
            await db.toolconfig.update(
                where={"id": config_id},
                data={
                    "lastTestAt": datetime.utcnow(),
                    "lastTestResult": success,
                    "lastTestError": error,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to update test result for {config_id}: {e}")

    def _prisma_tool_config_to_model(self, prisma_config: Any) -> ToolConfig:
        """Convert Prisma ToolConfig to Pydantic model."""
        # Decrypt password fields in connection_config
        decrypted_config = decrypt_json_passwords(
            prisma_config.connectionConfig, CONNECTION_CONFIG_PASSWORD_FIELDS
        )

        # Resolve group fields if relation or raw field present
        group_id = getattr(prisma_config, "groupId", None)
        group_name: str | None = None
        group_rel = getattr(prisma_config, "group", None)
        if group_rel is not None:
            group_name = getattr(group_rel, "name", None)

        return ToolConfig(
            id=prisma_config.id,
            name=prisma_config.name,
            tool_type=ToolType(prisma_config.toolType),
            enabled=prisma_config.enabled,
            description=prisma_config.description,
            connection_config=decrypted_config,
            max_results=prisma_config.maxResults,
            timeout=prisma_config.timeout,
            timeout_max_seconds=getattr(prisma_config, "timeoutMaxSeconds", 300),
            allow_write=prisma_config.allowWrite,
            group_id=group_id,
            group_name=group_name,
            last_test_at=prisma_config.lastTestAt,
            last_test_result=prisma_config.lastTestResult,
            last_test_error=prisma_config.lastTestError,
            created_at=prisma_config.createdAt,
            updated_at=prisma_config.updatedAt,
        )

    # -------------------------------------------------------------------------
    # Tool Group Operations
    # -------------------------------------------------------------------------

    async def list_tool_groups(self) -> list["ToolGroup"]:
        """List all tool groups ordered by sort_order then name."""
        db = await self._get_db()
        rows = await db.toolgroup.find_many(
            order=[{"sortOrder": "asc"}, {"name": "asc"}]
        )
        return [
            ToolGroup(
                id=r.id,
                name=r.name,
                description=r.description,
                sort_order=r.sortOrder,
                created_at=r.createdAt,
                updated_at=r.updatedAt,
            )
            for r in rows
        ]

    async def get_tool_group(self, group_id: str) -> "ToolGroup | None":
        db = await self._get_db()
        r = await db.toolgroup.find_unique(where={"id": group_id})
        if r is None:
            return None
        return ToolGroup(
            id=r.id,
            name=r.name,
            description=r.description,
            sort_order=r.sortOrder,
            created_at=r.createdAt,
            updated_at=r.updatedAt,
        )

    async def create_tool_group(
        self, name: str, description: str = "", sort_order: int = 0
    ) -> "ToolGroup":
        db = await self._get_db()
        r = await db.toolgroup.create(
            data={
                "name": name,
                "description": description,
                "sortOrder": sort_order,
            }
        )
        logger.info(f"Created tool group: {name}")
        return ToolGroup(
            id=r.id,
            name=r.name,
            description=r.description,
            sort_order=r.sortOrder,
            created_at=r.createdAt,
            updated_at=r.updatedAt,
        )

    async def update_tool_group(
        self, group_id: str, updates: dict[str, Any]
    ) -> "ToolGroup | None":
        db = await self._get_db()
        data: dict[str, Any] = {}
        if "name" in updates and updates["name"] is not None:
            data["name"] = updates["name"]
        if "description" in updates and updates["description"] is not None:
            data["description"] = updates["description"]
        if "sort_order" in updates and updates["sort_order"] is not None:
            data["sortOrder"] = updates["sort_order"]
        if not data:
            return await self.get_tool_group(group_id)
        try:
            r = await db.toolgroup.update(where={"id": group_id}, data=data)
        except Exception:
            return None
        return ToolGroup(
            id=r.id,
            name=r.name,
            description=r.description,
            sort_order=r.sortOrder,
            created_at=r.createdAt,
            updated_at=r.updatedAt,
        )

    async def delete_tool_group(self, group_id: str) -> bool:
        """Delete a tool group. Tools in the group become ungrouped."""
        db = await self._get_db()
        try:
            await db.toolgroup.delete(where={"id": group_id})
            logger.info(f"Deleted tool group {group_id}")
            return True
        except Exception:
            return False

    async def get_tool_ids_for_groups(self, group_ids: list[str]) -> list[str]:
        """Return enabled tool config IDs that belong to any of the given groups."""
        if not group_ids:
            return []
        db = await self._get_db()
        tools = await db.toolconfig.find_many(
            where={
                "groupId": {"in": group_ids},
                "enabled": True,
            },
        )
        return [t.id for t in tools]

    async def list_healthy_enabled_tool_ids(self) -> list[str]:
        """Return enabled tool config IDs that currently pass heartbeat checks."""
        tool_configs = await self.list_tool_configs(enabled_only=True)
        if not tool_configs:
            return []

        try:
            from ragtime.indexer.routes import _heartbeat_check
        except Exception as exc:
            logger.warning(
                "Heartbeat helper unavailable for default tool selection; "
                "falling back to enabled tools: %s",
                exc,
            )
            return [tool.id for tool in tool_configs if tool.id]

        async def check_single_tool(tool: ToolConfig) -> tuple[str | None, bool]:
            tool_id = tool.id
            if not tool_id:
                return None, False

            connection_config = tool.connection_config or {}
            heartbeat_timeout = get_heartbeat_timeout_seconds(connection_config)

            try:
                result = await asyncio.wait_for(
                    _heartbeat_check(tool.tool_type, connection_config),
                    timeout=heartbeat_timeout,
                )
                return tool_id, bool(result.success)
            except asyncio.TimeoutError:
                return tool_id, False
            except Exception:
                logger.debug(
                    "Default tool heartbeat check failed for %s",
                    tool.name,
                    exc_info=True,
                )
                return tool_id, False

        results = await asyncio.gather(
            *(check_single_tool(tool) for tool in tool_configs)
        )

        return [tool_id for tool_id, is_healthy in results if tool_id and is_healthy]

    # -------------------------------------------------------------------------
    # Conversation Operations
    # -------------------------------------------------------------------------

    async def create_conversation(
        self,
        title: str = "Untitled Chat",
        model: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> Conversation:
        """Create a new conversation."""
        db = await self._get_db()
        resolved_model = (model or "").strip()
        if not resolved_model:
            app_settings = await self.get_settings()
            resolved_model = _resolve_default_conversation_model(app_settings)

        prisma_conv = await db.conversation.create(
            data={
                "id": str(uuid.uuid4()),
                "title": title,
                "model": resolved_model,
                "messages": Json([]),
                "totalTokens": 0,
                "userId": user_id,
                "workspaceId": workspace_id,
            },
            include={"user": True},
        )

        default_tool_ids = await self.list_healthy_enabled_tool_ids()
        for tool_id in default_tool_ids:
            await db.conversationtoolselection.create(
                data={
                    "conversationId": prisma_conv.id,
                    "toolConfigId": tool_id,
                }
            )

        return self._prisma_conversation_to_model(prisma_conv)

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        db = await self._get_db()

        prisma_conv = await db.conversation.find_unique(
            where={"id": conversation_id}, include={"user": True}
        )

        if prisma_conv is None:
            return None

        conv = self._prisma_conversation_to_model(prisma_conv)
        return await self.attach_message_snapshot_links(conv)

    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        include_all: bool = False,
        workspace_id: Optional[str] = None,
    ) -> list[Conversation]:
        """
        List conversations, newest first.

        Args:
            user_id: Filter by user ID (required unless include_all=True)
            include_all: If True, return all conversations (admin only)
        """
        db = await self._get_db()

        where_clause: dict[str, Any] = {}
        if workspace_id is not None:
            if not include_all:
                if not user_id:
                    return []

                workspace = await db.workspace.find_unique(
                    where={"id": workspace_id},
                    include={"members": True},
                )
                if not workspace:
                    return []

                has_workspace_access = bool(
                    workspace.ownerUserId == user_id
                    or any(
                        getattr(member, "userId", None) == user_id
                        for member in list(getattr(workspace, "members", []) or [])
                    )
                )
                if not has_workspace_access:
                    return []

            where_clause["workspaceId"] = workspace_id
        else:
            where_clause["workspaceId"] = None

        if workspace_id is None and not include_all:
            if not user_id:
                return []
            where_clause["OR"] = [
                {"userId": user_id},
                {"members": {"some": {"userId": user_id}}},
            ]

        prisma_convs = await db.conversation.find_many(
            where=where_clause if where_clause else None,  # type: ignore[arg-type]
            order={"updatedAt": "desc"},
            include={"user": True},
        )

        conversations = [self._prisma_conversation_to_model(c) for c in prisma_convs]
        return await self.attach_message_snapshot_links_many(conversations)

    async def list_conversations_by_ids(
        self, conversation_ids: list[str]
    ) -> list[Conversation]:
        """List conversations by explicit IDs, newest first."""
        if not conversation_ids:
            return []

        db = await self._get_db()
        prisma_convs = await db.conversation.find_many(
            where={"id": {"in": conversation_ids}},
            order={"updatedAt": "desc"},
            include={"user": True},
        )
        conversations = [self._prisma_conversation_to_model(c) for c in prisma_convs]
        return await self.attach_message_snapshot_links_many(conversations)

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        events: Optional[List[dict]] = None,
    ) -> Optional[Conversation]:
        """Add a message to a conversation."""
        db = await self._get_db()

        # Sanitize content for PostgreSQL storage (remove null bytes)
        content = _sanitize_for_postgres(content)
        sanitized_events = (
            cast(List[dict], _sanitize_json_for_postgres(events)) if events else None
        )

        # Get current conversation
        prisma_conv = await db.conversation.find_unique(where={"id": conversation_id})
        if not prisma_conv:
            return None

        # Add new message
        messages: List[dict[str, Any]] = _normalize_message_payloads(
            prisma_conv.messages
        )
        new_message: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "message_id": str(uuid.uuid4()),
        }
        # Store chronological events only; compatibility tool_calls are derived on read.
        if sanitized_events:
            new_message["events"] = sanitized_events
        messages.append(new_message)

        # Recompute conversation token total from persisted messages (tiktoken-backed)
        total_tokens = _estimate_conversation_tokens(messages)

        # Update conversation
        updated = await db.conversation.update(
            where={"id": conversation_id},
            data={
                "messages": Json(messages),
                "totalTokens": total_tokens,
                "updatedAt": datetime.utcnow(),
            },
            include={"user": True},
        )

        return self._prisma_conversation_to_model(updated)

    async def update_conversation_title(
        self, conversation_id: str, title: str
    ) -> Optional[Conversation]:
        """Update a conversation's title."""
        db = await self._get_db()

        try:
            updated = await db.conversation.update(
                where={"id": conversation_id},
                data={"title": title, "updatedAt": datetime.utcnow()},
                include={"user": True},
            )
            return self._prisma_conversation_to_model(updated)
        except Exception as e:
            logger.warning(f"Failed to update conversation title: {e}")
            return None

    async def update_conversation_model(
        self, conversation_id: str, model: str
    ) -> Optional[Conversation]:
        """Update a conversation's model."""
        db = await self._get_db()

        try:
            updated = await db.conversation.update(
                where={"id": conversation_id},
                data={"model": model, "updatedAt": datetime.utcnow()},
                include={"user": True},
            )
            return self._prisma_conversation_to_model(updated)
        except Exception as e:
            logger.warning(f"Failed to update conversation model: {e}")
            return None

    async def update_conversation_tool_output_mode(
        self, conversation_id: str, tool_output_mode: str
    ) -> Optional[Conversation]:
        """Update a conversation's tool output mode."""
        db = await self._get_db()

        try:
            updated = await db.conversation.update(
                where={"id": conversation_id},
                data={
                    "toolOutputMode": tool_output_mode,
                    "updatedAt": datetime.utcnow(),
                },
                include={"user": True},
            )
            return self._prisma_conversation_to_model(updated)
        except Exception as e:
            logger.warning(f"Failed to update conversation tool output mode: {e}")
            return None

    async def truncate_messages(
        self, conversation_id: str, keep_count: int
    ) -> Optional[Conversation]:
        """Truncate messages to keep only the first N messages."""
        db = await self._get_db()

        try:
            prisma_conv = await db.conversation.find_unique(
                where={"id": conversation_id}
            )
            if not prisma_conv:
                return None

            messages: List[dict[str, Any]] = _normalize_message_payloads(
                prisma_conv.messages
            )
            truncated = messages[:keep_count]

            # Recalculate tokens, including tool calls and events
            total_tokens = _estimate_conversation_tokens(truncated)

            updated = await db.conversation.update(
                where={"id": conversation_id},
                data={
                    "messages": Json(truncated),
                    "totalTokens": total_tokens,
                    "updatedAt": datetime.utcnow(),
                },
                include={"user": True},
            )
            return self._prisma_conversation_to_model(updated)
        except Exception as e:
            logger.warning(f"Failed to truncate conversation: {e}")
            return None

    # -------------------------------------------------------------------------
    # Conversation Branch Operations
    # -------------------------------------------------------------------------

    async def create_conversation_branch(
        self,
        conversation_id: str,
        branch_point_index: int,
        branch_kind: Optional[ConversationBranchKind] = None,
        user_id: Optional[str] = None,
        parent_branch_id: Optional[str] = None,
        associated_snapshot_id: Optional[str] = None,
    ) -> Optional[ConversationBranch]:
        """Create a branch preserving messages from branch_point_index onward.

        The preserved messages are saved in the branch record, then the
        conversation is truncated to *branch_point_index* messages.
        The conversation stays on the live path after branch creation.
        """
        db = await self._get_db()
        try:
            async with self._get_conversation_branch_lock(conversation_id):
                async with db.tx() as tx:
                    prisma_conv = await tx.conversation.find_unique(
                        where={"id": conversation_id}
                    )
                    if not prisma_conv:
                        return None

                    messages: List[dict[str, Any]] = _normalize_message_payloads(
                        prisma_conv.messages
                    )

                    if branch_point_index < 0 or branch_point_index > len(messages):
                        return None

                    # Absorb existing sibling branches anchored AFTER this
                    # new branch point that share the same parent lineage.
                    # Their preserved content describes the same pre-delete
                    # tail, so consolidating them into this new branch keeps
                    # restoration coherent when the user has stacked
                    # consecutive deletes on adjacent turn messages. Without
                    # this, each delete leaves an orphan branch whose own
                    # `messages[:K]` base no longer contains the context it
                    # was created from, causing duplicate rows and split
                    # X/N nav handles on restore.
                    sibling_branches = await tx.conversationbranch.find_many(
                        where={
                            "conversationId": conversation_id,
                            "branchPointIndex": {"gt": branch_point_index},
                            "parentBranchId": parent_branch_id,
                        },
                        order=[{"branchPointIndex": "asc"}],
                    )

                    preserved: list[dict[str, Any]] = list(
                        messages[branch_point_index:]
                    )
                    absorbed_ids: list[str] = []
                    for sib in sibling_branches:
                        sib_preserved = _normalize_message_payloads(
                            sib.preservedMessages
                        )
                        preserved.extend(sib_preserved)
                        absorbed_ids.append(sib.id)

                    if absorbed_ids:
                        await tx.conversationbranch.delete_many(
                            where={"id": {"in": absorbed_ids}}
                        )

                    branch_id = str(uuid.uuid4())
                    await tx.conversationbranch.create(
                        data={
                            "id": branch_id,
                            "conversationId": conversation_id,
                            "parentBranchId": parent_branch_id,
                            "branchPointIndex": branch_point_index,
                            "branchKind": branch_kind.value if branch_kind else None,
                            "preservedMessages": Json(preserved),
                            "associatedSnapshotId": associated_snapshot_id,
                            "createdByUserId": user_id,
                        }
                    )

                    truncated = messages[:branch_point_index]
                    total_tokens = _estimate_conversation_tokens(truncated)
                    await tx.conversation.update(
                        where={"id": conversation_id},
                        data={
                            "messages": Json(truncated),
                            "totalTokens": total_tokens,
                            "activeBranchId": None,
                            "updatedAt": datetime.utcnow(),
                        },
                    )

                    created_branch = await tx.conversationbranch.find_unique(
                        where={"id": branch_id},
                        include={"createdByUser": True},
                    )

                return self._prisma_branch_to_model(created_branch)
        except Exception as e:
            logger.warning(f"Failed to create conversation branch: {e}")
            return None

    async def switch_conversation_branch(
        self,
        conversation_id: str,
        branch_id: str,
    ) -> Optional[Conversation]:
        """Switch to a different branch by restoring its preserved messages.

        If the conversation is on the live path (active_branch_id is None),
        the current downstream messages are saved into a new auto-created
        branch so they can be recovered later.  If it's already on a saved
        branch, the downstream messages are written back to that branch.
        """
        db = await self._get_db()
        try:
            async with self._get_conversation_branch_lock(conversation_id):
                async with db.tx() as tx:
                    prisma_conv = await tx.conversation.find_unique(
                        where={"id": conversation_id},
                        include={"user": True},
                    )
                    if not prisma_conv:
                        return None

                    current_branch_id = getattr(prisma_conv, "activeBranchId", None)
                    if current_branch_id == branch_id:
                        return self._prisma_conversation_to_model(prisma_conv)

                    target_branch = await tx.conversationbranch.find_unique(
                        where={"id": branch_id}
                    )
                    if (
                        not target_branch
                        or target_branch.conversationId != conversation_id
                    ):
                        return None

                    messages: List[dict[str, Any]] = _normalize_message_payloads(
                        prisma_conv.messages
                    )

                    branch_point = target_branch.branchPointIndex
                    current_downstream = messages[branch_point:]
                    if current_downstream:
                        if current_branch_id:
                            await tx.conversationbranch.update(
                                where={"id": current_branch_id},
                                data={
                                    "preservedMessages": Json(current_downstream),
                                    "updatedAt": datetime.utcnow(),
                                },
                            )
                        else:
                            user_id = (
                                str(prisma_conv.userId) if prisma_conv.userId else None
                            )
                            await tx.conversationbranch.create(
                                data={
                                    "id": str(uuid.uuid4()),
                                    "conversationId": conversation_id,
                                    "parentBranchId": getattr(
                                        target_branch, "parentBranchId", None
                                    ),
                                    "branchPointIndex": branch_point,
                                    "preservedMessages": Json(current_downstream),
                                    "createdByUserId": user_id,
                                }
                            )

                    base_messages = messages[:branch_point]
                    target_preserved: List[dict[str, Any]] = (
                        _normalize_message_payloads(target_branch.preservedMessages)
                    )
                    new_messages = base_messages + target_preserved
                    total_tokens = _estimate_conversation_tokens(new_messages)

                    updated = await tx.conversation.update(
                        where={"id": conversation_id},
                        data={
                            "messages": Json(new_messages),
                            "totalTokens": total_tokens,
                            "activeBranchId": branch_id,
                            "updatedAt": datetime.utcnow(),
                        },
                        include={"user": True},
                    )

                return self._prisma_conversation_to_model(updated)
        except Exception as e:
            logger.warning(f"Failed to switch conversation branch: {e}")
            return None

    async def get_conversation_branches(
        self,
        conversation_id: str,
    ) -> List[ConversationBranchSummary]:
        """List all branches for a conversation, grouped by branch point."""
        db = await self._get_db()
        try:
            branches = await db.conversationbranch.find_many(
                where={"conversationId": conversation_id},
                order=[{"createdAt": "asc"}],
                include={"createdByUser": True},
            )
            return [self._prisma_branch_to_summary(b) for b in branches]
        except Exception as e:
            logger.warning(f"Failed to list conversation branches: {e}")
            return []

    async def release_conversation_branch(
        self,
        conversation_id: str,
    ) -> Optional[Conversation]:
        """Release the active branch: stash current downstream back into it
        and return the conversation to the live (pre-branch) view.

        Used by the UI "Current" option in the branch nav so users can
        toggle between the restored state and the post-delete/live state
        without having to refresh or hunt for the original branch_id.
        """
        db = await self._get_db()
        try:
            async with self._get_conversation_branch_lock(conversation_id):
                async with db.tx() as tx:
                    prisma_conv = await tx.conversation.find_unique(
                        where={"id": conversation_id},
                        include={"user": True},
                    )
                    if not prisma_conv:
                        return None

                    active_branch_id = getattr(prisma_conv, "activeBranchId", None)
                    if not active_branch_id:
                        return self._prisma_conversation_to_model(prisma_conv)

                    active_branch = await tx.conversationbranch.find_unique(
                        where={"id": active_branch_id}
                    )
                    if (
                        not active_branch
                        or active_branch.conversationId != conversation_id
                    ):
                        # Active branch pointer is stale; just clear it.
                        updated = await tx.conversation.update(
                            where={"id": conversation_id},
                            data={
                                "activeBranchId": None,
                                "updatedAt": datetime.utcnow(),
                            },
                            include={"user": True},
                        )
                        return self._prisma_conversation_to_model(updated)

                    messages: List[dict[str, Any]] = _normalize_message_payloads(
                        prisma_conv.messages
                    )
                    branch_point = active_branch.branchPointIndex
                    downstream = messages[branch_point:]

                    if downstream:
                        await tx.conversationbranch.update(
                            where={"id": active_branch_id},
                            data={
                                "preservedMessages": Json(downstream),
                                "updatedAt": datetime.utcnow(),
                            },
                        )

                    truncated = messages[:branch_point]
                    total_tokens = _estimate_conversation_tokens(truncated)
                    updated = await tx.conversation.update(
                        where={"id": conversation_id},
                        data={
                            "messages": Json(truncated),
                            "totalTokens": total_tokens,
                            "activeBranchId": None,
                            "updatedAt": datetime.utcnow(),
                        },
                        include={"user": True},
                    )

                return self._prisma_conversation_to_model(updated)
        except Exception as e:
            logger.warning(f"Failed to release conversation branch: {e}")
            return None

    async def delete_conversation_branch(
        self,
        conversation_id: str,
        branch_id: str,
    ) -> bool:
        """Delete a branch. If it's the active branch, clear active_branch_id."""
        db = await self._get_db()
        try:
            async with self._get_conversation_branch_lock(conversation_id):
                async with db.tx() as tx:
                    branch = await tx.conversationbranch.find_unique(
                        where={"id": branch_id}
                    )
                    if not branch or branch.conversationId != conversation_id:
                        return False

                    conv = await tx.conversation.find_unique(
                        where={"id": conversation_id}
                    )
                    if conv and getattr(conv, "activeBranchId", None) == branch_id:
                        await tx.conversation.update(
                            where={"id": conversation_id},
                            data={
                                "activeBranchId": None,
                                "updatedAt": datetime.utcnow(),
                            },
                        )

                    await tx.conversationbranch.delete(where={"id": branch_id})

                return True
        except Exception as e:
            logger.warning(f"Failed to delete conversation branch: {e}")
            return False

    # -------------------------------------------------------------------------
    # Per-Message Snapshot Link Operations
    # -------------------------------------------------------------------------

    async def upsert_message_snapshot_link(
        self,
        *,
        conversation_id: str,
        workspace_id: str,
        message_id: str,
        snapshot_id: str,
        restore_message_count: int,
    ) -> bool:
        """Upsert a (conversation_id, message_id) → snapshot link.

        Latest write wins: if a prior link exists for the same
        (conversation_id, message_id), it is replaced.
        Returns True on success, False on failure.
        """
        if not message_id or not snapshot_id or restore_message_count < 0:
            return False
        db = await self._get_db()
        try:
            now = datetime.utcnow()
            await db.conversationmessagesnapshotlink.upsert(
                where={
                    "conversationId_messageId": {
                        "conversationId": conversation_id,
                        "messageId": message_id,
                    }
                },
                data={
                    "create": {
                        "id": str(uuid.uuid4()),
                        "conversationId": conversation_id,
                        "workspaceId": workspace_id,
                        "messageId": message_id,
                        "snapshotId": snapshot_id,
                        "restoreMessageCount": restore_message_count,
                    },
                    "update": {
                        "snapshotId": snapshot_id,
                        "restoreMessageCount": restore_message_count,
                        "workspaceId": workspace_id,
                        "updatedAt": now,
                    },
                },
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to upsert message snapshot link: {e}")
            return False

    async def get_message_snapshot_link(
        self, conversation_id: str, message_id: str
    ) -> Optional[dict[str, Any]]:
        """Look up a single link by (conversation_id, message_id)."""
        if not message_id:
            return None
        db = await self._get_db()
        try:
            link = await db.conversationmessagesnapshotlink.find_unique(
                where={
                    "conversationId_messageId": {
                        "conversationId": conversation_id,
                        "messageId": message_id,
                    }
                }
            )
        except Exception as e:
            logger.warning(f"Failed to fetch message snapshot link: {e}")
            return None
        if not link:
            return None
        return {
            "id": link.id,
            "snapshot_id": link.snapshotId,
            "workspace_id": link.workspaceId,
            "restore_message_count": link.restoreMessageCount,
            "created_at": link.createdAt,
            "updated_at": link.updatedAt,
        }

    async def get_message_snapshot_links_for_conversation(
        self, conversation_id: str
    ) -> dict[str, MessageSnapshotRestore]:
        """Return links keyed by message_id for one conversation."""
        db = await self._get_db()
        try:
            links = await db.conversationmessagesnapshotlink.find_many(
                where={"conversationId": conversation_id}
            )
        except Exception as e:
            logger.warning(f"Failed to list message snapshot links: {e}")
            return {}
        out: dict[str, MessageSnapshotRestore] = {}
        for link in links:
            out[link.messageId] = MessageSnapshotRestore(
                snapshot_id=link.snapshotId,
                restore_message_count=link.restoreMessageCount,
                created_at=link.updatedAt or link.createdAt,
            )
        return out

    async def attach_message_snapshot_links(
        self, conversation: Optional[Conversation]
    ) -> Optional[Conversation]:
        """Decorate each message with its snapshot_restore metadata when present."""
        if conversation is None or not conversation.messages:
            return conversation
        try:
            links = await self.get_message_snapshot_links_for_conversation(
                conversation.id
            )
        except Exception:
            return conversation
        if not links:
            return conversation
        for msg in conversation.messages:
            if msg.message_id and msg.message_id in links:
                msg.snapshot_restore = links[msg.message_id]
        return conversation

    async def attach_message_snapshot_links_many(
        self, conversations: list[Conversation]
    ) -> list[Conversation]:
        """Decorate a list of conversations with snapshot link metadata."""
        decorated: list[Conversation] = []
        for conversation in conversations:
            decorated.append(
                await self.attach_message_snapshot_links(conversation) or conversation
            )
        return decorated

    async def link_assistant_snapshot_tool_calls(
        self,
        conversation: Optional[Conversation],
        workspace_id: Optional[str],
    ) -> None:
        """Inspect the most recently persisted assistant message for
        ``create_userspace_snapshot`` tool calls and upsert a per-message
        snapshot link (latest snapshot in the turn wins).

        Best-effort: silently ignores parse/DB errors so chat persistence
        is never blocked.
        """
        if not conversation or not workspace_id or not conversation.messages:
            return
        last_msg = conversation.messages[-1]
        if (
            last_msg.role != "assistant"
            or not last_msg.message_id
            or not last_msg.events
        ):
            return
        snapshot_ids: list[str] = []
        for event in last_msg.events:
            if not isinstance(event, dict):
                continue
            if event.get("type") != "tool_end":
                continue
            if event.get("tool") != "create_userspace_snapshot":
                continue
            output = event.get("output")
            if not isinstance(output, str):
                continue
            try:
                payload = json.loads(output)
            except (TypeError, ValueError):
                continue
            snap = payload.get("snapshot") if isinstance(payload, dict) else None
            if isinstance(snap, dict):
                snap_id = snap.get("id")
                if isinstance(snap_id, str) and snap_id:
                    snapshot_ids.append(snap_id)
        if not snapshot_ids:
            return
        # Anchor on the assistant message; restore excludes that reply
        # ("before assistant reply" semantics).
        keep_count = max(len(conversation.messages) - 1, 0)
        await self.upsert_message_snapshot_link(
            conversation_id=conversation.id,
            workspace_id=workspace_id,
            message_id=last_msg.message_id,
            snapshot_id=snapshot_ids[-1],
            restore_message_count=keep_count,
        )

    def _prisma_branch_to_model(
        self, prisma_branch: Any
    ) -> Optional[ConversationBranch]:
        """Convert Prisma ConversationBranch to Pydantic model."""
        if not prisma_branch:
            return None
        preserved_raw = (
            prisma_branch.preservedMessages if prisma_branch.preservedMessages else []
        )
        preserved_messages = self._parse_messages_json(preserved_raw)
        return ConversationBranch(
            id=prisma_branch.id,
            conversation_id=prisma_branch.conversationId,
            parent_branch_id=getattr(prisma_branch, "parentBranchId", None),
            branch_point_index=prisma_branch.branchPointIndex,
            branch_kind=getattr(prisma_branch, "branchKind", None),
            preserved_messages=preserved_messages,
            associated_snapshot_id=getattr(prisma_branch, "associatedSnapshotId", None),
            created_by_user_id=getattr(prisma_branch, "createdByUserId", None),
            created_at=prisma_branch.createdAt,
            updated_at=prisma_branch.updatedAt,
        )

    def _prisma_branch_to_summary(
        self, prisma_branch: Any
    ) -> ConversationBranchSummary:
        """Convert Prisma ConversationBranch to lightweight summary."""
        preserved_raw = (
            prisma_branch.preservedMessages if prisma_branch.preservedMessages else []
        )
        message_count = len(preserved_raw) if isinstance(preserved_raw, list) else 0
        user = getattr(prisma_branch, "createdByUser", None)
        return ConversationBranchSummary(
            id=prisma_branch.id,
            conversation_id=prisma_branch.conversationId,
            parent_branch_id=getattr(prisma_branch, "parentBranchId", None),
            branch_point_index=prisma_branch.branchPointIndex,
            branch_kind=getattr(prisma_branch, "branchKind", None),
            message_count=message_count,
            associated_snapshot_id=getattr(prisma_branch, "associatedSnapshotId", None),
            created_by_user_id=getattr(prisma_branch, "createdByUserId", None),
            created_by_username=getattr(user, "username", None) if user else None,
            created_at=prisma_branch.createdAt,
        )

    def _parse_messages_json(self, messages_data: Any) -> List[ChatMessage]:
        """Parse a JSON messages array into ChatMessage list (reuses _prisma_conversation_to_model logic)."""
        if not messages_data:
            return []
        result: List[ChatMessage] = []
        for m in _normalize_message_payloads(messages_data):

            events = None
            if "events" in m and m["events"]:
                events = _normalize_message_events(m["events"])
            else:
                events = _synthesize_events_from_legacy_message(m)
            tool_calls = _extract_tool_calls_from_events(events)
            if tool_calls is None and "tool_calls" in m and m["tool_calls"]:
                tool_calls = [
                    ToolCallRecord(
                        tool=tc.get("tool", ""),
                        input=tc.get("input"),
                        output=tc.get("output"),
                        connection=tc.get("connection"),
                        presentation=normalize_tool_presentation(
                            str(tc.get("tool", "")),
                            cast(Optional[dict[str, Any]], tc.get("connection")),
                            cast(Optional[dict[str, Any]], tc.get("presentation")),
                        ),
                    )
                    for tc in m["tool_calls"]
                    if isinstance(tc, dict)
                ]
            result.append(
                ChatMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    timestamp=(
                        datetime.fromisoformat(m["timestamp"])
                        if "timestamp" in m
                        else datetime.utcnow()
                    ),
                    message_id=m.get("message_id"),
                    tool_calls=tool_calls,
                    events=events,
                )
            )
        return result

    async def check_conversation_access(
        self,
        conversation_id: str,
        user_id: Optional[str],
        is_admin: bool = False,
        workspace_id: Optional[str] = None,
    ) -> bool:
        """
        Check if user has access to a conversation.

        Args:
            conversation_id: The conversation ID to check
            user_id: The user's ID (None if not authenticated)
            is_admin: Whether the user is an admin (admins can access all)

        Returns:
            True if user has access, False otherwise
        """
        db = await self._get_db()
        conv = await db.conversation.find_unique(
            where={"id": conversation_id}, include={"members": True}
        )

        if not conv:
            return False

        conversation_workspace_id = getattr(conv, "workspaceId", None)
        if workspace_id is not None:
            if conversation_workspace_id != workspace_id:
                return False
        elif conversation_workspace_id is not None:
            return False

        if is_admin:
            return True

        if not user_id:
            return False

        if conversation_workspace_id:
            workspace = await db.workspace.find_unique(
                where={"id": conversation_workspace_id},
                include={"members": True},
            )
            if not workspace:
                return False

            if workspace.ownerUserId == user_id:
                return True

            return any(
                getattr(member, "userId", None) == user_id
                for member in list(getattr(workspace, "members", []) or [])
            )

        # Allow access if conversation has no owner (legacy) or matches user
        if conv.userId is None or conv.userId == user_id:
            return True

        return any(
            getattr(member, "userId", None) == user_id
            for member in list(getattr(conv, "members", []) or [])
        )

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        db = await self._get_db()

        try:
            await db.conversation.delete(where={"id": conversation_id})
            logger.info(f"Deleted conversation {conversation_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete conversation {conversation_id}: {e}")
            return False

    async def delete_workspace_conversations(self, workspace_id: str) -> int:
        """Delete all conversations linked to a workspace and return delete count."""
        db = await self._get_db()

        try:
            deleted = await db.conversation.delete_many(
                where={"workspaceId": workspace_id}
            )
            count = getattr(deleted, "count", 0)
            logger.info(
                "Deleted %s conversations for workspace %s", count, workspace_id
            )
            return int(count)
        except Exception as e:
            logger.warning(
                "Failed to delete conversations for workspace %s: %s",
                workspace_id,
                e,
            )
            return 0

    def _prisma_conversation_to_model(self, prisma_conv: Any) -> Conversation:
        """Convert Prisma Conversation to Pydantic model."""
        # Parse messages from JSON
        messages_data = _normalize_message_payloads(prisma_conv.messages)
        messages: List[ChatMessage] = []
        for m in messages_data:

            # Parse events if present; synthesize them for legacy tool_calls-only messages.
            events = None
            if "events" in m and m["events"]:
                events = _normalize_message_events(m["events"])
            else:
                events = _synthesize_events_from_legacy_message(m)

            # Legacy persisted tool_calls remain readable; new tool_calls are derived from events.
            tool_calls = _extract_tool_calls_from_events(events)
            if tool_calls is None and "tool_calls" in m and m["tool_calls"]:
                tool_calls = [
                    ToolCallRecord(
                        tool=tc.get("tool", ""),
                        input=tc.get("input"),
                        output=tc.get("output"),
                        connection=tc.get("connection"),
                        presentation=normalize_tool_presentation(
                            str(tc.get("tool", "")),
                            cast(Optional[dict[str, Any]], tc.get("connection")),
                            cast(Optional[dict[str, Any]], tc.get("presentation")),
                        ),
                    )
                    for tc in m["tool_calls"]
                    if isinstance(tc, dict)
                ]

            messages.append(
                ChatMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    timestamp=(
                        datetime.fromisoformat(m["timestamp"])
                        if "timestamp" in m
                        else datetime.utcnow()
                    ),
                    message_id=m.get("message_id"),
                    tool_calls=tool_calls,
                    events=events,
                )
            )

        user = getattr(prisma_conv, "user", None)

        # Parse tool_output_mode
        tool_output_mode_raw = getattr(prisma_conv, "toolOutputMode", "default")
        try:
            tool_output_mode = ToolOutputMode(tool_output_mode_raw)
        except ValueError:
            tool_output_mode = ToolOutputMode.DEFAULT

        return Conversation(
            id=prisma_conv.id,
            title=prisma_conv.title,
            model=prisma_conv.model,
            user_id=getattr(prisma_conv, "userId", None),
            workspace_id=getattr(prisma_conv, "workspaceId", None),
            username=getattr(user, "username", None) if user else None,
            display_name=getattr(user, "displayName", None) if user else None,
            messages=messages,
            total_tokens=prisma_conv.totalTokens,
            created_at=prisma_conv.createdAt,
            updated_at=prisma_conv.updatedAt,
            active_task_id=getattr(prisma_conv, "activeTaskId", None),
            active_branch_id=getattr(prisma_conv, "activeBranchId", None),
            tool_output_mode=tool_output_mode,
        )

    # -------------------------------------------------------------------------
    # Background Chat Task Operations
    # -------------------------------------------------------------------------

    async def create_chat_task(
        self, conversation_id: str, user_message: str
    ) -> ChatTask:
        """Create a new background chat task."""
        db = await self._get_db()

        # Sanitize user message for PostgreSQL storage (remove null bytes)
        user_message = _sanitize_for_postgres(user_message)

        prisma_task = await db.chattask.create(
            data={  # type: ignore[arg-type]
                "id": str(uuid.uuid4()),
                "conversation": {"connect": {"id": conversation_id}},
                "status": _to_prisma_task_status(ChatTaskStatus.pending),
                "userMessage": user_message,
            }
        )

        # Update conversation to track active task
        await db.conversation.update(
            where={"id": conversation_id}, data={"activeTaskId": prisma_task.id}
        )

        return self._prisma_task_to_model(prisma_task)

    async def get_chat_task(self, task_id: str) -> Optional[ChatTask]:
        """Get a chat task by ID."""
        db = await self._get_db()

        prisma_task = await db.chattask.find_unique(where={"id": task_id})

        if prisma_task is None:
            return None

        return self._prisma_task_to_model(prisma_task)

    async def get_active_task_for_conversation(
        self, conversation_id: str
    ) -> Optional[ChatTask]:
        """Get the active (pending/running) task for a conversation."""
        db = await self._get_db()

        prisma_task = await db.chattask.find_first(
            where=cast(
                Any,
                {
                    "conversationId": conversation_id,
                    "status": {
                        "in": [
                            PrismaChatTaskStatus.pending,
                            PrismaChatTaskStatus.running,
                        ]
                    },
                },
            ),
            order={"createdAt": "desc"},
        )  # type: ignore[arg-type]

        if prisma_task is None:
            return None

        return self._prisma_task_to_model(prisma_task)

    async def get_last_interrupted_task_for_conversation(
        self, conversation_id: str
    ) -> Optional[ChatTask]:
        """Get an interrupted task only when it is the latest task for a conversation."""
        db = await self._get_db()

        rows = await db.query_raw(f"""
            SELECT id, status
            FROM chat_tasks
            WHERE conversation_id = {_sql_quote_literal(conversation_id)}
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """)
        if not rows:
            return None

        latest_row = rows[0]
        if str(latest_row.get("status") or "") != ChatTaskStatus.interrupted.value:
            return None

        latest_task_id = str(latest_row.get("id") or "").strip()
        if not latest_task_id:
            return None

        return await self.get_chat_task(latest_task_id)

    async def get_interrupted_conversation_ids_for_workspace(
        self, workspace_id: str
    ) -> list[str]:
        """Return conversation IDs whose latest task is interrupted in a workspace."""
        db = await self._get_db()

        rows = await db.query_raw(f"""
            SELECT latest.conversation_id
            FROM (
                SELECT DISTINCT ON (ct.conversation_id)
                    ct.conversation_id,
                    ct.status
                FROM chat_tasks ct
                INNER JOIN conversations c ON c.id = ct.conversation_id
                WHERE c.workspace_id = {_sql_quote_literal(workspace_id)}
                ORDER BY ct.conversation_id, ct.created_at DESC, ct.id DESC
            ) latest
            WHERE latest.status = {_sql_quote_literal(ChatTaskStatus.interrupted.value)}
            """)

        return [
            str(row.get("conversation_id") or "").strip()
            for row in rows
            if str(row.get("conversation_id") or "").strip()
        ]

    async def get_workspace_task_state_summary(
        self, workspace_ids: list[str]
    ) -> tuple[set[str], set[str]]:
        """Return workspace IDs with live tasks and interrupted tasks."""
        deduped_workspace_ids = [
            wid.strip() for wid in dict.fromkeys(workspace_ids) if wid and wid.strip()
        ]
        if not deduped_workspace_ids:
            return set(), set()

        db = await self._get_db()

        live_rows = await db.conversation.find_many(
            where=cast(
                Any,
                {
                    "workspaceId": {"in": deduped_workspace_ids},
                    "activeTaskId": {"not": None},
                },
            ),
            distinct=["workspaceId"],
        )  # type: ignore[arg-type]

        workspace_id_clause = ", ".join(
            _sql_quote_literal(workspace_id) for workspace_id in deduped_workspace_ids
        )
        interrupted_rows = await db.query_raw(f"""
            SELECT DISTINCT latest.workspace_id
            FROM (
                SELECT DISTINCT ON (ct.conversation_id)
                    ct.conversation_id,
                    c.workspace_id,
                    ct.status
                FROM chat_tasks ct
                INNER JOIN conversations c ON c.id = ct.conversation_id
                WHERE c.workspace_id IN ({workspace_id_clause})
                ORDER BY ct.conversation_id, ct.created_at DESC, ct.id DESC
            ) latest
            WHERE latest.status = {_sql_quote_literal(ChatTaskStatus.interrupted.value)}
            """)

        live_workspace_ids = {
            str(getattr(row, "workspaceId", "") or "")
            for row in live_rows
            if getattr(row, "workspaceId", None)
        }
        interrupted_workspace_ids = {
            str((row.get("workspace_id") if isinstance(row, dict) else None) or "")
            for row in interrupted_rows
            if (row.get("workspace_id") if isinstance(row, dict) else None)
        }

        return live_workspace_ids, interrupted_workspace_ids

    async def update_chat_task_status(
        self, task_id: str, status: ChatTaskStatus, error_message: Optional[str] = None
    ) -> Optional[ChatTask]:
        """Update a chat task's status."""
        db = await self._get_db()

        update_data: dict[str, Any] = {
            "status": _to_prisma_task_status(status),
            "lastUpdateAt": datetime.utcnow(),
        }

        if status == ChatTaskStatus.running:
            update_data["startedAt"] = datetime.utcnow()
        elif status in (
            ChatTaskStatus.completed,
            ChatTaskStatus.failed,
            ChatTaskStatus.cancelled,
        ):
            update_data["completedAt"] = datetime.utcnow()

        if error_message:
            update_data["errorMessage"] = _sanitize_for_postgres(error_message)

        try:
            prisma_task = await db.chattask.update(
                where={"id": task_id}, data=update_data  # type: ignore[arg-type]
            )

            # Clear active task from conversation for terminal states
            if (
                status
                in (
                    ChatTaskStatus.completed,
                    ChatTaskStatus.failed,
                    ChatTaskStatus.cancelled,
                    ChatTaskStatus.interrupted,
                )
                and prisma_task
                and prisma_task.conversationId
            ):
                await db.conversation.update(
                    where={"id": prisma_task.conversationId},
                    data={"activeTaskId": None},
                )

            return self._prisma_task_to_model(prisma_task)
        except Exception as e:
            logger.warning(f"Failed to update chat task status: {e}")
            return None

    async def update_chat_task_streaming_state(
        self,
        task_id: str,
        content: str,
        events: List[dict],
        tool_calls: List[dict],
        hit_max_iterations: bool = False,
        current_version: int = 0,
    ) -> Optional[ChatTask]:
        """Update a chat task's streaming state with version tracking."""
        db = await self._get_db()

        # Sanitize content for PostgreSQL storage (remove null bytes)
        content = _sanitize_for_postgres(content)
        sanitized_events = cast(List[dict], _sanitize_json_for_postgres(events))
        sanitized_tool_calls = cast(List[dict], _sanitize_json_for_postgres(tool_calls))

        # Increment version for change detection
        new_version = current_version + 1

        streaming_state = {
            "content": content,
            "events": sanitized_events,
            "tool_calls": sanitized_tool_calls,
            "hit_max_iterations": hit_max_iterations,
            "version": new_version,
            "content_length": len(content),
        }

        try:
            prisma_task = await db.chattask.update(
                where={"id": task_id},
                data={
                    "streamingState": Json(streaming_state),
                    "lastUpdateAt": datetime.utcnow(),
                },
            )
            return self._prisma_task_to_model(prisma_task)
        except Exception as e:
            logger.warning(f"Failed to update chat task streaming state: {e}")
            return None

    async def complete_chat_task(
        self,
        task_id: str,
        response_content: str,
        final_events: List[dict],
        tool_calls: List[dict],
        hit_max_iterations: bool = False,
        current_version: int = 0,
    ) -> Optional[ChatTask]:
        """Mark a chat task as completed with the final response."""
        db = await self._get_db()

        # Sanitize content for PostgreSQL storage (remove null bytes)
        response_content = _sanitize_for_postgres(response_content)
        sanitized_final_events = cast(
            List[dict], _sanitize_json_for_postgres(final_events)
        )
        sanitized_tool_calls = cast(List[dict], _sanitize_json_for_postgres(tool_calls))

        # Final version increment to signal completion
        final_version = current_version + 1

        streaming_state = {
            "content": response_content,
            "events": sanitized_final_events,
            "tool_calls": sanitized_tool_calls,
            "hit_max_iterations": hit_max_iterations,
            "version": final_version,
            "content_length": len(response_content),
        }

        try:
            prisma_task = await db.chattask.update(
                where={"id": task_id},
                data={  # type: ignore[arg-type]
                    "status": _to_prisma_task_status(ChatTaskStatus.completed),
                    "responseContent": response_content,
                    "streamingState": Json(streaming_state),
                    "completedAt": datetime.utcnow(),
                    "lastUpdateAt": datetime.utcnow(),
                },
            )

            # Clear active task from conversation
            if prisma_task and prisma_task.conversationId:
                await db.conversation.update(
                    where={"id": prisma_task.conversationId},
                    data={"activeTaskId": None},
                )

            return self._prisma_task_to_model(prisma_task) if prisma_task else None
        except Exception as e:
            logger.warning(f"Failed to complete chat task: {e}")
            return None

    async def cancel_chat_task(self, task_id: str) -> Optional[ChatTask]:
        """Cancel a chat task."""
        db = await self._get_db()

        try:
            prisma_task = await db.chattask.update(
                where={"id": task_id},
                data={  # type: ignore[arg-type]
                    "status": _to_prisma_task_status(ChatTaskStatus.cancelled),
                    "completedAt": datetime.utcnow(),
                    "lastUpdateAt": datetime.utcnow(),
                },
            )

            # Clear active task from conversation
            if prisma_task and prisma_task.conversationId:
                await db.conversation.update(
                    where={"id": prisma_task.conversationId},
                    data={"activeTaskId": None},
                )

            return self._prisma_task_to_model(prisma_task) if prisma_task else None
        except Exception as e:
            logger.warning(f"Failed to cancel chat task: {e}")
            return None

    async def cleanup_stale_tasks(self, max_age_seconds: int = 3600) -> int:
        """Clean up stale tasks that have been running for too long."""
        db = await self._get_db()
        cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)

        try:
            # Find stale running tasks
            stale_tasks = await db.chattask.find_many(
                where=cast(
                    Any,
                    {
                        "status": {
                            "in": [
                                PrismaChatTaskStatus.pending,
                                PrismaChatTaskStatus.running,
                            ]
                        },
                        "lastUpdateAt": {"lt": cutoff},
                    },
                )
            )  # type: ignore[arg-type]

            count = 0
            for task in stale_tasks:
                await db.chattask.update(
                    where={"id": task.id},
                    data={  # type: ignore[arg-type]
                        "status": _to_prisma_task_status(ChatTaskStatus.failed),
                        "errorMessage": "Task timed out",
                        "completedAt": datetime.utcnow(),
                    },
                )
                # Clear from conversation
                if task.conversationId:
                    await db.conversation.update(
                        where={"id": task.conversationId}, data={"activeTaskId": None}
                    )
                count += 1

            if count > 0:
                logger.info(f"Cleaned up {count} stale chat tasks")
            return count
        except Exception as e:
            logger.warning(f"Failed to cleanup stale tasks: {e}")
            return 0

    async def create_provider_prompt_debug_record(
        self,
        *,
        conversation_id: str,
        provider: str,
        model: str,
        mode: str,
        request_kind: str,
        rendered_system_prompt: str,
        rendered_user_input: str,
        rendered_provider_messages: List[dict[str, Any]],
        rendered_chat_history: List[dict[str, Any]],
        tool_scope_prompt: str = "",
        prompt_additions: str = "",
        turn_reminders: str = "",
        debug_metadata: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
        chat_task_id: Optional[str] = None,
        message_index: Optional[int] = None,
    ) -> Optional[ProviderPromptDebugRecord]:
        """Persist a DEBUG-only provider prompt record for a single API call."""
        db = await self._get_db()

        try:
            sanitized_provider_messages = cast(
                List[dict[str, Any]],
                _sanitize_json_for_postgres(rendered_provider_messages),
            )
            sanitized_chat_history = cast(
                List[dict[str, Any]],
                _sanitize_json_for_postgres(rendered_chat_history),
            )
            sanitized_debug_metadata = cast(
                Optional[dict[str, Any]],
                (
                    _sanitize_json_for_postgres(debug_metadata)
                    if debug_metadata is not None
                    else None
                ),
            )

            create_data: dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "conversationId": conversation_id,
                "provider": provider,
                "model": model,
                "mode": mode,
                "requestKind": request_kind,
                "renderedSystemPrompt": _sanitize_for_postgres(rendered_system_prompt),
                "renderedUserInput": _sanitize_for_postgres(rendered_user_input),
                "renderedProviderMessages": Json(sanitized_provider_messages),
                "renderedChatHistory": Json(sanitized_chat_history),
                "toolScopePrompt": _sanitize_for_postgres(tool_scope_prompt),
                "promptAdditions": _sanitize_for_postgres(prompt_additions),
                "turnReminders": _sanitize_for_postgres(turn_reminders),
                "debugMetadata": (
                    Json(sanitized_debug_metadata)
                    if sanitized_debug_metadata is not None
                    else None
                ),
                "messageIndex": message_index,
            }
            if chat_task_id:
                create_data["chatTaskId"] = chat_task_id
            if user_id:
                create_data["userId"] = user_id

            prisma_record = await db.providerpromptdebugrecord.create(
                data=cast(Any, create_data)
            )
            return self._prisma_provider_prompt_debug_to_model(prisma_record)
        except Exception as e:
            logger.warning("Failed to persist provider prompt debug record: %s", e)
            return None

    async def list_provider_prompt_debug_records_for_conversation(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
        before: Optional[datetime] = None,
        message_index: Optional[int] = None,
    ) -> List[ProviderPromptDebugRecord]:
        """List provider prompt debug records for a conversation, newest first."""
        db = await self._get_db()

        where: dict[str, Any] = {"conversationId": conversation_id}
        if before is not None:
            where["createdAt"] = {"lt": before}
        if message_index is not None:
            where["messageIndex"] = message_index

        prisma_records = await db.providerpromptdebugrecord.find_many(
            where=cast(Any, where),
            order={"createdAt": "desc"},
            take=max(1, min(limit, 200)),
        )
        return [
            self._prisma_provider_prompt_debug_to_model(record)
            for record in prisma_records
        ]

    def _prisma_task_to_model(self, prisma_task: Any) -> ChatTask:
        """Convert Prisma ChatTask to Pydantic model."""
        streaming_state = None
        if prisma_task.streamingState:
            state_data = prisma_task.streamingState
            streaming_state = ChatTaskStreamingState(
                content=state_data.get("content", ""),
                events=state_data.get("events", []),
                tool_calls=state_data.get("tool_calls", []),
                hit_max_iterations=state_data.get("hit_max_iterations", False),
                version=state_data.get("version", 0),
                content_length=state_data.get("content_length", 0),
            )

        return ChatTask(
            id=prisma_task.id,
            conversation_id=prisma_task.conversationId,
            status=ChatTaskStatus(prisma_task.status),
            user_message=prisma_task.userMessage,
            streaming_state=streaming_state,
            response_content=prisma_task.responseContent,
            error_message=prisma_task.errorMessage,
            created_at=prisma_task.createdAt,
            started_at=prisma_task.startedAt,
            completed_at=prisma_task.completedAt,
            last_update_at=prisma_task.lastUpdateAt,
        )

    def _prisma_provider_prompt_debug_to_model(
        self, prisma_record: Any
    ) -> ProviderPromptDebugRecord:
        """Convert Prisma ProviderPromptDebugRecord to Pydantic model."""
        rendered_provider_messages = list(prisma_record.renderedProviderMessages or [])
        rendered_chat_history = list(prisma_record.renderedChatHistory or [])
        return ProviderPromptDebugRecord(
            id=prisma_record.id,
            conversation_id=prisma_record.conversationId,
            chat_task_id=getattr(prisma_record, "chatTaskId", None),
            user_id=getattr(prisma_record, "userId", None),
            provider=prisma_record.provider,
            model=prisma_record.model,
            mode=prisma_record.mode,
            request_kind=prisma_record.requestKind,
            rendered_system_prompt=prisma_record.renderedSystemPrompt,
            rendered_user_input=prisma_record.renderedUserInput,
            rendered_provider_messages=rendered_provider_messages,
            rendered_chat_history=rendered_chat_history,
            tool_scope_prompt=prisma_record.toolScopePrompt,
            prompt_additions=prisma_record.promptAdditions,
            turn_reminders=prisma_record.turnReminders,
            debug_metadata=getattr(prisma_record, "debugMetadata", None),
            message_index=getattr(prisma_record, "messageIndex", None),
            created_at=prisma_record.createdAt,
        )

    async def cleanup_orphaned_embeddings(self) -> dict[str, int]:
        """
        Remove orphaned embeddings from pgvector tables.

        Orphaned embeddings are those whose index_name doesn't match any
        existing tool configuration. This can happen when:
        - Tool configs were deleted before the embedding cleanup fix
        - The server crashed during a tool deletion

        Returns a dict with counts of deleted embeddings per table.
        """
        db = await self._get_db()
        deleted: dict[str, int] = {
            "filesystem_embeddings": 0,
            "filesystem_file_metadata": 0,
            "schema_embeddings": 0,
            "pdm_embeddings": 0,
            "pdm_document_metadata": 0,
        }

        try:
            # Get all tool configs to build valid index name lists
            all_tools = await self.list_tool_configs()

            # Build valid index names for each tool type
            valid_filesystem_indexes: set[str] = set()
            valid_schema_indexes: set[str] = set()
            valid_pdm_indexes: set[str] = set()

            from ragtime.indexer.utils import safe_tool_name

            for tool in all_tools:
                tool_safe_name = safe_tool_name(tool.name)

                if tool.tool_type == ToolType.FILESYSTEM_INDEXER:
                    # Filesystem index name comes from connection_config
                    if tool.connection_config:
                        index_name = tool.connection_config.get("index_name")
                        if index_name:
                            valid_filesystem_indexes.add(index_name)

                elif tool.tool_type in SCHEMA_INDEXER_CAPABLE_TOOL_TYPES:
                    # Schema indexes use format: schema_{tool_name} or schema_{tool_id}
                    if tool_safe_name:
                        valid_schema_indexes.add(f"schema_{tool_safe_name}")
                    valid_schema_indexes.add(f"schema_{tool.id}")

                elif tool.tool_type == ToolType.SOLIDWORKS_PDM:
                    # PDM indexes use format: pdm_{tool_name} or pdm_{tool_id}
                    if tool_safe_name:
                        valid_pdm_indexes.add(f"pdm_{tool_safe_name}")
                    valid_pdm_indexes.add(f"pdm_{tool.id}")

            # Clean up orphaned filesystem embeddings
            if valid_filesystem_indexes:
                valid_list = ", ".join(f"'{n}'" for n in valid_filesystem_indexes)
                result = await db.execute_raw(
                    f"DELETE FROM filesystem_embeddings WHERE index_name NOT IN ({valid_list})"
                )
                deleted["filesystem_embeddings"] = (
                    result if isinstance(result, int) else 0
                )

                result = await db.execute_raw(
                    f"DELETE FROM filesystem_file_metadata WHERE index_name NOT IN ({valid_list})"
                )
                deleted["filesystem_file_metadata"] = (
                    result if isinstance(result, int) else 0
                )
            else:
                # No valid indexes - delete all orphans (only if tables have data)
                result = await db.execute_raw("DELETE FROM filesystem_embeddings")
                deleted["filesystem_embeddings"] = (
                    result if isinstance(result, int) else 0
                )
                result = await db.execute_raw("DELETE FROM filesystem_file_metadata")
                deleted["filesystem_file_metadata"] = (
                    result if isinstance(result, int) else 0
                )

            # Clean up orphaned schema embeddings
            if valid_schema_indexes:
                valid_list = ", ".join(f"'{n}'" for n in valid_schema_indexes)
                result = await db.execute_raw(
                    f"DELETE FROM schema_embeddings WHERE index_name NOT IN ({valid_list})"
                )
                deleted["schema_embeddings"] = result if isinstance(result, int) else 0
            else:
                result = await db.execute_raw("DELETE FROM schema_embeddings")
                deleted["schema_embeddings"] = result if isinstance(result, int) else 0

            # Clean up orphaned PDM embeddings
            if valid_pdm_indexes:
                valid_list = ", ".join(f"'{n}'" for n in valid_pdm_indexes)
                result = await db.execute_raw(
                    f"DELETE FROM pdm_embeddings WHERE index_name NOT IN ({valid_list})"
                )
                deleted["pdm_embeddings"] = result if isinstance(result, int) else 0

                result = await db.execute_raw(
                    f"DELETE FROM pdm_document_metadata WHERE index_name NOT IN ({valid_list})"
                )
                deleted["pdm_document_metadata"] = (
                    result if isinstance(result, int) else 0
                )
            else:
                result = await db.execute_raw("DELETE FROM pdm_embeddings")
                deleted["pdm_embeddings"] = result if isinstance(result, int) else 0
                result = await db.execute_raw("DELETE FROM pdm_document_metadata")
                deleted["pdm_document_metadata"] = (
                    result if isinstance(result, int) else 0
                )

            total = sum(deleted.values())
            if total > 0:
                logger.info(
                    f"Garbage collection: removed {total} orphaned embedding(s): {deleted}"
                )
            return deleted

        except Exception as e:
            logger.warning(f"Failed to cleanup orphaned embeddings: {e}")
            return deleted


# Global repository instance
repository = IndexerRepository()
