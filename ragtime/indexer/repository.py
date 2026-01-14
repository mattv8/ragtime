"""
Database repository for indexer jobs and metadata.

Provides Prisma-backed persistence for IndexJob and IndexMetadata,
replacing the previous JSON file-based storage.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, List, Optional, cast

from prisma.enums import ChatTaskStatus as PrismaChatTaskStatus
from prisma.enums import IndexStatus as PrismaIndexStatus
from prisma.enums import ToolType as PrismaToolType
from prisma.models import IndexJob as PrismaIndexJob
from prisma.models import IndexMetadata as PrismaIndexMetadata

from prisma import Json, Prisma
from ragtime.core.database import get_db
from ragtime.core.encryption import (
    CONNECTION_CONFIG_PASSWORD_FIELDS,
    decrypt_json_passwords,
    decrypt_secret,
    encrypt_json_passwords,
    encrypt_secret,
)
from ragtime.core.logging import get_logger
from ragtime.indexer.models import (
    AppSettings,
    ChatMessage,
    ChatTask,
    ChatTaskStatus,
    ChatTaskStreamingState,
    Conversation,
    IndexConfig,
    IndexJob,
    IndexStatus,
    ToolCallRecord,
    ToolConfig,
    ToolType,
)

logger = get_logger(__name__)

CHARS_PER_TOKEN = 4


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


def _safe_serialize(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value) if value is not None else ""


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


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


def _to_prisma_task_status(status: ChatTaskStatus) -> PrismaChatTaskStatus:
    """Convert model ChatTaskStatus to Prisma ChatTaskStatus."""
    return PrismaChatTaskStatus(status.value)


class IndexerRepository:
    """Repository for indexer data persistence via Prisma."""

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
            "gitToken": job.git_token,
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

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID."""
        db = await self._get_db()

        try:
            await db.indexjob.delete(where={"id": job_id})
            logger.debug(f"Deleted job {job_id} from database")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete job {job_id}: {e}")
            return False

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
            git_token=getattr(prisma_job, "gitToken", None),
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
    ) -> None:
        """Create or update index metadata."""
        db = await self._get_db()

        # Build create/update data - only include configSnapshot if it has a value
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
            "gitToken": git_token,
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
            "gitToken": git_token,
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

        Updates the index_metadata name (primary key). The caller is responsible
        for renaming the FAISS index directory on disk.

        Args:
            old_name: Current index name (safe tool name)
            new_name: New index name (safe tool name)
            display_name: Human-readable display name (original user input)

        Returns:
            True if rename succeeded, False otherwise
        """
        import os

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
                "gitToken": getattr(metadata_any, "gitToken", None),
                "createdAt": metadata.createdAt,
                "lastModified": metadata.lastModified,
            }

            # Only include configSnapshot if we have actual data
            if metadata.configSnapshot is not None:
                create_data["configSnapshot"] = Json(metadata.configSnapshot)

            # Use a transaction to ensure atomicity - delete and create together
            async with db.tx() as tx:
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
        mcp_password = getattr(settings, "mcpDefaultRoutePassword", None)

        # Decrypt if encrypted (starts with 'enc::')
        if openai_key:
            openai_key = decrypt_secret(openai_key)
        if anthropic_key:
            anthropic_key = decrypt_secret(anthropic_key)
        if mcp_password:
            mcp_password = decrypt_secret(mcp_password)

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
            # LLM settings
            llm_provider=settings.llmProvider,
            llm_model=settings.llmModel,
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            allowed_chat_models=settings.allowedChatModels or [],
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
            # Search configuration
            search_results_k=getattr(settings, "searchResultsK", 5),
            aggregate_search=getattr(settings, "aggregateSearch", True),
            # Performance configuration
            sequential_index_loading=getattr(settings, "sequentialIndexLoading", False),
            # MCP configuration
            mcp_enabled=getattr(settings, "mcpEnabled", False),
            mcp_default_route_auth=getattr(settings, "mcpDefaultRouteAuth", False),
            mcp_default_route_password=mcp_password,
            has_mcp_default_password=getattr(settings, "mcpDefaultRoutePassword", None)
            is not None,
            # Embedding dimension tracking
            embedding_dimension=getattr(settings, "embeddingDimension", None),
            embedding_config_hash=getattr(settings, "embeddingConfigHash", None),
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
            # LLM settings
            "llm_provider": "llmProvider",
            "llm_model": "llmModel",
            "openai_api_key": "openaiApiKey",
            "anthropic_api_key": "anthropicApiKey",
            "allowed_chat_models": "allowedChatModels",
            "max_iterations": "maxIterations",
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
            # Performance configuration
            "sequential_index_loading": "sequentialIndexLoading",
            # MCP configuration
            "mcp_enabled": "mcpEnabled",
            "mcp_default_route_auth": "mcpDefaultRouteAuth",
            "mcp_default_route_password": "mcpDefaultRoutePassword",
            # Embedding dimension tracking (internal use)
            "embedding_dimension": "embeddingDimension",
            "embedding_config_hash": "embeddingConfigHash",
        }

        # Build update data with only provided fields
        update_data = {}
        for snake_key, camel_key in field_mapping.items():
            if snake_key in updates and updates[snake_key] is not None:
                update_data[camel_key] = updates[snake_key]

        # Encrypt secret fields before storage
        # These need to be reversibly encrypted so we can show them in the UI
        secret_fields = ["openai_api_key", "anthropic_api_key", "postgres_password"]
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

        prisma_config = await db.toolconfig.create(
            data={  # type: ignore[arg-type]
                "name": config.name,
                "toolType": _to_prisma_tool_type(config.tool_type),
                "enabled": config.enabled,
                "description": config.description,
                "connectionConfig": Json(encrypted_config),
                "maxResults": config.max_results,
                "timeout": config.timeout,
                "allowWrite": config.allow_write,
            }
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
            where=where, order={"createdAt": "desc"}  # type: ignore[arg-type]
        )

        return [self._prisma_tool_config_to_model(c) for c in prisma_configs]

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
                    # Encrypt password fields
                    value = encrypt_json_passwords(
                        value, CONNECTION_CONFIG_PASSWORD_FIELDS
                    )
                    value = Json(value)
                update_data[camel_key] = value

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

        return ToolConfig(
            id=prisma_config.id,
            name=prisma_config.name,
            tool_type=ToolType(prisma_config.toolType),
            enabled=prisma_config.enabled,
            description=prisma_config.description,
            connection_config=decrypted_config,
            max_results=prisma_config.maxResults,
            timeout=prisma_config.timeout,
            allow_write=prisma_config.allowWrite,
            last_test_at=prisma_config.lastTestAt,
            last_test_result=prisma_config.lastTestResult,
            last_test_error=prisma_config.lastTestError,
            created_at=prisma_config.createdAt,
            updated_at=prisma_config.updatedAt,
        )

    # -------------------------------------------------------------------------
    # Conversation Operations
    # -------------------------------------------------------------------------

    async def create_conversation(
        self,
        title: str = "New Chat",
        model: str = "gpt-4-turbo",
        user_id: Optional[str] = None,
    ) -> Conversation:
        """Create a new conversation."""
        db = await self._get_db()

        prisma_conv = await db.conversation.create(
            data={
                "id": str(uuid.uuid4()),
                "title": title,
                "model": model,
                "messages": Json([]),
                "totalTokens": 0,
                "userId": user_id,
            },
            include={"user": True},
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

        return self._prisma_conversation_to_model(prisma_conv)

    async def list_conversations(
        self, user_id: Optional[str] = None, include_all: bool = False
    ) -> list[Conversation]:
        """
        List conversations, newest first.

        Args:
            user_id: Filter by user ID (required unless include_all=True)
            include_all: If True, return all conversations (admin only)
        """
        db = await self._get_db()

        where_clause = {}
        if not include_all and user_id:
            where_clause = {"userId": user_id}

        prisma_convs = await db.conversation.find_many(
            where=where_clause if where_clause else None,  # type: ignore[arg-type]
            order={"updatedAt": "desc"},
            include={"user": True},
        )

        return [self._prisma_conversation_to_model(c) for c in prisma_convs]

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[dict]] = None,
        events: Optional[List[dict]] = None,
    ) -> Optional[Conversation]:
        """Add a message to a conversation."""
        db = await self._get_db()

        # Sanitize content for PostgreSQL storage (remove null bytes)
        content = _sanitize_for_postgres(content)

        # Get current conversation
        prisma_conv = await db.conversation.find_unique(where={"id": conversation_id})
        if not prisma_conv:
            return None

        # Add new message
        messages: List[dict[str, Any]] = cast(
            List[dict[str, Any]],
            list(prisma_conv.messages) if prisma_conv.messages else [],
        )
        new_message: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        # Add tool calls if provided (deprecated, for backward compatibility)
        if tool_calls:
            new_message["tool_calls"] = tool_calls
        # Add chronological events if provided (preferred)
        if events:
            new_message["events"] = events
        messages.append(new_message)

        # Estimate tokens (rough: 1 token ~= 4 chars) including tool calls and events
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

    async def clear_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Clear all messages in a conversation."""
        db = await self._get_db()

        try:
            updated = await db.conversation.update(
                where={"id": conversation_id},
                data={
                    "messages": Json([]),
                    "totalTokens": 0,
                    "updatedAt": datetime.utcnow(),
                },
                include={"user": True},
            )
            return self._prisma_conversation_to_model(updated)
        except Exception as e:
            logger.warning(f"Failed to clear conversation: {e}")
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

            messages: List[dict[str, Any]] = cast(
                List[dict[str, Any]],
                list(prisma_conv.messages) if prisma_conv.messages else [],
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

    async def check_conversation_access(
        self, conversation_id: str, user_id: Optional[str], is_admin: bool = False
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
        if is_admin:
            return True

        if not user_id:
            return False

        db = await self._get_db()
        conv = await db.conversation.find_unique(where={"id": conversation_id})

        if not conv:
            return False

        # Allow access if conversation has no owner (legacy) or matches user
        return conv.userId is None or conv.userId == user_id

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

    def _prisma_conversation_to_model(self, prisma_conv: Any) -> Conversation:
        """Convert Prisma Conversation to Pydantic model."""
        # Parse messages from JSON
        messages_data = prisma_conv.messages if prisma_conv.messages else []
        messages: List[ChatMessage] = []
        for m in messages_data:
            # Parse tool_calls if present (deprecated)
            tool_calls = None
            if "tool_calls" in m and m["tool_calls"]:
                tool_calls = [
                    ToolCallRecord(
                        tool=tc.get("tool", ""),
                        input=tc.get("input"),
                        output=tc.get("output"),
                    )
                    for tc in m["tool_calls"]
                ]

            # Parse events if present (preferred)
            events = None
            if "events" in m and m["events"]:
                events = m["events"]  # Keep as raw dicts for now

            messages.append(
                ChatMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    timestamp=(
                        datetime.fromisoformat(m["timestamp"])
                        if "timestamp" in m
                        else datetime.utcnow()
                    ),
                    tool_calls=tool_calls,
                    events=events,
                )
            )

        user = getattr(prisma_conv, "user", None)

        return Conversation(
            id=prisma_conv.id,
            title=prisma_conv.title,
            model=prisma_conv.model,
            user_id=getattr(prisma_conv, "userId", None),
            username=getattr(user, "username", None) if user else None,
            display_name=getattr(user, "displayName", None) if user else None,
            messages=messages,
            total_tokens=prisma_conv.totalTokens,
            created_at=prisma_conv.createdAt,
            updated_at=prisma_conv.updatedAt,
            active_task_id=getattr(prisma_conv, "activeTaskId", None),
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
            update_data["errorMessage"] = error_message

        try:
            prisma_task = await db.chattask.update(
                where={"id": task_id}, data=update_data  # type: ignore[arg-type]
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

        # Increment version for change detection
        new_version = current_version + 1

        streaming_state = {
            "content": content,
            "events": events,
            "tool_calls": tool_calls,
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

        # Final version increment to signal completion
        final_version = current_version + 1

        streaming_state = {
            "content": response_content,
            "events": final_events,
            "tool_calls": tool_calls,
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

                elif tool.tool_type in (ToolType.POSTGRES, ToolType.MSSQL):
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
