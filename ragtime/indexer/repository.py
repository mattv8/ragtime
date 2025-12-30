"""
Database repository for indexer jobs and metadata.

Provides Prisma-backed persistence for IndexJob and IndexMetadata,
replacing the previous JSON file-based storage.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Any, cast

from prisma import Prisma, Json
from prisma.models import IndexJob as PrismaIndexJob, IndexMetadata as PrismaIndexMetadata
from prisma.enums import (
    IndexStatus as PrismaIndexStatus,
    ToolType as PrismaToolType,
    ChatTaskStatus as PrismaChatTaskStatus,
)

from ragtime.core.logging import get_logger
from ragtime.core.database import get_db
from ragtime.indexer.models import (
    IndexJob,
    IndexStatus,
    IndexConfig,
    AppSettings,
    ToolConfig,
    ToolType,
    Conversation,
    ChatMessage,
    ChatTask,
    ChatTaskStatus,
    ChatTaskStreamingState,
    ToolCallRecord,
)

logger = get_logger(__name__)


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
        await db.indexjob.create(
            data={  # type: ignore[arg-type]
                "id": job.id,
                "name": job.name,
                "status": _to_prisma_index_status(job.status),
                "sourceType": job.source_type,
                "sourcePath": job.source_path,
                "gitUrl": job.git_url,
                "gitBranch": job.git_branch,
                "totalFiles": job.total_files,
                "processedFiles": job.processed_files,
                "totalChunks": job.total_chunks,
                "processedChunks": job.processed_chunks,
                "errorMessage": job.error_message,
                "createdAt": job.created_at,
                "startedAt": job.started_at,
                "completedAt": job.completed_at,
                "config": {
                    "create": config_data
                }
            },
            include={"config": True}
        )

        logger.debug(f"Created job {job.id} in database")
        return job

    async def get_job(self, job_id: str) -> Optional[IndexJob]:
        """Get a job by ID."""
        db = await self._get_db()

        prisma_job = await db.indexjob.find_unique(
            where={"id": job_id},
            include={"config": True}
        )

        if prisma_job is None:
            return None

        return self._prisma_job_to_model(prisma_job)

    async def list_jobs(self) -> list[IndexJob]:
        """List all jobs."""
        db = await self._get_db()

        prisma_jobs = await db.indexjob.find_many(
            include={"config": True},
            order={"createdAt": "desc"}
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
            }
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
    ) -> None:
        """Create or update index metadata."""
        db = await self._get_db()

        # Build create/update data - only include configSnapshot if it has a value
        create_data: dict = {
            "name": name,
            "description": description,
            "path": path,
            "documentCount": document_count,
            "chunkCount": chunk_count,
            "sizeBytes": size_bytes,
            "sourceType": source_type,
            "source": source,
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
            "lastModified": datetime.utcnow(),
        }

        # Only include configSnapshot if we have actual data
        if config_snapshot is not None:
            create_data["configSnapshot"] = Json(config_snapshot)
            update_data["configSnapshot"] = Json(config_snapshot)

        await db.indexmetadata.upsert(
            where={"name": name},
            data={  # type: ignore[arg-type]
                "create": create_data,
                "update": update_data,
            }
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
                where={"name": name},
                data={"enabled": enabled}
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
                where={"name": name},
                data={"description": description}
            )
            logger.debug(f"Updated description for index {name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to update description for index {name}: {e}")
            return False

    # -------------------------------------------------------------------------
    # Application Settings Operations
    # -------------------------------------------------------------------------

    async def get_settings(self) -> AppSettings:
        """Get application settings, creating defaults if needed."""
        db = await self._get_db()

        prisma_settings = await db.appsettings.find_unique(where={"id": "default"})

        if prisma_settings is None:
            # Create default settings
            prisma_settings = await db.appsettings.create(
                data={"id": "default"}
            )
            logger.info("Created default application settings")

        return AppSettings(
            id=prisma_settings.id,
            # Embedding settings
            embedding_provider=prisma_settings.embeddingProvider,
            embedding_model=prisma_settings.embeddingModel,
            ollama_protocol=prisma_settings.ollamaProtocol,
            ollama_host=prisma_settings.ollamaHost,
            ollama_port=prisma_settings.ollamaPort,
            ollama_base_url=prisma_settings.ollamaBaseUrl,
            # LLM settings
            llm_provider=prisma_settings.llmProvider,
            llm_model=prisma_settings.llmModel,
            openai_api_key=prisma_settings.openaiApiKey,
            anthropic_api_key=prisma_settings.anthropicApiKey,
            allowed_chat_models=prisma_settings.allowedChatModels or [],
            max_iterations=prisma_settings.maxIterations,
            # Tool settings
            enabled_tools=prisma_settings.enabledTools,
            odoo_container=prisma_settings.odooContainer,
            postgres_container=prisma_settings.postgresContainer,
            postgres_host=prisma_settings.postgresHost,
            postgres_port=prisma_settings.postgresPort,
            postgres_user=prisma_settings.postgresUser,
            postgres_password=prisma_settings.postgresPassword,
            postgres_database=prisma_settings.postgresDb,
            max_query_results=prisma_settings.maxQueryResults,
            query_timeout=prisma_settings.queryTimeout,
            enable_write_ops=prisma_settings.enableWriteOps,
            updated_at=prisma_settings.updatedAt,
        )

    async def update_settings(self, updates: dict) -> AppSettings:
        """Update application settings with provided fields."""
        db = await self._get_db()

        # Map snake_case to camelCase for Prisma
        field_mapping = {
            # Embedding settings
            "embedding_provider": "embeddingProvider",
            "embedding_model": "embeddingModel",
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
        }

        # Build update data with only provided fields
        update_data = {}
        for snake_key, camel_key in field_mapping.items():
            if snake_key in updates and updates[snake_key] is not None:
                update_data[camel_key] = updates[snake_key]

        if not update_data:
            return await self.get_settings()

        # Ensure default settings exist
        await self.get_settings()

        await db.appsettings.update(
            where={"id": "default"},
            data=update_data  # type: ignore[arg-type]
        )

        logger.info(f"Updated settings: {list(update_data.keys())}")
        return await self.get_settings()

    # -------------------------------------------------------------------------
    # Tool Configuration Operations
    # -------------------------------------------------------------------------

    async def create_tool_config(self, config: ToolConfig) -> ToolConfig:
        """Create a new tool configuration."""
        db = await self._get_db()

        prisma_config = await db.toolconfig.create(
            data={  # type: ignore[arg-type]
                "name": config.name,
                "toolType": _to_prisma_tool_type(config.tool_type),
                "enabled": config.enabled,
                "description": config.description,
                "connectionConfig": Json(config.connection_config),
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
            where=where,  # type: ignore[arg-type]
            order={"createdAt": "desc"}
        )

        return [self._prisma_tool_config_to_model(c) for c in prisma_configs]

    async def update_tool_config(self, config_id: str, updates: dict[str, Any]) -> Optional[ToolConfig]:
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
                    value = Json(value)
                update_data[camel_key] = value

        if not update_data:
            return await self.get_tool_config(config_id)

        try:
            await db.toolconfig.update(
                where={"id": config_id},
                data=update_data  # type: ignore[arg-type]
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
        self,
        config_id: str,
        success: bool,
        error: Optional[str] = None
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
                }
            )
        except Exception as e:
            logger.warning(f"Failed to update test result for {config_id}: {e}")

    def _prisma_tool_config_to_model(self, prisma_config: Any) -> ToolConfig:
        """Convert Prisma ToolConfig to Pydantic model."""
        return ToolConfig(
            id=prisma_config.id,
            name=prisma_config.name,
            tool_type=ToolType(prisma_config.toolType),
            enabled=prisma_config.enabled,
            description=prisma_config.description,
            connection_config=prisma_config.connectionConfig,
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
        user_id: Optional[str] = None
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
            include={"user": True}
        )

        return self._prisma_conversation_to_model(prisma_conv)

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        db = await self._get_db()

        prisma_conv = await db.conversation.find_unique(
            where={"id": conversation_id},
            include={"user": True}
        )

        if prisma_conv is None:
            return None

        return self._prisma_conversation_to_model(prisma_conv)

    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        include_all: bool = False
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
            include={"user": True}
        )

        return [self._prisma_conversation_to_model(c) for c in prisma_convs]

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[dict]] = None,
        events: Optional[List[dict]] = None
    ) -> Optional[Conversation]:
        """Add a message to a conversation."""
        db = await self._get_db()

        # Get current conversation
        prisma_conv = await db.conversation.find_unique(
            where={"id": conversation_id}
        )
        if not prisma_conv:
            return None

        # Add new message
        messages: List[dict[str, Any]] = cast(
            List[dict[str, Any]],
            list(prisma_conv.messages) if prisma_conv.messages else []
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

        # Estimate tokens (rough: 1 token ~= 4 chars)
        total_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4

        # Update conversation
        updated = await db.conversation.update(
            where={"id": conversation_id},
            data={
                "messages": Json(messages),
                "totalTokens": total_tokens,
                "updatedAt": datetime.utcnow(),
            },
            include={"user": True}
        )

        return self._prisma_conversation_to_model(updated)

    async def update_conversation_title(
        self,
        conversation_id: str,
        title: str
    ) -> Optional[Conversation]:
        """Update a conversation's title."""
        db = await self._get_db()

        try:
            updated = await db.conversation.update(
                where={"id": conversation_id},
                data={"title": title, "updatedAt": datetime.utcnow()},
                include={"user": True}
            )
            return self._prisma_conversation_to_model(updated)
        except Exception as e:
            logger.warning(f"Failed to update conversation title: {e}")
            return None

    async def update_conversation_model(
        self,
        conversation_id: str,
        model: str
    ) -> Optional[Conversation]:
        """Update a conversation's model."""
        db = await self._get_db()

        try:
            updated = await db.conversation.update(
                where={"id": conversation_id},
                data={"model": model, "updatedAt": datetime.utcnow()},
                include={"user": True}
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
                include={"user": True}
            )
            return self._prisma_conversation_to_model(updated)
        except Exception as e:
            logger.warning(f"Failed to clear conversation: {e}")
            return None

    async def truncate_messages(
        self,
        conversation_id: str,
        keep_count: int
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
                list(prisma_conv.messages) if prisma_conv.messages else []
            )
            truncated = messages[:keep_count]

            # Recalculate tokens
            total_tokens = sum(len(str(m.get("content", ""))) for m in truncated) // 4

            updated = await db.conversation.update(
                where={"id": conversation_id},
                data={
                    "messages": Json(truncated),
                    "totalTokens": total_tokens,
                    "updatedAt": datetime.utcnow(),
                },
                include={"user": True}
            )
            return self._prisma_conversation_to_model(updated)
        except Exception as e:
            logger.warning(f"Failed to truncate conversation: {e}")
            return None

    async def check_conversation_access(
        self,
        conversation_id: str,
        user_id: Optional[str],
        is_admin: bool = False
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
                        output=tc.get("output")
                    )
                    for tc in m["tool_calls"]
                ]

            # Parse events if present (preferred)
            events = None
            if "events" in m and m["events"]:
                events = m["events"]  # Keep as raw dicts for now

            messages.append(ChatMessage(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                timestamp=datetime.fromisoformat(m["timestamp"]) if "timestamp" in m else datetime.utcnow(),
                tool_calls=tool_calls,
                events=events,
            ))

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
            active_task_id=getattr(prisma_conv, 'activeTaskId', None),
        )

    # -------------------------------------------------------------------------
    # Background Chat Task Operations
    # -------------------------------------------------------------------------

    async def create_chat_task(
        self,
        conversation_id: str,
        user_message: str
    ) -> ChatTask:
        """Create a new background chat task."""
        db = await self._get_db()

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
            where={"id": conversation_id},
            data={"activeTaskId": prisma_task.id}
        )

        return self._prisma_task_to_model(prisma_task)

    async def get_chat_task(self, task_id: str) -> Optional[ChatTask]:
        """Get a chat task by ID."""
        db = await self._get_db()

        prisma_task = await db.chattask.find_unique(
            where={"id": task_id}
        )

        if prisma_task is None:
            return None

        return self._prisma_task_to_model(prisma_task)

    async def get_active_task_for_conversation(
        self,
        conversation_id: str
    ) -> Optional[ChatTask]:
        """Get the active (pending/running) task for a conversation."""
        db = await self._get_db()

        prisma_task = await db.chattask.find_first(
            where={  # type: ignore[arg-type]
                "conversationId": conversation_id,
                "status": {"in": ["pending", "running"]}
            },
            order={"createdAt": "desc"}
        )

        if prisma_task is None:
            return None

        return self._prisma_task_to_model(prisma_task)

    async def update_chat_task_status(
        self,
        task_id: str,
        status: ChatTaskStatus,
        error_message: Optional[str] = None
    ) -> Optional[ChatTask]:
        """Update a chat task's status."""
        db = await self._get_db()

        update_data: dict[str, Any] = {
            "status": _to_prisma_task_status(status),
            "lastUpdateAt": datetime.utcnow(),
        }

        if status == ChatTaskStatus.running:
            update_data["startedAt"] = datetime.utcnow()
        elif status in (ChatTaskStatus.completed, ChatTaskStatus.failed, ChatTaskStatus.cancelled):
            update_data["completedAt"] = datetime.utcnow()

        if error_message:
            update_data["errorMessage"] = error_message

        try:
            prisma_task = await db.chattask.update(
                where={"id": task_id},
                data=update_data  # type: ignore[arg-type]
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
        current_version: int = 0
    ) -> Optional[ChatTask]:
        """Update a chat task's streaming state with version tracking."""
        db = await self._get_db()

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
                }
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
        current_version: int = 0
    ) -> Optional[ChatTask]:
        """Mark a chat task as completed with the final response."""
        db = await self._get_db()

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
                }
            )

            # Clear active task from conversation
            if prisma_task and prisma_task.conversationId:
                await db.conversation.update(
                    where={"id": prisma_task.conversationId},
                    data={"activeTaskId": None}
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
                }
            )

            # Clear active task from conversation
            if prisma_task and prisma_task.conversationId:
                await db.conversation.update(
                    where={"id": prisma_task.conversationId},
                    data={"activeTaskId": None}
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
                where={  # type: ignore[arg-type]
                    "status": {"in": ["pending", "running"]},
                    "lastUpdateAt": {"lt": cutoff}
                }
            )

            count = 0
            for task in stale_tasks:
                await db.chattask.update(
                    where={"id": task.id},
                    data={  # type: ignore[arg-type]
                        "status": _to_prisma_task_status(ChatTaskStatus.failed),
                        "errorMessage": "Task timed out",
                        "completedAt": datetime.utcnow(),
                    }
                )
                # Clear from conversation
                if task.conversationId:
                    await db.conversation.update(
                        where={"id": task.conversationId},
                        data={"activeTaskId": None}
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


# Global repository instance
repository = IndexerRepository()
