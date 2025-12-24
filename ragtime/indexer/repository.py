"""
Database repository for indexer jobs and metadata.

Provides Prisma-backed persistence for IndexJob and IndexMetadata,
replacing the previous JSON file-based storage.
"""

import json
from datetime import datetime
from typing import Optional

from prisma import Prisma, Json
from prisma.models import IndexJob as PrismaIndexJob, IndexMetadata as PrismaIndexMetadata

from ragtime.core.logging import get_logger
from ragtime.core.database import get_db
from ragtime.indexer.models import IndexJob, IndexStatus, IndexConfig, AppSettings

logger = get_logger(__name__)


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
            data={
                "id": job.id,
                "name": job.name,
                "status": job.status.value,
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
            data={
                "status": job.status.value,
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

        await db.indexmetadata.upsert(
            where={"name": name},
            data={
                "create": {
                    "name": name,
                    "description": description,
                    "path": path,
                    "documentCount": document_count,
                    "chunkCount": chunk_count,
                    "sizeBytes": size_bytes,
                    "sourceType": source_type,
                    "source": source,
                    "configSnapshot": Json(config_snapshot) if config_snapshot else None,
                    "createdAt": datetime.utcnow(),
                    "lastModified": datetime.utcnow(),
                },
                "update": {
                    "description": description,
                    "path": path,
                    "documentCount": document_count,
                    "chunkCount": chunk_count,
                    "sizeBytes": size_bytes,
                    "sourceType": source_type,
                    "source": source,
                    "configSnapshot": Json(config_snapshot) if config_snapshot else None,
                    "lastModified": datetime.utcnow(),
                }
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
            data=update_data
        )

        logger.info(f"Updated settings: {list(update_data.keys())}")
        return await self.get_settings()

    # -------------------------------------------------------------------------
    # Tool Configuration Operations
    # -------------------------------------------------------------------------

    async def create_tool_config(self, config: "ToolConfig") -> "ToolConfig":
        """Create a new tool configuration."""
        from ragtime.indexer.models import ToolConfig as ToolConfigModel

        db = await self._get_db()

        prisma_config = await db.toolconfig.create(
            data={
                "name": config.name,
                "toolType": config.tool_type.value,
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

    async def get_tool_config(self, config_id: str) -> Optional["ToolConfig"]:
        """Get a tool configuration by ID."""
        db = await self._get_db()

        prisma_config = await db.toolconfig.find_unique(where={"id": config_id})
        if prisma_config is None:
            return None

        return self._prisma_tool_config_to_model(prisma_config)

    async def list_tool_configs(self, enabled_only: bool = False) -> list["ToolConfig"]:
        """List all tool configurations."""
        db = await self._get_db()

        where = {"enabled": True} if enabled_only else {}
        prisma_configs = await db.toolconfig.find_many(
            where=where,
            order={"createdAt": "desc"}
        )

        return [self._prisma_tool_config_to_model(c) for c in prisma_configs]

    async def update_tool_config(self, config_id: str, updates: dict) -> Optional["ToolConfig"]:
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
                data=update_data
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

    def _prisma_tool_config_to_model(self, prisma_config) -> "ToolConfig":
        """Convert Prisma ToolConfig to Pydantic model."""
        from ragtime.indexer.models import ToolConfig as ToolConfigModel, ToolType

        return ToolConfigModel(
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


# Global repository instance
repository = IndexerRepository()
