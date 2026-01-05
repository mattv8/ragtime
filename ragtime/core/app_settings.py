"""
Application settings provider - fetches settings from database.

This module provides async access to AppSettings stored in the database,
replacing environment-based configuration for tool settings.
"""

from typing import List, Optional

from ragtime.core.database import get_db
from ragtime.core.logging import get_logger

logger = get_logger(__name__)


class SettingsCache:
    """Simple in-memory cache for database settings."""

    _instance: Optional["SettingsCache"] = None
    _settings: Optional[dict] = None
    _tool_configs: Optional[List[dict]] = None

    @classmethod
    def get_instance(cls) -> "SettingsCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def invalidate(self) -> None:
        """Clear the cache to force a refresh."""
        self._settings = None
        self._tool_configs = None

    async def get_settings(self) -> dict:
        """Get settings from cache or database."""
        if self._settings is not None:
            return self._settings

        try:
            db = await get_db()
            prisma_settings = await db.appsettings.find_unique(where={"id": "default"})

            if prisma_settings is None:
                # Create default settings
                prisma_settings = await db.appsettings.create(data={"id": "default"})
                logger.info("Created default application settings")

            self._settings = {
                "server_name": prisma_settings.serverName,
                "enabled_tools": prisma_settings.enabledTools,
                "odoo_container": prisma_settings.odooContainer,
                "postgres_container": prisma_settings.postgresContainer,
                "postgres_host": prisma_settings.postgresHost,
                "postgres_port": prisma_settings.postgresPort,
                "postgres_user": prisma_settings.postgresUser,
                "postgres_password": prisma_settings.postgresPassword,
                "postgres_db": prisma_settings.postgresDb,
                "max_query_results": prisma_settings.maxQueryResults,
                "query_timeout": prisma_settings.queryTimeout,
                "enable_write_ops": prisma_settings.enableWriteOps,
                # Search configuration
                "search_results_k": prisma_settings.searchResultsK,
                "aggregate_search": prisma_settings.aggregateSearch,
                # LLM settings
                "llm_provider": prisma_settings.llmProvider,
                "llm_model": prisma_settings.llmModel,
                "openai_api_key": prisma_settings.openaiApiKey,
                "anthropic_api_key": prisma_settings.anthropicApiKey,
                "allowed_chat_models": prisma_settings.allowedChatModels or [],
                "max_iterations": prisma_settings.maxIterations,
                # Embedding settings
                "embedding_provider": prisma_settings.embeddingProvider,
                "embedding_model": prisma_settings.embeddingModel,
                "embedding_dimensions": prisma_settings.embeddingDimensions,
                "ollama_base_url": prisma_settings.ollamaBaseUrl,
            }
            return self._settings

        except Exception as e:
            logger.warning(
                f"Failed to load settings from database: {e}. Using defaults."
            )
            return {
                "server_name": "Ragtime",
                "enabled_tools": [],
                "odoo_container": "odoo-server",
                "postgres_container": "odoo-postgres",
                "postgres_host": "",
                "postgres_port": 5432,
                "postgres_user": "",
                "postgres_password": "",
                "postgres_db": "",
                "max_query_results": 100,
                "query_timeout": 30,
                "enable_write_ops": False,
                # Search configuration
                "search_results_k": 5,
                "aggregate_search": True,
                # LLM settings
                "llm_provider": "openai",
                "llm_model": "gpt-4-turbo",
                "openai_api_key": "",
                "anthropic_api_key": "",
                "allowed_chat_models": [],
                "max_iterations": 15,
                # Embedding settings
                "embedding_provider": "ollama",
                "embedding_model": "nomic-embed-text",
                "ollama_base_url": "http://localhost:11434",
            }

    async def get_tool_configs(self) -> List[dict]:
        """Get enabled tool configurations from database."""
        if self._tool_configs is not None:
            return self._tool_configs

        try:
            db = await get_db()
            prisma_configs = await db.toolconfig.find_many(
                where={"enabled": True}, order={"createdAt": "desc"}
            )

            self._tool_configs = [
                {
                    "id": cfg.id,
                    "name": cfg.name,
                    "tool_type": cfg.toolType,
                    "description": cfg.description,
                    "connection_config": cfg.connectionConfig,
                    "max_results": cfg.maxResults,
                    "timeout": cfg.timeout,
                    "allow_write": cfg.allowWrite,
                }
                for cfg in prisma_configs
            ]
            return self._tool_configs

        except Exception as e:
            logger.warning(f"Failed to load tool configs from database: {e}")
            return []


async def get_app_settings() -> dict:
    """Get application settings from database."""
    cache = SettingsCache.get_instance()
    return await cache.get_settings()


async def get_tool_configs() -> List[dict]:
    """Get enabled tool configurations from database."""
    cache = SettingsCache.get_instance()
    return await cache.get_tool_configs()


def invalidate_settings_cache() -> None:
    """Invalidate the settings cache to force a refresh."""
    cache = SettingsCache.get_instance()
    cache.invalidate()
