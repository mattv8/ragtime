"""
Application settings provider - fetches settings from database.

This module provides async access to AppSettings stored in the database,
replacing environment-based configuration for tool settings.
"""

import os
from typing import List, Optional

from ragtime.core.app_setting_defaults import (
    DEFAULT_AGGREGATE_SEARCH,
    DEFAULT_AUTHENTICATED_WEBGL_BACKGROUND_ENABLED,
    DEFAULT_CHAT_AUTO_COMPACTION_THRESHOLD_PERCENT,
    DEFAULT_CHAT_COMPACTION_THRESHOLD_PERCENT,
    DEFAULT_CHUNKING_MAX_BATCH_SIZE,
    DEFAULT_CHUNKING_MAX_WORKERS,
    DEFAULT_CHUNKING_USE_TOKENS,
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_GITHUB_COPILOT_BASE_URL,
    DEFAULT_HTTP_PROXY_SAFE_TIMEOUT_SECONDS,
    DEFAULT_IMAGE_PAYLOAD_LIMITS,
    DEFAULT_INCLUDE_COPILOT_THIRD_PARTY_MODELS,
    DEFAULT_IVFFLAT_LISTS,
    DEFAULT_LEGACY_ODOO_CONTAINER,
    DEFAULT_LEGACY_POSTGRES_CONTAINER,
    DEFAULT_LEGACY_POSTGRES_PORT,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_QUERY_RESULTS,
    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    DEFAULT_MCP_DEFAULT_ROUTE_AUTH,
    DEFAULT_MCP_DEFAULT_ROUTE_AUTH_METHOD,
    DEFAULT_MCP_ENABLED,
    DEFAULT_OCR_CONCURRENCY_LIMIT,
    DEFAULT_OCR_MODE,
    DEFAULT_OCR_PROVIDER,
    DEFAULT_OLLAMA_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_OPENAPI_MODEL_PREFIX_ENABLED,
    DEFAULT_OPENAPI_SYNC_CHAT_MODELS,
    DEFAULT_QUERY_TIMEOUT_SECONDS,
    DEFAULT_SCRATCHPAD_WINDOW_SIZE,
    DEFAULT_SEARCH_MMR_LAMBDA,
    DEFAULT_SEARCH_RESULTS_K,
    DEFAULT_SEARCH_USE_MMR,
    DEFAULT_SEQUENTIAL_INDEX_LOADING,
    DEFAULT_SERVER_NAME,
    DEFAULT_SNAPSHOT_RETENTION_DAYS,
    DEFAULT_SNAPSHOT_STALE_BRANCH_THRESHOLD,
    DEFAULT_TOOL_OUTPUT_MODE,
    DEFAULT_USERSPACE_CODE_INDEX_DEBOUNCE_SECONDS,
    DEFAULT_USERSPACE_CODE_INDEX_ENABLED,
    DEFAULT_USERSPACE_CODE_INDEX_MAX_ATTEMPTS,
    DEFAULT_USERSPACE_CODE_INDEX_RECONCILE_INTERVAL_SECONDS,
    DEFAULT_USERSPACE_DUPLICATE_COPY_CHATS,
    DEFAULT_USERSPACE_DUPLICATE_COPY_FILES,
    DEFAULT_USERSPACE_DUPLICATE_COPY_METADATA,
    DEFAULT_USERSPACE_DUPLICATE_COPY_MOUNTS,
    DEFAULT_USERSPACE_MOUNT_SYNC_INTERVAL_SECONDS,
)
from ragtime.core.claude_code import CLAUDE_CODE_OAUTH_TOKEN_ENV
from ragtime.core.database import get_db
from ragtime.core.encryption import (
    CONNECTION_CONFIG_PASSWORD_FIELDS,
    decrypt_json_passwords,
    decrypt_secret,
)
from ragtime.core.logging import get_logger
from ragtime.core.model_providers import (
    LLAMA_CPP_EMBEDDING_CONNECTION,
    LLAMA_CPP_LLM_CONNECTION,
    LMSTUDIO_CONNECTION,
    LMSTUDIO_LLM_CONNECTION,
    OLLAMA_EMBEDDING_CONNECTION,
    OLLAMA_LLM_CONNECTION,
    OMLX_CONNECTION,
    OMLX_LLM_CONNECTION,
    ProviderConnection,
)
from ragtime.core.openai_codex_auth import OPENAI_CODEX_DEFAULT_BASE_URL
from ragtime.core.userspace_limits import (
    ARCHIVE_MAX_FILE_COUNT_DEFAULT,
    ARCHIVE_MAX_TOTAL_SIZE_DEFAULT_BYTES,
)
from ragtime.core.userspace_preview_sandbox import (
    USERSPACE_PREVIEW_SANDBOX_DEFAULT_FLAGS,
    normalize_userspace_preview_sandbox_flags,
)

logger = get_logger(__name__)


def _normalize_default_ocr_mode(value: str | None) -> str:
    mode = str(value or "").strip().lower()
    if mode == "ollama":
        return "vision"
    return mode or "disabled"


PROVIDER_CONNECTION_PRISMA_FIELDS: tuple[tuple[ProviderConnection, dict[str, str]], ...] = (
    (
        OLLAMA_EMBEDDING_CONNECTION,
        {
            "protocol": "ollamaProtocol",
            "host": "ollamaHost",
            "port": "ollamaPort",
            "base_url": "ollamaBaseUrl",
        },
    ),
    (
        LLAMA_CPP_EMBEDDING_CONNECTION,
        {
            "protocol": "llamaCppProtocol",
            "host": "llamaCppHost",
            "port": "llamaCppPort",
            "base_url": "llamaCppBaseUrl",
        },
    ),
    (
        LMSTUDIO_CONNECTION,
        {
            "protocol": "lmstudioProtocol",
            "host": "lmstudioHost",
            "port": "lmstudioPort",
            "base_url": "lmstudioBaseUrl",
        },
    ),
    (
        OMLX_CONNECTION,
        {
            "protocol": "omlxProtocol",
            "host": "omlxHost",
            "port": "omlxPort",
            "base_url": "omlxBaseUrl",
        },
    ),
    (
        OLLAMA_LLM_CONNECTION,
        {
            "protocol": "llmOllamaProtocol",
            "host": "llmOllamaHost",
            "port": "llmOllamaPort",
            "base_url": "llmOllamaBaseUrl",
        },
    ),
    (
        LLAMA_CPP_LLM_CONNECTION,
        {
            "protocol": "llmLlamaCppProtocol",
            "host": "llmLlamaCppHost",
            "port": "llmLlamaCppPort",
            "base_url": "llmLlamaCppBaseUrl",
        },
    ),
    (
        LMSTUDIO_LLM_CONNECTION,
        {
            "protocol": "llmLmstudioProtocol",
            "host": "llmLmstudioHost",
            "port": "llmLmstudioPort",
            "base_url": "llmLmstudioBaseUrl",
        },
    ),
    (
        OMLX_LLM_CONNECTION,
        {
            "protocol": "llmOmlxProtocol",
            "host": "llmOmlxHost",
            "port": "llmOmlxPort",
            "base_url": "llmOmlxBaseUrl",
        },
    ),
)


def _provider_connection_settings(prisma_settings) -> dict:
    values = {}
    for connection, prisma_fields in PROVIDER_CONNECTION_PRISMA_FIELDS:
        values.update(
            {
                connection.protocol_field: getattr(
                    prisma_settings,
                    prisma_fields["protocol"],
                    connection.default_protocol,
                ),
                connection.host_field: getattr(
                    prisma_settings,
                    prisma_fields["host"],
                    connection.default_host,
                ),
                connection.port_field: getattr(
                    prisma_settings,
                    prisma_fields["port"],
                    connection.default_port,
                ),
                connection.base_url_field: getattr(
                    prisma_settings,
                    prisma_fields["base_url"],
                    connection.default_base_url,
                ),
            }
        )
    return values


def _provider_connection_defaults() -> dict:
    values = {}
    for connection, _prisma_fields in PROVIDER_CONNECTION_PRISMA_FIELDS:
        values.update(
            {
                connection.protocol_field: connection.default_protocol,
                connection.host_field: connection.default_host,
                connection.port_field: connection.default_port,
                connection.base_url_field: connection.default_base_url,
            }
        )
    return values


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

            # Decrypt secrets that may be encrypted
            openai_key = prisma_settings.openaiApiKey or ""
            anthropic_key = prisma_settings.anthropicApiKey or ""
            openrouter_key = getattr(prisma_settings, "openrouterApiKey", "") or ""
            github_models_api_token = getattr(prisma_settings, "githubModelsApiToken", "") or ""
            github_copilot_access_token = getattr(prisma_settings, "githubCopilotAccessToken", "") or ""
            github_copilot_refresh_token = getattr(prisma_settings, "githubCopilotRefreshToken", "") or ""
            github_copilot_oauth_refresh_token = getattr(prisma_settings, "githubCopilotOauthRefreshToken", "") or ""
            openai_codex_access_token = getattr(prisma_settings, "openaiCodexAccessToken", "") or ""
            openai_codex_refresh_token = getattr(prisma_settings, "openaiCodexRefreshToken", "") or ""
            postgres_password = prisma_settings.postgresPassword or ""
            mcp_password = prisma_settings.mcpDefaultRoutePassword
            lmstudio_api_key = getattr(prisma_settings, "lmstudioApiKey", None) or ""
            omlx_api_key = getattr(prisma_settings, "omlxApiKey", None) or ""

            openai_key = decrypt_secret(openai_key)
            anthropic_key = decrypt_secret(anthropic_key)
            openrouter_key = decrypt_secret(openrouter_key)
            github_models_api_token = decrypt_secret(github_models_api_token)
            github_copilot_access_token = decrypt_secret(github_copilot_access_token)
            github_copilot_refresh_token = decrypt_secret(github_copilot_refresh_token)
            github_copilot_oauth_refresh_token = decrypt_secret(github_copilot_oauth_refresh_token)
            openai_codex_access_token = decrypt_secret(openai_codex_access_token)
            openai_codex_refresh_token = decrypt_secret(openai_codex_refresh_token)
            postgres_password = decrypt_secret(postgres_password)
            lmstudio_api_key = decrypt_secret(lmstudio_api_key)
            omlx_api_key = decrypt_secret(omlx_api_key)
            # Note: mcp_default_route_password stays encrypted for auth verification
            # It's decrypted in the verification function

            try:
                userspace_preview_sandbox_flags = normalize_userspace_preview_sandbox_flags(getattr(prisma_settings, "userspacePreviewSandboxFlags", None))
            except ValueError as exc:
                logger.warning(
                    "Invalid userspace preview sandbox flags in app settings cache; falling back to defaults: %s",
                    exc,
                )
                userspace_preview_sandbox_flags = list(USERSPACE_PREVIEW_SANDBOX_DEFAULT_FLAGS)

            self._settings = {
                "server_name": prisma_settings.serverName,
                "default_theme_pack": getattr(prisma_settings, "defaultThemePack", "default"),
                "authenticated_webgl_background_enabled": getattr(
                    prisma_settings,
                    "authenticatedWebglBackgroundEnabled",
                    DEFAULT_AUTHENTICATED_WEBGL_BACKGROUND_ENABLED,
                ),
                "openapi_model_prefix_enabled": getattr(
                    prisma_settings,
                    "openapiModelPrefixEnabled",
                    DEFAULT_OPENAPI_MODEL_PREFIX_ENABLED,
                ),
                "enabled_tools": prisma_settings.enabledTools,
                "odoo_container": prisma_settings.odooContainer,
                "postgres_container": prisma_settings.postgresContainer,
                "postgres_host": prisma_settings.postgresHost,
                "postgres_port": prisma_settings.postgresPort,
                "postgres_user": prisma_settings.postgresUser,
                "postgres_password": postgres_password,
                "postgres_db": prisma_settings.postgresDb,
                "max_query_results": getattr(prisma_settings, "maxQueryResults", DEFAULT_MAX_QUERY_RESULTS),
                "query_timeout": getattr(prisma_settings, "queryTimeout", DEFAULT_QUERY_TIMEOUT_SECONDS),
                "http_proxy_safe_timeout_seconds": getattr(
                    prisma_settings,
                    "httpProxySafeTimeoutSeconds",
                    DEFAULT_HTTP_PROXY_SAFE_TIMEOUT_SECONDS,
                ),
                "enable_write_ops": prisma_settings.enableWriteOps,
                # Token optimization settings
                "max_tool_output_chars": getattr(
                    prisma_settings,
                    "maxToolOutputChars",
                    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
                ),
                "scratchpad_window_size": getattr(
                    prisma_settings,
                    "scratchpadWindowSize",
                    DEFAULT_SCRATCHPAD_WINDOW_SIZE,
                ),
                # Search configuration
                "search_results_k": getattr(prisma_settings, "searchResultsK", DEFAULT_SEARCH_RESULTS_K),
                "aggregate_search": getattr(
                    prisma_settings,
                    "aggregateSearch",
                    DEFAULT_AGGREGATE_SEARCH,
                ),
                "search_use_mmr": getattr(
                    prisma_settings,
                    "searchUseMmr",
                    DEFAULT_SEARCH_USE_MMR,
                ),
                "search_mmr_lambda": getattr(
                    prisma_settings,
                    "searchMmrLambda",
                    DEFAULT_SEARCH_MMR_LAMBDA,
                ),
                "context_token_budget": getattr(
                    prisma_settings,
                    "contextTokenBudget",
                    DEFAULT_CONTEXT_TOKEN_BUDGET,
                ),
                "chunking_use_tokens": getattr(
                    prisma_settings,
                    "chunkingUseTokens",
                    DEFAULT_CHUNKING_USE_TOKENS,
                ),
                # pgvector configuration
                "ivfflat_lists": getattr(prisma_settings, "ivfflatLists", DEFAULT_IVFFLAT_LISTS),
                # Performance / Memory configuration
                "sequential_index_loading": getattr(
                    prisma_settings,
                    "sequentialIndexLoading",
                    DEFAULT_SEQUENTIAL_INDEX_LOADING,
                ),
                "chunking_max_workers": getattr(
                    prisma_settings,
                    "chunkingMaxWorkers",
                    DEFAULT_CHUNKING_MAX_WORKERS,
                ),
                "chunking_max_batch_size": getattr(
                    prisma_settings,
                    "chunkingMaxBatchSize",
                    DEFAULT_CHUNKING_MAX_BATCH_SIZE,
                ),
                # API Tool Output configuration
                "tool_output_mode": getattr(
                    prisma_settings,
                    "toolOutputMode",
                    DEFAULT_TOOL_OUTPUT_MODE,
                ),
                # LLM settings
                "llm_provider": getattr(prisma_settings, "llmProvider", DEFAULT_LLM_PROVIDER),
                "llm_model": getattr(prisma_settings, "llmModel", DEFAULT_LLM_MODEL),
                "llm_max_tokens": getattr(prisma_settings, "llmMaxTokens", DEFAULT_LLM_MAX_TOKENS),
                "image_payload_max_width": getattr(
                    prisma_settings,
                    "imagePayloadMaxWidth",
                    DEFAULT_IMAGE_PAYLOAD_LIMITS["max_width"],
                ),
                "image_payload_max_height": getattr(
                    prisma_settings,
                    "imagePayloadMaxHeight",
                    DEFAULT_IMAGE_PAYLOAD_LIMITS["max_height"],
                ),
                "image_payload_max_pixels": getattr(
                    prisma_settings,
                    "imagePayloadMaxPixels",
                    DEFAULT_IMAGE_PAYLOAD_LIMITS["max_pixels"],
                ),
                "image_payload_max_bytes": getattr(
                    prisma_settings,
                    "imagePayloadMaxBytes",
                    DEFAULT_IMAGE_PAYLOAD_LIMITS["max_bytes"],
                ),
                "openai_api_key": openai_key,
                "openai_codex_access_token": openai_codex_access_token,
                "openai_codex_refresh_token": openai_codex_refresh_token,
                "openai_codex_token_expires_at": getattr(prisma_settings, "openaiCodexTokenExpiresAt", None),
                "openai_codex_account_id": getattr(prisma_settings, "openaiCodexAccountId", "") or "",
                "openai_codex_base_url": getattr(prisma_settings, "openaiCodexBaseUrl", OPENAI_CODEX_DEFAULT_BASE_URL) or OPENAI_CODEX_DEFAULT_BASE_URL,
                "anthropic_api_key": anthropic_key,
                "claude_code_oauth_token": os.getenv(CLAUDE_CODE_OAUTH_TOKEN_ENV, ""),
                "openrouter_api_key": openrouter_key,
                "lmstudio_api_key": lmstudio_api_key,
                "omlx_api_key": omlx_api_key,
                "github_models_api_token": github_models_api_token,
                "github_copilot_access_token": github_copilot_access_token,
                "github_copilot_refresh_token": github_copilot_refresh_token,
                "github_copilot_oauth_refresh_token": github_copilot_oauth_refresh_token,
                "github_copilot_token_expires_at": getattr(prisma_settings, "githubCopilotTokenExpiresAt", None),
                "github_copilot_enterprise_url": getattr(prisma_settings, "githubCopilotEnterpriseUrl", None),
                "github_copilot_base_url": getattr(
                    prisma_settings,
                    "githubCopilotBaseUrl",
                    DEFAULT_GITHUB_COPILOT_BASE_URL,
                ),
                "include_copilot_third_party_models": getattr(
                    prisma_settings,
                    "includeCopilotThirdPartyModels",
                    DEFAULT_INCLUDE_COPILOT_THIRD_PARTY_MODELS,
                ),
                "allowed_chat_models": prisma_settings.allowedChatModels or [],
                "default_chat_model": getattr(prisma_settings, "defaultChatModel", None),
                "allowed_openapi_models": getattr(prisma_settings, "allowedOpenapiModels", None) or [],
                "openapi_sync_chat_models": getattr(
                    prisma_settings,
                    "openapiSyncChatModels",
                    DEFAULT_OPENAPI_SYNC_CHAT_MODELS,
                ),
                "max_iterations": getattr(prisma_settings, "maxIterations", DEFAULT_MAX_ITERATIONS),
                "chat_compaction_threshold_percent": getattr(
                    prisma_settings,
                    "chatCompactionThresholdPercent",
                    DEFAULT_CHAT_COMPACTION_THRESHOLD_PERCENT,
                ),
                "chat_auto_compaction_threshold_percent": getattr(
                    prisma_settings,
                    "chatAutoCompactionThresholdPercent",
                    DEFAULT_CHAT_AUTO_COMPACTION_THRESHOLD_PERCENT,
                ),
                # Embedding settings
                "embedding_provider": prisma_settings.embeddingProvider,
                "embedding_model": prisma_settings.embeddingModel,
                "embedding_dimensions": prisma_settings.embeddingDimensions,
                **_provider_connection_settings(prisma_settings),
                # MCP settings
                "mcp_enabled": getattr(prisma_settings, "mcpEnabled", DEFAULT_MCP_ENABLED),
                "mcp_default_route_auth": getattr(
                    prisma_settings,
                    "mcpDefaultRouteAuth",
                    DEFAULT_MCP_DEFAULT_ROUTE_AUTH,
                ),
                "mcp_default_route_auth_method": getattr(
                    prisma_settings,
                    "mcpDefaultRouteAuthMethod",
                    DEFAULT_MCP_DEFAULT_ROUTE_AUTH_METHOD,
                ),
                "mcp_default_route_password": mcp_password,  # Kept encrypted
                "mcp_default_route_client_id": getattr(prisma_settings, "mcpDefaultRouteClientId", None),
                "mcp_default_route_allowed_group": prisma_settings.mcpDefaultRouteAllowedGroup,
                # OCR settings
                "default_ocr_mode": _normalize_default_ocr_mode(getattr(prisma_settings, "defaultOcrMode", DEFAULT_OCR_MODE)),
                "default_ocr_provider": getattr(
                    prisma_settings,
                    "defaultOcrProvider",
                    DEFAULT_OCR_PROVIDER,
                ),
                "default_ocr_vision_model": getattr(prisma_settings, "defaultOcrVisionModel", None),
                "ocr_concurrency_limit": getattr(
                    prisma_settings,
                    "ocrConcurrencyLimit",
                    DEFAULT_OCR_CONCURRENCY_LIMIT,
                ),
                "ollama_embedding_timeout_seconds": getattr(
                    prisma_settings,
                    "ollamaEmbeddingTimeoutSeconds",
                    DEFAULT_OLLAMA_EMBEDDING_TIMEOUT_SECONDS,
                ),
                "snapshot_retention_days": getattr(
                    prisma_settings,
                    "snapshotRetentionDays",
                    DEFAULT_SNAPSHOT_RETENTION_DAYS,
                ),
                "snapshot_stale_branch_threshold": getattr(
                    prisma_settings,
                    "snapshotStaleBranchThreshold",
                    DEFAULT_SNAPSHOT_STALE_BRANCH_THRESHOLD,
                ),
                # Index archive extraction limits
                "archive_max_total_size_bytes": getattr(prisma_settings, "archiveMaxTotalSizeBytes", ARCHIVE_MAX_TOTAL_SIZE_DEFAULT_BYTES),
                "archive_max_file_count": getattr(prisma_settings, "archiveMaxFileCount", ARCHIVE_MAX_FILE_COUNT_DEFAULT),
                # User Space configuration
                "userspace_preview_sandbox_flags": userspace_preview_sandbox_flags,
                "userspace_duplicate_copy_files_default": getattr(
                    prisma_settings,
                    "userspaceDuplicateCopyFilesDefault",
                    DEFAULT_USERSPACE_DUPLICATE_COPY_FILES,
                ),
                "userspace_duplicate_copy_metadata_default": getattr(
                    prisma_settings,
                    "userspaceDuplicateCopyMetadataDefault",
                    DEFAULT_USERSPACE_DUPLICATE_COPY_METADATA,
                ),
                "userspace_duplicate_copy_chats_default": getattr(
                    prisma_settings,
                    "userspaceDuplicateCopyChatsDefault",
                    DEFAULT_USERSPACE_DUPLICATE_COPY_CHATS,
                ),
                "userspace_duplicate_copy_mounts_default": getattr(
                    prisma_settings,
                    "userspaceDuplicateCopyMountsDefault",
                    DEFAULT_USERSPACE_DUPLICATE_COPY_MOUNTS,
                ),
                "userspace_mount_sync_interval_seconds": getattr(
                    prisma_settings,
                    "userspaceMountSyncIntervalSeconds",
                    DEFAULT_USERSPACE_MOUNT_SYNC_INTERVAL_SECONDS,
                ),
                "userspace_mount_sync_start_minute": getattr(
                    prisma_settings,
                    "userspaceMountSyncStartMinute",
                    None,
                ),
                "userspace_mount_sync_timezone": getattr(
                    prisma_settings,
                    "userspaceMountSyncTimezone",
                    None,
                ),
                "userspace_code_index_enabled": getattr(
                    prisma_settings,
                    "userspaceCodeIndexEnabled",
                    DEFAULT_USERSPACE_CODE_INDEX_ENABLED,
                ),
                "userspace_code_index_debounce_seconds": getattr(
                    prisma_settings,
                    "userspaceCodeIndexDebounceSeconds",
                    DEFAULT_USERSPACE_CODE_INDEX_DEBOUNCE_SECONDS,
                ),
                "userspace_code_index_reconcile_interval_seconds": getattr(
                    prisma_settings,
                    "userspaceCodeIndexReconcileIntervalSeconds",
                    DEFAULT_USERSPACE_CODE_INDEX_RECONCILE_INTERVAL_SECONDS,
                ),
                "userspace_code_index_max_attempts": getattr(
                    prisma_settings,
                    "userspaceCodeIndexMaxAttempts",
                    DEFAULT_USERSPACE_CODE_INDEX_MAX_ATTEMPTS,
                ),
            }
            return self._settings
        except Exception as e:
            logger.warning(f"Failed to load settings from database: {e}. Using defaults.")
            return {
                "server_name": DEFAULT_SERVER_NAME,
                "default_theme_pack": "default",
                "authenticated_webgl_background_enabled": DEFAULT_AUTHENTICATED_WEBGL_BACKGROUND_ENABLED,
                "openapi_model_prefix_enabled": DEFAULT_OPENAPI_MODEL_PREFIX_ENABLED,
                "enabled_tools": [],
                "odoo_container": DEFAULT_LEGACY_ODOO_CONTAINER,
                "postgres_container": DEFAULT_LEGACY_POSTGRES_CONTAINER,
                "postgres_host": "",
                "postgres_port": DEFAULT_LEGACY_POSTGRES_PORT,
                "postgres_user": "",
                "postgres_password": "",
                "postgres_db": "",
                "max_query_results": DEFAULT_MAX_QUERY_RESULTS,
                "query_timeout": DEFAULT_QUERY_TIMEOUT_SECONDS,
                "http_proxy_safe_timeout_seconds": DEFAULT_HTTP_PROXY_SAFE_TIMEOUT_SECONDS,
                "enable_write_ops": False,
                # Token optimization settings
                "max_tool_output_chars": DEFAULT_MAX_TOOL_OUTPUT_CHARS,
                "scratchpad_window_size": DEFAULT_SCRATCHPAD_WINDOW_SIZE,
                # Search configuration
                "search_results_k": DEFAULT_SEARCH_RESULTS_K,
                "aggregate_search": DEFAULT_AGGREGATE_SEARCH,
                "search_use_mmr": DEFAULT_SEARCH_USE_MMR,
                "search_mmr_lambda": DEFAULT_SEARCH_MMR_LAMBDA,
                "context_token_budget": DEFAULT_CONTEXT_TOKEN_BUDGET,
                "chunking_use_tokens": DEFAULT_CHUNKING_USE_TOKENS,
                # pgvector configuration
                "ivfflat_lists": DEFAULT_IVFFLAT_LISTS,
                # Performance / Memory configuration
                "sequential_index_loading": DEFAULT_SEQUENTIAL_INDEX_LOADING,
                "chunking_max_workers": DEFAULT_CHUNKING_MAX_WORKERS,
                "chunking_max_batch_size": DEFAULT_CHUNKING_MAX_BATCH_SIZE,
                # API Tool Output configuration
                "tool_output_mode": DEFAULT_TOOL_OUTPUT_MODE,
                # LLM settings
                "llm_provider": DEFAULT_LLM_PROVIDER,
                "llm_model": DEFAULT_LLM_MODEL,
                "llm_max_tokens": DEFAULT_LLM_MAX_TOKENS,
                "image_payload_max_width": DEFAULT_IMAGE_PAYLOAD_LIMITS["max_width"],
                "image_payload_max_height": DEFAULT_IMAGE_PAYLOAD_LIMITS["max_height"],
                "image_payload_max_pixels": DEFAULT_IMAGE_PAYLOAD_LIMITS["max_pixels"],
                "image_payload_max_bytes": DEFAULT_IMAGE_PAYLOAD_LIMITS["max_bytes"],
                "openai_api_key": "",
                "openai_codex_access_token": "",
                "openai_codex_refresh_token": "",
                "openai_codex_token_expires_at": None,
                "openai_codex_account_id": "",
                "openai_codex_base_url": OPENAI_CODEX_DEFAULT_BASE_URL,
                "anthropic_api_key": "",
                "claude_code_oauth_token": os.getenv(CLAUDE_CODE_OAUTH_TOKEN_ENV, ""),
                "openrouter_api_key": "",
                "lmstudio_api_key": "",
                "omlx_api_key": "",
                "github_models_api_token": "",
                "github_copilot_access_token": "",
                "github_copilot_refresh_token": "",
                "github_copilot_oauth_refresh_token": "",
                "github_copilot_token_expires_at": None,
                "github_copilot_enterprise_url": None,
                "github_copilot_base_url": DEFAULT_GITHUB_COPILOT_BASE_URL,
                "include_copilot_third_party_models": DEFAULT_INCLUDE_COPILOT_THIRD_PARTY_MODELS,
                "allowed_chat_models": [],
                "default_chat_model": None,
                "allowed_openapi_models": [],
                "openapi_sync_chat_models": DEFAULT_OPENAPI_SYNC_CHAT_MODELS,
                "max_iterations": DEFAULT_MAX_ITERATIONS,
                "chat_compaction_threshold_percent": DEFAULT_CHAT_COMPACTION_THRESHOLD_PERCENT,
                "chat_auto_compaction_threshold_percent": DEFAULT_CHAT_AUTO_COMPACTION_THRESHOLD_PERCENT,
                # Embedding settings
                "embedding_provider": DEFAULT_EMBEDDING_PROVIDER,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
                "embedding_dimensions": None,
                **_provider_connection_defaults(),
                # MCP settings
                "mcp_enabled": DEFAULT_MCP_ENABLED,
                "mcp_default_route_auth": DEFAULT_MCP_DEFAULT_ROUTE_AUTH,
                "mcp_default_route_auth_method": DEFAULT_MCP_DEFAULT_ROUTE_AUTH_METHOD,
                "mcp_default_route_password": None,
                "mcp_default_route_client_id": None,
                "mcp_default_route_allowed_group": None,
                # OCR settings
                "default_ocr_mode": DEFAULT_OCR_MODE,
                "default_ocr_provider": DEFAULT_OCR_PROVIDER,
                "default_ocr_vision_model": None,
                "ocr_concurrency_limit": DEFAULT_OCR_CONCURRENCY_LIMIT,
                "ollama_embedding_timeout_seconds": DEFAULT_OLLAMA_EMBEDDING_TIMEOUT_SECONDS,
                "snapshot_retention_days": DEFAULT_SNAPSHOT_RETENTION_DAYS,
                "snapshot_stale_branch_threshold": DEFAULT_SNAPSHOT_STALE_BRANCH_THRESHOLD,
                # Index archive extraction limits
                "archive_max_total_size_bytes": ARCHIVE_MAX_TOTAL_SIZE_DEFAULT_BYTES,
                "archive_max_file_count": ARCHIVE_MAX_FILE_COUNT_DEFAULT,
                # User Space configuration
                "userspace_preview_sandbox_flags": list(USERSPACE_PREVIEW_SANDBOX_DEFAULT_FLAGS),
                "userspace_duplicate_copy_files_default": DEFAULT_USERSPACE_DUPLICATE_COPY_FILES,
                "userspace_duplicate_copy_metadata_default": DEFAULT_USERSPACE_DUPLICATE_COPY_METADATA,
                "userspace_duplicate_copy_chats_default": DEFAULT_USERSPACE_DUPLICATE_COPY_CHATS,
                "userspace_duplicate_copy_mounts_default": DEFAULT_USERSPACE_DUPLICATE_COPY_MOUNTS,
                "userspace_mount_sync_interval_seconds": DEFAULT_USERSPACE_MOUNT_SYNC_INTERVAL_SECONDS,
                "userspace_mount_sync_start_minute": None,
                "userspace_mount_sync_timezone": None,
                "userspace_code_index_enabled": DEFAULT_USERSPACE_CODE_INDEX_ENABLED,
                "userspace_code_index_debounce_seconds": DEFAULT_USERSPACE_CODE_INDEX_DEBOUNCE_SECONDS,
                "userspace_code_index_reconcile_interval_seconds": DEFAULT_USERSPACE_CODE_INDEX_RECONCILE_INTERVAL_SECONDS,
                "userspace_code_index_max_attempts": DEFAULT_USERSPACE_CODE_INDEX_MAX_ATTEMPTS,
            }

    async def get_tool_configs(self) -> List[dict]:
        """Get enabled tool configurations from database."""
        if self._tool_configs is not None:
            return self._tool_configs

        try:
            db = await get_db()
            prisma_configs = await db.toolconfig.find_many(where={"enabled": True}, order={"createdAt": "desc"})

            enabled_tool_configs = [
                {
                    "id": cfg.id,
                    "name": cfg.name,
                    "tool_type": cfg.toolType,
                    "description": cfg.description,
                    "connection_config": decrypt_json_passwords(dict(cfg.connectionConfig), CONNECTION_CONFIG_PASSWORD_FIELDS),
                    "max_results": cfg.maxResults,
                    "timeout_max_seconds": getattr(cfg, "timeoutMaxSeconds", 300),
                    "allow_write": cfg.allowWrite,
                }
                for cfg in prisma_configs
            ]
            from ragtime.indexer.tool_health import tool_health_monitor

            self._tool_configs = tool_health_monitor.filter_healthy_tool_config_dicts(enabled_tool_configs)
            unavailable_count = len(enabled_tool_configs) - len(self._tool_configs)
            if unavailable_count > 0:
                logger.info(
                    "Filtered %d enabled tool config(s) without healthy heartbeats",
                    unavailable_count,
                )
            return self._tool_configs

        except Exception as e:
            logger.warning(f"Failed to load tool configs from database: {e}")
            return []


async def get_app_settings() -> dict:
    """Get application settings from database."""
    cache = SettingsCache.get_instance()
    was_empty = cache._settings is None
    settings = await cache.get_settings()
    if was_empty:
        _apply_runtime_setting_hooks(settings)
    return settings


def _apply_runtime_setting_hooks(settings: dict) -> None:
    """Push setting values into runtime singletons that need them."""
    try:
        from ragtime.indexer.chunking import configure_chunking_pool

        configure_chunking_pool(
            max_workers=settings.get("chunking_max_workers"),
            max_batch_size=settings.get("chunking_max_batch_size"),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to apply chunking pool settings: {exc}")


async def get_tool_configs() -> List[dict]:
    """Get enabled tool configurations from database."""
    cache = SettingsCache.get_instance()
    return await cache.get_tool_configs()


def invalidate_settings_cache() -> None:
    """Invalidate the settings cache to force a refresh."""
    cache = SettingsCache.get_instance()
    cache.invalidate()
