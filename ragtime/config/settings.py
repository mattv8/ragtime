"""
Application settings loaded from environment variables.

Note: LLM and embedding provider settings are stored in the database
and configured via the Settings UI at http://localhost:8001/indexes/ui
See: ragtime/core/app_settings.py and indexer/routes.py (GET/PUT /indexes/settings)
"""

import os
import secrets
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# File to persist auto-generated encryption key (in data volume)
# Used for both JWT signing and secrets encryption (Fernet)
ENCRYPTION_KEY_FILE = (
    Path(os.environ.get("INDEX_DATA_PATH", "/data")) / ".encryption_key"
)

SameSiteType = Literal["lax", "strict", "none"]


class Settings(BaseSettings):
    """
    Infrastructure settings loaded from environment variables.

    Note: LLM/embedding provider settings are in the database (Settings UI).
    """

    # Database (Prisma)
    database_url: str = Field(
        default="postgresql://ragtime:ragtime_dev@localhost:5434/ragtime",
        alias="DATABASE_URL",
    )
    prisma_timeout: int = Field(
        default=60,
        alias="PRISMA_TIMEOUT",
        description="Timeout in seconds for database operations",
    )

    # Security
    api_key: str = Field(default="", alias="API_KEY")  # API key for auth
    allowed_origins: str = Field(default="*", alias="ALLOWED_ORIGINS")
    external_base_url: str = Field(
        default="",
        alias="EXTERNAL_BASE_URL",
        description=(
            "Canonical external base URL (e.g. https://ragtime.example.com). "
            "When set, OAuth metadata and other public-facing URLs use this "
            "value instead of trusting X-Forwarded-* request headers."
        ),
    )
    enable_https: bool = Field(
        default=False,
        alias="ENABLE_HTTPS",
        description="Enable HTTPS with auto-generated self-signed certificate",
    )
    ssl_cert_file: str = Field(
        default="",
        alias="SSL_CERT_FILE",
        description="Path to SSL certificate file (auto-generated if ENABLE_HTTPS=true)",
    )
    ssl_key_file: str = Field(
        default="",
        alias="SSL_KEY_FILE",
        description="Path to SSL private key file (auto-generated if ENABLE_HTTPS=true)",
    )

    # Debug mode
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")

    # Indexer settings
    index_data_path: str = Field(default="/data", alias="INDEX_DATA_PATH")

    # User Space runtime provider settings
    userspace_runtime_manager_url: str = Field(
        default="http://runtime:8090",
        alias="RUNTIME_MANAGER_URL",
        description="Base URL for external runtime manager API",
    )
    userspace_runtime_manager_timeout_seconds: float = Field(
        default=60.0,
        alias="RUNTIME_MANAGER_TIMEOUT_SECONDS",
        description="Timeout in seconds for runtime manager HTTP requests",
    )
    userspace_runtime_manager_retry_attempts: int = Field(
        default=3,
        alias="RUNTIME_MANAGER_RETRY_ATTEMPTS",
        description="Retry attempts for runtime manager HTTP requests",
    )
    userspace_runtime_manager_retry_delay_seconds: float = Field(
        default=0.2,
        alias="RUNTIME_MANAGER_RETRY_DELAY_SECONDS",
        description="Base retry delay in seconds for runtime manager HTTP requests",
    )
    userspace_runtime_manager_auth_token: str = Field(
        default="",
        alias="RUNTIME_MANAGER_AUTH_TOKEN",
        description="Optional bearer token used when calling runtime manager",
    )
    userspace_runtime_worker_auth_token: str = Field(
        default="",
        alias="RUNTIME_WORKER_AUTH_TOKEN",
        description="Bearer token for authenticating preview proxy requests to the runtime worker",
    )

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    api_port: int = Field(default=8001, alias="API_PORT")

    # -------------------------------------------------------------------------
    # Authentication Settings
    # -------------------------------------------------------------------------

    # JWT Configuration
    encryption_key: str = Field(
        default="",
        alias="ENCRYPTION_KEY",
        description="Key for JWT signing and secrets encryption. Auto-generated if not set.",
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_hours: int = Field(default=24, alias="JWT_EXPIRE_HOURS")

    @field_validator("encryption_key", mode="after")
    @classmethod
    def generate_encryption_key_if_empty(cls, v: str) -> str:
        """
        Auto-generate encryption key if not provided or empty.

        The key is always persisted to a file in the data volume so it survives
        container restarts. This ensures encrypted secrets remain recoverable
        and user sessions stay valid.
        """
        # Try to load from persisted file first (even if env var is set, file takes precedence)
        if ENCRYPTION_KEY_FILE.exists():
            try:
                saved_key = ENCRYPTION_KEY_FILE.read_text().strip()
                if saved_key:
                    return saved_key
            except OSError:
                pass

        # Use provided value or generate new key
        key = v if v else secrets.token_urlsafe(32)

        # Persist to file for future restarts
        try:
            ENCRYPTION_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
            ENCRYPTION_KEY_FILE.write_text(key)
            print(
                f"[INFO] Encryption key saved to {ENCRYPTION_KEY_FILE}",
                file=sys.stderr,
            )
        except OSError as e:
            print(
                f"[WARNING] Could not persist encryption key to {ENCRYPTION_KEY_FILE}: {e}",
                file=sys.stderr,
            )
            print(
                "[WARNING] Key will be lost on container restart!",
                file=sys.stderr,
            )

        return key

    # Session cookie settings
    session_cookie_name: str = Field(
        default="ragtime_session", alias="SESSION_COOKIE_NAME"
    )
    session_cookie_secure: bool = Field(
        default=False, alias="SESSION_COOKIE_SECURE"
    )  # Set True in production with HTTPS
    session_cookie_httponly: bool = Field(default=True, alias="SESSION_COOKIE_HTTPONLY")
    session_cookie_samesite: SameSiteType = Field(
        default="lax", alias="SESSION_COOKIE_SAMESITE"
    )

    # Local fallback admin (works when LDAP is unreachable)
    # Username is auto-prefixed with "local:" to avoid collision with LDAP users
    local_admin_user: str = Field(default="admin", alias="LOCAL_ADMIN_USER")
    local_admin_password: str = Field(
        default="", alias="LOCAL_ADMIN_PASSWORD"
    )  # Must be set to enable local admin

    # -------------------------------------------------------------------------
    # MCP Server Settings
    # -------------------------------------------------------------------------
    mcp_heartbeat_cache_ttl: int = Field(
        default=30,
        alias="MCP_HEARTBEAT_CACHE_TTL",
        description="Seconds to cache heartbeat results (default 30)",
    )

    # Note: LDAP configuration is stored in the database and managed via the Settings UI
    # LDAP is enabled when serverUrl is configured in the database

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
