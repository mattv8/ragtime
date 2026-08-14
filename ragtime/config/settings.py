"""
Application settings loaded from environment variables.

Note: LLM and embedding provider settings are stored in the database
and configured via the Settings UI at http://localhost:8001/indexes/ui
See: ragtime/core/app_settings.py and indexer/routes.py (GET/PUT /indexes/settings)
"""

import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragtime.oauth_redirects import (
    DEFAULT_ALLOWED_ORIGINS,
    DEFAULT_TRUSTED_REDIRECT_URIS,
)

# File to persist the managed encryption key (in data volume)
# Used for both JWT signing and secrets encryption (Fernet)
ENCRYPTION_KEY_FILE = Path(os.environ.get("INDEX_DATA_PATH", "/data")) / ".encryption_key"

SameSiteType = Literal["lax", "strict", "none"]

# Generic token values shipped by old compose templates and docs. Any of these
# (or an empty token) triggers the admin security banner so operators migrate
# to a strong RUNTIME_AUTH_TOKEN. They are NOT filtered from resolution: legacy
# deployments keep working, they just get warned.
_RUNTIME_AUTH_TOKEN_GENERIC_DEFAULTS = {
    "runtime-auth-token",
    "runtime-manager-token",
    "runtime-worker-token",
    "dev-runtime-auth-token",
    "dev-runtime-manager-token",
    "dev-runtime-worker-token",
}


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
    allowed_origins: str = Field(
        default="",
        alias="ALLOWED_ORIGINS",
        description=(
            "Comma-separated list of allowed CORS origins. When empty, only "
            "loopback origins plus built-in trusted OAuth web origins "
            f"({', '.join(DEFAULT_ALLOWED_ORIGINS)}) are permitted. When set, "
            "explicit origins such as 'https://ragtime.example.com' are added "
            "to those defaults."
        ),
    )
    external_base_url: str = Field(
        default="",
        alias="EXTERNAL_BASE_URL",
        description=(
            "Canonical external base URL (e.g. https://ragtime.example.com). "
            "When set, OAuth metadata and other public-facing URLs use this "
            "value instead of trusting X-Forwarded-* request headers."
        ),
    )
    userspace_preview_base_domain: str = Field(
        default="",
        alias="USERSPACE_PREVIEW_BASE_DOMAIN",
        description=(
            "Optional wildcard base-domain override for per-workspace preview "
            "origins. Leave unset to derive preview hosts from the active "
            "Ragtime origin; set it when previews should use a different "
            "host family such as example-userspaces.com."
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
    userspace_runtime_auth_token: str = Field(
        default="",
        alias="RUNTIME_AUTH_TOKEN",
        description="Bearer token shared by Ragtime and the runtime service",
    )
    runtime_bridge_base_url: str = Field(
        default="",
        alias="RUNTIME_BRIDGE_BASE_URL",
        description="Optional internal Ragtime origin reachable by User Space runtime workers",
    )
    # DEPRECATED legacy bridge: older compose files set RUNTIME_MANAGER_AUTH_TOKEN /
    # RUNTIME_WORKER_AUTH_TOKEN instead of RUNTIME_AUTH_TOKEN. When the shared
    # token is unset we fall back to the legacy manager value so those
    # deployments keep working after upgrade; the admin UI shows a migration
    # warning (see runtime_auth_token_warning). The worker variable is ignored.
    # Remove this bridge once legacy deployments have migrated.
    userspace_runtime_manager_auth_token: str = Field(
        default="",
        alias="RUNTIME_MANAGER_AUTH_TOKEN",
        description="DEPRECATED: legacy fallback for RUNTIME_AUTH_TOKEN",
    )
    # Tracks whether the effective token came from the legacy bridge, so the
    # security banner can nudge migration even when the legacy value is strong.
    runtime_auth_token_from_legacy: bool = Field(default=False, exclude=True)
    tavily_api_key: str = Field(
        default="",
        alias="TAVILY_API_KEY",
        description=("API key for Tavily web search. When unset, chat diagnostics uses the bundled SearXNG service."),
    )
    claude_code_oauth_token: str = Field(
        default="",
        alias="CLAUDE_CODE_OAUTH_TOKEN",
        description="Optional Claude Code OAuth token passed through to the installed claude CLI.",
    )

    # Cloud userspace mount OAuth app credentials. Admins register one app per
    # provider; individual users authorize their own accounts against it.
    cloud_mount_microsoft_client_id: str = Field(default="", alias="CLOUD_MOUNT_MICROSOFT_CLIENT_ID")
    cloud_mount_microsoft_client_secret: str = Field(default="", alias="CLOUD_MOUNT_MICROSOFT_CLIENT_SECRET")
    cloud_mount_microsoft_tenant_id: str = Field(default="", alias="CLOUD_MOUNT_MICROSOFT_TENANT_ID")
    cloud_mount_google_client_id: str = Field(default="", alias="CLOUD_MOUNT_GOOGLE_CLIENT_ID")
    cloud_mount_google_client_secret: str = Field(default="", alias="CLOUD_MOUNT_GOOGLE_CLIENT_SECRET")

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
        description="Deprecated compatibility seed for the managed encryption key file.",
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_hours: int = Field(default=24, alias="JWT_EXPIRE_HOURS")

    @model_validator(mode="after")
    def apply_legacy_runtime_auth_token_bridge(self) -> "Settings":
        # Always derived, never accepted from the environment.
        self.runtime_auth_token_from_legacy = False
        shared_token = (self.userspace_runtime_auth_token or "").strip()
        legacy_token = (self.userspace_runtime_manager_auth_token or "").strip()
        if not shared_token and legacy_token:
            shared_token = legacy_token
            self.runtime_auth_token_from_legacy = True
        self.userspace_runtime_auth_token = shared_token
        return self

    def runtime_auth_token_warning(self) -> bool:
        """True when admins should be nudged to configure RUNTIME_AUTH_TOKEN.

        Warns when the effective token is empty, is a well-known generic
        default from old compose templates/docs, or was resolved through the
        deprecated legacy variable bridge. Suppressed in debug mode where the
        dev compose default is expected.
        """
        if self.debug_mode:
            return False
        token = self.userspace_runtime_auth_token
        return not token or token in _RUNTIME_AUTH_TOKEN_GENERIC_DEFAULTS or self.runtime_auth_token_from_legacy

    @field_validator("encryption_key", mode="after")
    @classmethod
    def generate_encryption_key_if_empty(cls, v: str) -> str:
        """
        Resolve the managed encryption key.

        The persisted .encryption_key file is authoritative. ENCRYPTION_KEY only
        seeds a missing file once for compatibility. When no key exists, a new
        key is generated and persisted with mode 0600.
        """
        saved_key = cls._read_managed_encryption_key()
        if saved_key:
            return saved_key

        key = v.strip() if v else ""
        if not key:
            key = secrets.token_urlsafe(32)

        try:
            cls._persist_managed_encryption_key(key)
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

    @staticmethod
    def _read_managed_encryption_key() -> str:
        if not ENCRYPTION_KEY_FILE.exists():
            return ""

        try:
            return ENCRYPTION_KEY_FILE.read_text().strip()
        except OSError:
            return ""

    @staticmethod
    def _persist_managed_encryption_key(key: str) -> None:
        ENCRYPTION_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        ENCRYPTION_KEY_FILE.write_text(key)
        ENCRYPTION_KEY_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)

    # Session cookie settings
    session_cookie_name: str = Field(default="ragtime_session", alias="SESSION_COOKIE_NAME")
    session_cookie_secure: bool = Field(default=False, alias="SESSION_COOKIE_SECURE")  # Set True in production with HTTPS
    session_cookie_httponly: bool = Field(default=True, alias="SESSION_COOKIE_HTTPONLY")
    session_cookie_samesite: SameSiteType = Field(default="lax", alias="SESSION_COOKIE_SAMESITE")

    # Local fallback admin (works when LDAP is unreachable)
    # Username is auto-prefixed with "local:" to avoid collision with LDAP users
    local_admin_user: str = Field(default="admin", alias="LOCAL_ADMIN_USER")
    local_admin_password: str = Field(default="", alias="LOCAL_ADMIN_PASSWORD")  # Must be set to enable local admin
    oauth_trusted_redirect_uris: str = Field(
        default="",
        alias="OAUTH_TRUSTED_REDIRECT_URIS",
        description=(
            "Comma-separated OAuth callback URLs to trust in addition to built-in "
            "loopback and IDE defaults (for example "
            f"{DEFAULT_TRUSTED_REDIRECT_URIS[0]},https://example.com/oauth/callback)."
        ),
    )

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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
