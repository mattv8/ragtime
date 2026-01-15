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

# File to persist auto-generated JWT secret (in data volume)
JWT_SECRET_FILE = Path(os.environ.get("INDEX_DATA_PATH", "/data")) / ".jwt_secret"

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

    # Security
    api_key: str = Field(default="", alias="API_KEY")  # API key for auth
    allowed_origins: str = Field(default="*", alias="ALLOWED_ORIGINS")
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

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    api_port: int = Field(default=8001, alias="API_PORT")

    # -------------------------------------------------------------------------
    # Authentication Settings
    # -------------------------------------------------------------------------

    # JWT Configuration
    jwt_secret_key: str = Field(
        default="",
        alias="JWT_SECRET_KEY",
        description="Secret key for JWT signing. Auto-generated if not set.",
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_hours: int = Field(default=24, alias="JWT_EXPIRE_HOURS")

    @field_validator("jwt_secret_key", mode="after")
    @classmethod
    def generate_jwt_secret_if_empty(cls, v: str) -> str:
        """
        Auto-generate JWT secret if not provided or empty.

        The secret is persisted to a file in the data volume so it survives
        container restarts. This ensures encrypted secrets remain recoverable
        and user sessions stay valid.
        """
        if v:
            return v

        # Try to load from persisted file first
        if JWT_SECRET_FILE.exists():
            try:
                saved_key = JWT_SECRET_FILE.read_text().strip()
                if saved_key:
                    return saved_key
            except OSError:
                pass

        # Generate new key
        new_key = secrets.token_urlsafe(32)

        # Persist to file for future restarts
        try:
            JWT_SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
            JWT_SECRET_FILE.write_text(new_key)
            # Log warning about auto-generation (to stderr since logging may not be set up yet)
            print(
                f"[WARNING] JWT_SECRET_KEY not set - auto-generated and saved to {JWT_SECRET_FILE}",
                file=sys.stderr,
            )
            print(
                "[WARNING] For production, set JWT_SECRET_KEY explicitly in .env",
                file=sys.stderr,
            )
        except OSError as e:
            print(
                f"[WARNING] Could not persist JWT secret to {JWT_SECRET_FILE}: {e}",
                file=sys.stderr,
            )
            print(
                "[WARNING] Secret will be lost on container restart!",
                file=sys.stderr,
            )

        return new_key

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
