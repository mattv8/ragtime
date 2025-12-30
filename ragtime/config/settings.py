"""
Application settings loaded from environment variables.

Note: LLM and embedding provider settings are stored in the database
and configured via the Settings UI at http://localhost:8001/indexes/ui
See: ragtime/core/app_settings.py and indexer/routes.py (GET/PUT /indexes/settings)
"""

import secrets
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

SameSiteType = Literal["lax", "strict", "none"]


class Settings(BaseSettings):
    """
    Infrastructure settings loaded from environment variables.

    Note: LLM/embedding provider settings are in the database (Settings UI).
    """

    # Database (Prisma)
    database_url: str = Field(
        default="postgresql://ragtime:ragtime_dev@localhost:5434/ragtime",
        alias="DATABASE_URL"
    )

    # Security
    api_key: str = Field(default="", alias="API_KEY")  # Optional API key for auth
    allowed_origins: str = Field(default="*", alias="ALLOWED_ORIGINS")

    # Debug mode
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")

    # Indexer settings
    index_data_path: str = Field(default="/app/data", alias="INDEX_DATA_PATH")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    api_port: int = Field(default=8001, alias="API_PORT")

    # -------------------------------------------------------------------------
    # Authentication Settings
    # -------------------------------------------------------------------------

    # JWT Configuration
    jwt_secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        alias="JWT_SECRET_KEY",
        description="Secret key for JWT signing. Auto-generated if not set."
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_hours: int = Field(default=24, alias="JWT_EXPIRE_HOURS")

    # Session cookie settings
    session_cookie_name: str = Field(default="ragtime_session", alias="SESSION_COOKIE_NAME")
    session_cookie_secure: bool = Field(default=False, alias="SESSION_COOKIE_SECURE")  # Set True in production with HTTPS
    session_cookie_httponly: bool = Field(default=True, alias="SESSION_COOKIE_HTTPONLY")
    session_cookie_samesite: SameSiteType = Field(default="lax", alias="SESSION_COOKIE_SAMESITE")

    # Local fallback admin (works when LDAP is unreachable)
    # Username is auto-prefixed with "local:" to avoid collision with LDAP users
    local_admin_user: str = Field(default="admin", alias="LOCAL_ADMIN_USER")
    local_admin_password: str = Field(default="", alias="LOCAL_ADMIN_PASSWORD")  # Must be set to enable local admin

    # Note: LDAP configuration is stored in the database and managed via the Settings UI
    # LDAP is enabled when serverUrl is configured in the database

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
