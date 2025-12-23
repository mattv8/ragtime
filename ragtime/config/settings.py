"""
Application settings loaded from environment variables.

Note: LLM and embedding provider settings are stored in the database
and configured via the Settings UI at http://localhost:8001/indexes/ui
See: ragtime/core/app_settings.py and indexer/routes.py (GET/PUT /indexes/settings)
"""

from pydantic import Field
from pydantic_settings import BaseSettings


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

    # FAISS indexes (comma-separated paths for multiple codebases)
    faiss_index_paths: str = Field(
        default="data/odoo,data/codebase",
        alias="FAISS_INDEX_PATHS"
    )

    # Security
    api_key: str = Field(default="", alias="API_KEY")  # Optional API key for auth
    allowed_origins: str = Field(default="*", alias="ALLOWED_ORIGINS")

    # Feature flags
    enable_tools: bool = Field(default=True, alias="ENABLE_TOOLS")
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")

    # Indexer settings
    enable_indexer: bool = Field(default=True, alias="ENABLE_INDEXER")
    index_data_path: str = Field(default="/app/data", alias="INDEX_DATA_PATH")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    indexer_port: int = Field(default=8001, alias="INDEXER_PORT")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
