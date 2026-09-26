"""
Database module for Prisma client initialization and lifecycle management.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional, cast

from prisma import Prisma
from prisma._types import PrismaMethod
from pydantic import BaseModel

from ragtime.config import settings
from ragtime.core.logging import get_logger
from ragtime.core.performance import track_operation

logger = get_logger(__name__)


class InstrumentedPrisma(Prisma):
    """Generated Prisma client with payload-free operation timings."""

    async def _execute(
        self,
        *,
        method: PrismaMethod,
        arguments: dict[str, Any],
        model: type[BaseModel] | None = None,
        root_selection: list[str] | None = None,
    ) -> Any:
        model_name = model.__name__ if model is not None else "raw"
        with track_operation(f"db.{model_name}.{method}"):
            parent = cast(Any, super())
            return await parent._execute(
                method=method,
                arguments=arguments,
                model=model,
                root_selection=root_selection,
            )


class DatabaseManager:
    """Manages the Prisma database connection."""

    def __init__(self) -> None:
        self._db: Optional[Prisma] = None

    @property
    def client(self) -> Prisma:
        """Get the Prisma client instance.

        Raises:
            RuntimeError: If the database is not connected.
        """
        if self._db is None or not self._db.is_connected():
            raise RuntimeError("Database is not connected. Call connect() first.")
        return self._db

    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._db is not None and self._db.is_connected()

    async def connect(self) -> Prisma:
        """Connect to the database and return the Prisma client."""
        if self._db is None:
            self._db = InstrumentedPrisma(http={"timeout": settings.prisma_timeout})

        if not self._db.is_connected():
            logger.info("Connecting to database...")
            await self._db.connect()
            logger.info("Database connected successfully")

        return self._db

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._db is not None and self._db.is_connected():
            logger.info("Disconnecting from database...")
            await self._db.disconnect()
            logger.info("Database disconnected")


# Global database manager instance
_manager = DatabaseManager()


async def get_db() -> Prisma:
    """Get the global Prisma client instance."""
    return _manager.client


async def connect_db() -> Prisma:
    """Connect to the database and return the Prisma client."""
    return await _manager.connect()


async def disconnect_db() -> None:
    """Disconnect from the database."""
    await _manager.disconnect()


@asynccontextmanager
async def db_lifespan() -> AsyncGenerator[Prisma, None]:
    """Context manager for database lifecycle.

    Usage:
        async with db_lifespan() as db:
            # use db
    """
    db = await connect_db()
    try:
        yield db
    finally:
        await disconnect_db()
