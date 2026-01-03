"""
Logging configuration for the application.
"""

import logging
import sys
from typing import Optional

from ragtime.config import settings


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """
    Set up and configure logging for the application.

    Args:
        name: Logger name. If None, returns root logger.

    Returns:
        Configured logger instance.
    """
    log_level = logging.DEBUG if settings.debug_mode else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("prisma").setLevel(logging.WARNING)

    return logging.getLogger(name or "rag_api")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__ of the module).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


# Default logger for the application
logger = setup_logging("rag_api")
