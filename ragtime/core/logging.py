"""
Logging configuration for the application.
"""

import logging
import re
import shutil
import sys
from typing import Optional

from ragtime.config import settings

# Custom Log Level
NOTICE = 25
logging.addLevelName(NOTICE, "NOTICE")

# ANSI Color Codes
BLUE = "\033[0;34m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
BOLD_RED = "\033[1;31m"
RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log records based on level."""

    # Keywords to highlight in message body
    # (Keyword, Color)
    KEYWORDS = [
        ("ERROR", RED),
        ("FAILED", RED),
        ("CRITICAL", BOLD_RED),
        ("WARNING", YELLOW),
        ("WARN", YELLOW),
        ("Exception", RED),
    ]

    # Base format matching the previous configuration
    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    FORMATS = {
        logging.DEBUG: BLUE + FMT + RESET,
        logging.INFO: FMT,
        NOTICE: "%(message)s",
        logging.WARNING: YELLOW + FMT + RESET,
        logging.ERROR: RED + FMT + RESET,
        logging.CRITICAL: BOLD_RED + FMT + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        # Get standard formatted message
        log_fmt = self.FORMATS.get(record.levelno, self.FMT)
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FMT)
        formatted_message = formatter.format(record)

        # Highlight keywords
        level_color = self.get_level_color(record.levelno)
        for keyword, color in self.KEYWORDS:
            # Use word boundaries to prevent partial matches (e.g. WARN matching WARNINGS)
            # escape keyword to handle special chars safe
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, formatted_message):
                # Color the keyword and then resume the level's base color
                replacement = f"{color}{keyword}{RESET}{level_color}"
                formatted_message = re.sub(pattern, replacement, formatted_message)

        return formatted_message

    def get_level_color(self, levelno: int) -> str:
        if levelno >= logging.CRITICAL:
            return BOLD_RED
        elif levelno >= logging.ERROR:
            return RED
        elif levelno >= logging.WARNING:
            return YELLOW
        elif levelno >= NOTICE:
            return RESET
        elif levelno >= logging.INFO:
            return RESET
        elif levelno >= logging.DEBUG:
            return BLUE
        return RESET


class UvicornAccessFilter(logging.Filter):
    """
    Filter to downgrade noisy polling/health endpoints to DEBUG level.
    Preserves INFO level for actual resource access (audit trail).
    """

    # Endpoints that generate frequent polling requests
    QUIET_PATHS = {
        "/health",
        "/events",  # SSE event streams
        "/task",  # Task status polling
        "/interrupted-task",  # Interrupted task polling
    }

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        # Only downgrade GET requests to noisy polling endpoints
        if '"GET ' in message:
            if any(path in message for path in self.QUIET_PATHS):
                record.levelno = logging.DEBUG
                record.levelname = "DEBUG"
        return True


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """
    Set up and configure logging for the application.

    Args:
        name: Logger name. If None, returns root logger.

    Returns:
        Configured logger instance.
    """
    log_level = logging.DEBUG if settings.debug_mode else logging.INFO

    # Create handler with colored formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[handler],
        force=True,  # Ensure we override any existing configuration
    )
    # Standardize Uvicorn logging to match application format
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        log_obj = logging.getLogger(logger_name)
        log_obj.handlers = [handler]  # Replace default uvicorn handlers
        log_obj.propagate = False  # Prevent duplication to root

    # Add filter to downgrade GET requests to DEBUG level
    logging.getLogger("uvicorn.access").addFilter(UvicornAccessFilter())

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("prisma").setLevel(logging.WARNING)
    # SSH/SFTP libraries - authentication success messages are noisy
    # logging.getLogger("paramiko").setLevel(logging.WARNING)
    # logging.getLogger("paramiko.transport").setLevel(logging.WARNING)
    # Async/network libraries
    # logging.getLogger("aiohttp").setLevel(logging.WARNING)
    # logging.getLogger("asyncio").setLevel(logging.WARNING)

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


def get_ui_width(deduction: int = 45) -> int:
    """
    Get the width for UI separators, deducting the log prefix length.
    Returns terminal width - deduction.
    """
    # Default to 80 if cannot determine (e.g. piped output)
    term_width = shutil.get_terminal_size((80, 24)).columns
    # Ensure at least 10 chars
    return max(10, term_width - deduction)


# Default logger for the application
logger = setup_logging("rag_api")
