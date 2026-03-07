import logging
import os
import sys

from runtime.manager.api import create_app as create_manager_app
from runtime.worker.api import create_app as create_worker_app
from runtime.worker.api import include_worker_routes


# Downgrade high-frequency polling endpoints from INFO to DEBUG.
class UvicornAccessFilter(logging.Filter):
    QUIET_PATHS = {
        "/health",
        "/sessions/",
    }

    def __init__(self, debug_mode: bool):
        super().__init__()
        self._debug_mode = debug_mode

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if '"GET ' in message and any(path in message for path in self.QUIET_PATHS):
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
            if not self._debug_mode:
                return False
        return True


def _setup_logging(name: str | None = None) -> logging.Logger:
    debug_mode = os.getenv("DEBUG_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    log_level = logging.DEBUG if debug_mode else logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))

    logging.basicConfig(
        level=log_level,
        handlers=[handler],
        force=True,
    )

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        log_obj = logging.getLogger(logger_name)
        log_obj.handlers = [handler]
        log_obj.propagate = False

    logging.getLogger("uvicorn.access").addFilter(UvicornAccessFilter(debug_mode))
    return logging.getLogger(name or "runtime")


# Initialize runtime-local logging for access log filtering.
_setup_logging("runtime")


def create_app():
    mode = os.getenv("RUNTIME_SERVICE_MODE", "manager").strip().lower()
    if mode == "worker":
        return create_worker_app()
    application = create_manager_app()
    include_worker_routes(application)
    return application


app = create_app()

__all__ = ["app", "create_app"]
