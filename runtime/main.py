import os

from runtime.manager.api import create_app as create_manager_app
from runtime.worker.api import create_app as create_worker_app
from runtime.worker.api import include_worker_routes


def create_app():
    mode = os.getenv("RUNTIME_SERVICE_MODE", "manager").strip().lower()
    if mode == "worker":
        return create_worker_app()
    application = create_manager_app()
    include_worker_routes(application)
    return application


app = create_app()

__all__ = ["app", "create_app"]
