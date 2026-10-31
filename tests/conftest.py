from __future__ import annotations

import os
import tempfile

# Ensure test imports never try writing to /data on host runs.
base_index_data_path = os.environ.setdefault(
    "INDEX_DATA_PATH",
    os.path.join(tempfile.gettempdir(), "ragtime-test-index-data"),
)
xdist_worker = os.environ.get("PYTEST_XDIST_WORKER")
if xdist_worker:
    os.environ["INDEX_DATA_PATH"] = os.path.join(base_index_data_path, xdist_worker)
    os.makedirs(os.environ["INDEX_DATA_PATH"], exist_ok=True)

# Import the real prompts module up front so individual tests that conditionally
# inject a fake module do not poison sys.modules during collection.
try:
    import ragtime.rag.prompts  # noqa: F401
except Exception:
    # Some narrow tests intentionally stub prompt imports; keep collection alive.
    pass
