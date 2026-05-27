from __future__ import annotations

import os
import tempfile

# Ensure test imports never try writing to /data on host runs.
os.environ.setdefault(
    "INDEX_DATA_PATH",
    os.path.join(tempfile.gettempdir(), "ragtime-test-index-data"),
)

# Import the real prompts module up front so individual tests that conditionally
# inject a fake module do not poison sys.modules during collection.
try:
    import ragtime.rag.prompts  # noqa: F401
except Exception:
    # Some narrow tests intentionally stub prompt imports; keep collection alive.
    pass
