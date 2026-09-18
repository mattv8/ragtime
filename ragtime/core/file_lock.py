from __future__ import annotations

import fcntl
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

BACKUP_LOCK_PATH: Path = Path(tempfile.gettempdir()) / "ragtime-server-backup.lock"


@contextmanager
def backup_restore_lock(lock_path: Path | None = None) -> Iterator[None]:
    """Serialize backup and restore filesystem mutations across processes."""
    path = lock_path or BACKUP_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
