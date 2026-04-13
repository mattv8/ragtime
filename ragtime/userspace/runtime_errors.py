
from __future__ import annotations

class RuntimeVersionConflictError(Exception):
    def __init__(self, expected_version: int, actual_version: int) -> None:
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Version conflict: expected {expected_version}, current {actual_version}"
        )
