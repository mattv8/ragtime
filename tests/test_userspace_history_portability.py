"""Workspace archive history portability contract.

The archive schema currently has no authenticated runtime history import route.
Snapshot exports therefore continue to contain only Git snapshot state, and do
not claim to include SQLite database-history images.
"""

from pathlib import Path


def test_workspace_snapshot_archive_has_no_sqlite_history_payload_contract() -> None:
    service_source = Path("ragtime/userspace/service.py").read_text(encoding="utf-8")
    archive_section = service_source[
        service_source.index("async def _run_workspace_archive_export_task") : service_source.index("async def _run_workspace_archive_import_task")
    ]
    assert 'manifest["sqlite_history"]' not in archive_section
    assert "stage_workspace_export" not in archive_section
