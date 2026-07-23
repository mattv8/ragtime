"""Shared helpers for database query tools."""

from __future__ import annotations

from ragtime.core.sql_utils import enforce_max_results, validate_sql_query


def validate_and_prepare_query(
    query: str,
    *,
    max_results: int,
    allow_write: bool,
    db_type: str,
    require_result_limit: bool,
    enforce_result_limit: bool,
) -> tuple[bool, str]:
    """Validate a query and apply the configured result limit when enabled.

    The returned string is either the unchanged/limited query or the validator
    reason when validation fails.  Keeping this orchestration shared avoids
    dialect tools drifting while leaving dialect-specific validation in
    ``sql_utils``.
    """
    is_safe, reason = validate_sql_query(
        query,
        enable_write=allow_write,
        db_type=db_type,
        require_result_limit=require_result_limit,
    )
    if not is_safe:
        return False, reason

    if enforce_result_limit:
        query = enforce_max_results(query, max_results, db_type=db_type)

    return True, query
