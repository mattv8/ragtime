"""Compatibility exports for shared workspace utility operations.

The runtime package owns these implementations because both the Ragtime app
and runtime services use them. Keep only Ragtime-specific defaults here.
"""

from runtime.core.workspace_ops import (
    PLATFORM_MANAGED_GITIGNORE_PATTERNS,
    SQLITE_FILE_EXTENSIONS,
    SQLITE_MANAGED_DIR_PREFIX,
    compute_file_hash,
    deduplicate_ancestor_paths,
    enforce_sqlite_managed_path,
    normalize_relative_file_path,
    normalize_runtime_file_path,
    sync_scope_relative_paths,
    workspace_mount_target_repo_relative_path,
    workspace_path_matches_mount_prefix,
)

WORKSPACE_DEFAULT_GITIGNORE_PATTERNS = (
    "node_modules/",
    "dist/",
    "__pycache__/",
)

__all__ = (
    "PLATFORM_MANAGED_GITIGNORE_PATTERNS",
    "SQLITE_FILE_EXTENSIONS",
    "SQLITE_MANAGED_DIR_PREFIX",
    "WORKSPACE_DEFAULT_GITIGNORE_PATTERNS",
    "compute_file_hash",
    "deduplicate_ancestor_paths",
    "enforce_sqlite_managed_path",
    "normalize_relative_file_path",
    "normalize_runtime_file_path",
    "sync_scope_relative_paths",
    "workspace_mount_target_repo_relative_path",
    "workspace_path_matches_mount_prefix",
)
