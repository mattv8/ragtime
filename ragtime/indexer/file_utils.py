"""
Shared file utilities for indexing services.

This module extracts common file handling functionality used by both
IndexerService (upload/git) and FilesystemIndexerService (local/SMB/NFS).
"""

import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from ragtime.core.file_constants import (
    BINARY_EXTENSIONS,
    MINIFIED_PATTERNS,
    PARSEABLE_DOCUMENT_EXTENSIONS,
    UNPARSEABLE_BINARY_EXTENSIONS,
)
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# ==============================================================================
# Hardcoded exclude patterns - directories that should never be indexed
# ==============================================================================
HARDCODED_EXCLUDES = [
    ".git/**",
    "__pycache__/**",
    "node_modules/**",
    ".venv/**",
    "venv/**",
    "**/vendor/**",  # Third-party libraries (JS/PHP/etc.)
    "**/static/vendor/**",  # Static vendor assets
]


# ==============================================================================
# Git Provider Token Authentication
# ==============================================================================
# Supported platforms and their HTTPS clone URL token formats:
#
# GitHub (github.com):
#   - Token prefixes: ghp_, gho_, ghu_, ghs_, ghr_, github_pat_
#   - Format: https://x-access-token:{token}@github.com/owner/repo.git
#
# GitLab (gitlab.com and self-hosted):
#   - Token prefixes: glpat-, glptt-, gldt-, glsoat-
#   - Format: https://oauth2:{token}@{host}/owner/repo.git
#
# Bitbucket (bitbucket.org):
#   - Format: https://x-bitbucket-api-token-auth:{token}@bitbucket.org/...
#
# Generic (Gitea, Gogs, Forgejo, etc.):
#   - Format: https://{token}@{host}/owner/repo.git
# ==============================================================================

_GITHUB_TOKEN_PREFIXES = ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")
_GITLAB_TOKEN_PREFIXES = ("glpat-", "glptt-", "gldt-", "glsoat-")


def build_authenticated_git_url(git_url: str, token: Optional[str] = None) -> str:
    """
    Build a Git clone URL with embedded token authentication.

    Detects the Git provider from the URL hostname or token prefix and applies
    the appropriate authentication format.

    Args:
        git_url: The original HTTPS Git URL (e.g., https://github.com/owner/repo.git)
        token: Optional personal access token for authentication

    Returns:
        The URL with embedded credentials, or the original URL if no token provided
    """
    if not token:
        return git_url

    # Parse URL: protocol://host/path
    match = re.match(r"(https?://)([^/]+)(/.*)$", git_url)
    if not match:
        return git_url

    protocol, host, path = match.groups()
    host_lower = host.lower()

    # Priority 1: Detect by hostname
    if "github.com" in host_lower:
        return f"{protocol}x-access-token:{token}@{host}{path}"

    if "gitlab" in host_lower:
        return f"{protocol}oauth2:{token}@{host}{path}"

    if "bitbucket.org" in host_lower:
        return f"{protocol}x-bitbucket-api-token-auth:{token}@{host}{path}"

    # Priority 2: Detect by token prefix
    if token.startswith(_GITHUB_TOKEN_PREFIXES):
        return f"{protocol}x-access-token:{token}@{host}{path}"

    if token.startswith(_GITLAB_TOKEN_PREFIXES):
        return f"{protocol}oauth2:{token}@{host}{path}"

    # Priority 3: Generic fallback
    return f"{protocol}{token}@{host}{path}"


def compute_file_hash(file_path: Path, hash_algorithm: str = "sha256") -> str:
    """
    Compute hash of a file for change detection.

    Args:
        file_path: Path to the file
        hash_algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Hex digest of the file hash
    """
    h = hashlib.new(hash_algorithm)
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def matches_pattern(path: str, patterns: List[str]) -> bool:
    """
    Check if a path matches any of the given glob patterns.

    Handles both directory-prefixed patterns (**/) and simple patterns.

    Args:
        path: Relative path to check (forward slashes)
        patterns: List of glob patterns to match against

    Returns:
        True if path matches any pattern
    """
    for pattern in patterns:
        # Strip common prefixes for matching
        clean_pattern = pattern.removeprefix("**/").removeprefix("*/")
        if fnmatch.fnmatch(path, clean_pattern):
            return True
        # Also try the original pattern for exact matches
        if fnmatch.fnmatch(path, pattern):
            return True
    return False


def separate_patterns(
    patterns: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Separate patterns into filename patterns and path patterns.

    Filename patterns (*.js, *.min.js) match against the filename only.
    Path patterns (**/dir/**, etc.) match against the full relative path.

    Args:
        patterns: List of glob patterns

    Returns:
        Tuple of (filename_patterns, path_patterns)
    """
    filename_patterns = []
    path_patterns = []

    for pattern in patterns:
        # Pattern is a filename pattern if it starts with * but has no path separators
        if pattern.startswith("*") and "/" not in pattern and "\\" not in pattern:
            filename_patterns.append(pattern)
        else:
            path_patterns.append(pattern)

    return filename_patterns, path_patterns


def should_include_file(
    file_path: Path,
    base_path: Path,
    include_patterns: List[str],
    exclude_patterns: List[str],
    skip_minified: bool = True,
    max_file_size_bytes: int = 10 * 1024 * 1024,  # 10 MB default
) -> Tuple[bool, Optional[str]]:
    """
    Determine if a file should be included in indexing.

    Checks file against:
    - Include patterns (must match at least one)
    - Exclude patterns (must not match any)
    - Hardcoded excludes (always excluded)
    - Minified file detection (optional)
    - File size limits

    Args:
        file_path: Absolute path to the file
        base_path: Base directory for relative path calculation
        include_patterns: Glob patterns files must match
        exclude_patterns: Glob patterns that exclude files
        skip_minified: Whether to skip minified files
        max_file_size_bytes: Maximum file size to include

    Returns:
        Tuple of (should_include, skip_reason) where skip_reason is None if included
    """
    try:
        # Get relative path for pattern matching
        rel_path = file_path.relative_to(base_path)
        rel_path_str = str(rel_path)
    except ValueError:
        return False, "path_outside_base"

    filename = file_path.name

    # Check file size first (cheap check)
    try:
        size = file_path.stat().st_size
        if size == 0:
            return False, "empty_file"
        if size > max_file_size_bytes:
            return False, "file_too_large"
    except OSError:
        return False, "stat_failed"

    # Combine user excludes with hardcoded excludes
    all_excludes = list(exclude_patterns) + HARDCODED_EXCLUDES

    # Separate filename patterns (*.js) from path patterns (**/dir/**)
    filename_excludes, path_excludes = separate_patterns(all_excludes)

    # Add hardcoded minified patterns to filename excludes if skip_minified is enabled
    if skip_minified:
        filename_excludes = list(filename_excludes) + list(MINIFIED_PATTERNS)

    # Check filename-based excludes (extension patterns like *.min.js)
    for exc_pattern in filename_excludes:
        if fnmatch.fnmatch(filename, exc_pattern):
            return False, "matched_exclude"

    # Check path-based excludes (directory patterns like **/vendor/**)
    if matches_pattern(rel_path_str, path_excludes):
        return False, "matched_exclude"

    # Check include patterns (must match at least one)
    if include_patterns:
        matched = False
        for pattern in include_patterns:
            clean_pattern = pattern.removeprefix("**/").removeprefix("*/")
            if fnmatch.fnmatch(rel_path_str, clean_pattern):
                matched = True
                break
            if fnmatch.fnmatch(filename, clean_pattern):
                matched = True
                break
        if not matched:
            return False, "no_include_match"

    return True, None


def is_binary_file(file_path: Path) -> bool:
    """
    Check if a file appears to be binary based on extension.

    Args:
        file_path: Path to check

    Returns:
        True if the file has a known binary extension
    """
    return file_path.suffix.lower() in BINARY_EXTENSIONS


def is_parseable_document(file_path: Path) -> bool:
    """
    Check if a file is a parseable document (PDF, Office, etc.).

    Args:
        file_path: Path to check

    Returns:
        True if the file has a parseable document extension
    """
    return file_path.suffix.lower() in PARSEABLE_DOCUMENT_EXTENSIONS


def is_unparseable_binary(file_path: Path) -> bool:
    """
    Check if a file is truly unparseable (executables, archives, etc.).

    Args:
        file_path: Path to check

    Returns:
        True if the file should never be parsed
    """
    return file_path.suffix.lower() in UNPARSEABLE_BINARY_EXTENSIONS


def extract_archive(
    archive_path: Path,
    extract_dir: Path,
    max_total_size: int = 5 * 1024 * 1024 * 1024,  # 5 GB
    max_file_count: int = 100000,
) -> None:
    """
    Extract a zip or tar archive to a directory.

    Supports: .zip, .tar, .tar.gz, .tgz, .tar.bz2

    Args:
        archive_path: Path to the archive file
        extract_dir: Directory to extract into
        max_total_size: Maximum total extracted size (protection against zip bombs)
        max_file_count: Maximum number of files to extract

    Raises:
        ValueError: If archive format is unsupported or extraction limits exceeded
    """
    archive_name = archive_path.name.lower()
    total_size = 0
    file_count = 0
    extract_dir_resolved = extract_dir.resolve()

    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                file_count += 1
                total_size += info.file_size
                if file_count > max_file_count:
                    raise ValueError(
                        f"Archive contains too many files (max: {max_file_count})"
                    )
                if total_size > max_total_size:
                    raise ValueError(
                        f"Archive too large when extracted (max: {max_total_size // (1024*1024*1024)} GB)"
                    )
                # Zip Slip protection: reject entries that escape extract_dir
                target = (extract_dir_resolved / info.filename).resolve()
                if not target.is_relative_to(extract_dir_resolved):
                    raise ValueError(
                        f"Archive member would escape target directory: {info.filename}"
                    )
            zf.extractall(extract_dir)

    elif archive_name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
        mode = "r:*"  # Auto-detect compression
        with tarfile.open(archive_path, mode) as tf:
            for member in tf.getmembers():
                file_count += 1
                total_size += member.size
                if file_count > max_file_count:
                    raise ValueError(
                        f"Archive contains too many files (max: {max_file_count})"
                    )
                if total_size > max_total_size:
                    raise ValueError(
                        f"Archive too large when extracted (max: {max_total_size // (1024*1024*1024)} GB)"
                    )
            # Use filter='data' to strip path traversal, absolute paths,
            # and dangerous special members (Python 3.12+).
            tf.extractall(extract_dir, filter="data")

    else:
        raise ValueError(
            f"Unsupported archive format: {archive_path.suffix}. "
            "Supported: .zip, .tar, .tar.gz, .tgz, .tar.bz2"
        )

    logger.info(
        f"Extracted {file_count} files ({total_size // (1024*1024)} MB) to {extract_dir}"
    )


def find_source_dir(extract_dir: Path) -> Path:
    """
    Find the actual source directory after archive extraction.

    Handles cases where archives contain a single top-level directory.

    Args:
        extract_dir: Directory where archive was extracted

    Returns:
        Path to the actual source directory
    """
    contents = list(extract_dir.iterdir())

    # If there's exactly one directory and no files, use that directory
    if len(contents) == 1 and contents[0].is_dir():
        nested_dir = contents[0]
        # Check if the nested directory contains the actual source
        nested_contents = list(nested_dir.iterdir())
        if nested_contents:
            logger.info(f"Using nested directory: {nested_dir.name}")
            return nested_dir

    return extract_dir


def collect_files_recursive(
    base_path: Path,
    include_patterns: List[str],
    exclude_patterns: List[str],
    max_files: int = 100000,
    skip_minified: bool = True,
    max_file_size_bytes: int = 10 * 1024 * 1024,
    follow_symlinks: bool = False,
) -> List[Tuple[Path, int]]:
    """
    Recursively collect files matching the given patterns.

    Args:
        base_path: Base directory to scan
        include_patterns: Glob patterns files must match
        exclude_patterns: Glob patterns that exclude files
        max_files: Maximum number of files to collect
        skip_minified: Whether to skip minified files
        max_file_size_bytes: Maximum file size to include
        follow_symlinks: Whether to follow symbolic links

    Returns:
        List of (file_path, size) tuples
    """
    results = []
    all_excludes = list(exclude_patterns) + HARDCODED_EXCLUDES

    # Separate filename patterns (*.js) from path patterns (**/dir/**)
    filename_excludes, path_excludes = separate_patterns(all_excludes)

    # Add hardcoded minified patterns to filename excludes if skip_minified is enabled
    if skip_minified:
        filename_excludes = list(filename_excludes) + list(MINIFIED_PATTERNS)

    for pattern in include_patterns:
        # Normalize pattern for rglob
        glob_pattern = pattern.removeprefix("**/").removeprefix("*/")

        for file_path in base_path.rglob(glob_pattern):
            if len(results) >= max_files:
                logger.warning(f"Reached max file limit ({max_files})")
                return results

            if not file_path.is_file():
                continue

            # Skip symlinks if not following them
            if not follow_symlinks and file_path.is_symlink():
                continue

            try:
                rel_path = file_path.relative_to(base_path)
                rel_path_str = str(rel_path)
            except ValueError:
                continue

            filename = file_path.name

            # Check filename-based excludes (extension patterns like *.min.js)
            if any(fnmatch.fnmatch(filename, exc) for exc in filename_excludes):
                continue

            # Check path-based excludes (directory patterns like **/vendor/**)
            if matches_pattern(rel_path_str, path_excludes):
                continue

            # Check size
            try:
                size = file_path.stat().st_size
                if size == 0 or size > max_file_size_bytes:
                    continue
            except OSError:
                continue

            # Avoid duplicates
            if file_path not in [r[0] for r in results]:
                results.append((file_path, size))

    return results
