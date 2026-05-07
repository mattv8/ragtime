"""
Shared file utilities for indexing services.

This module extracts common file handling functionality used by both
IndexerService (upload/git) and FilesystemIndexerService (local/SMB/NFS).
"""

import hashlib
import os
import re
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import fnmatch
import tarfile

from ragtime.core.file_constants import (
    MINIFIED_PATTERNS,
    OCR_EXTENSIONS,
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
    hasher = hashlib.new(hash_algorithm)
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_directory_size_bytes(path: Path) -> int:
    """Return total size of regular files under a directory using os.scandir."""
    total = 0
    stack = [path]

    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue

    return total


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
    filename = Path(path).name
    return get_matching_pattern(path, filename, patterns) is not None


def get_matching_pattern(
    rel_path: str,
    filename: str,
    patterns: List[str],
) -> Optional[str]:
    """Return the first pattern matching either a relative path or filename."""
    for pattern in patterns:
        clean_pattern = pattern.removeprefix("**/").removeprefix("*/")
        if fnmatch.fnmatch(rel_path, clean_pattern):
            return pattern
        if fnmatch.fnmatch(filename, clean_pattern):
            return pattern
        if fnmatch.fnmatch(rel_path, pattern):
            return pattern
        if fnmatch.fnmatch(filename, pattern):
            return pattern
    return None


def get_matching_file_pattern(
    file_path: Path,
    base_path: Path,
    patterns: List[str],
) -> Optional[str]:
    """Return the first pattern matching a file relative to a base path."""
    rel_path = file_path.relative_to(base_path).as_posix()
    return get_matching_pattern(rel_path, file_path.name, patterns)


def is_excluded_by_patterns(
    file_path: Path,
    base_path: Path,
    exclude_patterns: List[str],
    *,
    skip_minified: bool = True,
    include_hardcoded: bool = True,
) -> bool:
    """Check user, hardcoded, and optional minified excludes for a file."""
    patterns = list(exclude_patterns)
    if include_hardcoded:
        patterns.extend(HARDCODED_EXCLUDES)
    if skip_minified:
        patterns.extend(MINIFIED_PATTERNS)
    return get_matching_file_pattern(file_path, base_path, patterns) is not None


def is_excluded_directory(
    dir_path: Path,
    base_path: Path,
    exclude_patterns: List[str],
    *,
    include_hardcoded: bool = True,
) -> bool:
    """Check whether a directory should be pruned before descending into it."""
    patterns = list(exclude_patterns)
    if include_hardcoded:
        patterns.extend(HARDCODED_EXCLUDES)
    rel_path = dir_path.relative_to(base_path).as_posix()
    return (
        get_matching_pattern(rel_path, dir_path.name, patterns) is not None
        or get_matching_pattern(f"{rel_path}/", dir_path.name, patterns) is not None
    )


def has_binary_content(file_path: Path, sample_size: int = 8192) -> bool:
    """Check whether file content appears binary using a small byte sample."""
    try:
        with file_path.open("rb") as handle:
            sample = handle.read(sample_size)
    except OSError:
        return True

    if not sample:
        return False

    for bom, encoding in (
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    ):
        if sample.startswith(bom):
            try:
                sample.decode(encoding)
                return False
            except UnicodeDecodeError:
                break

    if b"\0" in sample:
        return True

    try:
        sample.decode("utf-8")
        return False
    except UnicodeDecodeError:
        pass

    text_bytes = set(range(32, 127)) | {8, 9, 10, 12, 13}
    non_text_bytes = sum(byte not in text_bytes for byte in sample)
    return (non_text_bytes / len(sample)) > 0.30


def should_index_file_type(
    file_path: Path,
    *,
    matches_include_pattern: bool,
    ocr_enabled: bool,
) -> bool:
    """
    Decide whether a file's type/content is eligible for indexing.

    The extension taxonomy lives in file_constants.py. This helper combines that
    taxonomy with a content sniff so explicit extension lists do not hide
    text-like files such as .plist files or extensionless scripts.
    """
    suffix = file_path.suffix.lower()

    if suffix in OCR_EXTENSIONS:
        return ocr_enabled

    if suffix in UNPARSEABLE_BINARY_EXTENSIONS:
        return False

    if suffix in PARSEABLE_DOCUMENT_EXTENSIONS:
        return matches_include_pattern

    return not has_binary_content(file_path)


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
    ocr_enabled: bool = False,
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
    results: List[Tuple[Path, int]] = []

    for dirpath, dirnames, filenames in os.walk(base_path, followlinks=follow_symlinks):
        current_dir = Path(dirpath)
        if not follow_symlinks:
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not (current_dir / dirname).is_symlink()
            ]
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not is_excluded_directory(
                current_dir / dirname,
                base_path,
                exclude_patterns,
            )
        ]

        for filename in filenames:
            if len(results) >= max_files:
                logger.warning(f"Reached max file limit ({max_files})")
                return results

            file_path = current_dir / filename

            if not file_path.is_file():
                continue

            if not follow_symlinks and file_path.is_symlink():
                continue

            if is_excluded_by_patterns(
                file_path,
                base_path,
                exclude_patterns,
                skip_minified=skip_minified,
            ):
                continue

            try:
                size = file_path.stat().st_size
                if size == 0 or size > max_file_size_bytes:
                    continue
            except OSError:
                continue

            matches_include = (
                get_matching_file_pattern(file_path, base_path, include_patterns)
                is not None
            )
            if not should_index_file_type(
                file_path,
                matches_include_pattern=matches_include,
                ocr_enabled=ocr_enabled,
            ):
                continue

            results.append((file_path, size))

    return results
