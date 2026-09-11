"""
CPU-optimized document chunking with semantic/language-aware splitting.

This module uses Chonkie for all text chunking:
1. `CodeChunker` with `language="auto"` for code files
   - Uses Magika (Google's ML model) for language detection
   - AST-based splitting via tree-sitter for semantic boundaries
   - Supports: Python, TypeScript, JavaScript, Rust, Go, Java, C, C++, C#, etc.
2. `RecursiveChunker` for plain text and documents
   - Delimiter-based splitting that respects paragraph/sentence boundaries
   - Used when CodeChunker can't detect a supported programming language

Text extraction from documents (PDF, DOCX, images, etc.) is handled by
document_parser.py BEFORE this module is called. This module only chunks text.
"""

import asyncio
import contextlib
import hashlib
import json
import multiprocessing
import os
import threading
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, get_args

from chonkie.chunker.code import CodeChunker
from chonkie.chunker.recursive import RecursiveChunker
from chonkie.refinery.overlap import OverlapRefinery
from chonkie.types import RecursiveLevel, RecursiveRules
from langchain_core.documents import Document

from ragtime.core.app_setting_defaults import (
    DEFAULT_CHUNKING_MAX_BATCH_SIZE,
    DEFAULT_CHUNKING_MAX_WORKERS,
)
from ragtime.core.file_constants import (
    ANYDOC_DOCUMENT_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    LANG_MAPPING,
)
from ragtime.core.logging import get_logger
from ragtime.core.tokenization import count_tokens
from ragtime.indexer.embedding_errors import iter_exception_chain
from ragtime.indexer.indexing_spool import SpoolTaskOutput
from ragtime.indexer.resource_governor import ResourceRequest, resource_governor
from ragtime.indexer.resource_workers import ResourceTaskError, run_resource_task

# Suppress Chonkie warnings we intentionally trigger:
# - tokenizers library: we use tiktoken intentionally
# - auto language: we use auto-detection as fallback when extension not mapped
warnings.filterwarnings(
    "ignore",
    message="'tokenizers' library not found",
    module="chonkie.tokenizer",
)
warnings.filterwarnings(
    "ignore",
    message="The language is set to `auto`",
    module="chonkie.chunker.code",
)

logger = get_logger(__name__)

# Tiktoken encoding for token-based chunking
# cl100k_base is used by GPT-4, text-embedding-3-*, and is a good general-purpose
# tokenizer that roughly matches most embedding model tokenization
TIKTOKEN_ENCODING = "cl100k_base"
CHUNKING_PIPELINE_SCHEMA_VERSION = 2


def chunking_implementation_fingerprint() -> dict[str, str | int]:
    """Return the parser versions that define persisted chunk output."""

    def package_version(distribution: str) -> str:
        try:
            return version(distribution)
        except PackageNotFoundError:
            return "not-installed"

    return {
        "pipeline_schema_version": CHUNKING_PIPELINE_SCHEMA_VERSION,
        "chonkie": package_version("chonkie"),
        "tree_sitter_language_pack": package_version("tree-sitter-language-pack"),
    }


# Pool sizing caps. Defaults preserve historical behavior; the active values
# can be overridden at runtime by `configure_chunking_pool()` (typically driven
# by app settings).
_GIB = 1024 * 1024 * 1024
_CHUNKING_WORKERS_HARD_CEILING = 16
_CHUNKING_BATCH_SIZE_HARD_CEILING = 500
_configured_max_workers: int = DEFAULT_CHUNKING_MAX_WORKERS
_configured_max_batch_size: int = DEFAULT_CHUNKING_MAX_BATCH_SIZE

# Sentinel pool key for callers that have no natural job identity (e.g. chat
# attachment chunking, rechunk fallbacks). Multiple callers may share this
# pool concurrently; per-job callers should pass their own key.
SHARED_CHUNKING_POOL_KEY = "__shared__"

# Time we give workers to drain after SIGTERM before escalating to SIGKILL.
# Chonkie's Rust-backed tree-sitter language detection does not unwind
# cleanly on SIGTERM, so we always need an escalation path.
_WORKER_SIGTERM_GRACE_SECONDS = 0.5
_WORKER_SIGKILL_GRACE_SECONDS = 0.5

# Poll interval for detecting pool closure while awaiting in-flight futures.
# Smaller values react faster to cancellation but add wakeup overhead.
_POOL_CLOSED_POLL_SECONDS = 0.5

# Bound how long one submitted wave may wait before falling back to recursive
# chunking for that entire wave and recreating the per-job pool.
_CHUNKING_WAVE_TIMEOUT_SECONDS = 300.0


class ChunkingPoolError(RuntimeError):
    """Raised when a chunking pool is unavailable (closed, terminated, or its
    owning job was cancelled)."""


@dataclass
class ChunkingPool:
    """A single chunking process pool owned by a job (or the shared pool)."""

    key: str
    executor: Any
    max_workers: int
    closed: threading.Event = field(default_factory=threading.Event)

    def is_closed(self) -> bool:
        return self.closed.is_set()


class ChunkingPoolManager:
    """Registry of active chunking pools keyed by job/pool identity.

    Replaces the previous global `_process_pool` singleton so that:
      * Per-job pools give each indexing job its own workers; cancelling or
        terminating one job never leaves another job's `asyncio.gather()`
        blocked on orphaned futures.
      * Chat-attach and rechunk-fallback callers share the sentinel
        `SHARED_CHUNKING_POOL_KEY` while still benefiting from graceful
        termination when the app shuts down.
      * The manager owns the SIGTERM -> SIGKILL escalation so workers that
        ignore SIGTERM (Chonkie's Rust code) cannot outlive their pool.
    """

    def __init__(self) -> None:
        self._pools: Dict[str, ChunkingPool] = {}
        self._lock = threading.Lock()

    def get_or_create(self, key: str, max_workers: int) -> ChunkingPool:
        # Per-job executors were the source of the indexing OOM incident.  New
        # work must go through run_resource_task(), where the process-wide
        # governor owns every live child.  Keep this registry's release APIs
        # for shutdown compatibility, but do not let a stale caller create a
        # policy-bypassing pool.
        raise ChunkingPoolError("private chunking pools are retired; use run_resource_task or chunk_spooled_batch")
        # Kept below temporarily for source compatibility with old tracebacks;
        # it is unreachable and can be removed with the next pool API cleanup.
        with self._lock:
            existing = self._pools.get(key)
            if existing is not None:
                if existing.is_closed():
                    self._pools.pop(key, None)
                else:
                    return existing

            executor = ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            pool = ChunkingPool(key=key, executor=executor, max_workers=max_workers)
            self._pools[key] = pool
            logger.info(f"Created chunking pool '{key}': {max_workers} workers (cap={_configured_max_workers}, batch_cap={_configured_max_batch_size})")
            return pool

    def get(self, key: str) -> Optional[ChunkingPool]:
        with self._lock:
            return self._pools.get(key)

    def release(self, key: str, *, terminate_workers: bool = True) -> bool:
        """Terminate the pool for `key` and remove it from the registry.

        Returns True if a pool was released, False if none existed.
        """
        with self._lock:
            pool = self._pools.pop(key, None)
        if pool is None:
            return False
        _terminate_pool(pool, terminate_workers=terminate_workers)
        return True

    def shutdown_all(self, *, terminate_workers: bool = True) -> None:
        """Terminate every registered pool. Used by app lifespan shutdown."""
        with self._lock:
            pools = list(self._pools.values())
            self._pools.clear()
        for pool in pools:
            _terminate_pool(pool, terminate_workers=terminate_workers)
        if pools:
            logger.info(f"Chunking pool manager shut down {len(pools)} pool(s)")

    def active_keys(self) -> List[str]:
        with self._lock:
            return [key for key, pool in self._pools.items() if not pool.is_closed()]


# Module-level singleton. Imported by callers via `pool_manager` below.
pool_manager = ChunkingPoolManager()


def configure_chunking_pool(
    max_workers: int | None = None,
    max_batch_size: int | None = None,
) -> None:
    """Apply runtime caps for chunking pools.

    Values are clamped to safe ranges. If `max_workers` changes, all existing
    pools are released so they are recreated lazily with the new size on next
    use. Safe to call repeatedly (e.g. on settings reload).
    """
    global _configured_max_workers, _configured_max_batch_size

    resized = False
    if max_workers is not None:
        clamped_workers = max(1, min(int(max_workers), _CHUNKING_WORKERS_HARD_CEILING))
        if clamped_workers != _configured_max_workers:
            _configured_max_workers = clamped_workers
            resized = True

    if max_batch_size is not None:
        _configured_max_batch_size = max(1, min(int(max_batch_size), _CHUNKING_BATCH_SIZE_HARD_CEILING))

    if resized:
        pool_manager.shutdown_all(terminate_workers=False)


def _read_cgroup_int(path: str) -> int | None:
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    # Docker sometimes reports host-ish sentinel values for unlimited memory.
    if parsed <= 0 or parsed > 1_000 * _GIB:
        return None
    return parsed


def _get_cgroup_memory_limit_bytes() -> int | None:
    """Return the container memory limit when cgroups expose one."""
    return _read_cgroup_int("/sys/fs/cgroup/memory.max") or _read_cgroup_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")


def _get_effective_memory_limit_bytes() -> int | None:
    cgroup_limit = _get_cgroup_memory_limit_bytes()
    if cgroup_limit is not None:
        return cgroup_limit
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * page_count)
    except (OSError, ValueError):
        return None


def _resolve_worker_count(cpu_count: int) -> int:
    """Return the worker count for a new pool, capped by configured ceiling."""
    cpu_limited = max(1, cpu_count // 4)
    return max(1, min(cpu_limited, _configured_max_workers))


def _effective_batch_size(requested_batch_size: int) -> int:
    return max(1, min(int(requested_batch_size), _configured_max_batch_size))


def _resolve_max_workers() -> tuple[int, int, int | None]:
    """Return (max_workers, cpu_count, memory_limit_bytes) for a new pool."""
    cpu_count = os.cpu_count() or 2
    return _resolve_worker_count(cpu_count), cpu_count, _get_effective_memory_limit_bytes()


def _pool_worker_processes(pool: ChunkingPool) -> list:
    """Return the live multiprocessing.Process objects for `pool`'s executor."""
    return list((getattr(pool.executor, "_processes", None) or {}).values())


def _terminate_pool(pool: ChunkingPool, *, terminate_workers: bool) -> None:
    """Tear down a pool's executor with SIGTERM-then-SIGKILL escalation.

    Chonkie's Rust-backed tree-sitter language detection does not unwind
    cleanly on SIGTERM, so a 0.5s grace window followed by SIGKILL is the
    only reliable way to release those workers.
    """
    if pool.closed.is_set():
        return
    pool.closed.set()

    processes = _pool_worker_processes(pool) if terminate_workers else []

    if terminate_workers:
        for process in processes:
            if getattr(process, "is_alive", lambda: False)():
                with contextlib.suppress(Exception):
                    process.terminate()

    try:
        pool.executor.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"Pool '{pool.key}' shutdown raised (ignored): {exc}")

    if terminate_workers:
        deadline = _WORKER_SIGTERM_GRACE_SECONDS
        for process in processes:
            if not getattr(process, "is_alive", lambda: False)():
                continue
            try:
                process.join(timeout=deadline)
            except Exception:
                pass
            if getattr(process, "is_alive", lambda: False)():
                with contextlib.suppress(Exception):
                    process.kill()
            with contextlib.suppress(Exception):
                process.join(timeout=_WORKER_SIGKILL_GRACE_SECONDS)
        survivors = [p for p in processes if getattr(p, "is_alive", lambda: False)()]
        if survivors:
            logger.warning(f"Chunking pool '{pool.key}': {len(survivors)} worker(s) still alive after SIGKILL; they may need parent reap")
        else:
            logger.info(f"Chunking pool '{pool.key}' shut down (workers terminated)")


# ---- Backwards-compatible module-level helpers ----------------------------
# `_get_process_pool` and `shutdown_process_pool` are kept as thin shims so
# existing callers (and tests that patched `_process_pool`) keep working.
# Indexing jobs should now pass an explicit `pool_key` to `chunk_documents_parallel`
# and release their pool via `pool_manager.release(job.id)` instead of relying
# on the global singleton.


def _get_process_pool() -> ProcessPoolExecutor:
    """Retired private pool factory; global admission is asynchronous."""
    raise ChunkingPoolError("_get_process_pool is retired; use chunk_documents_parallel")


def shutdown_process_pool(
    wait: bool = True,  # noqa: ARG001 - kept for backwards compatibility
    cancel_futures: bool = True,  # noqa: ARG001 - kept for backwards compatibility
    terminate_workers: bool = True,
) -> None:
    """Shut down every chunking pool. Used by app lifespan shutdown.

    `wait` and `cancel_futures` are accepted for back-compat with previous
    call sites but the manager always uses the safest behaviour (cancel
    pending + terminate workers).
    """
    pool_manager.shutdown_all(terminate_workers=terminate_workers)


# =============================================================================
# CODE CONTEXT EXTRACTION
# =============================================================================

# NOTE: Context extraction is handled by ragtime.indexer.code_extraction
# to leverage tree-sitter parsers instead of regex heuristics.


def _create_file_summary(
    file_path: str,
    imports: list[str],
    definitions: list[str],
    total_chunks: int,
) -> str:
    """
    Create a file-level summary chunk for hierarchical retrieval.

    This summary chunk provides an overview of the file contents, helping
    retrieval find the right file before drilling down into specific chunks.

    Args:
        file_path: Relative path to the source file
        imports: List of import statements
        definitions: List of top-level definitions
        total_chunks: Total number of chunks for this file

    Returns:
        Summary content string
    """
    lines = [
        f"# File Summary: {file_path}",
        f"# This file has {total_chunks} code chunks",
        "",
    ]

    if imports:
        lines.append("## Dependencies:")
        for imp in imports[:15]:
            lines.append(f"- {imp}")
        if len(imports) > 15:
            lines.append(f"- ... and {len(imports) - 15} more imports")
        lines.append("")

    if definitions:
        lines.append("## Definitions:")
        for defn in definitions[:25]:
            lines.append(f"- {defn}")
        if len(definitions) > 25:
            lines.append(f"- ... and {len(definitions) - 25} more definitions")
        lines.append("")

    return "\n".join(lines)


def _create_chunk_header(
    file_path: str,
    imports: list[str] | None = None,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> str:
    """
    Create a header for code chunks with file context.

    This header helps the LLM understand the context of the code chunk,
    including which file it's from and what dependencies are used.

    Args:
        file_path: Relative path to the source file
        imports: List of import statements (for first chunk only)
        chunk_index: Index of this chunk (0-based)
        total_chunks: Total number of chunks for this file

    Returns:
        Header string to prepend to chunk content
    """
    lines = [f"# File: {file_path}"]

    if total_chunks > 1:
        lines.append(f"# Chunk {chunk_index + 1}/{total_chunks}")

    # Include imports only in first chunk (or when there's just one chunk)
    if imports and chunk_index == 0:
        lines.append("# Imports:")
        for imp in imports[:10]:  # Limit to first 10 imports
            lines.append(f"#   {imp}")
        if len(imports) > 10:
            lines.append(f"#   ... and {len(imports) - 10} more")

    return "\n".join(lines) + "\n\n"


def _estimate_max_header_tokens(
    file_path: str,
    imports: list[str] | None = None,
    use_tokens: bool = False,
) -> int:
    """
    Estimate the maximum header size to reserve space before chunking.

    This calculates the worst-case header size (first chunk with imports)
    so we can reduce the effective chunk_size and ensure chunks with
    headers never exceed the embedding model's context limit.

    Args:
        file_path: Relative path to the source file
        imports: List of import statements
        use_tokens: If True, return token count; otherwise character count

    Returns:
        Size to reserve for headers (tokens or characters)
    """
    # Generate worst-case header (first chunk with imports, assuming multi-chunk)
    sample_header = _create_chunk_header(file_path, imports, chunk_index=0, total_chunks=99)

    if use_tokens:
        from ragtime.core.tokenization import count_tokens

        return count_tokens(sample_header, TIKTOKEN_ENCODING)

    return len(sample_header)


# =============================================================================
# CHUNKING IMPLEMENTATIONS
# =============================================================================

# Cache for the tree-sitter language set
_treesitter_langs_cache: set[str] | None = None


def _get_treesitter_langs() -> set[str]:
    """Get the set of supported tree-sitter language names.

    Handles both old Literal[...] unions and newer type definitions by
    attempting __args__ access and gracefully degrading if unavailable.
    If the exact list can't be determined, returns empty set — Magika's
    auto-detection and fallback to RecursiveChunker handle unknown languages.
    """
    global _treesitter_langs_cache
    if _treesitter_langs_cache is None:
        try:
            from tree_sitter_language_pack import SupportedLanguage

            # Try __args__ for Literal union types (pre-2024 versions)
            try:
                _treesitter_langs_cache = set(get_args(SupportedLanguage))
            except AttributeError:
                # Newer versions: fall back to an empty set.
                # This is safe because:
                # 1. LANG_MAPPING covers common extensions
                # 2. Magika auto-detects code languages
                # 3. We fall back to RecursiveChunker if CodeChunker fails
                logger.debug("Could not determine SupportedLanguage set; relying on LANG_MAPPING and recursive fallback")
                _treesitter_langs_cache = set()
        except ImportError:
            _treesitter_langs_cache = set()
    return _treesitter_langs_cache


def _normalize_ext(file_ext: str) -> tuple[str, str]:
    """Return (with_dot, without_dot) variants of an extension for map lookups.

    LANG_MAPPING mixes keys with and without a leading dot (e.g. ``pdf`` is
    present but ``.pdf`` is not). The chunker derives ``file_ext`` with a
    leading dot, so lookups must consider both forms to find the entry.
    """
    if not file_ext:
        return "", ""
    if file_ext.startswith("."):
        return file_ext, file_ext[1:]
    return f".{file_ext}", file_ext


def _is_extension_mapped_to_plain_text(file_ext: str) -> bool:
    """Return True when the extension is explicitly mapped to ``None`` in LANG_MAPPING.

    Such extensions represent text-style content (``.txt``, ``.csv``, ``.log``)
    or parseable documents whose text has already been extracted by the
    document parser (``.pdf``, ``.doc``, ``.docx``). Code-aware chunking adds
    no value here and chonkie's tree-sitter language detection can be
    extremely slow on long extracted text — but PDFs/Office docs do still
    need semantic chunking, so the dispatcher falls back to
    ``_chunk_with_recursive`` (paragraph/sentence boundaries), not to
    skipping entirely.

    Truly binary files are filtered out earlier by ``has_binary_content``
    in ``ragtime.indexer.file_utils`` so they never reach the chunker.
    """
    if not file_ext:
        return False
    with_dot, no_dot = _normalize_ext(file_ext)
    if with_dot in LANG_MAPPING and LANG_MAPPING[with_dot] is None:
        return True
    if no_dot in LANG_MAPPING and LANG_MAPPING[no_dot] is None:
        return True
    return False


def _is_known_document_or_code_extension(file_ext: str) -> bool:
    """Return True for extensions Ragtime intentionally treats as text/code."""
    with_dot, _no_dot = _normalize_ext(file_ext)
    return bool(with_dot and with_dot in DOCUMENT_EXTENSIONS)


def _resolve_language(key: str) -> str | None | str:
    """
    Resolve a file extension, filename, or Magika content type to tree-sitter language.

    Uses LANG_MAPPING from file_constants.py as the single source of truth,
    with auto-mapping for the 59+ Magika types that exactly match tree-sitter names.
    Lookup is normalized so both ``.py`` and ``py`` keys resolve.

    Args:
        key: File extension (e.g., ".py"), filename (e.g., "makefile"),
             or Magika content type (e.g., "shell")

    Returns:
        - tree-sitter language name if mapped
        - None if content should use RecursiveChunker
        - "__unknown__" if no mapping exists
    """
    key_lower = key.lower()

    # Normalize the lookup to handle both ``.ext`` and ``ext`` keys in LANG_MAPPING.
    with_dot, no_dot = _normalize_ext(key)
    if with_dot and with_dot in LANG_MAPPING:
        return LANG_MAPPING[with_dot]
    if no_dot and no_dot in LANG_MAPPING:
        return LANG_MAPPING[no_dot]

    # Check if it auto-maps (exact name match with tree-sitter)
    # This handles the 59+ Magika types like "python", "javascript", "rust", etc.
    if key_lower in _get_treesitter_langs():
        return key_lower

    return "__unknown__"


def _chunk_with_chonkie_code(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict,
    use_tokens: bool = False,
) -> List[Document]:
    """
    Chunk code using Chonkie's AST-based CodeChunker with auto language detection.

    Uses Magika (Google's ML model) to detect language, then tree-sitter for
    AST-based splitting that respects semantic boundaries (functions, classes, etc.)

    Features:
    - Adds file path and import context as header in each chunk
    - Applies OverlapRefinery to add context from adjacent chunks
    - Preserves semantic boundaries (functions, classes, blocks)
    - When use_tokens=True, chunk_size is in tokens (not characters)

    Args:
        text: Source code text to chunk
        chunk_size: Maximum size per chunk (characters if use_tokens=False, tokens otherwise)
        chunk_overlap: Overlap context size (characters or tokens)
        metadata: Metadata dict to attach to chunks
        use_tokens: If True, use tiktoken-based chunking for accurate token counts
    """
    source_path = metadata.get("source", "")
    # Get extension or filename for extensionless files like Makefile, Dockerfile
    if "." in source_path.rsplit("/", 1)[-1]:  # Has extension
        file_ext = "." + source_path.rsplit(".", 1)[-1]
    else:
        file_ext = source_path.rsplit("/", 1)[-1]  # Use filename itself

    # Check if extension/filename is explicitly mapped to plain text (None)
    # in LANG_MAPPING. Such content (text files and extracted PDF / Office
    # text) is best handled by the RecursiveChunker which splits at
    # paragraph and sentence boundaries. The dispatcher below catches the
    # raised ValueError and falls back to recursive chunking — semantic
    # boundaries are preserved, just not via tree-sitter.
    plain_text = file_ext and _is_extension_mapped_to_plain_text(file_ext)
    if plain_text:
        raise ValueError(f"Extension {file_ext.lower()} mapped to plain text chunker")

    # Truly unmapped extensions (CAD .stp/.x_t/.igs, PostScript .eps, etc.)
    # have no tree-sitter grammar. Chonkie's auto-detect picks the wrong
    # "best match" language and the iteration is slow on long text.
    # Keep known source/document extensions on their normal path because
    # LANG_MAPPING intentionally only lists non-obvious code aliases.
    # RecursiveChunker splits unknown text at paragraph/sentence boundaries,
    # which is both faster and more accurate for non-code formats.
    if file_ext and not plain_text:
        if not _is_known_document_or_code_extension(file_ext):
            # Phrase carries the "mapped to plain text" sentinel so the
            # dispatcher's fallback pattern matches and we get the
            # normal RecursiveChunker path (paragraph/sentence boundaries)
            # rather than the last-resort error branch.
            raise ValueError(f"Extension {file_ext.lower()} mapped to plain text chunker (no tree-sitter grammar)")

    # Extract imports and definitions for context/summary using Tree-sitter
    imports: list[str] = []
    definitions: list[str] = []
    if file_ext:
        from ragtime.indexer.code_extraction import extract_metadata

        imports, definitions = extract_metadata(text, file_ext)

    # Determine tokenizer based on use_tokens setting
    tokenizer = TIKTOKEN_ENCODING if use_tokens else "character"

    # Reserve space for headers BEFORE chunking to ensure chunks with headers
    # never exceed the embedding model's context limit
    header_reserve = 0
    if source_path:
        header_reserve = _estimate_max_header_tokens(source_path, imports, use_tokens)
        # Add 10% safety margin for tokenizer variations
        header_reserve = int(header_reserve * 1.1)

    # Also account for overlap refinement which adds chunk_overlap to each chunk
    # The final chunk size = base_chunk + overlap + header
    total_reserve = header_reserve + chunk_overlap

    # Effective chunk size after reserving header and overlap space
    effective_chunk_size = max(100, chunk_size - total_reserve)  # Minimum 100

    # Skip chunking if content is already small enough
    # For token mode, estimate tokens as chars/4 for this quick check
    size_threshold = effective_chunk_size * 4 if use_tokens else effective_chunk_size
    if len(text) <= size_threshold:
        new_meta = metadata.copy()
        new_meta["chunker"] = "no_chunk_small"
        # Add file context header even for small files
        if source_path:
            header = _create_chunk_header(source_path, imports, 0, 1)
            return [Document(page_content=header + text, metadata=new_meta)]
        return [Document(page_content=text, metadata=new_meta)]

    # Try to determine the language for tree-sitter
    # Priority: 1) file extension/name mapping, 2) auto-detection with Magika
    language: str = "auto"
    if file_ext:
        ext_lang = _resolve_language(file_ext)
        if ext_lang not in (None, "__unknown__"):
            language = ext_lang  # type: ignore
        elif ext_lang is None:
            # Extension mapped to None = use RecursiveChunker
            raise ValueError(f"Extension {file_ext} should use RecursiveChunker")

    # Also check filename for special cases (Makefile, Dockerfile, etc.)
    if language == "auto" and source_path:
        filename = os.path.basename(source_path)
        filename_lang = _resolve_language(filename)
        if filename_lang not in (None, "__unknown__"):
            language = filename_lang  # type: ignore

    # Create chunker - may raise if language not supported
    chunker = CodeChunker(
        tokenizer,
        chunk_size=effective_chunk_size,
        language=language,
    )

    try:
        chunks = chunker.chunk(text)
    except (ValueError, RuntimeError, LookupError) as e:
        # Magika detected an unsupported language - check our mapping
        err_str = str(e).lower()
        if "could not find language" in err_str or "not supported" in err_str:
            # Extract the detected language from error message
            # Format: "Could not find language library for <lang>"
            import re

            match = re.search(r"for (\w+)", str(e))
            if match:
                detected_lang = match.group(1).lower()
                # Use unified language resolution
                mapped_lang = _resolve_language(detected_lang)

                if mapped_lang is None:
                    # None means use RecursiveChunker (plain text content)
                    raise
                elif mapped_lang == "__unknown__":
                    # No mapping - re-raise for fallback handling
                    raise
                else:
                    # Valid mapping found (manual or auto)
                    logger.debug(f"Mapping detected language '{detected_lang}' to '{mapped_lang}' for {source_path}")
                    chunker = CodeChunker(
                        tokenizer,
                        chunk_size=effective_chunk_size,
                        language=mapped_lang,
                    )
                    chunks = chunker.chunk(text)
            else:
                raise
        else:
            raise

    # Apply overlap to add context from adjacent chunks
    # This helps retrieval when function calls reference other functions
    if chunk_overlap > 0 and len(chunks) > 1:
        refinery = OverlapRefinery(
            tokenizer,
            context_size=chunk_overlap,
            mode="recursive",  # Use delimiter-aware overlap
            method="suffix",  # Add context from previous chunk
            merge=True,
            inplace=True,
        )
        chunks = refinery.refine(chunks)

    docs = []
    total_chunks = len(chunks)

    # For files with multiple chunks, add a summary chunk first (hierarchical)
    # This helps retrieval find the right file before drilling into details
    if total_chunks > 2 and source_path and file_ext:
        if definitions:  # Only add summary if we found definitions
            summary = _create_file_summary(source_path, imports, definitions, total_chunks)
            # Truncate summary if it exceeds chunk_size
            if use_tokens:
                from ragtime.core.tokenization import count_tokens

                summary_tokens = count_tokens(summary, TIKTOKEN_ENCODING)
                if summary_tokens > chunk_size:
                    # Truncate to fit - rough estimate
                    ratio = chunk_size / summary_tokens
                    summary = summary[: int(len(summary) * ratio * 0.9)]
            elif len(summary) > chunk_size:
                summary = summary[: int(chunk_size * 0.9)]

            summary_meta = metadata.copy()
            summary_meta["chunker"] = "chonkie_code_summary"
            summary_meta["chunk_index"] = -1  # Special index for summary
            summary_meta["total_chunks"] = total_chunks
            summary_meta["is_summary"] = True
            docs.append(Document(page_content=summary, metadata=summary_meta))

    for i, c in enumerate(chunks):
        new_meta = metadata.copy()
        new_meta["chunker"] = "chonkie_code"
        new_meta["chunk_index"] = i
        new_meta["total_chunks"] = total_chunks

        # Add file context header
        if source_path:
            header = _create_chunk_header(source_path, imports, i, total_chunks)
            content = header + c.text
        else:
            content = c.text

        docs.append(Document(page_content=content, metadata=new_meta))
    return docs


# Extensions whose extracted or raw content is GitHub-Flavored Markdown:
# AnyDoc-converted documents plus native markdown files.
MARKDOWN_STRUCTURED_EXTENSIONS: frozenset[str] = frozenset(ANYDOC_DOCUMENT_EXTENSIONS) | {
    ".md",
    ".markdown",
}


def _markdown_recursive_rules() -> RecursiveRules:
    """Heading-first recursive splitting rules for markdown-structured text.

    Split at ATX heading boundaries first (keeping each heading attached to
    the section it introduces via include_delim="next"), then fall back to
    the paragraph/newline/sentence hierarchy of the default rules.
    """
    return RecursiveRules(
        levels=[
            # Known limitation: heading delimiters can also match "# " lines
            # inside fenced code blocks of native .md files, splitting the
            # fence. AnyDoc-extracted documents rarely contain fences, and a
            # split only occurs when a section already exceeds chunk_size.
            RecursiveLevel(
                delimiters=["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### "],
                include_delim="next",
            ),
            RecursiveLevel(delimiters=["\n\n", "\r\n\r\n"]),
            RecursiveLevel(delimiters=["\n", "\r\n"]),
            RecursiveLevel(delimiters=[". ", "! ", "? "]),
            RecursiveLevel(whitespace=True),
            RecursiveLevel(),
        ]
    )


def _is_markdown_structured_source(source_path: str) -> bool:
    """True when the chunk source's extension carries markdown-structured text."""
    name = source_path.rsplit("/", 1)[-1]
    if "." not in name:
        return False
    return ("." + name.rsplit(".", 1)[-1]).lower() in MARKDOWN_STRUCTURED_EXTENSIONS


def _chunk_with_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,  # noqa: ARG001
    metadata: dict,
    use_tokens: bool = False,
) -> List[Document]:
    """Chunk plain text using Chonkie's RecursiveChunker.

    Note: chunk_overlap is accepted for API compatibility but Chonkie's
    RecursiveChunker uses delimiter-based splitting rather than overlap.

    Args:
        text: Text content to chunk
        chunk_size: Maximum size per chunk (characters if use_tokens=False, tokens otherwise)
        chunk_overlap: Unused, kept for API compatibility
        metadata: Metadata dict to attach to chunks
        use_tokens: If True, use tiktoken-based chunking for accurate token counts
    """
    # Determine tokenizer based on use_tokens setting
    tokenizer = TIKTOKEN_ENCODING if use_tokens else "character"

    # Skip chunking if content is already small enough
    # For token mode, estimate tokens as chars/4 for this quick check
    size_threshold = chunk_size * 4 if use_tokens else chunk_size
    if len(text) <= size_threshold:
        new_meta = metadata.copy()
        new_meta["chunker"] = "no_chunk_small"
        return [Document(page_content=text, metadata=new_meta)]

    # min_characters_per_chunk needs adjustment for token mode
    min_chars = 20 if use_tokens else 50

    # Branch on markdown-structured source for heading-aware splitting
    source_path = metadata.get("source", "")
    is_markdown = _is_markdown_structured_source(source_path)

    if is_markdown:
        chunker = RecursiveChunker(
            tokenizer,
            chunk_size=chunk_size,
            rules=_markdown_recursive_rules(),
            min_characters_per_chunk=min_chars,
        )
        chunker_tag = "chonkie_recursive_markdown"
    else:
        chunker = RecursiveChunker(
            tokenizer,
            chunk_size=chunk_size,
            min_characters_per_chunk=min_chars,
        )
        chunker_tag = "chonkie_recursive"

    chunks = chunker.chunk(text)

    docs = []
    for c in chunks:
        if not c.text.strip():
            continue
        new_meta = metadata.copy()
        new_meta["chunker"] = chunker_tag
        docs.append(Document(page_content=c.text, metadata=new_meta))
    return docs


def chunk_semantic_segments(
    segments: list[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict,
) -> list[Document]:
    """
    Chunk content by semantic segments, keeping related content together.

    Each segment is chunked independently. Segments smaller than chunk_size
    stay as single chunks. Larger segments use RecursiveChunker but won't
    cross segment boundaries.

    This is ideal for vision OCR output where we want to keep:
    - OCR text separate from classification
    - Classification metadata (description + tags) always together

    Args:
        segments: List of (segment_type, content) tuples
        chunk_size: Max characters per chunk
        chunk_overlap: Character overlap between chunks (for large segments)
        metadata: Base metadata to attach to each chunk

    Returns:
        List of Document objects with semantic chunking
    """
    docs = []
    chunk_index = 0

    for segment_type, content in segments:
        if not content:
            continue

        # Create segment-specific metadata
        seg_meta = metadata.copy()
        seg_meta["segment_type"] = segment_type

        # If segment fits in one chunk, don't split it
        if len(content) <= chunk_size:
            seg_meta["chunker"] = "semantic_single"
            seg_meta["chunk_index"] = chunk_index
            docs.append(Document(page_content=content, metadata=seg_meta))
            chunk_index += 1
        else:
            # Large segment - use RecursiveChunker but only within this segment
            segment_docs = _chunk_with_recursive(content, chunk_size, chunk_overlap, seg_meta)
            for doc in segment_docs:
                doc.metadata["chunk_index"] = chunk_index
                # Intentional overwrite: semantic-segment chunking owns the tag.
                # These segments carry vision/OCR text, not AnyDoc markdown, even
                # when the source file extension is a document format.
                doc.metadata["chunker"] = "semantic_recursive"
                docs.append(doc)
                chunk_index += 1

    return docs


def is_recursive_fallback_error(error: Exception) -> bool:
    """True when a chunking error signals falling back to RecursiveChunker.

    Covers Chonkie/Magika "language not supported" errors and the intentional
    plain-text sentinel raised by _chunk_with_chonkie_code for extensions
    mapped to plain-text chunking.
    """
    err_lower = str(error).lower()
    return (
        "not supported" in err_lower
        or "detected language" in err_lower
        or "could not find language" in err_lower
        or "mapped to plain text" in err_lower
        or "should use recursivechunker" in err_lower
    )


def _chunk_document_batch_sync(
    batch_data: List[Tuple[str, dict]],
    chunk_size: int,
    chunk_overlap: int,
    use_tokens: bool,
) -> Tuple[List[Tuple[str, dict]], Dict[str, int]]:
    """
    Synchronous worker function to chunk a batch of documents.

    Strategy:
    1. Try Chonkie CodeChunker with auto language detection (Magika)
    2. If language not supported, fall back to RecursiveChunker

    Args:
        batch_data: List of (content, metadata) tuples to chunk
        chunk_size: Maximum chunk size (tokens if use_tokens=True, else characters)
        chunk_overlap: Overlap context size (tokens or characters)
        use_tokens: If True, use tiktoken-based chunking for accurate token counts

    Note: Text extraction from documents (PDF, DOCX, images) happens in
    document_parser.py before this function is called. We only chunk text here.
    """
    all_chunks = []
    splitter_counts: Dict[str, int] = {}

    for content, metadata in batch_data:
        file_path = metadata.get("source", "")
        docs = []

        try:
            # Try Chonkie CodeChunker with auto language detection
            try:
                docs = _chunk_with_chonkie_code(content, chunk_size, chunk_overlap, metadata, use_tokens)
                splitter_counts["chonkie_code"] = splitter_counts.get("chonkie_code", 0) + 1
            except (ValueError, RuntimeError, LookupError) as e:
                # Expected cases for falling back to RecursiveChunker:
                # - Extension explicitly mapped to plain text (e.g., .txt, .csv)
                # - Magika couldn't detect a supported language
                # - tree-sitter grammar not available
                if is_recursive_fallback_error(e):
                    logger.debug(f"Code chunking not available for {file_path}, using recursive: {e}")
                    docs = _chunk_with_recursive(content, chunk_size, chunk_overlap, metadata, use_tokens)
                    # Count markdown-aware recursive chunking separately so log
                    # summaries stay accurate; all other outcomes keep the
                    # historical "chonkie_recursive" key.
                    if docs and docs[0].metadata.get("chunker") == "chonkie_recursive_markdown":
                        recursive_key = "chonkie_recursive_markdown"
                    else:
                        recursive_key = "chonkie_recursive"
                    splitter_counts[recursive_key] = splitter_counts.get(recursive_key, 0) + 1
                else:
                    raise

        except Exception as e:
            logger.error(f"Chunking failed for {file_path or 'unknown'}: {e}")
            # Last resort: simple recursive text splitting
            docs = _chunk_with_recursive(content, chunk_size, chunk_overlap, metadata, use_tokens)
            splitter_counts["chonkie_recursive_error"] = splitter_counts.get("chonkie_recursive_error", 0) + 1

        for doc in docs:
            all_chunks.append((doc.page_content, doc.metadata))

    return all_chunks, splitter_counts


def _chunk_document_batch_recursive_sync(
    batch_data: List[Tuple[str, dict]],
    chunk_size: int,
    chunk_overlap: int,
    use_tokens: bool,
) -> Tuple[List[Tuple[str, dict]], Dict[str, int]]:
    """Chunk a batch without CodeChunker after a worker crash."""
    all_chunks = []
    splitter_counts: Dict[str, int] = {}

    for content, metadata in batch_data:
        file_path = metadata.get("source", "unknown")
        try:
            docs = _chunk_with_recursive(content, chunk_size, chunk_overlap, metadata, use_tokens)
            splitter_counts["chonkie_recursive_pool_fallback"] = splitter_counts.get("chonkie_recursive_pool_fallback", 0) + 1
        except Exception as e:
            logger.error(f"Recursive fallback chunking failed for {file_path}: {e}")
            fallback_meta = metadata.copy()
            fallback_meta["chunker"] = "whole_document_pool_fallback"
            docs = [Document(page_content=content, metadata=fallback_meta)]
            splitter_counts["whole_document_pool_fallback"] = splitter_counts.get("whole_document_pool_fallback", 0) + 1

        for doc in docs:
            all_chunks.append((doc.page_content, doc.metadata))

    return all_chunks, splitter_counts


def _attempt_relative_path(attempt_root: Path, path: Path) -> str:
    """Return a validated attempt-relative path for spool manifests."""
    root = attempt_root.resolve()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError as exc:
        raise ValueError("spool path escaped attempt root") from exc


def _chunk_spooled_batch_sync(
    attempt_root_text: str,
    records: tuple[tuple[str, str, int, dict], ...],
    chunk_size: int,
    chunk_overlap: int,
    use_tokens: bool,
    task_id: str,
    recursive_only: bool = False,
) -> tuple[str, int, int]:
    """Chunk a spool batch and write a compact JSONL descriptor manifest.

    This function receives only paths and small record metadata.  The parent
    ingests the manifest into SQLite after the worker has closed every file.
    """
    attempt_root = Path(attempt_root_text).resolve()
    task_dir = attempt_root / "tasks" / f"chunk-{task_id}"
    text_dir = task_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=False)
    manifest = task_dir / "chunks.jsonl"
    total_bytes = 0
    count = 0
    with manifest.open("w", encoding="utf-8") as output:
        for record_id, text_path, document_ordinal, metadata in records:
            source_path = attempt_root / text_path
            relative_source = _attempt_relative_path(attempt_root, source_path)
            content = source_path.read_text(encoding="utf-8")
            chunk_function = _chunk_document_batch_recursive_sync if recursive_only else _chunk_document_batch_sync
            chunks, _counts = chunk_function([(content, dict(metadata))], chunk_size, chunk_overlap, use_tokens)
            for output_ordinal, (chunk_text, chunk_metadata) in enumerate(chunks):
                digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                stable_id = hashlib.sha256(f"{record_id}\0{document_ordinal}\0{output_ordinal}\0{digest}".encode("utf-8")).hexdigest()
                text_file = text_dir / f"{document_ordinal:012d}-{output_ordinal:08d}.txt"
                text_file.write_text(chunk_text, encoding="utf-8")
                chunk_metadata = dict(chunk_metadata)
                chunk_metadata.update(
                    {
                        "chunk_id": stable_id,
                        "document_ordinal": document_ordinal,
                        "chunk_ordinal": output_ordinal,
                    }
                )
                item = {
                    "record_id": stable_id,
                    "source": chunk_metadata.get("source", metadata.get("source", relative_source)),
                    "ordinal": document_ordinal,
                    "text_path": _attempt_relative_path(attempt_root, text_file),
                    "metadata": chunk_metadata,
                    "text_bytes": len(chunk_text.encode("utf-8")),
                }
                output.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                total_bytes += item["text_bytes"]
                count += 1
    return _attempt_relative_path(attempt_root, manifest), count, total_bytes


def _combine_spool_manifests(attempt_root: Path, manifests: list[tuple[str, int, int]]) -> tuple[str, int, int]:
    """Stream retry manifests into one descriptor without materializing chunks."""
    output = attempt_root / "tasks" / f"chunk-combined-{uuid.uuid4().hex}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    count = text_bytes = 0
    with output.open("w", encoding="utf-8") as destination:
        for manifest_path, expected_count, expected_bytes in manifests:
            actual_count = actual_bytes = 0
            with (attempt_root / manifest_path).open(encoding="utf-8") as source:
                for line in source:
                    destination.write(line)
                    item = json.loads(line)
                    actual_count += 1
                    actual_bytes += int(item["text_bytes"])
            if (actual_count, actual_bytes) != (expected_count, expected_bytes):
                raise ValueError("chunk retry manifest does not match its task descriptor")
            count += actual_count
            text_bytes += actual_bytes
    return _attempt_relative_path(attempt_root, output), count, text_bytes


async def chunk_spooled_batch(
    job_id: str,
    attempt_root: Path,
    batch: Any,
    *,
    chunk_size: int,
    chunk_overlap: int,
    use_tokens: bool,
) -> Any:
    """Chunk a disk-backed batch in a supervised process.

    The result is ``SpoolTaskOutput`` only; chunk bodies remain task-private
    files.  Manifest ordering follows the source batch and per-document output
    ordinal, never worker completion order.
    """
    records = tuple((record.record_id, record.text_path, record.ordinal, dict(record.metadata)) for record in batch.records)
    diagnostic_source = ", ".join(str(record.metadata.get("source") or record.text_path) for record in batch.records[:3])
    text_bytes = sum(record.text_bytes for record in batch.records)
    request = resource_governor.estimate_request(
        job_id=job_id,
        stage="chunking",
        text_bytes=text_bytes,
        record_count=len(records),
        cpu_slots=1,
    )
    request = ResourceRequest(
        request.job_id,
        request.stage,
        request.estimated_peak_bytes,
        request.cpu_slots,
        request.provider_key,
        request.kind,
        diagnostic_source,
    )
    task_args = (str(attempt_root), records, chunk_size, chunk_overlap, use_tokens, uuid.uuid4().hex)
    try:
        manifest_path, record_count, output_bytes = await run_resource_task(request, _chunk_spooled_batch_sync, task_args)
    except (ResourceTaskError, asyncio.TimeoutError) as error:
        # A poisoned/native-heavy batch is retried one document at a time with
        # the deterministic recursive fallback, still in supervised children.
        # Descriptors are stream-merged on a thread; no chunk collection enters
        # the API process.
        logger.warning(
            "Chunking fallback: job=%s source=%s error=%s",
            job_id,
            diagnostic_source or "unknown",
            error,
        )
        outputs: list[tuple[str, int, int]] = []
        for record in records:
            single_request = resource_governor.estimate_request(
                job_id=job_id,
                stage="chunking",
                text_bytes=next(item.text_bytes for item in batch.records if item.record_id == record[0]),
                record_count=1,
                cpu_slots=1,
            )
            single_request = ResourceRequest(
                single_request.job_id,
                single_request.stage,
                single_request.estimated_peak_bytes,
                single_request.cpu_slots,
                single_request.provider_key,
                single_request.kind,
                str(record[3].get("source") or record[1]),
            )
            outputs.append(
                await run_resource_task(
                    single_request,
                    _chunk_spooled_batch_sync,
                    (str(attempt_root), (record,), chunk_size, chunk_overlap, use_tokens, uuid.uuid4().hex, True),
                )
            )
        manifest_path, record_count, output_bytes = await asyncio.to_thread(_combine_spool_manifests, attempt_root, outputs)
    return SpoolTaskOutput(
        manifest_path=manifest_path,
        record_count=record_count,
        text_bytes=output_bytes,
        peak_rss_bytes=0,
    )


async def _await_pool_futures_or_closed(
    futures: List,
    pool: ChunkingPool,
) -> List:
    """Await all futures, polling for pool closure.

    Plain `asyncio.gather()` would block forever on futures whose workers are
    stuck (e.g. inside Chonkie's Rust-backed tree-sitter detection, which
    ignores SIGTERM). Polling `pool.closed` lets us cancel pending futures
    and surface a `ChunkingPoolError` as soon as the pool is torn down.
    """
    pending = set(futures)
    while pending:
        done, pending = await asyncio.wait(
            pending,
            timeout=_POOL_CLOSED_POLL_SECONDS,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if pool.is_closed():
            for future in futures:
                if not future.done():
                    future.cancel()
            raise ChunkingPoolError(f"Chunking pool '{pool.key}' was terminated before all batches completed")
    return [future.result() for future in futures]


async def _retry_chunk_documents_individually(
    doc_data: List[Tuple[str, dict]],
    chunk_size: int,
    chunk_overlap: int,
    use_tokens: bool,
    pool: ChunkingPool,
) -> List[Tuple[List[Tuple[str, dict]], Dict[str, int]]]:
    """Retry a failed process-pool wave one document at a time."""
    loop = asyncio.get_event_loop()
    results: List[Tuple[List[Tuple[str, dict]], Dict[str, int]]] = []

    for idx, doc_item in enumerate(doc_data):
        if pool.is_closed():
            raise ChunkingPoolError(f"Chunking pool '{pool.key}' was terminated")

        try:
            result = await loop.run_in_executor(
                pool.executor,
                _chunk_document_batch_sync,
                [doc_item],
                chunk_size,
                chunk_overlap,
                use_tokens,
            )
        except BrokenProcessPool:
            file_path = doc_item[1].get("source", "unknown")
            logger.error(f"Chunking worker terminated while processing {file_path}; falling back to recursive chunking for that document")
            pool_was_already_closed = pool.is_closed()
            pool_manager.release(pool.key, terminate_workers=True)
            result = _chunk_document_batch_recursive_sync(
                [doc_item],
                chunk_size,
                chunk_overlap,
                use_tokens,
            )
            if not pool_was_already_closed:
                pool = pool_manager.get_or_create(pool.key, pool.max_workers)

        results.append(result)
        if idx % 10 == 9:
            await asyncio.sleep(0)

    return results


async def chunk_documents_parallel(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    use_tokens: bool,
    batch_size: int = 50,
    progress_callback: Optional[Callable[..., Any]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    pool_key: Optional[str] = None,
) -> List[Document]:
    """
    Chunk documents in parallel processes using Chonkie/Unstructured.

    Submits multiple batches concurrently to the process pool to fully utilize
    all worker processes, rather than submitting one batch at a time.

    Args:
        is_cancelled: Optional callback that returns True if the job has been cancelled.
                      Checked between waves to allow early exit.
        pool_key: Identifier for the chunking pool to use. Each indexing job
                  should pass its own `job.id` so its workers can be terminated
                  independently of other jobs. Defaults to
                  `SHARED_CHUNKING_POOL_KEY` for callers without a job identity.
    """
    if not documents:
        return []

    batch_size = _effective_batch_size(batch_size)
    key = pool_key or SHARED_CHUNKING_POOL_KEY

    # Compatibility callers still receive Documents, but execution is routed
    # through the same global governor as the spool pipeline.  The bounded IPC
    # guard in run_resource_task deliberately makes this unsuitable for a
    # corpus-sized result; document indexing uses chunk_spooled_batch instead.
    all_chunks: List[Document] = []
    for start in range(0, len(documents), batch_size):
        if is_cancelled and is_cancelled():
            raise asyncio.CancelledError("Job cancelled by user")
        batch_documents = documents[start : start + batch_size]
        batch_data = [(document.page_content, dict(document.metadata)) for document in batch_documents]
        text_bytes = sum(len(content.encode("utf-8")) for content, _metadata in batch_data)
        request = resource_governor.estimate_request(
            job_id=key,
            stage="chunking",
            text_bytes=text_bytes,
            record_count=len(batch_data),
            cpu_slots=1,
            kind="shared" if key == SHARED_CHUNKING_POOL_KEY else "document_job",
        )
        request = ResourceRequest(
            request.job_id,
            request.stage,
            request.estimated_peak_bytes,
            request.cpu_slots,
            request.provider_key,
            request.kind,
            ", ".join(str(metadata.get("source") or "unknown") for _content, metadata in batch_data[:3]),
        )
        try:
            result_chunks, _counts = await run_resource_task(
                request,
                _chunk_document_batch_sync,
                (batch_data, chunk_size, chunk_overlap, use_tokens),
            )
        except (ChunkingPoolError, BrokenProcessPool, ResourceTaskError, asyncio.TimeoutError):
            # A single recursive retry is still off-loop and supervised.  Do
            # not revive a private pool after a native failure.
            result_chunks, _counts = await run_resource_task(
                request,
                _chunk_document_batch_recursive_sync,
                (batch_data, chunk_size, chunk_overlap, use_tokens),
            )
        all_chunks.extend(Document(page_content=content, metadata=metadata) for content, metadata in result_chunks)
        if progress_callback:
            callback_result = progress_callback(min(start + len(batch_documents), len(documents)), len(documents))
            if asyncio.iscoroutine(callback_result):
                await callback_result
        await asyncio.sleep(0)
    return all_chunks


def rechunk_oversized_content(
    content: str,
    safe_token_limit: int,
    chunk_overlap: int = 0,
    metadata: dict | None = None,
) -> List[Document]:
    """
    Re-chunk oversized content into smaller pieces that fit within the token limit.

    This function properly re-chunks content that exceeds the embedding model's
    context limit, rather than blindly truncating. It uses RecursiveChunker to
    split at natural boundaries (paragraphs, sentences) and applies overlap
    using OverlapRefinery.

    Args:
        content: The oversized text content to re-chunk
        safe_token_limit: Maximum tokens per chunk (after safety margin applied)
        chunk_overlap: Token overlap between chunks (user-configured)
        metadata: Optional metadata dict to attach to each resulting chunk

    Returns:
        List of Document objects, each within the token limit
    """
    if metadata is None:
        metadata = {}

    # Account for overlap in the chunk size - overlap adds tokens to each chunk
    # so we need to reduce the base chunk size
    effective_chunk_size = max(100, safe_token_limit - chunk_overlap)

    # Use tiktoken for accurate token counting
    chunker = RecursiveChunker(
        TIKTOKEN_ENCODING,
        chunk_size=effective_chunk_size,
        min_characters_per_chunk=20,
    )

    chunks = chunker.chunk(content)

    # Apply overlap to add context from adjacent chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        refinery = OverlapRefinery(
            TIKTOKEN_ENCODING,
            context_size=chunk_overlap,
            mode="recursive",  # Use delimiter-aware overlap
            method="suffix",  # Add context from previous chunk
            merge=True,
            inplace=True,
        )
        chunks = refinery.refine(chunks)

    docs = []
    for i, c in enumerate(chunks):
        new_meta = metadata.copy()
        new_meta["chunker"] = "rechunk_oversized"
        new_meta["rechunk_part"] = i + 1
        new_meta["rechunk_total"] = len(chunks)
        docs.append(Document(page_content=c.text, metadata=new_meta))

    return docs


def rechunk_oversized_text(
    content: str,
    safe_token_limit: int,
    chunk_overlap: int = 0,
) -> List[str]:
    """
    Re-chunk oversized text content into smaller pieces that fit within the token limit.

    Thin wrapper around rechunk_oversized_content that returns plain strings.
    Used by filesystem indexer which works with raw text chunks.

    Args:
        content: The oversized text content to re-chunk
        safe_token_limit: Maximum tokens per chunk (after safety margin applied)
        chunk_overlap: Token overlap between chunks (user-configured)

    Returns:
        List of text strings, each within the token limit
    """
    docs = rechunk_oversized_content(content, safe_token_limit, chunk_overlap)
    return [doc.page_content for doc in docs]


# =============================================================================
# BATCH RECHUNKING (shared by git indexer and filesystem indexer)
# =============================================================================


def rechunk_documents_batch(
    chunks: List[Document],
    safe_token_limit: int,
    chunk_overlap: int = 0,
    max_warnings: int = 5,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> tuple[List[Document], int]:
    """
    Re-chunk oversized Document chunks that exceed the safe token limit.

    CPU-bound function intended to run in a thread pool. Iterates all chunks,
    re-chunks any that exceed the limit, and returns the full list with
    oversized chunks replaced by their smaller sub-chunks.

    Used by the git/upload (FAISS) indexer in service.py.

    Args:
        chunks: List of Document chunks to check and re-chunk
        safe_token_limit: Maximum tiktoken token count per chunk
        chunk_overlap: Token overlap between re-chunked pieces
        max_warnings: Number of individual re-chunk warnings to log
        is_cancelled: Optional callback that returns True if the job
                      has been cancelled. Checked periodically to allow
                      early exit from long-running re-chunking.

    Returns:
        Tuple of (result_chunks, rechunked_count)
    """
    result_chunks = []
    rc_count = 0
    for idx, chunk in enumerate(chunks):
        # Check for cancellation every 5000 chunks to avoid overhead
        if is_cancelled and idx % 5000 == 0 and is_cancelled():
            raise asyncio.CancelledError("Job cancelled by user")

        tokens = count_tokens(chunk.page_content)
        if tokens > safe_token_limit:
            source = chunk.metadata.get("source", "unknown")
            chunker_name = chunk.metadata.get("chunker", "unknown")

            sub_chunks = rechunk_oversized_content(
                chunk.page_content,
                safe_token_limit,
                chunk_overlap=chunk_overlap,
                metadata=chunk.metadata,
            )
            result_chunks.extend(sub_chunks)

            if rc_count < max_warnings:
                logger.warning(
                    f"Re-chunked oversized content from {tokens} tokens into "
                    f"{len(sub_chunks)} chunks (source: {source}, chunker: {chunker_name}, "
                    f"safe_limit: {safe_token_limit})"
                )
            rc_count += 1
        else:
            result_chunks.append(chunk)
    return result_chunks, rc_count


def rechunk_texts_batch(
    files_data: list[tuple[str, list[str], str, Path]],
    safe_token_limit: int,
    chunk_overlap: int = 0,
    max_warnings: int = 5,
) -> tuple[list[str], list[str], int]:
    """
    Re-chunk oversized text chunks that exceed the safe token limit.

    CPU-bound function intended to run in a thread pool. Processes file data
    tuples from the filesystem indexer, maintaining a parallel file_map for
    tracking which file each chunk belongs to.

    Used by the filesystem indexer in filesystem_service.py.

    Args:
        files_data: List of (rel_path, chunks, hash, file_path) tuples
        safe_token_limit: Maximum tiktoken token count per chunk
        chunk_overlap: Token overlap between re-chunked pieces
        max_warnings: Number of individual re-chunk warnings to log

    Returns:
        Tuple of (result_chunks, file_map, rechunked_count)
    """
    result_chunks: list[str] = []
    file_map: list[str] = []
    rc_count = 0
    for rel_path, chunks, *_ in files_data:
        for chunk in chunks:
            tokens = count_tokens(chunk)
            if tokens > safe_token_limit:
                sub_chunks = rechunk_oversized_text(chunk, safe_token_limit, chunk_overlap=chunk_overlap)
                if rc_count < max_warnings:
                    logger.warning(
                        f"Re-chunked oversized content from {tokens} tokens into {len(sub_chunks)} chunks (source: {rel_path}, safe_limit: {safe_token_limit})"
                    )
                rc_count += 1
                for sub_chunk in sub_chunks:
                    result_chunks.append(sub_chunk)
                    file_map.append(rel_path)
            else:
                result_chunks.append(chunk)
                file_map.append(rel_path)
    return result_chunks, file_map, rc_count


def is_context_length_error(exc: Exception) -> bool:
    """Detect embedding context length errors from Ollama and other providers."""
    for current in iter_exception_chain(exc):
        text = str(current).lower()
        if "input length exceeds" in text or "context length" in text or "maximum context length" in text or "token limit" in text:
            return True
    return False
