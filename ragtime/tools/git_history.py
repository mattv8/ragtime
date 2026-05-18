"""
Git History Search Tool - Query git repository history for detailed commit and file information.

This tool provides direct access to git repositories that have been cloned
for filesystem indexing. It allows the LLM to search commit history, get
detailed commit information, view file history, and perform git blame.

The tool works with persistent .git_repo directories that are maintained
for each git-based index.
"""

import asyncio
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ragtime.config.settings import settings
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Maximum number of changed files to display in commit details before truncating
MAX_FILES_DISPLAY_LIMIT = 50
# Maximum number of lines to display in diff output
MAX_DIFF_LINES = 200
# Number of git-log candidates to score when falling back to fuzzy local search
FUZZY_COMMIT_CANDIDATE_LIMIT = 500

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize_query(query: str) -> List[str]:
    """Tokenize natural-language git history queries for fuzzy matching."""
    normalized = query.lower().replace("_", " ").replace("-", " ")
    return [token for token in _TOKEN_RE.findall(normalized) if len(token) >= 2]


def _normalize_search_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower().replace("_", " ").replace("-", " ")))


def _parse_commit_subject(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("[Commit "):
            return re.sub(r"^\[Commit\s+[0-9a-fA-F]+\]\s*", "", stripped).strip()

    first_line = content.splitlines()[0] if content else ""
    return re.sub(r"^\[Commit\s+[0-9a-fA-F]+\]\s*", "", first_line).strip()


async def _get_repo_diagnostics(repo_path: Path) -> str:
    """Get diagnostic information about the git repository state."""
    diagnostics = []

    # Check last fetch time
    fetch_head = repo_path / ".git" / "FETCH_HEAD"
    if fetch_head.exists():
        mtime = fetch_head.stat().st_mtime
        last_fetch = datetime.fromtimestamp(mtime)
        time_since = datetime.now() - last_fetch

        # Format nice duration
        if time_since.days > 0:
            ago = f"{time_since.days} days ago"
        elif time_since.seconds > 3600:
            ago = f"{time_since.seconds // 3600} hours ago"
        elif time_since.seconds > 60:
            ago = f"{time_since.seconds // 60} minutes ago"
        else:
            ago = "just now"

        diagnostics.append(
            f"Last fetch: {ago} ({last_fetch.strftime('%Y-%m-%d %H:%M:%S')})"
        )
    else:
        diagnostics.append("Last fetch: Never (fresh clone?)")

    # Check if shallow
    if await _is_shallow(repo_path):
        diagnostics.append("Repo is shallow (incomplete history)")

    return "; ".join(diagnostics)


def _create_git_history_input_schema(available_repos: Optional[List[str]] = None):
    """Create input schema with available repos in description."""
    repo_desc = "Optional: specific index/repo name to search (searches all git repos if not specified)"
    if available_repos:
        repo_desc += f". Available repos: {', '.join(available_repos)}"

    class DynamicGitHistorySearchInput(BaseModel):
        """Input schema for git history search tool."""

        action: str = Field(
            description=(
                "The git action to perform. One of: "
                "'search_commits' - Semantic/fuzzy search over commit messages and changed files, "
                "'get_commit' - Get detailed info about a specific commit, "
                "'show_changes' (or 'get_diff') - Show the code changes (diff) for a commit, "
                "'file_history' - Get commit history for a specific file, "
                "'blame' - Show who last modified each line of a file, "
                "'find_files' - Find files matching a pattern (use before file_history/blame)"
            )
        )
        query: Optional[str] = Field(
            default=None,
            description="Natural language query for 'search_commits' action - searches embedded commit history when available and falls back to fuzzy matching",
        )
        commit_hash: Optional[str] = Field(
            default=None,
            description="Commit hash for 'get_commit' or 'show_changes' action",
        )
        file_path: Optional[str] = Field(
            default=None,
            description="File path for 'file_history' or 'blame' actions (relative to repo root), pattern for 'find_files', or optional filter for 'show_changes'",
        )
        index_name: Optional[str] = Field(
            default=None,
            description=repo_desc,
        )
        k: int = Field(
            default=10,
            ge=1,
            le=50,
            description="Maximum number of results to return for search operations",
        )

    return DynamicGitHistorySearchInput


# Default schema (for backwards compatibility and static tool)
GitHistorySearchInput = _create_git_history_input_schema()


async def _run_git_command(
    repo_path: Path, args: List[str], timeout: int = 30
) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    cmd = ["git", "-C", str(repo_path)] + args

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return (
            process.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except asyncio.TimeoutError:
        return -1, "", f"Git command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


async def _get_commit_count(repo_path: Path) -> int:
    """Get the number of commits in the repository (reachable from HEAD)."""
    returncode, stdout, _ = await _run_git_command(
        repo_path, ["rev-list", "--count", "HEAD"], timeout=5
    )
    if returncode == 0 and stdout.strip().isdigit():
        return int(stdout.strip())
    return 0


async def _is_shallow(repo_path: Path) -> bool:
    """Check if a directory is a shallow clone."""
    # Check for .git/shallow file (indicates shallow clone)
    shallow_file = repo_path / ".git" / "shallow"
    if shallow_file.exists():
        return True

    # Also verify with git command for robustness
    returncode, stdout, _ = await _run_git_command(
        repo_path, ["rev-parse", "--is-shallow-repository"], timeout=5
    )
    if returncode == 0 and stdout.strip().lower() == "true":
        return True

    return False


async def _is_shallow_repository(repo_path: Path) -> bool:
    """Check if a git repository is a shallow clone AND has minimal history (depth=1).

    Returns True ONLY if the repo is shallow AND has <= 1 commit.
    Returns False if the repo is full depth OR if it is shallow but has > 1 commit.
    """
    if not await _is_shallow(repo_path):
        return False

    # It is shallow, check if it has meaningful history (more than 1 commit)
    commit_count = await _get_commit_count(repo_path)
    if commit_count > 1:
        return False

    return True


async def _find_git_repos(index_name: Optional[str] = None) -> List[tuple[str, Path]]:
    """Find all git repos in the index base path.

    Returns list of (index_name, repo_path) tuples.
    Excludes repos with only 1 commit (depth=1 clones) which aren't useful.
    Repos with >1 commits are included even if technically "shallow".
    """
    index_base = Path(settings.index_data_path)
    repos: List[tuple[str, Path]] = []

    if not index_base.exists():
        return repos

    def _scan_dirs() -> List[tuple[str, Path]]:
        """Blocking directory scan offloaded to a thread."""
        found: List[tuple[str, Path]] = []
        for d in index_base.iterdir():
            if not d.is_dir():
                continue
            gr = d / ".git_repo"
            if gr.exists() and (gr / ".git").exists():
                if index_name and d.name != index_name:
                    continue
                found.append((d.name, gr))
        return found

    candidates = await asyncio.to_thread(_scan_dirs)

    for dir_name, git_repo in candidates:
        # Skip shallow clones - they have no meaningful history to search
        if await _is_shallow_repository(git_repo):
            logger.debug(
                f"Skipping shallow git repo for {dir_name}: no meaningful history available"
            )
            continue

        repos.append((dir_name, git_repo))

    return repos


def _semantic_result_to_match(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metadata = result.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("type") != "git_commit":
        return None

    commit_hash = str(metadata.get("commit_hash") or "").strip()
    if not commit_hash:
        source = str(metadata.get("source") or "")
        if source.startswith("git:commit:"):
            commit_hash = source.rsplit(":", 1)[-1]

    if not commit_hash:
        return None

    content = str(result.get("content") or "")
    return {
        "commit_hash": commit_hash,
        "author": str(metadata.get("author") or "?"),
        "date": str(metadata.get("date") or "")[:10],
        "subject": _parse_commit_subject(content),
        "semantic_similarity": float(result.get("similarity") or 0.0),
        "fuzzy_score": 0.0,
        "matched_terms": [],
    }


async def _search_commits_semantic(
    index_name: Optional[str], query: str, k: int
) -> List[Dict[str, Any]]:
    """Search embedded git commit-history documents when the index has them."""
    if not index_name:
        return []

    try:
        from ragtime.core.app_settings import get_app_settings
        from ragtime.indexer.vector_backends import get_faiss_backend
        from ragtime.indexer.vector_utils import (
            FILESYSTEM_COLUMNS,
            get_embeddings_model,
            search_pgvector_embeddings,
        )
        from ragtime.rag.components import rag

        app_settings = await get_app_settings()
        embeddings = await get_embeddings_model(
            app_settings,
            return_none_on_error=True,
            logger_override=logger,
        )
        if embeddings is None:
            return []

        try:
            query_embedding = await embeddings.aembed_query(query)
        except AttributeError:
            embedded_query = await asyncio.to_thread(embeddings.embed_documents, [query])
            if not embedded_query:
                return []
            query_embedding = embedded_query[0]

        search_limit = max(k * 10, 100)
        raw_results: List[Dict[str, Any]] = []

        try:
            raw_results.extend(
                await search_pgvector_embeddings(
                    table_name="filesystem_embeddings",
                    query_embedding=query_embedding,
                    index_name=index_name,
                    max_results=search_limit,
                    columns=FILESYSTEM_COLUMNS,
                    extra_where="metadata->>'type' = 'git_commit'",
                    logger_override=logger,
                )
            )
        except Exception as e:
            logger.debug(f"pgvector git commit semantic search skipped: {e}")

        try:
            if index_name in rag.faiss_dbs:
                docs_with_scores = await asyncio.to_thread(
                    rag.faiss_dbs[index_name].similarity_search_with_score,
                    query,
                    k=search_limit,
                )
                for doc, score in docs_with_scores:
                    raw_results.append(
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "similarity": 1 - (float(score) / 2),
                        }
                    )
            else:
                faiss_backend = get_faiss_backend()
                if index_name in faiss_backend.get_loaded_indexes():
                    faiss_results = await faiss_backend.search(
                        query_embedding=query_embedding,
                        index_name=index_name,
                        max_results=search_limit,
                    )
                    raw_results.extend(faiss_results)
        except Exception as e:
            logger.debug(f"FAISS git commit semantic search skipped: {e}")

        matches: List[Dict[str, Any]] = []
        seen_hashes: set[str] = set()
        for result in raw_results:
            match = _semantic_result_to_match(result)
            if not match:
                continue
            dedupe_key = str(match["commit_hash"])[:12]
            if dedupe_key in seen_hashes:
                continue
            seen_hashes.add(dedupe_key)
            matches.append(match)

        matches.sort(key=lambda item: item.get("semantic_similarity", 0.0), reverse=True)
        return matches[:k]
    except Exception as e:
        logger.debug(f"Git commit semantic search unavailable: {e}")
        return []


def _parse_git_log_candidates(stdout: str, record_sep: str, field_sep: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for block in stdout.split(record_sep):
        block = block.strip()
        if not block:
            continue

        parts = block.split(field_sep, 4)
        if len(parts) < 5:
            continue

        commit_hash, author, date, subject, details = parts
        candidates.append(
            {
                "commit_hash": commit_hash.strip(),
                "author": author.strip() or "?",
                "date": date.strip(),
                "subject": subject.strip(),
                "details": details.strip(),
            }
        )
    return candidates


async def _load_fuzzy_commit_candidates(
    repo_path: Path, query: str, k: int
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    tokens = _tokenize_query(query)
    record_sep = "\x1e"
    field_sep = "\x1f"
    candidate_limit = max(FUZZY_COMMIT_CANDIDATE_LIMIT, k * 50)
    format_arg = f"--format={record_sep}%H{field_sep}%an{field_sep}%ad{field_sep}%s{field_sep}%b"

    def build_args(use_grep: bool) -> List[str]:
        args = [
            "log",
            "--all",
            f"-n{candidate_limit}",
            format_arg,
            "--date=short",
            "--name-only",
        ]
        if use_grep and tokens:
            grep_pattern = "|".join(re.escape(token) for token in tokens)
            args.insert(2, "--extended-regexp")
            args.insert(3, "--regexp-ignore-case")
            args.insert(4, f"--grep={grep_pattern}")
        return args

    returncode, stdout, stderr = await _run_git_command(
        repo_path, build_args(use_grep=True), timeout=60
    )
    if returncode != 0:
        return f"Error searching commits: {stderr}", []

    candidates = _parse_git_log_candidates(stdout, record_sep, field_sep)
    if candidates:
        return None, candidates

    returncode, stdout, stderr = await _run_git_command(
        repo_path, build_args(use_grep=False), timeout=60
    )
    if returncode != 0:
        return f"Error searching commits: {stderr}", []

    return None, _parse_git_log_candidates(stdout, record_sep, field_sep)


def _score_fuzzy_commit(candidate: Dict[str, Any], query: str) -> Dict[str, Any]:
    tokens = _tokenize_query(query)
    if not tokens:
        return {**candidate, "fuzzy_score": 0.0, "matched_terms": []}

    subject = str(candidate.get("subject") or "")
    searchable_text = f"{subject}\n{candidate.get('details') or ''}"
    normalized_text = _normalize_search_text(searchable_text)
    normalized_subject = _normalize_search_text(subject)
    words = normalized_text.split()
    word_set = set(words)
    compact_text = normalized_text.replace(" ", "")
    normalized_query = " ".join(tokens)

    score = 0.0
    matched_terms: List[str] = []

    if normalized_query and normalized_query in normalized_text:
        score += len(tokens) + 2.0

    for token in tokens:
        token_score = 0.0
        if token in word_set:
            token_score = 1.5
        elif token in normalized_text or token in compact_text:
            token_score = 1.0
        elif words:
            best_ratio = max(SequenceMatcher(None, token, word).ratio() for word in words)
            if best_ratio >= 0.82:
                token_score = best_ratio * 0.8

        if token_score > 0:
            matched_terms.append(token)
            if token in normalized_subject:
                token_score += 0.4
            score += token_score

    coverage = len(set(matched_terms)) / len(tokens)
    score *= 0.5 + coverage

    return {
        **candidate,
        "fuzzy_score": score,
        "matched_terms": sorted(set(matched_terms)),
        "semantic_similarity": 0.0,
    }


async def _search_commits_fuzzy(
    repo_path: Path, query: str, k: int
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    error, candidates = await _load_fuzzy_commit_candidates(repo_path, query, k)
    if error:
        return error, []

    scored = [_score_fuzzy_commit(candidate, query) for candidate in candidates]
    threshold = 0.75 if len(_tokenize_query(query)) <= 1 else 1.2
    matches = [candidate for candidate in scored if candidate.get("fuzzy_score", 0.0) >= threshold]
    matches.sort(key=lambda item: item.get("fuzzy_score", 0.0), reverse=True)
    return None, matches[:k]


def _merge_commit_matches(
    semantic_matches: List[Dict[str, Any]], fuzzy_matches: List[Dict[str, Any]], k: int
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for match in semantic_matches + fuzzy_matches:
        commit_hash = str(match.get("commit_hash") or "")
        if not commit_hash:
            continue
        key = commit_hash[:12]
        existing = merged.get(key)
        if existing is None:
            merged[key] = match
            continue

        existing["semantic_similarity"] = max(
            float(existing.get("semantic_similarity") or 0.0),
            float(match.get("semantic_similarity") or 0.0),
        )
        existing["fuzzy_score"] = max(
            float(existing.get("fuzzy_score") or 0.0),
            float(match.get("fuzzy_score") or 0.0),
        )
        existing_terms = set(existing.get("matched_terms") or [])
        existing_terms.update(match.get("matched_terms") or [])
        existing["matched_terms"] = sorted(existing_terms)

    def sort_key(match: Dict[str, Any]) -> float:
        semantic = float(match.get("semantic_similarity") or 0.0)
        fuzzy = float(match.get("fuzzy_score") or 0.0)
        return max(semantic * 10.0, fuzzy)

    results = list(merged.values())
    results.sort(key=sort_key, reverse=True)
    return results[:k]


def _format_commit_matches(
    query: str,
    matches: List[Dict[str, Any]],
    semantic_available: bool,
) -> str:
    if not matches:
        return f"No commits found matching '{query}'"

    search_modes = ["semantic"] if semantic_available else []
    search_modes.append("fuzzy")
    lines = [
        f"Found {len(matches)} commit(s) matching '{query}' ({' + '.join(search_modes)} search):"
    ]

    for match in matches:
        commit_hash = str(match.get("commit_hash") or "")
        date = str(match.get("date") or "")[:10]
        author = str(match.get("author") or "?")
        subject = str(match.get("subject") or "").strip() or "(no subject)"
        score_bits: List[str] = []
        semantic = float(match.get("semantic_similarity") or 0.0)
        fuzzy = float(match.get("fuzzy_score") or 0.0)
        matched_terms = match.get("matched_terms") or []
        if semantic > 0:
            score_bits.append(f"semantic {semantic:.3f}")
        if fuzzy > 0:
            score_bits.append(f"fuzzy {fuzzy:.1f}")
        if matched_terms:
            score_bits.append("terms: " + ", ".join(matched_terms[:6]))

        suffix = f" ({'; '.join(score_bits)})" if score_bits else ""
        lines.append(f"- [{commit_hash[:8]}] {date} {author}: {subject}{suffix}")

    return "\n".join(lines)


async def _search_commits(
    repo_path: Path,
    query: str,
    k: int,
    index_name: Optional[str] = None,
) -> str:
    """Search commits using embedded history when available, with fuzzy git-log fallback."""
    semantic_matches = await _search_commits_semantic(index_name, query, k)
    fuzzy_error, fuzzy_matches = await _search_commits_fuzzy(repo_path, query, k)
    if fuzzy_error and not semantic_matches:
        return fuzzy_error

    matches = _merge_commit_matches(semantic_matches, fuzzy_matches, k)
    return _format_commit_matches(
        query=query,
        matches=matches,
        semantic_available=bool(semantic_matches),
    )


async def _get_commit_details(repo_path: Path, commit_hash: str) -> str:
    """Get detailed information about a specific commit.

    Truncates the file list if there are too many changed files to avoid context overflow.
    """
    # Get commit message and metadata (no diff/stat)
    args_msg = [
        "show",
        commit_hash,
        "--no-patch",
        "--format=Commit: %H%nAuthor: %an <%ae>%nDate: %ad%n%nSubject: %s%n%nBody:%n%b",
        "--date=iso",
    ]

    # Get stat separately to handle truncation safely
    args_stat = [
        "show",
        commit_hash,
        "--stat",
        "--format=",  # No commit info, just stat
    ]

    # Run in parallel
    (rc_msg, msg_stdout, msg_stderr), (rc_stat, stat_stdout, stat_stderr) = (
        await asyncio.gather(
            _run_git_command(repo_path, args_msg),
            _run_git_command(repo_path, args_stat),
        )
    )

    if rc_msg != 0:
        if any(
            msg in msg_stderr
            for msg in ["unknown revision", "bad revision", "bad object"]
        ):
            diagnostics = await _get_repo_diagnostics(repo_path)
            return (
                f"Commit '{commit_hash}' not found locally.\n"
                f"Repo Status: {diagnostics}.\n"
                "Try re-indexing to fetch latest commits, or verify the commit hash."
            )
        return f"Error getting commit details: {msg_stderr}"

    # Process stat output
    stat_output = stat_stdout.strip()
    stat_lines = stat_output.split("\n")

    # Truncate if too many files (e.g., > 50 files)
    # Stat output usually ends with a summary line, check if we have enough lines
    if len(stat_lines) > MAX_FILES_DISPLAY_LIMIT + 1:
        # Keep summary line (last line)
        summary_line = stat_lines[-1]

        # Take first MAX_FILES_DISPLAY_LIMIT lines
        truncated_files = stat_lines[:MAX_FILES_DISPLAY_LIMIT]
        remaining_count = len(stat_lines) - 1 - MAX_FILES_DISPLAY_LIMIT

        stat_output = "\n".join(truncated_files)
        stat_output += f"\n... ({remaining_count} more files changed) ...\n"
        stat_output += summary_line

    if not stat_output:
        return msg_stdout.strip()

    return f"{msg_stdout.strip()}\n\n{stat_output}"


async def _get_commit_diff(
    repo_path: Path, commit_hash: str, file_path: Optional[str] = None
) -> str:
    """Get the diff for a specific commit, optionally filtered by file."""
    # args = ["show", commit_hash, "--format="] # This works but git show output is slightly different than diff
    # Use git show with patch
    args = ["show", commit_hash, "--format=commit %H"]
    if file_path:
        args.extend(["--", file_path])

    returncode, stdout, stderr = await _run_git_command(repo_path, args)

    if returncode != 0:
        if "bad object" in stderr or "unknown revision" in stderr:
            diagnostics = await _get_repo_diagnostics(repo_path)
            return (
                f"Commit '{commit_hash}' not found locally.\n"
                f"Repo Status: {diagnostics}.\n"
                "The index may be out of date (try re-indexing to fetch latest commits)."
            )
        return f"Error getting diff: {stderr}"

    lines = stdout.splitlines()
    if len(lines) > MAX_DIFF_LINES:
        truncated = "\n".join(lines[:MAX_DIFF_LINES])
        return (
            f"{truncated}\n"
            f"\n... (Output truncated, showing {MAX_DIFF_LINES} of {len(lines)} lines) ..."
        )

    return stdout


async def _get_file_history(repo_path: Path, file_path: str, k: int) -> str:
    """Get commit history for a specific file."""
    args = [
        "log",
        f"-n{k}",
        "--format=%H|%an|%ad|%s",
        "--date=short",
        "--follow",  # Follow file renames
        "--",
        file_path,
    ]

    returncode, stdout, stderr = await _run_git_command(repo_path, args)

    if returncode != 0:
        return f"Error getting file history: {stderr}"

    if not stdout.strip():
        return (
            f"No history found for '{file_path}' (file may not exist or has no commits)"
        )

    lines = stdout.strip().split("\n")
    results = []
    for line in lines:
        parts = line.split("|", 3)
        if len(parts) >= 4:
            commit_hash, author, date, subject = parts
            results.append(f"- [{commit_hash[:8]}] {date} {author}: {subject}")

    return f"History for '{file_path}' ({len(results)} commit(s)):\n" + "\n".join(
        results
    )


async def _git_blame(repo_path: Path, file_path: str, max_lines: int = 100) -> str:
    """Show who last modified each line of a file."""
    # First check if file exists
    check_args = ["ls-files", "--", file_path]
    returncode, stdout, _ = await _run_git_command(repo_path, check_args)

    if not stdout.strip():
        return f"File '{file_path}' not found in repository"

    # Get blame with commit info
    args = [
        "blame",
        "--line-porcelain",
        file_path,
    ]

    returncode, stdout, stderr = await _run_git_command(repo_path, args, timeout=60)

    if returncode != 0:
        return f"Error running git blame: {stderr}"

    # Parse porcelain format into readable output
    lines = stdout.split("\n")
    blame_entries: List[str] = []
    current_entry: Dict[str, str] = {}
    line_count = 0

    for line in lines:
        if line.startswith("\t"):
            # This is the actual line content
            if current_entry:
                commit = current_entry.get("hash", "?")[:8]
                author = current_entry.get("author", "?")
                author_mail = current_entry.get("author_mail", "")
                line_num = current_entry.get("line", "?")
                summary = current_entry.get("summary", "")
                content = line[1:]  # Remove leading tab

                # Truncate summary
                if len(summary) > 40:
                    summary = summary[:37] + "..."

                blame_entries.append(
                    f"{line_num:4} {commit} ({author} {author_mail}) [{summary}]"
                )
                current_entry = {}
                line_count += 1

                if line_count >= max_lines:
                    break
        elif len(line) >= 40:
            # Commit hash line logic
            # A valid commit hash line in porcelain output starts with a 40-char hex string
            # Format: <40-char-sha1> <orig_line> <final_line> <group_lines>
            parts = line.split()
            if len(parts) >= 3 and len(parts[0]) == 40:
                # Strictly check if first token is a valid hex sha1
                is_sha1 = True
                for char in parts[0]:
                    if not (
                        (char >= "0" and char <= "9")
                        or (char >= "a" and char <= "f")
                        or (char >= "A" and char <= "F")
                    ):
                        is_sha1 = False
                        break

                if is_sha1:
                    current_entry = {
                        "hash": parts[0],
                        "line": parts[2],
                    }

        # Only parse headers if we have an active commit entry
        if current_entry:
            if line.startswith("author "):
                current_entry["author"] = line[7:]
            elif line.startswith("author-mail "):
                current_entry["author_mail"] = line[12:]
            elif line.startswith("summary "):
                current_entry["summary"] = line[8:]

    if not blame_entries:
        return f"Could not parse blame output for '{file_path}'"

    result = f"Blame for '{file_path}'"
    if line_count >= max_lines:
        result += f" (first {max_lines} lines)"

    result += "\nFormat: Line Hash (Author <Email>) [Subject]\n"
    result += "\n".join(blame_entries)

    return result


async def _find_files(repo_path: Path, pattern: str, k: int = 20) -> str:
    """Find files in the repository matching a pattern."""
    # Use git ls-files with pattern matching
    # First try exact match
    args = ["ls-files", "--", f"*{pattern}*"]
    returncode, stdout, stderr = await _run_git_command(repo_path, args)

    if returncode != 0:
        return f"Error finding files: {stderr}"

    files = [f.strip() for f in stdout.strip().split("\n") if f.strip()]

    if not files:
        # Try case-insensitive pattern with grep
        args = ["ls-files"]
        returncode, stdout, stderr = await _run_git_command(repo_path, args)
        if returncode == 0:
            all_files = stdout.strip().split("\n")
            pattern_lower = pattern.lower()
            files = [f for f in all_files if pattern_lower in f.lower()]

    if not files:
        return f"No files found matching '{pattern}'"

    # Limit results
    if len(files) > k:
        display_files = files[:k]
        return (
            f"Found {len(files)} files matching '{pattern}' "
            f"(showing first {k}):\n"
            + "\n".join(f"  {f}" for f in display_files)
        )

    return f"Found {len(files)} file(s) matching '{pattern}':\n" + "\n".join(
        f"  {f}" for f in files
    )


async def search_git_history(
    action: str,
    query: Optional[str] = None,
    commit_hash: Optional[str] = None,
    file_path: Optional[str] = None,
    index_name: Optional[str] = None,
    k: int = 10,
) -> str:
    """
    Search git repository history for commits, files, and blame information.

    This function queries the .git_repo directories maintained for git-based indexes.
    """
    # Find git repos
    repos = await _find_git_repos(index_name)

    if not repos:
        if index_name:
            return f"No git repository found for index '{index_name}'. The index may not be git-based or hasn't been indexed yet."
        return "No git repositories found. Index a git repository first."

    # Validate action and required parameters
    if action == "search_commits":
        if not query:
            return "Error: 'query' parameter is required for search_commits action"
    elif action in ("get_commit", "show_changes", "get_diff"):
        if not commit_hash:
            return f"Error: 'commit_hash' parameter is required for {action} action"
    elif action in ("file_history", "blame", "find_files"):
        if not file_path:
            return f"Error: 'file_path' parameter is required for {action} action"
    else:
        return (
            f"Unknown action: '{action}'. "
            "Valid actions: search_commits, get_commit, show_changes, file_history, blame, find_files"
        )

    # Execute action on each repo
    all_results = []

    for repo_name, repo_path in repos:
        try:
            result: str
            if action == "search_commits":
                assert query is not None  # Validated above
                result = await _search_commits(
                    repo_path, query, k, index_name=repo_name
                )
            elif action == "get_commit":
                assert commit_hash is not None  # Validated above
                result = await _get_commit_details(repo_path, commit_hash)
            elif action in ("show_changes", "get_diff"):
                assert commit_hash is not None  # Validated above
                result = await _get_commit_diff(repo_path, commit_hash, file_path)
            elif action == "file_history":
                assert file_path is not None  # Validated above
                result = await _get_file_history(repo_path, file_path, k)
            elif action == "find_files":
                assert file_path is not None  # Validated above
                result = await _find_files(repo_path, file_path, k)
            elif action == "blame":
                assert file_path is not None  # Validated above
                result = await _git_blame(repo_path, file_path)
            else:
                continue

            # Add repo context if searching multiple repos
            if len(repos) > 1:
                all_results.append(f"=== {repo_name} ===\n{result}")
            else:
                all_results.append(result)

        except Exception as e:
            logger.error(f"Error querying git repo {repo_name}: {e}", exc_info=True)
            all_results.append(f"=== {repo_name} ===\nError: {e}")

    return "\n\n".join(all_results)


def create_aggregate_git_history_tool(
    available_repos: Optional[List[str]] = None,
) -> StructuredTool:
    """Create aggregate git history tool with available repos in description.

    Args:
        available_repos: List of available repo/index names to include in description.
                        When None, searches all repos without listing them.
    """
    description = (
        "Search git repository history for detailed commit information. "
        "Actions: 'search_commits' (semantic/fuzzy search over commits), "
        "'get_commit' (show full commit details), "
        "'show_changes' (show code diff for a commit), "
        "'file_history' (show commits that modified a file), "
        "'blame' (show who last modified each line), "
        "'find_files' (find files matching a pattern - use before file_history/blame if unsure of path). "
    )
    if available_repos:
        description += f"Available repos: {', '.join(available_repos)}. "
    description += (
        "Use this to understand code evolution, find when bugs were introduced, "
        "or identify who worked on specific features."
    )

    # Create schema with available repos in field description
    input_schema = _create_git_history_input_schema(available_repos)

    return StructuredTool.from_function(
        coroutine=search_git_history,
        name="search_git_history",
        description=description,
        args_schema=input_schema,
    )


# Default tool for aggregate search mode (backwards compatibility)
git_history_tool = create_aggregate_git_history_tool()


def create_per_index_git_history_tool(
    index_name: str, repo_path: Path, description: str = ""
) -> StructuredTool:
    """Create a git history search tool for a specific index.

    This is used when aggregate_search is disabled.
    """

    class PerIndexGitHistoryInput(BaseModel):
        """Input schema for per-index git history search."""

        action: str = Field(
            description=(
                "The git action to perform. One of: "
                "'search_commits' - Semantic/fuzzy search over commit messages and changed files, "
                "'get_commit' - Get detailed info about a specific commit, "
                "'show_changes' (or 'get_diff') - Show the code changes (diff) for a commit, "
                "'file_history' - Get commit history for a specific file, "
                "'blame' - Show who last modified each line of a file, "
                "'find_files' - Find files matching a pattern (use before file_history/blame if unsure of path)"
            )
        )
        query: Optional[str] = Field(
            default=None,
            description="Natural language query for 'search_commits' action",
        )
        commit_hash: Optional[str] = Field(
            default=None,
            description="Commit hash for 'get_commit' or 'show_changes' action",
        )
        file_path: Optional[str] = Field(
            default=None,
            description="File path for 'file_history'/'blame' actions, optional filter for 'show_changes', or pattern for 'find_files' (e.g., 'index.php', '*.py')",
        )
        k: int = Field(
            default=10,
            ge=1,
            le=50,
            description="Maximum number of results",
        )

    async def search_this_repo(
        action: str,
        query: Optional[str] = None,
        commit_hash: Optional[str] = None,
        file_path: Optional[str] = None,
        k: int = 10,
        **_,
    ) -> str:
        """Search git history for this specific index."""
        # Validate the repo still exists
        if not repo_path.exists() or not (repo_path / ".git").exists():
            return f"Git repository for '{index_name}' not found. Re-index may be required."

        # Validate action and parameters
        if action == "search_commits":
            if not query:
                return "Error: 'query' parameter is required for search_commits"
            return await _search_commits(
                repo_path, query, k, index_name=index_name
            )
        elif action == "get_commit":
            if not commit_hash:
                return "Error: 'commit_hash' parameter is required for get_commit"
            return await _get_commit_details(repo_path, commit_hash)
        elif action in ("show_changes", "get_diff"):
            if not commit_hash:
                return f"Error: 'commit_hash' parameter is required for {action}"
            return await _get_commit_diff(repo_path, commit_hash, file_path)
        elif action == "file_history":
            if not file_path:
                return "Error: 'file_path' parameter is required for file_history"
            return await _get_file_history(repo_path, file_path, k)
        elif action == "blame":
            if not file_path:
                return "Error: 'file_path' parameter is required for blame"
            return await _git_blame(repo_path, file_path)
        else:
            return (
                f"Unknown action: '{action}'. "
                "Valid actions: search_commits, get_commit, show_changes, file_history, blame"
            )

    # Sanitize index name for tool name
    safe_name = index_name.replace("-", "_").replace(" ", "_").lower()

    tool_description = f"Search git history for the '{index_name}' repository. "
    if description:
        tool_description += f"{description} "
    tool_description += "Actions: 'search_commits', 'get_commit', 'show_changes', 'file_history', 'blame'."

    return StructuredTool.from_function(
        coroutine=search_this_repo,
        name=f"search_git_history_{safe_name}",
        description=tool_description,
        args_schema=PerIndexGitHistoryInput,
    )
