"""
Git History Search Tool - Query git repository history for detailed commit and file information.

This tool provides direct access to git repositories that have been cloned
for filesystem indexing. It allows the LLM to search commit history, get
detailed commit information, view file history, and perform git blame.

The tool works with persistent .git_repo directories that are maintained
for each git-based index.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ragtime.config.settings import settings
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Maximum number of changed files to display in commit details before truncating
MAX_FILES_DISPLAY_LIMIT = 50
# Maximum number of lines to display in diff output
MAX_DIFF_LINES = 200


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
                "'search_commits' - Search commit messages for keywords, "
                "'get_commit' - Get detailed info about a specific commit, "
                "'show_changes' (or 'get_diff') - Show the code changes (diff) for a commit, "
                "'file_history' - Get commit history for a specific file, "
                "'blame' - Show who last modified each line of a file, "
                "'find_files' - Find files matching a pattern (use before file_history/blame)"
            )
        )
        query: Optional[str] = Field(
            default=None,
            description="Search query for 'search_commits' action - keywords to find in commit messages",
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
        max_results: int = Field(
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

    for index_dir in index_base.iterdir():
        if not index_dir.is_dir():
            continue

        git_repo = index_dir / ".git_repo"
        if git_repo.exists() and (git_repo / ".git").exists():
            # Filter by index name if specified
            if index_name and index_dir.name != index_name:
                continue

            # Skip shallow clones - they have no meaningful history to search
            if await _is_shallow_repository(git_repo):
                logger.debug(
                    f"Skipping shallow git repo for {index_dir.name}: no meaningful history available"
                )
                continue

            repos.append((index_dir.name, git_repo))

    return repos


async def _search_commits(repo_path: Path, query: str, max_results: int) -> str:
    """Search commit messages for keywords."""
    # Use --all to search all branches, --grep for message search
    args = [
        "log",
        "--all",
        f"--grep={query}",
        "-i",  # Case insensitive
        f"-n{max_results}",
        "--format=%H|%an|%ad|%s",
        "--date=short",
    ]

    returncode, stdout, stderr = await _run_git_command(repo_path, args)

    if returncode != 0:
        return f"Error searching commits: {stderr}"

    if not stdout.strip():
        return f"No commits found matching '{query}'"

    lines = stdout.strip().split("\n")
    results = []
    for line in lines:
        parts = line.split("|", 3)
        if len(parts) >= 4:
            commit_hash, author, date, subject = parts
            results.append(f"- [{commit_hash[:8]}] {date} {author}: {subject}")

    return f"Found {len(results)} commit(s) matching '{query}':\n" + "\n".join(results)


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


async def _get_file_history(repo_path: Path, file_path: str, max_results: int) -> str:
    """Get commit history for a specific file."""
    args = [
        "log",
        f"-n{max_results}",
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


async def _find_files(repo_path: Path, pattern: str, max_results: int = 20) -> str:
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
    if len(files) > max_results:
        display_files = files[:max_results]
        return (
            f"Found {len(files)} files matching '{pattern}' "
            f"(showing first {max_results}):\n"
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
    max_results: int = 10,
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
                result = await _search_commits(repo_path, query, max_results)
            elif action == "get_commit":
                assert commit_hash is not None  # Validated above
                result = await _get_commit_details(repo_path, commit_hash)
            elif action in ("show_changes", "get_diff"):
                assert commit_hash is not None  # Validated above
                result = await _get_commit_diff(repo_path, commit_hash, file_path)
            elif action == "file_history":
                assert file_path is not None  # Validated above
                result = await _get_file_history(repo_path, file_path, max_results)
            elif action == "find_files":
                assert file_path is not None  # Validated above
                result = await _find_files(repo_path, file_path, max_results)
            elif action == "blame":
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
        "Actions: 'search_commits' (find commits by message keywords), "
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
                "'search_commits' - Search commit messages for keywords, "
                "'get_commit' - Get detailed info about a specific commit, "
                "'show_changes' (or 'get_diff') - Show the code changes (diff) for a commit, "
                "'file_history' - Get commit history for a specific file, "
                "'blame' - Show who last modified each line of a file, "
                "'find_files' - Find files matching a pattern (use before file_history/blame if unsure of path)"
            )
        )
        query: Optional[str] = Field(
            default=None,
            description="Search query for 'search_commits' action",
        )
        commit_hash: Optional[str] = Field(
            default=None,
            description="Commit hash for 'get_commit' or 'show_changes' action",
        )
        file_path: Optional[str] = Field(
            default=None,
            description="File path for 'file_history'/'blame' actions, optional filter for 'show_changes', or pattern for 'find_files' (e.g., 'index.php', '*.py')",
        )
        max_results: int = Field(
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
        max_results: int = 10,
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
            return await _search_commits(repo_path, query, max_results)
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
            return await _get_file_history(repo_path, file_path, max_results)
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
