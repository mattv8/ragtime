"""
Centralized Git API utilities.

Provides functions for interacting with GitHub and GitLab APIs,
including repository visibility checks and branch listing.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import httpx

from ragtime.core.logging import get_logger

logger = get_logger(__name__)


class GitProvider(str, Enum):
    """Supported Git providers."""

    GITHUB = "github"
    GITLAB = "gitlab"
    GENERIC = "generic"  # Self-hosted or unknown


class RepoVisibility(str, Enum):
    """Repository visibility status."""

    PUBLIC = "public"
    PRIVATE = "private"
    NOT_FOUND = "not_found"
    ERROR = "error"


@dataclass
class ParsedGitUrl:
    """Parsed components of a Git URL."""

    provider: GitProvider
    host: str
    owner: str
    repo: str

    @property
    def api_base_url(self) -> Optional[str]:
        """Get the API base URL for this provider."""
        if self.provider == GitProvider.GITHUB:
            if self.host == "github.com":
                return "https://api.github.com"
            # Enterprise GitHub
            return f"https://{self.host}/api/v3"
        elif self.provider == GitProvider.GITLAB:
            if self.host == "gitlab.com":
                return "https://gitlab.com/api/v4"
            # Self-hosted GitLab
            return f"https://{self.host}/api/v4"
        return None


@dataclass
class RepoCheckResult:
    """Result of checking repository accessibility."""

    visibility: RepoVisibility
    has_stored_token: bool = False
    needs_token: bool = False
    message: str = ""


# Token prefix patterns for provider detection
GITHUB_TOKEN_PREFIXES = ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")
GITLAB_TOKEN_PREFIXES = ("glpat-", "glptt-", "gldt-", "glsoat-")


def detect_provider_from_token(token: str) -> Optional[GitProvider]:
    """Detect Git provider from token prefix."""
    if any(token.startswith(prefix) for prefix in GITHUB_TOKEN_PREFIXES):
        return GitProvider.GITHUB
    if any(token.startswith(prefix) for prefix in GITLAB_TOKEN_PREFIXES):
        return GitProvider.GITLAB
    return None


def parse_git_url(url: str, token: Optional[str] = None) -> Optional[ParsedGitUrl]:
    """
    Parse a Git URL into its components.

    Supports:
    - HTTPS: https://github.com/owner/repo.git
    - SSH: git@github.com:owner/repo.git

    Args:
        url: Git repository URL
        token: Optional token for provider detection on generic hosts

    Returns:
        ParsedGitUrl or None if URL is invalid
    """
    if not url or not isinstance(url, str):
        return None

    # HTTPS format
    https_match = re.match(r"^https?://([^/]+)/([^/]+)/([^/]+?)(\.git)?/?$", url)
    if https_match:
        host, owner, repo, _ = https_match.groups()
        provider = _detect_provider_from_host(host, token)
        return ParsedGitUrl(provider=provider, host=host, owner=owner, repo=repo)

    # SSH format
    ssh_match = re.match(r"^git@([^:]+):([^/]+)/([^/]+?)(\.git)?$", url)
    if ssh_match:
        host, owner, repo, _ = ssh_match.groups()
        provider = _detect_provider_from_host(host, token)
        return ParsedGitUrl(provider=provider, host=host, owner=owner, repo=repo)

    return None


def _detect_provider_from_host(host: str, token: Optional[str] = None) -> GitProvider:
    """Detect Git provider from hostname."""
    host_lower = host.lower()

    if host_lower == "github.com":
        return GitProvider.GITHUB
    if host_lower == "gitlab.com" or "gitlab" in host_lower:
        return GitProvider.GITLAB

    # Try to detect from token if provided
    if token:
        provider = detect_provider_from_token(token)
        if provider:
            return provider

    return GitProvider.GENERIC


async def check_repo_visibility(
    url: str,
    stored_token: Optional[str] = None,
    timeout: float = 10.0,
) -> RepoCheckResult:
    """
    Check if a repository is publicly accessible.

    This is used to determine whether a token is needed for re-indexing.
    We first try without auth - if that works, repo is public.
    If it fails with 404, we try with stored token (if available).

    Args:
        url: Git repository URL
        stored_token: Token stored in database (if any)
        timeout: Request timeout in seconds

    Returns:
        RepoCheckResult with visibility and whether token is needed
    """
    parsed = parse_git_url(url)
    if not parsed:
        return RepoCheckResult(
            visibility=RepoVisibility.ERROR,
            message="Invalid Git URL format",
        )

    # Only GitHub and GitLab public APIs are supported
    if parsed.provider == GitProvider.GENERIC:
        # For generic/self-hosted, assume token is needed if we have one stored
        return RepoCheckResult(
            visibility=RepoVisibility.PRIVATE,  # Assume private for safety
            has_stored_token=bool(stored_token),
            needs_token=not bool(stored_token),
            message="Self-hosted Git server - using stored credentials if available",
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # First, try without authentication
            public_accessible = await _check_repo_access(client, parsed, token=None)

            if public_accessible:
                return RepoCheckResult(
                    visibility=RepoVisibility.PUBLIC,
                    has_stored_token=bool(stored_token),
                    needs_token=False,
                    message="Repository is publicly accessible",
                )

            # Not publicly accessible - check if we have a stored token
            if stored_token:
                # Verify stored token still works
                token_works = await _check_repo_access(
                    client, parsed, token=stored_token
                )
                if token_works:
                    return RepoCheckResult(
                        visibility=RepoVisibility.PRIVATE,
                        has_stored_token=True,
                        needs_token=False,
                        message="Private repository - will use stored token",
                    )
                else:
                    return RepoCheckResult(
                        visibility=RepoVisibility.PRIVATE,
                        has_stored_token=True,
                        needs_token=True,
                        message="Private repository - stored token no longer valid",
                    )

            # No stored token - user needs to provide one
            return RepoCheckResult(
                visibility=RepoVisibility.PRIVATE,
                has_stored_token=False,
                needs_token=True,
                message="Repository is private or not found - token required",
            )

    except httpx.TimeoutException:
        logger.warning(f"Timeout checking repo visibility: {url}")
        return RepoCheckResult(
            visibility=RepoVisibility.ERROR,
            has_stored_token=bool(stored_token),
            needs_token=not bool(stored_token),
            message="Timeout checking repository - using stored credentials if available",
        )
    except Exception as e:
        logger.warning(f"Error checking repo visibility: {url} - {e}")
        return RepoCheckResult(
            visibility=RepoVisibility.ERROR,
            has_stored_token=bool(stored_token),
            needs_token=not bool(stored_token),
            message=f"Error checking repository: {e}",
        )


async def _check_repo_access(
    client: httpx.AsyncClient,
    parsed: ParsedGitUrl,
    token: Optional[str] = None,
) -> bool:
    """
    Check if we can access a repository's API.

    Args:
        client: HTTP client
        parsed: Parsed Git URL
        token: Optional auth token

    Returns:
        True if accessible, False otherwise
    """
    headers: dict[str, str] = {}

    if parsed.provider == GitProvider.GITHUB:
        # GitHub API - check repo endpoint
        api_url = f"https://api.github.com/repos/{parsed.owner}/{parsed.repo}"
        headers["Accept"] = "application/vnd.github.v3+json"
        if token:
            headers["Authorization"] = f"token {token}"

    elif parsed.provider == GitProvider.GITLAB:
        # GitLab API - check project endpoint (supports self-hosted)
        project_path = f"{parsed.owner}/{parsed.repo}".replace("/", "%2F")
        api_url = f"https://{parsed.host}/api/v4/projects/{project_path}"
        if token:
            headers["PRIVATE-TOKEN"] = token

    else:
        return False  # Can't check generic providers

    try:
        response = await client.get(api_url, headers=headers)
        return response.status_code == 200
    except Exception:
        return False


async def fetch_branches(
    url: str,
    token: Optional[str] = None,
    timeout: float = 10.0,
) -> tuple[list[str], Optional[str]]:
    """
    Fetch list of branches from a Git repository.

    Args:
        url: Git repository URL
        token: Optional auth token
        timeout: Request timeout in seconds

    Returns:
        Tuple of (branch_names, error_message)
    """
    parsed = parse_git_url(url, token)
    if not parsed:
        return [], "Invalid Git URL format"

    if parsed.provider == GitProvider.GENERIC:
        return [], None  # Can't fetch branches for generic providers

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            headers: dict[str, str] = {}

            if parsed.provider == GitProvider.GITHUB:
                api_url = f"https://api.github.com/repos/{parsed.owner}/{parsed.repo}/branches?per_page=100"
                headers["Accept"] = "application/vnd.github.v3+json"
                if token:
                    headers["Authorization"] = f"token {token}"

            elif parsed.provider == GitProvider.GITLAB:
                # Supports self-hosted GitLab instances
                project_path = f"{parsed.owner}/{parsed.repo}".replace("/", "%2F")
                api_url = f"https://{parsed.host}/api/v4/projects/{project_path}/repository/branches?per_page=100"
                if token:
                    headers["PRIVATE-TOKEN"] = token

            else:
                return [], None

            response = await client.get(api_url, headers=headers)

            if response.status_code == 404:
                error = "Repository not found or is private"
                if token:
                    error = "Repository not found or token lacks access"
                return [], error

            if response.status_code == 401:
                return [], "Invalid or expired token"

            if response.status_code != 200:
                return [], f"API error: {response.status_code}"

            data = response.json()
            branch_names = [b["name"] for b in data]
            return branch_names, None

    except httpx.TimeoutException:
        return [], "Timeout fetching branches"
    except Exception as e:
        logger.warning(f"Error fetching branches: {url} - {e}")
        return [], "Failed to fetch branches"
