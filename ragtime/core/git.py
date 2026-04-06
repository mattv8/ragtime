"""
Centralized Git API utilities.

Provides functions for interacting with GitHub and GitLab APIs,
including repository visibility checks and branch listing.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import quote_plus

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


@dataclass
class RepoCreateResult:
    """Result of creating a remote repository."""

    success: bool
    provider: GitProvider
    git_url: str | None = None
    default_branch: str | None = None
    visibility: str | None = None
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
                message="Repository not found or is private. If private, a token is required.",
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
        return (
            [],
            "Cannot fetch branches for generic Git providers (only GitHub/GitLab supported)",
        )

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

            logger.debug(f"Fetching branches from: {api_url}")
            response = await client.get(api_url, headers=headers)

            if response.status_code == 404:
                error = "Repository not found or is private"
                if token:
                    error = "Repository not found or token lacks access"
                return [], error

            if response.status_code == 401:
                return [], "Invalid or expired token"

            if response.status_code == 403:
                # 403 often means token lacks required scopes
                error = "Access forbidden - token may lack required scopes"
                if parsed.provider == GitProvider.GITLAB:
                    error += (
                        " (GitLab tokens need 'read_api' or 'read_repository' scope)"
                    )
                return [], error

            if response.status_code != 200:
                logger.warning(
                    f"Git API error {response.status_code}: {response.text[:200]}"
                )
                return [], f"API error: {response.status_code}"

            data = response.json()
            branch_names = [b["name"] for b in data]
            return branch_names, None

    except httpx.TimeoutException:
        return [], "Timeout fetching branches"
    except Exception as e:
        logger.warning(f"Error fetching branches: {url} - {e}")
        return [], "Failed to fetch branches"


async def fetch_default_branch(
    url: str,
    token: Optional[str] = None,
    timeout: float = 10.0,
) -> tuple[Optional[str], Optional[str]]:
    """Fetch the default branch from a GitHub or GitLab repository."""
    parsed = parse_git_url(url, token)
    if not parsed:
        return None, "Invalid Git URL format"
    if parsed.provider == GitProvider.GENERIC:
        return None, "Cannot fetch default branch for generic Git providers"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await _get_repo_api_response(client, parsed, token)
            if response.status_code == 404:
                return None, "Repository not found or is private"
            if response.status_code == 401:
                return None, "Invalid or expired token"
            if response.status_code == 403:
                return None, "Access forbidden - token may lack required scopes"
            if response.status_code != 200:
                return None, f"API error: {response.status_code}"

            data = response.json()
            default_branch = data.get("default_branch")
            if isinstance(default_branch, str) and default_branch.strip():
                return default_branch.strip(), None
            return None, None
    except httpx.TimeoutException:
        return None, "Timeout fetching default branch"
    except Exception as e:
        logger.warning(f"Error fetching default branch: {url} - {e}")
        return None, "Failed to fetch default branch"


async def create_repository(
    url: str,
    token: str,
    *,
    private: bool = True,
    description: Optional[str] = None,
    timeout: float = 20.0,
) -> RepoCreateResult:
    """Create a repository on GitHub or GitLab from the desired Git URL."""
    parsed = parse_git_url(url, token)
    if not parsed:
        return RepoCreateResult(
            success=False,
            provider=GitProvider.GENERIC,
            message="Invalid Git URL format",
        )

    if parsed.provider == GitProvider.GENERIC:
        return RepoCreateResult(
            success=False,
            provider=parsed.provider,
            message="Repository creation is only supported for GitHub and GitLab",
        )

    if not token:
        return RepoCreateResult(
            success=False,
            provider=parsed.provider,
            message="A personal access token is required to create a repository",
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if parsed.provider == GitProvider.GITHUB:
                return await _create_github_repository(
                    client,
                    parsed,
                    token,
                    private=private,
                    description=description,
                )
            if parsed.provider == GitProvider.GITLAB:
                return await _create_gitlab_repository(
                    client,
                    parsed,
                    token,
                    private=private,
                    description=description,
                )
    except httpx.TimeoutException:
        return RepoCreateResult(
            success=False,
            provider=parsed.provider,
            message="Timeout creating repository",
        )
    except Exception as e:
        logger.warning(f"Error creating repository: {url} - {e}")
        return RepoCreateResult(
            success=False,
            provider=parsed.provider,
            message="Failed to create repository",
        )

    return RepoCreateResult(
        success=False,
        provider=parsed.provider,
        message="Unsupported provider",
    )


async def _get_repo_api_response(
    client: httpx.AsyncClient,
    parsed: ParsedGitUrl,
    token: Optional[str] = None,
) -> httpx.Response:
    headers: dict[str, str] = {}
    if parsed.provider == GitProvider.GITHUB:
        api_url = f"{parsed.api_base_url}/repos/{parsed.owner}/{parsed.repo}"
        headers["Accept"] = "application/vnd.github.v3+json"
        if token:
            headers["Authorization"] = f"token {token}"
    elif parsed.provider == GitProvider.GITLAB:
        project_path = quote_plus(f"{parsed.owner}/{parsed.repo}", safe="")
        api_url = f"{parsed.api_base_url}/projects/{project_path}"
        if token:
            headers["PRIVATE-TOKEN"] = token
    else:
        raise ValueError("Unsupported provider")
    return await client.get(api_url, headers=headers)


async def _create_github_repository(
    client: httpx.AsyncClient,
    parsed: ParsedGitUrl,
    token: str,
    *,
    private: bool,
    description: Optional[str],
) -> RepoCreateResult:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}",
    }
    user_resp = await client.get(f"{parsed.api_base_url}/user", headers=headers)
    if user_resp.status_code != 200:
        return RepoCreateResult(
            success=False,
            provider=parsed.provider,
            message="Unable to validate GitHub token for repository creation",
        )

    login = str(user_resp.json().get("login") or "")
    payload = {
        "name": parsed.repo,
        "private": private,
        "description": description or "",
        "auto_init": False,
    }
    if parsed.owner == login:
        create_url = f"{parsed.api_base_url}/user/repos"
    else:
        create_url = f"{parsed.api_base_url}/orgs/{parsed.owner}/repos"

    response = await client.post(create_url, headers=headers, json=payload)
    if response.status_code not in {201, 202}:
        return RepoCreateResult(
            success=False,
            provider=parsed.provider,
            message=_format_repository_create_error(response),
        )

    data = response.json()
    return RepoCreateResult(
        success=True,
        provider=parsed.provider,
        git_url=str(
            data.get("clone_url")
            or f"https://{parsed.host}/{parsed.owner}/{parsed.repo}.git"
        ),
        default_branch=str(data.get("default_branch") or "main"),
        visibility="private" if private else "public",
        message="Repository created",
    )


async def _create_gitlab_repository(
    client: httpx.AsyncClient,
    parsed: ParsedGitUrl,
    token: str,
    *,
    private: bool,
    description: Optional[str],
) -> RepoCreateResult:
    headers = {"PRIVATE-TOKEN": token}
    namespace_id: int | None = None

    namespace_resp = await client.get(
        f"{parsed.api_base_url}/namespaces?search={quote_plus(parsed.owner)}",
        headers=headers,
    )
    if namespace_resp.status_code == 200:
        for item in namespace_resp.json():
            full_path = str(item.get("full_path") or "")
            path = str(item.get("path") or "")
            name = str(item.get("name") or "")
            if parsed.owner in {full_path, path, name}:
                namespace_id = int(item.get("id"))
                break

    payload: dict[str, object] = {
        "name": parsed.repo,
        "path": parsed.repo,
        "visibility": "private" if private else "public",
        "description": description or "",
        "initialize_with_readme": False,
    }
    if namespace_id is not None:
        payload["namespace_id"] = namespace_id

    response = await client.post(
        f"{parsed.api_base_url}/projects",
        headers=headers,
        json=payload,
    )
    if response.status_code not in {201, 202}:
        return RepoCreateResult(
            success=False,
            provider=parsed.provider,
            message=_format_repository_create_error(response),
        )

    data = response.json()
    return RepoCreateResult(
        success=True,
        provider=parsed.provider,
        git_url=str(
            data.get("http_url_to_repo")
            or f"https://{parsed.host}/{parsed.owner}/{parsed.repo}.git"
        ),
        default_branch=str(data.get("default_branch") or "main"),
        visibility="private" if private else "public",
        message="Repository created",
    )


def _format_repository_create_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = None

    if response.status_code == 401:
        return "Invalid or expired token"
    if response.status_code == 403:
        return "Access forbidden - token may lack repository write scopes"
    if response.status_code == 404:
        return "Target owner or namespace not found"
    if response.status_code == 409:
        return "Repository already exists"
    if response.status_code == 422 and isinstance(payload, dict):
        errors = payload.get("errors")
        if errors:
            return f"Repository creation failed: {errors}"
    if isinstance(payload, dict):
        message = payload.get("message") or payload.get("error")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return f"API error: {response.status_code}"
