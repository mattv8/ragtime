"""
Centralized Git API utilities.

Provides functions for interacting with GitHub and GitLab APIs,
including repository visibility checks and branch listing.
"""

import asyncio
import base64
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional
from urllib.parse import quote, quote_plus, urlsplit

import httpx

from ragtime.core.git_auth import GITHUB_TOKEN_PREFIXES, GITLAB_TOKEN_PREFIXES, git_auth_pair
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

GIT_AUTH_FAILURE_MESSAGE = (
    "Authentication failed: confirm the token has Contents read-only access to this repository, "
    "the repository is selected for the token, and any organization approval or SSO authorization is complete."
)
GIT_REPOSITORY_ACCESS_FAILURE_MESSAGE = (
    "Repository could not be found or read: it may have moved, or the personal access token may not have this repository selected or readable."
)


@lru_cache(maxsize=1)
def _supports_git_config_environment() -> bool:
    """GIT_CONFIG_COUNT requires Git 2.31; never misclassify old-Git auth."""
    try:
        output = subprocess.check_output(["git", "--version"], text=True, stderr=subprocess.DEVNULL)
        match = re.search(r"(\d+)\.(\d+)", output)
        return bool(match and (int(match.group(1)), int(match.group(2))) >= (2, 31))
    except (OSError, subprocess.SubprocessError):
        return False


def is_git_auth_error(detail: str) -> bool:
    """Return whether Git's safe-to-display stderr indicates credential denial."""
    normalized = detail.lower()
    if re.search(r"rate[\s-]*limit|too many requests|\b429\b", normalized):
        return False
    return any(
        marker in normalized
        for marker in (
            "authentication failed",
            "invalid username or token",
            "bad credentials",
            "could not read username",
            "could not read password",
            "access denied",
            "access forbidden",
            "write access to repository not granted",
            "requested url returned error: 401",
            "requested url returned error: 403",
            "http basic: access denied",
            "repository not found",
        )
    )


def _safe_git_error(stderr: str, url: str, token: str | None = None) -> str:
    """Keep useful Git diagnostics while never returning credential material."""
    safe = re.sub(r"https?://[^\s/@]+@", "https://[REDACTED]@", stderr)
    if token:
        safe = safe.replace(token, "[REDACTED]")
        safe = safe.replace(quote(token, safe=""), "[REDACTED]")
        username, password = git_auth_pair(url, token)
        basic = base64.b64encode(f"{username}:{password}".encode()).decode()
        safe = safe.replace(basic, "[REDACTED]")
    return safe.strip()[:500]


def _git_probe_env(url: str, token: str | None) -> dict[str, str]:
    """Build noninteractive Git config for a one-off, host-scoped probe."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS"} and not re.fullmatch(r"GIT_CONFIG_(?:KEY|VALUE)_\d+", key)
    }
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.pop("GIT_ASKPASS", None)
    env.pop("SSH_ASKPASS", None)
    if not token:
        env["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes"
        return env
    parsed = urlsplit(url)
    base = f"https://{parsed.netloc}/"
    username, password = git_auth_pair(url, token)
    header = base64.b64encode(f"{username}:{password}".encode()).decode()
    # An empty extraheader resets inherited headers for this URL before adding
    # ours, preventing a process-level header from authenticating a probe.
    entries = [
        ("credential.helper", ""),
        ("http.followRedirects", "false"),
        (f"http.{base}.extraheader", ""),
        (f"http.{base}.extraheader", f"Authorization: Basic {header}"),
    ]
    env["GIT_CONFIG_COUNT"] = str(len(entries))
    for index, (key, value) in enumerate(entries):
        env[f"GIT_CONFIG_KEY_{index}"] = key
        env[f"GIT_CONFIG_VALUE_{index}"] = value
    return env


async def _probe_git_read_access(url: str, token: str | None, timeout: float = 10.0) -> tuple[list[str], str | None]:
    """Read refs through Git transport, returning only safe failure text."""
    parsed = urlsplit(url)
    is_ssh = bool(re.match(r"^git@[^:]+:[^/]+/[^/]+(?:\.git)?$", url))
    if is_ssh:
        if token:
            return [], "SSH repositories use SSH key access; personal access tokens cannot repair SSH authentication."
    elif token and parsed.scheme.lower() != "https":
        return [], "Personal access tokens can only be verified with an HTTPS Git URL"
    if not is_ssh and parsed.scheme.lower() not in {"https", "http"}:
        return [], "Invalid Git URL format"
    if not is_ssh and (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment):
        return [], "Invalid Git URL format"
    if shutil.which("git") is None:
        return [], "Git is unavailable; unable to verify repository access. Please retry after Git is available."
    if token and not _supports_git_config_environment():
        return [], "Git is too old to safely verify this token. Upgrade Git and retry."
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "ls-remote",
            "--heads",
            "--",
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_git_probe_env(url, token),
            start_new_session=os.name == "posix",
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            return [], "Repository access check timed out. Please retry."
    except FileNotFoundError:
        return [], "Git is unavailable; unable to verify repository access. Please retry after Git is available."
    except asyncio.CancelledError:
        raise
    except Exception:
        return [], "Unable to verify repository access. Please retry."
    finally:
        if process is not None and process.returncode is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except (OSError, ProcessLookupError):
                pass
            try:
                await process.wait()
            except Exception:
                pass

    if process is None:
        return [], "Unable to verify repository access. Please retry."

    if process.returncode == 0:
        branches = []
        for line in stdout.decode(errors="replace").splitlines():
            ref = line.split("\t", 1)[-1]
            if ref.startswith("refs/heads/"):
                branches.append(ref.removeprefix("refs/heads/"))
        return branches, None
    detail = _safe_git_error(stderr.decode(errors="replace"), url, token)
    if is_git_auth_error(detail):
        if "repository not found" in detail.lower():
            return [], GIT_REPOSITORY_ACCESS_FAILURE_MESSAGE
        return [], GIT_AUTH_FAILURE_MESSAGE
    return [], f"Unable to verify repository access: {detail or 'Git command failed'}"


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

    is_ssh = bool(re.match(r"^git@[^:]+:[^/]+/[^/]+(?:\.git)?$", url))
    # SSH keys, rather than PATs, authenticate SSH remotes. Probe them without
    # sending a stored HTTPS credential and keep the normal key workflow usable.
    if is_ssh:
        _, error = await _probe_git_read_access(url, None, timeout)
        if error is None:
            return RepoCheckResult(RepoVisibility.PUBLIC, bool(stored_token), False, "SSH key can read this repository")
        return RepoCheckResult(
            RepoVisibility.ERROR,
            bool(stored_token),
            False,
            "Could not verify SSH read access. Check SSH keys and repository configuration.",
        )

    # Only GitHub and GitLab public APIs are supported
    if parsed.provider == GitProvider.GENERIC:
        if stored_token:
            _, error = await _probe_git_read_access(url, stored_token, timeout)
            if error is None:
                return RepoCheckResult(RepoVisibility.PRIVATE, has_stored_token=True, message="Stored token can read this repository")
            return RepoCheckResult(
                RepoVisibility.ERROR,
                has_stored_token=True,
                needs_token=error in {GIT_AUTH_FAILURE_MESSAGE, GIT_REPOSITORY_ACCESS_FAILURE_MESSAGE},
                message=error,
            )
        _, error = await _probe_git_read_access(url, None, timeout)
        if error is None:
            return RepoCheckResult(RepoVisibility.PUBLIC, message="Repository is publicly accessible")
        return RepoCheckResult(
            visibility=RepoVisibility.ERROR,
            needs_token=error in {GIT_AUTH_FAILURE_MESSAGE, GIT_REPOSITORY_ACCESS_FAILURE_MESSAGE},
            message=error or "Could not verify repository access",
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # First, try without authentication
            public_accessible = await _check_repo_access(client, parsed, token=None)

            if public_accessible and not stored_token:
                return RepoCheckResult(
                    visibility=RepoVisibility.PUBLIC,
                    has_stored_token=bool(stored_token),
                    needs_token=False,
                    message="Repository is publicly accessible",
                )

            # API metadata can be public/readable even where Git Contents access
            # is denied. Always probe an existing credential through Git.
            if stored_token:
                _, probe_error = await _probe_git_read_access(url, stored_token, timeout)
                if probe_error is None:
                    return RepoCheckResult(
                        visibility=RepoVisibility.PUBLIC if public_accessible else RepoVisibility.PRIVATE,
                        has_stored_token=True,
                        needs_token=False,
                        message="Stored token can read this repository",
                    )
                needs_token = probe_error in {GIT_AUTH_FAILURE_MESSAGE, GIT_REPOSITORY_ACCESS_FAILURE_MESSAGE}
                return RepoCheckResult(
                    visibility=(RepoVisibility.PUBLIC if public_accessible else RepoVisibility.PRIVATE) if needs_token else RepoVisibility.ERROR,
                    has_stored_token=True,
                    needs_token=needs_token,
                    message=probe_error or "Unable to verify stored token",
                )

            # No stored token - the API rejected anonymous access.
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
            needs_token=False,
            message="Repository access check timed out. Please retry; stored credentials have not changed.",
        )
    except Exception:
        logger.warning("Unable to check repository visibility")
        return RepoCheckResult(
            visibility=RepoVisibility.ERROR,
            has_stored_token=bool(stored_token),
            needs_token=False,
            message="Could not check repository access. Please retry; stored credentials have not changed.",
        )


def _build_repo_api_request(
    parsed: ParsedGitUrl,
    token: Optional[str] = None,
    path: str = "",
) -> tuple[str, dict[str, str]]:
    """Build (api_url, headers) for a GitHub or GitLab repo API call.

    Args:
        parsed: Parsed Git URL with provider info.
        token: Optional personal access token for authentication.
        path: Suffix appended to the repo/project API endpoint path.

    Returns:
        Tuple of (full_api_url, request_headers).

    Raises:
        ValueError: If the provider is unsupported.
    """
    headers: dict[str, str] = {}

    if parsed.provider == GitProvider.GITHUB:
        api_url = f"{parsed.api_base_url}/repos/{parsed.owner}/{parsed.repo}{path}"
        headers["Accept"] = "application/vnd.github.v3+json"
        if token:
            headers["Authorization"] = f"token {token}"
    elif parsed.provider == GitProvider.GITLAB:
        project_path = quote_plus(f"{parsed.owner}/{parsed.repo}", safe="")
        api_url = f"{parsed.api_base_url}/projects/{project_path}{path}"
        if token:
            headers["PRIVATE-TOKEN"] = token
    else:
        raise ValueError("Unsupported provider")

    return api_url, headers


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
    api_url, headers = _build_repo_api_request(parsed, token)
    response = await client.get(api_url, headers=headers)
    if response.status_code == 200:
        return True
    rate_limited = (
        response.status_code == 429 or response.headers.get("x-ratelimit-remaining") == "0" or bool(re.search(r"rate[\s-]*limit", response.text, re.IGNORECASE))
    )
    if response.status_code in {401, 403, 404} and not rate_limited:
        return False
    response.raise_for_status()
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
    if not parse_git_url(url, token):
        return [], "Invalid Git URL format"
    return await _probe_git_read_access(url, token, timeout)


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
    api_url, headers = _build_repo_api_request(parsed, token)
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
        git_url=str(data.get("clone_url") or f"https://{parsed.host}/{parsed.owner}/{parsed.repo}.git"),
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
        git_url=str(data.get("http_url_to_repo") or f"https://{parsed.host}/{parsed.owner}/{parsed.repo}.git"),
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
