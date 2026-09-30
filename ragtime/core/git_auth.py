"""Provider credential formats shared by Git probes, clone and fetch."""

from urllib.parse import urlsplit

GITHUB_TOKEN_PREFIXES = ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")
GITLAB_TOKEN_PREFIXES = ("glpat-", "glptt-", "gldt-", "glsoat-")


def git_auth_pair(git_url: str, token: str) -> tuple[str, str]:
    """Keep host recognition ahead of token prefixes, matching clone semantics."""
    host = (urlsplit(git_url).hostname or "").lower()
    if "github.com" in host:
        return "x-access-token", token
    if "gitlab" in host:
        return "oauth2", token
    if "bitbucket.org" in host:
        return "x-bitbucket-api-token-auth", token
    if token.startswith(GITHUB_TOKEN_PREFIXES):
        return "x-access-token", token
    if token.startswith(GITLAB_TOKEN_PREFIXES):
        return "oauth2", token
    return token, ""
