#!/usr/bin/env python3
"""Promote a verified beta commit to main without force-pushing."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def run_git(repo_root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run git in *repo_root* and return its completed process."""
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )


def is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    """Return whether *ancestor* is an ancestor of (or equal to) *descendant*."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def repo_slug_from_remote(remote_url: str) -> str:
    """Extract the GitHub owner/repository slug from an origin URL."""
    url = remote_url.strip()
    if url.endswith(".git"):
        url = url[:-4]
    if url.startswith("git@") and ":" in url:
        path = url.split(":", 1)[1]
    elif "://" in url:
        path = url.split("://", 1)[1].split("/", 1)[1]
    else:
        raise ValueError("origin must use an SSH or HTTPS GitHub URL")
    parts = path.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("origin URL does not contain an owner/repository slug")
    return "/".join(parts)


def query_workflow_runs(repo_slug: str, sha: str) -> Dict[str, Any]:
    """Return build-container push workflow runs for *sha* through gh auth."""
    endpoint = "repos/" + repo_slug + "/actions/workflows/build-container.yml/runs?head_sha=" + sha + "&branch=beta" + "&event=push&per_page=100"
    result = subprocess.run(["gh", "api", endpoint], check=True, text=True, capture_output=True)
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("gh returned an unexpected workflow-runs response")
    return payload


def has_successful_evidence(payload: Dict[str, Any], candidate_sha: str) -> bool:
    """Return whether a workflow-runs response verifies *candidate_sha* on beta."""
    runs = payload.get("workflow_runs", [])
    return isinstance(runs, list) and any(
        isinstance(run, dict)
        and run.get("head_branch") == "beta"
        and run.get("head_sha") == candidate_sha
        and run.get("event") == "push"
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
        for run in runs
    )


def build_push_command(candidate_sha: str) -> List[str]:
    """Build the non-force command that advances main to *candidate_sha*."""
    return ["git", "push", "origin", candidate_sha + ":refs/heads/main"]


def resolve_repo_root(repo_root: Optional[str]) -> Path:
    """Resolve an explicit repository root or discover the current checkout root."""
    if repo_root:
        return Path(repo_root).resolve()
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        text=True,
        capture_output=True,
    )
    return Path(result.stdout.strip())


def _commit_sha(repo_root: Path, revision: str) -> str:
    return run_git(repo_root, ["rev-parse", "--verify", "--end-of-options", revision + "^{commit}"]).stdout.strip()


def _fetch_promotion_refs(repo_root: Path, *options: str) -> None:
    run_git(
        repo_root,
        [
            "fetch",
            *options,
            "origin",
            "refs/heads/main:refs/remotes/origin/main",
            "refs/heads/beta:refs/remotes/origin/beta",
        ],
    )


def _ensure_complete_history(repo_root: Path) -> None:
    shallow = run_git(repo_root, ["rev-parse", "--is-shallow-repository"]).stdout.strip()
    if shallow == "true":
        print("Repository is shallow; fetching complete history before ancestry checks.")
        _fetch_promotion_refs(repo_root, "--unshallow")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote a verified origin/beta commit to main without force-pushing.")
    parser.add_argument("--sha", help="Beta commit SHA to promote (defaults to origin/beta).")
    parser.add_argument("--repo-root", help="Checkout root (mainly useful for automation/tests).")
    parser.add_argument(
        "--skip-evidence",
        action="store_true",
        help="Emergency override: do not verify successful beta publication evidence.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Perform checks but do not push main.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run promotion checks and return a process exit code."""
    args = _parser().parse_args(argv)
    try:
        repo_root = resolve_repo_root(args.repo_root)
        _fetch_promotion_refs(repo_root)
        _ensure_complete_history(repo_root)

        beta_sha = _commit_sha(repo_root, "origin/beta")
        candidate_sha = _commit_sha(repo_root, args.sha) if args.sha else beta_sha
        if args.sha and not is_ancestor(repo_root, candidate_sha, beta_sha):
            print("Refusing promotion: --sha must be an ancestor of origin/beta.", file=sys.stderr)
            return 1

        main_sha = _commit_sha(repo_root, "origin/main")
        if not is_ancestor(repo_root, main_sha, candidate_sha):
            print(
                "Refusing promotion: origin/main is not an ancestor of the candidate. Reconcile main into beta first, then retry.",
                file=sys.stderr,
            )
            return 1

        if args.skip_evidence:
            print("WARNING: skipping beta publication evidence check (emergency override).")
        else:
            remote_url = run_git(repo_root, ["remote", "get-url", "origin"]).stdout
            repo_slug = repo_slug_from_remote(remote_url)
            if not has_successful_evidence(query_workflow_runs(repo_slug, candidate_sha), candidate_sha):
                print(
                    "Refusing promotion: no successful Build and Push Container push run was found for this candidate SHA.",
                    file=sys.stderr,
                )
                return 1

        command = build_push_command(candidate_sha)
        if args.dry_run:
            print("Dry run: checks passed; would run " + " ".join(command))
            return 0
        try:
            subprocess.run(command, cwd=repo_root, check=True, text=True)
        except subprocess.CalledProcessError:
            print(
                "Promotion push was rejected. Someone may have updated main concurrently; re-check ancestry and evidence, then retry.",
                file=sys.stderr,
            )
            return 1
    except (subprocess.CalledProcessError, ValueError, OSError, json.JSONDecodeError) as error:
        print("Promotion failed: " + str(error), file=sys.stderr)
        return 1

    print("Promoted " + candidate_sha + " to main.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
