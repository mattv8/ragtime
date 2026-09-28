#!/usr/bin/env python3
"""Coordinate manual beta-to-main promotion and main-to-beta reconciliation PRs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol, Sequence

from release_version import plan_release

PROMOTION_START = "<!-- release-preview:start -->"
PROMOTION_END = "<!-- release-preview:end -->"


class PullRequestApi(Protocol):
    def list_open(self, base: str) -> list[dict[str, Any]]: ...
    def create(self, title: str, head: str, base: str, body: str) -> dict[str, Any]: ...
    def update(self, number: int, body: str) -> None: ...


class GhApi:
    """Small, mockable boundary around the GitHub CLI."""

    def __init__(self, repository: str) -> None:
        self.repository = repository

    def _call(self, arguments: list[str]) -> Any:
        result = subprocess.run(["gh", "api", *arguments], check=True, text=True, capture_output=True)
        payload = json.loads(result.stdout)
        if not isinstance(payload, (dict, list)):
            raise ValueError("GitHub API returned malformed JSON")
        return payload

    def list_open(self, base: str) -> list[dict[str, Any]]:
        payload = self._call(["--paginate", "--slurp", f"repos/{self.repository}/pulls?state=open&base={base}&per_page=100"])
        if not isinstance(payload, list) or not all(isinstance(page, list) for page in payload):
            raise ValueError("GitHub pull request list response was malformed")
        prs = [pr for page in payload for pr in page]
        if not all(_valid_pr(pr) for pr in prs):
            raise ValueError("GitHub pull request item was malformed")
        return prs

    def create(self, title: str, head: str, base: str, body: str) -> dict[str, Any]:
        payload = self._call(
            ["-X", "POST", f"repos/{self.repository}/pulls", "-f", f"title={title}", "-f", f"head={head}", "-f", f"base={base}", "-f", f"body={body}"]
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("number"), int):
            raise ValueError("GitHub create pull request response was malformed")
        return payload

    def update(self, number: int, body: str) -> None:
        payload = self._call(["-X", "PATCH", f"repos/{self.repository}/pulls/{number}", "-f", f"body={body}"])
        if not isinstance(payload, dict):
            raise ValueError("GitHub update pull request response was malformed")


def _git(repo: Path, arguments: list[str]) -> str:
    return subprocess.run(["git", *arguments], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()


def _ancestor(repo: Path, older: str, newer: str) -> bool:
    result = subprocess.run(["git", "merge-base", "--is-ancestor", older, newer], cwd=repo, check=False)
    if result.returncode not in {0, 1}:
        raise subprocess.CalledProcessError(result.returncode, result.args)
    return result.returncode == 0


def _valid_pr(value: Any) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("number"), int):
        return False
    base, head = value.get("base"), value.get("head")
    if not isinstance(base, dict) or not isinstance(base.get("ref"), str) or not isinstance(head, dict) or not isinstance(head.get("ref"), str):
        return False
    head_repo = head.get("repo")
    if head_repo is not None and (not isinstance(head_repo, dict) or not isinstance(head_repo.get("full_name"), str)):
        return False
    return (
        isinstance(value.get("body"), (str, type(None)))
        and isinstance(value.get("labels", []), list)
        and all(isinstance(label, dict) and isinstance(label.get("name"), str) for label in value.get("labels", []))
    )


def _same_repository_pr(pr: dict[str, Any], repository: str, head: str, base: str) -> bool:
    try:
        return pr["base"]["ref"] == base and pr["head"]["ref"] == head and pr["head"]["repo"]["full_name"] == repository
    except (KeyError, TypeError):
        return False


def _find(api: PullRequestApi, repository: str, head: str, base: str) -> dict[str, Any] | None:
    return next((pr for pr in api.list_open(base) if _same_repository_pr(pr, repository, head, base)), None)


def _replace_preview(body: str, preview: str) -> str:
    block = f"{PROMOTION_START}\n{preview}\n{PROMOTION_END}"
    starts, ends = body.count(PROMOTION_START), body.count(PROMOTION_END)
    if starts != ends or starts > 1:
        raise ValueError("release preview markers are malformed; refusing to rewrite maintainer text")
    start, end = body.find(PROMOTION_START), body.find(PROMOTION_END)
    if starts == 1:
        if end <= start:
            raise ValueError("release preview markers are malformed; refusing to rewrite maintainer text")
        return body[:start] + block + body[end + len(PROMOTION_END) :]
    return (body.rstrip() + "\n\n" if body.strip() else "") + block


def _preview(repo_root: Path, initial_version: str, pr: dict[str, Any], beta_sha: str, ahead: int, repository: str) -> str:
    labels = {label["name"] for label in pr.get("labels", [])}
    selected = labels & {"release:patch", "release:minor", "release:major"}
    if len(selected) > 1:
        raise ValueError("promotion PR has multiple release bump labels")
    bump = selected.pop().split(":", 1)[1] if selected else "auto"
    previous = Path.cwd()
    try:
        os.chdir(repo_root)
        release = plan_release(beta_sha, initial_version, bump)
    finally:
        os.chdir(previous)
    return f"## Release preview\n\nPlanned stable release: `{release['tag']}`\n\nBeta commits ahead: `{ahead}`\n\n[Compare included changes](https://github.com/{repository}/compare/main...{beta_sha})\n\nSource beta SHA: `{release['sha']}`"


def coordinate_release(repo_root: Path, repository: str, api: PullRequestApi, event: str, initial_version: str = "1.0.0") -> dict[str, str]:
    """Apply the event's bounded PR action and return string status fields."""
    main, beta = _git(repo_root, ["rev-parse", "origin/main"]), _git(repo_root, ["rev-parse", "origin/beta"])
    promotion = _find(api, repository, "beta", "main")
    if not _ancestor(repo_root, main, beta):
        # A beta push is deliberately refresh-only: it must not create either
        # kind of PR. A main push (and an operator-triggered run) handles the
        # required reconciliation instead.
        if event == "beta-push":
            return {"action": "none", "number": "", "promotion": "false"}
        reconciliation = _find(api, repository, "main", "beta")
        if reconciliation is None:
            reconciliation = api.create("Reconcile main into beta", "main", "beta", "Bring stable-branch changes back into beta. Merge with a merge commit.")
        return {"action": "reconciliation", "number": str(reconciliation.get("number", "")), "promotion": "false"}
    if event == "beta-push":
        if promotion is not None:
            ahead = int(_git(repo_root, ["rev-list", "--count", "origin/main..origin/beta"]))
            api.update(
                int(promotion["number"]),
                _replace_preview(str(promotion.get("body") or ""), _preview(repo_root, initial_version, promotion, beta, ahead, repository)),
            )
            return {"action": "refreshed", "number": str(promotion["number"]), "promotion": "true"}
        return {"action": "none", "number": "", "promotion": "false"}
    if event not in {"schedule", "manual", "main-push"}:
        raise ValueError("event must be schedule, manual, beta-push, or main-push")
    if event == "main-push":
        return {"action": "none", "number": "", "promotion": "false"}
    ahead = int(_git(repo_root, ["rev-list", "--count", "origin/main..origin/beta"]))
    diff = subprocess.run(["git", "diff", "--quiet", "origin/main", "origin/beta"], cwd=repo_root)
    if diff.returncode not in {0, 1}:
        raise subprocess.CalledProcessError(diff.returncode, diff.args)
    changed_tree = diff.returncode == 1
    if promotion is not None:
        api.update(
            int(promotion["number"]),
            _replace_preview(str(promotion.get("body") or ""), _preview(repo_root, initial_version, promotion, beta, ahead, repository)),
        )
        return {"action": "reused", "number": str(promotion["number"]), "promotion": "true"}
    if ahead == 0 or not changed_tree:
        return {"action": "none", "number": "", "promotion": "false"}
    created = api.create(
        "Release: beta to main",
        "beta",
        "main",
        _replace_preview("", _preview(repo_root, initial_version, {"labels": []}, beta, ahead, repository)),
    )
    return {"action": "created", "number": str(created.get("number", "")), "promotion": "true"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, choices=("schedule", "manual", "beta-push", "main-push"))
    parser.add_argument("--repository", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--initial-version", default="1.0.0")
    args = parser.parse_args(argv)
    if not os.environ.get("GH_TOKEN"):
        print("PR_AUTOMATION_TOKEN is required for release PR preparation.", file=sys.stderr)
        return 1
    try:
        print(
            json.dumps(
                coordinate_release(Path(args.repo_root).resolve(), args.repository, GhApi(args.repository), args.event, args.initial_version), sort_keys=True
            )
        )
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"Release PR coordination failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
