#!/usr/bin/env python3
"""Plan the stable SemVer tag for a Git commit without changing the repository."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Sequence

_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_TAG = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_BUMPS = {"auto", "patch", "minor", "major"}


def _git(arguments: Sequence[str], repo: Path) -> str:
    return subprocess.run(["git", *arguments], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()


def _version(value: str) -> tuple[int, int, int]:
    match = _VERSION.fullmatch(value)
    if not match:
        raise ValueError("version must be a stable MAJOR.MINOR.PATCH SemVer value")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _bump(version: tuple[int, int, int], bump: str) -> tuple[int, int, int]:
    major, minor, patch = version
    if bump == "major":
        return major + 1, 0, 0
    if bump == "minor":
        return major, minor + 1, 0
    return major, minor, patch + 1


def _stable_tags(repo: Path, target: str) -> list[tuple[tuple[int, int, int], str, str]]:
    tags: list[tuple[tuple[int, int, int], str, str]] = []
    seen_versions: dict[tuple[int, int, int], str] = {}
    for tag in _git(["tag", "--list"], repo).splitlines():
        match = _TAG.fullmatch(tag)
        if not match:
            continue
        tag_sha = _git(["rev-list", "-n", "1", tag], repo)
        version = int(match.group(1)), int(match.group(2)), int(match.group(3))
        previous = seen_versions.get(version)
        if previous is not None and previous != tag_sha:
            raise ValueError(f"ambiguous release tags for v{'.'.join(map(str, version))}")
        seen_versions[version] = tag_sha
        ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", tag_sha, target], cwd=repo, check=False)
        if ancestry.returncode not in {0, 1}:
            raise subprocess.CalledProcessError(ancestry.returncode, ancestry.args)
        if ancestry.returncode == 0:
            tags.append((version, tag, tag_sha))
    return tags


def _inferred_bump(repo: Path, baseline: str, target: str) -> str:
    revisions = ["rev-list", "--no-merges", "--format=%B"]
    revisions.append(f"{baseline}..{target}" if baseline else target)
    messages = _git(revisions, repo)
    if not messages:
        return "patch"
    if re.search(r"(^|\n)(?:[A-Za-z0-9_-]+(?:\([^\n)]*\))?)!:\s*|BREAKING(?: |-)CHANGE:", messages, re.IGNORECASE):
        return "major"
    if re.search(r"(^|\n)feat(?:\([^\n)]*\))?:", messages, re.IGNORECASE):
        return "minor"
    return "patch"


def plan_release(ref: str, initial_version: str = "1.0.0", bump: str = "auto") -> dict[str, str]:
    """Return workflow-safe strings describing the stable release planned for *ref*."""
    if not ref or ref.startswith("-"):
        raise ValueError("ref must be a non-option Git revision")
    if bump not in _BUMPS:
        raise ValueError("bump must be auto, patch, minor, or major")
    initial = _version(initial_version)
    repo = Path.cwd()
    sha = _git(["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"], repo)
    tags = _stable_tags(repo, sha)
    at_sha = [item for item in tags if item[2] == sha]
    if len(at_sha) > 1:
        raise ValueError("ambiguous stable release tags at target SHA")
    if at_sha:
        version, tag, _ = at_sha[0]
        earlier = [item for item in tags if item[2] != sha]
        previous = max(earlier, default=None, key=lambda item: item[0])
        return {"tag": tag, "version": ".".join(map(str, version)), "previous_tag": previous[1] if previous else "", "sha": sha, "existing_tag": "true"}
    baseline = max(tags, default=None, key=lambda item: item[0])
    if baseline is None:
        version = initial
        previous_tag = ""
    else:
        selected_bump = bump if bump != "auto" else _inferred_bump(repo, baseline[2], sha)
        version = _bump(baseline[0], selected_bump)
        previous_tag = baseline[1]
    return {"tag": "v" + ".".join(map(str, version)), "version": ".".join(map(str, version)), "previous_tag": previous_tag, "sha": sha, "existing_tag": "false"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--initial-version", default="1.0.0")
    parser.add_argument("--bump", default="auto", choices=sorted(_BUMPS))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(plan_release(args.ref, args.initial_version, args.bump), sort_keys=True))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
