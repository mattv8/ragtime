#!/usr/bin/env python3
"""Calculate the safe, event-specific container publication plan for CI."""

from __future__ import annotations

import argparse
import re


def _branch(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", value).strip(".-") or "detached"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True)
    parser.add_argument("--ref-name", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pr-repository", default="")
    parser.add_argument("--pr-number", default="")
    parser.add_argument("--base-ref", default="")
    parser.add_argument("--container-changed", choices=("true", "false"), default="true")
    parser.add_argument("--build-images", choices=("true", "false"), required=True)
    parser.add_argument("--build-legacy", choices=("true", "false"), required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        parser.error("--sha must be a lowercase 40-character Git SHA")
    trusted = args.event in {"push", "workflow_dispatch"}
    is_pr = args.event == "pull_request"
    branch = _branch(f"pr-{args.pr_number}" if is_pr else args.ref_name)
    short_sha = args.sha[:7]
    release_ref = args.ref_name in {"main", "beta"}
    publish = args.build_images == "true" and trusted and release_ref
    main = publish and args.build_legacy != "true"
    legacy = publish and not is_pr and args.ref_name == "main"
    if publish:
        tags = [branch, "latest" if args.ref_name == "main" else "latest-beta", f"{branch}-{short_sha}"]
    else:
        tags = []
    environment_ref = args.base_ref if is_pr else args.ref_name
    environment = "beta" if environment_ref == "beta" else "main"
    cache_fallback = "beta" if environment_ref == "beta" else "main"
    legacy_tag = "legacy" if legacy else ""
    values = {
        "trusted": str(trusted).lower(),
        "branch": branch,
        "short_sha": short_sha,
        "environment": environment,
        "cache_scope": branch,
        "cache_fallback": cache_fallback,
        "build_main": str(main).lower(),
        "build_runtime": str(main).lower(),
        "build_legacy": str(legacy).lower(),
        "promote": str(main or legacy).lower(),
        "app_tags": ",".join(tags),
        "runtime_tags": ",".join(tags),
        "legacy_tag": legacy_tag,
    }
    for key, value in values.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
