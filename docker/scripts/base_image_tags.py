#!/usr/bin/env python3
"""Produce deterministic Harbor tags for source-free dependency images."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PLATFORM = b"linux/amd64\n"
BASE_REPOSITORY = "library/ragtime-base"
INPUTS: dict[str, tuple[str, ...]] = {
    "frontend": (
        "docker/Dockerfile",
        "ragtime/frontend/package.json",
        "ragtime/frontend/package-lock.json",
    ),
    "ci": (
        "docker/Dockerfile",
        "pyproject.toml",
        "docker/scripts/install_deps_from_pyproject.py",
        "prisma/schema.prisma",
    ),
    "production": (
        "docker/Dockerfile",
        "pyproject.toml",
        "docker/scripts/install_deps_from_pyproject.py",
        "ragtime/frontend/package.json",
        "ragtime/frontend/package-lock.json",
    ),
}


def _content_hash(root: Path, paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(PLATFORM)
    for relative_path in paths:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required base-image input is missing: {relative_path}")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def compute_tags(root: Path, registry: str) -> dict[str, str]:
    registry = registry.rstrip("/")
    if not registry:
        raise ValueError("registry must not be empty")
    return {
        "frontend_tag": f"{registry}/{BASE_REPOSITORY}:frontend-{_content_hash(root, INPUTS['frontend'])}",
        "python_ci_tag": f"{registry}/{BASE_REPOSITORY}:ci-{_content_hash(root, INPUTS['ci'])}",
        "production_tag": f"{registry}/{BASE_REPOSITORY}:production-{_content_hash(root, INPUTS['production'])}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    args = parser.parse_args()
    try:
        tags = compute_tags(Path(__file__).resolve().parents[2], args.registry)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    for key, value in tags.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
