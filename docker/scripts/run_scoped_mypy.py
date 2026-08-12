#!/usr/bin/env python3
from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_PATHS = ["docker/scripts", "ragtime", "runtime", "tests"]


def resolve_mypy_paths(scope: str, repo_root: Path) -> list[str] | None:
    if scope == "all":
        return DEFAULT_PATHS.copy()
    if scope == "none":
        return None
    if not scope.startswith("files:"):
        raise SystemExit(f"Unsupported MYPY_SCOPE: {scope}")

    try:
        decoded = base64.b64decode(scope[len("files:") :], validate=True).decode("utf-8")
    except Exception as exc:
        raise SystemExit(f"Invalid MYPY_SCOPE payload: {exc}") from exc

    paths: list[str] = []
    for rel_path in [entry for entry in decoded.split("\0") if entry]:
        candidate = Path(rel_path)
        if candidate.is_absolute():
            raise SystemExit(f"Unsupported MYPY_SCOPE path: {rel_path}")
        resolved = (repo_root / candidate).resolve(strict=False)
        try:
            resolved.relative_to(repo_root)
        except ValueError as exc:
            raise SystemExit(f"Unsupported MYPY_SCOPE path: {rel_path}") from exc
        if not resolved.exists():
            raise SystemExit(f"MYPY_SCOPE path does not exist: {rel_path}")
        paths.append(rel_path)

    if not paths:
        return None
    return paths


def main() -> int:
    scope = os.environ.get("MYPY_SCOPE", "all")
    paths = resolve_mypy_paths(scope, Path.cwd().resolve())
    if paths is None:
        if scope.startswith("files:"):
            print("Skipping mypy: selector returned an empty Python file list.")
        else:
            print("Skipping mypy: no changed Python files matched selector.")
        return 0

    subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", "pyproject.toml", *paths],
        check=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
