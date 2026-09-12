from __future__ import annotations

import base64
from pathlib import Path, PurePosixPath

DEFAULT_PYTHON_PATHS = ("docker/scripts", "ragtime", "runtime", "tests")
PYTHON_EXTENSIONS = (".py", ".pyi")


def resolve_python_scope(scope: str, scope_name: str, repo_root: Path) -> list[str] | None:
    if scope == "all":
        return list(DEFAULT_PYTHON_PATHS)
    if scope == "none":
        return None
    if not scope.startswith("files:"):
        raise SystemExit(f"Unsupported {scope_name}: {scope}")

    try:
        payload = base64.b64decode(scope[len("files:") :], validate=True)
        decoded = payload.decode("utf-8")
    except Exception as exc:
        raise SystemExit(f"Invalid {scope_name} payload: {exc}") from exc
    if decoded and not decoded.endswith("\0"):
        raise SystemExit(f"Invalid {scope_name} payload: paths must be NUL-terminated")

    paths: list[str] = []
    for rel_path in decoded.split("\0")[:-1]:
        _validate_python_path(rel_path, scope_name, repo_root)
        paths.append(rel_path)
    return paths or None


def _validate_python_path(rel_path: str, scope_name: str, repo_root: Path) -> None:
    path = PurePosixPath(rel_path)
    if (
        not rel_path
        or "\\" in rel_path
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or not rel_path.endswith(PYTHON_EXTENSIONS)
        or not _is_under_default_roots(rel_path)
    ):
        raise SystemExit(f"Unsupported {scope_name} path: {rel_path}")

    resolved = (repo_root / rel_path).resolve(strict=False)
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit(f"Unsupported {scope_name} path: {rel_path}") from exc
    if not resolved.is_file():
        raise SystemExit(f"{scope_name} path does not exist: {rel_path}")


def _is_under_default_roots(path: str) -> bool:
    return any(path.startswith(f"{root}/") for root in DEFAULT_PYTHON_PATHS)
