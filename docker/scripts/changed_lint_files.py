#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import subprocess
import sys
from pathlib import Path

EMPTY_TREE_HASH = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
ALL_ZERO_REF = "0" * 40
PYTHON_ROOTS = ("docker/scripts", "ragtime", "runtime", "tests")
PYTHON_EXTENSIONS = (".py", ".pyi")
FRONTEND_ROOTS = ("ragtime/frontend",)
FRONTEND_EXTENSIONS = (".js", ".jsx", ".cjs", ".mjs", ".ts", ".tsx")
PYTHON_FULL_SCOPE_PATHS = (
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "mypy.ini",
    ".mypy.ini",
    "ruff.toml",
    ".ruff.toml",
    "uv.lock",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
)
FRONTEND_FULL_SCOPE_PATHS = ("ragtime/frontend/package.json", "ragtime/frontend/package-lock.json")
QUALITY_ENVIRONMENT_PATHS = (
    "docker/Dockerfile",
    "docker/scripts/changed_lint_files.py",
    "docker/scripts/run_scoped_mypy.py",
    "docker/scripts/run_scoped_ruff.py",
    "docker/scripts/scoped_python_paths.py",
    "docker/scripts/fix_inline_imports.py",
    "docker/scripts/install_deps_from_pyproject.py",
    ".github/workflows/ci.yml",
    ".github/workflows/quality.yml",
    ".github/workflows/build-container.yml",
)
CONTAINER_PATHS = (
    "docker",
    "docker-compose.yml",
    ".dockerignore",
    ".jscpd.json",
    "ragtime",
    "runtime",
    "tests",
    "typings",
    "pyproject.toml",
    ".env.example",
    "README.md",
    "prisma",
    ".github/workflows/build-container.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/quality.yml",
    ".github/actions",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--all", action="store_true")
    parser.add_argument("--base-ref")
    parser.add_argument("--head-ref")
    parser.add_argument("--merge-base", action="store_true")
    args = parser.parse_args()
    if args.local or args.all:
        if args.base_ref or args.head_ref or args.merge_base:
            parser.error("--local/--all cannot be combined with range options")
    elif not (args.base_ref and args.head_ref):
        parser.error("--base-ref and --head-ref are required together")

    repo_root = Path(_git_output("rev-parse", "--show-toplevel").strip())
    if args.all:
        _print_scopes(all_checks=True)
        return 0

    try:
        if args.local:
            changed_paths, existing_paths = _local_changed_paths()
        else:
            base_ref = _comparison_base(args.base_ref, args.head_ref, args.merge_base)
            changed_paths, existing_paths = _range_changed_paths(base_ref, args.head_ref)
    except subprocess.CalledProcessError:
        _print_scopes(all_checks=True)
        return 0

    quality_environment_changed = _quality_environment_changed(changed_paths)
    python_full_scope = quality_environment_changed or _needs_python_full_scope(changed_paths)
    frontend_full_scope = quality_environment_changed or _needs_frontend_full_scope(changed_paths)
    mypy_paths = _filter_paths(repo_root, existing_paths, PYTHON_ROOTS, PYTHON_EXTENSIONS)
    eslint_paths = _filter_paths(repo_root, existing_paths, FRONTEND_ROOTS, FRONTEND_EXTENSIONS)
    _print_scopes(
        mypy_scope="all" if python_full_scope else _encode_scope(mypy_paths),
        eslint_scope="all" if frontend_full_scope else _encode_scope(eslint_paths),
        ruff_scope="all" if python_full_scope else _encode_scope(mypy_paths),
        container_changed=_container_changed(changed_paths),
    )
    return 0


def _comparison_base(base_ref: str, head_ref: str, use_merge_base: bool) -> str:
    if not _range_refs_are_usable(base_ref, head_ref):
        raise subprocess.CalledProcessError(1, ["git", "rev-parse"])
    if not use_merge_base or base_ref == ALL_ZERO_REF:
        return base_ref
    return _git_output("merge-base", base_ref, head_ref).strip()


def _range_changed_paths(base_ref: str, head_ref: str) -> tuple[set[str], set[str]]:
    diff_base = EMPTY_TREE_HASH if base_ref == ALL_ZERO_REF else base_ref
    changed_paths: set[str] = set()
    existing_paths: set[str] = set()
    for status, paths in _git_name_status("diff", f"{diff_base}..{head_ref}"):
        changed_paths.update(paths)
        if not status.startswith("D"):
            existing_paths.add(paths[-1])
    return changed_paths, existing_paths


def _local_changed_paths() -> tuple[set[str], set[str]]:
    upstream_ref = _local_upstream_ref()
    changed_paths, existing_paths = _range_changed_paths(upstream_ref, "HEAD")
    for args in (("diff", "--cached"), ("diff",)):
        for status, paths in _git_name_status(*args):
            changed_paths.update(paths)
            if status.startswith("D"):
                existing_paths.discard(paths[-1])
            else:
                existing_paths.add(paths[-1])
    for path in _git_z_list("ls-files", "--others", "--exclude-standard"):
        changed_paths.add(path)
        existing_paths.add(path)
    return changed_paths, existing_paths


def _local_upstream_ref() -> str:
    return _git_output("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}").strip()


def _range_refs_are_usable(base_ref: str, head_ref: str) -> bool:
    refs_to_verify = [head_ref]
    if base_ref != ALL_ZERO_REF:
        refs_to_verify.append(base_ref)
    for ref in refs_to_verify:
        try:
            _git_output("rev-parse", "--verify", f"{ref}^{{commit}}")
        except subprocess.CalledProcessError:
            return False
    return True


def _git_name_status(*args: str) -> list[tuple[str, tuple[str, ...]]]:
    fields = _git_z_list(*args, "--name-status", "--find-renames")
    result: list[tuple[str, tuple[str, ...]]] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        path_count = 2 if status.startswith(("R", "C")) else 1
        paths = tuple(fields[index : index + path_count])
        if len(paths) != path_count:
            raise subprocess.CalledProcessError(1, ["git", *args])
        index += path_count
        result.append((status, paths))
    return result


def _filter_paths(repo_root: Path, paths: set[str], roots: tuple[str, ...], extensions: tuple[str, ...]) -> list[str]:
    selected: list[str] = []
    for path in sorted(paths):
        if _is_under_roots(path, roots) and path.endswith(extensions) and (repo_root / path).is_file():
            selected.append(path)
    return selected


def _needs_python_full_scope(paths: set[str]) -> bool:
    return any(_is_python_full_scope_path(path) for path in paths)


def _is_python_full_scope_path(path: str) -> bool:
    filename = path.rsplit("/", 1)[-1]
    return (
        path in PYTHON_FULL_SCOPE_PATHS
        or filename in {"ruff.toml", ".ruff.toml", "mypy.ini", ".mypy.ini"}
        or path.startswith(("prisma/", "typings/"))
        or path.startswith(("requirements", "constraints"))
        and path.endswith((".txt", ".in"))
    )


def _needs_frontend_full_scope(paths: set[str]) -> bool:
    return any(
        path in FRONTEND_FULL_SCOPE_PATHS
        or path.startswith("ragtime/frontend/")
        and (path.rsplit("/", 1)[-1].startswith(("eslint.config.", ".eslintrc", "tsconfig")))
        for path in paths
    )


def _quality_environment_changed(paths: set[str]) -> bool:
    return any(path in QUALITY_ENVIRONMENT_PATHS for path in paths)


def _container_changed(paths: set[str]) -> bool:
    return any(_is_under_roots(path, CONTAINER_PATHS) for path in paths)


def _is_under_roots(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in roots)


def _encode_scope(paths: list[str]) -> str:
    if not paths:
        return "none"
    payload = "".join(f"{path}\0" for path in paths).encode("utf-8")
    return f"files:{base64.b64encode(payload).decode('ascii')}"


def _print_scopes(
    *,
    all_checks: bool = False,
    mypy_scope: str = "none",
    eslint_scope: str = "none",
    ruff_scope: str = "none",
    container_changed: bool = False,
) -> None:
    if all_checks:
        mypy_scope = eslint_scope = ruff_scope = "all"
        container_changed = True
    print(f"mypy_scope={mypy_scope}")
    print(f"eslint_scope={eslint_scope}")
    print(f"ruff_scope={ruff_scope}")
    print(f"container_changed={'true' if container_changed else 'false'}")


def _git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, check=True)
    return result.stdout.decode("utf-8")


def _git_z_list(*args: str) -> list[str]:
    output = _git_output(*args, "-z")
    return [item for item in output.split("\0") if item]


if __name__ == "__main__":
    sys.exit(main())
