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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--base-ref")
    parser.add_argument("--head-ref")
    args = parser.parse_args()

    if args.local:
        if args.base_ref or args.head_ref:
            parser.error("use either --local or --base-ref/--head-ref")
    elif not (args.base_ref and args.head_ref):
        parser.error("--base-ref and --head-ref are required together")

    repo_root = Path(_git_output("rev-parse", "--show-toplevel").strip())
    if args.local:
        changed_paths = _local_changed_paths()
    else:
        if not _range_refs_are_usable(args.base_ref, args.head_ref):
            print("mypy_scope=all")
            print("eslint_scope=all")
            return 0
        changed_paths = _range_changed_paths(args.base_ref, args.head_ref)

    mypy_paths = _filter_paths(repo_root, changed_paths, PYTHON_ROOTS, PYTHON_EXTENSIONS)
    eslint_paths = _filter_paths(repo_root, changed_paths, FRONTEND_ROOTS, FRONTEND_EXTENSIONS)

    print(f"mypy_scope={_encode_scope(mypy_paths)}")
    print(f"eslint_scope={_encode_scope(eslint_paths)}")
    return 0


def _range_changed_paths(base_ref: str, head_ref: str) -> set[str]:
    diff_base = EMPTY_TREE_HASH if base_ref == ALL_ZERO_REF else base_ref
    return set(_git_z_list("diff", "--name-only", "--diff-filter=d", f"{diff_base}..{head_ref}"))


def _local_changed_paths() -> set[str]:
    upstream_ref = _local_upstream_ref()
    changed_paths = _range_changed_paths(upstream_ref, "HEAD")
    changed_paths.update(_git_z_list("diff", "--name-only", "--diff-filter=d", "--cached"))
    changed_paths.update(_git_z_list("diff", "--name-only", "--diff-filter=d"))
    changed_paths.update(_git_z_list("ls-files", "--others", "--exclude-standard"))
    return changed_paths


def _local_upstream_ref() -> str:
    try:
        return _git_output("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}").strip()
    except subprocess.CalledProcessError:
        raise SystemExit("Unable to determine upstream for --local mode; configure an upstream branch.") from None


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


def _filter_paths(repo_root: Path, paths: set[str], roots: tuple[str, ...], extensions: tuple[str, ...]) -> list[str]:
    selected: list[str] = []
    for path in sorted(paths):
        if not _is_under_roots(path, roots):
            continue
        if not path.endswith(extensions):
            continue
        if not (repo_root / path).is_file():
            continue
        selected.append(path)
    return selected


def _is_under_roots(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in roots)


def _encode_scope(paths: list[str]) -> str:
    if not paths:
        return "none"
    payload = "".join(f"{path}\0" for path in paths).encode("utf-8")
    return f"files:{base64.b64encode(payload).decode('ascii')}"


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("utf-8")


def _git_z_list(*args: str) -> list[str]:
    output = _git_output(*args, "-z")
    return [item for item in output.split("\0") if item]


if __name__ == "__main__":
    sys.exit(main())
