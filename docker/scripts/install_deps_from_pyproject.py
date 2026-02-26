#!/usr/bin/env python3
from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "Usage: install_deps_from_pyproject.py <pyproject.toml> <group> <output.txt>",
            file=sys.stderr,
        )
        return 2

    pyproject_path = Path(sys.argv[1])
    group = sys.argv[2]
    output_path = Path(sys.argv[3])

    deps = _extract_group(pyproject_path.read_text(encoding="utf-8"), group)

    if not deps:
        print(
            f"Dependency group '{group}' not found in pyproject.toml", file=sys.stderr
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(deps) + "\n", encoding="utf-8")
    return 0


def _extract_group(pyproject_text: str, group: str) -> list[str]:
    lines = pyproject_text.splitlines()
    in_section = False
    collecting = False
    bracket_balance = 0
    buffer: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            in_section = line == "[project.optional-dependencies]"
            if collecting:
                break
            continue

        if not in_section:
            continue

        if not collecting:
            prefix = f"{group} ="
            if line.startswith(prefix):
                rhs = line[len(prefix) :].strip()
                collecting = True
                buffer.append(rhs)
                bracket_balance += rhs.count("[") - rhs.count("]")
                if bracket_balance <= 0:
                    break
            continue

        buffer.append(line)
        bracket_balance += line.count("[") - line.count("]")
        if bracket_balance <= 0:
            break

    if not buffer:
        return []

    value = "\n".join(buffer)
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []

    deps: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            deps.append(item)
    return deps


if __name__ == "__main__":
    raise SystemExit(main())
