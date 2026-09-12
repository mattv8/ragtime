#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from scoped_python_paths import resolve_python_scope


def main() -> int:
    scope = os.environ.get("RUFF_SCOPE", "all")
    paths = resolve_python_scope(scope, "RUFF_SCOPE", Path.cwd().resolve())
    if paths is None:
        print("Skipping Ruff: no changed Python files matched selector.")
        return 0

    for command in (
        ["format", "--check", "--", *paths],
        ["check", "--select", "E9,F821,F822,F823,I", "--", *paths],
    ):
        subprocess.run([sys.executable, "-m", "ruff", *command], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
