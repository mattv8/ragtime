#!/usr/bin/env bash
# Keep the user-facing entrypoint intentionally boring; the host implementation
# needs Python's structured subprocess, JSON, and signal handling facilities.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec python3 "$script_dir/switch_dev_worktree.py" "$@"
