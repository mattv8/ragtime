#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tests/check_frontend_format.sh [--check|--write]

Checks (or fixes) frontend JS/TS formatting and linting for ragtime/frontend.

  --check  (default) Fail if any JS/TS file is not Prettier-formatted
           or if ESLint reports any problem.
  --write  Format files with Prettier and apply ESLint autofixes.

Prettier is the formatting source of truth and matches the VSCode
"Format Document" (cmd+shift+f) output via .vscode/settings.json.
EOF
}

mode="check"

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  case "$1" in
    --check) mode="check" ;;
    --write) mode="write" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
frontend_dir="$(cd "${script_dir}/../ragtime/frontend" && pwd)"

cd "$frontend_dir"

if [[ ! -d node_modules ]]; then
  echo "Installing frontend dependencies (npm ci)..."
  npm ci --loglevel=error
fi

if [[ "$mode" == "write" ]]; then
  npm run format
  # ESLint autofix is best-effort; do not fail the fixer on unfixable issues.
  npm run lint:fix || true
  echo "Frontend JS/TS files formatted with Prettier; ESLint autofixes applied."
  exit 0
fi

# --check mode: Prettier formatting is a blocking gate.
if ! npm run format:check; then
  echo "Frontend formatting is out of sync." >&2
  echo "Run: bash tests/check_frontend_format.sh --write" >&2
  exit 1
fi

# ESLint is a blocking gate.
if ! npm run lint; then
  echo "ESLint reported problems above." >&2
  echo "Run: bash tests/check_frontend_format.sh --write to apply autofixes." >&2
  exit 1
fi

echo "Frontend format and lint checks passed."
exit 0
