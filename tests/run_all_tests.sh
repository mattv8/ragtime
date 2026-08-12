#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

selector_output="$(python3 "$ROOT_DIR/docker/scripts/changed_lint_files.py" --local)"

mypy_scope=""
eslint_scope=""
while IFS= read -r line; do
  case "$line" in
    mypy_scope=*)
      mypy_scope="${line#mypy_scope=}"
      ;;
    eslint_scope=*)
      eslint_scope="${line#eslint_scope=}"
      ;;
    '')
      ;;
    *)
      echo "Unexpected selector output: $line" >&2
      exit 1
      ;;
  esac
done <<EOF
$selector_output
EOF

case "$mypy_scope" in
  all|none|files:*) ;;
  *)
    echo "Invalid mypy_scope: $mypy_scope" >&2
    exit 1
    ;;
esac

case "$eslint_scope" in
  all|none|files:*) ;;
  *)
    echo "Invalid eslint_scope: $eslint_scope" >&2
    exit 1
    ;;
esac

bash "$ROOT_DIR/tests/check_readme_sync.sh" --check
docker build --progress=plain --target python-test -f "$ROOT_DIR/docker/Dockerfile" --build-arg "MYPY_SCOPE=$mypy_scope" "$ROOT_DIR"
docker build --progress=plain --target frontend-format-check -f "$ROOT_DIR/docker/Dockerfile" "$ROOT_DIR"
docker build --progress=plain --target frontend-lint -f "$ROOT_DIR/docker/Dockerfile" --build-arg "ESLINT_SCOPE=$eslint_scope" "$ROOT_DIR"
docker build --progress=plain --target frontend-builder -f "$ROOT_DIR/docker/Dockerfile" --build-arg ENVIRONMENT=local --build-arg APP_VERSION=local "$ROOT_DIR"
