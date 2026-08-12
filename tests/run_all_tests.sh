#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

bash "$ROOT_DIR/tests/check_readme_sync.sh" --check
docker build --progress=plain --target python-test -f "$ROOT_DIR/docker/Dockerfile" "$ROOT_DIR"
docker build --progress=plain --target frontend-format-check -f "$ROOT_DIR/docker/Dockerfile" "$ROOT_DIR"
docker build --progress=plain --target frontend-lint -f "$ROOT_DIR/docker/Dockerfile" --build-arg "ESLINT_SCOPE=all" "$ROOT_DIR"
docker build --progress=plain --target frontend-builder -f "$ROOT_DIR/docker/Dockerfile" --build-arg ENVIRONMENT=local --build-arg APP_VERSION=local "$ROOT_DIR"
