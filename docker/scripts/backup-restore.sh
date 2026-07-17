#!/bin/bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

if [[ "$SCRIPT_NAME" == "backup" || "$SCRIPT_NAME" == "restore" ]]; then
    exec python -m ragtime.core.server_backup "$SCRIPT_NAME" "$@"
fi

exec python -m ragtime.core.server_backup "$@"
