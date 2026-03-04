#!/bin/bash
set -e

PORT=${PORT:-8090}
DEBUG_MODE=${DEBUG_MODE:-false}
UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN=${UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN:-5}

if [ "$DEBUG_MODE" = "true" ]; then
	UVICORN_CMD=(
		uvicorn runtime.main:app
		--host 0.0.0.0
		--port "$PORT"
		--reload
		--timeout-graceful-shutdown "$UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN"
		--reload-dir /runtime/runtime
		--reload-exclude "runtime/**/__pycache__/*"
	)
	exec "${UVICORN_CMD[@]}"
fi

exec uvicorn runtime.main:app --host 0.0.0.0 --port "$PORT"
