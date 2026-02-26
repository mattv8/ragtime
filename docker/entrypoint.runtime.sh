#!/bin/bash
set -e

PORT=${PORT:-8090}
DEBUG_MODE=${DEBUG_MODE:-false}

if [ "$DEBUG_MODE" = "true" ]; then
	exec uvicorn runtime.main:app \
		--host 0.0.0.0 \
		--port "$PORT" \
		--reload \
		--reload-dir /runtime/runtime
fi

exec uvicorn runtime.main:app --host 0.0.0.0 --port "$PORT"
