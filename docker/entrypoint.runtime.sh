#!/bin/bash
set -e

PORT=${PORT:-8090}
DEBUG_MODE=${DEBUG_MODE:-false}
UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN=${UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN:-5}

# Runtime auth misconfiguration is surfaced in the Ragtime admin UI instead of
# blocking startup. Guarded routes reject all requests until a token is set.
if [ "$DEBUG_MODE" != "true" ] && [ -z "${RUNTIME_AUTH_TOKEN:-}" ] && [ -z "${RUNTIME_MANAGER_AUTH_TOKEN:-}" ]; then
	echo "WARNING: RUNTIME_AUTH_TOKEN is not set. All runtime API requests will be rejected until it is configured (generate with: openssl rand -base64 32)." >&2
fi

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
