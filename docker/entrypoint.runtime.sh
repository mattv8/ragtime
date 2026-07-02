#!/bin/bash
set -e

PORT=${PORT:-8090}
DEBUG_MODE=${DEBUG_MODE:-false}
UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN=${UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN:-5}

if [ "$DEBUG_MODE" != "true" ]; then
	case "${RUNTIME_MANAGER_AUTH_TOKEN:-}" in
		""|"runtime-manager-token"|"dev-runtime-manager-token")
			echo "ERROR: RUNTIME_MANAGER_AUTH_TOKEN must be set to a strong random value." >&2
			echo "Generate a secure token with: openssl rand -base64 32" >&2
			exit 1
			;;
	esac
	case "${RUNTIME_WORKER_AUTH_TOKEN:-}" in
		""|"runtime-worker-token"|"dev-runtime-worker-token")
			echo "ERROR: RUNTIME_WORKER_AUTH_TOKEN must be set to a strong random value." >&2
			echo "Generate a secure token with: openssl rand -base64 32" >&2
			exit 1
			;;
	esac
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
