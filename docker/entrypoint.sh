#!/bin/bash
# ====================================
# Ragtime RAG API Entrypoint
# ====================================

set -e

API_PORT=${API_PORT:-8000}
INDEXER_PORT=${INDEXER_PORT:-8001}
DEBUG_MODE=${DEBUG_MODE:-false}

echo "Ragtime API"
echo ""

# Wait for PostgreSQL if requested
if [ -n "$WAIT_FOR_POSTGRES" ]; then
    until pg_isready -h "${POSTGRES_HOST:-postgres}" -p "${POSTGRES_PORT:-5432}" -U "${POSTGRES_USER:-postgres}" > /dev/null 2>&1; do
        sleep 2
    done
fi

# Development mode: install dependencies
if [ "$DEBUG_MODE" = "true" ]; then
    echo "Running in DEBUG mode"
    echo ""
    pip install --no-cache-dir --upgrade pip -q
    pip install --no-cache-dir -r /ragtime/requirements.txt -q
fi

# Generate Prisma client and run migrations
cd /ragtime
if [ "$DEBUG_MODE" = "true" ]; then
    python -m prisma generate > /dev/null 2>&1
fi

if [ -n "$DATABASE_URL" ]; then
    python -m prisma db push --skip-generate --accept-data-loss > /dev/null 2>&1 || true
fi

# Warn on missing LLM keys
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Warning: No LLM API key configured"
fi

# Start services based on mode
if [ "$DEBUG_MODE" = "true" ]; then
    # Development mode with hot-reload
    if [ "${ENABLE_INDEXER:-true}" = "true" ]; then
        cd /ragtime/ragtime/frontend
        npm ci --loglevel=error > /dev/null 2>&1

        cd /ragtime
        uvicorn ragtime.main:app --host 0.0.0.0 --port $API_PORT --reload --reload-dir /ragtime/ragtime &
        sleep 2

        echo "  Indexer UI: http://localhost:$INDEXER_PORT"
        echo "  API:        http://localhost:$API_PORT"
        echo "  API Docs:   http://localhost:$API_PORT/docs"
        echo ""

        cd /ragtime/ragtime/frontend
        exec npm run dev -- --host 0.0.0.0
    else
        exec uvicorn ragtime.main:app --host 0.0.0.0 --port $API_PORT --reload --reload-dir /ragtime/ragtime
    fi
else
    # Production mode
    exec "$@"
fi
