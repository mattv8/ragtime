#!/bin/bash
# ====================================
# Ragtime RAG API Entrypoint
# ====================================

set -e

PORT=${PORT:-8000}
API_PORT=${API_PORT:-8001}
DEBUG_MODE=${DEBUG_MODE:-false}

echo "Ragtime API"
echo ""

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
    # Dev mode: sync schema directly (faster iteration)
    if [ -n "$DATABASE_URL" ]; then
        python -m prisma db push --skip-generate --accept-data-loss > /dev/null 2>&1 || true
    fi
else
    # Production mode: apply migrations safely
    if [ -n "$DATABASE_URL" ]; then
        echo "Applying database migrations..."
        python -m prisma migrate deploy 2>&1 || {
            echo "Warning: Migration failed, database may need manual intervention"
        }
    fi
fi

# Start services based on mode
if [ "$DEBUG_MODE" = "true" ]; then
    # Development mode with hot-reload
    # Start uvicorn (Python backend) on PORT (8000)
    # Start Vite (UI) on API_PORT (8001)
    cd /ragtime/ragtime/frontend
    npm ci --loglevel=error > /dev/null 2>&1

    cd /ragtime
    uvicorn ragtime.main:app --host 0.0.0.0 --port $PORT --reload --reload-dir /ragtime/ragtime &
    sleep 2

    echo "  UI:         http://localhost:$API_PORT"
    echo "  API:        http://localhost:$PORT"
    echo "  API Docs:   http://localhost:$PORT/docs"
    echo ""

    cd /ragtime/ragtime/frontend
    exec npm run dev -- --host 0.0.0.0
else
    # Production mode
    exec "$@"
fi
