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

if [ "$DEBUG_MODE" = "true" ]; then
    echo "Running in DEBUG mode"
    echo ""
fi

# Security check: Require explicit JWT_SECRET_KEY in production
if [ "$DEBUG_MODE" != "true" ] && [ -z "$JWT_SECRET_KEY" ]; then
    echo "ERROR: JWT_SECRET_KEY must be set in production mode."
    echo ""
    echo "Generate a secure key with: openssl rand -base64 32"
    echo "Add to your .env file: JWT_SECRET_KEY=<your-key>"
    echo ""
    echo "Or set DEBUG_MODE=true for development (not recommended for production)."
    exit 1
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

    # Install frontend dependencies if missing (node_modules is a named volume)
    if [ ! -d "/ragtime/ragtime/frontend/node_modules/.bin" ]; then
        echo "Installing frontend dependencies..."
        cd /ragtime/ragtime/frontend
        npm ci --loglevel=error
    fi

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
