#!/bin/bash
# ====================================
# Ragtime RAG API Entrypoint
# ====================================

set -e

PORT=${PORT:-8000}
API_PORT=${API_PORT:-8001}
DEBUG_MODE=${DEBUG_MODE:-false}
ENABLE_HTTPS=${ENABLE_HTTPS:-false}
INDEX_DATA_PATH=${INDEX_DATA_PATH:-/data}

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

# Security check: Require API_KEY in production
if [ "$DEBUG_MODE" != "true" ] && [ -z "$API_KEY" ]; then
    echo "ERROR: API_KEY must be set."
    echo ""
    echo "The OpenAI-compatible API endpoint requires authentication to prevent"
    echo "unauthorized access to your LLM (which may incur costs) and tools."
    echo ""
    echo "Generate a secure key with: openssl rand -base64 32"
    echo "Add to your .env file: API_KEY=<your-key>"
    exit 1
fi

# HTTPS setup: Generate self-signed certificate if needed
SSL_CERT_FILE=${SSL_CERT_FILE:-$INDEX_DATA_PATH/ssl/server.crt}
SSL_KEY_FILE=${SSL_KEY_FILE:-$INDEX_DATA_PATH/ssl/server.key}

if [ "$ENABLE_HTTPS" = "true" ]; then
    echo "HTTPS enabled"

    # Check if certificates exist, generate if not
    if [ ! -f "$SSL_CERT_FILE" ] || [ ! -f "$SSL_KEY_FILE" ]; then
        echo "Generating self-signed SSL certificate..."
        mkdir -p "$(dirname "$SSL_CERT_FILE")"

        HOSTNAME=$(hostname)
        openssl req -x509 -newkey rsa:4096 \
            -keyout "$SSL_KEY_FILE" \
            -out "$SSL_CERT_FILE" \
            -days 365 \
            -nodes \
            -subj "/CN=$HOSTNAME/O=Ragtime/OU=Self-Signed" \
            -addext "subjectAltName=DNS:localhost,DNS:$HOSTNAME,IP:127.0.0.1" \
            2>/dev/null

        echo "Self-signed certificate generated: $SSL_CERT_FILE"
        echo "WARNING: Browsers will show security warnings for self-signed certificates."
        echo "For production, use a reverse proxy with proper SSL certificates."
    else
        echo "Using existing SSL certificate: $SSL_CERT_FILE"
    fi

    # Auto-enable SESSION_COOKIE_SECURE when using HTTPS
    export SESSION_COOKIE_SECURE=true
    echo ""
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

    # Build uvicorn command with optional SSL
    UVICORN_CMD="uvicorn ragtime.main:app --host 0.0.0.0 --port $PORT --reload --reload-dir /ragtime/ragtime"
    if [ "$ENABLE_HTTPS" = "true" ]; then
        UVICORN_CMD="$UVICORN_CMD --ssl-keyfile=$SSL_KEY_FILE --ssl-certfile=$SSL_CERT_FILE"
        PROTOCOL="https"
    else
        PROTOCOL="http"
    fi

    $UVICORN_CMD &
    sleep 2

    echo "  UI:         $PROTOCOL://localhost:$API_PORT"
    echo "  API:        $PROTOCOL://localhost:$PORT"
    echo "  API Docs:   $PROTOCOL://localhost:$PORT/docs"
    echo ""

    cd /ragtime/ragtime/frontend
    # Pass SSL env vars to Vite
    ENABLE_HTTPS=$ENABLE_HTTPS SSL_CERT_FILE=$SSL_CERT_FILE SSL_KEY_FILE=$SSL_KEY_FILE exec npm run dev -- --host 0.0.0.0
else
    # Production mode
    exec "$@"
fi
