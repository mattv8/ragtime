#!/bin/bash
# ====================================
# Ragtime RAG API Entrypoint
# ====================================

set -e

# Set Logger prefix
LOG_PREFIX="ragtime.entrypoint"

# Source functions helper
if [ -f "/docker-scripts/functions.sh" ]; then
    source "/docker-scripts/functions.sh"
elif [ -f "./docker/scripts/functions.sh" ]; then
    source "./docker/scripts/functions.sh"
fi

PORT=${PORT:-8000}
API_PORT=${API_PORT:-8001}
DEBUG_MODE=${DEBUG_MODE:-false}
ENABLE_HTTPS=${ENABLE_HTTPS:-false}
INDEX_DATA_PATH=${INDEX_DATA_PATH:-/data}

if [ "$DEBUG_MODE" = "true" ]; then
    log "WARNING" "Running in DEBUG mode"
fi

# Security check: Require API_KEY in production
if [ "$DEBUG_MODE" != "true" ] && [ -z "$API_KEY" ]; then
    log "ERROR" "API_KEY must be set."
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
    log "INFO" "HTTPS enabled"

    # Check if certificates exist, generate if not
    if [ ! -f "$SSL_CERT_FILE" ] || [ ! -f "$SSL_KEY_FILE" ]; then
        log "INFO" "Generating self-signed SSL certificate..."
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

        log "INFO" "Self-signed certificate generated: $SSL_CERT_FILE"
        log "WARNING" "Browsers will show security warnings for self-signed certificates."
        echo "For production, use a reverse proxy with proper SSL certificates."
    else
        log "INFO" "Using existing SSL certificate: $SSL_CERT_FILE"
    fi

    # Auto-enable SESSION_COOKIE_SECURE when using HTTPS
    export SESSION_COOKIE_SECURE=true
    echo ""
fi

# Generate Prisma client and run migrations
cd /ragtime
if [ "$DEBUG_MODE" = "true" ]; then
    python -m prisma generate >/dev/null 2>&1
    # Dev mode: use migrations (same as production) to preserve pgvector columns
    # Note: We can't use "db push" because it drops vector columns not in Prisma schema
    if [ -n "$DATABASE_URL" ]; then
        python -m prisma migrate deploy >/dev/null 2>&1 || true
    fi
else
    # Production mode: apply migrations safely
    if [ -n "$DATABASE_URL" ]; then
        log "INFO" "Applying database migrations..."
        python -m prisma migrate deploy 2>&1 || {
            log "WARNING" "Migration failed, database may need manual intervention"
        }
    fi
fi

# Apply pgvector optimizations (autovacuum, probes, etc.)
if [ -x "/docker-scripts/init-pgvector.sh" ]; then
    /docker-scripts/init-pgvector.sh 2>&1 || {
        log "WARNING" "pgvector optimization script failed (non-fatal)"
    }
fi

# Determine protocol based on HTTPS setting
if [ "$ENABLE_HTTPS" = "true" ]; then
    PROTOCOL="https"
else
    PROTOCOL="http"
fi

# Print startup banner and URLs
print_banner() {
    local ui_port=$1
    local api_port=$2

    # ASCII Banner
    if [ -t 1 ]; then
        term_width=$(tput cols 2>/dev/null || echo 0)
    else
        term_width=0
    fi
    banner_lines=(
        "    ____              __  _                "
        "   / __ \____ _____ _/ /_(_)___ ___  ___   "
        "  / /_/ / __ \`/ __ \`/ __/ / __ \`__ \/ _ \  "
        " / _, _/ /_/ / /_/ / /_/ / / / / / /  __/  "
        "/_/ |_|\__,_/\__, /\__/_/_/ /_/ /_/\___/   "
        "            /____/                         "
    )

    echo ""
    for line in "${banner_lines[@]}"; do
        msg=$(printf "%*s\n" $(((${#line} + term_width) / 2)) "$line")
        log "NOTICE" "$msg"
    done
    echo ""

    # Get network IP for display
    local network_ip
    network_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    network_ip=${network_ip:-"<unknown>"}

    if [ "$ui_port" = "$api_port" ]; then
        # Production: single port serves both UI and API
        log "NOTICE" "${GREEN}  ➜  Local:    $PROTOCOL://localhost:$api_port${NC}"
        log "NOTICE" "${GREEN}  ➜  Network:  $PROTOCOL://$network_ip:$api_port${NC}"
        log "NOTICE" "${GREEN}  ➜  API Docs: $PROTOCOL://localhost:$api_port/docs${NC}"
    else
        # Development: separate ports for UI (Vite) and API (uvicorn)
        log "NOTICE" "${GREEN}  ➜  UI Local:    $PROTOCOL://localhost:$ui_port${NC}"
        log "NOTICE" "${GREEN}  ➜  UI Network:  $PROTOCOL://$network_ip:$ui_port${NC}"
        log "NOTICE" "${GREEN}  ➜  API:         $PROTOCOL://localhost:$api_port${NC}"
        log "NOTICE" "${GREEN}  ➜  API Docs:    $PROTOCOL://localhost:$api_port/docs${NC}"
    fi
    echo ""

    # Security warnings
    local warnings=()

    if [ -z "$API_KEY" ]; then
        warnings+=("API_KEY is not set. The OpenAI-compatible API endpoint is unprotected.")
    fi

    if [ "${ALLOWED_ORIGINS:-*}" = "*" ]; then
        warnings+=("ALLOWED_ORIGINS=* allows requests from any origin. Consider restricting to specific domains.")
    fi

    if [ "${SESSION_COOKIE_SECURE:-false}" != "true" ] && [ "$ENABLE_HTTPS" != "true" ]; then
        warnings+=("SESSION_COOKIE_SECURE=false and ENABLE_HTTPS=false. Credentials and API keys will be transmitted in plaintext.")
    fi

    if [ ${#warnings[@]} -gt 0 ]; then
        term_width=$(tput cols 2>/dev/null || echo 80)
        log "NOTICE" "${YELLOW}$(printf '=%.0s' $(seq 1 $term_width))${NC}"
        log "NOTICE" "${YELLOW}SECURITY WARNINGS${NC}"
        log "NOTICE" "${YELLOW}$(printf '=%.0s' $(seq 1 $term_width))${NC}"
        for warning in "${warnings[@]}"; do
            log "NOTICE" "${YELLOW}  ⚠️  ${warning}${NC}"
        done
        echo ""
    fi
}

# Start services based on mode
if [ "$DEBUG_MODE" = "true" ]; then
    # Development mode with hot-reload
    # Start uvicorn (Python backend) on PORT (8000)
    # Start Vite (UI) on API_PORT (8001)

    # Install frontend dependencies if missing (node_modules is a named volume)
    if [ ! -d "/ragtime/ragtime/frontend/node_modules/.bin" ]; then
        log "INFO" "Installing frontend dependencies..."
        cd /ragtime/ragtime/frontend
        npm ci --loglevel=error
    fi

    cd /ragtime

    # Build uvicorn command with optional SSL
    # Use custom logging config for consistent formatting
    UVICORN_CMD="uvicorn ragtime.main:app --host 0.0.0.0 --port $PORT --reload --reload-dir /ragtime/ragtime --log-config /ragtime/ragtime/core/logging_config.json"
    if [ "$ENABLE_HTTPS" = "true" ]; then
        UVICORN_CMD="$UVICORN_CMD --ssl-keyfile=$SSL_KEY_FILE --ssl-certfile=$SSL_CERT_FILE"
    fi

    $UVICORN_CMD &

    # Wait for backend to be ready (up to 30s) before starting frontend
    log "INFO" "Waiting for backend to be ready on port $PORT..."
    timeout=30
    while ! curl -s -k "$PROTOCOL://localhost:$PORT/health" >/dev/null; do
        sleep 1
        timeout=$((timeout - 1))
        if [ "$timeout" -le 0 ]; then
            log "WARNING" "Timed out waiting for backend to start."
            break
        fi
    done

    # Print banner with dev ports (UI on API_PORT, API on PORT)
    print_banner "$API_PORT" "$PORT"

    cd /ragtime/ragtime/frontend
    # Pass SSL env vars to Vite
    # Use --silent (npm) and --logLevel warn (vite) to suppress duplicate startup logs since we print our own banner
    ENABLE_HTTPS=$ENABLE_HTTPS SSL_CERT_FILE=$SSL_CERT_FILE SSL_KEY_FILE=$SSL_KEY_FILE exec npm run dev --silent -- --host 0.0.0.0 --logLevel warn
else
    # Production mode - print banner then start server
    print_banner "$PORT" "$PORT"
    exec "$@"
fi
