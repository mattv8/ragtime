#!/bin/bash
# ====================================
# Common Functions for Ragtime Scripts
# ====================================

# ANSI Color Codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Logging Helper
# Usage: log "LEVEL" "Message"
# Environment: LOG_PREFIX (optional)
# -----------------------------------------------------------------------------
log() {
    local level=$1
    shift
    local message="$*"
    local color=$NC
    local prefix=${LOG_PREFIX:-"script"}
    local out_fd=${LOG_FD:-1}

    case "$level" in
    ERROR)
        color=$RED
        ;;
    WARNING)
        color=$YELLOW
        ;;
    NOTICE)
        color="" # Caller responsible for color or raw message
        ;;
    INFO)
        color=$NC
        ;;
    DEBUG)
        color=$BLUE
        ;;
    esac

    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    local level_padded=$(printf "%-8s" "$level")
    local fmt_message="${timestamp} | ${level_padded} | ${prefix} | ${message}"

    if [ "$level" == "NOTICE" ]; then
        echo -e "${message}" >&$out_fd
    elif [ "$level" == "INFO" ]; then
        echo -e "${fmt_message}" >&$out_fd
    else
        echo -e "${color}${fmt_message}${NC}" >&$out_fd
    fi
}

# -----------------------------------------------------------------------------
# Parse DATABASE_URL into components
# Usage: parse_database_url
# Exports: DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME
# -----------------------------------------------------------------------------
parse_database_url() {
    if [ -z "$DATABASE_URL" ]; then
        return 1
    fi

    # Remove protocol
    local url="${DATABASE_URL#postgresql://}"

    # Extract user:pass
    local userpass="${url%%@*}"
    DB_USER="${userpass%%:*}"
    DB_PASS="${userpass#*:}"

    # Extract host:port/dbname
    local hostdb="${url#*@}"
    local hostport="${hostdb%%/*}"

    DB_HOST="${hostport%%:*}"

    # Extract port if present
    if [[ "$hostport" == *":"* ]]; then
        DB_PORT="${hostport#*:}"
    else
        DB_PORT="5432"
    fi

    # Extract DB name (remove query params if any)
    local dbname_raw="${hostdb#*/}"
    DB_NAME="${dbname_raw%%\?*}"

    export DB_USER DB_PASS DB_HOST DB_PORT DB_NAME
}

# -----------------------------------------------------------------------------
# Wait for PostgreSQL to be ready
# Usage: wait_for_postgres [retries]
# Relies on parsed DB variables (call parse_database_url first)
# -----------------------------------------------------------------------------
wait_for_postgres() {
    local retries=${1:-30}
    log "INFO" "Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."

    for i in $(seq 1 "$retries"); do
        if PGPASSWORD="$DB_PASS" pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
            return 0
        fi
        if [ "$i" -eq "$retries" ]; then
            log "WARNING" "PostgreSQL not ready after $retries attempts"
            return 1
        fi
        sleep 1
    done
}
