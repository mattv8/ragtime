#!/bin/bash
set -e

# Backup and Restore Script for Ragtime
# Creates/restores backup archives containing database + FAISS indexes
# Handles schema migrations for backups from older versions
#
# IMPORTANT: Secrets Encryption
# -----------------------------
# Ragtime encrypts sensitive data (API keys, passwords) using Fernet symmetric
# encryption. The encryption key is auto-generated on first startup and stored
# at data/.encryption_key.
#
# To ensure backups can be restored with working secrets:
#   1. Use --include-secret flag to include the encryption key in backups
#   2. Or manually backup the .encryption_key file
#   3. Or re-enter all passwords/API keys after restore
#
# Without the same encryption key, encrypted secrets become unrecoverable.

# Source functions helper
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
if [ -f "$SCRIPT_DIR/functions.sh" ]; then
    source "$SCRIPT_DIR/functions.sh"
elif [ -f "/docker-scripts/functions.sh" ]; then
    source "/docker-scripts/functions.sh"
fi

# Set Log Prefix for functions.sh
export LOG_PREFIX="BACKUP"

# Current schema version (update when adding migrations)
# Format: YYYYMMDDHHMMSS of latest migration
CURRENT_SCHEMA_VERSION="20260202000000"

# Paths
# FAISS indexes live at INDEX_DATA_PATH (same env the app uses).
# Default matches production compose mount (/app/data); dev overrides via env.
FAISS_DIR="${INDEX_DATA_PATH:-/app/data}"
TEMP_BASE="/tmp/ragtime_backup"
PRISMA_DIR="/ragtime/prisma"

# Initialize database connection vars
# parse_database_url is defined in functions.sh
parse_database_url || true

# Compatibility mapping: Scripts uses DB_* but backup script used POSTGRES_*
POSTGRES_USER="${POSTGRES_USER:-$DB_USER}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$DB_PASS}"
POSTGRES_HOST="${POSTGRES_HOST:-$DB_HOST}"
POSTGRES_DB="${POSTGRES_DB:-$DB_NAME}"
# Note: backup script didn't explicitly use PORT before, assuming it was handled by libpq defaults or psql
# Now we have DB_PORT available if needed.

# Get the latest applied migration from database

get_db_schema_version() {
    local version
    version=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c \
        "SELECT migration_name FROM _prisma_migrations ORDER BY finished_at DESC LIMIT 1" 2>/dev/null | tr -d '[:space:]' || echo "")
    # Extract timestamp from migration name (first 14 chars)
    if [ -n "$version" ]; then
        echo "${version:0:14}"
    else
        echo ""
    fi
}

# Check if _prisma_migrations table exists
check_migrations_table() {
    PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c \
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '_prisma_migrations')" 2>/dev/null | tr -d '[:space:]'
}

# Run Prisma migrations to bring schema up to date
run_migrations() {
    if [ -d "$PRISMA_DIR" ] && command -v python &>/dev/null; then
        log "DEBUG" "Running Prisma migrations to update schema..."
        if python -m prisma migrate deploy 2>&1; then
            log "DEBUG" "Schema migrations applied successfully"
            return 0
        else
            log "WARNING" "Some migrations may have failed - check manually"
            return 1
        fi
    else
        log "WARNING" "Prisma not available - skipping schema migration"
        log "WARNING" "Run 'python -m prisma migrate deploy' manually after restore"
        return 1
    fi
}

# Show usage for backup
show_backup_usage() {
    echo "Usage: backup [OPTIONS]"
    echo ""
    echo "Create a backup archive containing database and FAISS indexes."
    echo "Streams to stdout - redirect to a file on the host."
    echo ""
    echo "Options:"
    echo "  --db-only         Backup database only (no FAISS indexes)"
    echo "  --faiss-only      Backup FAISS indexes only (no database)"
    echo "  --include-secret  Include the .encryption_key file in backup"
    echo "                    (required to decrypt secrets on restore)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  docker exec ragtime backup > backup.tar.gz"
    echo "  docker exec ragtime backup --include-secret > backup.tar.gz"
    echo "  docker exec ragtime backup --db-only > db.tar.gz"
    echo ""
    echo "Environment variables required:"
    echo "  POSTGRES_USER     - Database user"
    echo "  POSTGRES_PASSWORD - Database password"
    echo "  POSTGRES_DB       - Database name"
}

# Show usage for restore
show_restore_usage() {
    echo "Usage: restore [OPTIONS] ARCHIVE_FILE"
    echo ""
    echo "Restore from a backup archive."
    echo "Copy the backup file into the container first, then run restore."
    echo ""
    echo "Options:"
    echo "  --db-only          Restore database only (skip FAISS indexes)"
    echo "  --faiss-only       Restore FAISS indexes only (skip database)"
    echo "  --skip-migrations  Skip automatic schema migration after restore"
    echo "  --data-only        Restore data only (no schema), then run migrations"
    echo "  --include-secret   Restore the .encryption_key file if present"
    echo "                     (overwrites existing key - use with caution!)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Arguments:"
    echo "  ARCHIVE_FILE   Path to backup archive inside container (required)"
    echo ""
    echo "Schema Migration:"
    echo "  After restoring a database backup, Prisma migrations are automatically"
    echo "  run to bring the schema up to the current version. This handles backups"
    echo "  from older Ragtime versions that may have an outdated schema."
    echo ""
    echo "Examples:"
    echo "  docker cp backup.tar.gz ragtime:/tmp/backup.tar.gz"
    echo "  docker exec ragtime restore /tmp/backup.tar.gz"
    echo "  docker exec ragtime restore --include-secret /tmp/backup.tar.gz"
    echo "  docker exec ragtime restore --db-only /tmp/backup.tar.gz"
    echo ""
    echo "Environment variables required:"
    echo "  POSTGRES_USER     - Database user"
    echo "  POSTGRES_PASSWORD - Database password"
    echo "  POSTGRES_DB       - Database name"
}

# Cleanup function
cleanup() {
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}

trap cleanup EXIT

# Perform backup
do_backup() {
    local db_only=false
    local faiss_only=false
    local include_secret=false
    local to_stdout=true
    local output_file=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
        --db-only)
            db_only=true
            shift
            ;;
        --faiss-only)
            faiss_only=true
            shift
            ;;
        --include-secret)
            include_secret=true
            shift
            ;;
        -h | --help)
            show_backup_usage
            exit 0
            ;;
        -*)
            log "ERROR" "Unknown option: $1"
            show_backup_usage
            exit 1
            ;;
        *)
            output_file="$1"
            to_stdout=false
            shift
            ;;
        esac
    done

    # Validate options
    if [ "$db_only" = true ] && [ "$faiss_only" = true ]; then
        log "ERROR" "Cannot specify both --db-only and --faiss-only"
        exit 1
    fi

    # Check required environment variables
    if [ -z "$POSTGRES_DB" ] || [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ]; then
        log "ERROR" "Missing required environment variables (POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD)"
        exit 1
    fi

    # Determine backup type for metadata
    local backup_type="full"
    if [ "$db_only" = true ]; then
        backup_type="database-only"
    elif [ "$faiss_only" = true ]; then
        backup_type="faiss-only"
    fi

    # Create temp directory
    TEMP_DIR="${TEMP_BASE}_$(date +%s)_$$"
    mkdir -p "$TEMP_DIR"

    # Configure logging output
    # If streaming backup to stdout (to_stdout=true), redirect log messages to stderr (fd 2)
    # Default is stdout (fd 1)
    if [ "$to_stdout" = true ]; then
        LOG_FD=2
    fi

    # Include encryption key if requested
    local includes_secret=false
    if [ "$include_secret" = true ]; then
        if [ -f "${FAISS_DIR}/.encryption_key" ]; then
            includes_secret=true
            log "INFO" "Including encryption key file..."
        else
            log "WARNING" "--include-secret specified but no .encryption_key file found"
            log "DEBUG" "Encryption key may not have been generated yet"
        fi
    fi

    log "INFO" "Creating $backup_type backup..."

    # Step 1: Database dump (unless faiss-only)
    if [ "$faiss_only" = true ]; then
        touch "$TEMP_DIR/database.dump"
        log "DEBUG" "Skipping database (faiss-only mode)"
    else
        log "INFO" "Dumping database..."
        if ! PGPASSWORD="$POSTGRES_PASSWORD" pg_dump -Fc -h "$POSTGRES_HOST" -U "$POSTGRES_USER" "$POSTGRES_DB" >"$TEMP_DIR/database.dump" 2>/dev/null; then
            log "ERROR" "Database dump failed"
            exit 1
        fi
        local db_size=$(du -h "$TEMP_DIR/database.dump" | cut -f1)
        log "DEBUG" "Database dump: $db_size"
    fi

    # Step 2: Copy FAISS indexes (unless db-only)
    log "DEBUG" "FAISS directory: ${FAISS_DIR}"
    mkdir -p "$TEMP_DIR/faiss"
    if [ "$db_only" = true ]; then
        log "DEBUG" "Skipping FAISS indexes (db-only mode)"
    else
        if [ ! -d "$FAISS_DIR" ]; then
            log "WARNING" "FAISS directory not found: $FAISS_DIR"
        elif [ -z "$(ls -A "$FAISS_DIR" 2>/dev/null)" ]; then
            log "DEBUG" "No FAISS indexes to backup"
        else
            log "INFO" "Copying FAISS indexes..."
            shopt -s nullglob
            for dir in "$FAISS_DIR"/*/; do
                dirname=$(basename "$dir")
                if [ "$dirname" = "_tmp" ]; then
                    continue
                fi
                cp -r "$dir" "$TEMP_DIR/faiss/"
                log "DEBUG" "Added index directory: $dirname"
            done
            shopt -u nullglob
            local index_count=$(find "$TEMP_DIR/faiss" -maxdepth 1 -mindepth 1 -type d | wc -l)
            log "DEBUG" "FAISS indexes: $index_count indexes"
        fi
    fi

    # Step 3: Create metadata
    # Include schema version for restore compatibility checking
    local schema_version=""
    if [ "$faiss_only" != true ]; then
        schema_version=$(get_db_schema_version)
    fi

    cat >"$TEMP_DIR/backup-meta.json" <<EOF
{
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "1.1",
    "source": "cli",
    "type": "$backup_type",
    "ragtime_version": "2.0.0",
    "schema_version": "${schema_version:-unknown}",
    "current_schema_version": "$CURRENT_SCHEMA_VERSION",
    "includes_secret": $includes_secret
}
EOF

    # Copy .encryption_key file if requested and exists
    if [ "$includes_secret" = true ]; then
        cp "${FAISS_DIR}/.encryption_key" "$TEMP_DIR/.encryption_key"
        log "DEBUG" "Encryption key file included in backup"
    fi

    # Step 4: Create archive
    if [ "$to_stdout" = true ]; then
        log "INFO" "Streaming archive to stdout..."
        tar -czf - -C "$TEMP_DIR" .
        log "INFO" "Backup complete"
    else
        log "INFO" "Creating archive..."
        tar -czf "$output_file" -C "$TEMP_DIR" .
        local archive_size=$(du -h "$output_file" | cut -f1)
        log "INFO" "Backup complete: $output_file ($archive_size)"
    fi

    # Print encryption key for user to save if not included in backup
    # Try to get it from Python settings
    local enc_key=""
    if command -v python &>/dev/null; then
        enc_key=$(python -c "from ragtime.config import settings; print(settings.encryption_key)" 2>/dev/null || echo "")
    fi
    # Fall back to persisted file
    if [ -z "$enc_key" ] && [ -f "${FAISS_DIR}/.encryption_key" ]; then
        enc_key=$(cat "${FAISS_DIR}/.encryption_key" 2>/dev/null || echo "")
    fi

    if [ "$includes_secret" = true ]; then
        log "INFO" ""
        log "INFO" "Encryption key included in backup."
        log "DEBUG" "Use 'restore --include-secret' to restore the key file."
    elif [ -n "$enc_key" ]; then
        log "INFO" ""
        log "WARNING" "Backup does not include encryption key."
        log "WARNING" "Use 'backup --include-secret' to include the key file in future backups."
        log "DEBUG" "Or manually backup: ${FAISS_DIR}/.encryption_key"
    else
        log "INFO" ""
        log "WARNING" "Could not determine encryption key - secrets may not be recoverable."
    fi
}

# Perform restore
do_restore() {
    local db_only=false
    local faiss_only=false
    local archive_file=""
    local skip_migrations=false
    local data_only=false
    local include_secret=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
        --db-only)
            db_only=true
            shift
            ;;
        --faiss-only)
            faiss_only=true
            shift
            ;;
        --skip-migrations)
            skip_migrations=true
            shift
            ;;
        --data-only)
            data_only=true
            shift
            ;;
        --include-secret)
            include_secret=true
            shift
            ;;
        -h | --help)
            show_restore_usage
            exit 0
            ;;
        -*)
            log "ERROR" "Unknown option: $1"
            show_restore_usage
            exit 1
            ;;
        *)
            archive_file="$1"
            shift
            ;;
        esac
    done

    # Validate options
    if [ "$db_only" = true ] && [ "$faiss_only" = true ]; then
        log "ERROR" "Cannot specify both --db-only and --faiss-only"
        exit 1
    fi

    # Archive file is required
    if [ -z "$archive_file" ]; then
        log "ERROR" "Archive file is required"
        log "DEBUG" "Copy the backup into the container first:"
        log "DEBUG" "  docker cp backup.tar.gz ragtime:/tmp/backup.tar.gz"
        log "DEBUG" "  docker exec ragtime restore /tmp/backup.tar.gz"
        exit 1
    fi

    if [ ! -f "$archive_file" ]; then
        log "ERROR" "Archive file not found: $archive_file"
        exit 1
    fi

    # Check required environment variables
    if [ -z "$POSTGRES_DB" ] || [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ]; then
        log "ERROR" "Missing required environment variables (POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD)"
        exit 1
    fi

    # Create temp directory
    TEMP_DIR="${TEMP_BASE}_restore_$(date +%s)_$$"
    mkdir -p "$TEMP_DIR"

    log "INFO" "Extracting archive: $archive_file"

    if ! tar -xzf "$archive_file" -C "$TEMP_DIR"; then
        log "ERROR" "Failed to extract archive: $archive_file"
        exit 1
    fi

    # Verify archive structure
    if [ ! -f "$TEMP_DIR/database.dump" ] && [ ! -d "$TEMP_DIR/faiss" ]; then
        log "ERROR" "Invalid backup archive: missing database.dump and faiss directory"
        exit 1
    fi

    # Show metadata and check schema version
    local backup_schema_version=""
    local backup_date=""
    local backup_type_meta=""
    if [ -f "$TEMP_DIR/backup-meta.json" ]; then
        backup_date=$(grep -o '"created_at"[[:space:]]*:[[:space:]]*"[^"]*"' "$TEMP_DIR/backup-meta.json" | cut -d'"' -f4)
        backup_type_meta=$(grep -o '"type"[[:space:]]*:[[:space:]]*"[^"]*"' "$TEMP_DIR/backup-meta.json" | cut -d'"' -f4)
        backup_schema_version=$(grep -o '"schema_version"[[:space:]]*:[[:space:]]*"[^"]*"' "$TEMP_DIR/backup-meta.json" | cut -d'"' -f4)
        log "DEBUG" "Backup date: ${backup_date:-unknown}"
        log "DEBUG" "Backup type: ${backup_type_meta:-full}"
        if [ -n "$backup_schema_version" ] && [ "$backup_schema_version" != "unknown" ]; then
            log "DEBUG" "Backup schema version: $backup_schema_version"
            log "DEBUG" "Current schema version: $CURRENT_SCHEMA_VERSION"
            if [ "$backup_schema_version" != "$CURRENT_SCHEMA_VERSION" ]; then
                log "WARNING" "Schema version mismatch detected - migrations will be applied after restore"
            fi
        fi
    fi

    # Step 1: Restore database (unless faiss-only)
    if [ "$faiss_only" = true ]; then
        log "DEBUG" "Skipping database restore (faiss-only mode)"
    elif [ -f "$TEMP_DIR/database.dump" ] && [ -s "$TEMP_DIR/database.dump" ]; then
        log "INFO" "Restoring database..."

        if [ "$data_only" = true ]; then
            # Data-only restore: only restore data, not schema
            # Useful when schema might be incompatible
            log "DEBUG" "Data-only mode: restoring data without schema changes"
            if ! PGPASSWORD="$POSTGRES_PASSWORD" pg_restore -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
                --data-only --disable-triggers "$TEMP_DIR/database.dump" 2>/dev/null; then
                log "WARNING" "Some data restore errors occurred (this may be expected for schema mismatches)"
            fi
        else
            # Full restore with schema
            # Note: pg_restore may report errors for objects that don't exist yet,
            # which is normal for --clean --if-exists. We capture stderr to a temp file
            # to avoid hiding real errors while suppressing expected ones.
            local restore_errors=$(mktemp)
            if ! PGPASSWORD="$POSTGRES_PASSWORD" pg_restore -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
                --clean --if-exists "$TEMP_DIR/database.dump" 2>"$restore_errors"; then
                # Check if errors are just "does not exist" warnings
                if grep -qE "(does not exist|no matching)" "$restore_errors" && ! grep -qE "FATAL|PANIC" "$restore_errors"; then
                    log "WARNING" "Some restore warnings occurred (objects didn't exist yet - this is normal)"
                else
                    log "WARNING" "Some restore errors occurred:"
                    cat "$restore_errors" | head -20 >&2
                fi
            fi
            rm -f "$restore_errors"
        fi

        # Verify restore actually worked by checking a table exists and has data
        local table_check
        table_check=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c \
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name NOT LIKE '_prisma%'" 2>/dev/null | tr -d '[:space:]')
        if [ -z "$table_check" ] || [ "$table_check" -lt 5 ]; then
            log "ERROR" "Database restore verification failed - expected tables not found"
            log "ERROR" "The pg_restore command may have failed silently. Check connection settings."
            exit 1
        fi

        log "DEBUG" "Database restored successfully (verified $table_check tables)"

        # Run migrations if not skipped and not faiss-only
        if [ "$skip_migrations" = true ]; then
            log "DEBUG" "Skipping automatic schema migration (--skip-migrations)"
            log "WARNING" "You may need to run 'python -m prisma migrate deploy' manually"
        else
            # Check if _prisma_migrations table exists and has data
            local migrations_exist
            migrations_exist=$(check_migrations_table)
            local restored_version
            restored_version=$(get_db_schema_version)

            if [ "$migrations_exist" = "t" ] && [ -n "$restored_version" ]; then
                # Backup included migration history - schema was fully restored
                # Only run migrations if there are newer ones available
                if [ "$restored_version" \< "$CURRENT_SCHEMA_VERSION" ]; then
                    log "DEBUG" "Detected older schema version ($restored_version), applying newer migrations..."
                    run_migrations
                else
                    log "DEBUG" "Schema restored from backup is up to date ($restored_version)"
                fi
            elif [ -z "$restored_version" ]; then
                # No migrations table or empty - this might be a very old backup
                log "WARNING" "No migration history found in restored database"
                log "DEBUG" "Attempting to run migrations to establish current schema..."
                run_migrations
            else
                log "DEBUG" "Schema is up to date, no migrations needed"
            fi
        fi
    else
        log "WARNING" "No database dump found or database.dump is empty, skipping database restore"
    fi

    # Step 2: Restore FAISS indexes (unless db-only)
    if [ "$db_only" = true ]; then
        log "DEBUG" "Skipping FAISS restore (db-only mode)"
    elif [ -d "$TEMP_DIR/faiss" ] && [ "$(ls -A $TEMP_DIR/faiss 2>/dev/null)" ]; then
        log "INFO" "Restoring FAISS indexes..."

        # Ensure FAISS directory exists
        mkdir -p "$FAISS_DIR"

        # Copy index directories
        cp -r "$TEMP_DIR/faiss"/* "$FAISS_DIR/" 2>/dev/null || true

        local index_count=$(find "$FAISS_DIR" -maxdepth 1 -type d | wc -l)
        index_count=$((index_count - 1))
        log "DEBUG" "FAISS indexes restored: $index_count indexes"
    else
        log "WARNING" "No FAISS indexes found in backup, skipping FAISS restore"
    fi

    # Step 3: Restore .encryption_key file if present and requested
    # Support both old (.jwt_secret) and new (.encryption_key) file names for backwards compatibility
    local backup_key_file=""
    if [ -f "$TEMP_DIR/.encryption_key" ]; then
        backup_key_file="$TEMP_DIR/.encryption_key"
    elif [ -f "$TEMP_DIR/.jwt_secret" ]; then
        backup_key_file="$TEMP_DIR/.jwt_secret"
        log "DEBUG" "Found legacy .jwt_secret file in backup (will restore as .encryption_key)"
    fi

    if [ "$include_secret" = true ]; then
        if [ -n "$backup_key_file" ]; then
            if [ -f "${FAISS_DIR}/.encryption_key" ]; then
                log "WARNING" "Overwriting existing .encryption_key file with backup version"
            fi
            mkdir -p "$FAISS_DIR"
            cp "$backup_key_file" "${FAISS_DIR}/.encryption_key"
            chmod 600 "${FAISS_DIR}/.encryption_key"
            log "DEBUG" "Encryption key file restored"
            log "DEBUG" "Restart the container to use the restored encryption key"
        else
            log "WARNING" "--include-secret specified but no encryption key file in backup"
            log "WARNING" "The backup may not have been created with --include-secret"
        fi
    elif [ -n "$backup_key_file" ]; then
        log "DEBUG" "Backup contains encryption key file but --include-secret not specified"
        log "DEBUG" "Use 'restore --include-secret' to restore the encryption key"
    fi

    log "INFO" "Restore complete!"

    # Warn about encryption keys (only if secret wasn't restored)
    if [ "$include_secret" != true ] || [ -z "$backup_key_file" ]; then
        log "WARNING" "IMPORTANT: Encrypted secrets (API keys, passwords) require the same encryption key"
        log "WARNING" "that was used when the backup was created. If the encryption key has changed,"
        log "WARNING" "you will need to re-enter all API keys and passwords in the Settings UI."
    fi
}

# Main entry point
# Detect if called via symlink (backup or restore) or directly
SCRIPT_NAME=$(basename "$0")

case "$SCRIPT_NAME" in
backup)
    do_backup "$@"
    ;;
restore)
    do_restore "$@"
    ;;
*)
    # Called as backup-restore.sh with subcommand
    case "${1:-}" in
    backup)
        shift
        do_backup "$@"
        ;;
    restore)
        shift
        do_restore "$@"
        ;;
    *)
        echo "Ragtime Backup/Restore Tool"
        echo ""
        echo "Usage:"
        echo "  $0 backup [OPTIONS] [OUTPUT_FILE]"
        echo "  $0 restore [OPTIONS] [ARCHIVE_FILE]"
        echo ""
        echo "Or via convenience commands:"
        echo "  backup [OPTIONS]     # Streams to stdout"
        echo "  restore [OPTIONS]    # Reads from stdin"
        echo ""
        echo "Run 'backup --help' or 'restore --help' for more information."
        exit 1
        ;;
    esac
    ;;
esac
