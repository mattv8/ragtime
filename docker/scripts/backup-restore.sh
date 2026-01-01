#!/bin/bash
set -e

# Backup and Restore Script for Ragtime
# Creates/restores backup archives containing database + FAISS indexes
# Handles schema migrations for backups from older versions

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Current schema version (update when adding migrations)
# Format: YYYYMMDDHHMMSS of latest migration
CURRENT_SCHEMA_VERSION="20251231180000"

log() {
    echo -e "${GREEN}[BACKUP]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Paths
FAISS_DIR="/ragtime/data"
TEMP_BASE="/tmp/ragtime_backup"
PRISMA_DIR="/ragtime/prisma"

# Parse DATABASE_URL if individual vars aren't set
# Format: postgresql://user:password@host:port/database
parse_database_url() {
    if [ -n "$DATABASE_URL" ]; then
        # Extract components from DATABASE_URL
        local url="${DATABASE_URL#postgresql://}"
        local userpass="${url%%@*}"
        local hostdb="${url#*@}"

        POSTGRES_USER="${POSTGRES_USER:-${userpass%%:*}}"
        POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-${userpass#*:}}"

        local hostport="${hostdb%%/*}"
        POSTGRES_HOST="${POSTGRES_HOST:-${hostport%%:*}}"
        POSTGRES_DB="${POSTGRES_DB:-${hostdb#*/}}"
    fi
}

# Initialize database connection vars
parse_database_url

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
    if [ -d "$PRISMA_DIR" ] && command -v python &> /dev/null; then
        info "Running Prisma migrations to update schema..."
        if python -m prisma migrate deploy 2>&1; then
            info "Schema migrations applied successfully"
            return 0
        else
            warn "Some migrations may have failed - check manually"
            return 1
        fi
    else
        warn "Prisma not available - skipping schema migration"
        warn "Run 'python -m prisma migrate deploy' manually after restore"
        return 1
    fi
}

# Show usage for backup
show_backup_usage() {
    echo "Usage: backup [OPTIONS] [OUTPUT_FILE]"
    echo ""
    echo "Create a backup archive containing database and FAISS indexes."
    echo "By default, streams to stdout for easy piping to local file."
    echo ""
    echo "Options:"
    echo "  --db-only      Backup database only (no FAISS indexes)"
    echo "  --faiss-only   Backup FAISS indexes only (no database)"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "Arguments:"
    echo "  OUTPUT_FILE    Output file path inside container (rarely needed)"
    echo ""
    echo "Examples:"
    echo "  backup > backup.tar.gz              # Full backup (default: stdout)"
    echo "  backup --db-only > db.tar.gz        # Database only"
    echo "  backup --faiss-only > faiss.tar.gz  # FAISS indexes only"
    echo ""
    echo "Environment variables required:"
    echo "  POSTGRES_USER     - Database user"
    echo "  POSTGRES_PASSWORD - Database password"
    echo "  POSTGRES_DB       - Database name"
}

# Show usage for restore
show_restore_usage() {
    echo "Usage: restore [OPTIONS] [ARCHIVE_FILE]"
    echo ""
    echo "Restore from a backup archive."
    echo "By default, reads from stdin for easy piping."
    echo ""
    echo "Options:"
    echo "  --db-only          Restore database only (skip FAISS indexes)"
    echo "  --faiss-only       Restore FAISS indexes only (skip database)"
    echo "  --skip-migrations  Skip automatic schema migration after restore"
    echo "  --data-only        Restore data only (no schema), then run migrations"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Arguments:"
    echo "  ARCHIVE_FILE   Path to backup archive inside container (rarely needed)"
    echo ""
    echo "Schema Migration:"
    echo "  After restoring a database backup, Prisma migrations are automatically"
    echo "  run to bring the schema up to the current version. This handles backups"
    echo "  from older Ragtime versions that may have an outdated schema."
    echo ""
    echo "Examples:"
    echo "  cat backup.tar.gz | restore            # Full restore (default: stdin)"
    echo "  cat backup.tar.gz | restore --db-only  # Database only"
    echo "  cat backup.tar.gz | restore --skip-migrations  # Skip auto-migration"
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
            -h|--help)
                show_backup_usage
                exit 0
                ;;
            -*)
                error "Unknown option: $1"
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
        error "Cannot specify both --db-only and --faiss-only"
        exit 1
    fi

    # Check required environment variables
    if [ -z "$POSTGRES_DB" ] || [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ]; then
        error "Missing required environment variables (POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD)"
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

    # Helper to log to stderr when streaming
    log_msg() {
        if [ "$to_stdout" = true ]; then
            echo -e "$1" >&2
        else
            echo -e "$1"
        fi
    }

    log_msg "${GREEN}[BACKUP]${NC} Creating $backup_type backup..."

    # Step 1: Database dump (unless faiss-only)
    if [ "$faiss_only" = true ]; then
        touch "$TEMP_DIR/database.dump"
        log_msg "${BLUE}[INFO]${NC} Skipping database (faiss-only mode)"
    else
        log_msg "${GREEN}[BACKUP]${NC} Dumping database..."
        if ! PGPASSWORD="$POSTGRES_PASSWORD" pg_dump -Fc -h "$POSTGRES_HOST" -U "$POSTGRES_USER" "$POSTGRES_DB" > "$TEMP_DIR/database.dump" 2>/dev/null; then
            echo -e "${RED}[ERROR]${NC} Database dump failed" >&2
            exit 1
        fi
        local db_size=$(du -h "$TEMP_DIR/database.dump" | cut -f1)
        log_msg "${BLUE}[INFO]${NC} Database dump: $db_size"
    fi

    # Step 2: Copy FAISS indexes (unless db-only)
    mkdir -p "$TEMP_DIR/faiss"
    if [ "$db_only" = true ]; then
        log_msg "${BLUE}[INFO]${NC} Skipping FAISS indexes (db-only mode)"
    else
        if [ -d "$FAISS_DIR" ] && [ "$(ls -A $FAISS_DIR 2>/dev/null)" ]; then
            log_msg "${GREEN}[BACKUP]${NC} Copying FAISS indexes..."
            # Copy all index directories (exclude _tmp)
            for dir in "$FAISS_DIR"/*/; do
                dirname=$(basename "$dir")
                if [ "$dirname" != "_tmp" ] && [ -d "$dir" ]; then
                    cp -r "$dir" "$TEMP_DIR/faiss/"
                fi
            done
            local index_count=$(find "$TEMP_DIR/faiss" -maxdepth 1 -type d | wc -l)
            index_count=$((index_count - 1))  # Subtract 1 for the faiss dir itself
            log_msg "${BLUE}[INFO]${NC} FAISS indexes: $index_count indexes"
        else
            log_msg "${BLUE}[INFO]${NC} No FAISS indexes to backup"
        fi
    fi

    # Step 3: Create metadata
    # Include schema version for restore compatibility checking
    local schema_version=""
    if [ "$faiss_only" != true ]; then
        schema_version=$(get_db_schema_version)
    fi

    cat > "$TEMP_DIR/backup-meta.json" << EOF
{
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "1.1",
    "source": "cli",
    "type": "$backup_type",
    "ragtime_version": "2.0.0",
    "schema_version": "${schema_version:-unknown}",
    "current_schema_version": "$CURRENT_SCHEMA_VERSION"
}
EOF

    # Step 4: Create archive
    if [ "$to_stdout" = true ]; then
        log_msg "${GREEN}[BACKUP]${NC} Streaming archive to stdout..."
        tar -czf - -C "$TEMP_DIR" .
        log_msg "${GREEN}[BACKUP]${NC} Backup complete"
    else
        log_msg "${GREEN}[BACKUP]${NC} Creating archive..."
        tar -czf "$output_file" -C "$TEMP_DIR" .
        local archive_size=$(du -h "$output_file" | cut -f1)
        log "Backup complete: $output_file ($archive_size)"
    fi
}

# Perform restore
do_restore() {
    local db_only=false
    local faiss_only=false
    local from_stdin=true
    local archive_file=""
    local skip_migrations=false
    local data_only=false

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
            -h|--help)
                show_restore_usage
                exit 0
                ;;
            -*)
                error "Unknown option: $1"
                show_restore_usage
                exit 1
                ;;
            *)
                archive_file="$1"
                from_stdin=false
                shift
                ;;
        esac
    done

    # Validate options
    if [ "$db_only" = true ] && [ "$faiss_only" = true ]; then
        error "Cannot specify both --db-only and --faiss-only"
        exit 1
    fi

    if [ "$from_stdin" = false ] && [ ! -f "$archive_file" ]; then
        error "Archive file not found: $archive_file"
        exit 1
    fi

    # Check required environment variables
    if [ -z "$POSTGRES_DB" ] || [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ]; then
        error "Missing required environment variables (POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD)"
        exit 1
    fi

    # Create temp directory
    TEMP_DIR="${TEMP_BASE}_restore_$(date +%s)_$$"
    mkdir -p "$TEMP_DIR"

    log "Extracting archive..."

    if [ "$from_stdin" = true ]; then
        tar -xzf - -C "$TEMP_DIR"
    else
        tar -xzf "$archive_file" -C "$TEMP_DIR"
    fi

    # Verify archive structure
    if [ ! -f "$TEMP_DIR/database.dump" ] && [ ! -d "$TEMP_DIR/faiss" ]; then
        error "Invalid backup archive: missing database.dump and faiss directory"
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
        info "Backup date: ${backup_date:-unknown}"
        info "Backup type: ${backup_type_meta:-full}"
        if [ -n "$backup_schema_version" ] && [ "$backup_schema_version" != "unknown" ]; then
            info "Backup schema version: $backup_schema_version"
            info "Current schema version: $CURRENT_SCHEMA_VERSION"
            if [ "$backup_schema_version" != "$CURRENT_SCHEMA_VERSION" ]; then
                warn "Schema version mismatch detected - migrations will be applied after restore"
            fi
        fi
    fi

    # Step 1: Restore database (unless faiss-only)
    if [ "$faiss_only" = true ]; then
        info "Skipping database restore (faiss-only mode)"
    elif [ -f "$TEMP_DIR/database.dump" ] && [ -s "$TEMP_DIR/database.dump" ]; then
        log "Restoring database..."

        if [ "$data_only" = true ]; then
            # Data-only restore: only restore data, not schema
            # Useful when schema might be incompatible
            info "Data-only mode: restoring data without schema changes"
            if ! PGPASSWORD="$POSTGRES_PASSWORD" pg_restore -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
                --data-only --disable-triggers "$TEMP_DIR/database.dump" 2>/dev/null; then
                warn "Some data restore errors occurred (this may be expected for schema mismatches)"
            fi
        else
            # Full restore with schema
            if ! PGPASSWORD="$POSTGRES_PASSWORD" pg_restore -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
                --clean --if-exists "$TEMP_DIR/database.dump" 2>/dev/null; then
                warn "Some restore errors occurred (this is normal if objects don't exist yet)"
            fi
        fi

        info "Database restored successfully"

        # Run migrations if not skipped and not faiss-only
        if [ "$skip_migrations" = true ]; then
            info "Skipping automatic schema migration (--skip-migrations)"
            warn "You may need to run 'python -m prisma migrate deploy' manually"
        else
            # Check if we need migrations
            local restored_version
            restored_version=$(get_db_schema_version)
            if [ -n "$restored_version" ] && [ "$restored_version" != "$CURRENT_SCHEMA_VERSION" ]; then
                info "Detected older schema version ($restored_version), applying migrations..."
                run_migrations
            elif [ -z "$restored_version" ]; then
                # No migrations table or empty - this might be a very old backup
                warn "No migration history found in restored database"
                info "Attempting to run migrations to establish current schema..."
                run_migrations
            else
                info "Schema is up to date, no migrations needed"
            fi
        fi
    else
        warn "No database dump found or database.dump is empty, skipping database restore"
    fi

    # Step 2: Restore FAISS indexes (unless db-only)
    if [ "$db_only" = true ]; then
        info "Skipping FAISS restore (db-only mode)"
    elif [ -d "$TEMP_DIR/faiss" ] && [ "$(ls -A $TEMP_DIR/faiss 2>/dev/null)" ]; then
        log "Restoring FAISS indexes..."

        # Ensure FAISS directory exists
        mkdir -p "$FAISS_DIR"

        # Copy index directories
        cp -r "$TEMP_DIR/faiss"/* "$FAISS_DIR/" 2>/dev/null || true

        local index_count=$(find "$FAISS_DIR" -maxdepth 1 -type d | wc -l)
        index_count=$((index_count - 1))
        info "FAISS indexes restored: $index_count indexes"
    else
        warn "No FAISS indexes found in backup, skipping FAISS restore"
    fi

    log "Restore complete!"
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
