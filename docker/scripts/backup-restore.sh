#!/bin/bash
set -e

# Backup and Restore Script for Ragtime
# Creates/restores backup archives containing database + FAISS indexes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    echo "  --db-only      Restore database only (skip FAISS indexes)"
    echo "  --faiss-only   Restore FAISS indexes only (skip database)"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "Arguments:"
    echo "  ARCHIVE_FILE   Path to backup archive inside container (rarely needed)"
    echo ""
    echo "Examples:"
    echo "  cat backup.tar.gz | restore            # Full restore (default: stdin)"
    echo "  cat backup.tar.gz | restore --db-only  # Database only"
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
    cat > "$TEMP_DIR/backup-meta.json" << EOF
{
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "1.0",
    "source": "cli",
    "type": "$backup_type",
    "ragtime_version": "2.0.0"
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

    # Show metadata if available
    if [ -f "$TEMP_DIR/backup-meta.json" ]; then
        local backup_date=$(grep -o '"created_at"[[:space:]]*:[[:space:]]*"[^"]*"' "$TEMP_DIR/backup-meta.json" | cut -d'"' -f4)
        local backup_type=$(grep -o '"type"[[:space:]]*:[[:space:]]*"[^"]*"' "$TEMP_DIR/backup-meta.json" | cut -d'"' -f4)
        info "Backup date: ${backup_date:-unknown}"
        info "Backup type: ${backup_type:-full}"
    fi

    # Step 1: Restore database (unless faiss-only)
    if [ "$faiss_only" = true ]; then
        info "Skipping database restore (faiss-only mode)"
    elif [ -f "$TEMP_DIR/database.dump" ] && [ -s "$TEMP_DIR/database.dump" ]; then
        log "Restoring database..."

        if ! PGPASSWORD="$POSTGRES_PASSWORD" pg_restore -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
            --clean --if-exists "$TEMP_DIR/database.dump" 2>/dev/null; then
            warn "Some restore errors occurred (this is normal if objects don't exist yet)"
        fi

        info "Database restored successfully"
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
