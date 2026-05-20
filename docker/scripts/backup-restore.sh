#!/bin/bash
set -euo pipefail

# Backup and Restore Script for Ragtime
# Creates/restores archives containing the PostgreSQL database and Ragtime data
# directory (indexes, userspace workspaces, SSL/runtime files, and optional
# encryption key). Handles schema migrations for backups from older versions.
#
# IMPORTANT: Secrets Encryption
# -----------------------------
# Ragtime encrypts sensitive data (API keys, passwords) using Fernet symmetric
# encryption. The encryption key is auto-generated on first startup and stored
# at data/.encryption_key.
#
# To ensure backups can be restored with working secrets:
#   1. Use --include-secret to include the encryption key in backups
#   2. Or manually backup the .encryption_key file
#   3. Or re-enter all passwords/API keys after restore
#
# Without the same encryption key, encrypted secrets become unrecoverable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/functions.sh" ]; then
    source "$SCRIPT_DIR/functions.sh"
elif [ -f "/docker-scripts/functions.sh" ]; then
    source "/docker-scripts/functions.sh"
fi

if ! declare -F log >/dev/null; then
    log() {
        local level=$1
        shift
        printf '%s | %-8s | %s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$level" "${LOG_PREFIX:-BACKUP}" "$*" >&"${LOG_FD:-1}"
    }
fi

export LOG_PREFIX="${LOG_PREFIX:-BACKUP}"

BACKUP_FORMAT_VERSION="2.0"
TEMP_BASE="${TEMP_BASE:-/tmp/ragtime_backup}"
PRISMA_DIR="${PRISMA_DIR:-/ragtime/prisma}"
DATA_DIR="${INDEX_DATA_PATH:-${FAISS_DIR:-/data}}"

DATABASE_URL_PARSED=false
if declare -F parse_database_url >/dev/null; then
    set +u
    if parse_database_url; then
        DATABASE_URL_PARSED=true
    fi
    set -u
fi

if [ "$DATABASE_URL_PARSED" = true ]; then
    POSTGRES_USER="${DB_USER:-}"
    POSTGRES_PASSWORD="${DB_PASS:-}"
    POSTGRES_HOST="${DB_HOST:-localhost}"
    POSTGRES_PORT="${DB_PORT:-5432}"
    POSTGRES_DB="${DB_NAME:-}"
else
    POSTGRES_USER="${POSTGRES_USER:-}"
    POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"
    POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
    POSTGRES_PORT="${POSTGRES_PORT:-5432}"
    POSTGRES_DB="${POSTGRES_DB:-}"
fi

TEMP_DIR=""

cleanup() {
    if [ -n "${TEMP_DIR:-}" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}

trap cleanup EXIT

die() {
    log "ERROR" "$*"
    exit 1
}

require_commands() {
    local missing=()
    local cmd
    for cmd in "$@"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing+=("$cmd")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        die "Missing required command(s): ${missing[*]}"
    fi
}

require_database_env() {
    if [ -z "$POSTGRES_DB" ] || [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ] || [ -z "$POSTGRES_HOST" ]; then
        die "Missing required database settings. Set DATABASE_URL or POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_HOST."
    fi
}

confirm_restore() {
    local confirmation_input=$1
    local confirmation_phrase="${RESTORE_CONFIRMATION_PHRASE:-RESTORE ${POSTGRES_DB:-RAGTIME}}"
    local typed_confirmation=""

    log "WARNING" "Restore is destructive and can replace the database and/or Ragtime data directory."
    if [ -n "${POSTGRES_DB:-}" ]; then
        log "WARNING" "Target database: $POSTGRES_DB on $POSTGRES_HOST:$POSTGRES_PORT"
    fi
    if [ -n "${DATA_DIR:-}" ]; then
        log "WARNING" "Target data directory: $DATA_DIR"
    fi

    if [ -n "$confirmation_input" ]; then
        typed_confirmation="$confirmation_input"
    elif [ -r /dev/tty ]; then
        printf "Type '%s' to continue with restore: " "$confirmation_phrase" >/dev/tty
        IFS= read -r typed_confirmation </dev/tty
    else
        die "Restore confirmation required. Re-run with --confirm-restore '$confirmation_phrase' or set RESTORE_CONFIRMATION='$confirmation_phrase'."
    fi

    if [ "$typed_confirmation" != "$confirmation_phrase" ]; then
        die "Restore confirmation did not match '$confirmation_phrase'; aborting restore"
    fi

    log "INFO" "Restore confirmation accepted"
}

psql_db() {
    PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -X \
        -v ON_ERROR_STOP=1 \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        "$@"
}

pg_dump_db() {
    PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
        -Fc \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        "$POSTGRES_DB"
}

pg_restore_db() {
    PGPASSWORD="$POSTGRES_PASSWORD" pg_restore \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        "$@"
}

get_current_schema_version() {
    local latest=""
    local migration_dir
    local migration_name
    local migration_version

    if [ -d "$PRISMA_DIR/migrations" ]; then
        shopt -s nullglob
        for migration_dir in "$PRISMA_DIR"/migrations/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*; do
            [ -d "$migration_dir" ] || continue
            migration_name="$(basename "$migration_dir")"
            migration_version="${migration_name%%_*}"
            if [[ "$migration_version" =~ ^[0-9]{14}$ ]] && { [ -z "$latest" ] || [[ "$migration_version" > "$latest" ]]; }; then
                latest="$migration_version"
            fi
        done
        shopt -u nullglob
    fi

    echo "${latest:-unknown}"
}

CURRENT_SCHEMA_VERSION="${CURRENT_SCHEMA_VERSION:-$(get_current_schema_version)}"

check_migrations_table() {
    psql_db -t -c "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '_prisma_migrations')" 2>/dev/null | tr -d '[:space:]' || true
}

get_db_schema_version() {
    local migrations_exist
    local version

    migrations_exist="$(check_migrations_table)"
    if [ "$migrations_exist" != "t" ]; then
        echo ""
        return 0
    fi

    version="$(psql_db -t -c "SELECT migration_name FROM _prisma_migrations WHERE finished_at IS NOT NULL ORDER BY finished_at DESC LIMIT 1" 2>/dev/null | tr -d '[:space:]' || true)"
    if [ -n "$version" ]; then
        echo "${version:0:14}"
    else
        echo ""
    fi
}

run_migrations() {
    if [ "$CURRENT_SCHEMA_VERSION" = "unknown" ]; then
        log "WARNING" "Current schema version could not be determined from $PRISMA_DIR/migrations"
    fi

    if [ -d "$PRISMA_DIR" ] && command -v python >/dev/null 2>&1; then
        log "INFO" "Running Prisma migrations to update schema..."
        if (cd /ragtime && python -m prisma migrate deploy 2>&1); then
            log "INFO" "Schema migrations applied successfully"
            return 0
        fi
        log "ERROR" "Prisma migrations failed"
        return 1
    fi

    log "WARNING" "Prisma is not available - skipping schema migration"
    log "WARNING" "Run 'python -m prisma migrate deploy' manually after restore"
    return 1
}

read_meta_field() {
    local meta_file=$1
    local field=$2

    python - "$meta_file" "$field" <<'PY' 2>/dev/null || true
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        value = json.load(handle).get(sys.argv[2], "")
except Exception:
    value = ""

if isinstance(value, bool):
    print("true" if value else "false")
elif value is not None:
    print(value)
PY
}

validate_tar_archive() {
    local archive_file=$1
    local entry
    local normalized

    require_commands tar
    if ! tar -tzf "$archive_file" >/dev/null; then
        die "Invalid or unreadable tar.gz archive: $archive_file"
    fi

    while IFS= read -r entry; do
        normalized="${entry#./}"
        if [[ "$entry" = /* ]] || [[ "$normalized" = ".." ]] || [[ "$normalized" = ../* ]] || [[ "$normalized" = */../* ]]; then
            die "Unsafe path in archive: $entry"
        fi
    done < <(tar -tzf "$archive_file")
}

terminate_other_database_connections() {
    psql_db -q -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid()" >/dev/null 2>&1 || true
}

normalize_local_admin_username() {
    local username=$1
    if [[ "$username" == local:* ]]; then
        echo "$username"
    else
        echo "local:$username"
    fi
}

select_admin_access_mirror_source() {
    local target_username=$1

    psql_db -At -v target_username="$target_username" <<'SQL' | head -n 1
WITH candidate_scores AS (
    SELECT
        u.username,
        COUNT(DISTINCT owned_workspaces.id) AS owned_workspace_count,
        COUNT(DISTINCT workspace_access.workspace_id) AS workspace_access_count,
        COUNT(DISTINCT owned_conversations.id) AS owned_conversation_count,
        COUNT(DISTINCT conversation_access.conversation_id) AS conversation_access_count
    FROM users u
    LEFT JOIN workspaces owned_workspaces ON owned_workspaces.owner_user_id = u.id
    LEFT JOIN workspace_members workspace_access ON workspace_access.user_id = u.id
    LEFT JOIN conversations owned_conversations ON owned_conversations.user_id = u.id
    LEFT JOIN conversation_members conversation_access ON conversation_access.user_id = u.id
    WHERE u.role = 'admin'
      AND u.username <> :'target_username'
      AND u.username NOT LIKE 'local:%'
    GROUP BY u.id, u.username
)
SELECT username
FROM candidate_scores
ORDER BY
    (owned_workspace_count + workspace_access_count + owned_conversation_count + conversation_access_count) DESC,
    owned_conversation_count DESC,
    workspace_access_count DESC,
    username ASC
LIMIT 1;
SQL
}

mirror_local_admin_access() {
    local source_username=$1
    local target_username=$2
    local selected_source=""
    local summary=""

    require_database_env

    target_username="$(normalize_local_admin_username "$target_username")"

    log "INFO" "Ensuring local admin user exists for access mirror: $target_username"
    psql_db -v target_username="$target_username" >/dev/null <<'SQL'
INSERT INTO users (
    id,
    username,
    auth_provider,
    cached_groups,
    display_name,
    role,
    role_manually_set,
    created_at,
    updated_at
)
VALUES (
    gen_random_uuid()::text,
    :'target_username',
    'local'::"AuthProvider",
    '[]'::jsonb,
    'Local Admin',
    'admin'::"UserRole",
    false,
    now(),
    now()
)
ON CONFLICT (username) DO UPDATE
SET role = 'admin'::"UserRole",
    updated_at = now();
SQL

    if [ -z "$source_username" ] || [ "$source_username" = "auto" ]; then
        selected_source="$(select_admin_access_mirror_source "$target_username" | tr -d '[:space:]')"
        if [ -z "$selected_source" ]; then
            log "WARNING" "No backed-up non-local admin user found to mirror access from"
            return 0
        fi
    else
        selected_source="$source_username"
    fi

    if [ "$selected_source" = "$target_username" ]; then
        log "WARNING" "Local admin access mirror source and target are the same user ($target_username); skipping"
        return 0
    fi

    if ! psql_db -At -v source_username="$selected_source" <<'SQL' | grep -q '^1$'; then
SELECT 1 FROM users WHERE username = :'source_username';
SQL
        die "Cannot mirror local admin access: source user not found: $selected_source"
    fi

    log "INFO" "Mirroring access from backed-up admin '$selected_source' to local admin '$target_username'"
    summary="$(psql_db -At -v source_username="$selected_source" -v target_username="$target_username" <<'SQL'
WITH source_user AS (
    SELECT id FROM users WHERE username = :'source_username'
),
target_user AS (
    SELECT id FROM users WHERE username = :'target_username'
),
workspace_source_access AS (
    SELECT w.id AS workspace_id, 'owner'::"WorkspaceRole" AS role
    FROM workspaces w
    JOIN source_user source ON source.id = w.owner_user_id
    UNION ALL
    SELECT wm.workspace_id, wm.role
    FROM workspace_members wm
    JOIN source_user source ON source.id = wm.user_id
),
workspace_ranked_access AS (
    SELECT DISTINCT ON (workspace_id)
        workspace_id,
        role
    FROM workspace_source_access
    ORDER BY workspace_id,
        CASE role
            WHEN 'owner'::"WorkspaceRole" THEN 3
            WHEN 'editor'::"WorkspaceRole" THEN 2
            ELSE 1
        END DESC
),
workspace_upsert AS (
    INSERT INTO workspace_members (id, workspace_id, user_id, role, created_at, updated_at)
    SELECT gen_random_uuid()::text, access.workspace_id, target.id, access.role, now(), now()
    FROM workspace_ranked_access access
    CROSS JOIN target_user target
    ON CONFLICT (workspace_id, user_id) DO UPDATE
    SET role = CASE
            WHEN CASE EXCLUDED.role
                    WHEN 'owner'::"WorkspaceRole" THEN 3
                    WHEN 'editor'::"WorkspaceRole" THEN 2
                    ELSE 1
                 END
               > CASE workspace_members.role
                    WHEN 'owner'::"WorkspaceRole" THEN 3
                    WHEN 'editor'::"WorkspaceRole" THEN 2
                    ELSE 1
                 END
            THEN EXCLUDED.role
            ELSE workspace_members.role
        END,
        updated_at = now()
    RETURNING 1
),
conversation_source_access AS (
    SELECT c.id AS conversation_id, 'owner'::"WorkspaceRole" AS role
    FROM conversations c
    JOIN source_user source ON source.id = c.user_id
    UNION ALL
    SELECT cm.conversation_id, cm.role
    FROM conversation_members cm
    JOIN source_user source ON source.id = cm.user_id
),
conversation_ranked_access AS (
    SELECT DISTINCT ON (conversation_id)
        conversation_id,
        role
    FROM conversation_source_access
    ORDER BY conversation_id,
        CASE role
            WHEN 'owner'::"WorkspaceRole" THEN 3
            WHEN 'editor'::"WorkspaceRole" THEN 2
            ELSE 1
        END DESC
),
conversation_upsert AS (
    INSERT INTO conversation_members (id, conversation_id, user_id, role, created_at, updated_at)
    SELECT gen_random_uuid()::text, access.conversation_id, target.id, access.role, now(), now()
    FROM conversation_ranked_access access
    CROSS JOIN target_user target
    ON CONFLICT (conversation_id, user_id) DO UPDATE
    SET role = CASE
            WHEN CASE EXCLUDED.role
                    WHEN 'owner'::"WorkspaceRole" THEN 3
                    WHEN 'editor'::"WorkspaceRole" THEN 2
                    ELSE 1
                 END
               > CASE conversation_members.role
                    WHEN 'owner'::"WorkspaceRole" THEN 3
                    WHEN 'editor'::"WorkspaceRole" THEN 2
                    ELSE 1
                 END
            THEN EXCLUDED.role
            ELSE conversation_members.role
        END,
        updated_at = now()
    RETURNING 1
),
auth_group_upsert AS (
    INSERT INTO auth_group_memberships (id, user_id, group_id, source_provider, source_synced_at, created_at, updated_at)
    SELECT gen_random_uuid()::text, target.id, membership.group_id, 'local_managed'::"AuthProvider", now(), now(), now()
    FROM auth_group_memberships membership
    JOIN source_user source ON source.id = membership.user_id
    CROSS JOIN target_user target
    ON CONFLICT (user_id, group_id) DO UPDATE
    SET updated_at = now()
    RETURNING 1
)
SELECT
    (SELECT COUNT(*) FROM workspace_upsert) || ' workspace memberships, ' ||
    (SELECT COUNT(*) FROM conversation_upsert) || ' conversation memberships, ' ||
    (SELECT COUNT(*) FROM auth_group_upsert) || ' auth group memberships';
SQL
)"
    log "INFO" "Local admin access mirror complete: $summary"
}

copy_data_to_temp() {
    local include_secret=$1
    local data_size
    local data_entries
    local tar_args=(
        --exclude='./_tmp'
        # Runtime sandboxes recreate rootfs trees and may contain devices/FIFOs.
        --exclude='./_userspace/workspaces/*/rootfs'
        # Bootstrap stamps certify local dependency caches, so they are not portable.
        --exclude='./_userspace/workspaces/*/files/.ragtime/.runtime-bootstrap.done'
    )

    mkdir -p "$TEMP_DIR/data"

    if [ ! -d "$DATA_DIR" ]; then
        log "WARNING" "Ragtime data directory not found: $DATA_DIR"
        return 0
    fi

    if [ "$include_secret" != true ]; then
        tar_args+=(--exclude='./.encryption_key' --exclude='./.jwt_secret')
    fi

    log "INFO" "Copying Ragtime data directory: $DATA_DIR"
    if ! tar -C "$DATA_DIR" "${tar_args[@]}" -cf - . | tar -C "$TEMP_DIR/data" -xf -; then
        die "Failed to copy Ragtime data directory"
    fi

    data_entries="$(find "$TEMP_DIR/data" -mindepth 1 -maxdepth 1 | wc -l | tr -d '[:space:]')"
    data_size="$(du -sh "$TEMP_DIR/data" | awk '{print $1}')"
    log "DEBUG" "Ragtime data: $data_entries top-level entries, $data_size"
}

restore_data_from_backup() {
    local extract_dir=$1
    local include_secret=$2
    local replace_data=$3
    local data_source=""
    local restored_size

    if [ -d "$extract_dir/data" ]; then
        data_source="$extract_dir/data"
    elif [ -d "$extract_dir/faiss" ]; then
        data_source="$extract_dir/faiss"
        log "WARNING" "Restoring legacy faiss/ archive contents into $DATA_DIR"
    else
        log "WARNING" "No Ragtime data directory found in backup, skipping data restore"
        return 0
    fi

    mkdir -p "$DATA_DIR"

    if [ "$include_secret" != true ]; then
        rm -f "$data_source/.encryption_key" "$data_source/.jwt_secret"
    fi

    if [ "$replace_data" = true ]; then
        log "WARNING" "Replacing existing contents of $DATA_DIR"
        if [ "$include_secret" = true ]; then
            find "$DATA_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
        else
            find "$DATA_DIR" -mindepth 1 -maxdepth 1 ! -name '.encryption_key' ! -name '.jwt_secret' -exec rm -rf {} +
        fi
    else
        log "INFO" "Merging backup data into $DATA_DIR (use --replace-existing-data for an exact data-directory restore)"
    fi

    if ! tar -C "$data_source" -cf - . | tar -C "$DATA_DIR" -xf -; then
        die "Failed to restore Ragtime data directory"
    fi

    if [ "$include_secret" = true ] && [ -f "$DATA_DIR/.encryption_key" ]; then
        chmod 600 "$DATA_DIR/.encryption_key"
    fi

    restored_size="$(du -sh "$DATA_DIR" | awk '{print $1}')"
    log "INFO" "Ragtime data restored to $DATA_DIR ($restored_size)"
}

invalidate_restored_workspace_runtime_artifacts() {
    local extract_dir=$1
    local data_source=""
    local workspace_source_root=""
    local workspace_source=""
    local workspace_id=""
    local workspace_target=""
    local removed_stamps=0
    local removed_rootfs=0

    if [ -d "$extract_dir/data" ]; then
        data_source="$extract_dir/data"
    elif [ -d "$extract_dir/faiss" ]; then
        data_source="$extract_dir/faiss"
    else
        return 0
    fi

    workspace_source_root="$data_source/_userspace/workspaces"
    if [ ! -d "$workspace_source_root" ]; then
        return 0
    fi

    shopt -s nullglob
    for workspace_source in "$workspace_source_root"/*; do
        [ -d "$workspace_source" ] || continue
        workspace_id="$(basename "$workspace_source")"
        workspace_target="$DATA_DIR/_userspace/workspaces/$workspace_id"

        if [ -f "$workspace_target/files/.ragtime/.runtime-bootstrap.done" ]; then
            rm -f "$workspace_target/files/.ragtime/.runtime-bootstrap.done"
            removed_stamps=$((removed_stamps + 1))
        fi

        if [ -d "$workspace_target/rootfs" ]; then
            rm -rf "$workspace_target/rootfs"
            removed_rootfs=$((removed_rootfs + 1))
        fi
    done
    shopt -u nullglob

    if [ "$removed_stamps" -gt 0 ] || [ "$removed_rootfs" -gt 0 ]; then
        log "INFO" "Invalidated restored workspace runtime artifacts: $removed_stamps bootstrap stamp(s), $removed_rootfs rootfs cache(s)"
        log "DEBUG" "Workspace runtime bootstraps will rerun lazily on next preview/session start"
    fi
}

invalidate_restored_runtime_sessions() {
    local active_count=""
    local invalidated_count=""

    require_database_env

    if [ "$(psql_db -At -c "SELECT to_regclass('public.userspace_runtime_sessions') IS NOT NULL" 2>/dev/null | tr -d '[:space:]' || true)" != "t" ]; then
        return 0
    fi

    active_count="$(psql_db -At <<'SQL' | tr -d '[:space:]'
SELECT COUNT(*)
FROM userspace_runtime_sessions
WHERE state IN ('starting'::"RuntimeSessionState", 'running'::"RuntimeSessionState", 'stopping'::"RuntimeSessionState");
SQL
)"
    if [ -z "$active_count" ] || [ "$active_count" -eq 0 ]; then
        return 0
    fi

    invalidated_count="$(psql_db -At <<'SQL' | tr -d '[:space:]'
WITH invalidated AS (
    UPDATE userspace_runtime_sessions
    SET state = 'stopped'::"RuntimeSessionState",
        provider_session_id = NULL,
        preview_internal_url = NULL,
        launch_port = NULL,
        last_heartbeat_at = now(),
        last_error = 'Backup restore invalidated active runtime state',
        updated_at = now()
    WHERE state IN ('starting'::"RuntimeSessionState", 'running'::"RuntimeSessionState", 'stopping'::"RuntimeSessionState")
    RETURNING 1
)
SELECT COUNT(*) FROM invalidated;
SQL
)"

    log "INFO" "Invalidated restored active runtime sessions: ${invalidated_count:-0}"
    log "DEBUG" "Workspace runtime sessions will be recreated lazily on next preview/session start"
}

restore_legacy_secret_if_requested() {
    local extract_dir=$1
    local include_secret=$2
    local backup_key_file=""

    if [ -f "$extract_dir/.encryption_key" ]; then
        backup_key_file="$extract_dir/.encryption_key"
    elif [ -f "$extract_dir/.jwt_secret" ]; then
        backup_key_file="$extract_dir/.jwt_secret"
        log "DEBUG" "Found legacy .jwt_secret file in backup (will restore as .encryption_key)"
    fi

    if [ "$include_secret" = true ]; then
        if [ -n "$backup_key_file" ]; then
            if [ -f "$DATA_DIR/.encryption_key" ]; then
                log "WARNING" "Overwriting existing .encryption_key file with backup version"
            fi
            mkdir -p "$DATA_DIR"
            cp "$backup_key_file" "$DATA_DIR/.encryption_key"
            chmod 600 "$DATA_DIR/.encryption_key"
            log "INFO" "Encryption key file restored"
            log "DEBUG" "Restart the container to use the restored encryption key"
        elif [ ! -f "$DATA_DIR/.encryption_key" ]; then
            log "WARNING" "--include-secret specified but no encryption key file was found in the backup"
        fi
    elif [ -n "$backup_key_file" ]; then
        log "DEBUG" "Backup contains an encryption key file but --include-secret was not specified"
        log "DEBUG" "Use 'restore --include-secret' to restore the encryption key"
    fi
}

show_backup_usage() {
    cat <<'EOF'
Usage: backup [OPTIONS] [OUTPUT_FILE]

Create a backup archive containing the database and Ragtime data directory.
Streams to stdout by default, so redirect to a file on the host.

Options:
  --db-only          Backup database only (no Ragtime data directory)
  --files-only       Backup Ragtime data directory only (no database)
  --data-dir-only    Alias for --files-only
  --faiss-only       Legacy alias for --files-only
  --include-secret   Include the .encryption_key file in backup
                     (required to decrypt secrets on restore)
  -h, --help         Show this help message

Examples:
  docker exec ragtime backup > backup.tar.gz
  docker exec ragtime backup --include-secret > backup.tar.gz
  docker exec ragtime backup --db-only > db.tar.gz
  docker exec ragtime backup --files-only > data.tar.gz

Environment variables required for database backup:
  DATABASE_URL, or POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_DB
EOF
}

show_restore_usage() {
    cat <<'EOF'
Usage: restore [OPTIONS] ARCHIVE_FILE

Restore from a Ragtime backup archive. Use '-' as ARCHIVE_FILE to read the
archive from stdin.

Options:
--db-only                 Restore database only (skip Ragtime data directory)
--files-only              Restore Ragtime data directory only (skip database)
--data-dir-only           Alias for --files-only
--faiss-only              Legacy alias for --files-only
--skip-migrations         Skip automatic schema migration after restore
--data-only               Restore database rows only (legacy pg_restore mode)
--pg-data-only            Alias for --data-only
--replace-existing-data   Remove existing data-directory contents before restore
--include-secret          Restore the .encryption_key file if present
                                                    (overwrites existing key - use with caution)
--confirm-restore TEXT    Required confirmation text for non-interactive restore.
                                                    By default, type: RESTORE <database-name>
                                                    Override the expected text with
                                                    RESTORE_CONFIRMATION_PHRASE.
--mirror-local-admin-access
                                                    Grant the configured local admin the same workspace,
                                                    conversation, and auth-group memberships as a
                                                    backed-up non-local admin user (auto-selected)
--mirror-local-admin-from USERNAME
                                                    Source username for --mirror-local-admin-access
                                                    (for example: mvisnovsky.admin)
--local-admin-username USERNAME
                                                    Local admin username to mirror access to
                                                    (default: LOCAL_ADMIN_USER or admin)
-h, --help                Show this help message

Examples:
docker cp backup.tar.gz ragtime:/tmp/backup.tar.gz
docker exec ragtime restore /tmp/backup.tar.gz
docker exec ragtime restore --include-secret /tmp/backup.tar.gz
docker exec ragtime restore --confirm-restore "RESTORE ragtime" /tmp/backup.tar.gz
docker exec ragtime restore --mirror-local-admin-access /tmp/backup.tar.gz
docker exec ragtime restore --mirror-local-admin-from mvisnovsky.admin /tmp/backup.tar.gz
docker exec ragtime restore --files-only --replace-existing-data /tmp/backup.tar.gz
cat backup.tar.gz | docker exec -i ragtime restore --include-secret --confirm-restore "RESTORE ragtime" -

Environment variables required for database restore:
DATABASE_URL, or POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_DB
EOF
}

do_backup() {
    local db_only=false
    local files_only=false
    local include_secret=false
    local to_stdout=true
    local output_file=""
    local backup_type="full"
    local includes_secret=false
    local schema_version=""
    local db_size=""
    local archive_size=""

    while [[ $# -gt 0 ]]; do
        case $1 in
        --db-only)
            db_only=true
            shift
            ;;
        --files-only | --data-dir-only | --faiss-only)
            files_only=true
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

    if [ "$db_only" = true ] && [ "$files_only" = true ]; then
        die "Cannot specify both --db-only and --files-only"
    fi

    if [ "$db_only" = true ]; then
        backup_type="database-only"
    elif [ "$files_only" = true ]; then
        backup_type="files-only"
    fi

    require_commands tar du awk date
    if [ "$files_only" != true ]; then
        require_commands pg_dump psql
        require_database_env
    fi

    TEMP_DIR="${TEMP_BASE}_$(date +%s)_$$"
    mkdir -p "$TEMP_DIR"

    if [ "$to_stdout" = true ]; then
        LOG_FD=2
    fi

    if [ "$include_secret" = true ]; then
        if [ -f "$DATA_DIR/.encryption_key" ]; then
            includes_secret=true
            log "INFO" "Including encryption key file"
        else
            log "WARNING" "--include-secret specified but no .encryption_key file found in $DATA_DIR"
        fi
    fi

    log "INFO" "Creating $backup_type backup"

    if [ "$files_only" != true ]; then
        log "INFO" "Dumping database $POSTGRES_DB from $POSTGRES_HOST:$POSTGRES_PORT"
        if ! pg_dump_db >"$TEMP_DIR/database.dump"; then
            die "Database dump failed"
        fi
        db_size="$(du -h "$TEMP_DIR/database.dump" | awk '{print $1}')"
        schema_version="$(get_db_schema_version)"
        log "DEBUG" "Database dump: $db_size"
    fi

    if [ "$db_only" != true ]; then
        copy_data_to_temp "$include_secret"
    fi

    if [ "$includes_secret" = true ]; then
        cp "$DATA_DIR/.encryption_key" "$TEMP_DIR/.encryption_key"
        chmod 600 "$TEMP_DIR/.encryption_key"
    fi

    cat >"$TEMP_DIR/backup-meta.json" <<EOF
{
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "format_version": "$BACKUP_FORMAT_VERSION",
    "version": "$BACKUP_FORMAT_VERSION",
    "source": "cli",
    "type": "$backup_type",
    "ragtime_version": "${RAGTIME_VERSION:-unknown}",
    "schema_version": "${schema_version:-unknown}",
    "current_schema_version": "$CURRENT_SCHEMA_VERSION",
    "data_path": "$DATA_DIR",
    "includes_secret": $includes_secret
}
EOF

    if [ "$to_stdout" = true ]; then
        log "INFO" "Streaming archive to stdout"
        tar -czf - -C "$TEMP_DIR" .
        log "INFO" "Backup complete"
    else
        log "INFO" "Creating archive: $output_file"
        tar -czf "$output_file" -C "$TEMP_DIR" .
        archive_size="$(du -h "$output_file" | awk '{print $1}')"
        log "INFO" "Backup complete: $output_file ($archive_size)"
    fi

    if [ "$includes_secret" = true ]; then
        log "INFO" "Encryption key included in backup"
        log "DEBUG" "Use 'restore --include-secret' to restore the key file"
    elif [ -f "$DATA_DIR/.encryption_key" ]; then
        log "WARNING" "Backup does not include encryption key"
        log "WARNING" "Use 'backup --include-secret' when encrypted secret portability matters"
        log "DEBUG" "Or manually backup: $DATA_DIR/.encryption_key"
    else
        log "WARNING" "No encryption key file found; encrypted secrets may not be recoverable"
    fi
}

do_restore() {
    local db_only=false
    local files_only=false
    local archive_file=""
    local archive_to_extract=""
    local skip_migrations=false
    local pg_data_only=false
    local include_secret=false
    local replace_data=false
    local restore_confirmation="${RESTORE_CONFIRMATION:-}"
    local mirror_local_admin_access="${RESTORE_MIRROR_LOCAL_ADMIN_ACCESS:-false}"
    local mirror_local_admin_source="${RESTORE_MIRROR_LOCAL_ADMIN_FROM:-auto}"
    local mirror_local_admin_target="${RESTORE_LOCAL_ADMIN_USERNAME:-${LOCAL_ADMIN_USER:-admin}}"
    local extract_dir=""
    local backup_schema_version=""
    local backup_date=""
    local backup_type_meta=""
    local restore_errors=""
    local table_check=""
    local migrations_exist=""
    local restored_version=""

    while [[ $# -gt 0 ]]; do
        case $1 in
        --db-only)
            db_only=true
            shift
            ;;
        --files-only | --data-dir-only | --faiss-only)
            files_only=true
            shift
            ;;
        --skip-migrations)
            skip_migrations=true
            shift
            ;;
        --data-only | --pg-data-only)
            pg_data_only=true
            shift
            ;;
        --replace-existing-data | --replace-data)
            replace_data=true
            shift
            ;;
        --include-secret)
            include_secret=true
            shift
            ;;
        --confirm-restore)
            if [ $# -lt 2 ]; then
                die "--confirm-restore requires confirmation text"
            fi
            restore_confirmation="$2"
            shift 2
            ;;
        --confirm-restore=*)
            restore_confirmation="${1#*=}"
            shift
            ;;
        --mirror-local-admin-access)
            mirror_local_admin_access=true
            shift
            ;;
        --mirror-local-admin-from)
            mirror_local_admin_access=true
            if [ $# -lt 2 ]; then
                die "--mirror-local-admin-from requires a source username"
            fi
            mirror_local_admin_source="$2"
            shift 2
            ;;
        --mirror-local-admin-from=*)
            mirror_local_admin_access=true
            mirror_local_admin_source="${1#*=}"
            shift
            ;;
        --local-admin-username)
            if [ $# -lt 2 ]; then
                die "--local-admin-username requires a username"
            fi
            mirror_local_admin_target="$2"
            shift 2
            ;;
        --local-admin-username=*)
            mirror_local_admin_target="${1#*=}"
            shift
            ;;
        -)
            archive_file="$1"
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

    if [ "$db_only" = true ] && [ "$files_only" = true ]; then
        die "Cannot specify both --db-only and --files-only"
    fi

    if [ -z "$archive_file" ]; then
        log "ERROR" "Archive file is required"
        show_restore_usage
        exit 1
    fi

    require_commands tar du awk date find
    if [ "$files_only" != true ]; then
        require_commands pg_restore psql
        require_database_env
    fi

    TEMP_DIR="${TEMP_BASE}_restore_$(date +%s)_$$"
    extract_dir="$TEMP_DIR/extract"
    mkdir -p "$extract_dir"

    if [ "$archive_file" = "-" ]; then
        archive_to_extract="$TEMP_DIR/stdin-backup.tar.gz"
        log "INFO" "Reading backup archive from stdin"
        cat >"$archive_to_extract"
    else
        archive_to_extract="$archive_file"
    fi

    if [ ! -f "$archive_to_extract" ]; then
        die "Archive file not found: $archive_to_extract"
    fi

    validate_tar_archive "$archive_to_extract"

    log "INFO" "Extracting archive: $archive_file"
    if ! tar -xzf "$archive_to_extract" -C "$extract_dir"; then
        die "Failed to extract archive: $archive_file"
    fi

    if [ ! -f "$extract_dir/database.dump" ] && [ ! -d "$extract_dir/data" ] && [ ! -d "$extract_dir/faiss" ]; then
        die "Invalid backup archive: expected database.dump, data/, or legacy faiss/"
    fi

    if [ -f "$extract_dir/backup-meta.json" ]; then
        backup_date="$(read_meta_field "$extract_dir/backup-meta.json" created_at)"
        backup_type_meta="$(read_meta_field "$extract_dir/backup-meta.json" type)"
        backup_schema_version="$(read_meta_field "$extract_dir/backup-meta.json" schema_version)"
        log "DEBUG" "Backup date: ${backup_date:-unknown}"
        log "DEBUG" "Backup type: ${backup_type_meta:-full}"
        if [ -n "$backup_schema_version" ] && [ "$backup_schema_version" != "unknown" ]; then
            log "DEBUG" "Backup schema version: $backup_schema_version"
            log "DEBUG" "Current schema version: $CURRENT_SCHEMA_VERSION"
            if [ "$CURRENT_SCHEMA_VERSION" != "unknown" ] && [ "$backup_schema_version" != "$CURRENT_SCHEMA_VERSION" ]; then
                log "WARNING" "Schema version mismatch detected - migrations will be applied after restore"
            fi
        fi
    fi

    confirm_restore "$restore_confirmation"

    if [ "$files_only" = true ]; then
        log "DEBUG" "Skipping database restore (files-only mode)"
    elif [ -f "$extract_dir/database.dump" ] && [ -s "$extract_dir/database.dump" ]; then
        log "INFO" "Restoring database $POSTGRES_DB on $POSTGRES_HOST:$POSTGRES_PORT"
        terminate_other_database_connections
        restore_errors="$(mktemp)"

        if [ "$pg_data_only" = true ]; then
            log "DEBUG" "PostgreSQL data-only mode: restoring rows without schema changes"
            if ! pg_restore_db --data-only --disable-triggers --no-owner --no-privileges --exit-on-error "$extract_dir/database.dump" 2>"$restore_errors"; then
                log "ERROR" "Database data-only restore failed"
                head -80 "$restore_errors" >&2 || true
                rm -f "$restore_errors"
                exit 1
            fi
        else
            if ! pg_restore_db --clean --if-exists --no-owner --no-privileges --exit-on-error "$extract_dir/database.dump" 2>"$restore_errors"; then
                log "ERROR" "Database restore failed"
                head -120 "$restore_errors" >&2 || true
                rm -f "$restore_errors"
                exit 1
            fi
        fi
        rm -f "$restore_errors"

        table_check="$(psql_db -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE' AND table_name NOT LIKE '_prisma%'" 2>/dev/null | tr -d '[:space:]' || true)"
        if [ -z "$table_check" ] || [ "$table_check" -lt 5 ]; then
            die "Database restore verification failed - expected application tables not found"
        fi

        log "INFO" "Database restored successfully (verified $table_check application tables)"

        if [ "$skip_migrations" = true ]; then
            log "DEBUG" "Skipping automatic schema migration (--skip-migrations)"
            log "WARNING" "You may need to run 'python -m prisma migrate deploy' manually"
        else
            migrations_exist="$(check_migrations_table)"
            restored_version="$(get_db_schema_version)"

            if [ "$migrations_exist" = "t" ] && [ -n "$restored_version" ]; then
                if [ "$CURRENT_SCHEMA_VERSION" != "unknown" ] && [[ "$restored_version" < "$CURRENT_SCHEMA_VERSION" ]]; then
                    log "INFO" "Detected older schema version ($restored_version), applying newer migrations"
                    run_migrations
                else
                    log "DEBUG" "Schema restored from backup is up to date ($restored_version)"
                fi
            else
                log "WARNING" "No migration history found in restored database"
                log "INFO" "Attempting to run migrations to establish current schema"
                run_migrations
            fi
        fi

        if [ "$mirror_local_admin_access" = true ]; then
            mirror_local_admin_access "$mirror_local_admin_source" "$mirror_local_admin_target"
        fi

        invalidate_restored_runtime_sessions
    else
        log "WARNING" "No database dump found or database.dump is empty, skipping database restore"
    fi

    if [ "$db_only" = true ]; then
        log "DEBUG" "Skipping Ragtime data restore (db-only mode)"
    else
        restore_data_from_backup "$extract_dir" "$include_secret" "$replace_data"
        invalidate_restored_workspace_runtime_artifacts "$extract_dir"
        restore_legacy_secret_if_requested "$extract_dir" "$include_secret"
    fi

    log "INFO" "Restore complete"

    if [ "$include_secret" != true ] && { [ -f "$extract_dir/.encryption_key" ] || [ -f "$extract_dir/data/.encryption_key" ] || [ -f "$extract_dir/.jwt_secret" ]; }; then
        log "WARNING" "Backup contained an encryption key, but it was not restored because --include-secret was not specified"
    fi

    if [ ! -f "$DATA_DIR/.encryption_key" ]; then
        log "WARNING" "IMPORTANT: Encrypted secrets require the same encryption key used when the backup was created"
        log "WARNING" "If the key is missing or different, re-enter API keys and passwords in the Settings UI"
    fi
}

SCRIPT_NAME="$(basename "$0")"

case "$SCRIPT_NAME" in
backup)
    do_backup "$@"
    ;;
restore)
    do_restore "$@"
    ;;
*)
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
        cat <<EOF
Ragtime Backup/Restore Tool

Usage:
  $0 backup [OPTIONS] [OUTPUT_FILE]
  $0 restore [OPTIONS] ARCHIVE_FILE

Or via convenience commands:
  backup [OPTIONS]
  restore [OPTIONS] ARCHIVE_FILE

Run 'backup --help' or 'restore --help' for more information.
EOF
        exit 1
        ;;
    esac
    ;;
esac