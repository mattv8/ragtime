#!/bin/bash
# =============================================================================
# PostgreSQL pgvector Initialization Script
# =============================================================================
# Configures optimal autovacuum and maintenance settings for pgvector workloads.
# This script runs from the app container and connects to the database container.
#
# Key optimizations:
# - Aggressive autovacuum for embedding tables (high UPDATE/DELETE churn)
# - Increased maintenance_work_mem for faster VACUUM and index builds
# - ivfflat.probes for query accuracy vs speed tradeoff
# =============================================================================

set -e

# Set Logger prefix
LOG_PREFIX="ragtime.db.init"

# Source functions helper
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
if [ -f "$SCRIPT_DIR/functions.sh" ]; then
    source "$SCRIPT_DIR/functions.sh"
elif [ -f "/docker-scripts/functions.sh" ]; then
    source "/docker-scripts/functions.sh"
fi

# Parse DATABASE_URL to extract connection details
# Format: postgresql://user:password@host:port/database
if ! parse_database_url; then
    log "WARNING" "DATABASE_URL not set, skipping pgvector configuration"
    exit 0
fi

# Wait for PostgreSQL to be ready
if ! wait_for_postgres 30; then
    log "WARNING" "Skipping pgvector configuration"
    exit 0
fi

log "INFO" "Configuring pgvector optimizations..."

PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=0 -o /dev/null <<-EOSQL
    -- ==========================================================================
    -- Session/Query Settings (set defaults for all connections)
    -- ==========================================================================

    -- ivfflat probes: higher = more accurate but slower (default 1, max ~sqrt(lists))
    -- For 100 lists, 10-20 probes is a good balance
    ALTER DATABASE ${DB_NAME} SET ivfflat.probes = 10;

    -- HNSW ef_search: higher = more accurate but slower (default 40)
    -- 100-200 is good for production accuracy
    ALTER DATABASE ${DB_NAME} SET hnsw.ef_search = 100;

    -- ==========================================================================
    -- Per-Table Autovacuum Configuration
    -- ==========================================================================
    -- These tables have high churn during indexing and need aggressive vacuum

    -- pdm_embeddings: 167k+ rows, frequent updates during PDM re-index
    ALTER TABLE IF EXISTS pdm_embeddings SET (
        autovacuum_vacuum_scale_factor = 0.05,      -- vacuum after 5% dead tuples (default 20%)
        autovacuum_analyze_scale_factor = 0.02,     -- analyze after 2% changes (default 10%)
        autovacuum_vacuum_threshold = 50,           -- minimum dead tuples before vacuum
        autovacuum_analyze_threshold = 50           -- minimum changes before analyze
    );

    -- schema_embeddings: moderate size, updated during schema re-index
    ALTER TABLE IF EXISTS schema_embeddings SET (
        autovacuum_vacuum_scale_factor = 0.05,
        autovacuum_analyze_scale_factor = 0.02,
        autovacuum_vacuum_threshold = 50,
        autovacuum_analyze_threshold = 50
    );

    -- filesystem_embeddings: can grow large, high churn during file indexing
    ALTER TABLE IF EXISTS filesystem_embeddings SET (
        autovacuum_vacuum_scale_factor = 0.05,
        autovacuum_analyze_scale_factor = 0.02,
        autovacuum_vacuum_threshold = 50,
        autovacuum_analyze_threshold = 50
    );

    -- ==========================================================================
    -- Ensure pgvector embedding columns exist
    -- These columns are not in Prisma schema (unsupported type) so db push drops them
    -- Read dimension from app_settings (set by first indexing operation)
    -- ==========================================================================
    DO \$\$
    DECLARE
        dim INTEGER;
    BEGIN
        -- Get configured dimension from app_settings, default to 768 (nomic-embed-text)
        SELECT COALESCE(embedding_dimension, 768) INTO dim FROM app_settings LIMIT 1;
        IF dim IS NULL THEN
            dim := 768;
        END IF;

        -- Add embedding columns if missing
        BEGIN
            EXECUTE format('ALTER TABLE filesystem_embeddings ADD COLUMN embedding vector(%s)', dim);
        EXCEPTION
            WHEN duplicate_column THEN NULL;
        END;

        BEGIN
            EXECUTE format('ALTER TABLE schema_embeddings ADD COLUMN embedding vector(%s)', dim);
        EXCEPTION
            WHEN duplicate_column THEN NULL;
        END;

        BEGIN
            EXECUTE format('ALTER TABLE pdm_embeddings ADD COLUMN embedding vector(%s)', dim);
        EXCEPTION
            WHEN duplicate_column THEN NULL;
        END;
    END \$\$;

    -- ==========================================================================
    -- Reload Configuration
    -- ==========================================================================
    SELECT pg_reload_conf();

EOSQL

log "INFO" "pgvector optimizations applied successfully"
