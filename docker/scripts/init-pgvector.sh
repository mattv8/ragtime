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

# Parse DATABASE_URL to extract connection details
# Format: postgresql://user:password@host:port/database
if [ -z "$DATABASE_URL" ]; then
    echo "DATABASE_URL not set, skipping pgvector configuration"
    exit 0
fi

# Extract components from DATABASE_URL
DB_USER=$(echo "$DATABASE_URL" | sed -n 's|.*://\([^:]*\):.*|\1|p')
DB_PASS=$(echo "$DATABASE_URL" | sed -n 's|.*://[^:]*:\([^@]*\)@.*|\1|p')
DB_HOST=$(echo "$DATABASE_URL" | sed -n 's|.*@\([^:]*\):.*|\1|p')
DB_PORT=$(echo "$DATABASE_URL" | sed -n 's|.*:\([0-9]*\)/.*|\1|p')
DB_NAME=$(echo "$DATABASE_URL" | sed -n 's|.*/\([^?]*\).*|\1|p')

# Fallback defaults
DB_PORT=${DB_PORT:-5432}

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."
for i in {1..30}; do
    if PGPASSWORD="$DB_PASS" pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
        break
    fi
    if [ $i -eq 30 ]; then
        echo "PostgreSQL not ready after 30 attempts, skipping pgvector configuration"
        exit 0
    fi
    sleep 1
done

echo "Configuring pgvector optimizations..."

PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=0 <<-EOSQL
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

echo "pgvector optimizations applied successfully"
