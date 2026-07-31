DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'FaissSearchConcurrencyMode') THEN
        CREATE TYPE "FaissSearchConcurrencyMode" AS ENUM ('per_index', 'global');
    END IF;
END $$;

ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "faiss_search_concurrency_mode" "FaissSearchConcurrencyMode" NOT NULL DEFAULT 'per_index';
