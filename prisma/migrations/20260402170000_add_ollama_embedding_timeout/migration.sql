-- Add configurable Ollama embedding timeout (seconds per sub-batch)
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "ollama_embedding_timeout_seconds" INTEGER NOT NULL DEFAULT 180;
