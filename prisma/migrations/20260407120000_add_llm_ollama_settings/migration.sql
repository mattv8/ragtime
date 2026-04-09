-- Add separate Ollama connection settings for chat/LLM usage.
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "llm_ollama_protocol" TEXT NOT NULL DEFAULT 'http';

ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "llm_ollama_host" TEXT NOT NULL DEFAULT 'localhost';

ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "llm_ollama_port" INTEGER NOT NULL DEFAULT 11434;

ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "llm_ollama_base_url" TEXT NOT NULL DEFAULT 'http://localhost:11434';