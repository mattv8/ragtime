-- Add generic semantic vision OCR provider support while preserving legacy Ollama OCR configs.
ALTER TYPE "OcrMode" ADD VALUE IF NOT EXISTS 'vision';

ALTER TABLE "index_metadata"
  ADD COLUMN IF NOT EXISTS "ocr_provider" TEXT;

ALTER TABLE "app_settings"
  ADD COLUMN IF NOT EXISTS "default_ocr_provider" TEXT DEFAULT 'ollama';

UPDATE "index_metadata"
SET "ocr_provider" = 'ollama'
WHERE "ocr_mode" = 'ollama' AND "ocr_provider" IS NULL;

UPDATE "app_settings"
SET "default_ocr_provider" = 'ollama'
WHERE "default_ocr_provider" IS NULL;
