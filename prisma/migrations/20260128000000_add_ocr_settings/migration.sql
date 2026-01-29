-- Add OCR mode enum and settings for vision-based OCR

-- Create OcrMode enum
DO $$ BEGIN
    CREATE TYPE "OcrMode" AS ENUM ('disabled', 'tesseract', 'ollama');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Add OCR settings to app_settings (default for new indexes)
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "default_ocr_mode" "OcrMode" NOT NULL DEFAULT 'disabled';
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "default_ocr_vision_model" TEXT;

-- Add per-index OCR mode to index_metadata (replaces enable_ocr boolean concept)
ALTER TABLE "index_metadata" ADD COLUMN IF NOT EXISTS "ocr_mode" "OcrMode" NOT NULL DEFAULT 'disabled';
ALTER TABLE "index_metadata" ADD COLUMN IF NOT EXISTS "ocr_vision_model" TEXT;
