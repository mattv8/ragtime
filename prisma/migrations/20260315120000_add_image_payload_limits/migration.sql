-- Add image payload downsampling limits to app_settings
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "image_payload_max_width" INTEGER NOT NULL DEFAULT 1024;
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "image_payload_max_height" INTEGER NOT NULL DEFAULT 1024;
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "image_payload_max_pixels" INTEGER NOT NULL DEFAULT 786432;
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "image_payload_max_bytes" INTEGER NOT NULL DEFAULT 350000;
