-- Per-user theme pack selection (null = inherit global default)
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "theme_pack" TEXT;

-- Instance-wide default theme pack for users without a personal choice
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "default_theme_pack" TEXT NOT NULL DEFAULT 'default';
