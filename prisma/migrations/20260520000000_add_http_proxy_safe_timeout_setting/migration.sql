ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "http_proxy_safe_timeout_seconds" INTEGER NOT NULL DEFAULT 90;
