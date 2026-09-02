-- Per-user chat model default
ALTER TABLE "users"
ADD COLUMN IF NOT EXISTS "default_chat_model" TEXT;

-- Per-user per-workspace chat model defaults
CREATE TABLE IF NOT EXISTS "workspace_user_preferences" (
  "id" TEXT NOT NULL,
  "workspace_id" TEXT NOT NULL,
  "user_id" TEXT NOT NULL,
  "default_chat_model" TEXT NOT NULL,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "workspace_user_preferences_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_user_preferences_workspace_id_user_id_key"
ON "workspace_user_preferences"("workspace_id", "user_id");

CREATE INDEX IF NOT EXISTS "workspace_user_preferences_user_id_idx"
ON "workspace_user_preferences"("user_id");

DO $$
BEGIN
  ALTER TABLE "workspace_user_preferences"
  ADD CONSTRAINT "workspace_user_preferences_workspace_id_fkey"
  FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

DO $$
BEGIN
  ALTER TABLE "workspace_user_preferences"
  ADD CONSTRAINT "workspace_user_preferences_user_id_fkey"
  FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;
