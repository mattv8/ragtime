CREATE TABLE IF NOT EXISTS "userspace_changed_file_acknowledgements" (
    "id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "path" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "userspace_changed_file_acknowledgements_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "userspace_changed_file_acknowledgements_workspace_id_fkey"
      FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "userspace_changed_file_acknowledgements_user_id_fkey"
      FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE
);

DO $$
BEGIN
  ALTER TABLE "userspace_changed_file_acknowledgements"
    ADD CONSTRAINT "userspace_changed_file_acknowledgements_workspace_id_user_id_path_key"
    UNIQUE ("workspace_id", "user_id", "path");
EXCEPTION
  WHEN duplicate_object THEN NULL;
END $$;

CREATE INDEX IF NOT EXISTS "userspace_changed_file_acknowledgements_workspace_id_user_id_idx"
  ON "userspace_changed_file_acknowledgements"("workspace_id", "user_id");

CREATE INDEX IF NOT EXISTS "userspace_changed_file_acknowledgements_workspace_id_idx"
  ON "userspace_changed_file_acknowledgements"("workspace_id");
