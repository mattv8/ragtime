CREATE TABLE IF NOT EXISTS "workspace_preview_diagnostics" (
    "id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "diagnostic_key" TEXT NOT NULL,
    "target_label" TEXT NOT NULL,
    "method" TEXT,
    "component_id" TEXT,
    "count" INTEGER NOT NULL DEFAULT 0,
    "error_count" INTEGER NOT NULL DEFAULT 0,
    "last_ms" INTEGER NOT NULL DEFAULT 0,
    "avg_ms" INTEGER NOT NULL DEFAULT 0,
    "max_ms" INTEGER NOT NULL DEFAULT 0,
    "last_error" TEXT,
    "last_status_code" INTEGER,
    "last_row_count" INTEGER,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "workspace_preview_diagnostics_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_preview_diagnostics_workspace_id_diagnostic_key_key"
    ON "workspace_preview_diagnostics"("workspace_id", "diagnostic_key");

CREATE INDEX IF NOT EXISTS "workspace_preview_diagnostics_workspace_id_updated_at_idx"
    ON "workspace_preview_diagnostics"("workspace_id", "updated_at");

CREATE INDEX IF NOT EXISTS "workspace_preview_diagnostics_workspace_id_max_ms_idx"
    ON "workspace_preview_diagnostics"("workspace_id", "max_ms");

DO $$
BEGIN
    ALTER TABLE "workspace_preview_diagnostics"
        ADD CONSTRAINT "workspace_preview_diagnostics_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;
