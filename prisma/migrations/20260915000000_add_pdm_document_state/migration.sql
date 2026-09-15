-- Structured checked-in PDM document state for deterministic lookup
CREATE TABLE IF NOT EXISTS "pdm_document_state" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "index_name" TEXT NOT NULL,
    "document_id" INTEGER NOT NULL,
    "filename" TEXT NOT NULL,
    "part_number" TEXT,
    "target_revision" INTEGER NOT NULL,
    "state_json" JSONB NOT NULL DEFAULT '{}',
    "metadata_hash" TEXT NOT NULL,
    "extracted_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS "pdm_document_state_index_doc_unique"
    ON "pdm_document_state" ("index_name", "document_id");

CREATE INDEX IF NOT EXISTS "pdm_document_state_index_name_idx"
    ON "pdm_document_state" ("index_name");

CREATE INDEX IF NOT EXISTS "pdm_document_state_part_number_idx"
    ON "pdm_document_state" ("part_number");
