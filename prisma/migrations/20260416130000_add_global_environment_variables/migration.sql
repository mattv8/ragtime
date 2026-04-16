CREATE TABLE IF NOT EXISTS global_environment_variables (
    id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT global_environment_variables_pkey PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS global_environment_variables_key_key
    ON global_environment_variables(key);

CREATE INDEX IF NOT EXISTS global_environment_variables_updated_at_idx
    ON global_environment_variables(updated_at);
