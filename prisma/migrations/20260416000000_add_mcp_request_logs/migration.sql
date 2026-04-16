-- CreateTable
CREATE TABLE IF NOT EXISTS "mcp_request_logs" (
    "id" TEXT NOT NULL,
    "user_id" TEXT,
    "username" TEXT,
    "route_name" TEXT NOT NULL DEFAULT 'default',
    "auth_method" TEXT NOT NULL DEFAULT 'none',
    "http_method" TEXT NOT NULL,
    "status_code" INTEGER NOT NULL DEFAULT 200,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "mcp_request_logs_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "mcp_request_logs_user_id_created_at_idx" ON "mcp_request_logs"("user_id", "created_at");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "mcp_request_logs_route_name_created_at_idx" ON "mcp_request_logs"("route_name", "created_at");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "mcp_request_logs_created_at_idx" ON "mcp_request_logs"("created_at");

-- AddForeignKey
DO $$
BEGIN
    ALTER TABLE "mcp_request_logs"
        ADD CONSTRAINT "mcp_request_logs_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;

-- CreateTable
CREATE TABLE IF NOT EXISTS "api_request_logs" (
    "id" TEXT NOT NULL,
    "user_id" TEXT,
    "provider" TEXT NOT NULL DEFAULT '',
    "model" TEXT NOT NULL DEFAULT '',
    "endpoint" TEXT NOT NULL,
    "http_method" TEXT NOT NULL,
    "status_code" INTEGER NOT NULL DEFAULT 200,
    "streaming" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "api_request_logs_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "api_request_logs_user_id_created_at_idx" ON "api_request_logs"("user_id", "created_at");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "api_request_logs_endpoint_created_at_idx" ON "api_request_logs"("endpoint", "created_at");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "api_request_logs_created_at_idx" ON "api_request_logs"("created_at");

-- AddForeignKey
DO $$
BEGIN
    ALTER TABLE "api_request_logs"
        ADD CONSTRAINT "api_request_logs_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;
