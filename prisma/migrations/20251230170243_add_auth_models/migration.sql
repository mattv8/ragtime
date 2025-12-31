-- CreateEnum
CREATE TYPE "IndexStatus" AS ENUM ('pending', 'processing', 'completed', 'failed');

-- CreateEnum
CREATE TYPE "ToolType" AS ENUM ('postgres', 'odoo_shell', 'ssh_shell');

-- CreateEnum
CREATE TYPE "AuthProvider" AS ENUM ('ldap', 'local');

-- CreateEnum
CREATE TYPE "UserRole" AS ENUM ('user', 'admin');

-- CreateEnum
CREATE TYPE "ChatTaskStatus" AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled');

-- CreateTable
CREATE TABLE "index_configs" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL DEFAULT '',
    "filePatterns" TEXT[] DEFAULT ARRAY['**/*.py', '**/*.md', '**/*.rst', '**/*.txt', '**/*.xml']::TEXT[],
    "excludePatterns" TEXT[] DEFAULT ARRAY['**/node_modules/**', '**/__pycache__/**', '**/venv/**', '**/.git/**']::TEXT[],
    "chunkSize" INTEGER NOT NULL DEFAULT 1000,
    "chunkOverlap" INTEGER NOT NULL DEFAULT 200,
    "embeddingModel" TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    "jobId" TEXT,

    CONSTRAINT "index_configs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "index_jobs" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "status" "IndexStatus" NOT NULL DEFAULT 'pending',
    "source_type" TEXT NOT NULL,
    "source_path" TEXT,
    "git_url" TEXT,
    "git_branch" TEXT NOT NULL DEFAULT 'main',
    "total_files" INTEGER NOT NULL DEFAULT 0,
    "processed_files" INTEGER NOT NULL DEFAULT 0,
    "total_chunks" INTEGER NOT NULL DEFAULT 0,
    "processed_chunks" INTEGER NOT NULL DEFAULT 0,
    "error_message" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "started_at" TIMESTAMP(3),
    "completed_at" TIMESTAMP(3),
    "config_id" TEXT NOT NULL,

    CONSTRAINT "index_jobs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "index_metadata" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL DEFAULT '',
    "path" TEXT NOT NULL,
    "document_count" INTEGER NOT NULL DEFAULT 0,
    "chunk_count" INTEGER NOT NULL DEFAULT 0,
    "size_bytes" BIGINT NOT NULL DEFAULT 0,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "source_type" TEXT NOT NULL,
    "source" TEXT,
    "config_snapshot" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "last_modified" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "index_metadata_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "app_settings" (
    "id" TEXT NOT NULL DEFAULT 'default',
    "embedding_provider" TEXT NOT NULL DEFAULT 'ollama',
    "embedding_model" TEXT NOT NULL DEFAULT 'nomic-embed-text',
    "ollama_protocol" TEXT NOT NULL DEFAULT 'http',
    "ollama_host" TEXT NOT NULL DEFAULT 'localhost',
    "ollama_port" INTEGER NOT NULL DEFAULT 11434,
    "ollama_base_url" TEXT NOT NULL DEFAULT 'http://localhost:11434',
    "llm_provider" TEXT NOT NULL DEFAULT 'openai',
    "llm_model" TEXT NOT NULL DEFAULT 'gpt-4-turbo',
    "openai_api_key" TEXT NOT NULL DEFAULT '',
    "anthropic_api_key" TEXT NOT NULL DEFAULT '',
    "allowed_chat_models" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "enabled_tools" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "odoo_container" TEXT NOT NULL DEFAULT 'odoo-server',
    "postgres_container" TEXT NOT NULL DEFAULT 'odoo-postgres',
    "postgres_host" TEXT NOT NULL DEFAULT '',
    "postgres_port" INTEGER NOT NULL DEFAULT 5432,
    "postgres_user" TEXT NOT NULL DEFAULT '',
    "postgres_password" TEXT NOT NULL DEFAULT '',
    "postgres_db" TEXT NOT NULL DEFAULT '',
    "max_query_results" INTEGER NOT NULL DEFAULT 100,
    "query_timeout" INTEGER NOT NULL DEFAULT 30,
    "max_iterations" INTEGER NOT NULL DEFAULT 15,
    "enable_write_ops" BOOLEAN NOT NULL DEFAULT false,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "app_settings_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "tool_configs" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "tool_type" "ToolType" NOT NULL,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "description" TEXT NOT NULL DEFAULT '',
    "connection_config" JSONB NOT NULL,
    "max_results" INTEGER NOT NULL DEFAULT 100,
    "timeout" INTEGER NOT NULL DEFAULT 30,
    "allow_write" BOOLEAN NOT NULL DEFAULT false,
    "last_test_at" TIMESTAMP(3),
    "last_test_result" BOOLEAN,
    "last_test_error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "tool_configs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "users" (
    "id" TEXT NOT NULL,
    "username" TEXT NOT NULL,
    "auth_provider" "AuthProvider" NOT NULL DEFAULT 'ldap',
    "ldap_dn" TEXT,
    "email" TEXT,
    "display_name" TEXT,
    "role" "UserRole" NOT NULL DEFAULT 'user',
    "last_login_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "sessions" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "token_hash" TEXT NOT NULL,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "user_agent" TEXT,
    "ip_address" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "sessions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ldap_config" (
    "id" TEXT NOT NULL DEFAULT 'default',
    "server_url" TEXT NOT NULL DEFAULT '',
    "bind_dn" TEXT NOT NULL DEFAULT '',
    "bind_password" TEXT NOT NULL DEFAULT '',
    "allow_self_signed" BOOLEAN NOT NULL DEFAULT false,
    "base_dn" TEXT NOT NULL DEFAULT '',
    "user_search_base" TEXT NOT NULL DEFAULT '',
    "user_search_filter" TEXT NOT NULL DEFAULT '(|(sAMAccountName={username})(uid={username}))',
    "admin_group_dn" TEXT NOT NULL DEFAULT '',
    "user_group_dn" TEXT NOT NULL DEFAULT '',
    "discovered_ous" JSONB NOT NULL DEFAULT '[]',
    "discovered_groups" JSONB NOT NULL DEFAULT '[]',
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ldap_config_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "conversations" (
    "id" TEXT NOT NULL,
    "title" TEXT NOT NULL DEFAULT 'New Chat',
    "model" TEXT NOT NULL DEFAULT 'gpt-4-turbo',
    "user_id" TEXT,
    "messages" JSONB NOT NULL DEFAULT '[]',
    "total_tokens" INTEGER NOT NULL DEFAULT 0,
    "active_task_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "conversations_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "chat_tasks" (
    "id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "status" "ChatTaskStatus" NOT NULL DEFAULT 'pending',
    "user_message" TEXT NOT NULL,
    "streaming_state" JSONB,
    "response_content" TEXT,
    "error_message" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "started_at" TIMESTAMP(3),
    "completed_at" TIMESTAMP(3),
    "last_update_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "chat_tasks_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "index_jobs_config_id_key" ON "index_jobs"("config_id");

-- CreateIndex
CREATE UNIQUE INDEX "index_metadata_name_key" ON "index_metadata"("name");

-- CreateIndex
CREATE UNIQUE INDEX "users_username_key" ON "users"("username");

-- AddForeignKey
ALTER TABLE "index_jobs" ADD CONSTRAINT "index_jobs_config_id_fkey" FOREIGN KEY ("config_id") REFERENCES "index_configs"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "sessions" ADD CONSTRAINT "sessions_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "conversations" ADD CONSTRAINT "conversations_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "chat_tasks" ADD CONSTRAINT "chat_tasks_conversation_id_fkey" FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE ON UPDATE CASCADE;
