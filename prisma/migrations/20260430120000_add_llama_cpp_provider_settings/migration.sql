ALTER TABLE "app_settings"
  ADD COLUMN IF NOT EXISTS "llm_llama_cpp_protocol" TEXT NOT NULL DEFAULT 'http',
  ADD COLUMN IF NOT EXISTS "llm_llama_cpp_host" TEXT NOT NULL DEFAULT 'host.docker.internal',
  ADD COLUMN IF NOT EXISTS "llm_llama_cpp_port" INTEGER NOT NULL DEFAULT 8080,
  ADD COLUMN IF NOT EXISTS "llm_llama_cpp_base_url" TEXT NOT NULL DEFAULT 'http://host.docker.internal:8080',
  ADD COLUMN IF NOT EXISTS "llama_cpp_protocol" TEXT NOT NULL DEFAULT 'http',
  ADD COLUMN IF NOT EXISTS "llama_cpp_host" TEXT NOT NULL DEFAULT 'host.docker.internal',
  ADD COLUMN IF NOT EXISTS "llama_cpp_port" INTEGER NOT NULL DEFAULT 8081,
  ADD COLUMN IF NOT EXISTS "llama_cpp_base_url" TEXT NOT NULL DEFAULT 'http://host.docker.internal:8081';
