-- Lets the default /mcp route own instruction skills. The owner CHECK and the
-- partial unique index cannot be expressed in schema.prisma: deploy with
-- `prisma migrate deploy` and reject any `migrate dev` diff that drops them.
-- Rolling the app back without down.sql leaves rows older code cannot summarize.
ALTER TABLE "instruction_skills" ADD COLUMN IF NOT EXISTS "default_route" BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE "instruction_skills" DROP CONSTRAINT IF EXISTS "instruction_skills_owner_check";
ALTER TABLE "instruction_skills" ADD CONSTRAINT "instruction_skills_owner_check"
    CHECK (num_nonnulls("tool_config_id", "mcp_route_id") + CASE WHEN "default_route" THEN 1 ELSE 0 END = 1);
CREATE UNIQUE INDEX IF NOT EXISTS "instruction_skills_default_route_name_key"
    ON "instruction_skills"("name") WHERE "default_route";
