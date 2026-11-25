DELETE FROM "instruction_skills" WHERE "default_route";
DROP INDEX IF EXISTS "instruction_skills_default_route_name_key";
ALTER TABLE "instruction_skills" DROP CONSTRAINT IF EXISTS "instruction_skills_owner_check";
ALTER TABLE "instruction_skills" ADD CONSTRAINT "instruction_skills_owner_check" CHECK (num_nonnulls("tool_config_id", "mcp_route_id") = 1);
ALTER TABLE "instruction_skills" DROP COLUMN IF EXISTS "default_route";
