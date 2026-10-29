-- PostgreSQL cannot drop the added "windows_shell" ToolType value; remove its rows instead.
DELETE FROM "tool_configs" WHERE "tool_type" = 'windows_shell';
DROP TABLE IF EXISTS "remote_pairings";
DROP TYPE IF EXISTS "RemotePairingStatus";
