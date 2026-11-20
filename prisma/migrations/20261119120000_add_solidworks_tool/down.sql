-- PostgreSQL cannot drop the added "solidworks" ToolType value; remove its rows instead.
DELETE FROM "tool_configs" WHERE "tool_type" = 'solidworks';
