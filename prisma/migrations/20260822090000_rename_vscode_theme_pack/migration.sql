UPDATE "users"
SET "theme_pack" = 'modern'
WHERE LOWER(BTRIM("theme_pack")) = 'vscode';

UPDATE "app_settings"
SET "default_theme_pack" = 'modern'
WHERE LOWER(BTRIM("default_theme_pack")) = 'vscode';