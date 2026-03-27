-- Make LDAP-safe uid lookup the default search filter and normalize legacy defaults.
ALTER TABLE "ldap_config"
ALTER COLUMN "user_search_filter" SET DEFAULT '(uid={username})';

-- Preserve custom filters; only normalize known legacy/default values.
UPDATE "ldap_config"
SET "user_search_filter" = '(uid={username})'
WHERE "user_search_filter" = '(|(sAMAccountName={username})(uid={username}))'
   OR "user_search_filter" = '';