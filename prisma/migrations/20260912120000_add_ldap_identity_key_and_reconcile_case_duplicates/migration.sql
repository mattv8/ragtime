-- Prisma sends PostgreSQL migration files as ordinary SQL; make the destructive
-- reconciliation atomic ourselves and block concurrent authentication writers.
BEGIN;
LOCK TABLE "users" IN SHARE ROW EXCLUSIVE MODE;

-- LDAP identity keys are populated lazily from entryUUID/objectGUID.  The reconciliation
-- below only joins legacy rows where the mutable DN agrees after case folding.
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "ldap_identity_key" TEXT;

DO $$
DECLARE
    duplicate_group record;
    loser_id text;
    newest record;
    enabled_factor_count integer;
    enabled_factor_id text;
    enabled_factor_user_id text;
    enabled_factor_type text;
    build_row record;
    workspace_row record;
    collision_counter integer;
    candidate_key text;
    foreign_key record;
    remaining_refs bigint;
BEGIN
    -- Validate every candidate before changing data.  A case collision is only safe when
    -- every row names the same non-empty DN and any already-known immutable keys agree.
    FOR duplicate_group IN
        SELECT lower(username) AS username_key,
               array_agg(id ORDER BY "created_at", id) AS ids,
               count(DISTINCT lower(NULLIF(btrim("ldap_dn"), ''))) AS dn_count,
               count(*) FILTER (WHERE NULLIF(btrim("ldap_dn"), '') IS NULL) AS empty_dn_count,
               count(DISTINCT "ldap_identity_key") FILTER (WHERE "ldap_identity_key" IS NOT NULL) AS identity_count
        FROM "users"
        WHERE "auth_provider" = 'ldap'::"AuthProvider"
        GROUP BY lower(username)
        HAVING count(*) > 1
    LOOP
        IF duplicate_group.dn_count <> 1 OR duplicate_group.empty_dn_count <> 0 THEN
            RAISE EXCEPTION 'LDAP case-duplicate migration refused username %: rows do not share one nonempty DN (ids: %)',
                duplicate_group.username_key, duplicate_group.ids;
        END IF;
        IF duplicate_group.identity_count > 1 THEN
            RAISE EXCEPTION 'LDAP case-duplicate migration refused username %: incompatible immutable identity keys (ids: %)',
                duplicate_group.username_key, duplicate_group.ids;
        END IF;
    END LOOP;

    FOR duplicate_group IN
        SELECT lower(username) AS username_key, array_agg(id ORDER BY "created_at", id) AS ids
        FROM "users"
        WHERE "auth_provider" = 'ldap'::"AuthProvider"
        GROUP BY lower(username)
        HAVING count(*) > 1
    LOOP
        -- Oldest ID is the stable survivor; newest profile projection wins except for
        -- explicitly managed roles, where a manual user role is the restrictive winner.
        SELECT * INTO newest FROM "users" WHERE id = ANY(duplicate_group.ids)
            ORDER BY "updated_at" DESC, id DESC LIMIT 1;

        UPDATE "users" survivor
        SET "ldap_dn" = newest."ldap_dn",
            "ldap_identity_key" = COALESCE(
                (SELECT "ldap_identity_key" FROM "users" WHERE id = ANY(duplicate_group.ids)
                 AND "ldap_identity_key" IS NOT NULL ORDER BY "updated_at" DESC, id DESC LIMIT 1),
                survivor."ldap_identity_key"),
            "source_provider" = newest."source_provider",
            "source_id" = newest."source_id",
            "cached_groups" = newest."cached_groups",
            "source_synced_at" = newest."source_synced_at",
            "source_expires_at" = newest."source_expires_at",
            "email" = newest.email,
            "display_name" = newest."display_name",
            "theme_pack" = newest."theme_pack",
            "default_chat_model" = newest."default_chat_model",
            "mfa_preferred_method" = newest."mfa_preferred_method",
            "last_login_at" = newest."last_login_at",
            role = CASE WHEN EXISTS (SELECT 1 FROM "users" WHERE id = ANY(duplicate_group.ids) AND "role_manually_set")
                        THEN CASE WHEN EXISTS (SELECT 1 FROM "users" WHERE id = ANY(duplicate_group.ids) AND "role_manually_set" AND role::text = 'user')
                                  THEN 'user'::"UserRole" ELSE 'admin'::"UserRole" END
                        ELSE newest.role END,
            "role_manually_set" = EXISTS (SELECT 1 FROM "users" WHERE id = ANY(duplicate_group.ids) AND "role_manually_set")
        WHERE survivor.id = duplicate_group.ids[1];

        SELECT count(*) FILTER (WHERE enabled), min(id) FILTER (WHERE enabled)
        INTO enabled_factor_count, enabled_factor_id
        FROM "user_mfa_factors" WHERE "user_id" = ANY(duplicate_group.ids) AND enabled;
        IF enabled_factor_count > 1 THEN
            RAISE EXCEPTION 'LDAP case-duplicate migration refused username %: multiple enabled MFA factors (ids: %)',
                duplicate_group.username_key, duplicate_group.ids;
        END IF;

        IF enabled_factor_count = 1 THEN
            SELECT "user_id", "factor_type" INTO enabled_factor_user_id, enabled_factor_type
            FROM "user_mfa_factors" WHERE id = enabled_factor_id;
        END IF;
        -- Recovery codes are TOTP credentials, not generic user metadata.  Only codes
        -- belonging to the retained enabled TOTP factor remain usable after the merge.
        IF enabled_factor_count = 1 AND enabled_factor_type = 'totp' THEN
            DELETE FROM "user_mfa_recovery_codes"
            WHERE "user_id" = ANY(duplicate_group.ids) AND "user_id" <> enabled_factor_user_id;
        ELSE
            DELETE FROM "user_mfa_recovery_codes" WHERE "user_id" = ANY(duplicate_group.ids);
        END IF;
        DELETE FROM "sessions" WHERE "user_id" = ANY(duplicate_group.ids);
        DELETE FROM "user_mfa_trusted_devices" WHERE "user_id" = ANY(duplicate_group.ids);

        FOR loser_id IN SELECT unnest(duplicate_group.ids[2:array_length(duplicate_group.ids, 1)]) LOOP
            -- Explicit composite-key conflict handling.  These rows are ACL/preference
            -- projections, so an existing survivor row is retained rather than granting a
            -- broader privilege from the duplicate account.
            DELETE FROM "auth_group_memberships" l USING "auth_group_memberships" s
              WHERE l."user_id" = loser_id AND s."user_id" = duplicate_group.ids[1] AND l."group_id" = s."group_id";
            DELETE FROM "workspace_members" l USING "workspace_members" s
              WHERE l."user_id" = loser_id AND s."user_id" = duplicate_group.ids[1] AND l."workspace_id" = s."workspace_id";
            DELETE FROM "conversation_members" l USING "conversation_members" s
              WHERE l."user_id" = loser_id AND s."user_id" = duplicate_group.ids[1] AND l."conversation_id" = s."conversation_id";
            DELETE FROM "userspace_changed_file_acknowledgements" l USING "userspace_changed_file_acknowledgements" s
              WHERE l."user_id" = loser_id AND s."user_id" = duplicate_group.ids[1]
                AND l."workspace_id" = s."workspace_id" AND l.path = s.path;
            DELETE FROM "workspace_user_preferences" l USING "workspace_user_preferences" s
              WHERE l."user_id" = loser_id AND s."user_id" = duplicate_group.ids[1] AND l."workspace_id" = s."workspace_id";
            DELETE FROM "tool_user_access" l USING "tool_user_access" s
              WHERE l."user_id" = loser_id AND s."user_id" = duplicate_group.ids[1] AND l."policy_id" = s."policy_id";

            -- The per-user external build idempotency key is not an FK.  Keep all audit
            -- rows: a full stable user ID plus a deterministic counter avoids a collision
            -- with either pre-existing keys or a prior interrupted/replayed rename.
            FOR build_row IN SELECT id, source, "request_id" FROM "external_build_requests" WHERE "user_id" = loser_id LOOP
                IF EXISTS (SELECT 1 FROM "external_build_requests" s WHERE s."user_id" = duplicate_group.ids[1]
                           AND s.source = build_row.source AND s."request_id" = build_row."request_id") THEN
                    collision_counter := 0;
                    LOOP
                        candidate_key := build_row."request_id" || ':merged:' || loser_id || ':' || collision_counter;
                        EXIT WHEN NOT EXISTS (SELECT 1 FROM "external_build_requests" e
                            WHERE e.source = build_row.source AND e."request_id" = candidate_key
                              AND e."user_id" IN (duplicate_group.ids[1], loser_id) AND e.id <> build_row.id);
                        collision_counter := collision_counter + 1;
                    END LOOP;
                    UPDATE "external_build_requests" SET "request_id" = candidate_key WHERE id = build_row.id;
                END IF;
            END LOOP;

            -- Do not lose either owned workspace.  Rename every name-normalized collision
            -- before reassignment using the complete immutable workspace ID and a stable
            -- counter, rather than a potentially ambiguous short ID prefix.
            FOR workspace_row IN SELECT id, name, "name_normalized" FROM "workspaces" WHERE "owner_user_id" = loser_id LOOP
                IF EXISTS (SELECT 1 FROM "workspaces" s WHERE s."owner_user_id" = duplicate_group.ids[1]
                           AND s."name_normalized" IS NOT DISTINCT FROM workspace_row."name_normalized") THEN
                    collision_counter := 0;
                    LOOP
                        candidate_key := COALESCE(workspace_row."name_normalized", lower(workspace_row.name))
                            || '-merged-' || workspace_row.id || '-' || collision_counter;
                        EXIT WHEN NOT EXISTS (SELECT 1 FROM "workspaces" e
                            WHERE e."owner_user_id" IN (duplicate_group.ids[1], loser_id)
                              AND e."name_normalized" = candidate_key AND e.id <> workspace_row.id);
                        collision_counter := collision_counter + 1;
                    END LOOP;
                    UPDATE "workspaces" SET name = workspace_row.name || ' (merged ' || workspace_row.id || ' ' || collision_counter || ')',
                        "name_normalized" = candidate_key WHERE id = workspace_row.id;
                END IF;
            END LOOP;

            -- A factor is unique per type.  Retain the single enabled secret; disabled
            -- duplicates are superseded before the generic FK transfer below.
            IF enabled_factor_count = 1 THEN
                DELETE FROM "user_mfa_factors" f
                 WHERE f."user_id" IN (duplicate_group.ids[1], loser_id) AND NOT f.enabled
                   AND f."factor_type" = (SELECT "factor_type" FROM "user_mfa_factors" WHERE id = enabled_factor_id)
                   AND f."user_id" <> (SELECT "user_id" FROM "user_mfa_factors" WHERE id = enabled_factor_id);
                DELETE FROM "user_mfa_factors" f
                 WHERE f."user_id" = loser_id AND f.id <> enabled_factor_id AND EXISTS (
                   SELECT 1 FROM "user_mfa_factors" s WHERE s."user_id" = duplicate_group.ids[1]
                     AND s."factor_type" = f."factor_type");
            ELSE
                DELETE FROM "user_mfa_factors" f
                 WHERE f."user_id" = loser_id AND EXISTS (
                   SELECT 1 FROM "user_mfa_factors" s WHERE s."user_id" = duplicate_group.ids[1]
                     AND s."factor_type" = f."factor_type");
            END IF;

            -- Explicit non-FK user IDs and known JSON ACL arrays.  Deliberately do not
            -- touch opaque audit/event payloads that may merely contain historical IDs.
            UPDATE "userspace_snapshots" SET "created_by_user_id" = duplicate_group.ids[1] WHERE "created_by_user_id" = loser_id;
            UPDATE "workspace_agent_access" SET "created_by_user_id" = duplicate_group.ids[1] WHERE "created_by_user_id" = loser_id;
            UPDATE "workspace_shares" SET "owner_user_id" = duplicate_group.ids[1] WHERE "owner_user_id" = loser_id;
            UPDATE "share_link_request_logs" SET "authenticated_user_id" = duplicate_group.ids[1] WHERE "authenticated_user_id" = loser_id;
            UPDATE "external_build_requests" SET "user_id" = duplicate_group.ids[1] WHERE "user_id" = loser_id;
            UPDATE "conversation_shares" SET "share_selected_user_ids" = (
                SELECT COALESCE(jsonb_agg(DISTINCT CASE WHEN value = to_jsonb(loser_id) THEN to_jsonb(duplicate_group.ids[1]) ELSE value END), '[]'::jsonb)
                FROM jsonb_array_elements("share_selected_user_ids") value) WHERE "share_selected_user_ids" ? loser_id;
            UPDATE "workspace_shares" SET "share_selected_user_ids" = (
                SELECT COALESCE(jsonb_agg(DISTINCT CASE WHEN value = to_jsonb(loser_id) THEN to_jsonb(duplicate_group.ids[1]) ELSE value END), '[]'::jsonb)
                FROM jsonb_array_elements("share_selected_user_ids") value) WHERE "share_selected_user_ids" ? loser_id;
            UPDATE "userspace_mount_sources" SET "access_user_ids" = (
                SELECT COALESCE(jsonb_agg(DISTINCT CASE WHEN value = to_jsonb(loser_id) THEN to_jsonb(duplicate_group.ids[1]) ELSE value END), '[]'::jsonb)
                FROM jsonb_array_elements("access_user_ids") value) WHERE "access_user_ids" ? loser_id;

            -- Transfer every actual FK discovered from the live catalog.  This makes the
            -- migration resilient to additive child tables while the explicit deletions
            -- above resolve all current user-containing composite uniqueness constraints.
            FOR foreign_key IN
                SELECT conrelid::regclass AS table_name, a.attname AS column_name
                FROM pg_constraint c
                JOIN unnest(c.conkey) WITH ORDINALITY keys(attnum, ord) ON true
                JOIN unnest(c.confkey) WITH ORDINALITY refs(attnum, ord) USING (ord)
                JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = keys.attnum
                WHERE c.contype = 'f' AND c.confrelid = 'users'::regclass
                  AND (SELECT attname FROM pg_attribute WHERE attrelid = c.confrelid AND attnum = refs.attnum) = 'id'
            LOOP
                EXECUTE format('UPDATE %s SET %I = $1 WHERE %I = $2', foreign_key.table_name, foreign_key.column_name, foreign_key.column_name)
                    USING duplicate_group.ids[1], loser_id;
            END LOOP;

            FOR foreign_key IN
                SELECT conrelid::regclass AS table_name, a.attname AS column_name
                FROM pg_constraint c
                JOIN unnest(c.conkey) WITH ORDINALITY keys(attnum, ord) ON true
                JOIN unnest(c.confkey) WITH ORDINALITY refs(attnum, ord) USING (ord)
                JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = keys.attnum
                WHERE c.contype = 'f' AND c.confrelid = 'users'::regclass
                  AND (SELECT attname FROM pg_attribute WHERE attrelid = c.confrelid AND attnum = refs.attnum) = 'id'
            LOOP
                EXECUTE format('SELECT count(*) FROM %s WHERE %I = $1', foreign_key.table_name, foreign_key.column_name)
                    INTO remaining_refs USING loser_id;
                IF remaining_refs <> 0 THEN
                    RAISE EXCEPTION 'LDAP case-duplicate migration left % references in %.%', remaining_refs, foreign_key.table_name, foreign_key.column_name;
                END IF;
            END LOOP;
            DELETE FROM "users" WHERE id = loser_id;
        END LOOP;
    END LOOP;
END $$;

-- The normal nullable btree unique index is the Prisma @unique contract.  Create it
-- only after loser rows are gone, so assigning an existing loser immutable key is safe.
CREATE UNIQUE INDEX IF NOT EXISTS "users_ldap_identity_key_key" ON "users" ("ldap_identity_key");
-- Prisma cannot express a provider-scoped functional uniqueness constraint.
CREATE UNIQUE INDEX IF NOT EXISTS "users_ldap_username_lower_unique"
    ON "users" (lower("username")) WHERE "auth_provider" = 'ldap'::"AuthProvider";
COMMIT;
