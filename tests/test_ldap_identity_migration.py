"""Disposable-Postgres coverage for the LDAP identity reconciliation migration.

Set LDAP_MIGRATION_TEST_DATABASE_URL to a database created solely for this test.
The fixture creates a random schema and never uses staging or the application database.
"""

import os
import unittest
import uuid
from pathlib import Path

try:
    import psycopg2  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - depends on the optional app dependency set
    psycopg2 = None


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "prisma" / "migrations" / "20260912120000_add_ldap_identity_key_and_reconcile_case_duplicates" / "migration.sql"
DATABASE_URL = os.environ.get("LDAP_MIGRATION_TEST_DATABASE_URL")


@unittest.skipUnless(DATABASE_URL and psycopg2, "requires disposable LDAP_MIGRATION_TEST_DATABASE_URL and psycopg2")
class LdapIdentityMigrationPostgresTests(unittest.TestCase):
    def setUp(self) -> None:
        assert psycopg2 is not None
        driver = psycopg2
        self.connection = driver.connect(DATABASE_URL)
        self.database_error = driver.Error
        self.connection.autocommit = True
        self.schema = f"ldap_migration_{uuid.uuid4().hex}"
        with self.connection.cursor() as cursor:
            cursor.execute(f'CREATE SCHEMA "{self.schema}"')
            cursor.execute(f'SET search_path TO "{self.schema}"')
            cursor.execute("CREATE TYPE \"AuthProvider\" AS ENUM ('ldap', 'local', 'local_managed')")
            cursor.execute("CREATE TYPE \"UserRole\" AS ENUM ('user', 'admin')")
            cursor.execute(
                """CREATE TABLE users (
                    id text PRIMARY KEY, username text NOT NULL UNIQUE, auth_provider "AuthProvider" NOT NULL,
                    ldap_dn text, source_provider "AuthProvider", source_id text,
                    cached_groups jsonb NOT NULL DEFAULT '[]', source_synced_at timestamptz,
                    source_expires_at timestamptz, email text, display_name text, role "UserRole" NOT NULL DEFAULT 'user',
                    role_manually_set boolean NOT NULL DEFAULT false, theme_pack text, default_chat_model text,
                    mfa_preferred_method text, last_login_at timestamptz, created_at timestamptz NOT NULL,
                    updated_at timestamptz NOT NULL)"""
            )
            cursor.execute("CREATE TABLE sessions (id text PRIMARY KEY, user_id text REFERENCES users(id))")
            cursor.execute("CREATE TABLE user_mfa_trusted_devices (id text PRIMARY KEY, user_id text REFERENCES users(id))")
            cursor.execute("CREATE TABLE user_mfa_recovery_codes (id text PRIMARY KEY, user_id text REFERENCES users(id))")
            cursor.execute(
                "CREATE TABLE user_mfa_factors (id text PRIMARY KEY, user_id text REFERENCES users(id), factor_type text, secret_encrypted text, enabled boolean NOT NULL, UNIQUE(user_id, factor_type))"
            )
            cursor.execute(
                "CREATE TABLE auth_group_memberships (id text PRIMARY KEY, user_id text REFERENCES users(id), group_id text, UNIQUE(user_id, group_id))"
            )
            cursor.execute(
                "CREATE TABLE workspace_members (id text PRIMARY KEY, user_id text REFERENCES users(id), workspace_id text, UNIQUE(workspace_id, user_id))"
            )
            cursor.execute(
                "CREATE TABLE conversation_members (id text PRIMARY KEY, user_id text REFERENCES users(id), conversation_id text, UNIQUE(conversation_id, user_id))"
            )
            cursor.execute(
                "CREATE TABLE userspace_changed_file_acknowledgements (id text PRIMARY KEY, user_id text REFERENCES users(id), workspace_id text, path text, UNIQUE(workspace_id, user_id, path))"
            )
            cursor.execute(
                "CREATE TABLE workspace_user_preferences (id text PRIMARY KEY, user_id text REFERENCES users(id), workspace_id text, UNIQUE(workspace_id, user_id))"
            )
            cursor.execute("CREATE TABLE tool_user_access (id text PRIMARY KEY, user_id text REFERENCES users(id), policy_id text, UNIQUE(policy_id, user_id))")
            cursor.execute(
                "CREATE TABLE external_build_requests (id text PRIMARY KEY, user_id text, source text, request_id text, UNIQUE(user_id, source, request_id))"
            )
            cursor.execute(
                "CREATE TABLE workspaces (id text PRIMARY KEY, owner_user_id text REFERENCES users(id), name text, name_normalized text, UNIQUE(owner_user_id, name_normalized))"
            )
            cursor.execute("CREATE TABLE userspace_snapshots (id text PRIMARY KEY, created_by_user_id text)")
            cursor.execute("CREATE TABLE workspace_agent_access (id text PRIMARY KEY, created_by_user_id text)")
            cursor.execute("CREATE TABLE workspace_shares (id text PRIMARY KEY, owner_user_id text, share_selected_user_ids jsonb NOT NULL DEFAULT '[]')")
            cursor.execute("CREATE TABLE share_link_request_logs (id text PRIMARY KEY, authenticated_user_id text)")
            cursor.execute(
                "CREATE TABLE conversation_shares (id text PRIMARY KEY, owner_user_id text REFERENCES users(id), share_selected_user_ids jsonb NOT NULL DEFAULT '[]')"
            )
            cursor.execute("CREATE TABLE userspace_mount_sources (id text PRIMARY KEY, access_user_ids jsonb NOT NULL DEFAULT '[]')")

    def tearDown(self) -> None:
        self._rollback_failed_migration()
        with self.connection.cursor() as cursor:
            cursor.execute(f'DROP SCHEMA "{self.schema}" CASCADE')
        self.connection.close()

    def _execute_migration(self) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))

    def _rollback_failed_migration(self) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute("ROLLBACK")

    def _insert_user(self, user_id: str, username: str, dn: str, created_at: str, *, manual_role: bool = False, role: str = "admin") -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO users (id, username, auth_provider, ldap_dn, email, display_name, role, role_manually_set, created_at, updated_at)
                   VALUES (%s, %s, 'ldap'::"AuthProvider", %s, %s, %s, %s::"UserRole", %s, %s::timestamptz, %s::timestamptz)""",
                (user_id, username, dn, f"{username}@example.test", username, role, manual_role, created_at, created_at),
            )

    def test_merges_staging_shape_and_preserves_resources(self) -> None:
        self._insert_user("old", "Matt", "uid=Matt,dc=example", "2026-01-01", manual_role=True, role="user")
        self._insert_user("new", "matt", "UID=matt,DC=example", "2026-02-01")
        with self.connection.cursor() as cursor:
            # Exercises safe assignment of a pre-existing immutable key held by the loser.
            cursor.execute("ALTER TABLE users ADD COLUMN ldap_identity_key text")
            cursor.execute("UPDATE users SET ldap_identity_key = 'entryuuid:6fcba6a6-5c5c-103c-8c0c-dd0077222c' WHERE id = 'new'")
            for number in range(12):
                cursor.execute("INSERT INTO auth_group_memberships VALUES (%s, %s, %s)", (f"old-group-{number}", "old", str(number)))
                cursor.execute("INSERT INTO auth_group_memberships VALUES (%s, %s, %s)", (f"new-group-{number}", "new", str(number)))
            cursor.execute("INSERT INTO user_mfa_factors VALUES ('new-factor', 'new', 'totp', 'encrypted-secret', true)")
            cursor.execute("INSERT INTO user_mfa_recovery_codes VALUES ('recovery', 'new')")
            cursor.execute("INSERT INTO user_mfa_recovery_codes VALUES ('old-recovery', 'old')")
            cursor.execute("INSERT INTO sessions VALUES ('old-session', 'old'), ('new-session', 'new')")
            cursor.execute("INSERT INTO user_mfa_trusted_devices VALUES ('device', 'new')")
            cursor.execute(
                "INSERT INTO workspaces VALUES ('survivor-ws', 'old', 'Project', 'project'), ('blocking-ws', 'old', 'Project copy', 'project-merged-loser-ws-0'), ('loser-ws', 'new', 'Project', 'project')"
            )
            cursor.execute("INSERT INTO workspace_members VALUES ('old-member', 'old', 'survivor-ws'), ('new-member', 'new', 'survivor-ws')")
            cursor.execute(
                "INSERT INTO conversation_members VALUES ('old-conversation-member', 'old', 'conversation'), ('new-conversation-member', 'new', 'conversation')"
            )
            cursor.execute("INSERT INTO workspace_user_preferences VALUES ('old-preference', 'old', 'survivor-ws'), ('new-preference', 'new', 'survivor-ws')")
            cursor.execute("INSERT INTO tool_user_access VALUES ('old-tool-access', 'old', 'policy'), ('new-tool-access', 'new', 'policy')")
            cursor.execute(
                "INSERT INTO userspace_changed_file_acknowledgements VALUES ('old-ack', 'old', 'survivor-ws', 'src/app.py'), ('new-ack', 'new', 'survivor-ws', 'src/app.py')"
            )
            cursor.execute(
                "INSERT INTO external_build_requests VALUES ('old-build', 'old', 'agent', 'same'), ('new-build', 'new', 'agent', 'same'), ('blocking-build', 'old', 'agent', 'same:merged:new:0')"
            )
            cursor.execute("INSERT INTO userspace_snapshots VALUES ('snapshot', 'new')")
            cursor.execute("INSERT INTO workspace_agent_access VALUES ('access', 'new')")
            cursor.execute("INSERT INTO workspace_shares VALUES ('share', 'new', '[\"new\"]')")
            cursor.execute("INSERT INTO conversation_shares VALUES ('conversation-share', 'new', '[\"new\"]')")
            cursor.execute("INSERT INTO userspace_mount_sources VALUES ('mount', '[\"new\"]')")
            cursor.execute("INSERT INTO share_link_request_logs VALUES ('share-log', 'new')")
        self._execute_migration()
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT id, role::text, email, ldap_identity_key FROM users ORDER BY id")
            self.assertEqual(cursor.fetchall(), [("old", "user", "matt@example.test", "entryuuid:6fcba6a6-5c5c-103c-8c0c-dd0077222c")])
            cursor.execute("SELECT count(*) FROM auth_group_memberships WHERE user_id = 'old'")
            self.assertEqual(cursor.fetchone(), (12,))
            cursor.execute("SELECT user_id, secret_encrypted FROM user_mfa_factors")
            self.assertEqual(cursor.fetchall(), [("old", "encrypted-secret")])
            cursor.execute("SELECT user_id FROM user_mfa_recovery_codes")
            self.assertEqual(cursor.fetchall(), [("old",)])
            cursor.execute("SELECT count(*) FROM sessions")
            self.assertEqual(cursor.fetchone(), (0,))
            cursor.execute("SELECT name_normalized FROM workspaces WHERE id = 'loser-ws'")
            self.assertEqual(cursor.fetchone(), ("project-merged-loser-ws-1",))
            for table_name in (
                "workspace_members",
                "conversation_members",
                "workspace_user_preferences",
                "tool_user_access",
                "userspace_changed_file_acknowledgements",
            ):
                cursor.execute(f"SELECT count(*) FROM {table_name} WHERE user_id = 'old'")
                self.assertEqual(cursor.fetchone(), (1,), table_name)
            cursor.execute("SELECT share_selected_user_ids FROM workspace_shares")
            self.assertEqual(cursor.fetchone(), (["old"],))
            cursor.execute("SELECT share_selected_user_ids FROM conversation_shares")
            self.assertEqual(cursor.fetchone(), (["old"],))
            cursor.execute("SELECT access_user_ids FROM userspace_mount_sources")
            self.assertEqual(cursor.fetchone(), (["old"],))
            cursor.execute("SELECT created_by_user_id FROM userspace_snapshots")
            self.assertEqual(cursor.fetchone(), ("old",))
            cursor.execute("SELECT created_by_user_id FROM workspace_agent_access")
            self.assertEqual(cursor.fetchone(), ("old",))
            cursor.execute("SELECT authenticated_user_id FROM share_link_request_logs")
            self.assertEqual(cursor.fetchone(), ("old",))
            cursor.execute("SELECT request_id FROM external_build_requests WHERE id = 'new-build'")
            self.assertEqual(cursor.fetchone(), ("same:merged:new:1",))
            with self.assertRaises(self.database_error):
                cursor.execute(
                    "INSERT INTO users (id, username, auth_provider, created_at, updated_at) VALUES ('ldap-case', 'MATT', 'ldap'::\"AuthProvider\", now(), now())"
                )
            cursor.execute(
                "INSERT INTO users (id, username, auth_provider, created_at, updated_at) VALUES ('local-case', 'MATT', 'local'::\"AuthProvider\", now(), now())"
            )
            with self.assertRaises(self.database_error):
                cursor.execute(
                    "INSERT INTO users (id, username, auth_provider, ldap_identity_key, created_at, updated_at) VALUES ('key-collision', 'other-ldap', 'ldap'::\"AuthProvider\", 'entryuuid:6fcba6a6-5c5c-103c-8c0c-dd0077222c', now(), now())"
                )
            cursor.execute(
                "INSERT INTO users (id, username, auth_provider, created_at, updated_at) VALUES ('null-key-one', 'null-one', 'local'::\"AuthProvider\", now(), now()), ('null-key-two', 'null-two', 'local'::\"AuthProvider\", now(), now())"
            )
        self._execute_migration()  # rerun is a no-op after the ledger would normally prevent it

    def test_conflicting_enabled_mfa_rolls_back(self) -> None:
        self._insert_user("old", "Matt", "uid=matt,dc=example", "2026-01-01")
        self._insert_user("new", "matt", "UID=MATT,DC=EXAMPLE", "2026-02-01")
        with self.connection.cursor() as cursor:
            cursor.execute("INSERT INTO user_mfa_factors VALUES ('one', 'old', 'totp', 'one', true), ('two', 'new', 'sms', 'two', true)")
        with self.assertRaises(self.database_error):
            self._execute_migration()
        self._rollback_failed_migration()
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM users")
            self.assertEqual(cursor.fetchone(), (2,))

    def test_matching_case_duplicate_merges_without_affecting_local_names(self) -> None:
        self._insert_user("old", "Matt", "uid=matt,dc=example", "2026-01-01")
        self._insert_user("new", "matt", "UID=MATT,DC=EXAMPLE", "2026-02-01")
        self._execute_migration()
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT id FROM users")
            self.assertEqual(cursor.fetchall(), [("old",)])
            cursor.execute(
                "INSERT INTO users (id, username, auth_provider, created_at, updated_at) VALUES ('local-one', 'Case', 'local'::\"AuthProvider\", now(), now()), ('local-two', 'case', 'local'::\"AuthProvider\", now(), now())"
            )
            cursor.execute("SELECT count(*) FROM users WHERE auth_provider = 'local'::\"AuthProvider\"")
            self.assertEqual(cursor.fetchone(), (2,))

    def test_recovery_codes_are_revoked_without_an_enabled_totp_factor(self) -> None:
        self._insert_user("old", "Matt", "uid=matt,dc=example", "2026-01-01")
        self._insert_user("new", "matt", "UID=MATT,DC=EXAMPLE", "2026-02-01")
        with self.connection.cursor() as cursor:
            cursor.execute("INSERT INTO user_mfa_factors VALUES ('sms', 'new', 'sms', 'secret', true)")
            cursor.execute("INSERT INTO user_mfa_recovery_codes VALUES ('old-code', 'old'), ('new-code', 'new')")
        self._execute_migration()
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM user_mfa_recovery_codes")
            self.assertEqual(cursor.fetchone(), (0,))

    def test_invalid_ldap_dn_group_rolls_back_all_groups(self) -> None:
        self._insert_user("safe-old", "Safe", "uid=safe,dc=example", "2026-01-01")
        self._insert_user("safe-new", "safe", "UID=SAFE,DC=example", "2026-02-01")
        self._insert_user("loser-one", "Matt", "uid=one,dc=example", "2026-01-01")
        self._insert_user("loser-two", "matt", "uid=two,dc=example", "2026-02-01")
        with self.connection.cursor() as cursor:
            for number in range(12):
                cursor.execute("INSERT INTO auth_group_memberships VALUES (%s, %s, %s)", (f"old-group-{number}", "safe-old", str(number)))
                cursor.execute("INSERT INTO auth_group_memberships VALUES (%s, %s, %s)", (f"one-group-{number}", "loser-one", str(number)))
                cursor.execute("INSERT INTO auth_group_memberships VALUES (%s, %s, %s)", (f"two-group-{number}", "loser-two", str(number)))
            cursor.execute(
                "INSERT INTO workspaces VALUES ('old-ws', 'safe-old', 'Project', 'project'), ('one-ws', 'loser-one', 'Project', 'project'), ('two-ws', 'loser-two', 'Project', 'project')"
            )
            cursor.execute("INSERT INTO user_mfa_factors VALUES ('last-factor', 'loser-two', 'totp', 'last-secret', true)")
            cursor.execute("INSERT INTO user_mfa_recovery_codes VALUES ('last-recovery', 'loser-two')")
        with self.assertRaises(self.database_error):
            self._execute_migration()
        self._rollback_failed_migration()
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT id FROM users ORDER BY id")
            self.assertEqual(cursor.fetchall(), [("loser-one",), ("loser-two",), ("safe-new",), ("safe-old",)])
            cursor.execute("SELECT count(*) FROM auth_group_memberships")
            self.assertEqual(cursor.fetchone(), (36,))
            cursor.execute("SELECT user_id, secret_encrypted FROM user_mfa_factors")
            self.assertEqual(cursor.fetchall(), [("loser-two", "last-secret")])
