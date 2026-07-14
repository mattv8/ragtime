import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

from cryptography.fernet import InvalidToken
from pydantic import ValidationError

from ragtime.core import encryption
from ragtime.core.encryption import CONNECTION_CONFIG_PASSWORD_FIELDS, ENCRYPTED_PREFIX, attempt_decrypt, decrypt_secret, encrypt_secret
from ragtime.indexer import routes
from ragtime.indexer.models import AppSettings, ToolConfig, ToolType
from ragtime.indexer.repository import IndexerRepository

EXPORT_PASSWORD = "Export-Test1!"


class PasswordEncryptionTests(unittest.TestCase):
    def test_encrypt_decrypt_roundtrip(self) -> None:
        plaintext = "hello world"
        password = "super-secret-password"
        salt, token = encryption.encrypt_with_password(plaintext, password)

        self.assertIsInstance(salt, str)
        self.assertIsInstance(token, str)
        self.assertGreater(len(salt), 0)
        self.assertGreater(len(token), 0)

        decrypted = encryption.decrypt_with_password(token, password, salt)
        self.assertEqual(decrypted, plaintext)

    def test_wrong_password_fails_decryption(self) -> None:
        plaintext = "hello world"
        password = "super-secret-password"
        salt, token = encryption.encrypt_with_password(plaintext, password)

        with self.assertRaises(InvalidToken):
            encryption.decrypt_with_password(token, "wrong-password", salt)

    def test_derived_key_depends_on_password_and_salt(self) -> None:
        password = "secret"
        salt = b"salt-1234567890ab"
        key1 = encryption.derive_password_fernet_key(password, salt)
        key2 = encryption.derive_password_fernet_key(password, b"different-salt-00")
        key3 = encryption.derive_password_fernet_key("other-password", salt)

        self.assertIsInstance(key1, bytes)
        self.assertEqual(len(key1), 44)  # base64-encoded 32-byte key
        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_docker_ssh_credentials_are_connection_secrets(self) -> None:
        self.assertIn("docker_ssh_password", CONNECTION_CONFIG_PASSWORD_FIELDS)
        self.assertIn("docker_ssh_key_content", CONNECTION_CONFIG_PASSWORD_FIELDS)
        self.assertIn("docker_ssh_key_passphrase", CONNECTION_CONFIG_PASSWORD_FIELDS)


class ToolExportImportHelpersTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_export_envelope_contains_only_importable_fields(self) -> None:
        config = ToolConfig(
            id="tool-123",
            name="My Postgres",
            tool_type=ToolType.POSTGRES,
            enabled=True,
            description="A test tool",
            connection_config={"host": "localhost", "password": "secret"},
            max_results=50,
            timeout_max_seconds=120,
            allow_write=True,
        )

        envelope = routes._build_tool_export_envelope(config, EXPORT_PASSWORD)

        self.assertEqual(envelope["format"], "ragtime-tool-config")
        self.assertEqual(envelope["version"], 1)
        self.assertEqual(envelope["kdf"]["algorithm"], "PBKDF2-SHA256")
        self.assertEqual(envelope["kdf"]["iterations"], 600000)
        self.assertIn("salt", envelope["kdf"])
        self.assertIsInstance(envelope["payload"], str)
        self.assertEqual(envelope["tool_hint"]["name"], "My Postgres")
        self.assertEqual(envelope["tool_hint"]["tool_type"], "postgres")

        # Payload must decrypt and contain only importable fields
        decrypted = encryption.decrypt_with_password(envelope["payload"], EXPORT_PASSWORD, envelope["kdf"]["salt"])
        payload = json.loads(decrypted)
        self.assertEqual(
            set(payload.keys()),
            {"name", "tool_type", "description", "connection_config", "max_results", "timeout_max_seconds", "allow_write"},
        )
        self.assertEqual(payload["name"], "My Postgres")
        self.assertEqual(payload["connection_config"]["password"], "secret")
        self.assertNotIn("id", payload)
        self.assertNotIn("enabled", payload)

    def test_validate_envelope_rejects_bad_format_version_or_kdf(self) -> None:
        good = routes._build_tool_export_envelope(
            ToolConfig(
                name="Tool",
                tool_type=ToolType.POSTGRES,
                connection_config={},
            ),
            EXPORT_PASSWORD,
        )

        with self.assertRaises(ValueError):
            routes._validate_tool_export_envelope({**good, "format": "other"})
        with self.assertRaises(ValueError):
            routes._validate_tool_export_envelope({**good, "version": 2})
        with self.assertRaises(ValueError):
            routes._validate_tool_export_envelope({**good, "kdf": {**good["kdf"], "algorithm": "PBKDF2-SHA512"}})
        with self.assertRaises(ValueError):
            routes._validate_tool_export_envelope({**good, "kdf": {**good["kdf"], "iterations": 100000}})

    def test_deduplicate_tool_name(self) -> None:
        existing = {"My Tool"}
        self.assertEqual(routes._deduplicate_tool_name("My Tool", existing), "My Tool Copy")
        existing.add("My Tool Copy")
        self.assertEqual(routes._deduplicate_tool_name("My Tool", existing), "My Tool Copy 2")
        self.assertEqual(routes._deduplicate_tool_name("New Tool", existing), "New Tool")


class ToolExportImportRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_export_then_import_roundtrip(self) -> None:
        original = ToolConfig(
            id="tool-abc",
            name="Exportable Postgres",
            tool_type=ToolType.POSTGRES,
            enabled=True,
            description="Round-trip test",
            connection_config={"host": "db.example.com", "password": "hunter2"},
            max_results=42,
            timeout_max_seconds=60,
            allow_write=True,
        )

        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(return_value=original)
        mock_repo.get_settings = mock.AsyncMock(return_value=AppSettings())
        mock_repo.list_tool_configs = mock.AsyncMock(return_value=[])
        mock_repo.create_tool_config = mock.AsyncMock(
            return_value=ToolConfig(
                id="imported-123",
                name="Exportable Postgres",
                tool_type=ToolType.POSTGRES,
                enabled=False,
                description="Round-trip test",
                connection_config={"host": "db.example.com", "password": "hunter2"},
                max_results=42,
                timeout_max_seconds=60,
                allow_write=True,
            )
        )

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.rag.initialize", mock.AsyncMock()),
            mock.patch("ragtime.indexer.routes.notify_tools_changed") as mock_notify,
            mock.patch("ragtime.indexer.routes.invalidate_settings_cache") as mock_invalidate,
        ):
            response = await routes.export_tool_config("tool-abc", routes.ToolExportRequest(password=EXPORT_PASSWORD))
            body_bytes = response.body if isinstance(response.body, bytes) else bytes(response.body)
            envelope = json.loads(body_bytes.decode("utf-8"))
            filename = response.headers["Content-Disposition"]
            self.assertIn("attachment", filename)
            self.assertIn(".json", filename)

            imported = await routes.import_tool_config(routes.ToolImportRequest(password=EXPORT_PASSWORD, file_content=json.dumps(envelope)))

        self.assertEqual(imported.name, "Exportable Postgres")
        self.assertEqual(imported.connection_config["password"], "hunter2")
        self.assertEqual(imported.enabled, False)
        mock_notify.assert_called_once()
        mock_invalidate.assert_called_once()

    async def test_import_renames_on_collision(self) -> None:
        envelope = routes._build_tool_export_envelope(
            ToolConfig(
                name="Existing Tool",
                tool_type=ToolType.POSTGRES,
                connection_config={},
            ),
            EXPORT_PASSWORD,
        )

        existing = SimpleNamespace(name="Existing Tool")
        mock_repo = mock.AsyncMock()
        mock_repo.list_tool_configs = mock.AsyncMock(return_value=[existing])

        created: ToolConfig | None = None

        async def fake_create(config: ToolConfig) -> ToolConfig:
            nonlocal created
            created = config
            data = config.model_dump()
            data["id"] = "imported-1"
            return ToolConfig(**data)

        mock_repo.create_tool_config = fake_create

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.rag.initialize", mock.AsyncMock()),
            mock.patch("ragtime.indexer.routes.notify_tools_changed"),
            mock.patch("ragtime.indexer.routes.invalidate_settings_cache"),
        ):
            await routes.import_tool_config(routes.ToolImportRequest(password=EXPORT_PASSWORD, file_content=json.dumps(envelope)))

        self.assertIsNotNone(created)
        created_config = cast(ToolConfig, created)
        self.assertEqual(created_config.name, "Existing Tool Copy")

    async def test_import_clears_schema_index_freshness_markers(self) -> None:
        envelope = routes._build_tool_export_envelope(
            ToolConfig(
                name="Schema Tool",
                tool_type=ToolType.POSTGRES,
                connection_config={
                    "schema_index_enabled": True,
                    "schema_index_interval_hours": 12,
                    "schema_index_timezone": "UTC",
                    "last_schema_indexed_at": "2026-01-01T00:00:00+00:00",
                    "schema_hash": "stale-hash",
                },
            ),
            EXPORT_PASSWORD,
        )
        mock_repo = mock.AsyncMock()
        mock_repo.list_tool_configs = mock.AsyncMock(return_value=[])
        created: ToolConfig | None = None

        async def fake_create(config: ToolConfig) -> ToolConfig:
            nonlocal created
            created = config
            return config

        mock_repo.create_tool_config = fake_create

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.rag.initialize", mock.AsyncMock()),
            mock.patch("ragtime.indexer.routes.notify_tools_changed"),
            mock.patch("ragtime.indexer.routes.invalidate_settings_cache"),
        ):
            await routes.import_tool_config(routes.ToolImportRequest(password=EXPORT_PASSWORD, file_content=json.dumps(envelope)))

        self.assertIsNotNone(created)
        imported_config = cast(ToolConfig, created).connection_config
        self.assertEqual(imported_config["schema_index_enabled"], True)
        self.assertEqual(imported_config["schema_index_interval_hours"], 12)
        self.assertEqual(imported_config["schema_index_timezone"], "UTC")
        self.assertNotIn("last_schema_indexed_at", imported_config)
        self.assertNotIn("schema_hash", imported_config)

    async def test_import_clears_filesystem_index_freshness_marker(self) -> None:
        envelope = routes._build_tool_export_envelope(
            ToolConfig(
                name="Filesystem Tool",
                tool_type=ToolType.FILESYSTEM_INDEXER,
                connection_config={
                    "mount_type": "local",
                    "base_path": "/data/files",
                    "index_name": "old-index",
                    "reindex_interval_hours": 12,
                    "last_indexed_at": "2026-01-01T00:00:00+00:00",
                },
            ),
            EXPORT_PASSWORD,
        )
        mock_repo = mock.AsyncMock()
        mock_repo.list_tool_configs = mock.AsyncMock(return_value=[])
        created: ToolConfig | None = None

        async def fake_create(config: ToolConfig) -> ToolConfig:
            nonlocal created
            created = config
            return config

        mock_repo.create_tool_config = fake_create

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.rag.initialize", mock.AsyncMock()),
            mock.patch("ragtime.indexer.routes.notify_tools_changed"),
            mock.patch("ragtime.indexer.routes.invalidate_settings_cache"),
        ):
            await routes.import_tool_config(routes.ToolImportRequest(password=EXPORT_PASSWORD, file_content=json.dumps(envelope)))

        self.assertIsNotNone(created)
        imported_config = cast(ToolConfig, created).connection_config
        self.assertEqual(imported_config["reindex_interval_hours"], 12)
        self.assertNotIn("last_indexed_at", imported_config)

    async def test_import_wrong_password_rejected(self) -> None:
        envelope = routes._build_tool_export_envelope(
            ToolConfig(
                name="Tool",
                tool_type=ToolType.POSTGRES,
                connection_config={},
            ),
            EXPORT_PASSWORD,
        )

        with self.assertRaises(Exception) as ctx:
            await routes.import_tool_config(routes.ToolImportRequest(password="wrong", file_content=json.dumps(envelope)))

        self.assertIn("password", str(ctx.exception).lower())

    def test_export_request_accepts_non_empty_password(self) -> None:
        request = routes.ToolExportRequest(password="x")
        self.assertEqual(request.password, "x")

    async def test_export_rejects_when_tool_has_undecryptable_credentials(self) -> None:
        config = ToolConfig(
            id="tool-abc",
            name="Broken Postgres",
            tool_type=ToolType.POSTGRES,
            connection_config={"host": "db.example.com", "password": ""},
            undecryptable_fields=["password"],
        )
        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(return_value=config)
        mock_repo.get_settings = mock.AsyncMock(return_value=AppSettings())

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.encryption_key_mismatch_detected", return_value=False),
        ):
            with self.assertRaises(Exception) as ctx:
                await routes.export_tool_config(
                    "tool-abc",
                    routes.ToolExportRequest(password=EXPORT_PASSWORD),
                )

        self.assertEqual(getattr(ctx.exception, "status_code", None), 409)
        detail = str(ctx.exception).lower()
        self.assertIn("password", detail)
        self.assertIn("clear", detail)

    async def test_export_allows_when_global_mismatch_but_tool_healthy(self) -> None:
        config = ToolConfig(
            id="tool-abc",
            name="Healthy Postgres",
            tool_type=ToolType.POSTGRES,
            connection_config={"host": "db.example.com", "password": "secret"},
            undecryptable_fields=[],
        )
        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(return_value=config)
        mock_repo.get_settings = mock.AsyncMock(return_value=AppSettings())

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.encryption_key_mismatch_detected", return_value=True),
            mock.patch("ragtime.indexer.routes.rag.initialize", mock.AsyncMock()),
        ):
            response = await routes.export_tool_config(
                "tool-abc",
                routes.ToolExportRequest(password=EXPORT_PASSWORD),
            )

        body = response.body if isinstance(response.body, bytes) else bytes(response.body)
        envelope = json.loads(body.decode("utf-8"))
        self.assertEqual(envelope["tool_hint"]["name"], "Healthy Postgres")

    async def test_import_invalid_decrypted_payload_returns_400(self) -> None:
        salt, token = encryption.encrypt_with_password(
            json.dumps(
                {
                    "name": "Invalid Tool",
                    "tool_type": "postgres",
                    "description": "",
                    "connection_config": {},
                    "max_results": "not-a-number",
                    "timeout_max_seconds": 300,
                    "allow_write": False,
                }
            ),
            EXPORT_PASSWORD,
        )
        envelope = {
            "format": "ragtime-tool-config",
            "version": 1,
            "kdf": {
                "algorithm": "PBKDF2-SHA256",
                "iterations": 600000,
                "salt": salt,
            },
            "tool_hint": {"name": "Invalid Tool", "tool_type": "postgres"},
            "payload": token,
        }
        mock_repo = mock.AsyncMock()
        mock_repo.list_tool_configs = mock.AsyncMock(return_value=[])

        with mock.patch("ragtime.indexer.routes.repository", mock_repo):
            with self.assertRaises(Exception) as ctx:
                await routes.import_tool_config(routes.ToolImportRequest(password=EXPORT_PASSWORD, file_content=json.dumps(envelope)))

        self.assertEqual(getattr(ctx.exception, "status_code", None), 400)
        self.assertIn("invalid imported tool config", str(ctx.exception).lower())

    async def test_import_filesystem_faiss_index_collision_returns_409(self) -> None:
        envelope = routes._build_tool_export_envelope(
            ToolConfig(
                name="Existing Filesystem",
                tool_type=ToolType.FILESYSTEM_INDEXER,
                connection_config={"vector_store_type": "faiss", "index_name": "existing_filesystem"},
            ),
            EXPORT_PASSWORD,
        )
        existing = SimpleNamespace(name="Existing Filesystem")
        mock_repo = mock.AsyncMock()
        mock_repo.list_tool_configs = mock.AsyncMock(return_value=[existing])

        with tempfile.TemporaryDirectory() as tmpdir:
            collision_name = routes.safe_tool_name("Existing Filesystem Copy")
            Path(tmpdir, collision_name).mkdir()
            with (
                mock.patch("ragtime.indexer.routes.repository", mock_repo),
                mock.patch("ragtime.indexer.vector_backends.FAISS_INDEX_BASE_PATH", Path(tmpdir)),
            ):
                with self.assertRaises(Exception) as ctx:
                    await routes.import_tool_config(routes.ToolImportRequest(password=EXPORT_PASSWORD, file_content=json.dumps(envelope)))

        self.assertEqual(getattr(ctx.exception, "status_code", None), 409)
        mock_repo.create_tool_config.assert_not_called()

    async def test_export_missing_tool_returns_404(self) -> None:
        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(return_value=None)
        mock_repo.get_settings = mock.AsyncMock(return_value=AppSettings())

        with mock.patch("ragtime.indexer.routes.repository", mock_repo):
            with self.assertRaises(Exception) as ctx:
                await routes.export_tool_config("missing", routes.ToolExportRequest(password=EXPORT_PASSWORD))

        self.assertEqual(getattr(ctx.exception, "status_code", None), 404)


class ExportPasswordPolicyTests(unittest.TestCase):
    def test_default_policy_rejects_short_password(self) -> None:
        settings = AppSettings()
        with self.assertRaises(Exception) as ctx:
            routes.validate_export_password_strength("Short1!", settings)
        self.assertEqual(getattr(ctx.exception, "status_code", None), 400)
        self.assertIn("at least 12 characters", str(getattr(ctx.exception, "detail", "")))

    def test_default_policy_rejects_missing_character_classes(self) -> None:
        settings = AppSettings()
        cases = [
            ("lower_only!@#no", "uppercase"),
            ("UPPER_ONLY!@#NO", "lowercase"),
            ("NoDigits!@#abcd", "number"),
            ("NoSpecial12345", "special"),
        ]
        for password, expected in cases:
            with self.subTest(password=password, expected=expected):
                with self.assertRaises(Exception) as ctx:
                    routes.validate_export_password_strength(password, settings)
                self.assertEqual(getattr(ctx.exception, "status_code", None), 400)
                self.assertIn(expected, str(getattr(ctx.exception, "detail", "")).lower())

    def test_default_policy_accepts_strong_password(self) -> None:
        settings = AppSettings()
        # Should not raise
        routes.validate_export_password_strength("Export-Test1!", settings)

    def test_relaxed_policy_accepts_weaker_password(self) -> None:
        settings = AppSettings(
            export_password_min_length=8,
            export_password_require_special=False,
        )
        # Should not raise
        routes.validate_export_password_strength("Abcdef12", settings)

    def test_relaxed_policy_still_enforces_minimum_length(self) -> None:
        settings = AppSettings(
            export_password_min_length=8,
            export_password_require_special=False,
        )
        with self.assertRaises(Exception) as ctx:
            routes.validate_export_password_strength("Abcd12", settings)
        self.assertEqual(getattr(ctx.exception, "status_code", None), 400)
        self.assertIn("at least 8 characters", str(getattr(ctx.exception, "detail", "")))


class ExportPasswordPolicyRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_export_route_enforces_default_password_policy(self) -> None:
        config = ToolConfig(
            id="tool-abc",
            name="Exportable Postgres",
            tool_type=ToolType.POSTGRES,
            connection_config={"host": "db.example.com", "password": ""},
        )
        mock_repo = mock.AsyncMock()
        mock_repo.get_tool_config = mock.AsyncMock(return_value=config)
        mock_repo.get_settings = mock.AsyncMock(return_value=AppSettings())

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.encryption_key_mismatch_detected", return_value=False),
        ):
            with self.assertRaises(Exception) as ctx:
                await routes.export_tool_config("tool-abc", routes.ToolExportRequest(password="weak"))

        self.assertEqual(getattr(ctx.exception, "status_code", None), 400)
        detail = str(getattr(ctx.exception, "detail", "")).lower()
        self.assertIn("password", detail)
        self.assertIn("at least 12 characters", detail)


class ToolConfigRepositoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        from ragtime.core.encryption import reset_key_mismatch_state

        reset_key_mismatch_state()

    def tearDown(self) -> None:
        from ragtime.core.encryption import reset_key_mismatch_state

        reset_key_mismatch_state()

    def _fake_prisma_tool(self, connection_config: dict, **overrides) -> SimpleNamespace:
        data = {
            "id": "tool-123",
            "name": "Test Tool",
            "toolType": "postgres",
            "enabled": True,
            "description": "",
            "connectionConfig": connection_config,
            "maxResults": 100,
            "timeoutMaxSeconds": 300,
            "allowWrite": False,
            "sortOrder": 0,
            "groupId": None,
            "group": None,
            "lastTestAt": None,
            "lastTestResult": None,
            "lastTestError": None,
            "createdAt": None,
            "updatedAt": None,
        }
        data.update(overrides)
        return SimpleNamespace(**data)

    def test_model_populates_undecryptable_fields(self) -> None:
        repo = IndexerRepository()
        bad_password = f"{ENCRYPTED_PREFIX}invalid-token-value"
        good_token = encrypt_secret("valid-token")
        raw_config = {
            "host": "db.example.com",
            "port": 5432,
            "password": bad_password,
            "token": good_token,
            "plain": "not-secret",
        }
        prisma_tool = self._fake_prisma_tool(connection_config=raw_config)

        model = repo._prisma_tool_config_to_model(prisma_tool)

        self.assertIn("password", model.undecryptable_fields)
        self.assertNotIn("token", model.undecryptable_fields)
        self.assertNotIn("plain", model.undecryptable_fields)
        self.assertEqual(model.connection_config["password"], "")
        self.assertEqual(model.connection_config["token"], "valid-token")
        self.assertEqual(model.connection_config["plain"], "not-secret")

    def test_model_no_undecryptable_fields_for_plaintext(self) -> None:
        repo = IndexerRepository()
        prisma_tool = self._fake_prisma_tool(connection_config={"host": "db.example.com", "password": "plaintext", "token": ""})

        model = repo._prisma_tool_config_to_model(prisma_tool)

        self.assertEqual(model.undecryptable_fields, [])
        self.assertEqual(model.connection_config["password"], "plaintext")

    async def test_create_encrypts_and_model_decrypts_docker_ssh_credentials(self) -> None:
        repo = IndexerRepository()
        captured_config: dict | None = None
        tool = self._fake_prisma_tool(
            connection_config={},
            id="tool-docker-ssh",
            name="Remote Docker",
        )

        async def fake_create(data, include=None):
            nonlocal captured_config
            del include
            captured_config = dict(data["connectionConfig"].data if hasattr(data["connectionConfig"], "data") else data["connectionConfig"])
            tool.connectionConfig = captured_config
            return tool

        db = mock.AsyncMock()
        db.toolconfig.create = fake_create

        config = ToolConfig(
            name="Remote Docker",
            tool_type=ToolType.POSTGRES,
            connection_config={
                "container": "postgres-postgres-1",
                "docker_ssh_host": "remote.example.com",
                "docker_ssh_user": "deploy",
                "docker_ssh_password": "ssh-secret",
                "docker_ssh_key_content": "private-key",
                "docker_ssh_key_passphrase": "key-passphrase",
            },
        )

        with mock.patch.object(repo, "_get_db", return_value=db):
            result = await repo.create_tool_config(config)

        assert captured_config is not None
        for field, plaintext in {
            "docker_ssh_password": "ssh-secret",
            "docker_ssh_key_content": "private-key",
            "docker_ssh_key_passphrase": "key-passphrase",
        }.items():
            encrypted = captured_config[field]
            self.assertTrue(encrypted.startswith(ENCRYPTED_PREFIX))
            self.assertEqual(decrypt_secret(encrypted), plaintext)

        self.assertEqual(result.connection_config["docker_ssh_password"], "ssh-secret")
        self.assertEqual(result.connection_config["docker_ssh_key_content"], "private-key")
        self.assertEqual(result.connection_config["docker_ssh_key_passphrase"], "key-passphrase")

    async def test_clear_undecryptable_credentials_removes_only_broken_secret_fields(self) -> None:
        repo = IndexerRepository()
        bad_password = f"{ENCRYPTED_PREFIX}bad-password-token"
        good_token = encrypt_secret("valid-token")
        raw_config = {
            "host": "db.example.com",
            "password": bad_password,
            "token": good_token,
            "user": "admin",
        }

        updated_config: dict | None = None

        async def fake_find_unique(where, include=None):
            del include
            if where.get("id") == "tool-123":
                return self._fake_prisma_tool(connection_config=updated_config if updated_config is not None else raw_config)
            return None

        async def fake_update(where, data):
            nonlocal updated_config
            if where.get("id") == "tool-123":
                json_value = data.get("connectionConfig")
                updated_config = dict(json_value.data if hasattr(json_value, "data") else json_value or raw_config)

        db = mock.AsyncMock()
        db.toolconfig.find_unique = fake_find_unique
        db.toolconfig.update = fake_update

        with mock.patch.object(repo, "_get_db", return_value=db):
            result = await repo.clear_tool_undecryptable_credentials("tool-123")

        assert result is not None
        self.assertNotIn("password", updated_config or {})
        self.assertEqual((updated_config or {}).get("token"), good_token)
        self.assertEqual((updated_config or {}).get("user"), "admin")
        self.assertEqual((updated_config or {}).get("host"), "db.example.com")
        self.assertEqual(result.undecryptable_fields, [])

    async def test_clear_undecryptable_credentials_returns_none_when_missing(self) -> None:
        repo = IndexerRepository()

        db = mock.AsyncMock()
        db.toolconfig.find_unique = mock.AsyncMock(return_value=None)

        with mock.patch.object(repo, "_get_db", return_value=db):
            result = await repo.clear_tool_undecryptable_credentials("missing")

        self.assertIsNone(result)
        db.toolconfig.update.assert_not_awaited()

    async def test_clear_undecryptable_credentials_returns_unchanged_when_nothing_broken(self) -> None:
        repo = IndexerRepository()
        raw_config = {"host": "db.example.com", "password": encrypt_secret("valid"), "user": "admin"}

        db = mock.AsyncMock()
        db.toolconfig.find_unique = mock.AsyncMock(return_value=self._fake_prisma_tool(connection_config=raw_config))

        with mock.patch.object(repo, "_get_db", return_value=db):
            result = await repo.clear_tool_undecryptable_credentials("tool-123")

        assert result is not None
        self.assertEqual(result.undecryptable_fields, [])
        db.toolconfig.update.assert_not_awaited()


class ClearUndecryptableCredentialsRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_clear_route_calls_side_effects_and_returns_updated_tool(self) -> None:
        updated = ToolConfig(
            id="tool-abc",
            name="Postgres",
            tool_type=ToolType.POSTGRES,
            connection_config={"host": "db.example.com", "user": "admin", "password": ""},
            undecryptable_fields=[],
        )
        mock_repo = mock.AsyncMock()
        mock_repo.clear_tool_undecryptable_credentials = mock.AsyncMock(return_value=updated)

        with (
            mock.patch("ragtime.indexer.routes.repository", mock_repo),
            mock.patch("ragtime.indexer.routes.recheck_encryption_key_health", new=mock.AsyncMock()) as mock_recheck,
            mock.patch("ragtime.indexer.routes.invalidate_settings_cache") as mock_invalidate,
            mock.patch("ragtime.indexer.routes.notify_tools_changed") as mock_notify,
            mock.patch("ragtime.indexer.routes.rag.initialize", new=mock.AsyncMock()) as mock_rag_init,
        ):
            result = await routes.clear_tool_undecryptable_credentials("tool-abc")

        self.assertEqual(result.id, "tool-abc")
        mock_repo.clear_tool_undecryptable_credentials.assert_awaited_once_with("tool-abc")
        mock_recheck.assert_awaited_once()
        mock_invalidate.assert_called_once()
        mock_notify.assert_called_once()
        mock_rag_init.assert_awaited_once()

    async def test_clear_route_returns_404_when_tool_missing(self) -> None:
        mock_repo = mock.AsyncMock()
        mock_repo.clear_tool_undecryptable_credentials = mock.AsyncMock(return_value=None)

        with mock.patch("ragtime.indexer.routes.repository", mock_repo):
            with self.assertRaises(Exception) as ctx:
                await routes.clear_tool_undecryptable_credentials("missing")

        self.assertEqual(getattr(ctx.exception, "status_code", None), 404)


if __name__ == "__main__":
    unittest.main()
