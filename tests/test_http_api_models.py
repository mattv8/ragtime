import subprocess
import sys
import unittest
from types import SimpleNamespace

from ragtime.http_api.models import HttpApiAuthMode, HttpApiMethodPolicy
from ragtime.indexer.models import ToolConfig, ToolType
from ragtime.indexer.repository import IndexerRepository


class HttpApiModelContractTests(unittest.TestCase):
    def test_http_api_connection_config_supports_nested_auth_fields_and_json_redacts_nested_secrets(self) -> None:
        from ragtime.http_api.models import HttpApiConnectionConfig

        config = HttpApiConnectionConfig.model_validate(
            {
                "base_url": "https://api.example.test",
                "auth_mode": "token_exchange",
                "request_headers": [{"name": "X-Tenant", "value": "tenant-secret"}],
                "token_request_headers": [{"name": "X-Token-Key", "value": "endpoint-secret"}],
                "token_request_fields": [
                    {"name": "grant_type", "value": "client_credentials", "secret": False},
                    {"name": "client_id", "value": "client-id", "secret": False},
                    {"name": "client_secret", "value": "client-secret", "secret": True},
                ],
            }
        )

        dumped = config.model_dump()
        json_dumped = config.model_dump(mode="json")

        self.assertEqual(dumped["auth_mode"], "token_exchange")
        self.assertEqual(dumped["request_headers"][0]["value"], "tenant-secret")
        self.assertEqual(dumped["token_request_headers"][0]["value"], "endpoint-secret")
        self.assertEqual(dumped["token_request_fields"][2]["value"], "client-secret")
        self.assertEqual(json_dumped["request_headers"], [{"name": "X-Tenant", "value": ""}])
        self.assertEqual(json_dumped["token_request_headers"], [{"name": "X-Token-Key", "value": ""}])
        self.assertEqual(
            json_dumped["token_request_fields"],
            [
                {"name": "grant_type", "value": "client_credentials", "secret": False},
                {"name": "client_id", "value": "client-id", "secret": False},
                {"name": "client_secret", "value": "", "secret": True},
            ],
        )

    def test_http_api_connection_config_rejects_duplicate_blocked_and_multiline_nested_headers(self) -> None:
        from pydantic import ValidationError

        from ragtime.http_api.models import HttpApiConnectionConfig

        invalid_cases = [
            {
                "request_headers": [
                    {"name": "X-Tenant", "value": "tenant-a"},
                    {"name": "x-tenant", "value": "tenant-b"},
                ]
            },
            {"request_headers": [{"name": "Host", "value": "forbidden"}]},
            {"token_request_headers": [{"name": "X-Token-Key", "value": "line1\nline2"}]},
        ]

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValidationError):
                    HttpApiConnectionConfig.model_validate(
                        {
                            "base_url": "https://api.example.test",
                            "auth_mode": "headers",
                            **case,
                        }
                    )

    def test_http_api_connection_config_allows_configured_authorization_and_authentication_headers(self) -> None:
        from ragtime.http_api.models import HttpApiConnectionConfig

        config = HttpApiConnectionConfig.model_validate(
            {
                "base_url": "https://api.example.test",
                "auth_mode": "headers",
                "request_headers": [
                    {"name": "Authorization", "value": "Bearer fake-token"},
                    {"name": "Authentication", "value": "Bearer fake-token"},
                ],
            }
        )

        self.assertEqual(
            config.model_dump()["request_headers"],
            [
                {"name": "Authorization", "value": "Bearer fake-token"},
                {"name": "Authentication", "value": "Bearer fake-token"},
            ],
        )

    def test_http_api_connection_config_validates_token_header_name_with_safe_header_policy(self) -> None:
        from pydantic import ValidationError

        from ragtime.http_api.models import HttpApiConnectionConfig

        valid = HttpApiConnectionConfig.model_validate(
            {
                "base_url": "https://api.example.test",
                "auth_mode": "token_exchange",
                "token_header_name": "X-Access-Token",
                "token_request_fields": [
                    {"name": "grant_type", "value": "client_credentials", "secret": False},
                ],
            }
        )
        self.assertEqual(valid.token_header_name, "X-Access-Token")

        for token_header_name in ("Host", "Cookie", "Content-Length", "X-Forwarded-For", "Bad Header", "Bad\r\nHeader"):
            with self.subTest(token_header_name=token_header_name):
                with self.assertRaises(ValidationError):
                    HttpApiConnectionConfig.model_validate(
                        {
                            "base_url": "https://api.example.test",
                            "auth_mode": "token_exchange",
                            "token_header_name": token_header_name,
                            "token_request_fields": [
                                {"name": "grant_type", "value": "client_credentials", "secret": False},
                            ],
                        }
                    )

    def test_http_api_connection_config_rejects_token_prefix_with_crlf(self) -> None:
        from pydantic import ValidationError

        from ragtime.http_api.models import HttpApiConnectionConfig

        with self.assertRaises(ValidationError):
            HttpApiConnectionConfig.model_validate(
                {
                    "base_url": "https://api.example.test",
                    "auth_mode": "token_exchange",
                    "token_prefix": "Bearer\r\nInjected",
                    "token_request_fields": [
                        {"name": "grant_type", "value": "client_credentials", "secret": False},
                    ],
                }
            )

    def test_http_api_connection_config_rejects_non_ascii_header_names(self) -> None:
        from pydantic import ValidationError

        from ragtime.http_api.models import HttpApiConnectionConfig

        with self.assertRaises(ValidationError):
            HttpApiConnectionConfig.model_validate(
                {
                    "base_url": "https://api.example.test",
                    "auth_mode": "headers",
                    "request_headers": [{"name": "X-Āpi-Key", "value": "nope"}],
                }
            )

    def test_core_encryption_imports_in_fresh_interpreter(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import ragtime.core.encryption"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_http_api_connection_config_uses_authoritative_field_names_and_defaults(self) -> None:
        from ragtime.http_api.models import HttpApiConnectionConfig

        config = HttpApiConnectionConfig(base_url="https://api.example.com")
        dumped = config.model_dump()

        self.assertEqual(dumped["auth_mode"], "none")
        self.assertEqual(dumped["api_key_location"], "header")
        self.assertEqual(dumped["login_path"], "")
        self.assertEqual(dumped["login_method"], "POST")
        self.assertEqual(dumped["login_body_format"], "json")
        self.assertEqual(dumped["token_response_path"], "access_token")
        self.assertEqual(dumped["token_header_name"], "Authorization")
        self.assertEqual(dumped["token_prefix"], "Bearer")
        self.assertEqual(dumped["send_api_key_to_requests"], False)
        self.assertEqual(dumped["default_response_selector"], "")
        self.assertEqual(dumped["approved_request_headers"], [])
        self.assertEqual(
            dumped["method_policies"],
            {
                "GET": "read",
                "HEAD": "read",
                "OPTIONS": "read",
                "POST": "disabled",
                "PUT": "disabled",
                "PATCH": "disabled",
                "DELETE": "disabled",
            },
        )

    def test_http_api_connection_config_python_dump_keeps_secret_values(self) -> None:
        from ragtime.http_api.models import HttpApiConnectionConfig

        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.LOGIN_EXCHANGE,
            bearer_token="secret-token",
            api_key="secret-key",
            basic_password="secret-password",
            login_password="login-secret",
        )

        dumped = config.model_dump()

        self.assertEqual(dumped["bearer_token"], "secret-token")
        self.assertEqual(dumped["api_key"], "secret-key")
        self.assertEqual(dumped["basic_password"], "secret-password")
        self.assertEqual(dumped["login_password"], "login-secret")

    def test_http_api_connection_config_json_dump_redacts_secret_values(self) -> None:
        from pydantic import ValidationError

        from ragtime.http_api.models import HttpApiConnectionConfig

        with self.assertRaises(ValidationError):
            HttpApiConnectionConfig.model_validate(
                {
                    "base_url": "https://api.example.com",
                    "unknown_field": "boom",
                }
            )

    def test_http_api_connection_config_json_dump_keeps_direct_secret_values(self) -> None:
        from ragtime.http_api.models import HttpApiConnectionConfig

        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.LOGIN_EXCHANGE,
            bearer_token="secret-token",
            api_key="secret-key",
            basic_password="secret-password",
            login_password="login-secret",
        )

        dumped = config.model_dump(mode="json")

        self.assertEqual(dumped["bearer_token"], "")
        self.assertEqual(dumped["api_key"], "")
        self.assertEqual(dumped["basic_password"], "")
        self.assertEqual(dumped["login_password"], "")

    def test_merge_http_api_secret_updates_preserves_missing_secrets(self) -> None:
        from ragtime.http_api.models import merge_http_api_secret_updates

        merged = merge_http_api_secret_updates(
            {
                "auth_mode": "bearer",
                "bearer_token": "stored-token",
                "api_key": "stored-key",
                "token_prefix": "Bearer",
            },
            {
                "auth_mode": "api_key",
                "send_api_key_to_requests": True,
            },
        )

        self.assertEqual(merged["auth_mode"], "api_key")
        self.assertEqual(merged["bearer_token"], "stored-token")
        self.assertEqual(merged["api_key"], "stored-key")
        self.assertEqual(merged["send_api_key_to_requests"], True)

    def test_merge_http_api_secret_updates_preserves_blank_nested_saved_secrets_by_name_and_removes_omitted_rows(self) -> None:
        from ragtime.http_api.models import merge_http_api_secret_updates

        merged = merge_http_api_secret_updates(
            {
                "auth_mode": "token_exchange",
                "request_headers": [
                    {"name": "X-Tenant", "value": "stored-tenant-secret"},
                    {"name": "X-Remove-Me", "value": "remove-me"},
                ],
                "token_request_fields": [
                    {"name": "grant_type", "value": "client_credentials", "secret": False},
                    {"name": "client_secret", "value": "stored-client-secret", "secret": True},
                ],
            },
            {
                "auth_mode": "token_exchange",
                "request_headers": [{"name": "X-Tenant", "value": ""}],
                "token_request_fields": [
                    {"name": "grant_type", "value": "client_credentials", "secret": False},
                    {"name": "client_secret", "value": "", "secret": True},
                ],
            },
        )

        self.assertEqual(merged["request_headers"], [{"name": "X-Tenant", "value": "stored-tenant-secret"}])
        self.assertEqual(
            merged["token_request_fields"],
            [
                {"name": "grant_type", "value": "client_credentials", "secret": False},
                {"name": "client_secret", "value": "stored-client-secret", "secret": True},
            ],
        )

    def test_merge_http_api_secret_updates_does_not_preserve_secret_when_row_toggles_to_non_secret_blank(self) -> None:
        from pydantic import ValidationError

        from ragtime.core.encryption import ENCRYPTED_PREFIX
        from ragtime.http_api.models import merge_http_api_secret_updates
        from ragtime.http_api.secrets import decrypt_http_api_nested_secrets, encrypt_http_api_nested_secrets

        existing = encrypt_http_api_nested_secrets(
            {
                "auth_mode": "token_exchange",
                "token_request_fields": [
                    {"name": "client_secret", "value": "stored-client-secret", "secret": True},
                ],
            }
        )

        self.assertTrue(existing["token_request_fields"][0]["value"].startswith(ENCRYPTED_PREFIX))
        self.assertEqual(
            decrypt_http_api_nested_secrets(existing)["token_request_fields"][0]["value"],
            "stored-client-secret",
        )

        with self.assertRaises(ValidationError):
            merge_http_api_secret_updates(
                existing,
                {
                    "auth_mode": "token_exchange",
                    "token_request_fields": [
                        {"name": "client_secret", "value": "", "secret": False},
                    ],
                },
            )

    def test_http_api_nested_secret_helpers_encrypt_decrypt_and_report_scoped_paths(self) -> None:
        from ragtime.core.encryption import ENCRYPTED_PREFIX
        from ragtime.http_api.secrets import (
            clear_undecryptable_http_api_nested_secrets,
            configured_http_api_secret_paths,
            decrypt_http_api_nested_secrets,
            encrypt_http_api_nested_secrets,
            iter_http_api_encrypted_secret_values,
            undecryptable_http_api_secret_paths,
        )

        config = {
            "base_url": "https://api.example.test",
            "auth_mode": "token_exchange",
            "request_headers": [{"name": "X-Tenant", "value": "tenant-secret"}],
            "token_request_headers": [{"name": "X-Token-Key", "value": "endpoint-secret"}],
            "token_request_fields": [
                {"name": "grant_type", "value": "client_credentials", "secret": False},
                {"name": "client_secret", "value": "client-secret", "secret": True},
            ],
        }

        encrypted = encrypt_http_api_nested_secrets(config)

        self.assertTrue(encrypted["request_headers"][0]["value"].startswith(ENCRYPTED_PREFIX))
        self.assertTrue(encrypted["token_request_headers"][0]["value"].startswith(ENCRYPTED_PREFIX))
        self.assertEqual(encrypted["token_request_fields"][0]["value"], "client_credentials")
        self.assertTrue(encrypted["token_request_fields"][1]["value"].startswith(ENCRYPTED_PREFIX))
        self.assertEqual(
            configured_http_api_secret_paths(encrypted),
            [
                "request_headers.X-Tenant",
                "token_request_headers.X-Token-Key",
                "token_request_fields.client_secret",
            ],
        )

        decrypted = decrypt_http_api_nested_secrets(encrypted)
        self.assertEqual(decrypted["request_headers"][0]["value"], "tenant-secret")
        self.assertEqual(decrypted["token_request_headers"][0]["value"], "endpoint-secret")
        self.assertEqual(decrypted["token_request_fields"][1]["value"], "client-secret")
        self.assertEqual(
            iter_http_api_encrypted_secret_values(encrypted),
            [
                encrypted["request_headers"][0]["value"],
                encrypted["token_request_headers"][0]["value"],
                encrypted["token_request_fields"][1]["value"],
            ],
        )

        broken = dict(encrypted)
        broken["token_request_fields"] = [{"name": "client_secret", "value": f"{ENCRYPTED_PREFIX}broken", "secret": True}]
        self.assertEqual(undecryptable_http_api_secret_paths(broken), ["token_request_fields.client_secret"])
        self.assertEqual(iter_http_api_encrypted_secret_values(decrypted), [])
        cleared, cleared_paths = clear_undecryptable_http_api_nested_secrets(broken)
        self.assertEqual(cleared_paths, ["token_request_fields.client_secret"])
        self.assertEqual(cleared["token_request_fields"], [])

    def test_method_policies_reject_invalid_keys_and_values(self) -> None:
        from pydantic import ValidationError

        from ragtime.http_api.models import HttpApiConnectionConfig

        with self.assertRaises(ValidationError):
            HttpApiConnectionConfig.model_validate(
                {
                    "base_url": "https://api.example.com",
                    "method_policies": {"TRACE": "read"},
                }
            )

        with self.assertRaises(ValidationError):
            HttpApiConnectionConfig.model_validate(
                {
                    "base_url": "https://api.example.com",
                    "method_policies": {"GET": "invalid"},
                }
            )

    def test_raw_openapi_content_is_omitted_from_persisted_dump(self) -> None:
        from ragtime.http_api.models import HttpApiConnectionConfig

        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            openapi_source_name="uploaded.yaml",
            openapi_source_hash="abc123",
            openapi_source_url="https://api.example.com/openapi.json",
            raw_openapi_document="openapi: 3.1.0",
        )

        dumped = config.model_dump()

        self.assertNotIn("raw_openapi_document", dumped)
        self.assertEqual(dumped["openapi_source_name"], "uploaded.yaml")
        self.assertEqual(dumped["openapi_source_hash"], "abc123")
        self.assertEqual(dumped["openapi_source_url"], "https://api.example.com/openapi.json")

    def test_validation_result_uses_test_connection_shape(self) -> None:
        from ragtime.http_api.models import HttpApiValidationResult

        result = HttpApiValidationResult(
            success=True,
            message="Configuration is valid - no live request was sent.",
            details={"auth_mode": "none"},
        )

        dumped = result.model_dump()

        self.assertEqual(
            dumped,
            {
                "success": True,
                "message": "Configuration is valid - no live request was sent.",
                "details": {"auth_mode": "none"},
            },
        )

    def test_merge_http_api_secret_updates_clears_explicit_empty_secrets(self) -> None:
        from ragtime.http_api.models import merge_http_api_secret_updates

        merged = merge_http_api_secret_updates(
            {
                "bearer_token": "stored-token",
                "basic_password": "stored-password",
            },
            {
                "bearer_token": "",
                "basic_password": "",
            },
        )

        self.assertEqual(merged["bearer_token"], "")
        self.assertEqual(merged["basic_password"], "")

    def test_tool_config_json_dump_redacts_http_api_secrets(self) -> None:
        tool = ToolConfig(
            name="HTTP API",
            tool_type=ToolType.HTTP_API,
            connection_config={
                "base_url": "https://api.example.com",
                "api_key": "secret-key",
                "bearer_token": "secret-token",
                "basic_password": "secret-password",
                "login_password": "login-secret",
                "raw_openapi_document": "openapi: 3.1.0",
            },
            configured_secret_fields=["api_key", "bearer_token", "basic_password", "login_password"],
        )

        dumped = tool.model_dump(mode="json")

        self.assertNotIn("api_key", dumped["connection_config"])
        self.assertNotIn("bearer_token", dumped["connection_config"])
        self.assertNotIn("basic_password", dumped["connection_config"])
        self.assertNotIn("login_password", dumped["connection_config"])
        self.assertNotIn("raw_openapi_document", dumped["connection_config"])
        self.assertEqual(
            dumped["configured_secret_fields"],
            ["api_key", "bearer_token", "basic_password", "login_password"],
        )

    def test_tool_config_json_dump_redacts_nested_http_api_secrets_but_keeps_scoped_names(self) -> None:
        tool = ToolConfig(
            name="HTTP API",
            tool_type=ToolType.HTTP_API,
            connection_config={
                "base_url": "https://api.example.test",
                "auth_mode": "token_exchange",
                "request_headers": [{"name": "X-Tenant", "value": "tenant-secret"}],
                "token_request_headers": [{"name": "X-Token-Key", "value": "endpoint-secret"}],
                "token_request_fields": [
                    {"name": "grant_type", "value": "client_credentials", "secret": False},
                    {"name": "client_secret", "value": "client-secret", "secret": True},
                ],
            },
            configured_secret_fields=[
                "request_headers.X-Tenant",
                "token_request_headers.X-Token-Key",
                "token_request_fields.client_secret",
            ],
        )

        dumped = tool.model_dump(mode="json")

        self.assertEqual(dumped["connection_config"]["request_headers"], [{"name": "X-Tenant", "value": ""}])
        self.assertEqual(dumped["connection_config"]["token_request_headers"], [{"name": "X-Token-Key", "value": ""}])
        self.assertEqual(
            dumped["connection_config"]["token_request_fields"],
            [
                {"name": "grant_type", "value": "client_credentials", "secret": False},
                {"name": "client_secret", "value": "", "secret": True},
            ],
        )
        self.assertEqual(
            dumped["configured_secret_fields"],
            [
                "request_headers.X-Tenant",
                "token_request_headers.X-Token-Key",
                "token_request_fields.client_secret",
            ],
        )

    def test_tool_config_python_dump_keeps_http_api_secrets(self) -> None:
        tool = ToolConfig(
            name="HTTP API",
            tool_type=ToolType.HTTP_API,
            connection_config={
                "base_url": "https://api.example.com",
                "api_key": "secret-key",
                "bearer_token": "secret-token",
                "raw_openapi_document": "openapi: 3.1.0",
            },
        )

        dumped = tool.model_dump()

        self.assertEqual(dumped["connection_config"]["api_key"], "secret-key")
        self.assertEqual(dumped["connection_config"]["bearer_token"], "secret-token")
        self.assertNotIn("raw_openapi_document", dumped["connection_config"])

    def test_http_api_execution_result_defaults_match_component_contract(self) -> None:
        from ragtime.http_api.models import HttpApiExecutionResult

        result = HttpApiExecutionResult(output={"ok": True})
        dumped = result.model_dump()

        self.assertIsNone(dumped["status"])
        self.assertEqual(dumped["output"], {"ok": True})
        self.assertEqual(dumped["rows"], [])
        self.assertEqual(dumped["columns"], [])
        self.assertIsNone(dumped["row_count"])
        self.assertIsNone(dumped["error"])
        self.assertIsNone(dumped["error_kind"])
        self.assertIsNone(dumped["response_bytes"])

    def test_repository_model_marks_configured_http_api_secret_fields(self) -> None:
        from ragtime.core.encryption import encrypt_secret

        repo = IndexerRepository()
        prisma_tool = SimpleNamespace(
            id="tool-http-api",
            name="HTTP API",
            toolType="http_api",
            enabled=True,
            description="",
            connectionConfig={
                "base_url": "https://api.example.com",
                "api_key": encrypt_secret("secret-key"),
                "bearer_token": encrypt_secret("secret-token"),
                "basic_password": "",
            },
            maxResults=100,
            timeoutMaxSeconds=300,
            allowWrite=False,
            sortOrder=0,
            groupId=None,
            group=None,
            lastTestAt=None,
            lastTestResult=None,
            lastTestError=None,
            createdAt=None,
            updatedAt=None,
        )

        model = repo._prisma_tool_config_to_model(prisma_tool)

        self.assertEqual(model.connection_config["api_key"], "secret-key")
        self.assertEqual(model.connection_config["bearer_token"], "secret-token")
        self.assertEqual(model.configured_secret_fields, ["api_key", "bearer_token"])

    def test_repository_model_marks_scoped_nested_http_api_secret_fields(self) -> None:
        from ragtime.core.encryption import encrypt_secret

        repo = IndexerRepository()
        prisma_tool = SimpleNamespace(
            id="tool-http-api",
            name="HTTP API",
            toolType="http_api",
            enabled=True,
            description="",
            connectionConfig={
                "base_url": "https://api.example.test",
                "auth_mode": "token_exchange",
                "request_headers": [{"name": "X-Tenant", "value": encrypt_secret("tenant-secret")}],
                "token_request_headers": [{"name": "X-Token-Key", "value": encrypt_secret("endpoint-secret")}],
                "token_request_fields": [
                    {"name": "grant_type", "value": "client_credentials", "secret": False},
                    {"name": "client_secret", "value": encrypt_secret("client-secret"), "secret": True},
                ],
            },
            maxResults=100,
            timeoutMaxSeconds=300,
            allowWrite=False,
            sortOrder=0,
            groupId=None,
            group=None,
            lastTestAt=None,
            lastTestResult=None,
            lastTestError=None,
            createdAt=None,
            updatedAt=None,
        )

        model = repo._prisma_tool_config_to_model(prisma_tool)

        self.assertEqual(model.connection_config["request_headers"][0]["value"], "tenant-secret")
        self.assertEqual(model.connection_config["token_request_headers"][0]["value"], "endpoint-secret")
        self.assertEqual(model.connection_config["token_request_fields"][1]["value"], "client-secret")
        self.assertEqual(
            model.configured_secret_fields,
            [
                "request_headers.X-Tenant",
                "token_request_headers.X-Token-Key",
                "token_request_fields.client_secret",
            ],
        )


if __name__ == "__main__":
    unittest.main()
