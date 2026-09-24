import asyncio
import logging
import ssl
from contextlib import ExitStack
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from ldap3.core.exceptions import LDAPInvalidCredentialsResult, LDAPSocketOpenError

from ragtime.core import auth
from ragtime.core.ldap_errors import AuthFailureCode, auth_failure_message


class _BoundConnection:
    def __init__(self, *_args, **_kwargs):
        self.bound = False
        self.closed = False
        self.unbound = False
        self.result: dict[str, object] = {}

    def open(self, **_kwargs):
        # ldap3 SyncStrategy.open() returns None on a successful open.
        return None

    def bind(self, **_kwargs):
        self.bound = True
        return True

    def unbind(self):
        self.unbound = True


def _ldap_config(**overrides):
    values = {
        "serverUrl": "ldap://directory",
        "bindDn": "CN=service,DC=example",
        "bindPassword": "service-password",
        "allowSelfSigned": False,
        "userSearchBase": "DC=example",
        "baseDn": "",
        "userSearchFilter": "(uid={username})",
        "adminGroupDns": [],
        "userGroupDns": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _service_connection():
    return SimpleNamespace(bound=True, entries=[], result={}, unbind=lambda: None)


def test_user_bind_is_explicit_once_and_releases_connection():
    connection = _BoundConnection()
    with patch.object(auth, "Connection", return_value=connection) as connection_factory:
        assert auth._bind_ldap_user("ldap://directory", "CN=Chris,DC=example", "password") is None

    assert connection_factory.call_count == 1
    assert connection.unbound is True
    assert connection_factory.call_args.kwargs["auto_bind"] == auth.AUTO_BIND_NONE


@pytest.mark.parametrize("server_url", ["ldap://directory", "ldaps://directory"])
def test_service_connection_uses_no_starttls_autobind_for_both_schemes(server_url):
    connection = _BoundConnection()
    with patch.object(auth, "Connection", return_value=connection) as connection_factory:
        assert auth._get_ldap_connection(server_url, "CN=service", "password", max_retries=1) is connection

    assert connection_factory.call_args.kwargs["auto_bind"] == auth.AUTO_BIND_NO_TLS


@pytest.mark.parametrize(("server_url", "expected_ssl"), [("ldap://directory", False), ("ldaps://directory", True)])
def test_service_and_user_bind_share_tls_server_mode(server_url, expected_ssl):
    service_connection = _BoundConnection()
    user_connection = _BoundConnection()
    with patch.object(auth, "Server", wraps=auth.Server) as server_factory, patch.object(auth, "Connection", side_effect=[service_connection, user_connection]):
        assert auth._get_ldap_connection(server_url, "CN=service", "password", max_retries=1) is service_connection
        assert auth._bind_ldap_user(server_url, "CN=user", "password") is None

    assert [call.kwargs["use_ssl"] for call in server_factory.call_args_list] == [expected_ssl, expected_ssl]


def test_ldaps_server_verifies_certificates_unless_self_signed_is_allowed():
    verified_server = auth._ldap_server("ldaps://directory", False, 5)
    self_signed_server = auth._ldap_server("ldaps://directory", True, 5)

    assert verified_server.tls.validate == ssl.CERT_REQUIRED
    assert self_signed_server.tls.validate == ssl.CERT_NONE


def test_user_ldaps_bind_does_not_start_tls_or_read_server_info():
    connection = _BoundConnection()
    start_tls = Mock()
    setattr(connection, "start_tls", start_tls)
    with patch.object(auth, "Connection", return_value=connection):
        assert auth._bind_ldap_user("ldaps://directory", "CN=Chris,DC=example", "password") is None

    start_tls.assert_not_called()


def test_user_bind_accepts_ldap3_successful_open_returning_none():
    connection = _BoundConnection()
    with patch.object(auth, "Connection", return_value=connection):
        assert auth._bind_ldap_user("ldap://directory", "CN=Chris,DC=example", "password") is None

    assert connection.bound is True


def test_empty_password_does_not_open_a_user_connection():
    with patch.object(auth, "Connection") as connection_factory:
        assert auth._bind_ldap_user("ldap://directory", "CN=Chris,DC=example", "") == AuthFailureCode.INVALID_CREDENTIALS
    connection_factory.assert_not_called()


def test_user_bind_preserves_ad_password_expired_diagnostic():
    class FailingConnection(_BoundConnection):
        def bind(self, **_kwargs):
            raise LDAPInvalidCredentialsResult(
                result=49,
                description="invalidCredentials",
                dn="",
                message="80090308: LdapErr: DSID-0C090527, data 532, v4563",
                response_type="bindResponse",
            )

    connection = FailingConnection()
    with patch.object(auth, "Connection", return_value=connection):
        failure = auth._bind_ldap_user("ldap://directory", "CN=Chris,DC=example", "password")

    assert failure == AuthFailureCode.PASSWORD_EXPIRED
    assert connection.unbound is True


@pytest.mark.parametrize(
    ("exc", "expected_type", "expected_result"),
    [
        (
            LDAPInvalidCredentialsResult(
                result=49,
                description="invalidCredentials",
                dn="CN=secret,DC=example",
                message="secret LDAP diagnostic",
                response_type="bindResponse",
            ),
            "LDAPInvalidCredentialsResult",
            "49",
        ),
        (RuntimeError("secret runtime diagnostic"), "RuntimeError", "None"),
        (TypeError("secret type diagnostic"), "TypeError", "None"),
    ],
)
def test_user_bind_exception_logging_is_safe_and_preserves_metadata(caplog, exc, expected_type, expected_result):
    class FailingConnection(_BoundConnection):
        def open(self, **_kwargs):
            raise exc

    caplog.set_level(logging.WARNING, logger=auth.logger.name)
    with patch.object(auth, "Connection", return_value=FailingConnection()):
        auth._bind_ldap_user("ldap://directory", "CN=Chris,DC=example", "password")

    messages = [record.getMessage() for record in caplog.records if "phase=user_bind" in record.getMessage()]
    assert messages
    assert any(f"exception_type={expected_type}" in message and f"ldap_result={expected_result}" in message for message in messages)
    assert all("secret" not in message for message in messages)


def test_rejected_user_bind_never_syncs_a_profile():
    config = SimpleNamespace(
        serverUrl="ldap://directory",
        bindDn="CN=service,DC=example",
        bindPassword="service-password",
        allowSelfSigned=False,
        userSearchBase="DC=example",
        baseDn="",
        userSearchFilter="(uid={username})",
        adminGroupDns=[],
        userGroupDns=[],
    )
    service_connection = SimpleNamespace(
        entries=[SimpleNamespace(entry_dn="CN=Chris,DC=example", memberOf=[])],
        bound=True,
        unbind=lambda: None,
    )

    async def authenticate_rejected_user():
        with (
            patch.object(auth, "get_ldap_config", AsyncMock(return_value=config)),
            patch.object(auth, "decrypt_secret", return_value="service-password"),
            patch.object(auth, "_get_ldap_connection", return_value=service_connection),
            patch.object(auth, "_search_first_matching_entry", return_value=service_connection.entries[0]),
            patch.object(auth, "_bind_ldap_user", return_value=AuthFailureCode.PASSWORD_EXPIRED),
            patch.object(auth, "_upsert_provider_user_profile", AsyncMock()) as sync,
        ):
            return await auth.authenticate_ldap("chris", "password"), sync

    result, sync = asyncio.run(authenticate_rejected_user())

    assert result.failure_code == AuthFailureCode.PASSWORD_EXPIRED
    assert result.error == auth_failure_message(AuthFailureCode.PASSWORD_EXPIRED)
    sync.assert_not_awaited()


def test_directory_user_bind_failure_is_not_reported_as_credentials():
    config = SimpleNamespace(
        serverUrl="ldap://directory",
        bindDn="CN=service,DC=example",
        bindPassword="service-password",
        allowSelfSigned=False,
        userSearchBase="DC=example",
        baseDn="",
        userSearchFilter="(uid={username})",
        adminGroupDns=[],
        userGroupDns=[],
    )
    service_connection = SimpleNamespace(
        entries=[SimpleNamespace(entry_dn="CN=Chris,DC=example", memberOf=[])],
        bound=True,
        unbind=lambda: None,
    )

    async def authenticate_unavailable_directory():
        with (
            patch.object(auth, "get_ldap_config", AsyncMock(return_value=config)),
            patch.object(auth, "decrypt_secret", return_value="service-password"),
            patch.object(auth, "_get_ldap_connection", return_value=service_connection),
            patch.object(auth, "_search_first_matching_entry", return_value=service_connection.entries[0]),
            patch.object(auth, "_bind_ldap_user", return_value=AuthFailureCode.DIRECTORY_UNAVAILABLE),
        ):
            return await auth.authenticate_ldap("chris", "password")

    result = asyncio.run(authenticate_unavailable_directory())

    assert result.failure_code == AuthFailureCode.DIRECTORY_UNAVAILABLE
    assert result.error == auth_failure_message(AuthFailureCode.DIRECTORY_UNAVAILABLE)


def test_socket_failure_is_classified_as_directory_failure_not_credentials():
    class FailingConnection(_BoundConnection):
        def open(self, **_kwargs):
            raise LDAPSocketOpenError("connection refused secret-password")

    socket_error = LDAPSocketOpenError("connection refused secret-password")
    connection = FailingConnection()
    with (
        patch.object(auth, "Connection", return_value=connection),
        patch.object(auth, "classify_user_bind_failure", wraps=auth.classify_user_bind_failure) as classifier,
    ):
        failure = auth._bind_ldap_user("ldap://directory", "CN=Chris,DC=example", "password")

    assert failure == AuthFailureCode.DIRECTORY_UNAVAILABLE
    assert isinstance(classifier.call_args.args[0], LDAPSocketOpenError)
    assert connection.unbound is True


def test_closed_user_connection_after_open_is_unavailable_and_released():
    connection = _BoundConnection()
    connection.closed = True
    with patch.object(auth, "Connection", return_value=connection):
        assert auth._bind_ldap_user("ldap://directory", "CN=Chris,DC=example", "password") is AuthFailureCode.DIRECTORY_UNAVAILABLE

    assert connection.unbound is True


def test_login_lookup_uses_later_match_after_earlier_operational_error():
    connection = SimpleNamespace(entries=[])
    matched_entry = SimpleNamespace(entry_dn="CN=Chris,DC=example")

    def search(*_args, **kwargs):
        if kwargs["search_filter"] == "(first=chris)":
            raise LDAPSocketOpenError("temporary lookup error")
        connection.entries = [matched_entry]
        return True

    with patch.object(auth, "_search_with_attribute_fallback", side_effect=search):
        result = auth._search_first_matching_entry(
            cast(auth.Connection, connection),
            "DC=example",
            ["(first=chris)", "(uid=chris)"],
            ["*"],
            "test",
            strict=True,
        )

    assert result is matched_entry


def test_login_lookup_raises_retained_error_only_when_no_filter_matches():
    connection = SimpleNamespace(entries=[])
    with patch.object(auth, "_search_with_attribute_fallback", side_effect=LDAPSocketOpenError("directory unavailable")):
        with pytest.raises(LDAPSocketOpenError):
            auth._search_first_matching_entry(cast(auth.Connection, connection), "DC=example", ["(uid=chris)"], ["*"], "test", strict=True)

    with patch.object(auth, "_search_with_attribute_fallback", return_value=False):
        assert auth._search_first_matching_entry(cast(auth.Connection, connection), "DC=example", ["(uid=chris)"], ["*"], "test", strict=True) is None


def test_false_search_result_with_operation_error_is_not_treated_as_unknown_user():
    connection = SimpleNamespace(entries=[], result={"result": 52})
    with patch.object(auth, "_search_with_attribute_fallback", return_value=False):
        with pytest.raises(auth.LDAPException):
            auth._search_first_matching_entry(cast(auth.Connection, connection), "DC=example", ["(uid=chris)"], ["*"], "test", strict=True)


def test_false_user_bind_uses_structured_transient_result_not_credentials(caplog):
    connection = _BoundConnection()
    connection.result = {"result": 52, "description": "unavailable", "message": "secret-password"}
    connection.bind = Mock(return_value=False)
    caplog.set_level(logging.WARNING, logger=auth.logger.name)
    with patch.object(auth, "Connection", return_value=connection):
        failure = auth._bind_ldap_user("ldap://directory", "CN=Chris,DC=example", "password")

    assert failure is AuthFailureCode.DIRECTORY_UNAVAILABLE
    messages = [record.getMessage() for record in caplog.records if "phase=user_bind" in record.getMessage()]
    assert any("exception_type=LDAPUnavailableResult" in message and "ldap_result=52" in message for message in messages)
    assert all("secret-password" not in message for message in messages)


def test_false_user_bind_result_49_remains_generic_credentials_failure():
    connection = _BoundConnection()
    connection.result = {"result": 49, "description": "invalidCredentials", "message": "not an AD diagnostic"}
    connection.bind = Mock(return_value=False)
    with patch.object(auth, "Connection", return_value=connection):
        failure = auth._bind_ldap_user("ldap://directory", "CN=Chris,DC=example", "password")

    assert failure is AuthFailureCode.INVALID_CREDENTIALS


def test_strict_group_lookup_treats_nonraising_operation_failure_as_unavailable():
    connection = SimpleNamespace(bound=True, entries=[], result={"result": 3, "message": "secret DN"}, search=Mock(return_value=False), unbind=Mock())
    config = _ldap_config(userGroupDns=["CN=Allowed,DC=example"])
    entry = SimpleNamespace(memberOf=[], primaryGroupID="513")
    with patch.object(auth, "_get_ldap_connection", return_value=connection):
        with pytest.raises(auth.LDAPOperationResult) as caught:
            auth._determine_ldap_role_for_entry(
                ldap_config=config,
                bind_password="service-password",
                user_entry=entry,
                ldap_username="chris",
                strict=True,
            )

    assert caught.value.result == 3


def test_legacy_group_lookup_keeps_nonraising_failure_as_no_membership():
    connection = SimpleNamespace(bound=True, entries=[], result={"result": 3}, search=Mock(return_value=False), unbind=Mock())
    with patch.object(auth, "_get_ldap_connection", return_value=connection):
        assert auth._ldap_group_rid(_ldap_config(), "service-password", "CN=Allowed,DC=example") is None


def test_strict_primary_group_lookup_allows_later_verified_alternative_group():
    config = _ldap_config(userGroupDns=["CN=First,DC=example", "CN=Allowed,DC=example"])
    entry = SimpleNamespace(memberOf=["CN=Allowed,DC=example"], primaryGroupID="513")
    with patch.object(auth, "_ldap_group_rid", side_effect=LDAPSocketOpenError("first group unavailable")):
        assert (
            auth._determine_ldap_role_for_entry(
                ldap_config=config,
                bind_password="service-password",
                user_entry=entry,
                ldap_username="chris",
                strict=True,
            )
            == auth.UserRole.user
        )


def test_strict_primary_group_lookup_failure_is_operational_but_no_matching_rid_is_denial():
    config = _ldap_config(userGroupDns=["CN=Allowed,DC=example"])
    entry = SimpleNamespace(memberOf=[], primaryGroupID="513")
    with patch.object(auth, "_ldap_group_rid", side_effect=LDAPSocketOpenError("directory unavailable")):
        with pytest.raises(LDAPSocketOpenError):
            auth._determine_ldap_role_for_entry(
                ldap_config=config,
                bind_password="service-password",
                user_entry=entry,
                ldap_username="chris",
                strict=True,
            )
    with patch.object(auth, "_ldap_group_rid", return_value=512):
        with pytest.raises(ValueError):
            auth._determine_ldap_role_for_entry(
                ldap_config=config,
                bind_password="service-password",
                user_entry=entry,
                ldap_username="chris",
                strict=True,
            )


def test_missing_connection_configuration_and_search_base_have_distinct_codes():
    async def authenticate_with(config):
        with patch.object(auth, "get_ldap_config", AsyncMock(return_value=config)):
            return await auth.authenticate_ldap("chris", "password")

    not_configured = asyncio.run(authenticate_with(_ldap_config(serverUrl="", bindDn="")))
    assert not_configured.failure_code == AuthFailureCode.NOT_CONFIGURED

    async def authenticate_without_base():
        with (
            patch.object(auth, "get_ldap_config", AsyncMock(return_value=_ldap_config(userSearchBase="", baseDn=""))),
            patch.object(auth, "decrypt_secret", return_value="service-password"),
            patch.object(auth, "_get_ldap_connection", return_value=_service_connection()),
        ):
            return await auth.authenticate_ldap("chris", "password")

    missing_base = asyncio.run(authenticate_without_base())
    assert missing_base.failure_code == AuthFailureCode.DIRECTORY_CONFIGURATION_ERROR


def test_unknown_user_and_wrong_password_have_the_same_public_failure():
    entry = SimpleNamespace(entry_dn="CN=Chris,DC=example", memberOf=[])

    async def authenticate_with(search_result, bind_failure):
        with (
            patch.object(auth, "get_ldap_config", AsyncMock(return_value=_ldap_config())),
            patch.object(auth, "decrypt_secret", return_value="service-password"),
            patch.object(auth, "_get_ldap_connection", return_value=_service_connection()),
            patch.object(auth, "_search_first_matching_entry", return_value=search_result),
            patch.object(auth, "_bind_ldap_user", return_value=bind_failure),
        ):
            return await auth.authenticate_ldap("chris", "wrong-password")

    unknown = asyncio.run(authenticate_with(None, None))
    wrong_password = asyncio.run(authenticate_with(entry, AuthFailureCode.INVALID_CREDENTIALS))
    assert unknown.failure_code == wrong_password.failure_code == AuthFailureCode.INVALID_CREDENTIALS
    assert unknown.error == wrong_password.error


def test_successful_ldap_authentication_syncs_after_verified_bind():
    entry = SimpleNamespace(entry_dn="CN=Chris,DC=example", memberOf=[], uid="chris")
    user = SimpleNamespace(id="user-id", username="chris", displayName="Chris", email="chris@example.com", role="user")

    async def authenticate_successfully():
        with (
            patch.object(auth, "get_ldap_config", AsyncMock(return_value=_ldap_config())),
            patch.object(auth, "decrypt_secret", return_value="service-password"),
            patch.object(auth, "_get_ldap_connection", return_value=_service_connection()),
            patch.object(auth, "_search_first_matching_entry", return_value=entry),
            patch.object(auth, "_bind_ldap_user", return_value=None),
            patch.object(auth, "get_auth_provider_config", AsyncMock(return_value=SimpleNamespace(ldap_lazy_sync_enabled=True))),
            patch.object(auth, "_upsert_provider_user_profile", AsyncMock(return_value=user)) as sync,
        ):
            return await auth.authenticate_ldap("chris", "password"), sync

    result, sync = asyncio.run(authenticate_successfully())
    assert result.success is True
    sync.assert_awaited_once()


def test_sync_exception_is_a_safe_internal_failure():
    entry = SimpleNamespace(entry_dn="CN=Chris,DC=example", memberOf=[], uid="chris")

    async def authenticate_with_sync_error():
        with (
            patch.object(auth, "get_ldap_config", AsyncMock(return_value=_ldap_config())),
            patch.object(auth, "decrypt_secret", return_value="service-password"),
            patch.object(auth, "_get_ldap_connection", return_value=_service_connection()),
            patch.object(auth, "_search_first_matching_entry", return_value=entry),
            patch.object(auth, "_bind_ldap_user", return_value=None),
            patch.object(auth, "get_auth_provider_config", AsyncMock(return_value=SimpleNamespace(ldap_lazy_sync_enabled=True))),
            patch.object(auth, "_upsert_provider_user_profile", AsyncMock(side_effect=RuntimeError("database unavailable"))),
        ):
            return await auth.authenticate_ldap("chris", "password")

    result = asyncio.run(authenticate_with_sync_error())
    assert result.failure_code == AuthFailureCode.INTERNAL_ERROR
    assert result.error == auth_failure_message(AuthFailureCode.INTERNAL_ERROR)


@pytest.mark.parametrize("operation", ["lookup", "group"])
def test_lookup_and_group_exceptions_return_safe_catalog_errors(operation):
    entry = SimpleNamespace(entry_dn="CN=Chris,DC=example", memberOf=[], uid="chris")
    secret = "ldap://directory/CN=secret,DC=example"

    async def run():
        patches = [
            patch.object(auth, "get_ldap_config", AsyncMock(return_value=_ldap_config())),
            patch.object(auth, "decrypt_secret", return_value="service-password"),
            patch.object(auth, "_get_ldap_connection", return_value=_service_connection()),
        ]
        if operation == "lookup":
            patches.append(patch.object(auth, "_search_first_matching_entry", side_effect=LDAPSocketOpenError(secret)))
        else:
            patches.extend(
                [
                    patch.object(auth, "_search_first_matching_entry", return_value=entry),
                    patch.object(auth, "_bind_ldap_user", return_value=None),
                    patch.object(auth, "_determine_ldap_role_for_entry", side_effect=LDAPSocketOpenError(secret)),
                ]
            )
        with ExitStack() as stack:
            for active_patch in patches:
                stack.enter_context(active_patch)
            warning = stack.enter_context(patch.object(auth.logger, "warning"))
            result = await auth.authenticate_ldap("chris", "password")
        return result, warning

    result, warning = asyncio.run(run())
    assert result.failure_code is AuthFailureCode.DIRECTORY_UNAVAILABLE
    assert result.error == auth_failure_message(AuthFailureCode.DIRECTORY_UNAVAILABLE)
    logged = " ".join(str(call.args) for call in warning.call_args_list)
    assert secret not in logged
    assert "directory" in logged
    assert "LDAPSocketOpenError" in logged


def test_authenticate_skips_ldap_when_not_configured():
    local_failure = auth.AuthResult(success=False, error="local failed")

    async def run():
        with (
            patch.object(auth, "authenticate_local_managed", AsyncMock(return_value=local_failure)),
            patch.object(auth, "get_ldap_config", AsyncMock(return_value=SimpleNamespace(serverUrl=""))),
            patch.object(auth, "authenticate_ldap", AsyncMock()) as ldap_authenticate,
        ):
            result = await auth.authenticate("ordinary-user", "password")
        return result, ldap_authenticate

    result, ldap_authenticate = asyncio.run(run())
    assert result.success is False
    ldap_authenticate.assert_not_awaited()


def test_authenticate_does_not_fall_through_after_configured_ldap_failure():
    ldap_failure = auth.AuthResult(
        success=False,
        error=auth_failure_message(AuthFailureCode.ACCOUNT_LOCKED),
        failure_code=AuthFailureCode.ACCOUNT_LOCKED,
    )

    async def run():
        with (
            patch.object(auth, "authenticate_local_managed", AsyncMock(return_value=auth.AuthResult(success=False))),
            patch.object(auth, "get_ldap_config", AsyncMock(return_value=SimpleNamespace(serverUrl="ldap://directory"))),
            patch.object(auth, "authenticate_ldap", AsyncMock(return_value=ldap_failure)),
            patch.object(auth, "authenticate_local", AsyncMock()) as local_authenticate,
        ):
            result = await auth.authenticate("ordinary-user", "password")
        return result, local_authenticate

    result, local_authenticate = asyncio.run(run())
    assert result is ldap_failure
    local_authenticate.assert_not_awaited()


def test_authenticate_preserves_configured_local_admin_fallback():
    local_failure = auth.AuthResult(success=False)
    local_success = auth.AuthResult(success=True, username="admin")
    ldap_failure = auth.AuthResult(success=False, failure_code=AuthFailureCode.DIRECTORY_UNAVAILABLE)

    async def run():
        with (
            patch.object(auth.settings, "local_admin_user", "admin"),
            patch.object(auth.settings, "local_admin_password", "configured"),
            patch.object(auth, "authenticate_local", AsyncMock(side_effect=[local_failure, local_success])) as local_authenticate,
            patch.object(auth, "authenticate_local_managed", AsyncMock(return_value=local_failure)),
            patch.object(auth, "get_ldap_config", AsyncMock(return_value=SimpleNamespace(serverUrl="ldap://directory"))),
            patch.object(auth, "authenticate_ldap", AsyncMock(return_value=ldap_failure)),
        ):
            result = await auth.authenticate("admin", "password")
        return result, local_authenticate

    result, local_authenticate = asyncio.run(run())
    assert result is local_success
    assert local_authenticate.await_count == 2


def test_authenticate_returns_successful_ldap_result():
    ldap_success = auth.AuthResult(success=True, username="directory-user")

    async def run():
        with (
            patch.object(auth, "authenticate_local_managed", AsyncMock(return_value=auth.AuthResult(success=False))),
            patch.object(auth, "get_ldap_config", AsyncMock(return_value=SimpleNamespace(serverUrl="ldap://directory"))),
            patch.object(auth, "authenticate_ldap", AsyncMock(return_value=ldap_success)),
        ):
            return await auth.authenticate("directory-user", "password")

    assert asyncio.run(run()) is ldap_success
