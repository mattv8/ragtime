"""Contracts for safe LDAP failure classification."""

import socket

import pytest
from ldap3.core.exceptions import (
    LDAPConfigurationError,
    LDAPInvalidCredentialsResult,
    LDAPInvalidDnError,
    LDAPInvalidFilterError,
    LDAPInvalidTlsSpecificationError,
    LDAPNoSuchObjectResult,
    LDAPOperationResult,
    LDAPResponseTimeoutError,
    LDAPSocketOpenError,
    LDAPStartTLSError,
)

from ragtime.core.ldap_errors import (
    AuthFailureCode,
    auth_failure_message,
    classify_directory_failure,
    classify_user_bind_failure,
)

AD_DIAGNOSTIC = "80090308: LdapErr: DSID-0C090527, comment: AcceptSecurityContext error, data 532, v4563"


@pytest.mark.parametrize(
    ("subcode", "expected"),
    [
        ("525", AuthFailureCode.INVALID_CREDENTIALS),
        ("52e", AuthFailureCode.INVALID_CREDENTIALS),
        ("532", AuthFailureCode.PASSWORD_EXPIRED),
        ("773", AuthFailureCode.PASSWORD_CHANGE_REQUIRED),
        ("775", AuthFailureCode.ACCOUNT_LOCKED),
        ("533", AuthFailureCode.ACCOUNT_DISABLED),
        ("701", AuthFailureCode.ACCOUNT_EXPIRED),
        ("530", AuthFailureCode.SIGN_IN_RESTRICTED),
        ("531", AuthFailureCode.SIGN_IN_RESTRICTED),
        ("534", AuthFailureCode.SIGN_IN_RESTRICTED),
    ],
)
def test_user_bind_classifies_structured_ad_invalid_credential_diagnostic(subcode, expected):
    """A changed AD subcode branch must not expose the wrong account state."""
    exc = LDAPInvalidCredentialsResult(result=49, message=f"AcceptSecurityContext error, data {subcode}, v4563")

    assert classify_user_bind_failure(exc) is expected


def test_user_bind_classifies_captured_password_expired_diagnostic():
    """Dropping the structured message parser must not turn an expired password generic."""
    exc = LDAPInvalidCredentialsResult(result=49, message=AD_DIAGNOSTIC)

    assert classify_user_bind_failure(exc) is AuthFailureCode.PASSWORD_EXPIRED


def test_user_bind_accepts_uppercase_hex_subcodes():
    """Case-sensitive hexadecimal parsing must not hide a valid AD diagnostic."""
    exc = LDAPInvalidCredentialsResult(result=49, message="AcceptSecurityContext error, data 52E, v4563")

    assert classify_user_bind_failure(exc) is AuthFailureCode.INVALID_CREDENTIALS


@pytest.mark.parametrize(
    "message",
    [
        "AcceptSecurityContext error, data 5320, v4563",
        "AcceptSecurityContext error, data 0x532, v4563",
        "AcceptSecurityContext error, data , v4563",
        "AcceptSecurityContext error, v4563",
        "AcceptSecurityContext error, data 999, v4563",
        "AcceptSecurityContext error, data 532, data 775, v4563",
        "AcceptSecurityContext error, data 532, data malformed, v4563",
        "AcceptSecurityContext error, data 532, data 0x775, v4563",
    ],
)
def test_user_bind_fails_closed_for_malformed_unknown_or_conflicting_diagnostics(message):
    """Loose or conflicting diagnostic parsing must not disclose account state."""
    exc = LDAPInvalidCredentialsResult(result=49, message=message)

    assert classify_user_bind_failure(exc) is AuthFailureCode.INVALID_CREDENTIALS


def test_user_bind_ignores_ad_code_in_unstructured_exception_text():
    """Substring matching arbitrary exception text must not disclose account state."""
    exc = RuntimeError(AD_DIAGNOSTIC)

    assert classify_user_bind_failure(exc) is AuthFailureCode.DIRECTORY_UNAVAILABLE


def test_user_bind_ignores_ad_code_from_non_49_operation_result():
    """Only result 49 may enrich a user-bind error with an AD account state."""
    exc = LDAPOperationResult(result=50, message=AD_DIAGNOSTIC)

    assert classify_user_bind_failure(exc) is AuthFailureCode.DIRECTORY_UNAVAILABLE


@pytest.mark.parametrize(
    "exc",
    [
        LDAPSocketOpenError("connection refused"),
        LDAPStartTLSError("TLS negotiation failed"),
        LDAPResponseTimeoutError("timed out"),
        socket.timeout("timed out"),
        TimeoutError("timed out"),
        OSError("network unreachable"),
    ],
)
def test_transport_failures_are_directory_unavailable(exc):
    """Transport failures must not be presented as invalid credentials."""
    assert classify_user_bind_failure(exc) is AuthFailureCode.DIRECTORY_UNAVAILABLE
    assert classify_directory_failure(exc) is AuthFailureCode.DIRECTORY_UNAVAILABLE


@pytest.mark.parametrize(
    "exc",
    [
        LDAPConfigurationError("bad config"),
        LDAPInvalidDnError("bad DN"),
        LDAPInvalidFilterError("bad filter"),
        LDAPInvalidTlsSpecificationError("bad TLS config"),
    ],
)
def test_explicit_ldap_configuration_errors_are_configuration_failures(exc):
    """Bad protocol configuration must not be retried as a directory outage."""
    assert classify_user_bind_failure(exc) is AuthFailureCode.DIRECTORY_CONFIGURATION_ERROR
    assert classify_directory_failure(exc) is AuthFailureCode.DIRECTORY_CONFIGURATION_ERROR


def test_directory_search_base_result_32_is_a_configuration_failure():
    """A missing search base must not be hidden as a transient directory outage."""
    exc = LDAPNoSuchObjectResult(result=32, message="base DN does not exist")

    assert classify_directory_failure(exc) is AuthFailureCode.DIRECTORY_CONFIGURATION_ERROR


def test_generic_user_invalid_credentials_remain_generic():
    """A result 49 without an AD diagnostic must preserve user enumeration resistance."""
    exc = LDAPInvalidCredentialsResult(result=49, message=None)

    assert classify_user_bind_failure(exc) is AuthFailureCode.INVALID_CREDENTIALS


def test_directory_bind_result_49_is_a_configuration_failure_not_account_state():
    """Service-bind diagnostics must never be attributed to the end user."""
    exc = LDAPInvalidCredentialsResult(result=49, message=AD_DIAGNOSTIC)

    assert classify_directory_failure(exc) is AuthFailureCode.DIRECTORY_CONFIGURATION_ERROR


def test_directory_generic_invalid_credentials_is_a_configuration_failure():
    """A rejected service account must report a configuration problem."""
    exc = LDAPInvalidCredentialsResult(result=49, message=None)

    assert classify_directory_failure(exc) is AuthFailureCode.DIRECTORY_CONFIGURATION_ERROR


@pytest.mark.parametrize("result", [8, 13])
def test_protocol_security_requirements_are_configuration_failures(result):
    """ldap3 protocol result exceptions must not be presented as outages."""
    exc = LDAPOperationResult(result=result, message="directory requires stronger security")

    assert classify_user_bind_failure(exc) is AuthFailureCode.DIRECTORY_CONFIGURATION_ERROR
    assert classify_directory_failure(exc) is AuthFailureCode.DIRECTORY_CONFIGURATION_ERROR


def test_safe_message_catalog_is_complete_and_static():
    """Every public classifier result must resolve to the approved safe message."""
    expected = {
        AuthFailureCode.INVALID_CREDENTIALS: "Invalid username or password.",
        AuthFailureCode.PASSWORD_EXPIRED: "Your organization password has expired. Change your password through your organization's password-management process, then try again. Contact IT if you need help.",
        AuthFailureCode.PASSWORD_CHANGE_REQUIRED: "You must change your organization password before signing in. Complete the password change through your organization's password-management process, then try again.",
        AuthFailureCode.ACCOUNT_LOCKED: "Your organization account is locked. Contact IT to unlock it before trying again.",
        AuthFailureCode.ACCOUNT_DISABLED: "Your organization account is disabled. Contact IT for help.",
        AuthFailureCode.ACCOUNT_EXPIRED: "Your organization account has expired. Contact IT for help.",
        AuthFailureCode.SIGN_IN_RESTRICTED: "Sign-in is blocked by directory policy. Contact IT for help.",
        AuthFailureCode.DIRECTORY_UNAVAILABLE: "The organization's sign-in service is temporarily unavailable. Try again later. Contact IT if the problem continues.",
        AuthFailureCode.DIRECTORY_CONFIGURATION_ERROR: "Organization sign-in is not available because of a server configuration problem. Contact IT for help.",
        AuthFailureCode.NOT_CONFIGURED: "Organization sign-in is not available because of a server configuration problem. Contact IT for help.",
        AuthFailureCode.ACCESS_DENIED: "Your organization account is not authorized to sign in to this application. Contact IT for access.",
        AuthFailureCode.IDENTITY_CONFLICT: "Your organization account could not be matched safely to this application. Contact IT for help.",
        AuthFailureCode.INTERNAL_ERROR: "Sign-in could not be completed because of a server problem. Try again later. Contact IT if the problem continues.",
    }

    assert set(AuthFailureCode) == set(expected)
    assert {code: auth_failure_message(code) for code in AuthFailureCode} == expected
