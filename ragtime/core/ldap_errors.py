"""Safe, dependency-free classification for LDAP authentication failures."""

from __future__ import annotations

import re
import socket
from enum import Enum

from ldap3.core.exceptions import (
    LDAPCertificateError,
    LDAPCommunicationError,
    LDAPConfigurationError,
    LDAPConfigurationParameterError,
    LDAPInvalidDnError,
    LDAPInvalidFilterError,
    LDAPInvalidPortError,
    LDAPInvalidServerError,
    LDAPInvalidTlsSpecificationError,
    LDAPOperationResult,
    LDAPResponseTimeoutError,
    LDAPSessionTerminatedByServerError,
    LDAPSocketCloseError,
    LDAPSocketOpenError,
    LDAPSocketReceiveError,
    LDAPSocketSendError,
    LDAPSSLConfigurationError,
    LDAPSSLNotSupportedError,
    LDAPStartTLSError,
)


class AuthFailureCode(str, Enum):
    INVALID_CREDENTIALS = "invalid_credentials"
    PASSWORD_EXPIRED = "password_expired"
    PASSWORD_CHANGE_REQUIRED = "password_change_required"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_DISABLED = "account_disabled"
    ACCOUNT_EXPIRED = "account_expired"
    SIGN_IN_RESTRICTED = "sign_in_restricted"
    DIRECTORY_UNAVAILABLE = "directory_unavailable"
    DIRECTORY_CONFIGURATION_ERROR = "directory_configuration_error"
    NOT_CONFIGURED = "not_configured"
    ACCESS_DENIED = "access_denied"
    IDENTITY_CONFLICT = "identity_conflict"
    INTERNAL_ERROR = "internal_error"


_MESSAGES = {
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

_AD_SUBCODES = {
    "525": AuthFailureCode.INVALID_CREDENTIALS,
    "52e": AuthFailureCode.INVALID_CREDENTIALS,
    "532": AuthFailureCode.PASSWORD_EXPIRED,
    "773": AuthFailureCode.PASSWORD_CHANGE_REQUIRED,
    "775": AuthFailureCode.ACCOUNT_LOCKED,
    "533": AuthFailureCode.ACCOUNT_DISABLED,
    "701": AuthFailureCode.ACCOUNT_EXPIRED,
    "530": AuthFailureCode.SIGN_IN_RESTRICTED,
    "531": AuthFailureCode.SIGN_IN_RESTRICTED,
    "534": AuthFailureCode.SIGN_IN_RESTRICTED,
}

_DATA_MARKER = re.compile(r"\bdata\b", re.IGNORECASE)
_HEX_TOKEN = re.compile(r"\s+([0-9a-fA-F]+)\b")

_CONFIGURATION_ERRORS = (
    LDAPConfigurationError,
    LDAPConfigurationParameterError,
    LDAPInvalidDnError,
    LDAPInvalidFilterError,
    LDAPInvalidPortError,
    LDAPInvalidServerError,
    LDAPInvalidTlsSpecificationError,
    LDAPSSLConfigurationError,
    LDAPSSLNotSupportedError,
)

_UNAVAILABLE_ERRORS = (
    LDAPCertificateError,
    LDAPCommunicationError,
    LDAPResponseTimeoutError,
    LDAPSessionTerminatedByServerError,
    LDAPSocketCloseError,
    LDAPSocketOpenError,
    LDAPSocketReceiveError,
    LDAPSocketSendError,
    LDAPStartTLSError,
    socket.timeout,
    TimeoutError,
    OSError,
)


def auth_failure_message(code: AuthFailureCode) -> str:
    """Return the approved public message for a normalized failure code."""
    return _MESSAGES[code]


def _classify_operational_failure(exc: Exception) -> AuthFailureCode:
    if isinstance(exc, _CONFIGURATION_ERRORS):
        return AuthFailureCode.DIRECTORY_CONFIGURATION_ERROR
    if isinstance(exc, _UNAVAILABLE_ERRORS):
        return AuthFailureCode.DIRECTORY_UNAVAILABLE
    return AuthFailureCode.DIRECTORY_UNAVAILABLE


def _ad_subcode(exc: Exception) -> str | None:
    """Return one known, unambiguous AD subcode from a result-49 message."""
    if not isinstance(exc, LDAPOperationResult) or exc.result != 49 or not isinstance(exc.message, str):
        return None

    markers = list(_DATA_MARKER.finditer(exc.message))
    if len(markers) != 1:
        return None
    token = _HEX_TOKEN.match(exc.message, markers[0].end())
    if token is None:
        return None
    subcode = token.group(1).lower()
    return subcode if subcode in _AD_SUBCODES else None


def classify_user_bind_failure(exc: Exception) -> AuthFailureCode:
    """Classify a user credential bind failure without parsing arbitrary text."""
    if isinstance(exc, LDAPOperationResult) and exc.result in {8, 13}:
        return AuthFailureCode.DIRECTORY_CONFIGURATION_ERROR
    subcode = _ad_subcode(exc)
    if subcode is not None:
        return _AD_SUBCODES[subcode]
    if isinstance(exc, LDAPOperationResult) and exc.result == 49:
        return AuthFailureCode.INVALID_CREDENTIALS
    return _classify_operational_failure(exc)


def classify_directory_failure(exc: Exception) -> AuthFailureCode:
    """Classify service, lookup, and group failures without user-state enrichment."""
    if isinstance(exc, LDAPOperationResult) and exc.result in {8, 13, 32, 49}:
        return AuthFailureCode.DIRECTORY_CONFIGURATION_ERROR
    return _classify_operational_failure(exc)
