"""
Reversible encryption utilities for storing secrets.

Uses Fernet symmetric encryption with a key derived from the effective
application encryption key. ENCRYPTION_KEY can explicitly provide that key;
otherwise settings auto-generates one and persists it in the data volume.
This allows secrets to be decrypted for display in the frontend or for
backup/restore operations.

IMPORTANT: backups that must preserve encrypted secrets should use
backup --include-secret. Without the original key, users must re-enter stored
API keys and connection passwords.
"""

import base64
import hashlib
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from ragtime.config.settings import settings
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Prefix to identify encrypted values (helps distinguish from legacy plaintext)
ENCRYPTED_PREFIX = "enc::"

# Sticky flag set the first time an encrypted secret fails to decrypt with the
# active key. This is the recoverable "key changed / key file lost" condition
# (e.g. .data/.encryption_key was deleted). We record it so the platform can
# surface a single, actionable warning instead of silently running with empty
# secrets, and we use it to log the failure only once instead of flooding the
# logs as every encrypted field is loaded in turn.
_key_mismatch_detected = False


def encryption_key_mismatch_detected() -> bool:
    """Return True if an encrypted secret failed to decrypt with the active key.

    Indicates the encryption key changed or its key file was lost. Existing
    encrypted secrets cannot be recovered until the original key is restored.
    """
    return _key_mismatch_detected


def reset_key_mismatch_state() -> None:
    """Reset the key-mismatch flag. Primarily for tests."""
    global _key_mismatch_detected
    _key_mismatch_detected = False


def encryption_recovery_hint() -> str:
    """Return standard, actionable recovery guidance for a key mismatch.

    ENCRYPTION_KEY takes precedence over the persisted key file. The file is a
    fallback and backup artifact so backup --include-secret can carry the
    effective key between deployments.
    """
    return (
        "Set ENCRYPTION_KEY to the original key and restart, or restore from a backup "
        "created with --include-secret. If the key is lost, re-enter API keys and "
        "connection passwords in Settings."
    )


def _mark_key_mismatch() -> None:
    """Record an undecryptable secret and log the cause exactly once."""
    global _key_mismatch_detected
    if not _key_mismatch_detected:
        _key_mismatch_detected = True
        logger.error(
            "Failed to decrypt secret: invalid token (key may have changed). "
            "Encrypted settings are unavailable until the original encryption key "
            "is restored (e.g. .data/.encryption_key). Further decryption failures "
            "will be suppressed until restart."
        )


@lru_cache(maxsize=1)
def _get_fernet() -> Fernet:
    """
    Get a Fernet instance using a key derived from ENCRYPTION_KEY.

    Fernet requires a 32-byte base64-encoded key. We derive this from
    ENCRYPTION_KEY using SHA256 to ensure consistent key length.
    """
    # Use SHA256 to get exactly 32 bytes from ENCRYPTION_KEY
    key_bytes = hashlib.sha256(settings.encryption_key.encode()).digest()
    # Fernet requires base64-encoded key
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    return Fernet(fernet_key)


def encrypt_secret(plaintext: str) -> str:
    """
    Encrypt a plaintext secret.

    Args:
        plaintext: The secret to encrypt

    Returns:
        Encrypted string with 'enc::' prefix
    """
    if not plaintext:
        return ""

    # Already encrypted? Return as-is
    if plaintext.startswith(ENCRYPTED_PREFIX):
        return plaintext

    try:
        fernet = _get_fernet()
        encrypted_bytes = fernet.encrypt(plaintext.encode())
        return ENCRYPTED_PREFIX + encrypted_bytes.decode()
    except Exception as e:
        logger.error(f"Failed to encrypt secret: {e}")
        raise ValueError("Encryption failed") from e


def decrypt_secret(encrypted: str) -> str:
    """
    Decrypt an encrypted secret.

    Args:
        encrypted: The encrypted string (with or without 'enc::' prefix)

    Returns:
        Decrypted plaintext, or empty string if decryption fails

    Note:
        If the value doesn't have the 'enc::' prefix, it's treated as
        legacy plaintext and returned as-is.
    """
    if not encrypted:
        return ""

    # Not encrypted (legacy value)? Return as-is
    if not encrypted.startswith(ENCRYPTED_PREFIX):
        return encrypted

    try:
        fernet = _get_fernet()
        encrypted_data = encrypted[len(ENCRYPTED_PREFIX) :]
        decrypted_bytes = fernet.decrypt(encrypted_data.encode())
        return decrypted_bytes.decode()
    except InvalidToken:
        _mark_key_mismatch()
        return ""
    except Exception as e:
        logger.error(f"Failed to decrypt secret: {e}")
        return ""


def is_encrypted(value: str) -> bool:
    """Check if a value is encrypted (has the enc:: prefix)."""
    return value.startswith(ENCRYPTED_PREFIX) if value else False


def migrate_plaintext_to_encrypted(value: str | None) -> str | None:
    """
    Migrate a potentially plaintext value to encrypted format.

    Args:
        value: Plaintext or already-encrypted value

    Returns:
        Encrypted value, or None if input was None
    """
    if value is None:
        return None

    if not value:
        return ""

    # Already encrypted
    if is_encrypted(value):
        return value

    # Encrypt plaintext
    return encrypt_secret(value)


def encrypt_json_passwords(data: dict, password_fields: list[str]) -> dict:
    """
    Encrypt password fields in a JSON dict (e.g., connection_config).

    Args:
        data: Dictionary potentially containing password fields
        password_fields: List of field names to encrypt

    Returns:
        Copy of dict with specified fields encrypted
    """
    if not data:
        return data

    result = dict(data)
    for field in password_fields:
        if field in result and result[field]:
            result[field] = encrypt_secret(result[field])
    return result


def decrypt_json_passwords(data: dict, password_fields: list[str]) -> dict:
    """
    Decrypt password fields in a JSON dict (e.g., connection_config).

    Args:
        data: Dictionary potentially containing encrypted password fields
        password_fields: List of field names to decrypt

    Returns:
        Copy of dict with specified fields decrypted
    """
    if not data:
        return data

    result = dict(data)
    for field in password_fields:
        if field in result and result[field]:
            result[field] = decrypt_secret(result[field])
    return result


# Password fields in connection_config JSON
CONNECTION_CONFIG_PASSWORD_FIELDS = [
    "password",  # postgres, mssql, solidworks_pdm
    "token",  # influxdb
    "ssh_password",  # odoo_shell ssh mode
    "ssh_key_passphrase",  # odoo_shell ssh mode
    "ssh_key_content",  # ssh private key (sensitive)
    "key_passphrase",  # ssh_shell
    "key_content",  # ssh private key (sensitive)
    "smb_password",  # filesystem smb mount
    "access_token",  # cloud userspace mounts
    "oauth_token",  # cloud userspace mounts (legacy alias)
    "refresh_token",  # cloud userspace mounts
    "oauth_refresh_token",  # cloud userspace mounts (legacy alias)
    "client_secret",  # cloud OAuth app config when stored in JSON
    # SSH tunnel fields (postgres, mysql, mssql, pdm)
    "ssh_tunnel_password",
    "ssh_tunnel_key_content",
    "ssh_tunnel_key_passphrase",
]
