"""
Reversible encryption utilities for storing secrets.

Uses Fernet symmetric encryption with a key derived from ENCRYPTION_KEY.
This allows secrets to be decrypted for display in the frontend or for
backup/restore operations.

IMPORTANT: The encryption key is auto-generated on first startup and persisted
to data/.encryption_key. If this file is lost, all encrypted secrets become
unrecoverable. Always include the encryption key in backups using the
--include-secret flag.
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
        logger.error("Failed to decrypt secret: invalid token (key may have changed)")
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
    "ssh_password",  # odoo_shell ssh mode
    "ssh_key_passphrase",  # odoo_shell ssh mode
    "ssh_key_content",  # ssh private key (sensitive)
    "key_passphrase",  # ssh_shell
    "key_content",  # ssh private key (sensitive)
    "smb_password",  # filesystem smb mount
]
