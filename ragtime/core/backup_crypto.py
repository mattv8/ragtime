from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
from typing import BinaryIO, Protocol, cast

from cryptography.fernet import Fernet, InvalidToken

from ragtime.core.encryption import PASSWORD_EXPORT_KDF_ALGORITHM, PASSWORD_EXPORT_KDF_ITERATIONS, derive_password_fernet_key

MAGIC = b"RAGBAK\n"
FORMAT_VERSION = 1
_HEADER_STRUCT = struct.Struct("!H")
_FRAME_STRUCT = struct.Struct("!I")
_MAX_CHUNK_SIZE = 8 * 1024 * 1024
_MAX_FRAME_OVERHEAD = 64 * 1024


class _PeekableBinaryIO(Protocol):
    def peek(self, size: int = ...) -> bytes | bytearray | memoryview: ...


class BackupCryptoError(ValueError):
    pass


def _build_header(*, salt: bytes, chunk_size: int) -> bytes:
    header = {
        "version": FORMAT_VERSION,
        "kdf": {
            "algorithm": PASSWORD_EXPORT_KDF_ALGORITHM,
            "iterations": PASSWORD_EXPORT_KDF_ITERATIONS,
        },
        "salt": base64.urlsafe_b64encode(salt).decode("ascii"),
        "chunk_size": chunk_size,
    }
    return json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    payload = stream.read(size)
    if payload is None or len(payload) != size:
        raise BackupCryptoError("Unexpected end of encrypted backup stream")
    return payload


def _read_header(source: BinaryIO) -> tuple[dict[str, object], bytes]:
    magic = _read_exact(source, len(MAGIC))
    if magic != MAGIC:
        raise BackupCryptoError("Not an encrypted Ragtime backup")
    header_length = _HEADER_STRUCT.unpack(_read_exact(source, _HEADER_STRUCT.size))[0]
    if header_length <= 0 or header_length > 4096:
        raise BackupCryptoError("Invalid encrypted backup header length")
    header_bytes = _read_exact(source, header_length)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupCryptoError("Invalid encrypted backup header") from exc
    if header.get("version") != FORMAT_VERSION:
        raise BackupCryptoError("Unsupported encrypted backup version")
    kdf = header.get("kdf") or {}
    if kdf.get("algorithm") != PASSWORD_EXPORT_KDF_ALGORITHM:
        raise BackupCryptoError("Unsupported encrypted backup KDF algorithm")
    if kdf.get("iterations") != PASSWORD_EXPORT_KDF_ITERATIONS:
        raise BackupCryptoError("Unsupported encrypted backup KDF iterations")
    chunk_size = header.get("chunk_size")
    if not isinstance(chunk_size, int) or chunk_size <= 0 or chunk_size > _MAX_CHUNK_SIZE:
        raise BackupCryptoError("Invalid encrypted backup chunk size")
    if not isinstance(header.get("salt"), str) or not header["salt"]:
        raise BackupCryptoError("Encrypted backup header is missing salt")
    return header, header_bytes


def encrypt_stream(source: BinaryIO, destination: BinaryIO, password: str, *, chunk_size: int = 1024 * 1024) -> None:
    if not password:
        raise BackupCryptoError("Backup password is required")
    if chunk_size <= 0 or chunk_size > _MAX_CHUNK_SIZE:
        raise BackupCryptoError("Invalid encrypted backup chunk size")

    salt_bytes = os.urandom(16)
    header_bytes = _build_header(salt=salt_bytes, chunk_size=chunk_size)
    header_digest = hashlib.sha256(header_bytes).hexdigest()
    fernet = Fernet(derive_password_fernet_key(password, salt_bytes))
    digest = hashlib.sha256()
    chunk_count = 0
    byte_count = 0

    destination.write(MAGIC)
    destination.write(_HEADER_STRUCT.pack(len(header_bytes)))
    destination.write(header_bytes)

    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
        byte_count += len(chunk)
        payload = json.dumps(
            {
                "type": "chunk",
                "sequence": chunk_count,
                "data": base64.b64encode(chunk).decode("ascii"),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        token = fernet.encrypt(payload)
        destination.write(_FRAME_STRUCT.pack(len(token)))
        destination.write(token)
        chunk_count += 1

    final_payload = json.dumps(
        {
            "type": "final",
            "sequence": chunk_count,
            "header_digest": header_digest,
            "chunk_count": chunk_count,
            "byte_count": byte_count,
            "sha256": digest.hexdigest(),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    final_token = fernet.encrypt(final_payload)
    destination.write(_FRAME_STRUCT.pack(len(final_token)))
    destination.write(final_token)


def decrypt_stream(source: BinaryIO, destination: BinaryIO, password: str) -> None:
    if not password:
        raise BackupCryptoError("Backup password is required")

    header, header_bytes = _read_header(source)
    chunk_size_value = header.get("chunk_size")
    if not isinstance(chunk_size_value, int):
        raise BackupCryptoError("Encrypted backup chunk size is invalid")
    chunk_size = chunk_size_value
    header_digest = hashlib.sha256(header_bytes).hexdigest()
    salt = base64.urlsafe_b64decode(str(header["salt"]).encode("ascii"))
    fernet = Fernet(derive_password_fernet_key(password, salt))
    digest = hashlib.sha256()
    expected_sequence = 0
    byte_count = 0
    saw_final = False
    max_frame_length = chunk_size * 4 + _MAX_FRAME_OVERHEAD

    while True:
        length_bytes = source.read(_FRAME_STRUCT.size)
        if not length_bytes:
            break
        if len(length_bytes) != _FRAME_STRUCT.size:
            raise BackupCryptoError("Encrypted backup frame header is truncated")
        frame_length = _FRAME_STRUCT.unpack(length_bytes)[0]
        if frame_length <= 0 or frame_length > max_frame_length:
            raise BackupCryptoError("Encrypted backup frame length is invalid")
        token = _read_exact(source, frame_length)
        try:
            payload = json.loads(fernet.decrypt(token).decode("utf-8"))
        except (InvalidToken, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackupCryptoError("Encrypted backup could not be decrypted") from exc

        record_type = payload.get("type")
        if saw_final and record_type == "final":
            raise BackupCryptoError("Encrypted backup contains more than one final record")
        if saw_final and record_type == "chunk":
            raise BackupCryptoError("Encrypted backup contains chunk data after final record")

        if payload.get("sequence") != expected_sequence:
            raise BackupCryptoError("Encrypted backup record ordering is invalid")

        if record_type == "chunk":
            chunk = base64.b64decode(payload.get("data", ""))
            if len(chunk) > chunk_size:
                raise BackupCryptoError("Encrypted backup chunk exceeds header chunk size")
            destination.write(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
        elif record_type == "final":
            saw_final = True
            if payload.get("header_digest") != header_digest:
                raise BackupCryptoError("Encrypted backup header digest mismatch")
            if payload.get("chunk_count") != expected_sequence:
                raise BackupCryptoError("Encrypted backup chunk count mismatch")
            if payload.get("byte_count") != byte_count:
                raise BackupCryptoError("Encrypted backup byte count mismatch")
            if payload.get("sha256") != digest.hexdigest():
                raise BackupCryptoError("Encrypted backup digest mismatch")
        else:
            raise BackupCryptoError("Encrypted backup record type is invalid")
        expected_sequence += 1

    if not saw_final:
        raise BackupCryptoError("Encrypted backup is missing final digest record")


def is_encrypted_backup(source: BinaryIO) -> bool:
    if hasattr(source, "tell") and hasattr(source, "seek"):
        offset = source.tell()
        source.seek(0)
        prefix = source.read(len(MAGIC))
        source.seek(offset)
        return prefix == MAGIC
    if hasattr(source, "peek"):
        peekable_source = cast(_PeekableBinaryIO, source)
        return bytes(peekable_source.peek(len(MAGIC))[: len(MAGIC)]) == MAGIC
    raise BackupCryptoError("Encrypted backup detection requires a seekable or peekable stream")
