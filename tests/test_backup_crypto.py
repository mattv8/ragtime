import io
import json
import struct
import unittest

from ragtime.core.backup_crypto import MAGIC


def _read_records(payload: bytes):
    header_length = struct.unpack("!H", payload[len(MAGIC) : len(MAGIC) + 2])[0]
    offset = len(MAGIC) + 2 + header_length
    records = []
    while offset < len(payload):
        frame_length = struct.unpack("!I", payload[offset : offset + 4])[0]
        start = offset + 4
        end = start + frame_length
        records.append(payload[offset:end])
        offset = end
    return payload[: len(MAGIC) + 2 + header_length], records


class BackupCryptoTests(unittest.TestCase):
    def _encrypt(self, data: bytes = b"hello world" * 1024, *, chunk_size: int = 256) -> bytes:
        from ragtime.core.backup_crypto import encrypt_stream

        encrypted = io.BytesIO()
        encrypt_stream(io.BytesIO(data), encrypted, "secret-password", chunk_size=chunk_size)
        return encrypted.getvalue()

    def test_encrypt_round_trip_and_detection(self) -> None:
        from ragtime.core.backup_crypto import decrypt_stream, is_encrypted_backup

        encrypted = io.BytesIO(self._encrypt())

        self.assertTrue(is_encrypted_backup(encrypted))
        self.assertEqual(encrypted.tell(), 0)

        encrypted.seek(0)
        decrypted = io.BytesIO()
        decrypt_stream(encrypted, decrypted, "secret-password")
        self.assertEqual(decrypted.getvalue(), b"hello world" * 1024)

    def test_wrong_password_fails(self) -> None:
        from ragtime.core.backup_crypto import BackupCryptoError, decrypt_stream

        encrypted = io.BytesIO(self._encrypt(b"payload", chunk_size=64))
        with self.assertRaises(BackupCryptoError):
            decrypt_stream(encrypted, io.BytesIO(), "wrong-password")

    def test_truncated_stream_fails_validation(self) -> None:
        from ragtime.core.backup_crypto import BackupCryptoError, decrypt_stream

        encrypted = self._encrypt(b"payload" * 128, chunk_size=128)
        with self.assertRaises(BackupCryptoError):
            decrypt_stream(io.BytesIO(encrypted[:-10]), io.BytesIO(), "secret-password")

    def test_reordered_stream_fails_validation(self) -> None:
        from ragtime.core.backup_crypto import BackupCryptoError, decrypt_stream

        header, records = _read_records(self._encrypt(b"A" * 1024, chunk_size=128))
        reordered = header + records[1] + records[0] + b"".join(records[2:])

        with self.assertRaises(BackupCryptoError):
            decrypt_stream(io.BytesIO(reordered), io.BytesIO(), "secret-password")

    def test_duplicate_record_fails_validation(self) -> None:
        from ragtime.core.backup_crypto import BackupCryptoError, decrypt_stream

        header, records = _read_records(self._encrypt(b"A" * 1024, chunk_size=128))
        duplicated = header + records[0] + records[0] + b"".join(records[1:])

        with self.assertRaises(BackupCryptoError):
            decrypt_stream(io.BytesIO(duplicated), io.BytesIO(), "secret-password")

    def test_oversized_frame_length_fails_validation(self) -> None:
        from ragtime.core.backup_crypto import MAGIC, BackupCryptoError, decrypt_stream

        encrypted = bytearray(self._encrypt(b"payload" * 128, chunk_size=128))
        header_length = struct.unpack("!H", encrypted[len(MAGIC) : len(MAGIC) + 2])[0]
        offset = len(MAGIC) + 2 + header_length
        encrypted[offset : offset + 4] = struct.pack("!I", 10_000_000)

        with self.assertRaises(BackupCryptoError):
            decrypt_stream(io.BytesIO(bytes(encrypted)), io.BytesIO(), "secret-password")

    def test_chunk_after_final_fails_validation(self) -> None:
        from ragtime.core.backup_crypto import BackupCryptoError, decrypt_stream

        header, records = _read_records(self._encrypt(b"A" * 1024, chunk_size=128))
        after_final = header + b"".join(records) + records[0]

        with self.assertRaises(BackupCryptoError):
            decrypt_stream(io.BytesIO(after_final), io.BytesIO(), "secret-password")

    def test_second_final_record_is_rejected(self) -> None:
        from ragtime.core.backup_crypto import BackupCryptoError, decrypt_stream

        header, records = _read_records(self._encrypt(b"A" * 1024, chunk_size=128))
        duplicate_final = header + b"".join(records) + records[-1]

        with self.assertRaises(BackupCryptoError):
            decrypt_stream(io.BytesIO(duplicate_final), io.BytesIO(), "secret-password")


if __name__ == "__main__":
    unittest.main()
