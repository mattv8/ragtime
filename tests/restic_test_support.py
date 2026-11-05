"""Test-only Restic fixtures with an intentionally weak KDF; never import from runtime code."""

from __future__ import annotations

import atexit
import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.poly1305 import Poly1305

_FAST_SCRYPT_N = 1024
_FAST_SCRYPT_R = 1
_FAST_SCRYPT_P = 1
_RESTIC_BINARY = Path("/opt/ragtime-backup/bin/restic")
_TEMPLATE_LOCK = threading.Lock()
_TEMPLATES: dict[Path, ResticTestTemplate] = {}


@dataclass(frozen=True)
class ResticTestTemplate:
    repository_path: Path
    password: bytes
    repository_id: str


def seed_fast_restic_repository(
    repository_path: Path,
    cache_path: Path,
    password_path: Path,
    *,
    binary: Path = _RESTIC_BINARY,
) -> str:
    """Copy an empty, real Restic repository with a test-cost access key."""
    template = _template_for(binary)
    if repository_path.exists():
        raise FileExistsError(repository_path)
    shutil.copytree(template.repository_path, repository_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    password_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    password_path.write_bytes(template.password)
    password_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return template.repository_id


def _template_for(binary: Path) -> ResticTestTemplate:
    resolved_binary = binary.resolve()
    with _TEMPLATE_LOCK:
        template = _TEMPLATES.get(resolved_binary)
        if template is None:
            template = _create_template(resolved_binary)
            _TEMPLATES[resolved_binary] = template
        return template


def _create_template(binary: Path) -> ResticTestTemplate:
    root = Path(tempfile.mkdtemp(prefix="restic-test-template-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    repository = root / "repository"
    password_path = root / "password"
    password = base64.urlsafe_b64encode(os.urandom(48)) + b"\n"
    password_path.write_bytes(password)
    password_path.chmod(0o600)
    command = [str(binary), "--repo", str(repository), "--password-file", str(password_path), "--cache-dir", str(root / "cache")]
    subprocess.run([*command, "init", "--repository-version", "2"], check=True, capture_output=True, timeout=120)
    _lower_key_cost(repository, password.decode().strip())
    config = json.loads(subprocess.run([*command, "cat", "config"], check=True, capture_output=True, text=True, timeout=120).stdout)
    return ResticTestTemplate(repository, password, config["id"])


def _lower_key_cost(repository_path: Path, password: str) -> None:
    key_paths = list((repository_path / "keys").iterdir())
    if len(key_paths) != 1:
        raise ValueError(f"Expected exactly one Restic key file, found {len(key_paths)}")
    key_path = key_paths[0]
    key_document = json.loads(key_path.read_bytes())
    master_key = _open_key_document(key_document, password)
    salt = os.urandom(64)
    user_key = _derive_user_key(password, salt, _FAST_SCRYPT_N, _FAST_SCRYPT_R, _FAST_SCRYPT_P)
    key_document.update(
        {
            "N": _FAST_SCRYPT_N,
            "r": _FAST_SCRYPT_R,
            "p": _FAST_SCRYPT_P,
            "salt": base64.b64encode(salt).decode(),
            "data": base64.b64encode(_seal(user_key, master_key)).decode(),
        }
    )
    encoded = json.dumps(key_document, separators=(",", ":")).encode()
    replacement = key_path.with_name(hashlib.sha256(encoded).hexdigest())
    replacement.write_bytes(encoded)
    key_path.unlink()


def _open_key_document(document: dict[str, object], password: str) -> bytes:
    salt = base64.b64decode(str(document["salt"]))
    user_key = _derive_user_key(password, salt, _document_int(document, "N"), _document_int(document, "r"), _document_int(document, "p"))
    encrypted = base64.b64decode(str(document["data"]))
    nonce, ciphertext, tag = encrypted[:16], encrypted[16:-16], encrypted[-16:]
    Poly1305.verify_tag(_poly1305_key(user_key, nonce), ciphertext, tag)
    decryptor = Cipher(algorithms.AES(user_key[:32]), modes.CTR(nonce)).decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()


def _document_int(document: dict[str, object], key: str) -> int:
    value = document[key]
    if isinstance(value, int):
        return value
    raise ValueError(f"Restic key {key} is not an integer")


def _seal(user_key: bytes, plaintext: bytes) -> bytes:
    nonce = os.urandom(16)
    encryptor = Cipher(algorithms.AES(user_key[:32]), modes.CTR(nonce)).encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    tag = Poly1305.generate_tag(_poly1305_key(user_key, nonce), ciphertext)
    return nonce + ciphertext + tag


def _poly1305_key(user_key: bytes, nonce: bytes) -> bytes:
    encryptor = Cipher(algorithms.AES(user_key[32:48]), modes.ECB()).encryptor()
    one_time_pad = encryptor.update(nonce) + encryptor.finalize()
    return user_key[48:64] + one_time_pad


def _derive_user_key(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return Scrypt(salt=salt, length=64, n=n, r=r, p=p).derive(password.encode())
