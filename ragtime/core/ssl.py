"""
SSL certificate management for HTTPS support.

Features:
- Auto-detects certificate files in the SSL directory
- Validates certificates for common issues (expired, mismatched key, etc.)
- Generates self-signed certificates when needed
- Falls back to HTTP with clear error messages on failure
"""

import os
import subprocess
from pathlib import Path

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Default paths for certificates
DEFAULT_CERT_DIR = Path(os.environ.get("INDEX_DATA_PATH", "/data")) / "ssl"
DEFAULT_CERT_FILE = DEFAULT_CERT_DIR / "server.crt"
DEFAULT_KEY_FILE = DEFAULT_CERT_DIR / "server.key"

# Common certificate file patterns
CERT_PATTERNS = ["*.crt", "*.pem", "*.cert", "*certificate*"]
KEY_PATTERNS = ["*.key", "*private*", "*.pem"]

# Certificate validity period (1 year)
CERT_VALIDITY_DAYS = 365


class SSLError(Exception):
    """SSL configuration error with user-friendly message."""

    pass


class SSLValidationResult:
    """Result of SSL certificate validation."""

    def __init__(self) -> None:
        self.valid = True
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.cert_path: Path | None = None
        self.key_path: Path | None = None

    def add_error(self, msg: str) -> None:
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def setup_ssl(
    cert_file: str | None = None,
    key_file: str | None = None,
    auto_generate: bool = True,
) -> tuple[Path, Path] | None:
    """
    Set up SSL certificates for HTTPS.

    This function:
    1. Looks for existing certificates (explicit paths or auto-detect)
    2. Validates them thoroughly
    3. Generates self-signed certs if none found and auto_generate=True
    4. Returns None with clear error messages if SSL cannot be configured

    Args:
        cert_file: Explicit path to certificate file
        key_file: Explicit path to private key file
        auto_generate: Generate self-signed cert if none found

    Returns:
        Tuple of (cert_path, key_path) if successful, None if SSL should be disabled
    """
    result = SSLValidationResult()

    # Step 1: Find certificate files
    if cert_file and key_file:
        # Explicit paths provided
        result.cert_path = Path(cert_file)
        result.key_path = Path(key_file)
    else:
        # Try to auto-detect in SSL directory
        detected = _detect_certificates(DEFAULT_CERT_DIR)
        if detected:
            result.cert_path, result.key_path = detected
            logger.info(
                f"Auto-detected certificates: {result.cert_path.name}, {result.key_path.name}"
            )
        elif auto_generate:
            # No certs found, generate new ones
            logger.info("No certificates found, generating self-signed certificate...")
            try:
                _generate_self_signed_cert(DEFAULT_CERT_FILE, DEFAULT_KEY_FILE)
                result.cert_path = DEFAULT_CERT_FILE
                result.key_path = DEFAULT_KEY_FILE
                return result.cert_path, result.key_path
            except Exception as e:
                logger.error(f"Failed to generate SSL certificate: {e}")
                logger.warning("Falling back to HTTP")
                return None
        else:
            logger.error(f"No SSL certificates found in {DEFAULT_CERT_DIR}")
            logger.warning("Falling back to HTTP")
            return None

    # Step 2: Validate certificates
    _validate_ssl_files(result)

    # Log results
    for warning in result.warnings:
        logger.warning(f"SSL: {warning}")

    if not result.valid:
        logger.error("=" * 60)
        logger.error("SSL CERTIFICATE ERRORS")
        logger.error("=" * 60)
        for error in result.errors:
            logger.error(f"  - {error}")
        logger.error("-" * 60)
        logger.error(
            "Fix these issues or remove the certificates to auto-generate new ones."
        )
        logger.error("Falling back to HTTP")
        logger.error("=" * 60)
        return None

    logger.info(f"SSL certificates validated: {result.cert_path}")
    return result.cert_path, result.key_path


def _detect_certificates(ssl_dir: Path) -> tuple[Path, Path] | None:
    """
    Auto-detect certificate and key files in a directory.

    Uses content inspection (not file extensions) to reliably identify
    certificate vs key files. Prefers user-provided certificates over
    Ragtime self-signed ones.

    Returns (cert_path, key_path) or None if not found.
    """
    if not ssl_dir.exists():
        return None

    # Separate user certs from self-signed ones
    user_certs: list[Path] = []
    self_signed_certs: list[Path] = []
    user_keys: list[Path] = []
    self_signed_keys: list[Path] = []

    for f in ssl_dir.iterdir():
        if not f.is_file():
            continue

        # Content-based detection - actually inspect the file
        file_type = _peek_file_type(f)

        if file_type == "certificate":
            if _is_ragtime_self_signed(f):
                self_signed_certs.append(f)
            else:
                user_certs.append(f)

        elif file_type == "key":
            # Check if this key pairs with a self-signed cert
            # (heuristic: server.key next to self-signed server.crt)
            paired_cert = ssl_dir / f.stem.replace("_key", "").replace("-key", "")
            for ext in [".crt", ".pem", ".cert"]:
                potential_cert = ssl_dir / (f.stem + ext)
                if potential_cert.exists() and _is_ragtime_self_signed(potential_cert):
                    self_signed_keys.append(f)
                    break
            else:
                # Default: treat as user key unless it's our standard name
                if f.name == "server.key":
                    server_crt = ssl_dir / "server.crt"
                    if server_crt.exists() and _is_ragtime_self_signed(server_crt):
                        self_signed_keys.append(f)
                    else:
                        user_keys.append(f)
                else:
                    user_keys.append(f)

    # Prefer user certs over self-signed
    cert_file = (
        user_certs[0]
        if user_certs
        else (self_signed_certs[0] if self_signed_certs else None)
    )
    key_file = (
        user_keys[0]
        if user_keys
        else (self_signed_keys[0] if self_signed_keys else None)
    )

    if cert_file and key_file:
        if user_certs:
            logger.info(f"Using user-provided certificate: {cert_file.name}")
        return cert_file, key_file

    return None


def _is_ragtime_self_signed(cert_path: Path) -> bool:
    """Check if a certificate is a Ragtime-generated self-signed cert."""
    try:
        result = subprocess.run(
            ["openssl", "x509", "-in", str(cert_path), "-noout", "-subject"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # Our self-signed certs have "O=Ragtime/OU=Self-Signed" in subject
            subject = result.stdout.lower()
            return "ragtime" in subject and "self-signed" in subject
    except Exception:
        pass
    return False


def _peek_file_type(path: Path) -> str | None:
    """Peek at a PEM file to determine if it's a certificate or key."""
    try:
        content = path.read_text()[:500]
        if "CERTIFICATE" in content:
            return "certificate"
        if "PRIVATE KEY" in content:
            return "key"
    except Exception:
        pass
    return None


def _validate_ssl_files(result: SSLValidationResult) -> None:
    """
    Thoroughly validate SSL certificate and key files.

    Checks for:
    - File existence and readability
    - Swapped cert/key files
    - Certificate expiration
    - Cert/key pair mismatch
    - Weak algorithms
    """
    cert_path = result.cert_path
    key_path = result.key_path

    if not cert_path or not key_path:
        result.add_error("Certificate or key path not set")
        return

    # Check files exist
    if not cert_path.exists():
        result.add_error(f"Certificate file not found: {cert_path}")
        return

    if not key_path.exists():
        result.add_error(f"Key file not found: {key_path}")
        return

    # Check for swapped files (common mistake)
    cert_content = cert_path.read_text()[:500]
    key_content = key_path.read_text()[:500]

    if "PRIVATE KEY" in cert_content:
        result.add_error(
            f"Certificate file contains a private key - files may be swapped. "
            f"Expected certificate in {cert_path.name}, got key content."
        )
        return

    if "CERTIFICATE" in key_content and "PRIVATE KEY" not in key_content:
        result.add_error(
            f"Key file contains a certificate - files may be swapped. "
            f"Expected private key in {key_path.name}, got certificate content."
        )
        return

    if "CERTIFICATE" not in cert_content:
        result.add_error(
            f"Certificate file does not contain a valid certificate: {cert_path.name}"
        )
        return

    if "PRIVATE KEY" not in key_content:
        result.add_error(
            f"Key file does not contain a valid private key: {key_path.name}"
        )
        return

    # Check certificate expiration
    try:
        exp_result = subprocess.run(
            ["openssl", "x509", "-in", str(cert_path), "-noout", "-checkend", "0"],
            capture_output=True,
            text=True,
        )
        if exp_result.returncode != 0:
            result.add_error("Certificate has expired")
            return

        # Check if expiring within 30 days
        exp_30 = subprocess.run(
            [
                "openssl",
                "x509",
                "-in",
                str(cert_path),
                "-noout",
                "-checkend",
                "2592000",
            ],
            capture_output=True,
            text=True,
        )
        if exp_30.returncode != 0:
            result.add_warning("Certificate expires within 30 days")

    except FileNotFoundError:
        result.add_error("openssl command not found - cannot validate certificate")
        return
    except Exception as e:
        result.add_error(f"Failed to check certificate expiration: {e}")
        return

    # Check cert/key pair match (modulus comparison)
    try:
        cert_mod = subprocess.run(
            ["openssl", "x509", "-in", str(cert_path), "-noout", "-modulus"],
            capture_output=True,
            text=True,
        )
        key_mod = subprocess.run(
            ["openssl", "rsa", "-in", str(key_path), "-noout", "-modulus"],
            capture_output=True,
            text=True,
        )

        if cert_mod.returncode != 0:
            result.add_error(
                f"Cannot read certificate modulus: {cert_mod.stderr.strip()}"
            )
            return

        if key_mod.returncode != 0:
            # Try EC key
            key_mod = subprocess.run(
                ["openssl", "ec", "-in", str(key_path), "-noout", "-text"],
                capture_output=True,
                text=True,
            )
            if key_mod.returncode != 0:
                result.add_error(f"Cannot read private key: {key_mod.stderr.strip()}")
                return
            # For EC keys, we can't easily compare modulus, skip this check
        else:
            # RSA key - compare modulus
            cert_modulus = cert_mod.stdout.strip()
            key_modulus = key_mod.stdout.strip()

            if cert_modulus != key_modulus:
                result.add_error(
                    "Certificate and private key do not match. "
                    "The key was not used to generate this certificate."
                )
                return

    except Exception as e:
        result.add_error(f"Failed to validate cert/key pair: {e}")
        return

    # Check for weak signature algorithms
    try:
        sig_result = subprocess.run(
            ["openssl", "x509", "-in", str(cert_path), "-noout", "-text"],
            capture_output=True,
            text=True,
        )
        if sig_result.returncode == 0:
            sig_text = sig_result.stdout.lower()
            if "sha1" in sig_text and "signature algorithm" in sig_text:
                result.add_warning(
                    "Certificate uses SHA-1 signature which is deprecated. "
                    "Consider regenerating with SHA-256."
                )
            if "md5" in sig_text:
                result.add_error(
                    "Certificate uses MD5 signature which is insecure. "
                    "Regenerate with SHA-256 or stronger."
                )
    except Exception:
        pass  # Non-critical check


def _generate_self_signed_cert(cert_path: Path, key_path: Path) -> None:
    """Generate a self-signed SSL certificate using openssl."""
    import socket

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    hostname = socket.gethostname()
    subject = f"/CN={hostname}/O=Ragtime/OU=Self-Signed"

    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:4096",
                "-sha256",
                "-keyout",
                str(key_path),
                "-out",
                str(cert_path),
                "-days",
                str(CERT_VALIDITY_DAYS),
                "-nodes",
                "-subj",
                subject,
                "-addext",
                f"subjectAltName=DNS:localhost,DNS:{hostname},IP:127.0.0.1",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Generated self-signed certificate: {cert_path}")
        logger.info(f"Certificate valid for {CERT_VALIDITY_DAYS} days")
        logger.warning(
            "This is a self-signed certificate. Browsers will show security warnings. "
            "For production, use a reverse proxy with proper SSL certificates."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to generate SSL certificate: {e.stderr}") from e
    except FileNotFoundError as exc:
        raise RuntimeError(
            "openssl command not found. Ensure openssl is installed in the container."
        ) from exc
