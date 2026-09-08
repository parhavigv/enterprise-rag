"""Password hashing hardened for credential storage.

Uses PBKDF2-HMAC-SHA256 with a per-user random salt and the OWASP-recommended
iteration count for 2023+ (600k). The stored value is versioned and
self-describing so the scheme can be upgraded without a data migration::

    $pbkdf2-sha256$i=600000$<salt_hex>$<digest_hex>

Properties that make this production-suitable (vs. a bare SHA-256 digest):

    * One-way and salted per user, so identical passwords never share a hash
      and rainbow tables / hash-cracking GPUs do not amortise.
    * Keyed by an output-size-constrained HMAC construction, not a fast hash.
    * Constant-time digest comparison (``hmac.compare_digest``).
    * Versioned encoding so a stronger KDF (argon2id) can be adopted later
      while old records keep verifying.

This is stdlib-only and FIPS-friendly; no native dependencies.

>>> hashed = hash_password("hunter2!")
>>> verify_password("hunter2!", hashed)
True
>>> verify_password("wrong", hashed)
False
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

_ALGORITHM = "pbkdf2-sha256"
_DIGEST = hashlib.sha256
_PBKDF2_HMAC = "sha256"
_DERIVED_KEY_LENGTH = 32  # bytes
_SALT_BYTES = 16
DEFAULT_ITERATIONS = 600_000
MAX_PASSWORD_LENGTH = 4096  # NIST SP 800-63B: pre-hash / cap excessively long secrets


class PasswordSchemeError(ValueError):
    """Raised when a stored hash uses an unknown or malformed scheme."""


def _format(iterations: int, salt_hex: str, digest_hex: str) -> str:
    return f"${_ALGORITHM}$i={iterations}${salt_hex}${digest_hex}"


def hash_password(
    password: str,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    salt: str | None = None,
) -> str:
    """Return a versioned PBKDF2-HMAC-SHA256 hash of ``password``.

    Args:
        password: the plaintext secret (unicode).
        iterations: PBKDF2 work factor (``i=`` in the output).
        salt: optional hex salt (random 16 bytes when omitted).
    """
    if not isinstance(password, str):
        raise TypeError("password must be a str")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"password exceeds maximum length of {MAX_PASSWORD_LENGTH} characters")
    salt = salt or secrets.token_hex(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        _PBKDF2_HMAC,
        password.encode("utf-8"),
        salt.encode("ascii"),
        iterations,
        dklen=_DERIVED_KEY_LENGTH,
    )
    return _format(iterations, salt, digest.hex())


def verify_password(password: str, encoded: str) -> bool:
    """Compare ``password`` against a versioned hash in constant time.

    Returns ``False`` (never raises) for malformed or foreign-scheme records
    so a corrupt store degrades to a failed login, not a 500.
    """
    try:
        _algo, salt, iterations, digest = _parse(encoded)
    except PasswordSchemeError:
        return False
    if _algo != _ALGORITHM:
        return False
    candidate = hashlib.pbkdf2_hmac(
        _PBKDF2_HMAC,
        password.encode("utf-8"),
        salt.encode("ascii"),
        iterations,
        dklen=_DERIVED_KEY_LENGTH,
    )
    return hmac.compare_digest(candidate.hex(), digest)


def _parse(encoded: str) -> tuple[str, str, int, str]:
    """Split ``$algo$i=<n>$<salt>$<digest>`` into its parts."""
    if not isinstance(encoded, str):
        raise PasswordSchemeError("password hash must be a string")
    parts = encoded.split("$")
    if len(parts) != 5 or parts[0] != "":
        raise PasswordSchemeError("malformed password hash")
    _, algo, iterations_tag, salt, digest = parts
    if not iterations_tag.startswith("i="):
        raise PasswordSchemeError("missing iteration count")
    try:
        iterations = int(iterations_tag[2:])
    except ValueError as exc:  # noqa: PERF203 - re-raise with context
        raise PasswordSchemeError("non-numeric iteration count") from exc
    if iterations <= 0 or not salt or not digest:
        raise PasswordSchemeError("invalid salt or digest")
    return algo, salt, iterations, digest


if __name__ == "__main__":  # pragma: no cover - dev CLI
    import argparse

    parser = argparse.ArgumentParser(description="Generate a PBKDF2 password hash")
    parser.add_argument("password", help="plaintext password to hash")
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"PBKDF2 work factor (default {DEFAULT_ITERATIONS})",
    )
    args = parser.parse_args()
    print(hash_password(args.password, iterations=args.iterations))
