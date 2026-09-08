"""User store backing the RBAC demo.

Loads the static developer registry (``app/auth/users.json``) and validates
credentials against PBKDF2-hashed records via :mod:`app.auth.passwords`.

Production note: this file is a *dev/test stand-in*. A real deployment should
resolve identities against an IdP (Okta, Entra ID, Keycloak/OIDC) or a
scalable identity table behind the user store interface, and rotate the store
out of the container image entirely. The interface used here — ``authenticate
(username, password) -> StoredUser | None`` — is the seam to swap in such a
backend without touching the API layer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from app.auth.passwords import verify_password
from app.core.logging import get_logger

logger = get_logger(__name__)

_USERS_PATH = Path(__file__).resolve().parent / "users.json"


@dataclass(frozen=True)
class StoredUser:
    username: str
    password_hash: str
    role: str
    name: str
    departments: tuple[str, ...]


def _load_users(path: Path) -> list[StoredUser]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: list[StoredUser] = []
    for entry in raw:
        out.append(
            StoredUser(
                username=entry["username"],
                password_hash=entry["password_hash"],
                role=entry["role"],
                name=entry.get("name", entry["username"]),
                departments=tuple(entry.get("departments") or ()),
            )
        )
    return out


class UserStore:
    """Validates credentials against the user registry."""

    def __init__(self, *, path: Path | None = None) -> None:
        self._path = path or _USERS_PATH
        self._users = _load_users(self._path)

    @property
    def users(self) -> tuple[StoredUser, ...]:
        return tuple(self._users)

    def get(self, username: str) -> StoredUser | None:
        """Look up a user by username without verifying credentials."""
        return next((u for u in self._users if u.username == username), None)

    def authenticate(self, username: str, password: str) -> StoredUser | None:
        """Return the user on valid credentials, else ``None``.

        Unknown usernames and failed hashes follow the same code path so the
        endpoint cannot be used to enumerate accounts (no timing or status
        side-channel distinguishing "bad user" from "bad password").
        """
        user = self.get(username)
        if user is None:
            # Burn comparable work even for unknown users to blunt timing probes.
            verify_password(password, _DUMMY_HASH)
            return None
        if not verify_password(password, user.password_hash):
            logger.warning("Failed login attempt | actor={}", username)
            return None
        return user


# Constant-time work equaliser for unknown usernames (see authenticate).
_DUMMY_HASH = (
    "$pbkdf2-sha256$i=600000$"
    + "0" * 32
    + "$"
    + hashlib.sha256(b"enterprise-rag-unknown-user").hexdigest()
)
