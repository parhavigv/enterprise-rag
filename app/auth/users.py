"""Development user store for the RBAC demo.

In production this would back onto an identity provider (Okta, Entra ID,
OIDC) or a real DB; here a small static list is enough to demonstrate the
permission model and guardrail tests. Passwords are hashed with SHA-256
(salted) purely so we are not storing plaintext in the repo.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path

_USERS_PATH = Path(__file__).resolve().parent / "users.json"


@dataclass(frozen=True)
class StoredUser:
    username: str
    password_hash: str
    salt: str
    role: str
    name: str
    departments: tuple[str, ...]


def _hash(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()


def _load_users() -> list[StoredUser]:
    import json

    with open(_USERS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    out: list[StoredUser] = []
    for entry in raw:
        salt = entry.get("salt", entry["username"])
        out.append(
            StoredUser(
                username=entry["username"],
                password_hash=entry["password_hash"],
                salt=salt,
                role=entry["role"],
                name=entry.get("name", entry["username"]),
                departments=tuple(entry.get("departments") or []),
            )
        )
    return out


class UserStore:
    """Validates credentials against the static user registry."""

    def __init__(self, *, path: Path | None = None) -> None:
        self._path = str(path or _USERS_PATH)
        self._users = _load_users()

    def authenticate(self, username: str, password: str) -> StoredUser | None:
        user = next((u for u in self._users if u.username == username), None)
        if user is None:
            return None
        if not hmac.compare_digest(user.password_hash, _hash(password, user.salt)):
            return None
        return user
