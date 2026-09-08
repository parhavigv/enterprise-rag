"""Identity store backing the RBAC layer.

Two implementations live behind a single seam:

* :class:`SqliteUserStore` (default) - a real, durable identity table backed
  by SQLite (stdlib, WAL journal) with account lockout and a login-attempt
  audit trail. Usernames are case-insensitive; passwords are stored as
  PBKDF2 hashes via :mod:`app.auth.passwords`.
* :class:`JsonUserStore` (legacy) - a read-only registry loaded from
  ``app/auth/users.json``. Kept for local dev/demo scenarios where a file is
  convenient; credentials are still PBKDF2-hashed.

:func:`build_user_store` selects the backend from settings, and
:func:`seed_demo_users` provisions the four demo identities (used by the
dev auto-seed and the ``scripts/manage_users.py`` CLI).

Production note: SQLite scales comfortably to single-instance deployments but
is not an IdP. For many services, SSO federation, or fine-grained accounts,
swap this seam for Okta / Entra ID / Keycloak (OIDC) or a central user table.
The interface below - ``authenticate(username, password, client_ip) ->
StoredUser | None`` plus the account-management methods - is the swap point;
the API layer never touches the store directly.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.auth.models import ROLES, get_role
from app.auth.passwords import hash_password, verify_password
from app.core.logging import get_logger

logger = get_logger(__name__)

_USERS_PATH = Path(__file__).resolve().parent / "users.json"


# --------------------------------------------------------------------- #
# Shared data type + timing equaliser
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class StoredUser:
    username: str
    password_hash: str
    role: str
    name: str
    departments: tuple[str, ...]


# Constant-time work equaliser for unknown usernames (see authenticate).
_DUMMY_HASH = (
    "$pbkdf2-sha256$i=600000$"
    + "0" * 32
    + "$"
    + hashlib.sha256(b"enterprise-rag-unknown-user").hexdigest()
)

_DEMO_USERS: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    (
        "alice",
        "admin",
        "admin-password!",
        "Alice Admin",
        ("PUBLIC", "ENGINEERING", "PRODUCT", "FINANCE", "HR", "LEGAL", "OPS"),
    ),
    (
        "bob",
        "manager",
        "manager-password!",
        "Bob Manager",
        ("PUBLIC", "ENGINEERING", "PRODUCT", "FINANCE", "HR", "OPS"),
    ),
    (
        "carol",
        "employee",
        "employee-password!",
        "Carol Employee",
        ("PUBLIC", "ENGINEERING", "OPS"),
    ),
    (
        "dave",
        "intern",
        "intern-password!",
        "Dave Intern",
        ("PUBLIC",),
    ),
)


def _dummy_pbkdf2(password: str) -> bool:
    """Verifies against the dummy hash so unknown users cost similar work."""
    return verify_password(password, _DUMMY_HASH)


# --------------------------------------------------------------------- #
# Port
# --------------------------------------------------------------------- #
class UserStore:
    """Common interface for identity backends."""

    @property
    def users(self) -> tuple[StoredUser, ...]:
        raise NotImplementedError

    def get(self, username: str) -> StoredUser | None:
        raise NotImplementedError

    def authenticate(
        self,
        username: str,
        password: str,
        *,
        client_ip: str | None = None,
    ) -> StoredUser | None:
        """Return the matching user on valid credentials, else ``None``."""
        raise NotImplementedError


# --------------------------------------------------------------------- #
# Legacy JSON backend (read-only, dev convenience)
# --------------------------------------------------------------------- #
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


class JsonUserStore(UserStore):
    """Validates credentials against a static JSON registry (dev/demo only)."""

    def __init__(self, *, path: Path | None = None) -> None:
        self._path = path or _USERS_PATH
        self._users = _load_users(self._path)

    @property
    def users(self) -> tuple[StoredUser, ...]:
        return tuple(self._users)

    def get(self, username: str) -> StoredUser | None:
        return next((u for u in self._users if u.username == username), None)

    def authenticate(
        self,
        username: str,
        password: str,
        *,
        client_ip: str | None = None,  # noqa: ARG002 - parity with the port
    ) -> StoredUser | None:
        """Return the user on valid credentials, else ``None``.

        Unknown usernames and failed hashes follow the same code path so the
        endpoint cannot be used to enumerate accounts (no timing or status
        side-channel distinguishing "bad user" from "bad password").
        """
        user = self.get(username)
        if user is None:
            _dummy_pbkdf2(password)
            return None
        if not verify_password(password, user.password_hash):
            logger.warning("Failed login attempt | actor={}", username)
            return None
        return user


# --------------------------------------------------------------------- #
# SQLite backend (production default)
# --------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username        TEXT PRIMARY KEY COLLATE NOCASE,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT '',
    departments     TEXT NOT NULL DEFAULT '',
    is_active       INTEGER NOT NULL DEFAULT 1,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    REAL,
    last_login_at   REAL,
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS login_attempts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    username     TEXT NOT NULL COLLATE NOCASE,
    success      INTEGER NOT NULL,
    client_ip    TEXT,
    attempted_at REAL NOT NULL
);
"""


class SqliteUserStore(UserStore):
    """Durable, lockout-aware identity store on SQLite.

    Features:

    * case-insensitive unique usernames (``COLLATE NOCASE`` primary key)
    * account lockout after N consecutive failures (locked for ``lockout_seconds``)
    * ``login_attempts`` audit table recording success/failure + source IP
    * concurrent-safe writes guarded by a process lock (SQLite WAL)
    """

    def __init__(
        self,
        *,
        path: Path | str,
        max_failed_attempts: int = 5,
        lockout_seconds: int = 300,
    ) -> None:
        self._path = Path(path)
        self._max_failed_attempts = max(1, int(max_failed_attempts))
        self._lockout_seconds = max(0, int(lockout_seconds))
        self._lock = threading.Lock()
        if self._path.parent:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # -- internals ----------------------------------------------------- #
    def close(self) -> None:
        self._db.close()

    def _row_by_username(self, username: str) -> sqlite3.Row | None:
        cur = self._db.execute("SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,))
        return cur.fetchone()

    def _to_user(self, row: sqlite3.Row) -> StoredUser:
        deps = tuple(d for d in (row["departments"] or "").split(",") if d)
        return StoredUser(
            username=row["username"],
            password_hash=row["password_hash"],
            role=row["role"],
            name=row["name"],
            departments=deps,
        )

    def _record_attempt(self, username: str, success: bool, client_ip: str | None) -> None:
        self._db.execute(
            "INSERT INTO login_attempts (username, success, client_ip, attempted_at) "
            "VALUES (?, ?, ?, ?)",
            (username, 1 if success else 0, client_ip, time.time()),
        )
        self._db.commit()

    def _account_disabled(self, row: sqlite3.Row) -> bool:
        return not bool(row["is_active"])

    def _account_locked(self, row: sqlite3.Row) -> bool:
        until = row["locked_until"]
        return until is not None and until > time.time()

    # -- port ---------------------------------------------------------- #
    @property
    def users(self) -> tuple[StoredUser, ...]:
        cur = self._db.execute("SELECT * FROM users ORDER BY username")
        return tuple(self._to_user(r) for r in cur.fetchall())

    def get(self, username: str) -> StoredUser | None:
        row = self._row_by_username(username)
        return None if row is None else self._to_user(row)

    def authenticate(
        self,
        username: str,
        password: str,
        *,
        client_ip: str | None = None,
    ) -> StoredUser | None:
        """Verify credentials with timing equalisation + account lockout.

        A disabled/locked account is reported exactly like a failed login
        (``None``) so callers cannot distinguish reasons by status code.
        """
        with self._lock:
            row = self._row_by_username(username)
            if row is None:
                _dummy_pbkdf2(password)
                self._record_attempt(username, False, client_ip)
                logger.warning("Failed login attempt | actor={}", username)
                return None
            if self._account_disabled(row) or self._account_locked(row):
                self._record_attempt(username, False, client_ip)
                logger.warning("Rejected login - disabled/locked account | actor={}", username)
                return None
            if verify_password(password, row["password_hash"]):
                self._db.execute(
                    "UPDATE users SET last_login_at = ?, failed_attempts = 0 " "WHERE username = ?",
                    (time.time(), row["username"]),
                )
                self._record_attempt(username, True, client_ip)
                self._db.commit()
                return self._to_user(row)
            attempts = int(row["failed_attempts"]) + 1
            locked_until = None
            if attempts >= self._max_failed_attempts:
                locked_until = time.time() + self._lockout_seconds
                attempts = 0
                logger.warning(
                    "Account locked after failed attempts | actor={} | window={}s",
                    username,
                    self._lockout_seconds,
                )
            self._db.execute(
                "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE username = ?",
                (attempts, locked_until, row["username"]),
            )
            self._record_attempt(username, False, client_ip)
            self._db.commit()
            logger.warning("Failed login attempt | actor={}", username)
            return None

    # -- account management -------------------------------------------- #
    def is_empty(self) -> bool:
        cur = self._db.execute("SELECT COUNT(*) AS n FROM users")
        return cur.fetchone()["n"] == 0

    def create_user(
        self,
        username: str,
        password: str,
        role: str,
        *,
        name: str | None = None,
        departments: tuple[str, ...] = (),
        if_exists: str = "error",
    ) -> StoredUser:
        """Provision an account. ``if_exists``: ``error`` | ``skip`` | ``replace``."""
        if_exists = if_exists.lower()
        if if_exists not in ("error", "skip", "replace"):
            raise ValueError(f"if_exists must be 'error', 'skip' or 'replace', got {if_exists!r}")
        normalized = username.strip().lower()
        if not normalized:
            raise ValueError("username must not be empty")
        role_key = role.strip().lower()
        if role_key not in ROLES:
            raise ValueError(f"Unknown role {role!r}. Known roles: {sorted(ROLES)}")
        role_obj = get_role(role_key)
        deps = sorted({d.strip().upper() for d in departments if d and d.strip()})
        if not deps or "PUBLIC" not in deps:
            deps = ["PUBLIC", *[d for d in deps if d != "PUBLIC"]]
        payload = StoredUser(
            username=normalized,
            password_hash=hash_password(password),
            role=role_obj.name,
            name=(name or normalized).strip(),
            departments=tuple(deps),
        )
        with self._lock:
            existing = self._row_by_username(normalized)
            if existing is not None:
                if if_exists == "skip":
                    return self._to_user(existing)
                if if_exists == "replace":
                    self._update_user(row=existing, user=payload)
                    return payload
                raise ValueError(f"User {normalized!r} already exists")
            self._db.execute(
                "INSERT INTO users "
                "(username, password_hash, role, name, departments, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    payload.username,
                    payload.password_hash,
                    payload.role,
                    payload.name,
                    ",".join(payload.departments),
                    time.time(),
                ),
            )
            self._db.commit()
        logger.info("Created user | actor={} | role={}", payload.username, payload.role)
        return payload

    def _update_user(self, *, row: sqlite3.Row, user: StoredUser) -> None:
        self._db.execute(
            "UPDATE users SET password_hash = ?, role = ?, name = ?, departments = ? "
            "WHERE username = ?",
            (
                user.password_hash,
                user.role,
                user.name,
                ",".join(user.departments),
                row["username"],
            ),
        )
        self._db.commit()

    def set_password(self, username: str, password: str) -> None:
        """Rotate a password and clear lockout/failure counters."""
        with self._lock:
            row = self._row_by_username(username)
            if row is None:
                raise ValueError(f"Unknown user {username!r}")
            self._db.execute(
                "UPDATE users SET password_hash = ?, failed_attempts = 0, locked_until = NULL "
                "WHERE username = ?",
                (hash_password(password), row["username"]),
            )
            self._db.commit()
        logger.info("Password rotated | actor={}", row["username"])

    def set_active(self, username: str, active: bool) -> None:
        with self._lock:
            row = self._row_by_username(username)
            if row is None:
                raise ValueError(f"Unknown user {username!r}")
            self._db.execute(
                "UPDATE users SET is_active = ? WHERE username = ?",
                (1 if active else 0, row["username"]),
            )
            self._db.commit()
        logger.info("Account status set | actor={} | active={}", row["username"], active)


# --------------------------------------------------------------------- #
# Provisioning helpers
# --------------------------------------------------------------------- #
def seed_demo_users(store: UserStore) -> list[StoredUser]:
    """Provision the four demo identities (idempotent, skip-if-exists).

    Only write-capable stores can be seeded; ``JsonUserStore`` raises.
    """
    if not callable(getattr(store, "create_user", None)):
        raise TypeError(
            f"{store.__class__.__name__} is read-only; seed demo users into a "
            "SqliteUserStore or build one via build_user_store()"
        )
    created: list[StoredUser] = []
    for username, role, password, name, deps in _DEMO_USERS:
        created.append(
            store.create_user(
                username,
                password,
                role,
                name=name,
                departments=deps,
                if_exists="skip",
            )
        )
    return created


def build_user_store(settings) -> UserStore:
    """Construct the configured identity backend from :class:`Settings`."""
    backend = getattr(settings, "auth_user_store_backend", "sqlite").lower()
    if backend == "json":
        return JsonUserStore()
    if backend == "sqlite":
        raw_path = getattr(settings, "auth_sqlite_path", "") or None
        path = Path(raw_path) if raw_path else Path(settings.data_dir) / "users.db"
        return SqliteUserStore(
            path=path,
            max_failed_attempts=getattr(settings, "auth_lockout_max_attempts", 5),
            lockout_seconds=getattr(settings, "auth_lockout_seconds", 300),
        )
    raise ValueError(f"Unknown AUTH_USER_STORE_BACKEND {backend!r}. Use 'sqlite' or 'json'.")
