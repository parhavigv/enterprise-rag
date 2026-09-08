"""JWT issuance and verification for the RBAC layer.

Uses ``PyJWT`` with HS256. Secrets come from settings; in production set
``auth_jwt_secret`` to a long random value and ``auth_jwt_algorithm`` /
``auth_jwt_expiry_seconds`` as needed. When ``auth_jwt_secret`` is empty (the
dev/test default) HMAC falls back to a generated in-memory secret so the app
still boots without configuration while signalling that tokens are not shared
across restarts.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

import jwt

from app.auth.models import AuthUser, Role, get_role

_ensure_secret = hashlib.sha256(uuid.uuid4().hex.encode()).hexdigest()


class AuthService:
    """Signs and verifies JWTs, and issues request-scoped principals."""

    def __init__(
        self,
        secret: str | None = None,
        algorithm: str = "HS256",
        expiry_seconds: int = 3600,
        issuer: str = "enterprise-rag",
    ) -> None:
        self._secret = secret or _ensure_secret
        self._algorithm = algorithm
        self._expiry_seconds = expiry_seconds
        self._issuer = issuer

    @property
    def expiry_seconds(self) -> int:
        return self._expiry_seconds

    # ------------------------------------------------------------------ #
    # Token issuance
    # ------------------------------------------------------------------ #
    def issue_token(
        self,
        subject: str,
        role: str,
        *,
        departments: list[str] | None = None,
        name: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Create a signed JWT for the given subject/role."""
        now = int(time.time())
        payload: dict[str, Any] = {
            "sub": subject,
            "role": role,
            "departments": departments or [],
            "name": name or subject,
            "iss": self._issuer,
            "iat": now,
            "exp": now + self._expiry_seconds,
            "jti": uuid.uuid4().hex,
        }
        if extra:
            payload.update(extra)
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    # ------------------------------------------------------------------ #
    # Verification
    # ------------------------------------------------------------------ #
    def decode(self, token: str) -> dict[str, Any]:
        """Decode and validate a token, raising on invalid/expired tokens."""
        return jwt.decode(
            token,
            self._secret,
            algorithms=[self._algorithm],
            issuer=self._issuer,
            options={"require": ["sub", "role", "exp"]},
        )

    def verify(self, token: str) -> AuthUser:
        """Verify a token and return an authenticated :class:`AuthUser`."""
        payload = self.decode(token)
        role: Role = get_role(payload.get("role"))
        departments: set[str] = set(payload.get("departments") or [])
        if role.departments is not None and not departments:
            departments = set(role.departments)
        return AuthUser(
            sub=str(payload["sub"]),
            name=str(payload.get("name") or payload["sub"]),
            role=role,
            departments=departments,
        )
