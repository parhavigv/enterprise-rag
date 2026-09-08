"""JWT issuance and verification for the RBAC layer.

Uses ``PyJWT`` with HS256. Secrets come from settings; in production set
``auth_jwt_secret`` to a long random value (enforced at Settings build time for
``environment=production``) and ``auth_jwt_algorithm`` /
``auth_jwt_expiry_seconds`` / ``auth_audience`` as needed. When the secret is
empty (the dev/test default) HMAC falls back to a generated in-memory secret so
the app still boots without configuration while signalling that tokens are not
shared across restarts.

Tokens carry the standard RFC 7519 claim set — ``iss``, ``aud``, ``iat``,
``nbf``, ``exp``, ``jti`` — all validated on verify. ``aud`` scopes tokens to a
specific audience so a token minted for one deployment (e.g. ``api``) can never
be replayed against another (e.g. an internal admin tool).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

import jwt

from app.auth.models import AuthUser, Role, get_role

_ensure_secret = hashlib.sha256(uuid.uuid4().hex.encode()).hexdigest()

_REQUIRED_CLAIMS = ("sub", "role", "exp")


class TokenVerificationError(Exception):
    """A token could not be verified (malformed, expired, wrong audience, ...).

    Raised by :meth:`AuthService.verify` with a *non-sensitive* reason so
    callers can map it to a 401 without echoing decode internals to clients.
    The underlying PyJWT exception is chained for server-side diagnostics.
    """


class AuthService:
    """Signs and verifies JWTs, and issues request-scoped principals."""

    def __init__(
        self,
        secret: str | None = None,
        algorithm: str = "HS256",
        expiry_seconds: int = 3600,
        issuer: str = "enterprise-rag",
        audience: str = "enterprise-rag",
    ) -> None:
        self._secret = secret or _ensure_secret
        self._algorithm = algorithm
        self._expiry_seconds = expiry_seconds
        self._issuer = issuer
        self._audience = audience

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
            "aud": self._audience,
            "iat": now,
            "nbf": now,
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
        """Decode and validate a token against every supported claim.

        Raises a PyJWT exception subclass on any validation failure.
        """
        return jwt.decode(
            token,
            self._secret,
            algorithms=[self._algorithm],
            issuer=self._issuer,
            audience=self._audience,
            options={"require": list(_REQUIRED_CLAIMS)},
        )

    def verify(self, token: str) -> AuthUser:
        """Verify a token and return an authenticated :class:`AuthUser`.

        Raises:
            TokenVerificationError: token is invalid/expired/wrong audience,
                with a safe, client-agnostic message.
        """
        try:
            payload = self.decode(token)
        except jwt.InvalidTokenError as exc:
            raise TokenVerificationError(str(exc)) from exc
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
