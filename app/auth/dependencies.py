"""FastAPI dependencies for extracting the authenticated user from a JWT.

The ``Authorization: Bearer <token>`` header is decoded via :class:`AuthService`
into an :class:`AuthUser`. If authentication is disabled (``auth_enabled=0``) a
default admin-like principal is used so the service still runs locally without a
token while tests can exercise the full RBAC path with explicit credentials.

All dependencies resolve the service container through ``get_container`` so
tests can swap it via ``app.dependency_overrides`` exactly as they do for the
rest of the application.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from app.api.deps import get_container
from app.auth.models import AuthUser, get_role
from app.auth.service import AuthService, TokenVerificationError
from app.auth.users import UserStore
from app.core.logging import get_logger
from app.services.container import Container

logger = get_logger(__name__)

_WWW_AUTH = {"WWW-Authenticate": "Bearer"}


def get_auth_service(container: Container = Depends(get_container)) -> AuthService:
    return container.auth()


def get_user_store(container: Container = Depends(get_container)) -> UserStore:
    return container.user_store()


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _unauthorized(detail: str) -> HTTPException:
    """401 with an RFC 6750 ``WWW-Authenticate`` challenge."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers=_WWW_AUTH,
    )


def get_current_user(
    request: Request,
    container: Container = Depends(get_container),
    auth: AuthService = Depends(get_auth_service),
) -> AuthUser:
    """Resolve the request principal from its bearer token."""
    if getattr(container.settings, "auth_enabled", True) is False:
        # Unsecured mode: treat the caller as an admin (no filtering).
        return AuthUser(
            sub="anonymous",
            name="anonymous",
            role=get_role("admin"),
            departments=set(),
        )

    token = _extract_bearer(request.headers.get("Authorization"))
    if token is None:
        logger.info("Rejected request without bearer token | path={}", request.url.path)
        raise _unauthorized("Authentication required. Send 'Authorization: Bearer <token>'.")

    try:
        return auth.verify(token)
    except TokenVerificationError as exc:
        # Log the specific PyJWT reason server-side; never echo it to clients
        # (it can carry algorithm/claim details useful to an attacker).
        logger.warning(
            "Rejected invalid token | path={} | reason={}",
            request.url.path,
            exc.__cause__ or exc,
        )
        raise _unauthorized("Invalid or expired token.") from exc
    except Exception as exc:  # noqa: BLE001 - any unexpected decode failure is a 401
        logger.error("Token verification failed unexpectedly | reason={}", exc)
        raise _unauthorized("Invalid or expired token.") from exc


def require_role(*minimum: str):
    """Dependency factory enforcing a minimum role for an endpoint.

    Usage::

        @router.get("/audit", dependencies=[Depends(require_role("admin", "executive"))])
    """

    def _check(user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if user.role.name not in set(minimum):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role.name}' is not permitted. Required: {', '.join(minimum)}",
            )
        return user

    return _check
