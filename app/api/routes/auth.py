"""Authentication endpoints - issue and inspect JWTs.

``/login`` and ``/token-info`` are rate-limited to blunt brute-force and
token-farming attempts (see :mod:`app.api.rate_limit`). All 401s include an
RFC 6750 ``WWW-Authenticate: Bearer`` challenge and never echo decode
internals back to the client.
"""

from __future__ import annotations

from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from app.api.deps import get_container
from app.api.models import LoginRequest, TokenResponse
from app.api.rate_limit import rate_limit
from app.auth.dependencies import get_auth_service, get_current_user, get_user_store
from app.auth.models import AuthUser
from app.auth.service import AuthService
from app.auth.users import UserStore
from app.core.logging import get_logger
from app.services.container import Container

logger = get_logger(__name__)

router = APIRouter(tags=["auth"])

_WWW_AUTH = {"WWW-Authenticate": "Bearer"}


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Exchange credentials for a JWT",
    dependencies=[Depends(rate_limit("login"))],
)
def login(
    req: LoginRequest,
    request: Request,
    service: AuthService = Depends(get_auth_service),
    store: UserStore = Depends(get_user_store),
    container: Container = Depends(get_container),
) -> TokenResponse:
    client_ip = request.client.host if request.client else None
    user = store.authenticate(req.username, req.password, client_ip=client_ip)
    audit = getattr(container, "audit", lambda: None)()
    if user is None:
        if audit is not None:
            audit.log_auth_result(
                actor=req.username,
                role=None,
                success=False,
                client_ip=client_ip,
                reason="invalid credentials",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers=_WWW_AUTH,
        )
    if audit is not None:
        audit.log_auth_result(
            actor=user.username,
            role=user.role,
            success=True,
            client_ip=client_ip,
            reason=None,
        )
    token = service.issue_token(
        subject=user.username,
        role=user.role,
        departments=list(user.departments),
        name=user.name,
    )
    return TokenResponse(
        access_token=token,
        expires_in=service.expiry_seconds,
        user={
            "username": user.username,
            "name": user.name,
            "role": user.role,
            "departments": list(user.departments),
        },
    )


@router.get("/me", summary="Return the caller's identity")
def me(user: AuthUser = Depends(get_current_user)) -> dict:
    return {
        "username": user.sub,
        "name": user.name,
        "role": user.role.name,
        "max_clearance": user.role.max_clearance.name,
        "departments": sorted(user.departments or set()),
    }


@router.get(
    "/token-info",
    summary="Decode the caller's token claims",
    dependencies=[Depends(rate_limit("token-info"))],
)
def token_info(
    authorization: Annotated[str | None, Header()] = None,
    service: AuthService = Depends(get_auth_service),
) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Send 'Authorization: Bearer <token>'.",
            headers=_WWW_AUTH,
        )
    try:
        payload = service.decode(authorization.split(" ", 1)[1])
    except jwt.InvalidTokenError as exc:
        logger.warning("Rejected token-info request | reason={}", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers=_WWW_AUTH,
        ) from exc
    return {
        "sub": payload.get("sub"),
        "role": payload.get("role"),
        "name": payload.get("name"),
        "departments": payload.get("departments"),
        "iss": payload.get("iss"),
        "aud": payload.get("aud"),
        "exp": payload.get("exp"),
    }
