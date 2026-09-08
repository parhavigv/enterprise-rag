"""Authentication endpoints - issue and inspect JWTs."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.api.models import LoginRequest, TokenResponse
from app.auth.dependencies import get_auth_service, get_current_user, get_user_store
from app.auth.models import AuthUser
from app.auth.service import AuthService
from app.auth.users import UserStore

router = APIRouter(tags=["auth"])


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Exchange credentials for a JWT",
)
def login(
    req: LoginRequest,
    service: AuthService = Depends(get_auth_service),
    store: UserStore = Depends(get_user_store),
) -> TokenResponse:
    user = store.authenticate(req.username, req.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
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


@router.get("/token-info", summary="Decode the caller's token claims")
def token_info(
    authorization: Annotated[str | None, Header()] = None,
    service: AuthService = Depends(get_auth_service),
) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )
    try:
        payload = service.decode(authorization.split(" ", 1)[1])
    except Exception as exc:  # noqa: BLE001 - invalid token surfaces as 401
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        ) from exc
    return {
        "sub": payload.get("sub"),
        "role": payload.get("role"),
        "name": payload.get("name"),
        "departments": payload.get("departments"),
        "iss": payload.get("iss"),
        "exp": payload.get("exp"),
    }
