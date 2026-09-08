"""Audit endpoints - expose access-control records to privileged roles.

In this dev build the audit trail lives only in the structured log stream
(an enterprise build would persist to a dedicated datastore). This route
demonstrates the endpoint contract and its RBAC gate: only `admin` and
`executive` roles may read it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.dependencies import require_role
from app.auth.models import AuthUser

router = APIRouter(tags=["audit"])


@router.get("/audit/access", summary="Access-control decision summary")
def audit_summary(
    _: AuthUser = Depends(require_role("admin", "executive")),
) -> dict:
    return {
        "endpoint": "audit/access",
        "notice": "Full audit records are streamed to the structured log",
        "fields": ["event", "actor", "role", "query", "requested", "returned", "filtered"],
    }
