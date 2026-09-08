"""Pre-retrieval ACL filtering of candidate documents.

The RBAC guarantee lives here: before dense/sparse hits are fused or
re-ranked, every candidate is checked against the requesting user's role.
Chunks the user cannot read are dropped from the pool *entirely* so they can
never influence ranking or reach the LLM context.
"""

from __future__ import annotations

from app.auth.models import ACLMetadata, AuthUser, ClearanceLevel
from app.core.logging import get_logger
from retrieval.types import RetrievedDocument

logger = get_logger(__name__)


def filter_by_permission(
    documents: list[RetrievedDocument],
    user: AuthUser | None,
) -> list[RetrievedDocument]:
    """Drop chunks the user may not read, returning only permitted docs.

    ``user`` of ``None`` (no auth context) falls back to a PUBLIC-only,
    PUBLIC-department principal: the most restrictive safe default.
    """
    effective = user or _public_only_user()
    kept: list[RetrievedDocument] = []
    blocked = 0
    for doc in documents:
        acl = ACLMetadata.from_metadata(doc.metadata or {})
        if effective.can_access(acl):
            kept.append(doc)
        else:
            blocked += 1
    if blocked:
        logger.info(
            "ACL filter | user={} | kept={} | blocked={}",
            getattr(effective, "sub", "anonymous"),
            len(kept),
            blocked,
        )
    return kept


def visibility_filter(where: dict | None, user: AuthUser | None) -> dict | None:
    """Build a ChromaDB-safe ``$and`` metadata filter combining any existing
    ``where`` with the user's clearance/department visibility constraints.

    Dense retrieval applies this as a metadata filter at query time (a
    second, cheap pre-retrieval gate in addition to the post-hoc
    :func:`filter_by_permission` sweep that also covers BM25 sparse hits).
    """
    if user is None:
        user = _public_only_user()
    if user.role.is_admin:
        return where

    allowed_levels = [lv.name for lv in ClearanceLevel if lv <= user.role.max_clearance]
    deps = list(user.departments or set())

    clauses: list[dict] = []
    if where:
        clauses.append(where)
    clauses.append({"clearance_level": {"$in": allowed_levels}})
    clauses.append({"department": {"$in": deps}})
    return {"$and": clauses}


def _public_only_user() -> AuthUser:
    from app.auth.models import get_role

    role = get_role("intern")
    return AuthUser(
        sub="anonymous",
        name="anonymous",
        role=role,
        departments=set(role.departments or {"PUBLIC"}),
    )
