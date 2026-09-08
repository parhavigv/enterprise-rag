"""Audit logging for access-control decisions.

Captures *who asked what*, *how many results were filtered out* and *why*,
producing a structured, queryable record for governance reviews. Records are
written through loguru so they flow into the same JSON log pipeline as the
rest of the service.
"""

from __future__ import annotations

import time
from typing import Any

from app.auth.models import AuthUser
from app.core.logging import get_logger

logger = get_logger("auth.audit")


class AuditLogger:
    """Structured audit trail for RBAC + query activity."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def log_access(
        self,
        *,
        user: AuthUser | None,
        query: str,
        requested: int,
        returned: int,
        filtered: int,
        reason: str | None = None,
    ) -> None:
        """Record a single retrieval/access decision."""
        if not self._enabled:
            return
        record: dict[str, Any] = {
            "event": "acl.retrieval",
            "actor": getattr(user, "sub", "anonymous"),
            "role": getattr(user, "role.name", "anonymous"),
            "query": query,
            "requested": requested,
            "returned": returned,
            "filtered": filtered,
            "reason": reason,
            "ts": time.time(),
        }
        level = "warning" if filtered else "info"
        getattr(logger, level)(
            "{event} | actor={actor} | role={role} | filtered={filtered}", **record
        )

    def log_denied(self, *, user: AuthUser | None, resource: str, reason: str) -> None:
        if not self._enabled:
            return
        logger.warning(
            "event=acl.denied | actor={} | resource={} | reason={}",
            getattr(user, "sub", "anonymous"),
            resource,
            reason,
        )

    def log_auth_result(
        self,
        *,
        actor: str | None,
        role: str | None,
        success: bool,
        client_ip: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Record an authentication outcome (login success/failure)."""
        if not self._enabled:
            return
        record: dict[str, Any] = {
            "event": "auth.login",
            "actor": actor or "unknown",
            "role": role or "n/a",
            "success": success,
            "client_ip": client_ip or "unknown",
            "reason": reason,
            "ts": time.time(),
        }
        (logger.info if success else logger.warning)(
            "{event} | actor={actor} | role={role} | success={success} | client_ip={client_ip}",
            **record,
        )
