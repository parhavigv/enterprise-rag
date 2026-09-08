"""RBAC / access-control layer for Enterprise RAG.

Provides:
    - Role-based clearance model (public < internal < confidential < secret)
    - Document-level ACL metadata (department, clearance_level, owner)
    - Pre-retrieval filtering based on the requesting user's role
    - JWT authentication for API endpoints
    - Audit logging of access-control decisions
"""

# NOTE: ``dependencies`` is intentionally NOT imported here to avoid a heavy
# import chain (dependencies -> app.api.deps -> services.container). Route
# modules import ``get_current_user`` / ``require_role`` directly.
from app.auth.models import (
    ACLMetadata,
    AuthUser,
    ClearanceLevel,
    Role,
)
from app.auth.service import AuthService

__all__ = [
    "ACLMetadata",
    "AuthService",
    "AuthUser",
    "ClearanceLevel",
    "Role",
]
