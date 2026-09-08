"""Permission-model data types for the RBAC layer.

The model is deliberately simple but sound:

    - Every document chunk is tagged with ACL metadata during ingestion:
      ``department`` (which team owns it), ``clearance_level`` (how
      sensitive it is) and ``owner`` (the responsible principal).
    - Every user is assigned a single :class:`Role` that maps to a maximum
      :class:`ClearanceLevel` plus a set of visible departments.
    - A user may read a chunk iff its clearance_level <= the user's max and
      its department is in the user's visible departments (or the user owns
      it, or the user is an admin).

Enforcement happens *pre-retrieval* (before RRF fusion / re-ranking) so
sensitive chunks never even enter the candidate pool the LLM sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any


class ClearanceLevel(IntEnum):
    """Ordered sensitivity ladder; higher = more sensitive."""

    PUBLIC = auto()
    INTERNAL = auto()
    CONFIDENTIAL = auto()
    SECRET = auto()

    @classmethod
    def parse(cls, value: str | int | None) -> ClearanceLevel:
        """Coerce a raw string/int into a valid clearance level."""
        if value is None:
            return cls.PUBLIC
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        normalized = str(value).strip().upper()
        aliases = {
            "PUBLIC": cls.PUBLIC,
            "OPEN": cls.PUBLIC,
            "INTERNAL": cls.INTERNAL,
            "CONFIDENTIAL": cls.CONFIDENTIAL,
            "RESTRICTED": cls.CONFIDENTIAL,
            "SECRET": cls.SECRET,
            "TOP_SECRET": cls.SECRET,
            "TOP-SECRET": cls.SECRET,
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unknown clearance level '{value}'. Must be one of: "
                "public, internal, confidential, secret"
            )
        return aliases[normalized]


class Role:
    """A named role with a max clearance and visible departments."""

    def __init__(
        self,
        name: str,
        max_clearance: ClearanceLevel,
        departments: set[str] | None = None,
        *,
        is_admin: bool = False,
        can_audit: bool = False,
    ) -> None:
        self.name = name
        self.max_clearance = max_clearance
        self.departments = departments or {"PUBLIC"}
        self.is_admin = is_admin
        self.can_audit = can_audit

    def can_access(self, acl: ACLMetadata) -> bool:
        """True if the role may read a chunk with the given ACL metadata."""
        if self.is_admin:
            return True
        if acl.clearance_level > self.max_clearance:
            return False
        return acl.department in self.departments


# --------------------------------------------------------------------- #
# Built-in roles
# --------------------------------------------------------------------- #
ROLE_INTERN = Role(
    "intern",
    max_clearance=ClearanceLevel.PUBLIC,
    departments={"PUBLIC"},
)
ROLE_EMPLOYEE = Role(
    "employee",
    max_clearance=ClearanceLevel.INTERNAL,
    departments={"PUBLIC", "ENGINEERING", "PRODUCT", "OPS"},
)
ROLE_MANAGER = Role(
    "manager",
    max_clearance=ClearanceLevel.CONFIDENTIAL,
    departments={"PUBLIC", "ENGINEERING", "PRODUCT", "OPS", "FINANCE", "HR"},
)
ROLE_EXECUTIVE = Role(
    "executive",
    max_clearance=ClearanceLevel.SECRET,
    departments={"PUBLIC", "ENGINEERING", "PRODUCT", "OPS", "FINANCE", "HR", "LEGAL"},
)
ROLE_ADMIN = Role(
    "admin",
    max_clearance=ClearanceLevel.SECRET,
    departments=None,  # unrestricted
    is_admin=True,
    can_audit=True,
)

ROLES: dict[str, Role] = {
    "intern": ROLE_INTERN,
    "employee": ROLE_EMPLOYEE,
    "manager": ROLE_MANAGER,
    "executive": ROLE_EXECUTIVE,
    "admin": ROLE_ADMIN,
}


def get_role(name: str | None) -> Role:
    """Resolve a role name to a :class:`Role`; defaults to intern."""
    if name is None:
        return ROLE_INTERN
    return ROLES.get(name.strip().lower(), ROLE_INTERN)


@dataclass(slots=True)
class ACLMetadata:
    """Document-level / chunk-level access-control metadata."""

    department: str = "PUBLIC"
    clearance_level: ClearanceLevel = ClearanceLevel.PUBLIC
    owner: str | None = None

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> ACLMetadata:
        """Coerce a stored metadata dict into ACL values (filtering path).

        Unknown/malformed clearance values default to PUBLIC (the most
        restrictive reading) so a junk metadata field can never crash a query.
        Ingestion-time validation is strict (:meth:`merge_defaults`); this is
        deliberately the tolerant, safe-to-degrade reader.
        """
        try:
            level = ClearanceLevel.parse(metadata.get("clearance_level"))
        except ValueError:
            level = ClearanceLevel.PUBLIC
        return cls(
            department=str(metadata.get("department") or "PUBLIC").upper(),
            clearance_level=level,
            owner=metadata.get("owner"),
        )

    @classmethod
    def merge_defaults(
        cls, metadata: dict[str, Any], defaults: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Attach safe-to-persist ACL keys to a metadata dict, applying defaults.

        ``defaults`` may carry ``department`` / ``clearance_level`` / ``owner``
        supplied at ingestion time.

        Raises:
            ValueError: an explicit clearance level is not one of the known
                levels. Silently downgrading an unparseable value to PUBLIC
                would *loosen* a document's access policy - failing fast is the
                safe behaviour.
        """
        deps = defaults or {}
        out = dict(metadata)
        out["department"] = str(deps.get("department") or out.pop("department", "PUBLIC")).upper()
        raw_level = deps.get("clearance_level") or out.pop("clearance_level", None)
        cl = ClearanceLevel.PUBLIC if raw_level is None else ClearanceLevel.parse(raw_level)
        out["clearance_level"] = cl.name
        out["owner"] = out.get("owner") or deps.get("owner")
        return out


@dataclass(slots=True)
class AuthUser:
    """Authenticated principal extracted from a JWT."""

    sub: str
    name: str
    role: Role
    departments: set[str] = field(default_factory=set)

    @property
    def username(self) -> str:
        return self.sub

    def can_access(self, acl: ACLMetadata) -> bool:
        """Shortcut: enforce the user's role clearance + user-scoped departments."""
        if self.role.is_admin:
            return True
        if acl.clearance_level > self.role.max_clearance:
            return False
        effective_depts = self.departments if self.departments else (self.role.departments or set())
        return acl.department in effective_depts
