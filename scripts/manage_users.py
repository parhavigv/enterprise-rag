"""Admin CLI for the identity store.

Manages accounts in the configured :class:`SqliteUserStore` (or any store
built by ``build_user_store``). Passwords are never accepted via plain CLI
arguments - they are prompted with ``getpass`` so they stay off the shell
history and process list.

Examples::

    python -m scripts.manage_users list
    python -m scripts.manage_users seed-demo
    python -m scripts.manage_users create --username jen --role employee --name "Jen Doe"
    python -m scripts.manage_users set-password --username jen
    python -m scripts.manage_users disable --username jen
    python -m scripts.manage_users enable --username jen

Settings (``ENVIRONMENT``, ``AUTH_USER_STORE_BACKEND``, ``AUTH_SQLITE_PATH``,
lockout parameters) are read from the environment / ``.env`` exactly as the
application does.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from collections.abc import Sequence

from app.auth.users import build_user_store, seed_demo_users
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _store():
    return build_user_store(get_settings())


def _prompt_password(label: str = "Password: ") -> str:
    try:
        return getpass.getpass(label)
    except (EOFError, getpass.GetPassWarning) as exc:
        raise SystemExit(f"Could not read password interactively: {exc}") from exc


def _cmd_init(_args: argparse.Namespace) -> int:
    store = _store()
    print(f"Identity store ready: {store.__class__.__name__}")
    return 0


def _cmd_seed(_args: argparse.Namespace) -> int:
    store = _store()
    created = seed_demo_users(store)
    print(f"Demo identities ready: {', '.join(u.username for u in created)}")
    return 0


def _cmd_list(_args: argparse.Namespace) -> int:
    store = _store()
    rows = store.users
    if not rows:
        print("(no users)")
        return 0
    width = max(len(u.username) for u in rows)
    for u in rows:
        active = "active"
        if getattr(u, "is_active", True) is False:
            active = "disabled"
        print(f"{u.username:<{width}}  {u.role:<10}  {u.name}  {active}")
    return 0


def _cmd_create(args: argparse.Namespace) -> int:
    store = _store()
    password = args.password or _prompt_password(f"Password for {args.username}: ")
    departments = tuple(d.strip().upper() for d in (args.departments or "").split(",") if d.strip())
    user = store.create_user(
        args.username,
        password,
        args.role,
        name=args.name,
        departments=departments,
        if_exists=args.if_exists,
    )
    print(f"Created {user.username} ({user.role})")
    return 0


def _cmd_set_password(args: argparse.Namespace) -> int:
    store = _store()
    password = args.password or _prompt_password(f"New password for {args.username}: ")
    store.set_password(args.username, password)
    print(f"Password rotated for {args.username}")
    return 0


def _cmd_enable(args: argparse.Namespace) -> int:
    _store().set_active(args.username, True)
    print(f"Enabled {args.username}")
    return 0


def _cmd_disable(args: argparse.Namespace) -> int:
    _store().set_active(args.username, False)
    print(f"Disabled {args.username}")
    return 0


COMMANDS = {
    "init": _cmd_init,
    "seed-demo": _cmd_seed,
    "list": _cmd_list,
    "create": _cmd_create,
    "set-password": _cmd_set_password,
    "enable": _cmd_enable,
    "disable": _cmd_disable,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create the identity-store schema")
    sub.add_parser("seed-demo", help="Provision the four demo identities (skip-if-exists)")
    sub.add_parser("list", help="List accounts")

    p_create = sub.add_parser("create", help="Create an account")
    p_create.add_argument("--username", required=True)
    p_create.add_argument("--role", required=True, help="intern|employee|manager|executive|admin")
    p_create.add_argument("--name", default=None)
    p_create.add_argument("--departments", default="", help="Comma-separated (e.g. PUBLIC,OPS)")
    p_create.add_argument("--password", default=None, help="If omitted you are prompted")
    p_create.add_argument("--if-exists", default="error", choices=("error", "skip", "replace"))

    p_pass = sub.add_parser("set-password", help="Rotate an account password")
    p_pass.add_argument("--username", required=True)
    p_pass.add_argument("--password", default=None, help="If omitted you are prompted")

    for cmd_name in ("enable", "disable"):
        p = sub.add_parser(cmd_name, help=f"{cmd_name.capitalize()} an account")
        p.add_argument("--username", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (ValueError, OSError) as exc:
        logger.error("manage_users failed: {}", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
