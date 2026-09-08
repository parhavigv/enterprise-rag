"""Tests for the SQLite-backed identity store.

Verifies the production record: durable accounts, case-insensitive unique
usernames, PBKDF2-hashed credentials, account lockout, the login-attempt
audit trail, and the settings-driven backend factory.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from app.auth.users import (
    JsonUserStore,
    SqliteUserStore,
    build_user_store,
    seed_demo_users,
)
from app.core.config import Settings


@pytest.fixture()
def store(tmp_path):
    s = SqliteUserStore(
        path=tmp_path / "users.db",
        max_failed_attempts=3,
        lockout_seconds=3600,
    )
    yield s
    s.close()


def _attempts_count(store: SqliteUserStore) -> int:
    cur = store._db.execute("SELECT COUNT(*) AS n FROM login_attempts")
    return cur.fetchone()["n"]


# --------------------------------------------------------------------- #
# Schema + identity provisioning
# --------------------------------------------------------------------- #
class TestSchema:
    def test_tables_are_created(self, tmp_path):
        SqliteUserStore(path=tmp_path / "users.db").close()
        con = sqlite3.connect(tmp_path / "users.db")
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"users", "login_attempts"}.issubset(tables)
        con.close()

    def test_is_empty_tracks_provisioning(self, store):
        assert store.is_empty() is True
        store.create_user("jane", "pw", "employee")
        assert store.is_empty() is False


class TestCreateAndLookup:
    def test_round_trip(self, store):
        user = store.create_user("jane", "pw", "employee", name="Jane")
        assert store.get("jane") == user
        assert store.get("JANE") == user  # case-insensitive lookup

    def test_unknown_role_rejected(self, store):
        with pytest.raises(ValueError, match="Unknown role"):
            store.create_user("jane", "pw", "ceo-made-up")

    def test_duplicate_username_rejected(self, store):
        store.create_user("jane", "pw", "employee")
        with pytest.raises(ValueError, match="already exists"):
            store.create_user("JANE", "pw2", "manager")

    def test_if_exists_skip_and_replace(self, store):
        store.create_user("jane", "pw", "employee")
        skipped = store.create_user("JANE", "pw", "manager", if_exists="skip")
        assert skipped.role == "employee"
        replaced = store.create_user("JANE", "pw", "manager", if_exists="replace")
        assert replaced.role == "manager"

    def test_departments_normalised_with_public_default(self, store):
        user = store.create_user("jane", "pw", "employee", departments=("ops",))
        assert user.departments == ("PUBLIC", "OPS")

    def test_users_sorted_and_hashed(self, store):
        seed_demo_users(store)
        names = [u.username for u in store.users]
        assert names == sorted(names)
        assert all(u.password_hash.startswith("$pbkdf2-sha256$") for u in store.users)


# --------------------------------------------------------------------- #
# Authentication lifecycle
# --------------------------------------------------------------------- #
class TestAuthenticate:
    def test_success_updates_last_login_and_records_attempt(self, store):
        store.create_user("jane", "pw", "employee")
        before = time.time()
        user = store.authenticate("jane", "pw", client_ip="10.0.0.7")
        assert user is not None and user.username == "jane"
        row = store._row_by_username("jane")
        assert row["last_login_at"] is not None and row["last_login_at"] >= before
        assert _attempts_count(store) == 1

    def test_failure_counts_and_no_last_login(self, store):
        store.create_user("jane", "pw", "employee")
        assert store.authenticate("jane", "wrong") is None
        row = store._row_by_username("jane")
        assert row["failed_attempts"] == 1
        assert row["last_login_at"] is None

    def test_unknown_user_returns_none_without_errors(self, store):
        assert store.authenticate("ghost", "pw") is None
        assert _attempts_count(store) == 1

    def test_client_ip_is_persisted(self, store):
        store.create_user("jane", "pw", "employee")
        store.authenticate("jane", "wrong", client_ip="10.1.2.3")
        row = store._db.execute("SELECT * FROM login_attempts").fetchone()
        assert row["client_ip"] == "10.1.2.3"
        assert row["success"] == 0


class TestLockout:
    def test_lockout_after_max_failures(self, store):
        store.create_user("jane", "pw", "employee")
        for _ in range(3):
            assert store.authenticate("jane", "wrong") is None
        row = store._row_by_username("jane")
        assert row["failed_attempts"] == 0  # counter reset when lockout triggers
        assert row["locked_until"] and row["locked_until"] > time.time()
        # Even the correct password is rejected while locked.
        assert store.authenticate("jane", "pw") is None

    def test_lockout_only_affects_target_account(self, store):
        store.create_user("jane", "pw", "employee", departments=("ops",))
        store.create_user("jo", "pw", "intern")
        for _ in range(3):
            store.authenticate("jane", "wrong")
        assert store.authenticate("jo", "pw") is not None

    def test_lock_releases_after_window(self, store):
        store.create_user("jane", "pw", "employee")
        for _ in range(3):
            store.authenticate("jane", "wrong")
        # Simulate the window elapsing.
        store._db.execute(
            "UPDATE users SET locked_until = ? WHERE username = 'jane'", (time.time() - 1,)
        )
        store._db.commit()
        assert store.authenticate("jane", "pw") is not None

    def test_disabled_account_never_succeeds(self, store):
        store.create_user("jane", "pw", "employee")
        store.set_active("jane", False)
        assert store.authenticate("jane", "pw") is None
        store.set_active("jane", True)
        assert store.authenticate("jane", "pw") is not None

    def test_disabled_and_locked_indistinguishable(self, store):
        store.create_user("evil", "pw", "employee")
        store.set_active("evil", False)
        store.create_user("bad", "pw", "intern")
        for _ in range(3):
            store.authenticate("bad", "wrong")
        # Callers see the same "invalid credentials" surface either way.
        assert store.authenticate("evil", "pw") is None
        assert store.authenticate("bad", "pw") is None


class TestAccountManagement:
    def test_set_password_clears_lock_and_rotates(self, store):
        store.create_user("jane", "pw", "employee")
        for _ in range(3):
            store.authenticate("jane", "wrong")
        store.set_password("jane", "newpw")
        assert store.authenticate("jane", "pw") is None
        assert store.authenticate("jane", "newpw") is not None

    def test_unknown_user_management_raises(self, store):
        with pytest.raises(ValueError, match="Unknown user"):
            store.set_password("ghost", "pw")
        with pytest.raises(ValueError, match="Unknown user"):
            store.set_active("ghost", False)


# --------------------------------------------------------------------- #
# Seeding + backend factory
# --------------------------------------------------------------------- #
class TestSeeding:
    def test_seed_is_idempotent(self, store):
        first = seed_demo_users(store)
        second = seed_demo_users(store)
        assert {u.username for u in first} == {u.username for u in second}
        assert store.is_empty() is False
        assert store.authenticate("alice", "admin-password!").role == "admin"
        assert store.authenticate("dave", "intern-password!").role == "intern"

    def test_seed_rejects_read_only_store(self):
        store = JsonUserStore()  # legacy registry is read-only
        with pytest.raises(TypeError, match="read-only"):
            seed_demo_users(store)


class TestFactory:
    def test_sqlite_default_path_uses_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        settings = Settings(
            environment="test",
            auth_user_store_backend="sqlite",
            auth_sqlite_path="",
            data_dir="./data",
        )
        store = build_user_store(settings)
        assert isinstance(store, SqliteUserStore)
        store.close()
        assert (tmp_path / "data" / "users.db").exists()

    def test_sqlite_respects_explicit_path(self, tmp_path):
        settings = Settings(
            environment="test",
            auth_user_store_backend="sqlite",
            auth_sqlite_path=str(tmp_path / "custom.db"),
        )
        store = build_user_store(settings)
        assert isinstance(store, SqliteUserStore)
        store.close()
        assert (tmp_path / "custom.db").exists()

    def test_json_backend_returns_legacy_store(self):
        store = build_user_store(Settings(environment="test", auth_user_store_backend="json"))
        assert isinstance(store, JsonUserStore)

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValueError, match="AUTH_USER_STORE_BACKEND"):
            Settings(auth_user_store_backend="ldap")

    def test_production_still_validates_secret_with_sqlite(self):
        Settings(
            environment="production",
            auth_jwt_secret="x" * 64,
            auth_user_store_backend="sqlite",
        )


# --------------------------------------------------------------------- #
# Admin CLI (scripts/manage_users.py)
# --------------------------------------------------------------------- #
def _cli_settings(tmp_path) -> Settings:
    return Settings(
        environment="development",
        auth_user_store_backend="sqlite",
        auth_sqlite_path=str(tmp_path / "users.db"),
    )


class TestManageUsersCli:
    def test_create_and_list(self, tmp_path, monkeypatch, capsys):
        import scripts.manage_users as cli

        monkeypatch.setattr(cli, "get_settings", lambda: _cli_settings(tmp_path))
        assert (
            cli.main(["create", "--username", "jen", "--role", "employee", "--password", "s3cr3t!"])
            == 0
        )
        assert cli.main(["list"]) == 0
        out = capsys.readouterr().out
        assert "jen" in out and "employee" in out

    def test_duplicate_create_returns_nonzero(self, tmp_path, monkeypatch):
        import scripts.manage_users as cli

        monkeypatch.setattr(cli, "get_settings", lambda: _cli_settings(tmp_path))
        assert (
            cli.main(["create", "--username", "jen", "--role", "employee", "--password", "s3cr3t!"])
            == 0
        )
        assert (
            cli.main(["create", "--username", "jen", "--role", "employee", "--password", "x"]) == 1
        )

    def test_unknown_command_field_rejected(self, tmp_path, monkeypatch):
        import scripts.manage_users as cli

        monkeypatch.setattr(cli, "get_settings", lambda: _cli_settings(tmp_path))
        assert cli.main(["create", "--username", "jim", "--role", "ceo", "--password", "x"]) == 1

    def test_disable_and_enable_via_cli(self, tmp_path, monkeypatch):
        import scripts.manage_users as cli

        monkeypatch.setattr(cli, "get_settings", lambda: _cli_settings(tmp_path))
        cli.main(["create", "--username", "jen", "--role", "employee", "--password", "s3cr3t!"])
        assert cli.main(["disable", "--username", "jen"]) == 0
        store = build_user_store(_cli_settings(tmp_path))
        assert store.authenticate("jen", "s3cr3t!") is None
        store.close()
        assert cli.main(["enable", "--username", "jen"]) == 0
