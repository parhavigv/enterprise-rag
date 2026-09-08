"""Security-hardening tests for the RBAC layer.

Covers the production-grade details that turn a demo auth layer into a
defensible one:

    * credential hashing (PBKDF2, per-user salt, constant-time compare)
    * JWT claim validation (audience, not-before, required claims)
    * fail-fast production configuration (no empty/weak JWT secret)
    * generic 401 responses that never leak decode internals
    * brute-force rate limiting on credential endpoints
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth.passwords import (
    DEFAULT_ITERATIONS,
    PasswordSchemeError,
    _parse,
    hash_password,
    verify_password,
)
from app.auth.service import AuthService, TokenVerificationError
from app.auth.users import UserStore
from app.core.config import Settings


# --------------------------------------------------------------------- #
# 1. Password hashing
# --------------------------------------------------------------------- #
class TestPasswordHashing:
    def test_round_trip(self):
        assert verify_password("hunter2!", hash_password("hunter2!"))

    def test_wrong_password_rejected(self):
        assert verify_password("not-the-password", hash_password("hunter2!")) is False

    def test_hashes_are_unique_per_user(self):
        a, b = hash_password("same-password"), hash_password("same-password")
        assert a != b  # random per-user salt -> no shared digests

    def test_scheme_is_versioned_and_self_describing(self):
        hashed = hash_password("x", iterations=1000, salt="0" * 32)
        algo, salt, iterations, digest = _parse(hashed)
        assert algo == "pbkdf2-sha256"
        assert salt == "0" * 32
        assert iterations == 1000
        assert len(digest) == 64  # sha256 hex

    def test_verify_uses_stored_iterations(self):
        hashed = hash_password("x", iterations=5000)
        assert verify_password("x", hashed)

    def test_malformed_hash_returns_false_not_exception(self):
        assert verify_password("x", "not-a-hash") is False
        assert verify_password("x", "$md5$i=1$abc$def") is False

    def test_parse_rejects_malformed(self):
        for bad in ["", "abc", "$pbkdf2-sha256$i=x$s$d", "a$b$c$d$e$f"]:
            with pytest.raises(PasswordSchemeError):
                _parse(bad)

    def test_excessively_long_password_rejected(self):
        with pytest.raises(ValueError):
            hash_password("p" * 5000)

    def test_default_iteration_count_meets_owasp_floor(self):
        assert DEFAULT_ITERATIONS >= 600_000


# --------------------------------------------------------------------- #
# 2. JWT claim validation
# --------------------------------------------------------------------- #
class TestJwtClaims:
    def test_audience_is_bound_to_token(self):
        svc = AuthService(secret="s", audience="api")
        token = svc.issue_token("alice", "manager")
        other = AuthService(secret="s", audience="admin-console")
        with pytest.raises(TokenVerificationError):
            other.verify(token)

    def test_token_carries_standard_claims(self):
        svc = AuthService(secret="s", expiry_seconds=60)
        token = svc.issue_token("alice", "manager")
        payload = svc.decode(token)
        assert {"iss", "aud", "iat", "nbf", "exp", "jti"}.issubset(payload)
        assert payload["aud"] == "enterprise-rag"

    def test_future_notbefore_token_is_rejected(self):
        import time

        import jwt as _jwt

        svc = AuthService(secret="s")
        token = svc.issue_token("alice", "manager")
        payload = _jwt.decode(token, "s", algorithms=["HS256"], audience="enterprise-rag")
        payload["nbf"] = int(time.time()) + 3600
        future = _jwt.encode(payload, "s", algorithm="HS256")
        with pytest.raises(TokenVerificationError):
            svc.verify(future)

    def test_missing_required_claims_are_rejected(self):
        import jwt as _jwt

        svc = AuthService(secret="s")
        token = svc.issue_token("alice", "manager")
        payload = _jwt.decode(token, "s", algorithms=["HS256"], audience="enterprise-rag")
        payload.pop("sub")
        tampered = _jwt.encode(payload, "s", algorithm="HS256")
        with pytest.raises(TokenVerificationError):
            svc.verify(tampered)

    def test_verify_generic_error_never_exposes_internals(self):
        svc = AuthService(secret="s")
        with pytest.raises(TokenVerificationError) as excinfo:
            svc.verify("garbage.token.value")
        assert "garbage" not in str(excinfo.value)


# --------------------------------------------------------------------- #
# 3. Fail-fast production configuration
# --------------------------------------------------------------------- #
class TestProductionConfigGuard:
    def test_production_requires_secret(self):
        with pytest.raises(ValueError, match="AUTH_JWT_SECRET"):
            Settings(environment="production", auth_jwt_secret="")

    def test_production_rejects_short_secret(self):
        with pytest.raises(ValueError, match="AUTH_JWT_SECRET"):
            Settings(environment="production", auth_jwt_secret="short")

    def test_production_accepts_strong_secret(self):
        s = Settings(environment="production", auth_jwt_secret="x" * 64)
        assert s.auth_jwt_secret == "x" * 64

    def test_development_allows_ephemeral_secret(self):
        assert Settings(environment="development", auth_jwt_secret="").auth_jwt_secret == ""


# --------------------------------------------------------------------- #
# 4. User store: credential verification + timing equalisation
# --------------------------------------------------------------------- #
class TestUserStoreHardening:
    @pytest.fixture(scope="class")
    def store(self):
        users_path = Path(__file__).resolve().parent.parent / "app" / "auth" / "users.json"
        return UserStore(path=users_path)

    def test_stored_hashes_use_pbkdf2(self, store):
        for user in store.users:
            assert user.password_hash.startswith("$pbkdf2-sha256$")

    def test_valid_credentials(self, store):
        assert store.authenticate("bob", "manager-password!").role == "manager"

    def test_bad_password_and_unknown_user_both_return_none(self, store):
        assert store.authenticate("carol", "wrong") is None
        assert store.authenticate("nobody", "wrong") is None


# --------------------------------------------------------------------- #
# 5. API: generic 401s + rate limiting
# --------------------------------------------------------------------- #
class TestApiHardening:
    @pytest.fixture()
    def client(self):
        from types import SimpleNamespace

        from app.api.deps import get_container
        from app.auth.service import AuthService
        from app.main import create_app

        class _Container:
            settings = SimpleNamespace(
                api_prefix="/api/v1",
                request_id_header="X-Request-ID",
                cors_origin_list=["*"],
                environment="test",
                app_name="enterprise-rag",
                debug=False,
                log_level="WARNING",
                log_format="console",
                chroma_collection="enterprise_rag",
                bm25_index_path="./data/bm25_index.pkl",
                llm_model="llama3.1",
                embed_model="nomic-embed-text",
                auth_enabled=True,
                auth_jwt_secret="api-test-secret",
                auth_jwt_expiry_seconds=3600,
                auth_jwt_algorithm="HS256",
                auth_issuer="enterprise-rag",
                auth_audience="enterprise-rag",
                auth_rate_limit_enabled=True,
                auth_rate_limit_max_requests=3,
                auth_rate_limit_window_seconds=60,
            )

            def auth(self):
                return AuthService(secret="api-test-secret", expiry_seconds=3600)

            def user_store(self):
                users_path = Path(__file__).resolve().parent.parent / "app" / "auth" / "users.json"
                return UserStore(path=users_path)

            def audit(self):
                return None

        app = create_app(settings=_Container.settings)
        fake = _Container()
        app.dependency_overrides[get_container] = lambda: fake
        with TestClient(app) as c:
            c.app.state.container = fake
            yield c

    def test_invalid_token_401_is_generic_with_challenge(self, client):
        r = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer junk.token.value"})
        assert r.status_code == 401
        assert r.headers.get("WWW-Authenticate", "").startswith("Bearer")
        assert "junk" not in r.json()["detail"].lower()
        assert "expired" in r.json()["detail"].lower()

    def test_missing_token_401_has_challenge(self, client):
        r = client.get("/api/v1/auth/me")
        assert r.status_code == 401
        assert r.headers.get("WWW-Authenticate", "").startswith("Bearer")

    def test_login_is_rate_limited(self, client):
        for _ in range(3):
            r = client.post(
                "/api/v1/auth/login",
                json={"username": "carol", "password": "employee-password!"},
            )
            assert r.status_code == 200
        r = client.post(
            "/api/v1/auth/login",
            json={"username": "carol", "password": "employee-password!"},
        )
        assert r.status_code == 429
        assert r.headers.get("Retry-After")

    def test_rate_limited_bad_credentials_still_throttled(self, client):
        # Brute-force attempts count against the same budget.
        for _ in range(3):
            client.post("/api/v1/auth/login", json={"username": "carol", "password": "wrong"})
        r = client.post("/api/v1/auth/login", json={"username": "carol", "password": "wrong"})
        assert r.status_code == 429
