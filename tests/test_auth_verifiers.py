"""
Tests for server-side credential verifiers and MultiProtocolAuthBackend.

Covers: JWTTokenVerifier, ApiKeyVerifier, MultiProtocolAuthBackend,
401/403 differentiation, exempt paths, WWW-Authenticate generation,
Flask integration hook.
"""

import time
from typing import Dict, Optional, Set

import pytest

from python_a2a.auth.verifiers import (
    AccessPrincipal,
    ApiKeyVerifier,
    CredentialVerifier,
    InsufficientScopeError,
    JWTTokenVerifier,
    MultiProtocolAuthBackend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jwt_token(
    claims: dict,
    key: str = "test-secret",
    algorithm: str = "HS256",
) -> str:
    """Create a signed JWT for testing."""
    import jwt

    return jwt.encode(claims, key, algorithm=algorithm)


# ---------------------------------------------------------------------------
# JWTTokenVerifier
# ---------------------------------------------------------------------------


class TestJWTTokenVerifier:
    def test_valid_token(self):
        key = "test-secret"
        claims = {
            "sub": "client-123",
            "scope": "a2a:call read",
            "iss": "https://auth.example.com",
            "exp": time.time() + 3600,
        }
        token = _make_jwt_token(claims, key=key)

        verifier = JWTTokenVerifier(
            secret_or_public_key=key,
            algorithms=["HS256"],
            issuer="https://auth.example.com",
        )
        principal = verifier.verify({"Authorization": f"Bearer {token}"})
        assert principal is not None
        assert principal.subject == "client-123"
        assert principal.scopes == {"a2a:call", "read"}
        assert principal.issuer == "https://auth.example.com"

    def test_expired_token(self):
        key = "test-secret"
        claims = {"sub": "c", "exp": time.time() - 10}
        token = _make_jwt_token(claims, key=key)

        verifier = JWTTokenVerifier(secret_or_public_key=key, algorithms=["HS256"])
        principal = verifier.verify({"Authorization": f"Bearer {token}"})
        assert principal is None

    def test_invalid_signature(self):
        token = _make_jwt_token(
            {"sub": "c", "exp": time.time() + 3600},
            key="wrong-key",
        )
        verifier = JWTTokenVerifier(
            secret_or_public_key="correct-key", algorithms=["HS256"],
        )
        principal = verifier.verify({"Authorization": f"Bearer {token}"})
        assert principal is None

    def test_missing_authorization_header(self):
        verifier = JWTTokenVerifier(
            secret_or_public_key="key", algorithms=["HS256"],
        )
        assert verifier.verify({}) is None

    def test_non_bearer_scheme(self):
        verifier = JWTTokenVerifier(
            secret_or_public_key="key", algorithms=["HS256"],
        )
        assert verifier.verify({"Authorization": "Basic abc123"}) is None

    def test_get_challenge(self):
        verifier = JWTTokenVerifier(
            secret_or_public_key="k", algorithms=["HS256"], realm="my-a2a",
        )
        assert verifier.get_challenge() == 'Bearer realm="my-a2a"'

    def test_audience_validation(self):
        key = "test-secret"
        claims = {"sub": "c", "aud": "agent-1", "exp": time.time() + 3600}
        token = _make_jwt_token(claims, key=key)

        verifier = JWTTokenVerifier(
            secret_or_public_key=key, algorithms=["HS256"], audience="agent-1",
        )
        assert verifier.verify({"Authorization": f"Bearer {token}"}) is not None

        # Wrong audience
        verifier_wrong = JWTTokenVerifier(
            secret_or_public_key=key, algorithms=["HS256"], audience="agent-2",
        )
        assert verifier_wrong.verify({"Authorization": f"Bearer {token}"}) is None

    def test_no_scope_claim(self):
        key = "test-secret"
        claims = {"sub": "c", "exp": time.time() + 3600}
        token = _make_jwt_token(claims, key=key)

        verifier = JWTTokenVerifier(
            secret_or_public_key=key, algorithms=["HS256"],
        )
        principal = verifier.verify({"Authorization": f"Bearer {token}"})
        assert principal is not None
        assert principal.scopes == set()


# ---------------------------------------------------------------------------
# ApiKeyVerifier
# ---------------------------------------------------------------------------


class TestApiKeyVerifier:
    def test_valid_key(self):
        verifier = ApiKeyVerifier(
            valid_keys={"key-abc-123"},
            scopes={"read"},
        )
        principal = verifier.verify({"X-API-Key": "key-abc-123"})
        assert principal is not None
        assert principal.scopes == {"read"}

    def test_invalid_key(self):
        verifier = ApiKeyVerifier(valid_keys={"key-abc-123"})
        assert verifier.verify({"X-API-Key": "wrong-key"}) is None

    def test_missing_header(self):
        verifier = ApiKeyVerifier(valid_keys={"k"})
        assert verifier.verify({}) is None

    def test_custom_header_name(self):
        verifier = ApiKeyVerifier(
            valid_keys={"k"}, header_name="X-Custom-Auth",
        )
        assert verifier.verify({"X-Custom-Auth": "k"}) is not None
        assert verifier.verify({"X-API-Key": "k"}) is None

    def test_case_insensitive_header(self):
        verifier = ApiKeyVerifier(valid_keys={"k"})
        assert verifier.verify({"x-api-key": "k"}) is not None

    def test_get_challenge_returns_none(self):
        verifier = ApiKeyVerifier(valid_keys={"k"})
        assert verifier.get_challenge() is None


# ---------------------------------------------------------------------------
# MultiProtocolAuthBackend
# ---------------------------------------------------------------------------


class TestMultiProtocolAuthBackend:
    def _jwt_verifier(self, key="secret"):
        return JWTTokenVerifier(
            secret_or_public_key=key,
            algorithms=["HS256"],
        )

    def _apikey_verifier(self, keys=None):
        return ApiKeyVerifier(valid_keys=keys or {"test-key"})

    def test_jwt_verifier_first_match(self):
        key = "secret"
        token = _make_jwt_token(
            {"sub": "c", "scope": "a2a:call", "exp": time.time() + 3600},
            key=key,
        )
        backend = MultiProtocolAuthBackend(
            verifiers=[self._jwt_verifier(key), self._apikey_verifier()],
        )
        principal = backend.verify(
            headers={"Authorization": f"Bearer {token}"},
            path="/tasks/send",
        )
        assert principal is not None
        assert principal.subject == "c"

    def test_apikey_fallback(self):
        backend = MultiProtocolAuthBackend(
            verifiers=[self._jwt_verifier(), self._apikey_verifier()],
        )
        principal = backend.verify(
            headers={"X-API-Key": "test-key"},
            path="/tasks/send",
        )
        assert principal is not None
        assert "apikey:" in principal.subject

    def test_no_match_returns_none(self):
        backend = MultiProtocolAuthBackend(
            verifiers=[self._jwt_verifier(), self._apikey_verifier()],
        )
        principal = backend.verify(headers={}, path="/tasks/send")
        assert principal is None

    def test_exempt_path_bypasses_auth(self):
        backend = MultiProtocolAuthBackend(
            verifiers=[self._jwt_verifier()],
        )
        # Default exempt path
        principal = backend.verify(
            headers={}, path="/.well-known/agent.json",
        )
        assert principal is not None
        assert principal.subject == "anonymous"

    def test_custom_exempt_paths(self):
        backend = MultiProtocolAuthBackend(
            verifiers=[self._jwt_verifier()],
            exempt_paths={"/health", "/ready"},
        )
        assert backend.verify(headers={}, path="/health") is not None
        # Default exempts no longer apply
        assert backend.verify(headers={}, path="/.well-known/agent.json") is None

    def test_required_scopes_pass(self):
        key = "secret"
        token = _make_jwt_token(
            {"sub": "c", "scope": "a2a:call read", "exp": time.time() + 3600},
            key=key,
        )
        backend = MultiProtocolAuthBackend(verifiers=[self._jwt_verifier(key)])
        principal = backend.verify(
            headers={"Authorization": f"Bearer {token}"},
            path="/tasks/send",
            required_scopes=["a2a:call"],
        )
        assert principal is not None

    def test_required_scopes_fail_raises_insufficient_scope(self):
        key = "secret"
        token = _make_jwt_token(
            {"sub": "c", "scope": "read", "exp": time.time() + 3600},
            key=key,
        )
        backend = MultiProtocolAuthBackend(verifiers=[self._jwt_verifier(key)])
        with pytest.raises(InsufficientScopeError) as exc_info:
            backend.verify(
                headers={"Authorization": f"Bearer {token}"},
                path="/tasks/send",
                required_scopes=["a2a:call", "write"],
            )
        assert "a2a:call" in str(exc_info.value)

    def test_get_www_authenticate_jwt_only(self):
        backend = MultiProtocolAuthBackend(
            verifiers=[self._jwt_verifier()],
        )
        assert backend.get_www_authenticate() == 'Bearer realm="a2a"'

    def test_get_www_authenticate_mixed(self):
        backend = MultiProtocolAuthBackend(
            verifiers=[self._jwt_verifier(), self._apikey_verifier()],
        )
        # ApiKey returns None, so only Bearer challenge
        assert backend.get_www_authenticate() == 'Bearer realm="a2a"'


# ---------------------------------------------------------------------------
# Flask integration
# ---------------------------------------------------------------------------


class TestFlaskAuthIntegration:
    def _make_flask_app(self, backend):
        """Create a minimal Flask app with auth backend."""
        from python_a2a import A2AServer, AgentCard

        card = AgentCard(
            name="TestAgent",
            description="Test",
            url="http://localhost:5000",
        )
        agent = A2AServer(agent_card=card)

        from python_a2a.server.http import create_flask_app

        return create_flask_app(agent, auth_backend=backend)

    def test_exempt_path_no_auth_required(self):
        key = "secret"
        backend = MultiProtocolAuthBackend(
            verifiers=[JWTTokenVerifier(secret_or_public_key=key, algorithms=["HS256"])],
        )
        app = self._make_flask_app(backend)
        client = app.test_client()

        response = client.get("/.well-known/agent.json")
        # Should not be 401 (exempt path)
        assert response.status_code != 401

    def test_protected_path_no_credentials_returns_401(self):
        key = "secret"
        backend = MultiProtocolAuthBackend(
            verifiers=[JWTTokenVerifier(secret_or_public_key=key, algorithms=["HS256"])],
        )
        app = self._make_flask_app(backend)
        client = app.test_client()

        response = client.post(
            "/",
            json={"jsonrpc": "2.0", "method": "tasks/send", "params": {}, "id": 1},
        )
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers
        assert "Bearer" in response.headers["WWW-Authenticate"]

    def test_protected_path_valid_jwt_passes(self):
        key = "secret"
        token = _make_jwt_token(
            {"sub": "c", "exp": time.time() + 3600}, key=key,
        )
        backend = MultiProtocolAuthBackend(
            verifiers=[JWTTokenVerifier(secret_or_public_key=key, algorithms=["HS256"])],
        )
        app = self._make_flask_app(backend)
        client = app.test_client()

        response = client.post(
            "/",
            json={"jsonrpc": "2.0", "method": "tasks/send", "params": {}, "id": 1},
            headers={"Authorization": f"Bearer {token}"},
        )
        # Should pass auth (actual endpoint logic may produce other status codes)
        assert response.status_code != 401

    def test_no_auth_backend_backward_compat(self):
        """Without auth_backend, all requests are accepted."""
        from python_a2a import A2AServer, AgentCard
        from python_a2a.server.http import create_flask_app

        card = AgentCard(
            name="TestAgent",
            description="Test",
            url="http://localhost:5000",
        )
        agent = A2AServer(agent_card=card)
        app = create_flask_app(agent)  # No auth_backend
        client = app.test_client()

        response = client.post(
            "/",
            json={"jsonrpc": "2.0", "method": "tasks/send", "params": {}, "id": 1},
        )
        # Should not be 401 (no auth enforcement)
        assert response.status_code != 401

    def test_insufficient_scope_returns_403(self):
        """Valid credentials but insufficient scopes -> 403."""
        key = "secret"
        token = _make_jwt_token(
            {"sub": "c", "scope": "read", "exp": time.time() + 3600}, key=key,
        )

        class ScopeRequiringBackend(MultiProtocolAuthBackend):
            def verify(self, headers, path, required_scopes=None):
                return super().verify(
                    headers, path, required_scopes=["a2a:call"],
                )

        backend = ScopeRequiringBackend(
            verifiers=[JWTTokenVerifier(secret_or_public_key=key, algorithms=["HS256"])],
        )
        app = self._make_flask_app(backend)
        client = app.test_client()

        response = client.post(
            "/",
            json={"jsonrpc": "2.0", "method": "tasks/send", "params": {}, "id": 1},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403
        assert "insufficient_scope" in response.headers.get("WWW-Authenticate", "")
