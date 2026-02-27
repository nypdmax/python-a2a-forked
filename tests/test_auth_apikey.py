"""
Tests for ApiKeyProtocol (client-side) and ApiKeyVerifier Bearer fallback
(server-side).

Covers:
  - ApiKeyProtocol.authenticate() / prepare_headers()
  - Custom header_name resolution (local_config > SecurityScheme.name > default)
  - ApiKeyVerifier dedicated header, Bearer fallback, both-missing rejection
  - MultiProtocolAuthBackend ordering: JWT Bearer not consumed by ApiKeyVerifier
"""

import time

import pytest

from python_a2a.auth.protocol import ApiKeyCredentials, AuthContext
from python_a2a.auth.protocols.apikey import ApiKeyProtocol
from python_a2a.auth.verifiers import (
    AccessPrincipal,
    ApiKeyVerifier,
    JWTTokenVerifier,
    MultiProtocolAuthBackend,
)
from python_a2a.exceptions import A2AAuthenticationError
from python_a2a.models.agent import SecurityScheme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_apikey_context(
    api_key: str = "test-key-12345678",
    header_name: str | None = None,
    scheme_name: str | None = None,
) -> AuthContext:
    """Build an AuthContext for apiKey authentication."""
    local_config: dict = {"api_key": api_key}
    if header_name is not None:
        local_config["header_name"] = header_name

    scheme = SecurityScheme(type="apiKey")
    if scheme_name is not None:
        scheme.name = scheme_name

    return AuthContext(
        agent_url="https://agent.example.com",
        security_scheme=scheme,
        local_config=local_config,
    )


HMAC_TEST_KEY = "test-secret-key-that-is-at-least-32-bytes-long"


def _make_jwt_token(
    claims: dict,
    key: str = HMAC_TEST_KEY,
    algorithm: str = "HS256",
) -> str:
    import jwt

    return jwt.encode(claims, key, algorithm=algorithm)


# ---------------------------------------------------------------------------
# ApiKeyProtocol – client-side
# ---------------------------------------------------------------------------


class TestApiKeyProtocolAuthenticate:
    def test_returns_credentials_with_default_header(self):
        ctx = _make_apikey_context()
        proto = ApiKeyProtocol()
        creds = proto.authenticate(ctx)

        assert isinstance(creds, ApiKeyCredentials)
        assert creds.api_key == "test-key-12345678"
        assert creds.header_name == "X-API-Key"

    def test_header_from_local_config_overrides_all(self):
        ctx = _make_apikey_context(
            header_name="X-Custom-Key", scheme_name="X-From-Scheme",
        )
        proto = ApiKeyProtocol()
        creds = proto.authenticate(ctx)
        assert creds.header_name == "X-Custom-Key"

    def test_header_from_security_scheme_name(self):
        ctx = _make_apikey_context(scheme_name="X-Scheme-Key")
        proto = ApiKeyProtocol()
        creds = proto.authenticate(ctx)
        assert creds.header_name == "X-Scheme-Key"

    def test_missing_api_key_raises(self):
        ctx = AuthContext(
            agent_url="https://agent.example.com",
            security_scheme=SecurityScheme(type="apiKey"),
            local_config={},
        )
        proto = ApiKeyProtocol()
        with pytest.raises(A2AAuthenticationError, match="api_key"):
            proto.authenticate(ctx)

    def test_protocol_id(self):
        assert ApiKeyProtocol().protocol_id == "apiKey"


class TestApiKeyProtocolPrepareHeaders:
    def test_default_header_injection(self):
        creds = ApiKeyCredentials(
            protocol_id="apiKey", api_key="abc123", header_name="X-API-Key",
        )
        headers = ApiKeyProtocol().prepare_headers(creds)
        assert headers == {"X-API-Key": "abc123"}

    def test_custom_header_injection(self):
        creds = ApiKeyCredentials(
            protocol_id="apiKey", api_key="abc123", header_name="X-Custom",
        )
        headers = ApiKeyProtocol().prepare_headers(creds)
        assert headers == {"X-Custom": "abc123"}


# ---------------------------------------------------------------------------
# ApiKeyVerifier – Bearer fallback
# ---------------------------------------------------------------------------


class TestApiKeyVerifierBearerFallback:
    def test_dedicated_header_preferred_over_bearer(self):
        verifier = ApiKeyVerifier(
            valid_keys={"key-abc"}, bearer_fallback=True,
        )
        principal = verifier.verify({
            "X-API-Key": "key-abc",
            "Authorization": "Bearer key-other",
        })
        assert principal is not None
        assert principal.claims["api_key_header"] == "X-API-Key"

    def test_bearer_fallback_when_dedicated_missing(self):
        verifier = ApiKeyVerifier(
            valid_keys={"key-abc"}, bearer_fallback=True,
        )
        principal = verifier.verify({"Authorization": "Bearer key-abc"})
        assert principal is not None
        assert principal.claims["api_key_header"] == "Authorization"

    def test_bearer_fallback_invalid_key(self):
        verifier = ApiKeyVerifier(
            valid_keys={"key-abc"}, bearer_fallback=True,
        )
        assert verifier.verify({"Authorization": "Bearer wrong"}) is None

    def test_no_fallback_by_default(self):
        verifier = ApiKeyVerifier(valid_keys={"key-abc"})
        assert verifier.verify({"Authorization": "Bearer key-abc"}) is None

    def test_both_headers_missing(self):
        verifier = ApiKeyVerifier(
            valid_keys={"key-abc"}, bearer_fallback=True,
        )
        assert verifier.verify({}) is None

    def test_non_bearer_auth_ignored(self):
        verifier = ApiKeyVerifier(
            valid_keys={"key-abc"}, bearer_fallback=True,
        )
        assert verifier.verify({"Authorization": "Basic key-abc"}) is None


# ---------------------------------------------------------------------------
# MultiProtocolAuthBackend – verifier ordering constraint (R1 gate)
# ---------------------------------------------------------------------------


class TestVerifierOrdering:
    """JWT Bearer must not be consumed by ApiKeyVerifier when both are present.

    Per R1 排序约束: JWTTokenVerifier MUST precede ApiKeyVerifier in the
    verifier chain so that a real JWT token sent via ``Authorization: Bearer``
    is validated as JWT, not misinterpreted as an API key.
    """

    def test_jwt_bearer_not_consumed_by_apikey_verifier(self):
        token = _make_jwt_token(
            {"sub": "user-1", "scope": "read", "exp": time.time() + 3600},
        )
        backend = MultiProtocolAuthBackend(
            verifiers=[
                JWTTokenVerifier(
                    secret_or_public_key=HMAC_TEST_KEY, algorithms=["HS256"],
                ),
                ApiKeyVerifier(
                    valid_keys={token}, bearer_fallback=True,
                ),
            ],
        )
        principal = backend.verify(
            headers={"Authorization": f"Bearer {token}"},
            path="/tasks/send",
        )
        assert principal is not None
        assert principal.subject == "user-1"
        assert "api_key_header" not in principal.claims

    def test_apikey_via_dedicated_header_still_works(self):
        backend = MultiProtocolAuthBackend(
            verifiers=[
                JWTTokenVerifier(
                    secret_or_public_key=HMAC_TEST_KEY, algorithms=["HS256"],
                ),
                ApiKeyVerifier(
                    valid_keys={"my-api-key"}, bearer_fallback=True,
                ),
            ],
        )
        principal = backend.verify(
            headers={"X-API-Key": "my-api-key"},
            path="/tasks/send",
        )
        assert principal is not None
        assert "apikey:" in principal.subject

    def test_apikey_bearer_fallback_when_jwt_rejects(self):
        """API key via Bearer is accepted when JWT verifier does not match."""
        api_key = "simple-api-key-value"
        backend = MultiProtocolAuthBackend(
            verifiers=[
                JWTTokenVerifier(
                    secret_or_public_key=HMAC_TEST_KEY, algorithms=["HS256"],
                ),
                ApiKeyVerifier(
                    valid_keys={api_key}, bearer_fallback=True,
                ),
            ],
        )
        principal = backend.verify(
            headers={"Authorization": f"Bearer {api_key}"},
            path="/tasks/send",
        )
        assert principal is not None
        assert "apikey:" in principal.subject
