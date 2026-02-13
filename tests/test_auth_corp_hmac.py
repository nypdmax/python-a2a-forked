"""
Tests for the Corp HMAC-SHA256 enterprise protocol example.

Covers: CorpHmacProtocol (client), CorpHmacVerifier (server),
signature validation, timestamp replay protection, vendor extension reading.
"""

import time

import pytest

from python_a2a.auth.protocol import AuthContext
from python_a2a.auth.protocols.corp_hmac_example import (
    HMAC_AUTH_HEADER,
    HMAC_KEY_ID_HEADER,
    HMAC_TIMESTAMP_HEADER,
    CorpHmacCredentials,
    CorpHmacProtocol,
    CorpHmacVerifier,
)
from python_a2a.exceptions import A2AAuthenticationError
from python_a2a.models.agent import SecurityScheme


CORP_SCHEME = SecurityScheme(
    type="http",
    scheme="hmac-sha256",
    description="Corp HMAC-SHA256 signing",
    extra_fields={"x-metadata-url": "https://auth.corp.com/.well-known/hmac-config"},
)


def _make_context(**overrides):
    defaults = {
        "agent_url": "https://agent.corp.com",
        "security_scheme": CORP_SCHEME,
        "local_config": {
            "key_id": "app-123",
            "secret_key": "super-secret-key-32-bytes-long!!",
        },
    }
    defaults.update(overrides)
    return AuthContext(**defaults)


class TestCorpHmacProtocol:
    def test_authenticate_success(self):
        protocol = CorpHmacProtocol()
        ctx = _make_context()
        creds = protocol.authenticate(ctx)
        assert isinstance(creds, CorpHmacCredentials)
        assert creds.key_id == "app-123"
        assert creds.secret_key == "super-secret-key-32-bytes-long!!"

    def test_authenticate_missing_key_id(self):
        protocol = CorpHmacProtocol()
        ctx = _make_context(local_config={"secret_key": "s"})
        with pytest.raises(A2AAuthenticationError, match="key_id"):
            protocol.authenticate(ctx)

    def test_prepare_headers(self):
        protocol = CorpHmacProtocol()
        creds = CorpHmacCredentials(
            key_id="app-123",
            secret_key="super-secret-key-32-bytes-long!!",
        )
        headers = protocol.prepare_headers(creds)
        assert HMAC_KEY_ID_HEADER in headers
        assert HMAC_TIMESTAMP_HEADER in headers
        assert HMAC_AUTH_HEADER in headers
        assert headers[HMAC_KEY_ID_HEADER] == "app-123"

    def test_vendor_extension_accessible(self):
        ctx = _make_context()
        assert ctx.security_scheme.extra_fields["x-metadata-url"] == (
            "https://auth.corp.com/.well-known/hmac-config"
        )

    def test_protocol_id(self):
        assert CorpHmacProtocol().protocol_id == "http"


class TestCorpHmacVerifier:
    def _make_valid_headers(self, key_id="app-123", secret="super-secret-key-32-bytes-long!!"):
        timestamp = str(int(time.time()))
        sig = CorpHmacProtocol._compute_signature(secret, key_id, timestamp)
        return {
            HMAC_KEY_ID_HEADER: key_id,
            HMAC_TIMESTAMP_HEADER: timestamp,
            HMAC_AUTH_HEADER: sig,
        }

    def test_verify_success(self):
        verifier = CorpHmacVerifier(
            valid_keys={"app-123": "super-secret-key-32-bytes-long!!"},
            scopes={"a2a:call"},
        )
        headers = self._make_valid_headers()
        principal = verifier.verify(headers)
        assert principal is not None
        assert principal.subject == "hmac:app-123"
        assert principal.scopes == {"a2a:call"}

    def test_verify_unknown_key_id(self):
        verifier = CorpHmacVerifier(valid_keys={"other-key": "secret"})
        headers = self._make_valid_headers()
        assert verifier.verify(headers) is None

    def test_verify_wrong_signature(self):
        verifier = CorpHmacVerifier(
            valid_keys={"app-123": "different-secret-key-here!!!!!!!"},
        )
        headers = self._make_valid_headers()
        assert verifier.verify(headers) is None

    def test_verify_expired_timestamp(self):
        verifier = CorpHmacVerifier(
            valid_keys={"app-123": "super-secret-key-32-bytes-long!!"},
            max_timestamp_skew=60,
        )
        old_timestamp = str(int(time.time()) - 120)
        sig = CorpHmacProtocol._compute_signature(
            "super-secret-key-32-bytes-long!!", "app-123", old_timestamp,
        )
        headers = {
            HMAC_KEY_ID_HEADER: "app-123",
            HMAC_TIMESTAMP_HEADER: old_timestamp,
            HMAC_AUTH_HEADER: sig,
        }
        assert verifier.verify(headers) is None

    def test_verify_missing_headers(self):
        verifier = CorpHmacVerifier(valid_keys={"k": "s"})
        assert verifier.verify({}) is None
        assert verifier.verify({HMAC_KEY_ID_HEADER: "k"}) is None

    def test_get_challenge_returns_none(self):
        verifier = CorpHmacVerifier(valid_keys={})
        assert verifier.get_challenge() is None

    def test_roundtrip_client_to_server(self):
        """Full round-trip: client generates headers, server verifies them."""
        protocol = CorpHmacProtocol()
        ctx = _make_context()
        creds = protocol.authenticate(ctx)
        headers = protocol.prepare_headers(creds)

        verifier = CorpHmacVerifier(
            valid_keys={"app-123": "super-secret-key-32-bytes-long!!"},
        )
        principal = verifier.verify(headers)
        assert principal is not None
        assert principal.subject == "hmac:app-123"
