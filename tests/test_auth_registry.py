"""
Tests for AuthProtocolRegistry and SelectedRequirement.

Covers: protocol registration, select_requirement priority/ordering,
single-scheme selection, AND-composite skip-with-warning, no-match error.
"""

import logging
from typing import Dict

import pytest

from python_a2a.auth.protocol import (
    AuthContext,
    AuthCredentials,
    AuthProtocol,
    OAuthCredentials,
    ApiKeyCredentials,
    HttpCredentials,
)
from python_a2a.auth.registry import (
    AuthProtocolRegistry,
    SchemeBinding,
    SelectedRequirement,
)
from python_a2a.exceptions import A2AAuthenticationError
from python_a2a.models.agent import OAuthFlow, OAuthFlows, SecurityScheme


# ---------------------------------------------------------------------------
# Stub protocol implementations for testing
# ---------------------------------------------------------------------------


class StubOAuth2Protocol(AuthProtocol):
    @property
    def protocol_id(self) -> str:
        return "oauth2"

    def authenticate(self, context: AuthContext) -> AuthCredentials:
        return OAuthCredentials(protocol_id="oauth2", access_token="tok")

    def prepare_headers(self, credentials: AuthCredentials) -> Dict[str, str]:
        assert isinstance(credentials, OAuthCredentials)
        return {"Authorization": f"Bearer {credentials.access_token}"}


class StubApiKeyProtocol(AuthProtocol):
    @property
    def protocol_id(self) -> str:
        return "apiKey"

    def authenticate(self, context: AuthContext) -> AuthCredentials:
        return ApiKeyCredentials(protocol_id="apiKey", api_key="key123")

    def prepare_headers(self, credentials: AuthCredentials) -> Dict[str, str]:
        assert isinstance(credentials, ApiKeyCredentials)
        return {credentials.header_name: credentials.api_key}


class StubHttpProtocol(AuthProtocol):
    @property
    def protocol_id(self) -> str:
        return "http"

    def authenticate(self, context: AuthContext) -> AuthCredentials:
        return HttpCredentials(protocol_id="http", scheme="bearer", token="t")

    def prepare_headers(self, credentials: AuthCredentials) -> Dict[str, str]:
        assert isinstance(credentials, HttpCredentials)
        return {"Authorization": f"{credentials.scheme} {credentials.token}"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OAUTH2_SCHEME = SecurityScheme(
    type="oauth2",
    flows=OAuthFlows(
        client_credentials=OAuthFlow(
            token_url="https://auth.example.com/token",
            scopes={"a2a:call": "Call agent"},
        ),
    ),
)

APIKEY_SCHEME = SecurityScheme(
    type="apiKey",
    in_location="header",
    name="X-API-Key",
)

MTLS_SCHEME = SecurityScheme(type="mutualTLS")

HTTP_BEARER_SCHEME = SecurityScheme(type="http", scheme="bearer")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAuthProtocolRegistry:
    def test_register_and_lookup(self):
        registry = AuthProtocolRegistry()
        protocol = StubOAuth2Protocol()
        registry.register(protocol)
        assert registry.get("oauth2") is protocol
        assert registry.get("unknown") is None
        assert "oauth2" in registry.registered_ids

    def test_select_single_scheme_first_match(self):
        """First satisfiable single-scheme requirement is selected."""
        registry = AuthProtocolRegistry()
        registry.register(StubOAuth2Protocol())
        registry.register(StubApiKeyProtocol())

        card_schemes = {"oauth2": OAUTH2_SCHEME, "apiKey": APIKEY_SCHEME}
        card_security = [
            {"oauth2": ["a2a:call"]},
            {"apiKey": []},
        ]

        result = registry.select_requirement(card_security, card_schemes)
        assert len(result.bindings) == 1
        assert result.bindings[0].scheme_name == "oauth2"
        assert result.bindings[0].scopes == ["a2a:call"]
        assert result.bindings[0].protocol.protocol_id == "oauth2"
        assert not result.is_composite

    def test_select_falls_through_to_second(self):
        """If first requirement's scheme type is not registered, skip to next."""
        registry = AuthProtocolRegistry()
        # Only register apiKey, not oauth2
        registry.register(StubApiKeyProtocol())

        card_schemes = {"oauth2": OAUTH2_SCHEME, "apiKey": APIKEY_SCHEME}
        card_security = [
            {"oauth2": ["a2a:call"]},
            {"apiKey": []},
        ]

        result = registry.select_requirement(card_security, card_schemes)
        assert len(result.bindings) == 1
        assert result.bindings[0].scheme_name == "apiKey"

    def test_select_no_match_raises(self):
        """Error when no requirement can be satisfied."""
        registry = AuthProtocolRegistry()
        # No protocols registered

        card_schemes = {"oauth2": OAUTH2_SCHEME}
        card_security = [{"oauth2": ["a2a:call"]}]

        with pytest.raises(A2AAuthenticationError, match="No satisfiable"):
            registry.select_requirement(card_security, card_schemes)

    def test_select_skips_and_composite_with_warning(self, caplog):
        """AND-composite requirements are skipped with a warning."""
        registry = AuthProtocolRegistry()
        registry.register(StubOAuth2Protocol())
        registry.register(StubApiKeyProtocol())

        card_schemes = {"oauth2": OAUTH2_SCHEME, "apiKey": APIKEY_SCHEME}
        # Single requirement with AND (both oauth2 AND apiKey)
        card_security = [{"oauth2": ["a2a:call"], "apiKey": []}]

        with caplog.at_level(logging.WARNING):
            with pytest.raises(A2AAuthenticationError, match="No satisfiable"):
                registry.select_requirement(card_security, card_schemes)

        assert "AND-composite" in caplog.text

    def test_select_and_composite_skipped_but_single_after(self, caplog):
        """AND-composite is skipped; next single-scheme requirement is used."""
        registry = AuthProtocolRegistry()
        registry.register(StubOAuth2Protocol())
        registry.register(StubApiKeyProtocol())

        card_schemes = {"oauth2": OAUTH2_SCHEME, "apiKey": APIKEY_SCHEME}
        card_security = [
            {"oauth2": ["a2a:call"], "apiKey": []},  # AND -> skip
            {"apiKey": []},                           # single -> match
        ]

        with caplog.at_level(logging.WARNING):
            result = registry.select_requirement(card_security, card_schemes)

        assert "AND-composite" in caplog.text
        assert len(result.bindings) == 1
        assert result.bindings[0].scheme_name == "apiKey"

    def test_select_undefined_scheme_in_security_skipped(self):
        """Requirement referencing undefined scheme is silently skipped."""
        registry = AuthProtocolRegistry()
        registry.register(StubApiKeyProtocol())

        card_schemes = {"apiKey": APIKEY_SCHEME}
        # First req references "unknown" which is not in card_schemes
        card_security = [
            {"unknown": []},
            {"apiKey": []},
        ]

        result = registry.select_requirement(card_security, card_schemes)
        assert result.bindings[0].scheme_name == "apiKey"

    def test_empty_security_list_raises(self):
        """Empty security list means no requirements -> error."""
        registry = AuthProtocolRegistry()
        registry.register(StubOAuth2Protocol())

        with pytest.raises(A2AAuthenticationError, match="No satisfiable"):
            registry.select_requirement([], {"oauth2": OAUTH2_SCHEME})


class TestSelectedRequirement:
    def test_is_composite_false_for_single(self):
        binding = SchemeBinding(
            scheme_name="oauth2",
            security_scheme=OAUTH2_SCHEME,
            scopes=["a2a:call"],
            protocol=StubOAuth2Protocol(),
        )
        selected = SelectedRequirement(bindings=[binding])
        assert not selected.is_composite

    def test_is_composite_true_for_multiple(self):
        b1 = SchemeBinding(
            scheme_name="oauth2",
            security_scheme=OAUTH2_SCHEME,
            scopes=[],
            protocol=StubOAuth2Protocol(),
        )
        b2 = SchemeBinding(
            scheme_name="apiKey",
            security_scheme=APIKEY_SCHEME,
            scopes=[],
            protocol=StubApiKeyProtocol(),
        )
        selected = SelectedRequirement(bindings=[b1, b2])
        assert selected.is_composite
