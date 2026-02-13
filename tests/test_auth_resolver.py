"""
Tests for AgentCardSecurityResolver.

Covers: new security fields resolution, legacy authentication fallback,
undefined scheme reference handling, empty/absent fields.
"""

import logging

import pytest

from python_a2a.auth.resolver import AgentCardSecurityResolver, SecurityRequirement
from python_a2a.models.agent import (
    AgentCard,
    OAuthFlow,
    OAuthFlows,
    SecurityScheme,
)


def _make_card(**kwargs) -> AgentCard:
    """Minimal AgentCard factory."""
    defaults = {
        "name": "TestAgent",
        "description": "Test",
        "url": "https://agent.example.com",
    }
    defaults.update(kwargs)
    return AgentCard(**defaults)


class TestAgentCardSecurityResolver:
    def setup_method(self):
        self.resolver = AgentCardSecurityResolver()

    # ------------------------------------------------------------------
    # New security fields
    # ------------------------------------------------------------------

    def test_resolve_new_fields_single_requirement(self):
        card = _make_card(
            security_schemes={
                "oauth2": SecurityScheme(
                    type="oauth2",
                    flows=OAuthFlows(
                        client_credentials=OAuthFlow(
                            token_url="https://auth.example.com/token",
                        ),
                    ),
                ),
            },
            security=[{"oauth2": ["a2a:call"]}],
        )
        requirements = self.resolver.resolve(card)
        assert len(requirements) == 1
        req = requirements[0]
        assert "oauth2" in req.schemes
        entry = req.schemes["oauth2"]
        assert entry.security_scheme.type == "oauth2"
        assert entry.scopes == ["a2a:call"]

    def test_resolve_new_fields_multiple_or_requirements(self):
        card = _make_card(
            security_schemes={
                "oauth2": SecurityScheme(type="oauth2"),
                "apiKey": SecurityScheme(type="apiKey", in_location="header", name="X-API-Key"),
            },
            security=[
                {"oauth2": ["a2a:call"]},
                {"apiKey": []},
            ],
        )
        requirements = self.resolver.resolve(card)
        assert len(requirements) == 2
        assert "oauth2" in requirements[0].schemes
        assert "apiKey" in requirements[1].schemes

    def test_resolve_new_fields_and_requirement(self):
        """AND: single requirement with multiple schemes."""
        card = _make_card(
            security_schemes={
                "oauth2": SecurityScheme(type="oauth2"),
                "mtls": SecurityScheme(type="mutualTLS"),
            },
            security=[{"oauth2": ["scope1"], "mtls": []}],
        )
        requirements = self.resolver.resolve(card)
        assert len(requirements) == 1
        req = requirements[0]
        assert len(req.schemes) == 2
        assert "oauth2" in req.schemes
        assert "mtls" in req.schemes

    def test_resolve_skips_requirement_with_undefined_scheme(self, caplog):
        card = _make_card(
            security_schemes={
                "oauth2": SecurityScheme(type="oauth2"),
            },
            security=[
                {"undefined_scheme": []},   # references missing scheme
                {"oauth2": ["a2a:call"]},   # this one is valid
            ],
        )
        with caplog.at_level(logging.WARNING):
            requirements = self.resolver.resolve(card)
        assert len(requirements) == 1
        assert "oauth2" in requirements[0].schemes
        assert "undefined scheme" in caplog.text.lower()

    def test_resolve_new_fields_take_precedence_over_legacy(self):
        """When both new and legacy fields exist, new fields win."""
        card = _make_card(
            authentication="bearer_token",
            security_schemes={
                "oauth2": SecurityScheme(type="oauth2"),
            },
            security=[{"oauth2": []}],
        )
        requirements = self.resolver.resolve(card)
        assert len(requirements) == 1
        assert "oauth2" in requirements[0].schemes
        # Legacy should be ignored

    # ------------------------------------------------------------------
    # Legacy authentication fallback
    # ------------------------------------------------------------------

    def test_resolve_legacy_bearer(self):
        card = _make_card(authentication="bearer")
        requirements = self.resolver.resolve(card)
        assert len(requirements) == 1
        req = requirements[0]
        scheme_name = list(req.schemes.keys())[0]
        assert scheme_name.startswith("_legacy_")
        entry = req.schemes[scheme_name]
        assert entry.security_scheme.type == "http"
        assert entry.security_scheme.scheme == "bearer"
        assert entry.scopes == []

    def test_resolve_legacy_bearer_token(self):
        card = _make_card(authentication="bearer_token")
        requirements = self.resolver.resolve(card)
        assert len(requirements) == 1
        entry = list(requirements[0].schemes.values())[0]
        assert entry.security_scheme.type == "http"
        assert entry.security_scheme.scheme == "bearer"

    def test_resolve_legacy_api_key(self):
        card = _make_card(authentication="api_key")
        requirements = self.resolver.resolve(card)
        assert len(requirements) == 1
        entry = list(requirements[0].schemes.values())[0]
        assert entry.security_scheme.type == "apiKey"
        assert entry.security_scheme.in_location == "header"
        assert entry.security_scheme.name == "X-API-Key"

    def test_resolve_legacy_unknown_value_returns_empty(self, caplog):
        card = _make_card(authentication="custom_unknown")
        with caplog.at_level(logging.WARNING):
            requirements = self.resolver.resolve(card)
        assert requirements == []
        assert "Unknown legacy authentication" in caplog.text

    def test_resolve_legacy_case_insensitive(self):
        card = _make_card(authentication="BEARER")
        requirements = self.resolver.resolve(card)
        assert len(requirements) == 1

    # ------------------------------------------------------------------
    # No authentication
    # ------------------------------------------------------------------

    def test_resolve_no_auth_returns_empty(self):
        card = _make_card()
        requirements = self.resolver.resolve(card)
        assert requirements == []

    def test_resolve_empty_security_schemes_and_security(self):
        """Empty dicts/lists for new fields are treated as absent."""
        card = _make_card(
            security_schemes={},
            security=[],
        )
        requirements = self.resolver.resolve(card)
        assert requirements == []

    def test_resolve_security_schemes_only_no_security_list(self):
        """securitySchemes present but security list absent -> no auth."""
        card = _make_card(
            security_schemes={
                "oauth2": SecurityScheme(type="oauth2"),
            },
            security=None,
        )
        requirements = self.resolver.resolve(card)
        assert requirements == []
