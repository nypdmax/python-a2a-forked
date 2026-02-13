"""
Tests for A2A v0.3.0 security types on AgentCard.

Covers: OAuthFlow, OAuthFlows, SecurityScheme, AgentCard
securitySchemes / security fields, serialization round-trip,
backward compatibility with legacy ``authentication`` field.
"""

import json
import logging

import pytest

from python_a2a.models.agent import (
    AgentCard,
    AgentSkill,
    OAuthFlow,
    OAuthFlows,
    SecurityScheme,
)


# ---------------------------------------------------------------------------
# OAuthFlow
# ---------------------------------------------------------------------------


class TestOAuthFlow:
    def test_minimal_flow(self):
        flow = OAuthFlow(token_url="https://auth.example.com/token")
        assert flow.token_url == "https://auth.example.com/token"
        assert flow.authorization_url is None
        assert flow.refresh_url is None
        assert flow.scopes == {}

    def test_full_flow_roundtrip(self):
        flow = OAuthFlow(
            token_url="https://auth.example.com/token",
            authorization_url="https://auth.example.com/authorize",
            refresh_url="https://auth.example.com/refresh",
            scopes={"read": "Read access", "write": "Write access"},
        )
        data = flow.to_dict()
        assert data == {
            "tokenUrl": "https://auth.example.com/token",
            "authorizationUrl": "https://auth.example.com/authorize",
            "refreshUrl": "https://auth.example.com/refresh",
            "scopes": {"read": "Read access", "write": "Write access"},
        }
        restored = OAuthFlow.from_dict(data)
        assert restored.token_url == flow.token_url
        assert restored.authorization_url == flow.authorization_url
        assert restored.refresh_url == flow.refresh_url
        assert restored.scopes == flow.scopes

    def test_from_dict_missing_token_url_defaults_to_empty(self):
        flow = OAuthFlow.from_dict({})
        assert flow.token_url == ""

    def test_empty_scopes_omitted_in_dict(self):
        flow = OAuthFlow(token_url="https://t.co/token")
        data = flow.to_dict()
        assert "scopes" not in data


# ---------------------------------------------------------------------------
# OAuthFlows
# ---------------------------------------------------------------------------


class TestOAuthFlows:
    def test_empty_flows(self):
        flows = OAuthFlows()
        assert flows.to_dict() == {}

    def test_client_credentials_roundtrip(self):
        flows = OAuthFlows(
            client_credentials=OAuthFlow(
                token_url="https://auth.example.com/token",
                scopes={"a2a": "A2A communication"},
            ),
        )
        data = flows.to_dict()
        assert "clientCredentials" in data
        assert "authorizationCode" not in data

        restored = OAuthFlows.from_dict(data)
        assert restored.client_credentials is not None
        assert restored.client_credentials.token_url == "https://auth.example.com/token"
        assert restored.authorization_code is None


# ---------------------------------------------------------------------------
# SecurityScheme
# ---------------------------------------------------------------------------


class TestSecurityScheme:
    def test_apikey_scheme(self):
        scheme = SecurityScheme(
            type="apiKey",
            in_location="header",
            name="X-API-Key",
            description="API key authentication",
        )
        data = scheme.to_dict()
        assert data == {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key authentication",
        }
        restored = SecurityScheme.from_dict(data)
        assert restored.type == "apiKey"
        assert restored.in_location == "header"
        assert restored.name == "X-API-Key"

    def test_http_bearer_scheme(self):
        scheme = SecurityScheme(type="http", scheme="bearer")
        data = scheme.to_dict()
        assert data == {"type": "http", "scheme": "bearer"}
        restored = SecurityScheme.from_dict(data)
        assert restored.type == "http"
        assert restored.scheme == "bearer"

    def test_oauth2_scheme_roundtrip(self):
        scheme = SecurityScheme(
            type="oauth2",
            flows=OAuthFlows(
                client_credentials=OAuthFlow(
                    token_url="https://auth.example.com/token",
                    scopes={"a2a:call": "Call remote agent"},
                ),
            ),
        )
        data = scheme.to_dict()
        assert data["type"] == "oauth2"
        assert "flows" in data
        assert "clientCredentials" in data["flows"]

        restored = SecurityScheme.from_dict(data)
        assert restored.type == "oauth2"
        assert restored.flows is not None
        assert restored.flows.client_credentials is not None
        assert restored.flows.client_credentials.token_url == "https://auth.example.com/token"
        assert restored.flows.client_credentials.scopes == {"a2a:call": "Call remote agent"}

    def test_openidconnect_scheme(self):
        scheme = SecurityScheme(
            type="openIdConnect",
            open_id_connect_url="https://auth.example.com/.well-known/openid-configuration",
        )
        data = scheme.to_dict()
        assert data["openIdConnectUrl"] == "https://auth.example.com/.well-known/openid-configuration"
        restored = SecurityScheme.from_dict(data)
        assert restored.open_id_connect_url == "https://auth.example.com/.well-known/openid-configuration"

    def test_vendor_extensions_preserved(self):
        raw = {
            "type": "oauth2",
            "flows": {
                "clientCredentials": {
                    "tokenUrl": "https://auth.example.com/token",
                },
            },
            "x-enterprise-idp": "https://idp.internal.example.com",
            "x-audience": "a2a-agent-prod",
        }
        scheme = SecurityScheme.from_dict(raw)
        assert scheme.extra_fields["x-enterprise-idp"] == "https://idp.internal.example.com"
        assert scheme.extra_fields["x-audience"] == "a2a-agent-prod"

        # Round-trip preserves extensions
        data = scheme.to_dict()
        assert data["x-enterprise-idp"] == "https://idp.internal.example.com"
        assert data["x-audience"] == "a2a-agent-prod"

    def test_from_dict_unknown_type_preserved(self):
        """Schemes with non-standard types still parse without error."""
        scheme = SecurityScheme.from_dict({"type": "mutualTLS"})
        assert scheme.type == "mutualTLS"


# ---------------------------------------------------------------------------
# AgentCard – security fields
# ---------------------------------------------------------------------------


def _make_agent_card_dict(
    *,
    with_security: bool = False,
    with_legacy_auth: bool = False,
) -> dict:
    """Helper to create a representative AgentCard dict."""
    card: dict = {
        "name": "TestAgent",
        "description": "A test agent",
        "url": "https://agent.example.com",
        "version": "1.0.0",
        "protocolVersion": "0.3.0",
        "capabilities": {"streaming": True},
        "skills": [],
    }
    if with_security:
        card["securitySchemes"] = {
            "oauth2": {
                "type": "oauth2",
                "flows": {
                    "clientCredentials": {
                        "tokenUrl": "https://auth.example.com/token",
                        "scopes": {"a2a:call": "Call agent"},
                    },
                },
            },
            "apiKey": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
            },
        }
        card["security"] = [
            {"oauth2": ["a2a:call"]},
            {"apiKey": []},
        ]
    if with_legacy_auth:
        card["authentication"] = "bearer_token"
    return card


class TestAgentCardSecurity:
    def test_card_without_security_fields(self):
        """AgentCard without security fields still works (backward compat)."""
        data = _make_agent_card_dict()
        card = AgentCard.from_dict(data)
        assert card.security_schemes is None
        assert card.security is None

        # Round-trip must not add security fields
        out = card.to_dict()
        assert "securitySchemes" not in out
        assert "security" not in out

    def test_card_with_security_roundtrip(self):
        data = _make_agent_card_dict(with_security=True)
        card = AgentCard.from_dict(data)
        assert card.security_schemes is not None
        assert len(card.security_schemes) == 2
        assert "oauth2" in card.security_schemes
        assert "apiKey" in card.security_schemes
        assert card.security == [
            {"oauth2": ["a2a:call"]},
            {"apiKey": []},
        ]

        # Verify scheme details
        oauth = card.security_schemes["oauth2"]
        assert oauth.type == "oauth2"
        assert oauth.flows is not None
        assert oauth.flows.client_credentials is not None
        assert oauth.flows.client_credentials.token_url == "https://auth.example.com/token"

        api_key = card.security_schemes["apiKey"]
        assert api_key.type == "apiKey"
        assert api_key.in_location == "header"
        assert api_key.name == "X-API-Key"

        # Round-trip
        out = card.to_dict()
        assert "securitySchemes" in out
        assert "security" in out
        # JSON round-trip
        restored = AgentCard.from_dict(json.loads(json.dumps(out)))
        assert len(restored.security_schemes) == 2
        assert restored.security == card.security

    def test_card_legacy_auth_only(self):
        data = _make_agent_card_dict(with_legacy_auth=True)
        card = AgentCard.from_dict(data)
        assert card.authentication == "bearer_token"
        assert card.security_schemes is None

    def test_card_both_legacy_and_new_logs_warning(self, caplog):
        data = _make_agent_card_dict(with_security=True, with_legacy_auth=True)
        with caplog.at_level(logging.WARNING):
            card = AgentCard.from_dict(data)
        assert "securitySchemes/security take precedence" in caplog.text
        # New fields should still be populated
        assert card.security_schemes is not None
        assert card.authentication == "bearer_token"

    def test_card_to_json_includes_security(self):
        card = AgentCard(
            name="Secure",
            description="Secured agent",
            url="https://agent.example.com",
            security_schemes={
                "bearer": SecurityScheme(type="http", scheme="bearer"),
            },
            security=[{"bearer": []}],
        )
        json_str = card.to_json()
        parsed = json.loads(json_str)
        assert "securitySchemes" in parsed
        assert parsed["securitySchemes"]["bearer"]["type"] == "http"
        assert parsed["security"] == [{"bearer": []}]

    def test_card_validate_a2a_protocol_unchanged(self):
        """Validate that adding security fields doesn't break the validator."""
        card = AgentCard.from_dict(_make_agent_card_dict(with_security=True))
        violations = card.validate_a2a_protocol()
        assert violations == []

    def test_empty_security_schemes_dict(self):
        """An empty securitySchemes dict is valid but yields empty map."""
        data = _make_agent_card_dict()
        data["securitySchemes"] = {}
        card = AgentCard.from_dict(data)
        assert card.security_schemes == {}

    def test_security_schemes_non_dict_ignored(self):
        """If securitySchemes is not a dict, treat as absent."""
        data = _make_agent_card_dict()
        data["securitySchemes"] = "invalid"
        card = AgentCard.from_dict(data)
        assert card.security_schemes is None

    def test_security_or_of_and_structure(self):
        """Verify the OR-of-AND list structure."""
        data = _make_agent_card_dict()
        data["securitySchemes"] = {
            "oauth2": {"type": "oauth2", "flows": {
                "clientCredentials": {"tokenUrl": "https://t.co/token"},
            }},
            "mtls": {"type": "mutualTLS"},
        }
        # AND: both oauth2 and mtls required together
        data["security"] = [{"oauth2": ["scope1"], "mtls": []}]
        card = AgentCard.from_dict(data)
        assert len(card.security) == 1
        requirement = card.security[0]
        assert "oauth2" in requirement
        assert "mtls" in requirement
