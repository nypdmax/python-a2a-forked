"""
Tests for UnifiedAuthProvider and client integration hooks.

Covers:
- UnifiedAuthProvider header generation
- 401 refresh-and-retry logic (retry once, then stop)
- 403 does not trigger retry
- A2AClient auth_provider parameter wiring
- StreamingClient auth_provider parameter wiring (async)
"""

import json
from typing import Dict
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from python_a2a.auth.protocol import (
    AuthContext,
    AuthCredentials,
    AuthProtocol,
    OAuthCredentials,
    ApiKeyCredentials,
)
from python_a2a.auth.provider import UnifiedAuthProvider
from python_a2a.auth.registry import SchemeBinding, SelectedRequirement
from python_a2a.models.agent import OAuthFlow, OAuthFlows, SecurityScheme


# ---------------------------------------------------------------------------
# Stub protocol for testing
# ---------------------------------------------------------------------------


class StubOAuth2Protocol(AuthProtocol):
    """Stub that returns a static Bearer token."""

    def __init__(self, token: str = "tok-initial"):
        self._token = token
        self._call_count = 0

    @property
    def protocol_id(self) -> str:
        return "oauth2"

    def authenticate(self, context: AuthContext) -> OAuthCredentials:
        self._call_count += 1
        return OAuthCredentials(
            protocol_id="oauth2",
            access_token=f"{self._token}-{self._call_count}",
            token_type="Bearer",
        )

    def prepare_headers(self, credentials: AuthCredentials) -> Dict[str, str]:
        assert isinstance(credentials, OAuthCredentials)
        return {"Authorization": f"Bearer {credentials.access_token}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OAUTH_SCHEME = SecurityScheme(
    type="oauth2",
    flows=OAuthFlows(
        client_credentials=OAuthFlow(token_url="https://auth.example.com/token"),
    ),
)


def _make_provider(protocol: AuthProtocol = None) -> UnifiedAuthProvider:
    protocol = protocol or StubOAuth2Protocol()
    binding = SchemeBinding(
        scheme_name="oauth2",
        security_scheme=OAUTH_SCHEME,
        scopes=["a2a:call"],
        protocol=protocol,
    )
    selected = SelectedRequirement(bindings=[binding])
    return UnifiedAuthProvider(
        selected=selected,
        local_config={"client_id": "cid", "client_secret": "csec"},
        agent_url="https://agent.example.com",
    )


# ---------------------------------------------------------------------------
# UnifiedAuthProvider
# ---------------------------------------------------------------------------


class TestUnifiedAuthProvider:
    def test_get_auth_headers(self):
        provider = _make_provider()
        headers = provider.get_auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer tok-initial-")

    def test_should_retry_on_401_first_time(self):
        provider = _make_provider()
        # Must have credentials first
        provider.get_auth_headers()
        assert provider.should_retry_on_401() is True

    def test_should_retry_on_401_second_time_false(self):
        provider = _make_provider()
        provider.get_auth_headers()
        provider.force_refresh()
        assert provider.should_retry_on_401() is False

    def test_reset_retry(self):
        provider = _make_provider()
        provider.get_auth_headers()
        provider.force_refresh()
        assert provider.should_retry_on_401() is False
        provider.reset_retry()
        assert provider.should_retry_on_401() is True

    def test_force_refresh_returns_new_headers(self):
        provider = _make_provider()
        headers1 = provider.get_auth_headers()
        headers2 = provider.force_refresh()
        # The stub increments its counter, so tokens differ
        assert headers1["Authorization"] != headers2["Authorization"]

    def test_no_credentials_no_retry(self):
        """If get_auth_headers was never called, retry flag is False."""
        provider = _make_provider()
        assert provider.should_retry_on_401() is False

    @pytest.mark.anyio
    async def test_get_auth_headers_async(self):
        provider = _make_provider()
        headers = await provider.get_auth_headers_async()
        assert "Authorization" in headers

    @pytest.mark.anyio
    async def test_force_refresh_async(self):
        provider = _make_provider()
        h1 = await provider.get_auth_headers_async()
        h2 = await provider.force_refresh_async()
        assert h1["Authorization"] != h2["Authorization"]


# ---------------------------------------------------------------------------
# A2AClient with auth_provider
# ---------------------------------------------------------------------------


class TestA2AClientAuthIntegration:
    def _mock_agent_card_response(self):
        """Return a mock response for agent card fetch."""
        card_resp = MagicMock()
        card_resp.status_code = 200
        card_resp.headers = {"Content-Type": "application/json"}
        card_resp.json.return_value = {
            "name": "TestAgent",
            "description": "Test",
            "url": "https://agent.example.com",
            "version": "1.0.0",
            "capabilities": {},
            "skills": [],
        }
        card_resp.raise_for_status = MagicMock()
        return card_resp

    def _mock_task_response(self, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.headers = {"Content-Type": "application/json", "WWW-Authenticate": "Bearer"}
        resp.json.return_value = {
            "jsonrpc": "2.0",
            "result": {
                "id": "task-1",
                "status": {"state": "completed"},
                "artifacts": [{"parts": [{"type": "text", "text": "Hello"}]}],
            },
        }
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            import requests as req_lib
            resp.raise_for_status.side_effect = req_lib.HTTPError(response=resp)
        return resp

    def test_auth_provider_injects_headers(self):
        """Auth provider headers are included in task send requests."""
        from python_a2a.client.http import A2AClient

        card_resp = self._mock_agent_card_response()
        task_resp = self._mock_task_response()

        with patch("requests.get", return_value=card_resp):
            with patch("requests.post", return_value=task_resp) as mock_post:
                provider = _make_provider()
                client = A2AClient(
                    "https://agent.example.com",
                    auth_provider=provider,
                )
                merged = client._get_merged_headers()
                assert "Authorization" in merged

    def test_auth_provider_none_backward_compat(self):
        """Without auth_provider, headers are unchanged."""
        from python_a2a.client.http import A2AClient

        card_resp = self._mock_agent_card_response()

        with patch("requests.get", return_value=card_resp):
            client = A2AClient(
                "https://agent.example.com",
                headers={"X-Custom": "value"},
            )
            merged = client._get_merged_headers()
            assert merged == {"Content-Type": "application/json", "X-Custom": "value"}

    def test_user_headers_override_auth(self):
        """User-supplied headers take priority over auth provider."""
        from python_a2a.client.http import A2AClient

        card_resp = self._mock_agent_card_response()

        with patch("requests.get", return_value=card_resp):
            provider = _make_provider()
            client = A2AClient(
                "https://agent.example.com",
                headers={"Authorization": "Bearer user-override"},
                auth_provider=provider,
            )
            merged = client._get_merged_headers()
            assert merged["Authorization"] == "Bearer user-override"

    def test_post_with_auth_retry_on_401(self):
        """_post_with_auth_retry retries once on 401."""
        from python_a2a.client.http import A2AClient

        card_resp = self._mock_agent_card_response()
        resp_401 = self._mock_task_response(status_code=401)
        resp_200 = self._mock_task_response(status_code=200)

        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return resp_401
            return resp_200

        with patch("requests.get", return_value=card_resp):
            with patch("requests.post", side_effect=_side_effect):
                provider = _make_provider()
                client = A2AClient(
                    "https://agent.example.com",
                    auth_provider=provider,
                )
                # Ensure provider has credentials
                provider.get_auth_headers()
                response = client._post_with_auth_retry(
                    "https://agent.example.com/tasks/send",
                    {"test": "data"},
                )
                assert response.status_code == 200
                assert call_count == 2

    def test_post_with_auth_retry_gives_up_on_second_401(self):
        """After retry, if still 401, return the 401 response."""
        from python_a2a.client.http import A2AClient

        card_resp = self._mock_agent_card_response()
        resp_401 = self._mock_task_response(status_code=401)

        with patch("requests.get", return_value=card_resp):
            with patch("requests.post", return_value=resp_401):
                provider = _make_provider()
                client = A2AClient(
                    "https://agent.example.com",
                    auth_provider=provider,
                )
                provider.get_auth_headers()
                response = client._post_with_auth_retry(
                    "https://agent.example.com/tasks/send",
                    {"test": "data"},
                )
                # Should return 401 after exhausting retry
                assert response.status_code == 401

    def test_403_does_not_trigger_retry(self):
        """403 responses do not trigger refresh."""
        from python_a2a.client.http import A2AClient

        card_resp = self._mock_agent_card_response()
        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_403.headers = {"WWW-Authenticate": 'Bearer error="insufficient_scope"'}

        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return resp_403

        with patch("requests.get", return_value=card_resp):
            with patch("requests.post", side_effect=_side_effect):
                provider = _make_provider()
                client = A2AClient(
                    "https://agent.example.com",
                    auth_provider=provider,
                )
                provider.get_auth_headers()
                response = client._post_with_auth_retry(
                    "https://agent.example.com/tasks/send",
                    {"test": "data"},
                )
                assert response.status_code == 403
                # Only one call, no retry
                assert call_count == 1


# ---------------------------------------------------------------------------
# StreamingClient with auth_provider
# ---------------------------------------------------------------------------


class TestStreamingClientAuthIntegration:
    def test_auth_provider_parameter_accepted(self):
        """StreamingClient accepts auth_provider kwarg."""
        from python_a2a.client.streaming import StreamingClient

        provider = _make_provider()
        client = StreamingClient(
            url="https://agent.example.com",
            auth_provider=provider,
        )
        assert client._auth_provider is provider

    def test_no_auth_provider_backward_compat(self):
        from python_a2a.client.streaming import StreamingClient

        client = StreamingClient(url="https://agent.example.com")
        assert client._auth_provider is None

    @pytest.mark.anyio
    async def test_get_merged_headers_async_with_provider(self):
        from python_a2a.client.streaming import StreamingClient

        provider = _make_provider()
        client = StreamingClient(
            url="https://agent.example.com",
            auth_provider=provider,
        )
        headers = await client._get_merged_headers_async()
        assert "Authorization" in headers

    @pytest.mark.anyio
    async def test_get_merged_headers_async_no_provider(self):
        from python_a2a.client.streaming import StreamingClient

        client = StreamingClient(
            url="https://agent.example.com",
            headers={"X-Custom": "v"},
        )
        headers = await client._get_merged_headers_async()
        assert headers == {"Content-Type": "application/json", "X-Custom": "v"}
