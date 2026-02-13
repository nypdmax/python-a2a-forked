"""
Tests for OAuth2 client_credentials protocol, token fetcher, and token cache.

Covers: successful token fetch, fetch failure, cache hit/miss, early refresh,
forced refresh (401 scenario), concurrent debounce, client_secret_basic auth.
"""

import asyncio
import base64
import time
import threading
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from python_a2a.auth.protocol import AuthContext, OAuthCredentials
from python_a2a.auth.protocols.oauth2 import OAuth2ClientCredentialsProtocol
from python_a2a.auth.token_cache import (
    CacheKey,
    TokenCache,
    make_cache_key,
    DEFAULT_REFRESH_SKEW_SECONDS,
)
from python_a2a.auth.token_fetcher import (
    SyncTokenFetcher,
    AsyncTokenFetcher,
    TokenResponse,
    _build_token_request_params,
    _parse_token_response,
    CLIENT_SECRET_BASIC,
    CLIENT_SECRET_POST,
)
from python_a2a.exceptions import A2AAuthenticationError
from python_a2a.models.agent import OAuthFlow, OAuthFlows, SecurityScheme


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


def _make_context(**overrides) -> AuthContext:
    defaults = {
        "agent_url": "https://agent.example.com",
        "security_scheme": OAUTH2_SCHEME,
        "required_scopes": ["a2a:call"],
        "local_config": {
            "client_id": "test-client",
            "client_secret": "test-secret",
        },
    }
    defaults.update(overrides)
    return AuthContext(**defaults)


# ---------------------------------------------------------------------------
# Token request parameter assembly
# ---------------------------------------------------------------------------


class TestBuildTokenRequestParams:
    def test_client_secret_post(self):
        data, headers = _build_token_request_params(
            "cid", "csec", ["s1", "s2"], "aud1", CLIENT_SECRET_POST,
        )
        assert data["grant_type"] == "client_credentials"
        assert data["scope"] == "s1 s2"
        assert data["audience"] == "aud1"
        assert data["client_id"] == "cid"
        assert data["client_secret"] == "csec"
        assert headers == {}

    def test_client_secret_basic(self):
        data, headers = _build_token_request_params(
            "cid", "csec", [], None, CLIENT_SECRET_BASIC,
        )
        assert "client_id" not in data
        assert "client_secret" not in data
        expected = base64.b64encode(b"cid:csec").decode()
        assert headers["Authorization"] == f"Basic {expected}"

    def test_no_scopes_no_audience(self):
        data, _ = _build_token_request_params(
            "c", "s", [], None, CLIENT_SECRET_POST,
        )
        assert "scope" not in data
        assert "audience" not in data


class TestParseTokenResponse:
    def test_success(self):
        resp = _parse_token_response(200, {
            "access_token": "tok123",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "a2a:call",
        })
        assert resp.access_token == "tok123"
        assert resp.token_type == "Bearer"
        assert resp.expires_in == 3600
        assert resp.scope == "a2a:call"
        assert resp.expires_at is not None

    def test_missing_access_token(self):
        with pytest.raises(A2AAuthenticationError, match="missing 'access_token'"):
            _parse_token_response(200, {"token_type": "Bearer"})

    def test_non_200_status(self):
        with pytest.raises(A2AAuthenticationError, match="invalid_client"):
            _parse_token_response(401, {
                "error": "invalid_client",
                "error_description": "Bad credentials",
            })

    def test_no_expires_in(self):
        resp = _parse_token_response(200, {"access_token": "tok"})
        assert resp.expires_in is None
        assert resp.expires_at is None


# ---------------------------------------------------------------------------
# SyncTokenFetcher
# ---------------------------------------------------------------------------


class TestSyncTokenFetcher:
    def test_fetch_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "tok-sync",
            "token_type": "Bearer",
            "expires_in": 1800,
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            fetcher = SyncTokenFetcher()
            result = fetcher.fetch(
                token_url="https://auth.example.com/token",
                client_id="cid",
                client_secret="csec",
                scopes=["a2a:call"],
            )

        assert result.access_token == "tok-sync"
        assert result.expires_in == 1800
        mock_post.assert_called_once()

    def test_fetch_network_error(self):
        import requests as req_lib

        with patch("requests.post", side_effect=req_lib.ConnectionError("fail")):
            fetcher = SyncTokenFetcher()
            with pytest.raises(A2AAuthenticationError, match="failed"):
                fetcher.fetch(
                    token_url="https://auth.example.com/token",
                    client_id="cid",
                    client_secret="csec",
                )


# ---------------------------------------------------------------------------
# TokenCache
# ---------------------------------------------------------------------------


class TestTokenCache:
    def test_get_hit(self):
        cache = TokenCache()
        key = make_cache_key("url", "cid", ["s"], "aud")
        creds = OAuthCredentials(
            protocol_id="oauth2",
            access_token="t",
            expires_at=time.time() + 3600,
        )
        cache.put(key, creds)
        assert cache.get(key) is creds

    def test_get_miss(self):
        cache = TokenCache()
        key = make_cache_key("url", "cid", [], None)
        assert cache.get(key) is None

    def test_get_expired(self):
        cache = TokenCache(refresh_skew_seconds=0)
        key = make_cache_key("url", "cid", [], None)
        creds = OAuthCredentials(
            protocol_id="oauth2",
            access_token="t",
            expires_at=time.time() - 1,  # already expired
        )
        cache.put(key, creds)
        assert cache.get(key) is None

    def test_early_refresh_skew(self):
        """Token within skew window is considered expired."""
        skew = 60
        cache = TokenCache(refresh_skew_seconds=skew)
        key = make_cache_key("url", "cid", [], None)
        creds = OAuthCredentials(
            protocol_id="oauth2",
            access_token="t",
            expires_at=time.time() + 30,  # 30s left < 60s skew
        )
        cache.put(key, creds)
        assert cache.get(key) is None

    def test_no_expires_at_never_expires(self):
        cache = TokenCache()
        key = make_cache_key("url", "cid", [], None)
        creds = OAuthCredentials(
            protocol_id="oauth2",
            access_token="t",
            expires_at=None,
        )
        cache.put(key, creds)
        assert cache.get(key) is creds

    def test_invalidate(self):
        cache = TokenCache()
        key = make_cache_key("url", "cid", [], None)
        creds = OAuthCredentials(protocol_id="oauth2", access_token="t")
        cache.put(key, creds)
        cache.invalidate(key)
        assert cache.get(key) is None

    def test_get_or_refresh_cache_hit(self):
        cache = TokenCache()
        key = make_cache_key("url", "cid", [], None)
        creds = OAuthCredentials(
            protocol_id="oauth2",
            access_token="cached",
            expires_at=time.time() + 3600,
        )
        cache.put(key, creds)

        refresh_called = False

        def _refresh():
            nonlocal refresh_called
            refresh_called = True
            return OAuthCredentials(protocol_id="oauth2", access_token="new")

        result = cache.get_or_refresh(key, _refresh)
        assert result.access_token == "cached"
        assert not refresh_called

    def test_get_or_refresh_cache_miss(self):
        cache = TokenCache()
        key = make_cache_key("url", "cid", [], None)

        def _refresh():
            return OAuthCredentials(
                protocol_id="oauth2",
                access_token="fresh",
                expires_at=time.time() + 3600,
            )

        result = cache.get_or_refresh(key, _refresh)
        assert result.access_token == "fresh"

    def test_get_or_refresh_force_ignores_cache(self):
        cache = TokenCache()
        key = make_cache_key("url", "cid", [], None)
        cache.put(
            key,
            OAuthCredentials(
                protocol_id="oauth2",
                access_token="old",
                expires_at=time.time() + 3600,
            ),
        )

        def _refresh():
            return OAuthCredentials(
                protocol_id="oauth2",
                access_token="forced",
                expires_at=time.time() + 3600,
            )

        result = cache.get_or_refresh(key, _refresh, force=True)
        assert result.access_token == "forced"

    def test_concurrent_debounce(self):
        """Only one refresh call when multiple threads race."""
        cache = TokenCache()
        key = make_cache_key("url", "cid", [], None)
        call_count = 0
        call_lock = threading.Lock()

        def _slow_refresh():
            nonlocal call_count
            time.sleep(0.1)
            with call_lock:
                call_count += 1
            return OAuthCredentials(
                protocol_id="oauth2",
                access_token="debounced",
                expires_at=time.time() + 3600,
            )

        threads = [
            threading.Thread(target=cache.get_or_refresh, args=(key, _slow_refresh))
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With debounce, refresh should be called exactly once (first thread
        # acquires lock, others wait and find cache populated)
        assert call_count == 1
        assert cache.get(key).access_token == "debounced"


# ---------------------------------------------------------------------------
# TokenCache async
# ---------------------------------------------------------------------------


class TestTokenCacheAsync:
    @pytest.mark.anyio
    async def test_get_or_refresh_async_cache_miss(self):
        cache = TokenCache()
        key = make_cache_key("url", "cid", [], None)

        async def _refresh():
            return OAuthCredentials(
                protocol_id="oauth2",
                access_token="async-fresh",
                expires_at=time.time() + 3600,
            )

        result = await cache.get_or_refresh_async(key, _refresh)
        assert result.access_token == "async-fresh"

    @pytest.mark.anyio
    async def test_get_or_refresh_async_force(self):
        cache = TokenCache()
        key = make_cache_key("url", "cid", [], None)
        cache.put(
            key,
            OAuthCredentials(
                protocol_id="oauth2",
                access_token="old",
                expires_at=time.time() + 3600,
            ),
        )

        async def _refresh():
            return OAuthCredentials(
                protocol_id="oauth2",
                access_token="async-forced",
                expires_at=time.time() + 3600,
            )

        result = await cache.get_or_refresh_async(key, _refresh, force=True)
        assert result.access_token == "async-forced"


# ---------------------------------------------------------------------------
# OAuth2ClientCredentialsProtocol
# ---------------------------------------------------------------------------


class TestOAuth2ClientCredentialsProtocol:
    def test_authenticate_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "proto-tok",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with patch("requests.post", return_value=mock_response):
            protocol = OAuth2ClientCredentialsProtocol()
            ctx = _make_context()
            creds = protocol.authenticate(ctx)

        assert isinstance(creds, OAuthCredentials)
        assert creds.access_token == "proto-tok"
        assert creds.protocol_id == "oauth2"

    def test_authenticate_caches_result(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "cached-tok",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            protocol = OAuth2ClientCredentialsProtocol()
            ctx = _make_context()
            creds1 = protocol.authenticate(ctx)
            creds2 = protocol.authenticate(ctx)

        # Second call should hit cache, no extra network call
        assert mock_post.call_count == 1
        assert creds1.access_token == creds2.access_token

    def test_authenticate_missing_client_id(self):
        protocol = OAuth2ClientCredentialsProtocol()
        ctx = _make_context(local_config={"client_secret": "s"})
        with pytest.raises(A2AAuthenticationError, match="client_id"):
            protocol.authenticate(ctx)

    def test_authenticate_missing_token_url(self):
        protocol = OAuth2ClientCredentialsProtocol()
        ctx = _make_context(
            security_scheme=SecurityScheme(type="oauth2"),  # no flows
        )
        with pytest.raises(A2AAuthenticationError, match="token_url"):
            protocol.authenticate(ctx)

    def test_force_refresh_ignores_cache(self):
        call_count = 0

        mock_response = MagicMock()
        mock_response.status_code = 200

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response.json.return_value = {
                "access_token": f"tok-{call_count}",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
            return mock_response

        with patch("requests.post", side_effect=_side_effect):
            protocol = OAuth2ClientCredentialsProtocol()
            ctx = _make_context()
            creds1 = protocol.authenticate(ctx)
            creds2 = protocol.force_refresh(ctx)

        assert creds1.access_token == "tok-1"
        assert creds2.access_token == "tok-2"
        assert call_count == 2

    def test_prepare_headers(self):
        protocol = OAuth2ClientCredentialsProtocol()
        creds = OAuthCredentials(
            protocol_id="oauth2",
            access_token="test-token",
            token_type="Bearer",
        )
        headers = protocol.prepare_headers(creds)
        assert headers == {"Authorization": "Bearer test-token"}

    def test_protocol_id(self):
        protocol = OAuth2ClientCredentialsProtocol()
        assert protocol.protocol_id == "oauth2"

    @pytest.mark.anyio
    async def test_authenticate_async_success(self):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "access_token": "async-tok",
            "token_type": "Bearer",
            "expires_in": 3600,
        })

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            protocol = OAuth2ClientCredentialsProtocol()
            ctx = _make_context()
            creds = await protocol.authenticate_async(ctx)

        assert isinstance(creds, OAuthCredentials)
        assert creds.access_token == "async-tok"
