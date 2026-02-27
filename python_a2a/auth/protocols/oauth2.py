"""
OAuth 2.0 protocol implementations.

Provides:
  - ``OAuth2ClientCredentialsProtocol`` — client_credentials grant.
  - ``OAuth2AuthorizationCodeProtocol`` — authorization_code grant with PKCE.
"""

import hmac
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

from ...exceptions import A2AAuthenticationError
from ...models.agent import SecurityScheme
from ..pkce import PKCEParameters, generate_state
from ..protocol import AuthContext, AuthCredentials, AuthProtocol, OAuthCredentials
from ..token_cache import CacheKey, TokenCache, make_cache_key
from ..token_fetcher import (
    DEFAULT_CLIENT_AUTH_METHOD,
    AsyncTokenFetcher,
    SyncTokenFetcher,
    TokenResponse,
)

logger = logging.getLogger(__name__)


class OAuth2ClientCredentialsProtocol(AuthProtocol):
    """OAuth 2.0 client_credentials grant protocol.

    This protocol:
    1. Reads ``token_url`` from ``SecurityScheme.flows.client_credentials``.
    2. Reads ``client_id`` / ``client_secret`` from ``AuthContext.local_config``.
    3. Fetches a token via ``SyncTokenFetcher`` (or async variant).
    4. Caches the token in a ``TokenCache`` with automatic refresh.
    """

    def __init__(
        self,
        token_cache: Optional[TokenCache] = None,
        sync_fetcher: Optional[SyncTokenFetcher] = None,
        async_fetcher: Optional[AsyncTokenFetcher] = None,
    ) -> None:
        self._cache = token_cache or TokenCache()
        self._sync_fetcher = sync_fetcher or SyncTokenFetcher()
        self._async_fetcher = async_fetcher or AsyncTokenFetcher()

    @property
    def protocol_id(self) -> str:
        return "oauth2"

    @property
    def token_cache(self) -> TokenCache:
        return self._cache

    def authenticate(self, context: AuthContext) -> OAuthCredentials:
        """Synchronously obtain OAuth credentials (with caching)."""
        token_url = self._extract_token_url(context.security_scheme)
        client_id = self._require_config(context, "client_id")
        client_secret = self._require_config(context, "client_secret")
        audience = context.local_config.get("audience")
        client_auth_method = context.local_config.get(
            "client_auth_method", DEFAULT_CLIENT_AUTH_METHOD,
        )

        cache_key = make_cache_key(
            token_url, client_id, context.required_scopes, audience,
        )

        def _refresh() -> OAuthCredentials:
            response = self._sync_fetcher.fetch(
                token_url=token_url,
                client_id=client_id,
                client_secret=client_secret,
                scopes=context.required_scopes,
                audience=audience,
                client_auth_method=client_auth_method,
            )
            return self._to_credentials(response)

        return self._cache.get_or_refresh(cache_key, _refresh)

    async def authenticate_async(self, context: AuthContext) -> OAuthCredentials:
        """Asynchronously obtain OAuth credentials (with caching)."""
        token_url = self._extract_token_url(context.security_scheme)
        client_id = self._require_config(context, "client_id")
        client_secret = self._require_config(context, "client_secret")
        audience = context.local_config.get("audience")
        client_auth_method = context.local_config.get(
            "client_auth_method", DEFAULT_CLIENT_AUTH_METHOD,
        )

        cache_key = make_cache_key(
            token_url, client_id, context.required_scopes, audience,
        )

        async def _refresh() -> OAuthCredentials:
            response = await self._async_fetcher.fetch(
                token_url=token_url,
                client_id=client_id,
                client_secret=client_secret,
                scopes=context.required_scopes,
                audience=audience,
                client_auth_method=client_auth_method,
            )
            return self._to_credentials(response)

        return await self._cache.get_or_refresh_async(cache_key, _refresh)

    def force_refresh(self, context: AuthContext) -> OAuthCredentials:
        """Synchronous forced refresh (ignores cache), for 401 retry."""
        token_url = self._extract_token_url(context.security_scheme)
        client_id = self._require_config(context, "client_id")
        client_secret = self._require_config(context, "client_secret")
        audience = context.local_config.get("audience")
        client_auth_method = context.local_config.get(
            "client_auth_method", DEFAULT_CLIENT_AUTH_METHOD,
        )

        cache_key = make_cache_key(
            token_url, client_id, context.required_scopes, audience,
        )

        def _refresh() -> OAuthCredentials:
            response = self._sync_fetcher.fetch(
                token_url=token_url,
                client_id=client_id,
                client_secret=client_secret,
                scopes=context.required_scopes,
                audience=audience,
                client_auth_method=client_auth_method,
            )
            return self._to_credentials(response)

        return self._cache.get_or_refresh(cache_key, _refresh, force=True)

    async def force_refresh_async(self, context: AuthContext) -> OAuthCredentials:
        """Async forced refresh (ignores cache), for 401 retry."""
        token_url = self._extract_token_url(context.security_scheme)
        client_id = self._require_config(context, "client_id")
        client_secret = self._require_config(context, "client_secret")
        audience = context.local_config.get("audience")
        client_auth_method = context.local_config.get(
            "client_auth_method", DEFAULT_CLIENT_AUTH_METHOD,
        )

        cache_key = make_cache_key(
            token_url, client_id, context.required_scopes, audience,
        )

        async def _refresh() -> OAuthCredentials:
            response = await self._async_fetcher.fetch(
                token_url=token_url,
                client_id=client_id,
                client_secret=client_secret,
                scopes=context.required_scopes,
                audience=audience,
                client_auth_method=client_auth_method,
            )
            return self._to_credentials(response)

        return await self._cache.get_or_refresh_async(cache_key, _refresh, force=True)

    def prepare_headers(self, credentials: AuthCredentials) -> Dict[str, str]:
        assert isinstance(credentials, OAuthCredentials)
        token_type = credentials.token_type or "Bearer"
        return {"Authorization": f"{token_type} {credentials.access_token}"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_token_url(scheme: SecurityScheme) -> str:
        if scheme.flows and scheme.flows.client_credentials:
            token_url = scheme.flows.client_credentials.token_url
            if token_url:
                return token_url
        raise A2AAuthenticationError(
            "SecurityScheme does not contain a client_credentials flow "
            "with a token_url"
        )

    @staticmethod
    def _require_config(context: AuthContext, key: str) -> str:
        value = context.local_config.get(key)
        if not value:
            raise A2AAuthenticationError(
                f"Missing required config '{key}' in local_config"
            )
        return value

    @staticmethod
    def _to_credentials(response: TokenResponse) -> OAuthCredentials:
        return OAuthCredentials(
            protocol_id="oauth2",
            access_token=response.access_token,
            token_type=response.token_type,
            expires_at=response.expires_at,
        )


class OAuth2AuthorizationCodeProtocol(AuthProtocol):
    """OAuth 2.0 authorization_code grant with PKCE.

    Flow:
    1. Generate PKCE pair and high-entropy state.
    2. Build ``/authorize`` URL and invoke ``redirect_handler``.
    3. Await ``callback_handler`` → ``(code, state)``.
    4. Validate state (constant-time).
    5. Exchange code + code_verifier for tokens at ``token_url``.

    ``AuthContext`` fields used:
        - ``security_scheme.flows.authorization_code`` → endpoints
        - ``redirect_uri``, ``redirect_handler``, ``callback_handler``
        - ``local_config["client_id"]``, ``local_config["client_secret"]``
    """

    def __init__(
        self,
        token_cache: Optional[TokenCache] = None,
        sync_fetcher: Optional[SyncTokenFetcher] = None,
    ) -> None:
        self._cache = token_cache or TokenCache()
        self._sync_fetcher = sync_fetcher or SyncTokenFetcher()

    @property
    def protocol_id(self) -> str:
        return "oauth2"

    def authenticate(self, context: AuthContext) -> OAuthCredentials:
        authorization_url, token_url = self._extract_endpoints(context.security_scheme)
        client_id = self._require_config(context, "client_id")
        client_secret = self._require_config(context, "client_secret")
        redirect_uri = context.redirect_uri
        client_auth_method = context.local_config.get(
            "client_auth_method", DEFAULT_CLIENT_AUTH_METHOD,
        )

        if not redirect_uri:
            raise A2AAuthenticationError(
                "redirect_uri is required for authorization_code flow"
            )
        if context.redirect_handler is None:
            raise A2AAuthenticationError(
                "redirect_handler is required for authorization_code flow"
            )
        if context.callback_handler is None:
            raise A2AAuthenticationError(
                "callback_handler is required for authorization_code flow"
            )

        pkce = PKCEParameters.generate()
        state = generate_state()

        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": pkce.code_challenge,
            "code_challenge_method": "S256",
        }
        if context.required_scopes:
            params["scope"] = " ".join(context.required_scopes)

        auth_url_with_params = f"{authorization_url}?{urlencode(params)}"

        context.redirect_handler(auth_url_with_params)
        code, returned_state = context.callback_handler()

        if not hmac.compare_digest(state, returned_state):
            raise A2AAuthenticationError(
                "OAuth2 state mismatch — possible CSRF attack"
            )

        response = self._sync_fetcher.exchange_code(
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret,
            code=code,
            redirect_uri=redirect_uri,
            code_verifier=pkce.code_verifier,
            client_auth_method=client_auth_method,
        )
        return self._to_credentials(response)

    def prepare_headers(self, credentials: AuthCredentials) -> Dict[str, str]:
        assert isinstance(credentials, OAuthCredentials)
        token_type = credentials.token_type or "Bearer"
        return {"Authorization": f"{token_type} {credentials.access_token}"}

    @staticmethod
    def _extract_endpoints(scheme: SecurityScheme) -> Tuple[str, str]:
        """Return (authorization_url, token_url) from the authorization_code flow."""
        if scheme.flows and scheme.flows.authorization_code:
            flow = scheme.flows.authorization_code
            authorization_url = flow.authorization_url
            token_url = flow.token_url
            if not authorization_url:
                raise A2AAuthenticationError(
                    "authorization_code flow requires authorization_url"
                )
            if not token_url:
                raise A2AAuthenticationError(
                    "authorization_code flow requires token_url"
                )
            return authorization_url, token_url
        raise A2AAuthenticationError(
            "SecurityScheme does not contain an authorization_code flow"
        )

    @staticmethod
    def _require_config(context: AuthContext, key: str) -> str:
        value = context.local_config.get(key)
        if not value:
            raise A2AAuthenticationError(
                f"Missing required config '{key}' in local_config"
            )
        return value

    @staticmethod
    def _to_credentials(response: TokenResponse) -> OAuthCredentials:
        return OAuthCredentials(
            protocol_id="oauth2",
            access_token=response.access_token,
            token_type=response.token_type,
            expires_at=response.expires_at,
        )
