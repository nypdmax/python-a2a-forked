"""
Token fetcher implementations for OAuth 2.0 client_credentials grant.

Provides ``SyncTokenFetcher`` (uses ``requests``) and
``AsyncTokenFetcher`` (uses ``aiohttp``), both sharing the same
parameter assembly logic.
"""

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..exceptions import A2AAuthenticationError

logger = logging.getLogger(__name__)

# Client authentication methods
CLIENT_SECRET_BASIC = "client_secret_basic"
CLIENT_SECRET_POST = "client_secret_post"

DEFAULT_CLIENT_AUTH_METHOD = CLIENT_SECRET_POST


@dataclass
class TokenResponse:
    """Parsed token endpoint response."""

    access_token: str
    token_type: str
    expires_in: Optional[int]
    scope: Optional[str]
    raw: Dict[str, Any]

    @property
    def expires_at(self) -> Optional[float]:
        """Absolute unix timestamp when the token expires."""
        if self.expires_in is not None:
            return time.time() + self.expires_in
        return None


def _build_token_request_params(
    client_id: str,
    client_secret: str,
    scopes: List[str],
    audience: Optional[str],
    client_auth_method: str,
) -> tuple:
    """Build (data dict, headers dict) for a token request.

    Returns:
        A tuple of (form data, extra headers).
    """
    data: Dict[str, str] = {"grant_type": "client_credentials"}

    if scopes:
        data["scope"] = " ".join(scopes)

    if audience:
        data["audience"] = audience

    headers: Dict[str, str] = {}

    if client_auth_method == CLIENT_SECRET_BASIC:
        encoded = base64.b64encode(
            f"{client_id}:{client_secret}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {encoded}"
    else:
        # client_secret_post: include credentials in body
        data["client_id"] = client_id
        data["client_secret"] = client_secret

    return data, headers


def _parse_token_response(
    status_code: int,
    body: Dict[str, Any],
) -> TokenResponse:
    """Parse and validate a token endpoint response.

    Raises:
        A2AAuthenticationError: On non-200 status or missing access_token.
    """
    if status_code != 200:
        error = body.get("error", "unknown_error")
        description = body.get("error_description", "")
        raise A2AAuthenticationError(
            f"Token request failed: {error} - {description} "
            f"(HTTP {status_code})"
        )

    access_token = body.get("access_token")
    if not access_token:
        raise A2AAuthenticationError(
            "Token response missing 'access_token'"
        )

    return TokenResponse(
        access_token=access_token,
        token_type=body.get("token_type", "Bearer"),
        expires_in=body.get("expires_in"),
        scope=body.get("scope"),
        raw=body,
    )


def _build_authcode_token_request_params(
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    client_auth_method: str,
) -> tuple:
    """Build (data dict, headers dict) for an authorization_code token exchange."""
    data: Dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }

    headers: Dict[str, str] = {}

    if client_auth_method == CLIENT_SECRET_BASIC:
        encoded = base64.b64encode(
            f"{client_id}:{client_secret}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {encoded}"
    else:
        data["client_id"] = client_id
        data["client_secret"] = client_secret

    return data, headers


class SyncTokenFetcher:
    """Synchronous token fetcher using ``requests``."""

    def fetch(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        scopes: Optional[List[str]] = None,
        audience: Optional[str] = None,
        client_auth_method: str = DEFAULT_CLIENT_AUTH_METHOD,
        timeout: float = 30.0,
    ) -> TokenResponse:
        """Fetch an access token from the token endpoint.

        Raises:
            A2AAuthenticationError: On network error or bad response.
        """
        import requests

        data, extra_headers = _build_token_request_params(
            client_id, client_secret, scopes or [], audience, client_auth_method,
        )
        try:
            response = requests.post(
                token_url,
                data=data,
                headers=extra_headers,
                timeout=timeout,
            )
            body = response.json()
        except requests.RequestException:
            logger.exception("Token request to %s failed", token_url)
            raise A2AAuthenticationError(
                f"Token request to {token_url} failed"
            )

        return _parse_token_response(response.status_code, body)

    def exchange_code(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        client_auth_method: str = DEFAULT_CLIENT_AUTH_METHOD,
        timeout: float = 30.0,
    ) -> TokenResponse:
        """Exchange an authorization code for tokens (sync)."""
        import requests

        data, extra_headers = _build_authcode_token_request_params(
            client_id, client_secret, code, redirect_uri,
            code_verifier, client_auth_method,
        )
        try:
            response = requests.post(
                token_url, data=data, headers=extra_headers, timeout=timeout,
            )
            body = response.json()
        except requests.RequestException:
            logger.exception("Auth-code token exchange to %s failed", token_url)
            raise A2AAuthenticationError(
                f"Auth-code token exchange to {token_url} failed"
            )

        return _parse_token_response(response.status_code, body)


class AsyncTokenFetcher:
    """Asynchronous token fetcher using ``aiohttp``."""

    async def fetch(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        scopes: Optional[List[str]] = None,
        audience: Optional[str] = None,
        client_auth_method: str = DEFAULT_CLIENT_AUTH_METHOD,
        timeout: float = 30.0,
    ) -> TokenResponse:
        """Fetch an access token from the token endpoint.

        Raises:
            A2AAuthenticationError: On network error or bad response.
        """
        import aiohttp

        data, extra_headers = _build_token_request_params(
            client_id, client_secret, scopes or [], audience, client_auth_method,
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    token_url,
                    data=data,
                    headers=extra_headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    body = await response.json(content_type=None)
                    status_code = response.status
        except aiohttp.ClientError:
            logger.exception("Async token request to %s failed", token_url)
            raise A2AAuthenticationError(
                f"Token request to {token_url} failed"
            )

        return _parse_token_response(status_code, body)

    async def exchange_code(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        client_auth_method: str = DEFAULT_CLIENT_AUTH_METHOD,
        timeout: float = 30.0,
    ) -> TokenResponse:
        """Exchange an authorization code for tokens (async)."""
        import aiohttp

        data, extra_headers = _build_authcode_token_request_params(
            client_id, client_secret, code, redirect_uri,
            code_verifier, client_auth_method,
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    token_url,
                    data=data,
                    headers=extra_headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    body = await response.json(content_type=None)
                    status_code = response.status
        except aiohttp.ClientError:
            logger.exception("Async auth-code token exchange to %s failed", token_url)
            raise A2AAuthenticationError(
                f"Auth-code token exchange to {token_url} failed"
            )

        return _parse_token_response(status_code, body)
