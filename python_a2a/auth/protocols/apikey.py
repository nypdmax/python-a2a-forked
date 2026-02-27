"""
API Key authentication protocol implementation.

Implements ``AuthProtocol`` for ``SecurityScheme.type == "apiKey"``
by reading the key from ``ApiKeyCredentials`` and injecting it into
the configured header (default ``X-API-Key``).
"""

import logging
from typing import Dict

from ...exceptions import A2AAuthenticationError
from ..protocol import ApiKeyCredentials, AuthContext, AuthCredentials, AuthProtocol

logger = logging.getLogger(__name__)


class ApiKeyProtocol(AuthProtocol):
    """API Key authentication protocol.

    This protocol:
    1. Reads ``api_key`` and optional ``header_name`` from
       ``AuthContext.local_config``.
    2. Returns ``ApiKeyCredentials`` with no expiration.
    3. Injects ``{header_name: api_key}`` via ``prepare_headers()``.

    ``local_config`` keys:
        - ``api_key`` (required): The API key string.
        - ``header_name`` (optional): Header to inject. Falls back to
          ``SecurityScheme.name`` if present, otherwise ``X-API-Key``.
    """

    @property
    def protocol_id(self) -> str:
        return "apiKey"

    def authenticate(self, context: AuthContext) -> AuthCredentials:
        config = context.local_config
        api_key = config.get("api_key", "")
        if not api_key:
            raise A2AAuthenticationError(
                "ApiKeyProtocol requires 'api_key' in local_config"
            )

        header_name = self._resolve_header_name(context)

        return ApiKeyCredentials(
            protocol_id=self.protocol_id,
            api_key=api_key,
            header_name=header_name,
        )

    def prepare_headers(self, credentials: AuthCredentials) -> Dict[str, str]:
        assert isinstance(credentials, ApiKeyCredentials)
        return {credentials.header_name: credentials.api_key}

    @staticmethod
    def _resolve_header_name(context: AuthContext) -> str:
        """Determine the header name from config or SecurityScheme."""
        explicit = context.local_config.get("header_name")
        if explicit:
            return explicit

        scheme = context.security_scheme
        if scheme.type == "apiKey" and scheme.name:
            return scheme.name

        return "X-API-Key"
