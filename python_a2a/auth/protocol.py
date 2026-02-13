"""
Core authentication abstractions for the A2A unified auth framework.

Defines the protocol plugin interface (``AuthProtocol``), credential
data model (``AuthCredentials`` and typed subclasses), and the per-request
authentication context (``AuthContext``).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models.agent import SecurityScheme


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@dataclass
class AuthCredentials:
    """Base class for authentication credentials.

    Every credential carries ``protocol_id`` (corresponding to
    ``SecurityScheme.type``) and an optional ``expires_at`` unix timestamp.
    """

    protocol_id: str
    expires_at: Optional[float] = None


@dataclass
class OAuthCredentials(AuthCredentials):
    """Credentials produced by an OAuth 2.0 flow."""

    access_token: str = ""
    token_type: str = "Bearer"


@dataclass
class ApiKeyCredentials(AuthCredentials):
    """Credentials for API-key based authentication."""

    api_key: str = ""
    header_name: str = "X-API-Key"


@dataclass
class HttpCredentials(AuthCredentials):
    """Credentials for HTTP authentication schemes (e.g. Bearer, Basic)."""

    scheme: str = "Bearer"
    token: str = ""


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class AuthContext:
    """Context provided to ``AuthProtocol.authenticate()``.

    Bundles everything a protocol plugin needs to obtain or refresh
    credentials for a specific agent endpoint.
    """

    agent_url: str
    security_scheme: SecurityScheme
    required_scopes: List[str] = field(default_factory=list)
    local_config: Dict[str, Any] = field(default_factory=dict)
    current_credentials: Optional[AuthCredentials] = None


# ---------------------------------------------------------------------------
# AuthProtocol (plugin interface)
# ---------------------------------------------------------------------------


class AuthProtocol(ABC):
    """Abstract interface for an authentication protocol plugin.

    Each implementation handles one ``SecurityScheme.type`` value
    (e.g. ``"oauth2"``, ``"apiKey"``, ``"http"``).

    Attributes:
        protocol_id: Matches ``SecurityScheme.type``.
        injection_layer: How credentials are injected into a request.
            ``"header"`` — via HTTP header (OAuth Bearer, API Key in header).
            ``"query"``  — via URL query parameter.
            ``"transport"`` — via transport layer (e.g. mTLS SSL context).
    """

    @property
    @abstractmethod
    def protocol_id(self) -> str:
        """SecurityScheme type this protocol handles."""
        ...

    @property
    def injection_layer(self) -> str:
        """Credential injection mechanism. Default ``"header"``."""
        return "header"

    @abstractmethod
    def authenticate(self, context: AuthContext) -> AuthCredentials:
        """Execute authentication (e.g. token fetch) and return credentials."""
        ...

    @abstractmethod
    def prepare_headers(self, credentials: AuthCredentials) -> Dict[str, str]:
        """Generate HTTP headers from credentials for a single request."""
        ...

    def validate_credentials(self, credentials: AuthCredentials) -> bool:
        """Check whether *credentials* are still valid (not expired).

        Default implementation checks ``expires_at`` against current time.
        """
        if credentials.expires_at is None:
            return True
        import time

        return time.time() < credentials.expires_at
