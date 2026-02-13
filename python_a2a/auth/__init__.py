"""
A2A unified authentication framework.

Client-side abstractions:
    AuthProtocol, AuthCredentials, AuthContext,
    AuthProtocolRegistry, SelectedRequirement, SchemeBinding,
    AgentCardSecurityResolver, SecurityRequirement, SchemeEntry.
"""

from .protocol import (
    AuthContext,
    AuthCredentials,
    AuthProtocol,
    ApiKeyCredentials,
    HttpCredentials,
    OAuthCredentials,
)
from .registry import (
    AuthProtocolRegistry,
    SchemeBinding,
    SelectedRequirement,
)
from .resolver import (
    AgentCardSecurityResolver,
    SchemeEntry,
    SecurityRequirement,
)
from .provider import UnifiedAuthProvider
from .verifiers import (
    AccessPrincipal,
    ApiKeyVerifier,
    CredentialVerifier,
    InsufficientScopeError,
    JWTTokenVerifier,
    MultiProtocolAuthBackend,
)

__all__ = [
    "AuthContext",
    "AuthCredentials",
    "AuthProtocol",
    "ApiKeyCredentials",
    "HttpCredentials",
    "OAuthCredentials",
    "AuthProtocolRegistry",
    "SchemeBinding",
    "SelectedRequirement",
    "AgentCardSecurityResolver",
    "SchemeEntry",
    "SecurityRequirement",
    "UnifiedAuthProvider",
    "AccessPrincipal",
    "ApiKeyVerifier",
    "CredentialVerifier",
    "InsufficientScopeError",
    "JWTTokenVerifier",
    "MultiProtocolAuthBackend",
]
