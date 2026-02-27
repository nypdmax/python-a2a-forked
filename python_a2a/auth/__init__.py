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
from .dpop import DPoPKeyPair, DPoPProofGenerator, compute_jwk_thumbprint
from .dpop_verifier import (
    DPoPProofInfo,
    DPoPProofVerifier,
    DPoPVerificationError,
    InMemoryJTIReplayStore,
    extract_dpop_proof,
)
from .protocols.apikey import ApiKeyProtocol
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
    "ApiKeyProtocol",
    "DPoPKeyPair",
    "DPoPProofGenerator",
    "DPoPProofInfo",
    "DPoPProofVerifier",
    "DPoPVerificationError",
    "InMemoryJTIReplayStore",
    "compute_jwk_thumbprint",
    "extract_dpop_proof",
    "UnifiedAuthProvider",
    "AccessPrincipal",
    "ApiKeyVerifier",
    "CredentialVerifier",
    "InsufficientScopeError",
    "JWTTokenVerifier",
    "MultiProtocolAuthBackend",
]
