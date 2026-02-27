"""
Server-side credential verification.

Framework-agnostic verifier chain for validating request credentials.
First phase: synchronous ``verify()`` + JWT local validation (no I/O).

Introspection (opaque token -> AS endpoint) is deferred as a future
enhancement requiring ``async verify_async()``.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..exceptions import A2AAuthenticationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Access principal
# ---------------------------------------------------------------------------


@dataclass
class AccessPrincipal:
    """Identity + scopes extracted from a verified credential.

    Attributes:
        subject: Unique identifier (e.g. ``client_id``, ``sub`` claim).
        scopes: Set of granted scopes (may be empty).
        issuer: Token issuer (e.g. AS URL).
        claims: Full decoded claims dict (JWT) or metadata dict (API key).
    """

    subject: str
    scopes: Set[str] = field(default_factory=set)
    issuer: Optional[str] = None
    claims: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InsufficientScopeError(A2AAuthenticationError):
    """Raised when credentials are valid but scopes are insufficient.

    Server adapters should map this to **403 Forbidden** with a
    ``WWW-Authenticate: Bearer error="insufficient_scope"`` header.
    """

    def __init__(self, required: List[str], granted: Set[str]) -> None:
        self.required = required
        self.granted = granted
        missing = set(required) - granted
        super().__init__(
            f"Insufficient scope: required={required}, "
            f"granted={sorted(granted)}, missing={sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# CredentialVerifier interface
# ---------------------------------------------------------------------------


class CredentialVerifier(ABC):
    """Abstract base for request credential verifiers.

    Implementations validate a single authentication scheme (e.g. JWT
    Bearer, API Key).
    """

    @abstractmethod
    def verify(self, headers: Dict[str, str]) -> Optional[AccessPrincipal]:
        """Validate credentials in request headers.

        Returns:
            An ``AccessPrincipal`` on success, ``None`` if this verifier
            does not recognise the credentials (allowing the next
            verifier in the chain to try).
        """
        ...

    def get_challenge(self) -> Optional[str]:
        """Return a ``WWW-Authenticate`` challenge string for this verifier.

        Verifiers for schemes that do not use HTTP authentication (e.g.
        API Key in header) should return ``None`` (the default).
        """
        return None


# ---------------------------------------------------------------------------
# JWT verifier
# ---------------------------------------------------------------------------


class JWTTokenVerifier(CredentialVerifier):
    """Verify ``Authorization: Bearer <jwt>`` using local JWT validation.

    Validates: signature, ``exp``, ``iss``, ``aud``, and extracts ``scope``
    from the token claims.

    Requires the ``PyJWT`` library (``pip install PyJWT[crypto]``).

    Args:
        secret_or_public_key: The secret (HMAC) or public key (RSA/EC)
            used to verify the JWT signature.
        algorithms: Allowed signing algorithms (default ``["RS256"]``).
        issuer: Expected ``iss`` claim (optional).
        audience: Expected ``aud`` claim (optional).
        realm: Realm string for the ``WWW-Authenticate`` challenge.
    """

    def __init__(
        self,
        secret_or_public_key: Any,
        algorithms: Optional[List[str]] = None,
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
        realm: str = "a2a",
    ) -> None:
        self._key = secret_or_public_key
        self._algorithms = algorithms or ["RS256"]
        self._issuer = issuer
        self._audience = audience
        self._realm = realm

    def verify(self, headers: Dict[str, str]) -> Optional[AccessPrincipal]:
        auth_header = headers.get("Authorization") or headers.get("authorization")
        if not auth_header:
            return None

        parts = auth_header.split(None, 1)
        if len(parts) != 2:
            return None

        scheme = parts[0].lower()
        if scheme not in ("bearer", "dpop"):
            return None

        token = parts[1]
        return self._decode_and_validate(token)

    def get_challenge(self) -> Optional[str]:
        return f'Bearer realm="{self._realm}"'

    def _decode_and_validate(self, token: str) -> Optional[AccessPrincipal]:
        try:
            import jwt
        except ImportError:
            logger.error(
                "PyJWT is not installed; cannot verify JWT tokens. "
                "Install with: pip install PyJWT[crypto]"
            )
            return None

        decode_options: Dict[str, Any] = {}
        kwargs: Dict[str, Any] = {
            "key": self._key,
            "algorithms": self._algorithms,
            "options": decode_options,
        }
        if self._issuer:
            kwargs["issuer"] = self._issuer
        if self._audience:
            kwargs["audience"] = self._audience

        try:
            claims = jwt.decode(token, **kwargs)
        except jwt.ExpiredSignatureError:
            logger.debug("JWT token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.debug("JWT token is invalid")
            return None

        # Extract scopes from 'scope' claim (space-separated string)
        scope_str = claims.get("scope", "")
        scopes = set(scope_str.split()) if scope_str else set()

        return AccessPrincipal(
            subject=claims.get("sub", claims.get("client_id", "unknown")),
            scopes=scopes,
            issuer=claims.get("iss"),
            claims=claims,
        )


# ---------------------------------------------------------------------------
# API Key verifier
# ---------------------------------------------------------------------------


class ApiKeyVerifier(CredentialVerifier):
    """Verify API key passed in a request header.

    Checks the dedicated header (default ``X-API-Key``) first.  When
    ``bearer_fallback`` is enabled and the dedicated header is absent,
    the verifier also accepts ``Authorization: Bearer <key>`` as an API
    key, aligning with MCP ``APIKeyVerifier`` semantics.

    When using ``bearer_fallback`` together with ``JWTTokenVerifier`` in
    a ``MultiProtocolAuthBackend``, place ``JWTTokenVerifier`` **before**
    this verifier so that real JWT tokens are handled by the correct
    verifier first.

    Args:
        valid_keys: Set of accepted API key strings.
        header_name: Header to read (default ``X-API-Key``).
        scopes: Scopes to grant when API key is valid (default all).
        bearer_fallback: If ``True``, fall back to reading
            ``Authorization: Bearer <value>`` when the dedicated header
            is absent.
    """

    def __init__(
        self,
        valid_keys: Set[str],
        header_name: str = "X-API-Key",
        scopes: Optional[Set[str]] = None,
        bearer_fallback: bool = False,
    ) -> None:
        self._valid_keys = valid_keys
        self._header_name = header_name
        self._scopes = scopes or set()
        self._bearer_fallback = bearer_fallback

    def verify(self, headers: Dict[str, str]) -> Optional[AccessPrincipal]:
        key_value = self._extract_from_dedicated_header(headers)
        source = self._header_name

        if key_value is None and self._bearer_fallback:
            key_value = self._extract_from_bearer(headers)
            source = "Authorization"

        if key_value is None:
            return None

        if key_value not in self._valid_keys:
            return None

        return AccessPrincipal(
            subject=f"apikey:{key_value[:8]}...",
            scopes=set(self._scopes),
            claims={"api_key_header": source},
        )

    def get_challenge(self) -> Optional[str]:
        return None

    def _extract_from_dedicated_header(
        self, headers: Dict[str, str],
    ) -> Optional[str]:
        target = self._header_name.lower()
        for name, value in headers.items():
            if name.lower() == target:
                return value
        return None

    @staticmethod
    def _extract_from_bearer(headers: Dict[str, str]) -> Optional[str]:
        auth = headers.get("Authorization") or headers.get("authorization")
        if not auth:
            return None
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
        return None


# ---------------------------------------------------------------------------
# Multi-protocol auth backend
# ---------------------------------------------------------------------------


class MultiProtocolAuthBackend:
    """Chains multiple ``CredentialVerifier`` instances.

    Walks verifiers in order; the first one that returns a non-None
    ``AccessPrincipal`` wins.

    Exempt paths (e.g. the agent card endpoint) bypass verification entirely.

    Args:
        verifiers: Ordered list of verifier instances.
        exempt_paths: Paths that skip authentication (e.g. ``{"/.well-known/agent.json"}``).
    """

    # Default paths that should be exempt from auth
    DEFAULT_EXEMPT_PATHS = {
        "/.well-known/agent.json",
        "/agent.json",
        "/a2a/agent.json",
    }

    def __init__(
        self,
        verifiers: List[CredentialVerifier],
        exempt_paths: Optional[Set[str]] = None,
    ) -> None:
        assert verifiers, "At least one verifier is required"
        self._verifiers = verifiers
        self._exempt_paths = exempt_paths if exempt_paths is not None else self.DEFAULT_EXEMPT_PATHS

    def verify(
        self,
        headers: Dict[str, str],
        path: str,
        required_scopes: Optional[List[str]] = None,
    ) -> Optional[AccessPrincipal]:
        """Verify request credentials.

        Args:
            headers: Request headers (case-sensitive as received).
            path: Request path (used for exempt check).
            required_scopes: If specified, the principal must have all
                these scopes; otherwise ``InsufficientScopeError`` is raised.

        Returns:
            An ``AccessPrincipal`` on success, ``None`` if no verifier
            matched (caller should return 401).

        Raises:
            InsufficientScopeError: If credentials are valid but scopes
                are insufficient (caller should return 403).
        """
        if path in self._exempt_paths:
            return AccessPrincipal(subject="anonymous", scopes=set())

        for verifier in self._verifiers:
            principal = verifier.verify(headers)
            if principal is not None:
                if required_scopes:
                    missing = set(required_scopes) - principal.scopes
                    if missing:
                        raise InsufficientScopeError(
                            required=required_scopes,
                            granted=principal.scopes,
                        )
                return principal

        return None

    def get_www_authenticate(self) -> str:
        """Aggregate ``WWW-Authenticate`` challenges from all verifiers.

        Only non-None challenges are included.
        """
        challenges = []
        for verifier in self._verifiers:
            challenge = verifier.get_challenge()
            if challenge is not None:
                challenges.append(challenge)
        return ", ".join(challenges) if challenges else "Bearer"
