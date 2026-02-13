"""
Example: Enterprise HMAC-SHA256 authentication protocol.

Demonstrates how to extend the A2A unified auth framework with a
custom enterprise protocol using OpenAPI ``x-`` vendor extensions.

AgentCard ``securitySchemes`` declaration::

    {
        "corpHmacV1": {
            "type": "http",
            "scheme": "hmac-sha256",
            "description": "Corp HMAC-SHA256 signing",
            "x-metadata-url": "https://auth.corp.com/.well-known/hmac-config"
        }
    }

Client side:
    Register ``CorpHmacProtocol`` with the ``AuthProtocolRegistry``.

Server side:
    Add ``CorpHmacVerifier`` to the ``MultiProtocolAuthBackend`` verifier list.
"""

import hashlib
import hmac
import logging
import time
from typing import Any, Dict, List, Optional, Set

from ..protocol import AuthContext, AuthCredentials, AuthProtocol
from ..verifiers import AccessPrincipal, CredentialVerifier

logger = logging.getLogger(__name__)

# Header names used by this protocol
HMAC_AUTH_HEADER = "X-Corp-HMAC-Signature"
HMAC_KEY_ID_HEADER = "X-Corp-Key-ID"
HMAC_TIMESTAMP_HEADER = "X-Corp-Timestamp"

# Maximum allowed clock skew (seconds) for replay protection
MAX_TIMESTAMP_SKEW = 300


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


class CorpHmacCredentials(AuthCredentials):
    """Credentials for Corp HMAC-SHA256 authentication."""

    key_id: str = ""
    secret_key: str = ""

    def __init__(self, key_id: str = "", secret_key: str = "", **kwargs: Any) -> None:
        super().__init__(protocol_id="http", **kwargs)
        self.key_id = key_id
        self.secret_key = secret_key


# ---------------------------------------------------------------------------
# Client-side protocol
# ---------------------------------------------------------------------------


class CorpHmacProtocol(AuthProtocol):
    """Client-side Corp HMAC-SHA256 authentication protocol.

    Reads ``key_id`` and ``secret_key`` from ``AuthContext.local_config``.
    Optionally reads ``x-metadata-url`` from ``SecurityScheme.extra_fields``
    for dynamic config discovery (not implemented in this example).

    ``protocol_id`` is ``"http"`` because the SecurityScheme uses
    ``type: "http", scheme: "hmac-sha256"``. In practice, for a custom
    protocol_id the registry matching logic would need to inspect
    ``scheme`` as well; this example keeps it simple.
    """

    @property
    def protocol_id(self) -> str:
        return "http"

    def authenticate(self, context: AuthContext) -> CorpHmacCredentials:
        key_id = context.local_config.get("key_id", "")
        secret_key = context.local_config.get("secret_key", "")
        if not key_id or not secret_key:
            from ...exceptions import A2AAuthenticationError

            raise A2AAuthenticationError(
                "CorpHmacProtocol requires 'key_id' and 'secret_key' in local_config"
            )

        # Optionally read enterprise metadata URL from vendor extension
        metadata_url = context.security_scheme.extra_fields.get("x-metadata-url")
        if metadata_url:
            logger.debug("Corp HMAC metadata URL: %s (not fetched in example)", metadata_url)

        return CorpHmacCredentials(key_id=key_id, secret_key=secret_key)

    def prepare_headers(self, credentials: AuthCredentials) -> Dict[str, str]:
        assert isinstance(credentials, CorpHmacCredentials)
        timestamp = str(int(time.time()))
        signature = self._compute_signature(
            credentials.secret_key, credentials.key_id, timestamp,
        )
        return {
            HMAC_KEY_ID_HEADER: credentials.key_id,
            HMAC_TIMESTAMP_HEADER: timestamp,
            HMAC_AUTH_HEADER: signature,
        }

    @staticmethod
    def _compute_signature(secret_key: str, key_id: str, timestamp: str) -> str:
        message = f"{key_id}:{timestamp}".encode("utf-8")
        return hmac.new(
            secret_key.encode("utf-8"), message, hashlib.sha256,
        ).hexdigest()


# ---------------------------------------------------------------------------
# Server-side verifier
# ---------------------------------------------------------------------------


class CorpHmacVerifier(CredentialVerifier):
    """Server-side Corp HMAC-SHA256 credential verifier.

    Args:
        valid_keys: Mapping of key_id -> secret_key.
        max_timestamp_skew: Maximum allowed clock skew in seconds.
        scopes: Scopes to grant on successful verification.
    """

    def __init__(
        self,
        valid_keys: Dict[str, str],
        max_timestamp_skew: int = MAX_TIMESTAMP_SKEW,
        scopes: Optional[Set[str]] = None,
    ) -> None:
        self._valid_keys = valid_keys
        self._max_skew = max_timestamp_skew
        self._scopes = scopes or set()

    def verify(self, headers: Dict[str, str]) -> Optional[AccessPrincipal]:
        # Extract headers (case-insensitive)
        signature = self._get_header(headers, HMAC_AUTH_HEADER)
        key_id = self._get_header(headers, HMAC_KEY_ID_HEADER)
        timestamp_str = self._get_header(headers, HMAC_TIMESTAMP_HEADER)

        if not all([signature, key_id, timestamp_str]):
            return None

        # Validate timestamp for replay protection
        try:
            request_time = int(timestamp_str)
        except ValueError:
            logger.debug("Invalid timestamp format: %s", timestamp_str)
            return None

        current_time = int(time.time())
        if abs(current_time - request_time) > self._max_skew:
            logger.debug(
                "Timestamp skew too large: request=%d, now=%d, max=%d",
                request_time, current_time, self._max_skew,
            )
            return None

        # Look up secret key
        secret_key = self._valid_keys.get(key_id)
        if secret_key is None:
            logger.debug("Unknown key_id: %s", key_id)
            return None

        # Verify signature
        expected = CorpHmacProtocol._compute_signature(
            secret_key, key_id, timestamp_str,
        )
        if not hmac.compare_digest(signature, expected):
            logger.debug("HMAC signature mismatch for key_id: %s", key_id)
            return None

        return AccessPrincipal(
            subject=f"hmac:{key_id}",
            scopes=set(self._scopes),
            claims={"key_id": key_id, "auth_method": "hmac-sha256"},
        )

    def get_challenge(self) -> Optional[str]:
        # Custom scheme — no standard WWW-Authenticate challenge
        return None

    @staticmethod
    def _get_header(headers: Dict[str, str], name: str) -> Optional[str]:
        for key, value in headers.items():
            if key.lower() == name.lower():
                return value
        return None
