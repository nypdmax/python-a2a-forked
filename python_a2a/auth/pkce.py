"""
PKCE (Proof Key for Code Exchange) utility for OAuth 2.0 authorization code flow.

Generates ``code_verifier`` and ``code_challenge`` (S256) per RFC 7636,
aligned with the MCP SDK's ``PKCEParameters`` generation rules.
"""

import base64
import hashlib
import secrets
import string
from dataclasses import dataclass

PKCE_CHARSET = string.ascii_letters + string.digits + "-._~"
PKCE_VERIFIER_LENGTH = 128


@dataclass(frozen=True)
class PKCEParameters:
    """Immutable PKCE parameter pair."""

    code_verifier: str
    code_challenge: str

    @classmethod
    def generate(cls) -> "PKCEParameters":
        code_verifier = "".join(
            secrets.choice(PKCE_CHARSET) for _ in range(PKCE_VERIFIER_LENGTH)
        )
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        code_challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        return cls(code_verifier=code_verifier, code_challenge=code_challenge)


def generate_state() -> str:
    """Generate a high-entropy random state parameter."""
    return secrets.token_urlsafe(32)
