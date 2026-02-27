"""
Client-side DPoP (Demonstrating Proof-of-Possession) implementation.

RFC 9449: OAuth 2.0 Demonstrating Proof of Possession (DPoP).
Provides ``DPoPKeyPair``, ``DPoPProofGenerator``, and ``compute_jwk_thumbprint``
for generating DPoP proof JWTs.

Aligned with the MCP SDK's ``DPoPProofGeneratorImpl`` generation rules.
"""

import base64
import hashlib
import json
import logging
import secrets
import time
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

DPoPAlgorithm = Literal["ES256", "RS256"]

_BITS_PER_BYTE = 8
RSA_KEY_SIZE_DEFAULT = 2048
_RSA_PUBLIC_EXPONENT = 65537


def _int_to_base64url(num: int, *, fixed_length: Optional[int] = None) -> str:
    """Encode integer to base64url without padding."""
    if fixed_length is not None:
        size = fixed_length
    else:
        size = (num.bit_length() + _BITS_PER_BYTE - 1) // _BITS_PER_BYTE
    data = num.to_bytes(size, "big")
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


class DPoPKeyPair:
    """DPoP key pair holding a private key and its public JWK representation."""

    def __init__(self, private_key: Any, algorithm: DPoPAlgorithm = "ES256") -> None:
        self._private_key = private_key
        self._algorithm = algorithm
        self._public_jwk = _key_to_jwk(private_key)

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @property
    def public_key_jwk(self) -> Dict[str, Any]:
        return self._public_jwk.copy()

    @classmethod
    def generate(
        cls,
        algorithm: DPoPAlgorithm = "ES256",
        *,
        rsa_key_size: int = RSA_KEY_SIZE_DEFAULT,
    ) -> "DPoPKeyPair":
        """Generate a new DPoP key pair.

        Args:
            algorithm: ``"ES256"`` (default) or ``"RS256"``.
            rsa_key_size: RSA key size in bits (minimum 2048).

        Raises:
            ValueError: If algorithm is unsupported or rsa_key_size < 2048.
        """
        from cryptography.hazmat.primitives.asymmetric.ec import (
            SECP256R1,
            generate_private_key as ec_generate,
        )
        from cryptography.hazmat.primitives.asymmetric.rsa import (
            generate_private_key as rsa_generate,
        )

        if algorithm == "ES256":
            key = ec_generate(SECP256R1())
        elif algorithm == "RS256":
            if rsa_key_size < RSA_KEY_SIZE_DEFAULT:
                raise ValueError(
                    f"RSA key size must be at least {RSA_KEY_SIZE_DEFAULT} bits, "
                    f"got {rsa_key_size}"
                )
            key = rsa_generate(
                public_exponent=_RSA_PUBLIC_EXPONENT, key_size=rsa_key_size,
            )
        else:
            raise ValueError(f"Unsupported DPoP algorithm: {algorithm}")
        return cls(key, algorithm)

    def sign_dpop_jwt(
        self, payload: Dict[str, Any], headers: Dict[str, Any],
    ) -> str:
        """Sign a DPoP JWT with the private key."""
        import jwt

        return jwt.encode(
            payload, self._private_key, algorithm=self._algorithm, headers=headers,
        )


def _key_to_jwk(key: Any) -> Dict[str, Any]:
    """Convert a private key to public JWK (no private components)."""
    from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

    if isinstance(key, EllipticCurvePrivateKey):
        pub = key.public_key()
        nums = pub.public_numbers()
        ec_coord_length = 32
        return {
            "kty": "EC",
            "crv": "P-256",
            "x": _int_to_base64url(nums.x, fixed_length=ec_coord_length),
            "y": _int_to_base64url(nums.y, fixed_length=ec_coord_length),
        }
    assert isinstance(key, RSAPrivateKey)
    pub = key.public_key()
    nums = pub.public_numbers()
    return {
        "kty": "RSA",
        "n": _int_to_base64url(nums.n),
        "e": _int_to_base64url(nums.e),
    }


class DPoPProofGenerator:
    """Generates DPoP proof JWTs per RFC 9449."""

    def __init__(self, key_pair: DPoPKeyPair) -> None:
        self._key_pair = key_pair

    def generate_proof(
        self,
        method: str,
        url: str,
        access_token: Optional[str] = None,
        nonce: Optional[str] = None,
    ) -> str:
        """Create a signed DPoP proof JWT.

        Args:
            method: HTTP method (uppercased in the ``htm`` claim).
            url: Request URL (query/fragment stripped for ``htu``).
            access_token: If present, SHA-256 hash stored as ``ath``.
            nonce: Server-provided nonce to include.
        """
        htu = normalize_htu(url)
        payload: Dict[str, Any] = {
            "jti": secrets.token_urlsafe(32),
            "htm": method.upper(),
            "htu": htu,
            "iat": int(time.time()),
        }
        if access_token:
            payload["ath"] = ath_hash(access_token)
        if nonce:
            payload["nonce"] = nonce

        headers: Dict[str, Any] = {
            "typ": "dpop+jwt",
            "alg": self._key_pair.algorithm,
            "jwk": self._key_pair.public_key_jwk,
        }

        return self._key_pair.sign_dpop_jwt(payload, headers)

    @property
    def public_key_jwk(self) -> Dict[str, Any]:
        return self._key_pair.public_key_jwk


def normalize_htu(uri: str) -> str:
    """Strip query and fragment from URI per RFC 9449 ``htu`` claim."""
    parsed = urlparse(uri)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def ath_hash(access_token: str) -> str:
    """Base64url-encoded SHA-256 hash of the access token."""
    digest = hashlib.sha256(access_token.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def compute_jwk_thumbprint(jwk: Dict[str, Any]) -> str:
    """Compute JWK Thumbprint (RFC 7638) for ``cnf.jkt`` binding."""
    kty = jwk.get("kty")
    if kty == "EC":
        canonical = {"crv": jwk["crv"], "kty": "EC", "x": jwk["x"], "y": jwk["y"]}
    elif kty == "RSA":
        canonical = {"e": jwk["e"], "kty": "RSA", "n": jwk["n"]}
    else:
        raise ValueError(f"Unsupported key type: {kty}")
    data = json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
