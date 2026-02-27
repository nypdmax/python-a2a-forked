"""
Server-side DPoP (Demonstrating Proof-of-Possession) verification.

RFC 9449: OAuth 2.0 Demonstrating Proof of Possession (DPoP).
Provides ``DPoPProofVerifier`` for validating DPoP proof JWTs,
``InMemoryJTIReplayStore`` for jti replay protection, and
``extract_dpop_proof`` for case-insensitive header extraction.

Aligned with the MCP SDK's ``DPoPProofVerifier`` verification rules.
"""

import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set
from urllib.parse import urlparse, urlunparse

from ..exceptions import A2AAuthenticationError
from .dpop import compute_jwk_thumbprint as _shared_compute_thumbprint

logger = logging.getLogger(__name__)

SUPPORTED_ALGORITHMS: Set[str] = {
    "ES256", "ES384", "ES512",
    "RS256", "RS384", "RS512",
    "PS256", "PS384", "PS512",
}
PRIVATE_KEY_FIELDS: Set[str] = {"d", "p", "q", "dp", "dq", "qi", "k"}
DEFAULT_IAT_WINDOW = 300


class DPoPVerificationError(A2AAuthenticationError):
    """DPoP verification failure with an error code.

    ``error_code`` follows RFC 9449 error semantics.
    """

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        super().__init__(message)


@dataclass
class DPoPProofInfo:
    """Validated DPoP proof payload."""

    jti: str
    htm: str
    htu: str
    iat: int
    ath: Optional[str]
    nonce: Optional[str]
    jwk: Dict[str, Any]
    jwk_thumbprint: str


class InMemoryJTIReplayStore:
    """In-memory jti replay store with automatic eviction.

    Not suitable for distributed systems; use a shared cache (e.g. Redis)
    for multi-instance deployments.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._store: Dict[str, float] = {}
        self._max_size = max_size

    def check_and_store(self, jti: str, exp_time: float) -> bool:
        """Return ``True`` if *jti* is new; ``False`` if it's a replay."""
        now = time.time()
        if len(self._store) > self._max_size * 0.9:
            self._store = {k: v for k, v in self._store.items() if v > now}
        if jti in self._store:
            return False
        self._store[jti] = exp_time
        return True


class DPoPProofVerifier:
    """Verify DPoP proof JWTs per RFC 9449 §4.3.

    Validation steps:
    1. Parse JWT header — check ``typ``, ``alg``, ``jwk``.
    2. Verify signature using the embedded public JWK.
    3. Validate required claims: ``jti``, ``htm``, ``htu``, ``iat``.
    4. Match ``htm`` / ``htu`` against the actual request.
    5. Check ``iat`` within the configured time window.
    6. Replay protection via ``jti_store``.
    7. If an access token is provided, verify ``ath``.

    Args:
        jti_store: Optional replay store. When ``None`` replay checking is
            skipped (not recommended for production).
        iat_window: Maximum age of ``iat`` in seconds (default 300).
    """

    def __init__(
        self,
        *,
        jti_store: Optional[InMemoryJTIReplayStore] = None,
        iat_window: int = DEFAULT_IAT_WINDOW,
    ) -> None:
        self._jti_store = jti_store
        self._iat_window = iat_window

    def verify(
        self,
        dpop_proof: str,
        http_method: str,
        http_uri: str,
        *,
        access_token: Optional[str] = None,
    ) -> DPoPProofInfo:
        """Verify a DPoP proof (synchronous).

        Raises:
            DPoPVerificationError: On any verification failure.
        """
        import jwt as pyjwt
        from jwt import PyJWK

        try:
            header = pyjwt.get_unverified_header(dpop_proof)
        except pyjwt.exceptions.DecodeError as exc:
            raise DPoPVerificationError(
                "invalid_dpop_proof", f"Malformed JWT: {exc}",
            ) from exc

        if header.get("typ") != "dpop+jwt":
            raise DPoPVerificationError("invalid_dpop_proof", "Invalid typ")

        alg = header.get("alg")
        if not alg or alg == "none" or alg not in SUPPORTED_ALGORITHMS:
            raise DPoPVerificationError(
                "invalid_dpop_proof", f"Invalid algorithm: {alg}",
            )

        jwk_raw = header.get("jwk")
        if not jwk_raw or not isinstance(jwk_raw, dict):
            raise DPoPVerificationError(
                "invalid_dpop_proof", "Missing or invalid jwk",
            )
        if PRIVATE_KEY_FIELDS & set(jwk_raw.keys()):
            raise DPoPVerificationError(
                "invalid_dpop_proof", "jwk contains private key material",
            )

        try:
            payload = pyjwt.decode(
                dpop_proof,
                key=PyJWK.from_dict(jwk_raw),
                algorithms=[alg],
                options={
                    "verify_signature": True,
                    "verify_exp": False,
                    "verify_nbf": False,
                    "verify_iat": False,
                    "verify_aud": False,
                    "verify_iss": False,
                    "require": [],
                },
            )
        except pyjwt.exceptions.InvalidSignatureError as exc:
            raise DPoPVerificationError(
                "invalid_dpop_proof", "Signature verification failed",
            ) from exc
        except pyjwt.exceptions.DecodeError as exc:
            raise DPoPVerificationError(
                "invalid_dpop_proof", f"Decode failed: {exc}",
            ) from exc

        for claim in ("jti", "htm", "htu", "iat"):
            if claim not in payload:
                raise DPoPVerificationError(
                    "invalid_dpop_proof", f"Missing required claim: {claim}",
                )

        jti = payload["jti"]
        htm = payload["htm"]
        htu = payload["htu"]
        iat = payload["iat"]

        if not isinstance(jti, str) or not jti:
            raise DPoPVerificationError(
                "invalid_dpop_proof", "jti must be a non-empty string",
            )

        if htm.upper() != http_method.upper():
            raise DPoPVerificationError("invalid_dpop_proof", "htm mismatch")

        if htu != _normalize_uri(http_uri):
            raise DPoPVerificationError("invalid_dpop_proof", "htu mismatch")

        now = time.time()
        if not isinstance(iat, (int, float)) or abs(now - iat) > self._iat_window:
            raise DPoPVerificationError("invalid_dpop_proof", "iat out of window")

        if self._jti_store is not None:
            if not self._jti_store.check_and_store(jti, now + self._iat_window):
                raise DPoPVerificationError("invalid_dpop_proof", "Replay detected")

        ath_value = payload.get("ath")
        if access_token is not None:
            expected_ath = _compute_ath(access_token)
            if ath_value != expected_ath:
                raise DPoPVerificationError("invalid_dpop_proof", "ath mismatch")

        thumbprint = _compute_thumbprint(jwk_raw)

        return DPoPProofInfo(
            jti=jti,
            htm=htm,
            htu=htu,
            iat=int(iat),
            ath=ath_value,
            nonce=payload.get("nonce"),
            jwk=jwk_raw,
            jwk_thumbprint=thumbprint,
        )


def extract_dpop_proof(headers: Dict[str, str]) -> Optional[str]:
    """Extract the DPoP proof from request headers (case-insensitive)."""
    for name, value in headers.items():
        if name.lower() == "dpop":
            return value
    return None


def _normalize_uri(uri: str) -> str:
    parsed = urlparse(uri)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _compute_ath(token: str) -> str:
    digest = hashlib.sha256(token.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def _compute_thumbprint(jwk: Dict[str, Any]) -> str:
    try:
        return _shared_compute_thumbprint(jwk)
    except ValueError as exc:
        raise DPoPVerificationError(
            "invalid_dpop_proof", str(exc),
        ) from exc
