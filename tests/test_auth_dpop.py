"""
Tests for client-side DPoP key generation, proof generation, and utilities.

Covers:
  - DPoPKeyPair generation (ES256 / RS256)
  - DPoPProofGenerator proof claims (jti, htm, htu, iat, ath, nonce)
  - htu normalization (strip query/fragment)
  - ath SHA-256 hash
  - JWK thumbprint (RFC 7638)
  - ES256 / RS256 proof signatures
"""

import base64
import hashlib
import json
import time
from typing import Any

import jwt
import pytest

from python_a2a.auth.dpop import (
    DPoPKeyPair,
    DPoPProofGenerator,
    ath_hash,
    compute_jwk_thumbprint,
    normalize_htu,
)


# ---------------------------------------------------------------------------
# DPoPKeyPair
# ---------------------------------------------------------------------------


class TestDPoPKeyPair:
    def test_generate_es256(self):
        kp = DPoPKeyPair.generate("ES256")
        assert kp.algorithm == "ES256"
        jwk = kp.public_key_jwk
        assert jwk["kty"] == "EC"
        assert jwk["crv"] == "P-256"
        assert "x" in jwk and "y" in jwk

    def test_generate_rs256(self):
        kp = DPoPKeyPair.generate("RS256")
        assert kp.algorithm == "RS256"
        jwk = kp.public_key_jwk
        assert jwk["kty"] == "RSA"
        assert "n" in jwk and "e" in jwk

    def test_rs256_key_size_too_small_raises(self):
        with pytest.raises(ValueError, match="at least 2048"):
            DPoPKeyPair.generate("RS256", rsa_key_size=1024)

    def test_unsupported_algorithm_raises(self):
        unsupported_alg: Any = "PS256"
        with pytest.raises(ValueError, match="Unsupported"):
            DPoPKeyPair.generate(unsupported_alg)

    def test_public_key_jwk_returns_copy(self):
        kp = DPoPKeyPair.generate("ES256")
        jwk1 = kp.public_key_jwk
        jwk2 = kp.public_key_jwk
        assert jwk1 == jwk2
        assert jwk1 is not jwk2


# ---------------------------------------------------------------------------
# DPoPProofGenerator – claims
# ---------------------------------------------------------------------------


class TestDPoPProofGenerator:
    def _decode_proof(self, proof: str, kp: DPoPKeyPair) -> dict:
        from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey

        pub = kp._private_key.public_key()
        return jwt.decode(
            proof,
            pub,
            algorithms=[kp.algorithm],
            options={"verify_exp": False, "verify_aud": False},
        )

    def _decode_headers(self, proof: str) -> dict:
        header_segment = proof.split(".")[0]
        padded = header_segment + "=" * (-len(header_segment) % 4)
        return json.loads(base64.urlsafe_b64decode(padded))

    def test_proof_has_required_claims(self):
        kp = DPoPKeyPair.generate("ES256")
        gen = DPoPProofGenerator(kp)
        proof = gen.generate_proof("POST", "https://agent.example.com/tasks/send")

        claims = self._decode_proof(proof, kp)
        assert "jti" in claims
        assert claims["htm"] == "POST"
        assert claims["htu"] == "https://agent.example.com/tasks/send"
        assert abs(claims["iat"] - int(time.time())) <= 2

    def test_proof_headers(self):
        kp = DPoPKeyPair.generate("ES256")
        gen = DPoPProofGenerator(kp)
        proof = gen.generate_proof("GET", "https://example.com/resource")

        headers = self._decode_headers(proof)
        assert headers["typ"] == "dpop+jwt"
        assert headers["alg"] == "ES256"
        assert "jwk" in headers
        assert headers["jwk"]["kty"] == "EC"

    def test_ath_included_when_token_provided(self):
        kp = DPoPKeyPair.generate("ES256")
        gen = DPoPProofGenerator(kp)
        proof = gen.generate_proof(
            "POST", "https://example.com/api", access_token="my-token",
        )
        claims = self._decode_proof(proof, kp)
        expected_ath = ath_hash("my-token")
        assert claims["ath"] == expected_ath

    def test_ath_absent_when_no_token(self):
        kp = DPoPKeyPair.generate("ES256")
        gen = DPoPProofGenerator(kp)
        proof = gen.generate_proof("POST", "https://example.com/api")
        claims = self._decode_proof(proof, kp)
        assert "ath" not in claims

    def test_nonce_included(self):
        kp = DPoPKeyPair.generate("ES256")
        gen = DPoPProofGenerator(kp)
        proof = gen.generate_proof(
            "POST", "https://example.com/api", nonce="server-nonce-123",
        )
        claims = self._decode_proof(proof, kp)
        assert claims["nonce"] == "server-nonce-123"

    def test_jti_unique_per_call(self):
        kp = DPoPKeyPair.generate("ES256")
        gen = DPoPProofGenerator(kp)
        proof1 = gen.generate_proof("POST", "https://example.com/api")
        proof2 = gen.generate_proof("POST", "https://example.com/api")
        jti1 = self._decode_proof(proof1, kp)["jti"]
        jti2 = self._decode_proof(proof2, kp)["jti"]
        assert jti1 != jti2

    def test_htm_uppercased(self):
        kp = DPoPKeyPair.generate("ES256")
        gen = DPoPProofGenerator(kp)
        proof = gen.generate_proof("post", "https://example.com/api")
        claims = self._decode_proof(proof, kp)
        assert claims["htm"] == "POST"

    def test_rs256_proof(self):
        kp = DPoPKeyPair.generate("RS256")
        gen = DPoPProofGenerator(kp)
        proof = gen.generate_proof("GET", "https://example.com/resource")
        claims = self._decode_proof(proof, kp)
        assert claims["htm"] == "GET"
        headers = self._decode_headers(proof)
        assert headers["alg"] == "RS256"


# ---------------------------------------------------------------------------
# htu normalization
# ---------------------------------------------------------------------------


class TestNormalizeHtu:
    def test_strips_query(self):
        assert normalize_htu("https://example.com/path?foo=bar") == "https://example.com/path"

    def test_strips_fragment(self):
        assert normalize_htu("https://example.com/path#section") == "https://example.com/path"

    def test_strips_query_and_fragment(self):
        assert normalize_htu("https://example.com/path?a=1#s") == "https://example.com/path"

    def test_preserves_path(self):
        assert normalize_htu("https://example.com/a/b/c") == "https://example.com/a/b/c"

    def test_preserves_scheme_and_host(self):
        assert normalize_htu("http://localhost:8080/api") == "http://localhost:8080/api"


# ---------------------------------------------------------------------------
# ath hash
# ---------------------------------------------------------------------------


class TestAthHash:
    def test_matches_sha256_base64url(self):
        token = "test-access-token"
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(token.encode("ascii")).digest()
        ).decode().rstrip("=")
        assert ath_hash(token) == expected


# ---------------------------------------------------------------------------
# JWK Thumbprint
# ---------------------------------------------------------------------------


class TestComputeJwkThumbprint:
    def test_ec_thumbprint(self):
        kp = DPoPKeyPair.generate("ES256")
        jwk = kp.public_key_jwk
        tp = compute_jwk_thumbprint(jwk)
        assert isinstance(tp, str)
        assert len(tp) > 10

    def test_rsa_thumbprint(self):
        kp = DPoPKeyPair.generate("RS256")
        jwk = kp.public_key_jwk
        tp = compute_jwk_thumbprint(jwk)
        assert isinstance(tp, str)
        assert len(tp) > 10

    def test_deterministic(self):
        kp = DPoPKeyPair.generate("ES256")
        jwk = kp.public_key_jwk
        assert compute_jwk_thumbprint(jwk) == compute_jwk_thumbprint(jwk)

    def test_unsupported_kty_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            compute_jwk_thumbprint({"kty": "OKP"})
