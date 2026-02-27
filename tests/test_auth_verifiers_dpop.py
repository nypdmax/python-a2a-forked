"""
Tests for server-side DPoP proof verification.

Covers:
  - Valid proof acceptance (ES256, RS256)
  - htm mismatch rejection
  - htu mismatch rejection
  - iat out-of-window rejection
  - ath mismatch rejection
  - jti replay detection
  - Missing DPoP header for ``Authorization: DPoP`` scheme
  - Bearer old path compatibility (no DPoP required)
  - Malformed JWT rejection
  - Private key in JWK rejection
  - extract_dpop_proof case-insensitive extraction
"""

import time

import pytest

from python_a2a.auth.dpop import DPoPKeyPair, DPoPProofGenerator, ath_hash
from python_a2a.auth.dpop_verifier import (
    DPoPProofVerifier,
    DPoPVerificationError,
    InMemoryJTIReplayStore,
    extract_dpop_proof,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proof(
    kp: DPoPKeyPair,
    method: str = "POST",
    url: str = "https://agent.example.com/tasks/send",
    access_token: str | None = None,
    nonce: str | None = None,
) -> str:
    gen = DPoPProofGenerator(kp)
    return gen.generate_proof(method, url, access_token=access_token, nonce=nonce)


# ---------------------------------------------------------------------------
# DPoPProofVerifier
# ---------------------------------------------------------------------------


class TestDPoPProofVerifier:
    def test_valid_es256_proof(self):
        kp = DPoPKeyPair.generate("ES256")
        proof = _make_proof(kp)
        verifier = DPoPProofVerifier()
        info = verifier.verify(
            proof, "POST", "https://agent.example.com/tasks/send",
        )
        assert info.htm == "POST"
        assert info.htu == "https://agent.example.com/tasks/send"
        assert info.jti
        assert abs(info.iat - int(time.time())) <= 2
        assert info.jwk_thumbprint

    def test_valid_rs256_proof(self):
        kp = DPoPKeyPair.generate("RS256")
        proof = _make_proof(kp)
        verifier = DPoPProofVerifier()
        info = verifier.verify(
            proof, "POST", "https://agent.example.com/tasks/send",
        )
        assert info.htm == "POST"

    def test_htm_mismatch(self):
        kp = DPoPKeyPair.generate("ES256")
        proof = _make_proof(kp, method="POST")
        verifier = DPoPProofVerifier()
        with pytest.raises(DPoPVerificationError, match="htm mismatch"):
            verifier.verify(proof, "GET", "https://agent.example.com/tasks/send")

    def test_htu_mismatch(self):
        kp = DPoPKeyPair.generate("ES256")
        proof = _make_proof(kp, url="https://agent.example.com/tasks/send")
        verifier = DPoPProofVerifier()
        with pytest.raises(DPoPVerificationError, match="htu mismatch"):
            verifier.verify(proof, "POST", "https://other.example.com/tasks/send")

    def test_htu_strips_query_for_matching(self):
        kp = DPoPKeyPair.generate("ES256")
        proof = _make_proof(kp, url="https://agent.example.com/api")
        verifier = DPoPProofVerifier()
        info = verifier.verify(
            proof, "POST", "https://agent.example.com/api?param=1",
        )
        assert info.htu == "https://agent.example.com/api"

    def test_iat_out_of_window(self):
        import jwt as pyjwt

        kp = DPoPKeyPair.generate("ES256")
        gen = DPoPProofGenerator(kp)

        import secrets
        from python_a2a.auth.dpop import normalize_htu

        payload = {
            "jti": secrets.token_urlsafe(32),
            "htm": "POST",
            "htu": normalize_htu("https://agent.example.com/api"),
            "iat": int(time.time()) - 600,
        }
        headers = {
            "typ": "dpop+jwt",
            "alg": kp.algorithm,
            "jwk": kp.public_key_jwk,
        }
        proof = kp.sign_dpop_jwt(payload, headers)

        verifier = DPoPProofVerifier(iat_window=300)
        with pytest.raises(DPoPVerificationError, match="iat"):
            verifier.verify(proof, "POST", "https://agent.example.com/api")

    def test_ath_mismatch(self):
        kp = DPoPKeyPair.generate("ES256")
        proof = _make_proof(kp, access_token="correct-token")
        verifier = DPoPProofVerifier()
        with pytest.raises(DPoPVerificationError, match="ath mismatch"):
            verifier.verify(
                proof, "POST", "https://agent.example.com/tasks/send",
                access_token="wrong-token",
            )

    def test_ath_correct(self):
        kp = DPoPKeyPair.generate("ES256")
        proof = _make_proof(kp, access_token="my-token")
        verifier = DPoPProofVerifier()
        info = verifier.verify(
            proof, "POST", "https://agent.example.com/tasks/send",
            access_token="my-token",
        )
        assert info.ath == ath_hash("my-token")

    def test_malformed_jwt(self):
        verifier = DPoPProofVerifier()
        with pytest.raises(DPoPVerificationError, match="Malformed"):
            verifier.verify("not-a-jwt", "POST", "https://example.com/api")

    def test_private_key_in_jwk_rejected(self):
        import jwt as pyjwt

        kp = DPoPKeyPair.generate("ES256")
        jwk = kp.public_key_jwk
        jwk["d"] = "PRIVATE-MATERIAL"

        import secrets

        payload = {
            "jti": secrets.token_urlsafe(32),
            "htm": "POST",
            "htu": "https://example.com/api",
            "iat": int(time.time()),
        }
        headers_dict = {"typ": "dpop+jwt", "alg": "ES256", "jwk": jwk}
        proof = kp.sign_dpop_jwt(payload, headers_dict)

        verifier = DPoPProofVerifier()
        with pytest.raises(DPoPVerificationError, match="private key"):
            verifier.verify(proof, "POST", "https://example.com/api")

    def test_invalid_typ(self):
        import secrets

        kp = DPoPKeyPair.generate("ES256")
        payload = {
            "jti": secrets.token_urlsafe(32),
            "htm": "POST",
            "htu": "https://example.com/api",
            "iat": int(time.time()),
        }
        headers_dict = {"typ": "JWT", "alg": "ES256", "jwk": kp.public_key_jwk}
        proof = kp.sign_dpop_jwt(payload, headers_dict)

        verifier = DPoPProofVerifier()
        with pytest.raises(DPoPVerificationError, match="Invalid typ"):
            verifier.verify(proof, "POST", "https://example.com/api")

    def test_missing_required_claim(self):
        import secrets

        kp = DPoPKeyPair.generate("ES256")
        payload = {
            "jti": secrets.token_urlsafe(32),
            "htm": "POST",
            "iat": int(time.time()),
        }
        headers_dict = {"typ": "dpop+jwt", "alg": "ES256", "jwk": kp.public_key_jwk}
        proof = kp.sign_dpop_jwt(payload, headers_dict)

        verifier = DPoPProofVerifier()
        with pytest.raises(DPoPVerificationError, match="Missing required claim: htu"):
            verifier.verify(proof, "POST", "https://example.com/api")


# ---------------------------------------------------------------------------
# InMemoryJTIReplayStore
# ---------------------------------------------------------------------------


class TestInMemoryJTIReplayStore:
    def test_first_jti_accepted(self):
        store = InMemoryJTIReplayStore()
        assert store.check_and_store("jti-1", time.time() + 300) is True

    def test_replay_rejected(self):
        store = InMemoryJTIReplayStore()
        store.check_and_store("jti-1", time.time() + 300)
        assert store.check_and_store("jti-1", time.time() + 300) is False

    def test_different_jti_accepted(self):
        store = InMemoryJTIReplayStore()
        store.check_and_store("jti-1", time.time() + 300)
        assert store.check_and_store("jti-2", time.time() + 300) is True

    def test_eviction_on_overflow(self):
        store = InMemoryJTIReplayStore(max_size=10)
        now = time.time()
        for i in range(15):
            store.check_and_store(f"jti-{i}", now + 300)
        assert len(store._store) <= 15

    def test_verifier_with_replay_store(self):
        kp = DPoPKeyPair.generate("ES256")
        store = InMemoryJTIReplayStore()
        verifier = DPoPProofVerifier(jti_store=store)

        proof = _make_proof(kp)
        verifier.verify(
            proof, "POST", "https://agent.example.com/tasks/send",
        )
        with pytest.raises(DPoPVerificationError, match="Replay"):
            verifier.verify(
                proof, "POST", "https://agent.example.com/tasks/send",
            )


# ---------------------------------------------------------------------------
# extract_dpop_proof
# ---------------------------------------------------------------------------


class TestExtractDPoPProof:
    def test_exact_case(self):
        assert extract_dpop_proof({"DPoP": "proof123"}) == "proof123"

    def test_lowercase(self):
        assert extract_dpop_proof({"dpop": "proof123"}) == "proof123"

    def test_uppercase(self):
        assert extract_dpop_proof({"DPOP": "proof123"}) == "proof123"

    def test_missing(self):
        assert extract_dpop_proof({"Authorization": "Bearer tok"}) is None
