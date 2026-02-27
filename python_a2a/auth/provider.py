"""
Unified authentication provider for A2A clients.

``UnifiedAuthProvider`` is the lightweight coordinator that sits between
the A2A client (``A2AClient`` / ``StreamingClient``) and the auth
protocol subsystem.  It:

- Generates per-request auth headers from the selected requirement.
- Handles 401 refresh-and-retry logic (one retry).
- Leaves 403 (insufficient_scope) to the caller.
- Optionally injects DPoP proofs when ``dpop_enabled`` is set.
"""

import logging
from typing import Any, Dict, List, Optional

from ..exceptions import A2AAuthenticationError
from ..models.agent import SecurityScheme
from .protocol import AuthContext, AuthCredentials, AuthProtocol, OAuthCredentials
from .registry import SchemeBinding, SelectedRequirement

logger = logging.getLogger(__name__)


class UnifiedAuthProvider:
    """Coordinate per-request auth header generation and 401 refresh.

    Args:
        selected: The ``SelectedRequirement`` from registry selection.
        local_config: Client-local credentials (client_id/secret, api_key, etc.).
        agent_url: The target agent URL.
    """

    def __init__(
        self,
        selected: SelectedRequirement,
        local_config: Dict[str, Any],
        agent_url: str,
    ) -> None:
        self._selected = selected
        self._local_config = local_config
        self._agent_url = agent_url
        self._credentials: Dict[str, AuthCredentials] = {}
        self._retry_attempted = False
        self._dpop_generator: Optional[Any] = None
        self._dpop_nonce: Optional[str] = None

        if local_config.get("dpop_enabled"):
            self._init_dpop(local_config)

    def _init_dpop(self, config: Dict[str, Any]) -> None:
        from .dpop import DPoPKeyPair, DPoPProofGenerator

        algorithm = config.get("dpop_algorithm", "ES256")
        rsa_key_size = config.get("dpop_rsa_key_size", 2048)
        key_pair = DPoPKeyPair.generate(algorithm, rsa_key_size=rsa_key_size)
        self._dpop_generator = DPoPProofGenerator(key_pair)

    # ------------------------------------------------------------------
    # Header generation
    # ------------------------------------------------------------------

    def get_auth_headers(
        self,
        method: Optional[str] = None,
        url: Optional[str] = None,
    ) -> Dict[str, str]:
        """Synchronous: obtain auth headers for all bindings (AND merged).

        When DPoP is enabled, ``method`` and ``url`` are used to generate
        a DPoP proof header and upgrade ``Authorization`` to ``DPoP`` scheme.
        """
        merged: Dict[str, str] = {}
        for binding in self._selected.bindings:
            creds = self._ensure_credentials(binding)
            headers = binding.protocol.prepare_headers(creds)
            merged.update(headers)

        if self._dpop_generator is not None and method and url:
            access_token = self._extract_access_token(merged)
            proof = self._dpop_generator.generate_proof(
                method=method,
                url=url,
                access_token=access_token,
                nonce=self._dpop_nonce,
            )
            merged["DPoP"] = proof
            if access_token:
                merged["Authorization"] = f"DPoP {access_token}"

        return merged

    async def get_auth_headers_async(
        self,
        method: Optional[str] = None,
        url: Optional[str] = None,
    ) -> Dict[str, str]:
        """Asynchronous: obtain auth headers for all bindings (AND merged)."""
        merged: Dict[str, str] = {}
        for binding in self._selected.bindings:
            creds = await self._ensure_credentials_async(binding)
            headers = binding.protocol.prepare_headers(creds)
            merged.update(headers)

        if self._dpop_generator is not None and method and url:
            access_token = self._extract_access_token(merged)
            proof = self._dpop_generator.generate_proof(
                method=method,
                url=url,
                access_token=access_token,
                nonce=self._dpop_nonce,
            )
            merged["DPoP"] = proof
            if access_token:
                merged["Authorization"] = f"DPoP {access_token}"

        return merged

    # ------------------------------------------------------------------
    # 401 handling
    # ------------------------------------------------------------------

    def should_retry_on_401(self, www_authenticate: Optional[str] = None) -> bool:
        """Determine whether a 401 should trigger a refresh-and-retry.

        Returns True at most once per provider lifecycle.
        """
        if self._retry_attempted:
            return False
        # Only retry if we actually have credentials to refresh
        if not self._credentials:
            return False
        return True

    def force_refresh(
        self,
        method: Optional[str] = None,
        url: Optional[str] = None,
    ) -> Dict[str, str]:
        """Synchronous forced refresh of all binding credentials.

        A fresh DPoP proof (new ``jti``) is generated when DPoP is enabled.
        """
        self._retry_attempted = True
        merged: Dict[str, str] = {}
        for binding in self._selected.bindings:
            creds = self._do_authenticate(binding)
            self._credentials[binding.scheme_name] = creds
            headers = binding.protocol.prepare_headers(creds)
            merged.update(headers)

        if self._dpop_generator is not None and method and url:
            access_token = self._extract_access_token(merged)
            proof = self._dpop_generator.generate_proof(
                method=method,
                url=url,
                access_token=access_token,
                nonce=self._dpop_nonce,
            )
            merged["DPoP"] = proof
            if access_token:
                merged["Authorization"] = f"DPoP {access_token}"

        return merged

    async def force_refresh_async(
        self,
        method: Optional[str] = None,
        url: Optional[str] = None,
    ) -> Dict[str, str]:
        """Asynchronous forced refresh of all binding credentials."""
        self._retry_attempted = True
        merged: Dict[str, str] = {}
        for binding in self._selected.bindings:
            creds = await self._do_authenticate_async(binding)
            self._credentials[binding.scheme_name] = creds
            headers = binding.protocol.prepare_headers(creds)
            merged.update(headers)

        if self._dpop_generator is not None and method and url:
            access_token = self._extract_access_token(merged)
            proof = self._dpop_generator.generate_proof(
                method=method,
                url=url,
                access_token=access_token,
                nonce=self._dpop_nonce,
            )
            merged["DPoP"] = proof
            if access_token:
                merged["Authorization"] = f"DPoP {access_token}"

        return merged

    def reset_retry(self) -> None:
        """Reset the retry flag (call after a successful request)."""
        self._retry_attempted = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_context(self, binding: SchemeBinding) -> AuthContext:
        cfg = self._local_config
        return AuthContext(
            agent_url=self._agent_url,
            security_scheme=binding.security_scheme,
            required_scopes=binding.scopes,
            local_config=cfg,
            current_credentials=self._credentials.get(binding.scheme_name),
            grant_type=cfg.get("grant_type", "client_credentials"),
            redirect_uri=cfg.get("redirect_uri"),
            redirect_handler=cfg.get("redirect_handler"),
            callback_handler=cfg.get("callback_handler"),
            dpop_enabled=cfg.get("dpop_enabled", False),
            dpop_algorithm=cfg.get("dpop_algorithm", "ES256"),
            dpop_rsa_key_size=cfg.get("dpop_rsa_key_size", 2048),
        )

    def _ensure_credentials(self, binding: SchemeBinding) -> AuthCredentials:
        cached = self._credentials.get(binding.scheme_name)
        if cached is not None and binding.protocol.validate_credentials(cached):
            return cached
        creds = self._do_authenticate(binding)
        self._credentials[binding.scheme_name] = creds
        return creds

    async def _ensure_credentials_async(self, binding: SchemeBinding) -> AuthCredentials:
        cached = self._credentials.get(binding.scheme_name)
        if cached is not None and binding.protocol.validate_credentials(cached):
            return cached
        creds = await self._do_authenticate_async(binding)
        self._credentials[binding.scheme_name] = creds
        return creds

    def _do_authenticate(self, binding: SchemeBinding) -> AuthCredentials:
        context = self._make_context(binding)
        return binding.protocol.authenticate(context)

    async def _do_authenticate_async(self, binding: SchemeBinding) -> AuthCredentials:
        context = self._make_context(binding)
        protocol = binding.protocol
        if hasattr(protocol, "authenticate_async"):
            return await protocol.authenticate_async(context)
        return protocol.authenticate(context)

    @staticmethod
    def _extract_access_token(headers: Dict[str, str]) -> Optional[str]:
        """Extract the access token from an ``Authorization`` header."""
        auth = headers.get("Authorization", "")
        parts = auth.split(None, 1)
        if len(parts) == 2:
            return parts[1]
        return None
