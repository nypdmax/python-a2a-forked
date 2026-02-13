"""
Unified authentication provider for A2A clients.

``UnifiedAuthProvider`` is the lightweight coordinator that sits between
the A2A client (``A2AClient`` / ``StreamingClient``) and the auth
protocol subsystem.  It:

- Generates per-request auth headers from the selected requirement.
- Handles 401 refresh-and-retry logic (one retry).
- Leaves 403 (insufficient_scope) to the caller.
"""

import logging
import re
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
        # Cache credentials per binding (indexed by scheme_name)
        self._credentials: Dict[str, AuthCredentials] = {}
        self._retry_attempted = False

    # ------------------------------------------------------------------
    # Header generation
    # ------------------------------------------------------------------

    def get_auth_headers(self) -> Dict[str, str]:
        """Synchronous: obtain auth headers for all bindings (AND merged)."""
        merged: Dict[str, str] = {}
        for binding in self._selected.bindings:
            creds = self._ensure_credentials(binding)
            headers = binding.protocol.prepare_headers(creds)
            merged.update(headers)
        return merged

    async def get_auth_headers_async(self) -> Dict[str, str]:
        """Asynchronous: obtain auth headers for all bindings (AND merged)."""
        merged: Dict[str, str] = {}
        for binding in self._selected.bindings:
            creds = await self._ensure_credentials_async(binding)
            headers = binding.protocol.prepare_headers(creds)
            merged.update(headers)
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

    def force_refresh(self) -> Dict[str, str]:
        """Synchronous forced refresh of all binding credentials."""
        self._retry_attempted = True
        merged: Dict[str, str] = {}
        for binding in self._selected.bindings:
            creds = self._do_authenticate(binding)
            self._credentials[binding.scheme_name] = creds
            headers = binding.protocol.prepare_headers(creds)
            merged.update(headers)
        return merged

    async def force_refresh_async(self) -> Dict[str, str]:
        """Asynchronous forced refresh of all binding credentials."""
        self._retry_attempted = True
        merged: Dict[str, str] = {}
        for binding in self._selected.bindings:
            creds = await self._do_authenticate_async(binding)
            self._credentials[binding.scheme_name] = creds
            headers = binding.protocol.prepare_headers(creds)
            merged.update(headers)
        return merged

    def reset_retry(self) -> None:
        """Reset the retry flag (call after a successful request)."""
        self._retry_attempted = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_context(self, binding: SchemeBinding) -> AuthContext:
        return AuthContext(
            agent_url=self._agent_url,
            security_scheme=binding.security_scheme,
            required_scopes=binding.scopes,
            local_config=self._local_config,
            current_credentials=self._credentials.get(binding.scheme_name),
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
        # Use async path if available
        if hasattr(protocol, "authenticate_async"):
            return await protocol.authenticate_async(context)
        return protocol.authenticate(context)
