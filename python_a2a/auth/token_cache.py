"""
Token cache with expiry-aware refresh and in-process concurrency debounce.

``TokenCache`` stores ``OAuthCredentials`` keyed by
``(token_url, client_id, scopes_key, audience)`` and supports:
- Automatic expiry detection with configurable skew.
- Forced refresh for 401 retry scenarios.
- Per-key ``threading.Lock`` / ``asyncio.Lock`` to debounce concurrent
  refresh attempts within the same process.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple

from .protocol import OAuthCredentials

logger = logging.getLogger(__name__)

# Default: treat token as expired 60 s before actual expiry
DEFAULT_REFRESH_SKEW_SECONDS = 60


@dataclass(frozen=True)
class CacheKey:
    """Immutable key for the token cache."""

    token_url: str
    client_id: str
    scopes_key: str  # sorted, space-joined scopes
    audience: str


def make_cache_key(
    token_url: str,
    client_id: str,
    scopes: list,
    audience: Optional[str] = None,
) -> CacheKey:
    """Create a ``CacheKey`` from parameters."""
    return CacheKey(
        token_url=token_url,
        client_id=client_id,
        scopes_key=" ".join(sorted(scopes)),
        audience=audience or "",
    )


class TokenCache:
    """In-process token cache with concurrency debounce.

    Args:
        refresh_skew_seconds: How many seconds before actual expiry to
            consider a token as needing refresh.
    """

    def __init__(self, refresh_skew_seconds: float = DEFAULT_REFRESH_SKEW_SECONDS) -> None:
        self._refresh_skew = refresh_skew_seconds
        self._store: Dict[CacheKey, OAuthCredentials] = {}
        self._locks: Dict[CacheKey, threading.Lock] = {}
        self._async_locks: Dict[CacheKey, asyncio.Lock] = {}
        self._global_lock = threading.Lock()

    def get(self, key: CacheKey) -> Optional[OAuthCredentials]:
        """Return cached credentials if still valid, else ``None``."""
        creds = self._store.get(key)
        if creds is None:
            return None
        if self._is_expired(creds):
            return None
        return creds

    def put(self, key: CacheKey, credentials: OAuthCredentials) -> None:
        """Store credentials in the cache."""
        self._store[key] = credentials

    def invalidate(self, key: CacheKey) -> None:
        """Remove credentials for the given key."""
        self._store.pop(key, None)

    def get_or_refresh(
        self,
        key: CacheKey,
        refresh_fn: Callable[[], OAuthCredentials],
        force: bool = False,
    ) -> OAuthCredentials:
        """Get cached credentials or refresh synchronously.

        Uses a per-key lock to debounce concurrent refresh calls.
        If ``force`` is True, ignores the cache and always calls ``refresh_fn``.
        """
        if not force:
            cached = self.get(key)
            if cached is not None:
                return cached

        lock = self._get_sync_lock(key)
        with lock:
            # Double-check after acquiring lock (another thread may have refreshed)
            if not force:
                cached = self.get(key)
                if cached is not None:
                    return cached

            credentials = refresh_fn()
            self.put(key, credentials)
            return credentials

    async def get_or_refresh_async(
        self,
        key: CacheKey,
        refresh_fn: Callable[[], Coroutine[Any, Any, OAuthCredentials]],
        force: bool = False,
    ) -> OAuthCredentials:
        """Get cached credentials or refresh asynchronously.

        Uses a per-key ``asyncio.Lock`` to debounce concurrent refresh.
        """
        if not force:
            cached = self.get(key)
            if cached is not None:
                return cached

        lock = self._get_async_lock(key)
        async with lock:
            if not force:
                cached = self.get(key)
                if cached is not None:
                    return cached

            credentials = await refresh_fn()
            self.put(key, credentials)
            return credentials

    def _is_expired(self, credentials: OAuthCredentials) -> bool:
        if credentials.expires_at is None:
            return False
        return time.time() >= (credentials.expires_at - self._refresh_skew)

    def _get_sync_lock(self, key: CacheKey) -> threading.Lock:
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _get_async_lock(self, key: CacheKey) -> asyncio.Lock:
        # asyncio.Lock must be created in the running event loop context.
        # We lazily create and store it; caller is always in async context.
        if key not in self._async_locks:
            self._async_locks[key] = asyncio.Lock()
        return self._async_locks[key]
