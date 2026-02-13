"""
AgentCard security resolver.

Extracts ordered security requirements from an AgentCard, handling
the priority between new (``securitySchemes``/``security``) and legacy
(``authentication``) fields.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..models.agent import AgentCard, SecurityScheme

logger = logging.getLogger(__name__)

# Mapping from legacy ``authentication`` string values to equivalent
# SecurityScheme definitions, used for backward compatibility.
_LEGACY_AUTH_MAPPING: Dict[str, SecurityScheme] = {
    "bearer": SecurityScheme(type="http", scheme="bearer"),
    "bearer_token": SecurityScheme(type="http", scheme="bearer"),
    "api_key": SecurityScheme(
        type="apiKey", in_location="header", name="X-API-Key",
    ),
}


@dataclass
class SecurityRequirement:
    """A single security requirement extracted from an AgentCard.

    ``schemes`` maps scheme name -> (SecurityScheme, required scopes).
    The mapping preserves insertion order so that AND semantics are
    deterministic.
    """

    schemes: Dict[str, "SchemeEntry"] = field(default_factory=dict)


@dataclass
class SchemeEntry:
    """One scheme reference within a ``SecurityRequirement``."""

    security_scheme: SecurityScheme
    scopes: List[str] = field(default_factory=list)


class AgentCardSecurityResolver:
    """Resolves security requirements from an ``AgentCard``.

    Priority rules:
        1. If ``security_schemes`` **and** ``security`` are present
           (not None / not empty), use them exclusively.
        2. Otherwise fall back to the legacy ``authentication`` field
           and map it to an equivalent ``SecurityScheme``.
        3. If neither is present, the agent requires no authentication
           and an empty list is returned.
    """

    def resolve(self, card: AgentCard) -> List[SecurityRequirement]:
        """Return ordered security requirements (OR-of-AND).

        The list order reflects priority: first element = highest
        priority, matching the ``security`` list order in the AgentCard.
        """
        if self._has_new_security_fields(card):
            return self._resolve_from_new_fields(card)

        if card.authentication:
            return self._resolve_from_legacy(card)

        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_new_security_fields(card: AgentCard) -> bool:
        """Check whether the card carries v0.3.0 security fields."""
        has_schemes = card.security_schemes is not None and len(card.security_schemes) > 0
        has_security = card.security is not None and len(card.security) > 0
        return has_schemes and has_security

    @staticmethod
    def _resolve_from_new_fields(card: AgentCard) -> List[SecurityRequirement]:
        assert card.security_schemes is not None
        assert card.security is not None

        requirements: List[SecurityRequirement] = []
        for raw_req in card.security:
            entries: Dict[str, SchemeEntry] = {}
            skip = False
            for scheme_name, scopes in raw_req.items():
                scheme_def = card.security_schemes.get(scheme_name)
                if scheme_def is None:
                    logger.warning(
                        "Security requirement references undefined scheme '%s'; "
                        "skipping this requirement",
                        scheme_name,
                    )
                    skip = True
                    break
                entries[scheme_name] = SchemeEntry(
                    security_scheme=scheme_def,
                    scopes=list(scopes),
                )
            if not skip and entries:
                requirements.append(SecurityRequirement(schemes=entries))
        return requirements

    @staticmethod
    def _resolve_from_legacy(card: AgentCard) -> List[SecurityRequirement]:
        assert card.authentication is not None
        auth_value = card.authentication.lower().strip()
        scheme_def = _LEGACY_AUTH_MAPPING.get(auth_value)
        if scheme_def is None:
            logger.warning(
                "Unknown legacy authentication value '%s'; "
                "cannot map to SecurityScheme",
                card.authentication,
            )
            return []

        # Synthesise a single requirement with a generated scheme name
        synthetic_name = f"_legacy_{auth_value}"
        entry = SchemeEntry(security_scheme=scheme_def, scopes=[])
        return [SecurityRequirement(schemes={synthetic_name: entry})]
