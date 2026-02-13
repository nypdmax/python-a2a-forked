"""
Protocol registry and security requirement selection.

``AuthProtocolRegistry`` maintains a mapping of protocol plugins and
selects the best requirement from an AgentCard's ``security`` list.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..exceptions import A2AAuthenticationError
from ..models.agent import SecurityScheme
from .protocol import AuthProtocol

logger = logging.getLogger(__name__)


@dataclass
class SchemeBinding:
    """One scheme binding within a security requirement (AND element).

    Attributes:
        scheme_name: Key in ``securitySchemes``.
        security_scheme: The resolved ``SecurityScheme`` definition.
        scopes: Scopes required for this scheme.
        protocol: The matched client-side ``AuthProtocol`` implementation.
    """

    scheme_name: str
    security_scheme: SecurityScheme
    scopes: List[str]
    protocol: AuthProtocol


@dataclass
class SelectedRequirement:
    """A security requirement selected from the ``security`` list.

    Contains one or more ``SchemeBinding`` entries in an AND relationship
    (all bindings must be satisfied simultaneously).
    """

    bindings: List[SchemeBinding] = field(default_factory=list)

    @property
    def is_composite(self) -> bool:
        """Whether this requirement has multiple AND-combined schemes."""
        return len(self.bindings) > 1


class AuthProtocolRegistry:
    """Registry of authentication protocol plugins.

    Protocols are registered by ``protocol_id`` (matching
    ``SecurityScheme.type``).  ``select_requirement`` walks the
    ``security`` list in order (first = highest priority) and returns
    the first fully satisfiable requirement.

    First-phase limitation: only single-scheme requirements are
    supported. AND-composite requirements are skipped with a warning.
    """

    def __init__(self) -> None:
        self._protocols: Dict[str, AuthProtocol] = {}

    def register(self, protocol: AuthProtocol) -> None:
        """Register a protocol plugin, keyed by ``protocol.protocol_id``."""
        self._protocols[protocol.protocol_id] = protocol

    def get(self, protocol_id: str) -> Optional[AuthProtocol]:
        """Look up a registered protocol by id."""
        return self._protocols.get(protocol_id)

    @property
    def registered_ids(self) -> List[str]:
        """Return list of registered protocol ids."""
        return list(self._protocols.keys())

    def select_requirement(
        self,
        card_security: List[Dict[str, List[str]]],
        card_schemes: Dict[str, SecurityScheme],
    ) -> SelectedRequirement:
        """Select the first satisfiable security requirement.

        Args:
            card_security: The ``security`` list from AgentCard (OR-of-AND).
            card_schemes: The ``securitySchemes`` dict from AgentCard.

        Returns:
            A ``SelectedRequirement`` whose bindings are all matched to
            registered protocols.

        Raises:
            A2AAuthenticationError: If no requirement can be satisfied.
        """
        for requirement in card_security:
            bindings = self._try_resolve_requirement(requirement, card_schemes)
            if bindings is None:
                continue

            selected = SelectedRequirement(bindings=bindings)

            # First-phase limitation: skip AND-composite requirements
            if selected.is_composite:
                scheme_names = [b.scheme_name for b in bindings]
                logger.warning(
                    "Skipping AND-composite security requirement %s; "
                    "only single-scheme requirements are supported in this phase",
                    scheme_names,
                )
                continue

            return selected

        registered = self.registered_ids
        raise A2AAuthenticationError(
            f"No satisfiable security requirement found. "
            f"Registered protocols: {registered}"
        )

    def _try_resolve_requirement(
        self,
        requirement: Dict[str, List[str]],
        card_schemes: Dict[str, SecurityScheme],
    ) -> Optional[List[SchemeBinding]]:
        """Try to resolve all scheme references in a single requirement.

        Returns a list of ``SchemeBinding`` if every scheme in the
        requirement can be matched to a registered protocol, otherwise
        ``None``.
        """
        bindings: List[SchemeBinding] = []
        for scheme_name, scopes in requirement.items():
            scheme = card_schemes.get(scheme_name)
            if scheme is None:
                logger.debug(
                    "Scheme '%s' referenced in security but not defined "
                    "in securitySchemes; skipping requirement",
                    scheme_name,
                )
                return None
            protocol = self._protocols.get(scheme.type)
            if protocol is None:
                logger.debug(
                    "No registered protocol for scheme type '%s' "
                    "(scheme '%s'); skipping requirement",
                    scheme.type,
                    scheme_name,
                )
                return None
            bindings.append(
                SchemeBinding(
                    scheme_name=scheme_name,
                    security_scheme=scheme,
                    scopes=list(scopes),
                    protocol=protocol,
                )
            )
        return bindings if bindings else None
