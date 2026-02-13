"""
Agent-related models for the A2A protocol.

Includes security types aligned with A2A v0.3.0 specification
(securitySchemes/security following OpenAPI 3.0 Security Scheme Object).
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .base import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Security types (A2A v0.3.0 / OpenAPI 3.0)
# ---------------------------------------------------------------------------

@dataclass
class OAuthFlow:
    """Represents an OAuth 2.0 flow definition (OpenAPI 3.0 OAuth Flow Object)."""

    token_url: str
    authorization_url: Optional[str] = None
    refresh_url: Optional[str] = None
    scopes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"tokenUrl": self.token_url}
        if self.authorization_url is not None:
            result["authorizationUrl"] = self.authorization_url
        if self.refresh_url is not None:
            result["refreshUrl"] = self.refresh_url
        if self.scopes:
            result["scopes"] = dict(self.scopes)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OAuthFlow":
        return cls(
            token_url=data.get("tokenUrl", ""),
            authorization_url=data.get("authorizationUrl"),
            refresh_url=data.get("refreshUrl"),
            scopes=data.get("scopes", {}),
        )


@dataclass
class OAuthFlows:
    """Represents OAuth 2.0 flow definitions (OpenAPI 3.0 OAuth Flows Object)."""

    client_credentials: Optional[OAuthFlow] = None
    authorization_code: Optional[OAuthFlow] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.client_credentials is not None:
            result["clientCredentials"] = self.client_credentials.to_dict()
        if self.authorization_code is not None:
            result["authorizationCode"] = self.authorization_code.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OAuthFlows":
        client_creds_data = data.get("clientCredentials")
        auth_code_data = data.get("authorizationCode")
        return cls(
            client_credentials=(
                OAuthFlow.from_dict(client_creds_data)
                if client_creds_data is not None
                else None
            ),
            authorization_code=(
                OAuthFlow.from_dict(auth_code_data)
                if auth_code_data is not None
                else None
            ),
        )


@dataclass
class SecurityScheme:
    """Represents a security scheme (OpenAPI 3.0 Security Scheme Object).

    ``type`` must be one of: ``apiKey``, ``http``, ``oauth2``,
    ``openIdConnect``, ``mutualTLS``.

    Fields prefixed with ``x-`` in the source JSON are preserved in
    ``extra_fields`` so that vendor extensions (e.g. enterprise-specific
    metadata URLs) remain accessible.
    """

    type: str

    # apiKey-specific
    in_location: Optional[str] = None
    name: Optional[str] = None

    # http-specific
    scheme: Optional[str] = None

    # oauth2-specific
    flows: Optional[OAuthFlows] = None

    # openIdConnect-specific
    open_id_connect_url: Optional[str] = None

    # common
    description: Optional[str] = None

    # vendor extensions (x-* fields)
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": self.type}
        if self.in_location is not None:
            result["in"] = self.in_location
        if self.name is not None:
            result["name"] = self.name
        if self.scheme is not None:
            result["scheme"] = self.scheme
        if self.flows is not None:
            result["flows"] = self.flows.to_dict()
        if self.open_id_connect_url is not None:
            result["openIdConnectUrl"] = self.open_id_connect_url
        if self.description is not None:
            result["description"] = self.description
        for key, value in self.extra_fields.items():
            result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityScheme":
        extra: Dict[str, Any] = {}
        known_keys = {
            "type", "in", "name", "scheme", "flows",
            "openIdConnectUrl", "description",
        }
        for key, value in data.items():
            if key not in known_keys:
                extra[key] = value

        flows_data = data.get("flows")
        return cls(
            type=data.get("type", ""),
            in_location=data.get("in"),
            name=data.get("name"),
            scheme=data.get("scheme"),
            flows=OAuthFlows.from_dict(flows_data) if flows_data is not None else None,
            open_id_connect_url=data.get("openIdConnectUrl"),
            description=data.get("description"),
            extra_fields=extra,
        )


@dataclass
class AgentSkill(BaseModel):
    """Represents a skill in an A2A agent card"""
    name: str
    description: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    input_modes: List[str] = field(default_factory=lambda: ["text/plain"])
    output_modes: List[str] = field(default_factory=lambda: ["text/plain"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags
        }
        
        if self.examples:
            result["examples"] = self.examples
            
        if self.input_modes:
            result["inputModes"] = self.input_modes
            
        if self.output_modes:
            result["outputModes"] = self.output_modes
            
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentSkill':
        """Create an AgentSkill from a dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            examples=data.get("examples", []),
            input_modes=data.get("inputModes", ["text/plain"]),
            output_modes=data.get("outputModes", ["text/plain"])
        )


@dataclass
class AgentCard(BaseModel):
    """Represents an A2A agent card for discovery.

    Security model (A2A v0.3.0):
        ``security_schemes`` maps scheme names to ``SecurityScheme`` definitions
        (serialised as camelCase ``securitySchemes``).
        ``security`` is the list of security requirements (OR-of-AND), each item
        mapping scheme name -> required scopes.

    Field priority:
        If ``security_schemes`` and ``security`` are present (not None/empty),
        they take full precedence over the legacy ``authentication`` field.
        Only when both are absent does the resolver fall back to ``authentication``.
    """

    name: str
    description: str
    url: str
    version: str = "1.0.0"
    protocol_version: str = "0.3.0"  # A2A protocol version
    preferred_transport: str = "JSONRPC"  # A2A specification default
    authentication: Optional[str] = None
    capabilities: Dict[str, Any] = field(default_factory=lambda: {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False
    })
    default_input_modes: List[str] = field(default_factory=lambda: ["text/plain"])
    default_output_modes: List[str] = field(default_factory=lambda: ["text/plain"])
    skills: List[AgentSkill] = field(default_factory=list)
    provider: Optional[str] = None
    documentation_url: Optional[str] = None
    # A2A v0.3.0 security fields
    security_schemes: Optional[Dict[str, SecurityScheme]] = None
    security: Optional[List[Dict[str, List[str]]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "protocolVersion": self.protocol_version,
            "preferredTransport": self.preferred_transport,
            "capabilities": self.capabilities,
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "skills": [skill.to_dict() for skill in self.skills],
        }

        if self.authentication:
            result["authentication"] = self.authentication

        if self.provider:
            result["provider"] = self.provider

        if self.documentation_url:
            result["documentationUrl"] = self.documentation_url

        if self.security_schemes is not None:
            result["securitySchemes"] = {
                name: scheme.to_dict()
                for name, scheme in self.security_schemes.items()
            }

        if self.security is not None:
            result["security"] = self.security

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """Create an AgentCard from a dictionary."""
        skills = [
            AgentSkill.from_dict(skill)
            for skill in data.get("skills", [])
        ]

        # Parse securitySchemes
        raw_schemes = data.get("securitySchemes")
        security_schemes: Optional[Dict[str, SecurityScheme]] = None
        if raw_schemes is not None and isinstance(raw_schemes, dict):
            security_schemes = {
                name: SecurityScheme.from_dict(scheme_data)
                for name, scheme_data in raw_schemes.items()
            }

        security = data.get("security")

        # Warn on conflict between legacy and new fields
        authentication = data.get("authentication")
        if authentication and security_schemes:
            logger.warning(
                "AgentCard has both 'authentication' and 'securitySchemes'; "
                "securitySchemes/security take precedence"
            )

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            url=data.get("url", ""),
            version=data.get("version", "1.0.0"),
            protocol_version=data.get("protocolVersion", "0.3.0"),
            preferred_transport=data.get("preferredTransport", "JSONRPC"),
            authentication=authentication,
            capabilities=data.get("capabilities", {
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": False,
            }),
            default_input_modes=data.get("defaultInputModes", ["text/plain"]),
            default_output_modes=data.get("defaultOutputModes", ["text/plain"]),
            skills=skills,
            provider=data.get("provider"),
            documentation_url=data.get("documentationUrl"),
            security_schemes=security_schemes,
            security=security,
        )
    
    def validate_a2a_protocol(self) -> List[str]:
        """
        Validate agent card against A2A protocol specification.
        
        Returns:
            List of protocol violations (empty if compliant)
        """
        violations = []
        
        # A2A specification required fields
        required_fields = ['name', 'description', 'url', 'version', 'protocolVersion', 'skills', 'capabilities', 'defaultInputModes', 'defaultOutputModes']
        
        for field in required_fields:
            field_name = field.replace('protocolVersion', 'protocol_version').replace('defaultInputModes', 'default_input_modes').replace('defaultOutputModes', 'default_output_modes')
            field_value = getattr(self, field_name, None)
            
            if field_value is None:
                violations.append(f"A2A protocol requires field: {field}")
            elif field in ['name', 'description', 'url', 'version', 'protocolVersion'] and not field_value:
                violations.append(f"A2A protocol requires non-empty field: {field}")
            # For lists (skills, defaultInputModes, defaultOutputModes), empty lists are valid
        
        # Check capabilities is a dictionary
        if not isinstance(self.capabilities, dict):
            violations.append("A2A protocol requires capabilities to be an object")
        
        return violations