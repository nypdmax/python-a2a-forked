"""Built-in authentication protocol implementations."""

from .apikey import ApiKeyProtocol
from .oauth2 import OAuth2AuthorizationCodeProtocol, OAuth2ClientCredentialsProtocol

__all__ = [
    "ApiKeyProtocol",
    "OAuth2AuthorizationCodeProtocol",
    "OAuth2ClientCredentialsProtocol",
]
