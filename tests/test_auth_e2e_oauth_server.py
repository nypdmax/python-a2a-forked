"""
End-to-end auth matrix tests with an in-process OAuth test server.

This module provides a minimal OAuth authorization server + A2A resource
server (single process) to validate full request chains for:
- OAuth2 client_credentials
- OAuth2 authorization_code + PKCE
- DPoP-enabled OAuth requests
- API key header auth
- API key bearer fallback auth
"""

import base64
import hashlib
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import jwt
import pytest
import requests

from python_a2a.auth.dpop_verifier import DPoPProofVerifier, InMemoryJTIReplayStore
from python_a2a.auth.protocols.apikey import ApiKeyProtocol
from python_a2a.auth.protocols.oauth2 import (
    OAuth2AuthorizationCodeProtocol,
    OAuth2ClientCredentialsProtocol,
)
from python_a2a.auth.provider import UnifiedAuthProvider
from python_a2a.auth.registry import SchemeBinding, SelectedRequirement
from python_a2a.auth.verifiers import (
    ApiKeyVerifier,
    InsufficientScopeError,
    JWTTokenVerifier,
    MultiProtocolAuthBackend,
)
from python_a2a.client.http import A2AClient
from python_a2a.models.agent import OAuthFlow, OAuthFlows, SecurityScheme

JSON_HEADERS = {"Content-Type": "application/json"}
SCOPE_A2A_CALL = "a2a:call"


@dataclass
class E2EServerState:
    host: str
    port: int
    jwt_key: str = "e2e-secret-key-should-be-at-least-32-bytes"
    api_key: str = "demo-e2e-api-key"
    client_id: str = "demo-client"
    client_secret: str = "demo-secret"
    redirect_uri: str = "http://127.0.0.1:3031/callback"
    auth_codes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    auth_code_exchange_count: int = 0
    last_auth_mode: Optional[str] = None
    last_dpop_verified: bool = False
    replay_store: InMemoryJTIReplayStore = field(
        default_factory=InMemoryJTIReplayStore
    )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def issue_jwt(self, subject: str, scope: str = SCOPE_A2A_CALL) -> str:
        now = int(time.time())
        payload = {
            "sub": subject,
            "scope": scope,
            "iss": self.base_url,
            "aud": "a2a-resource",
            "iat": now,
            "exp": now + 3600,
        }
        return jwt.encode(payload, self.jwt_key, algorithm="HS256")


class _OAuthA2AHandler(BaseHTTPRequestHandler):
    server: ThreadingHTTPServer

    def do_GET(self) -> None:
        if self.path.startswith("/.well-known/agent.json"):
            self._handle_agent_card()
            return
        if self.path.startswith("/oauth/authorize"):
            self._handle_authorize()
            return
        self._json_response(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path.startswith("/oauth/token"):
            self._handle_token()
            return
        if self.path.startswith("/tasks/send") or self.path.startswith("/a2a/tasks/send"):
            self._handle_tasks_send()
            return
        self._json_response(404, {"error": "not_found"})

    def _state(self) -> E2EServerState:
        return self.server.state  # type: ignore[attr-defined]

    def _read_form_body(self) -> Dict[str, str]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        parsed = parse_qs(raw, keep_blank_values=True)
        return {key: values[0] for key, values in parsed.items()}

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        if not raw:
            return {}
        return json.loads(raw)

    def _decode_basic_client(self) -> tuple[Optional[str], Optional[str]]:
        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Basic "):
            return None, None
        encoded = auth_header.split(" ", 1)[1]
        decoded = base64.b64decode(encoded).decode("utf-8")
        if ":" not in decoded:
            return None, None
        client_id, client_secret = decoded.split(":", 1)
        return client_id, client_secret

    def _validate_client(self, form: Dict[str, str]) -> bool:
        state = self._state()
        basic_client_id, basic_client_secret = self._decode_basic_client()
        if basic_client_id is not None:
            return (
                basic_client_id == state.client_id
                and basic_client_secret == state.client_secret
            )
        return (
            form.get("client_id") == state.client_id
            and form.get("client_secret") == state.client_secret
        )

    def _handle_agent_card(self) -> None:
        state = self._state()
        card = {
            "name": "E2E Auth Agent",
            "description": "OAuth and auth E2E test agent",
            "url": state.base_url,
            "version": "1.0.0",
            "capabilities": {},
            "skills": [],
            "securitySchemes": {
                "oauthCC": {
                    "type": "oauth2",
                    "flows": {
                        "clientCredentials": {
                            "tokenUrl": f"{state.base_url}/oauth/token",
                            "scopes": {SCOPE_A2A_CALL: "A2A call scope"},
                        }
                    },
                },
                "oauthCode": {
                    "type": "oauth2",
                    "flows": {
                        "authorizationCode": {
                            "authorizationUrl": f"{state.base_url}/oauth/authorize",
                            "tokenUrl": f"{state.base_url}/oauth/token",
                            "scopes": {SCOPE_A2A_CALL: "A2A call scope"},
                        }
                    },
                },
                "apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
            },
            "security": [
                {"oauthCC": [SCOPE_A2A_CALL]},
                {"oauthCode": [SCOPE_A2A_CALL]},
                {"apiKey": []},
            ],
        }
        self._json_response(200, card)

    def _handle_authorize(self) -> None:
        state = self._state()
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        client_id = query.get("client_id", [""])[0]
        redirect_uri = query.get("redirect_uri", [""])[0]
        state_value = query.get("state", [""])[0]
        code_challenge = query.get("code_challenge", [""])[0]
        scope = query.get("scope", [SCOPE_A2A_CALL])[0]

        if (
            client_id != state.client_id
            or redirect_uri != state.redirect_uri
            or not state_value
            or not code_challenge
        ):
            self._json_response(400, {"error": "invalid_request"})
            return

        auth_code = secrets.token_urlsafe(24)
        state.auth_codes[auth_code] = {
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "scope": scope,
        }
        location = f"{redirect_uri}?{urlencode({'code': auth_code, 'state': state_value})}"
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def _handle_token(self) -> None:
        state = self._state()
        form = self._read_form_body()
        if not self._validate_client(form):
            self._json_response(401, {"error": "invalid_client"})
            return

        grant_type = form.get("grant_type", "")
        if grant_type == "client_credentials":
            scope = form.get("scope", SCOPE_A2A_CALL)
            token = state.issue_jwt(subject=state.client_id, scope=scope)
            self._json_response(
                200,
                {
                    "access_token": token,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": scope,
                },
            )
            return

        if grant_type == "authorization_code":
            code = form.get("code", "")
            redirect_uri = form.get("redirect_uri", "")
            code_verifier = form.get("code_verifier", "")
            code_record = state.auth_codes.get(code)
            if not code_record:
                self._json_response(400, {"error": "invalid_grant"})
                return
            if redirect_uri != code_record["redirect_uri"]:
                self._json_response(400, {"error": "invalid_grant"})
                return
            expected_challenge = (
                base64.urlsafe_b64encode(
                    hashlib.sha256(code_verifier.encode("ascii")).digest()
                )
                .decode("ascii")
                .rstrip("=")
            )
            if expected_challenge != code_record["code_challenge"]:
                self._json_response(400, {"error": "invalid_grant"})
                return
            state.auth_code_exchange_count += 1
            scope = code_record["scope"] or SCOPE_A2A_CALL
            token = state.issue_jwt(subject="demo-user", scope=scope)
            self._json_response(
                200,
                {
                    "access_token": token,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": scope,
                },
            )
            return

        self._json_response(400, {"error": "unsupported_grant_type"})

    def _handle_tasks_send(self) -> None:
        state = self._state()
        state.last_auth_mode = None
        state.last_dpop_verified = False

        _ = self._read_json_body()
        headers = {name: value for name, value in self.headers.items()}
        authorization = headers.get("Authorization", "")

        if authorization.lower().startswith("dpop "):
            proof = headers.get("DPoP")
            if not proof:
                self._json_response(
                    401,
                    {"error": "missing_dpop"},
                    {"WWW-Authenticate": "DPoP"},
                )
                return
            token = authorization.split(" ", 1)[1]
            verifier = DPoPProofVerifier(jti_store=state.replay_store)
            try:
                verifier.verify(
                    proof,
                    http_method="POST",
                    http_uri=f"{state.base_url}/tasks/send",
                    access_token=token,
                )
            except Exception:
                self._json_response(
                    401,
                    {"error": "invalid_dpop"},
                    {"WWW-Authenticate": "DPoP"},
                )
                return
            state.last_dpop_verified = True
            state.last_auth_mode = "dpop_jwt"

        backend = MultiProtocolAuthBackend(
            verifiers=[
                JWTTokenVerifier(
                    secret_or_public_key=state.jwt_key,
                    algorithms=["HS256"],
                    audience="a2a-resource",
                ),
                ApiKeyVerifier(
                    valid_keys={state.api_key},
                    scopes={SCOPE_A2A_CALL},
                    bearer_fallback=True,
                ),
            ]
        )

        try:
            principal = backend.verify(
                headers=headers, path="/tasks/send", required_scopes=[SCOPE_A2A_CALL]
            )
        except InsufficientScopeError:
            self._json_response(403, {"error": "insufficient_scope"})
            return

        if principal is None:
            self._json_response(
                401,
                {"error": "unauthorized"},
                {"WWW-Authenticate": backend.get_www_authenticate()},
            )
            return

        if state.last_auth_mode is None:
            source = principal.claims.get("api_key_header")
            if source == "X-API-Key":
                state.last_auth_mode = "x_api_key"
            elif source == "Authorization":
                state.last_auth_mode = "bearer_api_key"
            else:
                state.last_auth_mode = "bearer_jwt"

        response = {
            "jsonrpc": "2.0",
            "result": {
                "id": "task-e2e-1",
                "status": {"state": "completed"},
                "artifacts": [
                    {
                        "parts": [
                            {
                                "type": "text",
                                "text": f"ok:{state.last_auth_mode}",
                            }
                        ]
                    }
                ],
            },
        }
        self._json_response(200, response)

    def _json_response(
        self, status: int, payload: Dict[str, Any], extra_headers: Optional[Dict[str, str]] = None
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


class _OAuthA2AServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: E2EServerState) -> None:
        self.state = state
        super().__init__(server_address, _OAuthA2AHandler)


@pytest.fixture
def oauth_a2a_server() -> E2EServerState:
    host = "127.0.0.1"
    probe = ThreadingHTTPServer((host, 0), _OAuthA2AHandler)
    port = probe.server_address[1]
    probe.server_close()

    state = E2EServerState(host=host, port=port)
    server = _OAuthA2AServer((host, port), state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def _client_credentials_selected(base_url: str) -> SelectedRequirement:
    scheme = SecurityScheme(
        type="oauth2",
        flows=OAuthFlows(
            client_credentials=OAuthFlow(token_url=f"{base_url}/oauth/token")
        ),
    )
    return SelectedRequirement(
        bindings=[
            SchemeBinding(
                scheme_name="oauthCC",
                security_scheme=scheme,
                scopes=[SCOPE_A2A_CALL],
                protocol=OAuth2ClientCredentialsProtocol(),
            )
        ]
    )


def _provider_for_client_credentials(base_url: str) -> UnifiedAuthProvider:
    selected = _client_credentials_selected(base_url)
    return UnifiedAuthProvider(
        selected=selected,
        local_config={"client_id": "demo-client", "client_secret": "demo-secret"},
        agent_url=base_url,
    )


def _provider_for_auth_code(base_url: str, callback_box: Dict[str, str]) -> UnifiedAuthProvider:
    scheme = SecurityScheme(
        type="oauth2",
        flows=OAuthFlows(
            authorization_code=OAuthFlow(
                token_url=f"{base_url}/oauth/token",
                authorization_url=f"{base_url}/oauth/authorize",
            )
        ),
    )

    def redirect_handler(auth_url: str) -> None:
        response = requests.get(auth_url, allow_redirects=False, timeout=5)
        assert response.status_code == 302
        location = response.headers["Location"]
        query = parse_qs(urlparse(location).query)
        callback_box["code"] = query["code"][0]
        callback_box["state"] = query["state"][0]

    def callback_handler() -> tuple[str, str]:
        return callback_box["code"], callback_box["state"]

    selected = SelectedRequirement(
        bindings=[
            SchemeBinding(
                scheme_name="oauthCode",
                security_scheme=scheme,
                scopes=[SCOPE_A2A_CALL],
                protocol=OAuth2AuthorizationCodeProtocol(),
            )
        ]
    )
    return UnifiedAuthProvider(
        selected=selected,
        local_config={
            "client_id": "demo-client",
            "client_secret": "demo-secret",
            "redirect_handler": redirect_handler,
            "callback_handler": callback_handler,
            "redirect_uri": "http://127.0.0.1:3031/callback",
        },
        agent_url=base_url,
    )


def _provider_for_apikey(base_url: str, api_key: str) -> UnifiedAuthProvider:
    scheme = SecurityScheme(type="apiKey", in_location="header", name="X-API-Key")
    selected = SelectedRequirement(
        bindings=[
            SchemeBinding(
                scheme_name="apiKey",
                security_scheme=scheme,
                scopes=[],
                protocol=ApiKeyProtocol(),
            )
        ]
    )
    return UnifiedAuthProvider(
        selected=selected,
        local_config={"api_key": api_key},
        agent_url=base_url,
    )


def test_e2e_client_credentials(oauth_a2a_server: E2EServerState) -> None:
    client = A2AClient(
        oauth_a2a_server.base_url,
        auth_provider=_provider_for_client_credentials(oauth_a2a_server.base_url),
    )
    task = client._create_task("hello")
    result = client._send_task(task)
    assert result.id == "task-e2e-1"
    assert oauth_a2a_server.last_auth_mode == "bearer_jwt"


def test_e2e_authorization_code_pkce(oauth_a2a_server: E2EServerState) -> None:
    callback_box: Dict[str, str] = {}
    client = A2AClient(
        oauth_a2a_server.base_url,
        auth_provider=_provider_for_auth_code(oauth_a2a_server.base_url, callback_box),
    )
    task = client._create_task("hello")
    result = client._send_task(task)
    assert result.id == "task-e2e-1"
    assert oauth_a2a_server.auth_code_exchange_count == 1
    assert oauth_a2a_server.last_auth_mode == "bearer_jwt"


def test_e2e_dpop_client_credentials(oauth_a2a_server: E2EServerState) -> None:
    selected = _client_credentials_selected(oauth_a2a_server.base_url)
    provider_with_dpop = UnifiedAuthProvider(
        selected=selected,
        local_config={
            "client_id": "demo-client",
            "client_secret": "demo-secret",
            "dpop_enabled": True,
            "dpop_algorithm": "ES256",
        },
        agent_url=oauth_a2a_server.base_url,
    )
    client = A2AClient(oauth_a2a_server.base_url, auth_provider=provider_with_dpop)
    task = client._create_task("hello")
    result = client._send_task(task)
    assert result.id == "task-e2e-1"
    assert oauth_a2a_server.last_auth_mode == "dpop_jwt"
    assert oauth_a2a_server.last_dpop_verified is True


def test_e2e_apikey_header(oauth_a2a_server: E2EServerState) -> None:
    client = A2AClient(
        oauth_a2a_server.base_url,
        auth_provider=_provider_for_apikey(
            oauth_a2a_server.base_url, oauth_a2a_server.api_key
        ),
    )
    task = client._create_task("hello")
    result = client._send_task(task)
    assert result.id == "task-e2e-1"
    assert oauth_a2a_server.last_auth_mode == "x_api_key"


def test_e2e_apikey_bearer_fallback(oauth_a2a_server: E2EServerState) -> None:
    client = A2AClient(
        oauth_a2a_server.base_url,
        headers={
            "Authorization": f"Bearer {oauth_a2a_server.api_key}",
            **JSON_HEADERS,
        },
    )
    task = client._create_task("hello")
    result = client._send_task(task)
    assert result.id == "task-e2e-1"
    assert oauth_a2a_server.last_auth_mode == "bearer_api_key"
