#!/usr/bin/env python
"""
Reusable OAuth + A2A test server for end-to-end auth flows.

This module provides a standalone local server that combines:
- OAuth Authorization Server endpoints:
  - GET /oauth/authorize
  - POST /oauth/token
- A2A Resource Server endpoints:
  - GET /.well-known/agent.json
  - POST /tasks/send
  - POST /a2a/tasks/send

It is designed for local E2E tests and examples, not production use.

Interactive login credentials:
- username: demo_user
- password: demo_password
"""

import argparse
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

from python_a2a.auth.dpop_verifier import DPoPProofVerifier, InMemoryJTIReplayStore
from python_a2a.auth.verifiers import (
    ApiKeyVerifier,
    InsufficientScopeError,
    JWTTokenVerifier,
    MultiProtocolAuthBackend,
)

SCOPE_A2A_CALL = "a2a:call"
JWT_KEY_DEFAULT = "e2e-secret-key-should-be-at-least-32-bytes"
CLIENT_ID_DEFAULT = "demo-client"
CLIENT_SECRET_DEFAULT = "demo-secret"
API_KEY_DEFAULT = "demo-e2e-api-key"
DEMO_USERNAME = "demo_user"
DEMO_PASSWORD = "demo_password"


@dataclass
class OAuthA2ATestState:
    """Mutable runtime state for the OAuth+A2A test server."""

    host: str
    port: int
    jwt_key: str = JWT_KEY_DEFAULT
    api_key: str = API_KEY_DEFAULT
    client_id: str = CLIENT_ID_DEFAULT
    client_secret: str = CLIENT_SECRET_DEFAULT
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
        """Issue a signed HS256 JWT for local test use."""
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


class _OAuthA2AHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying shared test state."""

    def __init__(self, server_address: tuple[str, int], state: OAuthA2ATestState) -> None:
        self.state = state
        super().__init__(server_address, _OAuthA2AHandler)


class _OAuthA2AHandler(BaseHTTPRequestHandler):
    """Request handler for OAuth+A2A test endpoints."""

    server: _OAuthA2AHTTPServer

    def do_GET(self) -> None:
        if self.path.startswith("/.well-known/agent.json"):
            self._handle_agent_card()
            return
        if self.path.startswith("/oauth/authorize"):
            self._handle_authorize()
            return
        self._json_response(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path.startswith("/oauth/login"):
            self._handle_login()
            return
        if self.path.startswith("/oauth/token"):
            self._handle_token()
            return
        if self.path.startswith("/tasks/send") or self.path.startswith("/a2a/tasks/send"):
            self._handle_tasks_send()
            return
        self._json_response(404, {"error": "not_found"})

    def _state(self) -> OAuthA2ATestState:
        return self.server.state

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

        # Backward-compatible auto-approve mode for non-interactive E2E tests.
        auto_approve = query.get("auto_approve", ["0"])[0] in {"1", "true", "yes"}
        if not auto_approve:
            self._render_login_page(
                client_id=client_id,
                redirect_uri=redirect_uri,
                state_value=state_value,
                code_challenge=code_challenge,
                scope=scope,
            )
            return

        self._issue_auth_code_redirect(
            redirect_uri=redirect_uri,
            state_value=state_value,
            code_challenge=code_challenge,
            scope=scope,
        )

    def _handle_login(self) -> None:
        form = self._read_form_body()
        username = form.get("username", "")
        password = form.get("password", "")
        client_id = form.get("client_id", "")
        redirect_uri = form.get("redirect_uri", "")
        state_value = form.get("state", "")
        code_challenge = form.get("code_challenge", "")
        scope = form.get("scope", SCOPE_A2A_CALL)

        if username != DEMO_USERNAME or password != DEMO_PASSWORD:
            self._render_login_page(
                client_id=client_id,
                redirect_uri=redirect_uri,
                state_value=state_value,
                code_challenge=code_challenge,
                scope=scope,
                error="Invalid credentials. Try demo_user / demo_password.",
            )
            return

        state = self._state()
        if client_id != state.client_id or redirect_uri != state.redirect_uri:
            self._json_response(400, {"error": "invalid_request"})
            return
        if not state_value or not code_challenge:
            self._json_response(400, {"error": "invalid_request"})
            return

        self._issue_auth_code_redirect(
            redirect_uri=redirect_uri,
            state_value=state_value,
            code_challenge=code_challenge,
            scope=scope,
        )

    def _issue_auth_code_redirect(
        self,
        *,
        redirect_uri: str,
        state_value: str,
        code_challenge: str,
        scope: str,
    ) -> None:
        state = self._state()
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

    def _render_login_page(
        self,
        *,
        client_id: str,
        redirect_uri: str,
        state_value: str,
        code_challenge: str,
        scope: str,
        error: Optional[str] = None,
    ) -> None:
        error_block = ""
        if error:
            error_block = f'<p style="color:#b91c1c;">{error}</p>'

        html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>OAuth Test Login</title>
  </head>
  <body style="font-family: sans-serif; max-width: 560px; margin: 40px auto;">
    <h2>OAuth Test Server Login</h2>
    <p>Use credentials: <code>{DEMO_USERNAME}</code> / <code>{DEMO_PASSWORD}</code></p>
    {error_block}
    <form method="post" action="/oauth/login">
      <label>Username <input name="username" /></label><br /><br />
      <label>Password <input type="password" name="password" /></label>
      <input type="hidden" name="client_id" value="{client_id}" />
      <input type="hidden" name="redirect_uri" value="{redirect_uri}" />
      <input type="hidden" name="state" value="{state_value}" />
      <input type="hidden" name="code_challenge" value="{code_challenge}" />
      <input type="hidden" name="scope" value="{scope}" />
      <br /><br />
      <button type="submit">Sign in</button>
    </form>
  </body>
</html>"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
            del state.auth_codes[code]
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
        self,
        status: int,
        payload: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
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


class OAuthA2ATestServer:
    """Reusable local OAuth+A2A test server wrapper."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        redirect_uri: str = "http://127.0.0.1:3031/callback",
        client_id: str = CLIENT_ID_DEFAULT,
        client_secret: str = CLIENT_SECRET_DEFAULT,
        api_key: str = API_KEY_DEFAULT,
        jwt_key: str = JWT_KEY_DEFAULT,
    ) -> None:
        self._host = host
        self._port = port
        self._server: Optional[_OAuthA2AHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._state = OAuthA2ATestState(
            host=host,
            port=port,
            redirect_uri=redirect_uri,
            client_id=client_id,
            client_secret=client_secret,
            api_key=api_key,
            jwt_key=jwt_key,
        )

    @property
    def state(self) -> OAuthA2ATestState:
        return self._state

    @property
    def base_url(self) -> str:
        return self._state.base_url

    def start(self) -> "OAuthA2ATestServer":
        """Start the server in a background thread."""
        if self._server is not None:
            return self
        self._server = _OAuthA2AHTTPServer((self._host, self._port), self._state)
        bound_host, bound_port = self._server.server_address
        self._state.host = str(bound_host)
        self._state.port = int(bound_port)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the server and wait for thread exit."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def __enter__(self) -> "OAuthA2ATestServer":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reusable OAuth+A2A local test server."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8008, help="Bind port.")
    parser.add_argument(
        "--redirect-uri",
        default="http://127.0.0.1:3031/callback",
        help="Expected redirect URI for authorization_code flow.",
    )
    args = parser.parse_args()

    server = OAuthA2ATestServer(
        host=args.host, port=args.port, redirect_uri=args.redirect_uri
    ).start()
    print(f"OAuth+A2A test server running at {server.base_url}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
