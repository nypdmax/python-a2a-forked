#!/usr/bin/env python
"""
A2A 认证端到端使用示例：在本地 OAuth 测试服务器上演示多种鉴权方式。

本示例为使用样例（非回归测试）。包含五类场景：
- 四类自动化流程：client_credentials、client_credentials + DPoP、
  X-API-Key、Authorization: Bearer（API Key 回退）
- 一类需用户参与：authorization_code + PKCE（打开浏览器、在登录页输入
  凭据、回调后换取 token 再请求 A2A）

运行：
    python examples/auth/e2e_oauth_matrix_example.py

授权码场景登录凭据：demo_user / demo_password
"""

import threading
import time
import subprocess
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from python_a2a.auth.protocols.apikey import ApiKeyProtocol
from python_a2a.auth.protocols.oauth2 import (
    OAuth2AuthorizationCodeProtocol,
    OAuth2ClientCredentialsProtocol,
)
from python_a2a.auth.provider import UnifiedAuthProvider
from python_a2a.auth.registry import SchemeBinding, SelectedRequirement
from python_a2a.client.http import A2AClient
from python_a2a.models.agent import OAuthFlow, OAuthFlows, SecurityScheme

from oauth_test_server import DEMO_PASSWORD, DEMO_USERNAME, OAuthA2ATestServer

SCOPE_A2A_CALL = "a2a:call"
JSON_HEADERS = {"Content-Type": "application/json"}
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 3031
CALLBACK_PATH = "/callback"


# ---------------------------------------------------------------------------
# 本地回调服务器（仅用于 authorization_code 场景）
# ---------------------------------------------------------------------------


class _CallbackServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, host: str, port: int) -> None:
        self.callback_data: Dict[str, Optional[str]] = {
            "code": None,
            "state": None,
            "error": None,
        }
        super().__init__((host, port), _CallbackHandler)


class _CallbackHandler(BaseHTTPRequestHandler):
    server: _CallbackServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return
        query = parse_qs(parsed.query)
        if "error" in query:
            self.server.callback_data["error"] = query["error"][0]
            self._write_page("Authorization failed", query["error"][0])
            return
        if "code" in query:
            self.server.callback_data["code"] = query["code"][0]
            self.server.callback_data["state"] = query.get("state", [""])[0]
            self._write_page("Authorization complete", "You can close this page now.")
            return
        self.send_response(400)
        self.end_headers()

    def _write_page(self, title: str, message: str) -> None:
        html = f"""<!doctype html>
<html><body style="font-family:sans-serif;max-width:540px;margin:40px auto;">
<h2>{title}</h2><p>{message}</p></body></html>"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def _wait_for_callback(server: _CallbackServer, timeout_seconds: float = 180.0) -> Tuple[str, str]:
    start = time.monotonic()
    while time.monotonic() - start < timeout_seconds:
        error = server.callback_data["error"]
        if error:
            raise RuntimeError(f"OAuth error from callback: {error}")
        code = server.callback_data["code"]
        state = server.callback_data["state"]
        if code is not None and state is not None:
            return code, state
        time.sleep(0.1)
    raise TimeoutError("Timed out waiting for OAuth callback")


# ---------------------------------------------------------------------------
# 鉴权 provider 与请求封装
# ---------------------------------------------------------------------------


def _client_credentials_provider(
    base_url: str, *, dpop_enabled: bool = False
) -> UnifiedAuthProvider:
    scheme = SecurityScheme(
        type="oauth2",
        flows=OAuthFlows(
            client_credentials=OAuthFlow(token_url=f"{base_url}/oauth/token")
        ),
    )
    selected = SelectedRequirement(
        bindings=[
            SchemeBinding(
                scheme_name="oauthCC",
                security_scheme=scheme,
                scopes=[SCOPE_A2A_CALL],
                protocol=OAuth2ClientCredentialsProtocol(),
            )
        ]
    )
    local_config: Dict[str, object] = {
        "client_id": "demo-client",
        "client_secret": "demo-secret",
    }
    if dpop_enabled:
        local_config["dpop_enabled"] = True
        local_config["dpop_algorithm"] = "ES256"
    return UnifiedAuthProvider(
        selected=selected,
        local_config=local_config,
        agent_url=base_url,
    )


def _authorization_code_provider(
    base_url: str,
    redirect_uri: str,
    client_id: str,
    client_secret: str,
    callback_server: _CallbackServer,
) -> UnifiedAuthProvider:
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
        print(f"  打开浏览器登录: {auth_url}")
        opened = False
        try:
            opened = webbrowser.open(auth_url)
        except Exception:
            opened = False
        if opened:
            return
        print("  自动拉起默认浏览器失败，尝试使用 macOS open 命令...")
        try:
            open_result = subprocess.run(
                ["open", auth_url], check=False, capture_output=True, text=True
            )
            if open_result.returncode == 0:
                return
            print(
                "  open 命令失败，请手动复制上面的链接到浏览器打开。"
            )
        except (OSError, ValueError):
            print("  无法自动打开浏览器，请手动复制上面的链接到浏览器打开。")

    def callback_handler() -> Tuple[str, str]:
        return _wait_for_callback(callback_server)

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
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "redirect_handler": redirect_handler,
            "callback_handler": callback_handler,
        },
        agent_url=base_url,
    )


def _apikey_provider(base_url: str, api_key: str) -> UnifiedAuthProvider:
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


def _send_one(client: A2AClient) -> str:
    task = client._create_task("hello from e2e example")
    result = client._send_task(task)
    part = result.artifacts[0]["parts"][0]
    return str(part["text"])


# ---------------------------------------------------------------------------
# 主流程：先跑 4 个自动化场景，再跑 1 个交互式授权码场景
# ---------------------------------------------------------------------------

def _cleanup_callback_server(
    callback_server: _CallbackServer,
    callback_thread: threading.Thread,
) -> None:
    """关闭回调服务器并等待线程结束，单步失败不影响后续清理。"""
    try:
        callback_server.shutdown()
    except Exception:
        pass
    try:
        callback_server.server_close()
    except Exception:
        pass
    try:
        callback_thread.join(timeout=1.0)
    except Exception:
        pass


def main() -> int:
    redirect_uri = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
    callback_server: Optional[_CallbackServer] = None
    callback_thread: Optional[threading.Thread] = None
    auth_server: Optional[OAuthA2ATestServer] = None
    exit_code = 1

    try:
        try:
            callback_server = _CallbackServer(CALLBACK_HOST, CALLBACK_PORT)
        except OSError as error:
            print(
                f"回调服务器启动失败: {CALLBACK_HOST}:{CALLBACK_PORT} 被占用或不可用 ({error})"
            )
            return 1
        callback_thread = threading.Thread(
            target=callback_server.serve_forever,
            daemon=True,
        )
        callback_thread.start()
        with OAuthA2ATestServer(port=0, redirect_uri=redirect_uri) as server:
            auth_server = server
            base_url = server.base_url
            print(f"OAuth+A2A 服务器: {base_url}")
            print(f"回调地址: {redirect_uri}")
            print()

            scenarios: List[Tuple[str, Callable[[], str], str, bool]] = [
                (
                    "oauth_client_credentials",
                    lambda: _send_one(
                        A2AClient(base_url, auth_provider=_client_credentials_provider(base_url))
                    ),
                    "ok:bearer_jwt",
                    False,
                ),
                (
                    "oauth_client_credentials_dpop",
                    lambda: _send_one(
                        A2AClient(
                            base_url,
                            auth_provider=_client_credentials_provider(base_url, dpop_enabled=True),
                        ),
                    ),
                    "ok:dpop_jwt",
                    False,
                ),
                (
                    "apikey_header",
                    lambda: _send_one(
                        A2AClient(
                            base_url,
                            auth_provider=_apikey_provider(base_url, server.state.api_key),
                        )
                    ),
                    "ok:x_api_key",
                    False,
                ),
                (
                    "apikey_bearer_fallback",
                    lambda: _send_one(
                        A2AClient(
                            base_url,
                            headers={
                                "Authorization": f"Bearer {server.state.api_key}",
                                **JSON_HEADERS,
                            },
                        )
                    ),
                    "ok:bearer_api_key",
                    False,
                ),
                (
                    "oauth_authorization_code_pkce",
                    lambda: _send_one(
                        A2AClient(
                            base_url,
                            auth_provider=_authorization_code_provider(
                                base_url,
                                redirect_uri=redirect_uri,
                                client_id=server.state.client_id,
                                client_secret=server.state.client_secret,
                                callback_server=callback_server,
                            ),
                        )
                    ),
                    "ok:bearer_jwt",
                    True,
                ),
            ]

            passed = 0
            for name, run, expected, is_interactive in scenarios:
                if is_interactive:
                    print(f"[交互] {name}: 请在浏览器中登录 ({DEMO_USERNAME} / {DEMO_PASSWORD})，完成后再继续...")
                try:
                    actual = run()
                    ok = actual == expected
                    status = "PASS" if ok else "FAIL"
                    print(f"  [{status}] actual={actual!r} expected={expected!r}")
                    if ok:
                        passed += 1
                except Exception as exc:
                    print(f"  [FAIL] exception={exc}")
                print()

            total = len(scenarios)
            print(f"结果: {passed}/{total} 个场景通过")
            exit_code = 0 if passed == total else 1
    finally:
        if auth_server is not None:
            try:
                auth_server.stop()
            except Exception as error:
                print(f"认证服务器清理失败: {error}")
        if callback_server is not None and callback_thread is not None:
            try:
                _cleanup_callback_server(callback_server, callback_thread)
            except Exception as error:
                print(f"回调服务器清理失败: {error}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
