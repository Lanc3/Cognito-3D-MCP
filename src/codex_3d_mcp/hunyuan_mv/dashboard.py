"""Small, local-only queue dashboard; importing it never loads a model runtime."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import unquote, urlsplit

_WEB = Path(__file__).with_name("web")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_MAX_CONTROL_BODY = 4096
_STATIC = {
    "/app.js": ("app.js", "text/javascript"),
    "/viewer.js": ("viewer.js", "text/javascript"),
    "/viewer.css": ("viewer.css", "text/css"),
    "/style.css": ("style.css", "text/css"),
    **{
        f"/vendor/{name}": (f"vendor/{name}", "text/javascript")
        for name in (
            "three.module.js", "three.core.js", "OrbitControls.js", "GLTFLoader.js",
            "BufferGeometryUtils.js", "RoomEnvironment.js",
        )
    },
    "/vendor/LICENSE.txt": ("vendor/LICENSE.txt", "text/plain"),
}
_IMAGES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
    "connect-src 'self' blob:; font-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
    "form-action 'none'"
)


def _json_bytes(value: Any) -> bytes:
    # Also safe if a client later embeds this JSON in an HTML document.
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for literal, escaped in (
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("&", "\\u0026"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        encoded = encoded.replace(literal, escaped)
    return encoded.encode("utf-8")


def _segments(path: str) -> list[str]:
    if re.search(r"%(?![0-9a-fA-F]{2})", path):
        raise ValueError("Malformed path")
    result = [unquote(part, errors="strict") for part in path.split("/")[1:]]
    for part in result:
        if (
            not part
            or part.endswith((".", " "))
            or any(char in part for char in "\\/:\x00%")
        ):
            raise ValueError("Invalid path")
        if any(ord(char) < 32 or ord(char) == 127 for char in part):
            raise ValueError("Invalid path")
    return result


class QueueDashboard:
    """Serve a manager's queue snapshot and explicitly scoped artifact/control APIs.

    The manager owns durable state and artifact-root authorization. It implements
    ``queue_snapshot``, ``resolve_artifact``, and pause/resume/cancel_batch. The
    dashboard intentionally has no start-job API and no filesystem browser.
    """

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._token = ""
        self._url = ""

    @property
    def url(self) -> str:
        return self._url

    def start(self, open_browser: bool = False, port: int = 0) -> str:
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("Dashboard port must be an integer from 0 to 65535")
        with self._lock:
            if self._server is not None and port not in (0, self._server.server_port):
                raise ValueError("Close the existing dashboard before changing its port")
            if self._server is None:
                self._token = secrets.token_urlsafe(32)
                dashboard = self

                class Handler(_DashboardHandler):
                    owner = dashboard

                server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
                server.daemon_threads = True
                self._server = server
                self._url = f"http://127.0.0.1:{server.server_port}"
                self._thread = threading.Thread(
                    target=server.serve_forever,
                    kwargs={"poll_interval": 0.2},
                    name="cognito-3d-queue-dashboard",
                    daemon=True,
                )
                self._thread.start()
            url = self._url
        if open_browser:
            webbrowser.open(url, new=2)
        return url

    def open(self) -> str:
        return self.start(open_browser=True)

    def close(self) -> None:
        with self._lock:
            server, thread = self._server, self._thread
            if server is None:
                return
            # Hold the lifecycle lock so a concurrent start cannot reuse stale state.
            server.shutdown()
            server.server_close()
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2)
            self._server = None
            self._thread = None
            self._url = ""
            self._token = ""


class _DashboardHandler(BaseHTTPRequestHandler):
    owner: QueueDashboard
    server_version = "Cognito-3D-mcp"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format: str, *args: Any) -> None:
        # MCP reserves stdout for protocol traffic; poll requests need no log spam.
        return

    def _headers(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self._headers(status, content_type, len(body))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, value: Any) -> None:
        self._send(status, _json_bytes(value), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._discard_rejected_control_body()
        self._json(status, {"error": message})

    def _discard_rejected_control_body(self) -> None:
        # Closing HTTP/1.0 with unread POST bytes can reset the connection on
        # Windows before the client receives its error response. Consume only
        # unambiguously framed, small bodies; this never parses or authorizes
        # a command and does not raise the accepted control-body limit.
        if self.command != "POST" or getattr(self, "_control_body_consumed", False):
            return
        self._control_body_consumed = True
        lengths = self.headers.get_all("Content-Length") or []
        if (
            self.headers.get_all("Transfer-Encoding") is not None
            or len(lengths) != 1
            or not re.fullmatch(r"[0-9]{1,10}", lengths[0])
        ):
            return
        length = int(lengths[0])
        if not 0 < length <= 64 * 1024:
            return
        timeout = self.connection.gettimeout()
        deadline = monotonic() + 1.0
        remaining = length
        try:
            while remaining:
                budget = deadline - monotonic()
                if budget <= 0:
                    self.close_connection = True
                    break
                self.connection.settimeout(min(timeout, budget) if timeout is not None else budget)
                # A drip-fed body cannot keep restarting the discard timeout.
                chunk = self.rfile.read1(min(remaining, 8192))
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            self.close_connection = True
        finally:
            self.connection.settimeout(timeout)

    def _request_path(self) -> str | None:
        expected_host = urlsplit(self.owner.url).netloc
        if self.headers.get_all("Host") != [expected_host]:
            self._error(HTTPStatus.FORBIDDEN, "Unexpected host")
            return None
        origins = self.headers.get_all("Origin")
        if origins is not None and origins != [self.owner.url]:
            self._error(HTTPStatus.FORBIDDEN, "Unexpected origin")
            return None
        # Browser image/navigation requests can omit Origin. Fetch metadata also
        # prevents another localhost application from embedding private artifacts.
        sites = self.headers.get_all("Sec-Fetch-Site")
        if sites is not None and sites not in (["same-origin"], ["none"]):
            self._error(HTTPStatus.FORBIDDEN, "Cross-origin browser request")
            return None
        try:
            parsed = urlsplit(self.path)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "Invalid request path")
            return None
        if parsed.scheme or parsed.netloc or parsed.fragment:
            self._error(HTTPStatus.BAD_REQUEST, "Invalid request path")
            return None
        return parsed.path

    def _authorized(self, require_origin: bool = False) -> bool:
        tokens = self.headers.get_all("X-Queue-Token")
        valid_token = (
            tokens is not None
            and len(tokens) == 1
            and secrets.compare_digest(tokens[0].encode("utf-8"), self.owner._token.encode("ascii"))
        )
        valid_origin = not require_origin or self.headers.get_all("Origin") == [self.owner.url]
        if not valid_token or not valid_origin:
            self._error(HTTPStatus.FORBIDDEN, "Dashboard authorization required")
            return False
        return True

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        path = self._request_path()
        if path is None:
            return
        if path == "/":
            try:
                html = (_WEB / "index.html").read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard assets are unavailable")
                return
            self._send(
                HTTPStatus.OK,
                html.replace("__QUEUE_SESSION_TOKEN__", self.owner._token).encode("utf-8"),
                "text/html; charset=utf-8",
            )
        elif path in _STATIC:
            filename, content_type = _STATIC[path]
            try:
                content = (_WEB / filename).read_bytes()
            except OSError:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
                return
            self._send(
                HTTPStatus.OK, content, content_type + "; charset=utf-8"
            )
        elif path == "/api/queue":
            if self._authorized():
                try:
                    payload = _json_bytes(self.owner.manager.queue_snapshot())
                except (TypeError, ValueError, RuntimeError):
                    self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Queue snapshot is unavailable")
                    return
                self._send(HTTPStatus.OK, payload, "application/json; charset=utf-8")
        elif path.startswith("/artifacts/"):
            self._artifact(path)
        elif path.startswith("/models/"):
            self._model(path)
        else:
            self._error(HTTPStatus.NOT_FOUND, "Not found")

    def _artifact(self, path: str) -> None:
        try:
            parts = _segments(path)
            if len(parts) < 4 or not all(_IDENTIFIER.fullmatch(part) for part in parts[1:3]):
                raise ValueError("Invalid artifact path")
            relative_path = "/".join(parts[3:])
            # The manager resolves only within this asset's owned roots, including symlinks.
            file = Path(self.owner.manager.resolve_artifact(parts[1], parts[2], relative_path))
        except (OSError, ValueError, KeyError, RuntimeError):
            self._error(HTTPStatus.NOT_FOUND, "Artifact not found")
            return
        self._stream_file(file)

    def _model(self, path: str) -> None:
        try:
            parts = _segments(path)
            if (
                len(parts) != 4 or parts[3] != "original-shape.glb"
                or not all(_IDENTIFIER.fullmatch(part) for part in parts[1:3])
            ):
                raise ValueError("Invalid model alias")
            file = Path(self.owner.manager.resolve_model(parts[1], parts[2], parts[3]))
        except (OSError, ValueError, KeyError, RuntimeError):
            self._error(HTTPStatus.NOT_FOUND, "Model not found")
            return
        self._stream_file(file)

    def _stream_file(self, file: Path) -> None:
        try:
            handle = file.open("rb")
        except OSError:
            self._error(HTTPStatus.NOT_FOUND, "Artifact not found")
            return
        with handle:
            handle.seek(0, 2)
            length = handle.tell()
            handle.seek(0)
            content_type = _IMAGES.get(file.suffix.lower(), "application/octet-stream")
            self._headers(HTTPStatus.OK, content_type, length)
            if file.suffix.lower() not in _IMAGES:
                # Never execute HTML/SVG/JavaScript even if it is a legitimate artifact.
                self.send_header("Content-Disposition", "attachment")
            self.end_headers()
            if self.command != "HEAD":
                shutil.copyfileobj(handle, self.wfile, 256 * 1024)

    def do_POST(self) -> None:
        self._control_body_consumed = False
        path = self._request_path()
        if path is None or not self._authorized(require_origin=True):
            return
        try:
            parts = _segments(path)
            if len(parts) != 4 or parts[:2] != ["api", "batches"]:
                raise ValueError("Unknown command")
            batch_id, action = parts[2:]
            if not _IDENTIFIER.fullmatch(batch_id) or action not in {"pause", "resume", "cancel"}:
                raise ValueError("Unknown command")
        except ValueError:
            self._error(HTTPStatus.NOT_FOUND, "Unknown command")
            return
        content_types = self.headers.get_all("Content-Type") or []
        if (
            len(content_types) != 1
            or content_types[0].split(";", 1)[0].strip().lower() != "application/json"
        ):
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Send application/json")
            return
        lengths = self.headers.get_all("Content-Length") or []
        if self.headers.get_all("Transfer-Encoding") is not None or len(lengths) != 1:
            self._error(HTTPStatus.BAD_REQUEST, "A single Content-Length is required")
            return
        if not re.fullmatch(r"[0-9]{1,10}", lengths[0]):
            self._error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
            return
        length = int(lengths[0])
        if length > _MAX_CONTROL_BODY:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Command body is too large")
            return
        try:
            self._control_body_consumed = True
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ValueError("Incomplete command body")
            body = json.loads(payload)
            if body != {}:
                raise ValueError("Expected an empty object")
        except (ValueError, UnicodeDecodeError, RecursionError):
            self._error(HTTPStatus.BAD_REQUEST, "Send an empty JSON object")
            return
        except TimeoutError:
            self._error(HTTPStatus.REQUEST_TIMEOUT, "Command body timed out")
            return
        try:
            getattr(self.owner.manager, f"{action}_batch")(batch_id)
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "Batch not found")
            return
        except (ValueError, RuntimeError):
            self._error(
                HTTPStatus.CONFLICT, "Batch cannot perform this action in its current state"
            )
            return
        self._json(HTTPStatus.OK, {"ok": True, "batch_id": batch_id, "action": action})
