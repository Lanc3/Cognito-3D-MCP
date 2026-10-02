from __future__ import annotations

import http.client
import io
import json
import re
import socket
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from codex_3d_mcp.hunyuan_mv import dashboard as dashboard_module
from codex_3d_mcp.hunyuan_mv.dashboard import QueueDashboard, _DashboardHandler


class FakeManager:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.actions: list[tuple[str, str]] = []
        self.resolutions: list[tuple[str, str, str]] = []
        self.model_resolutions: list[tuple[str, str, str]] = []
        self.snapshot = {
            "active_batch_id": "batch-1",
            "active_asset_id": "asset-1",
            "active_resource": "GPU shape",
            "batches": [
                {
                    "batch_id": "batch-1",
                    "name": '</script><img src=x onerror="alert(1)"> & café',
                    "state": "awaiting_review",
                    "stage": "shape",
                    "assets": [
                        {
                            "asset_id": "asset-1",
                            "name": "Stone tower",
                            "state": "awaiting_review",
                            "stage": "shape",
                            "attempt": 2,
                            "error": {"message": "Check silhouette"},
                            "previews": {"front": "/artifacts/batch-1/asset-1/previews/front.png"},
                            "artifacts": [{"relative_path": "game.glb"}],
                        }
                    ],
                }
            ],
        }

    def queue_snapshot(self):
        return self.snapshot

    def resolve_artifact(self, batch_id: str, asset_id: str, relative_path: str) -> Path:
        self.resolutions.append((batch_id, asset_id, relative_path))
        if (batch_id, asset_id) != ("batch-1", "asset-1"):
            raise KeyError("Unknown asset")
        path = (self.directory / relative_path).resolve()
        path.relative_to(self.directory.resolve())
        return path

    def pause_batch(self, batch_id: str) -> None:
        self.actions.append(("pause", batch_id))

    def resolve_model(self, batch_id: str, asset_id: str, name: str) -> Path:
        self.model_resolutions.append((batch_id, asset_id, name))
        if (batch_id, asset_id, name) != ("batch-1", "asset-1", "original-shape.glb"):
            raise KeyError("Unknown model alias")
        return self.directory / "game.glb"

    def resume_batch(self, batch_id: str) -> None:
        self.actions.append(("resume", batch_id))

    def cancel_batch(self, batch_id: str) -> None:
        self.actions.append(("cancel", batch_id))


@pytest.fixture
def dashboard(tmp_path: Path):
    manager = FakeManager(tmp_path)
    (tmp_path / "previews").mkdir()
    (tmp_path / "previews" / "front.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    (tmp_path / "game.glb").write_bytes(b"glTFfixture")
    (tmp_path / "unsafe.html").write_text("<script>alert(1)</script>", encoding="utf-8")
    dashboard = QueueDashboard(manager)
    dashboard.start()
    try:
        yield dashboard, manager
    finally:
        dashboard.close()


def request(dashboard: QueueDashboard, path: str, *, method="GET", body=None, headers=None):
    address = urlsplit(dashboard.url)
    connection = http.client.HTTPConnection(address.hostname, address.port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def session_token(dashboard: QueueDashboard) -> str:
    status, headers, body = request(dashboard, "/")
    assert status == 200
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    match = re.search(rb'name="queue-session-token" content="([^"]+)"', body)
    assert match is not None
    return match[1].decode()


def control_headers(dashboard: QueueDashboard) -> dict[str, str]:
    return {
        "X-Queue-Token": session_token(dashboard),
        "Origin": dashboard.url,
        "Content-Type": "application/json",
    }


def raw_control_request(server, headers, body=b"{}", *, finish_body=False):
    """Preserve duplicate/framing headers that high-level clients normalize."""
    address = urlsplit(server.url)
    with socket.create_connection((address.hostname, address.port), timeout=5) as connection:
        values = [
            ("Host", address.netloc),
            ("Origin", server.url),
            ("X-Queue-Token", session_token(server)),
            *headers,
        ]
        request_head = "POST /api/batches/batch-1/pause HTTP/1.0\r\n" + "".join(
            f"{name}: {value}\r\n" for name, value in values
        )
        connection.sendall(request_head.encode("ascii") + b"\r\n" + body)
        if finish_body:
            connection.shutdown(socket.SHUT_WR)
        response = http.client.HTTPResponse(connection)
        response.begin()
        return response.status, response.read()


def test_lifecycle_binds_only_loopback_and_opens_browser_explicitly(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url, new: opened.append((url, new)))
    manager = FakeManager(tmp_path)
    dashboard = QueueDashboard(manager)
    try:
        url = dashboard.start()
        assert url.startswith("http://127.0.0.1:")
        assert int(urlsplit(url).port) > 0
        assert dashboard.start() == url
        assert opened == []
        assert dashboard.open() == url
        assert opened == [(url, 2)]
        old_token = session_token(dashboard)
        dashboard.close()
        assert dashboard.url == ""
        dashboard.close()
        dashboard.start()
        assert session_token(dashboard) != old_token
    finally:
        dashboard.close()


def test_dashboard_can_reuse_a_specific_loopback_port(dashboard):
    server, _ = dashboard
    port = urlsplit(server.url).port
    assert server.start(port=port) == server.url
    with pytest.raises(ValueError, match="Close the existing dashboard"):
        server.start(port=1 if port != 1 else 2)
    server.close()
    assert urlsplit(server.start(port=port)).port == port
    assert urlsplit(server.url).hostname == "127.0.0.1"
    for value in (-1, 65536, "61981", True, 2.5):
        with pytest.raises(ValueError, match="integer from 0 to 65535"):
            server.start(port=value)


def test_snapshot_is_authenticated_script_safe_json(dashboard):
    server, manager = dashboard
    assert request(server, "/api/queue")[0] == 403
    token = session_token(server)
    status, headers, body = request(server, "/api/queue", headers={"X-Queue-Token": token})
    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert headers["Cache-Control"] == "no-store"
    assert "Access-Control-Allow-Origin" not in headers
    assert b"</script>" not in body
    assert b"\\u003c/script\\u003e" in body
    assert json.loads(body) == manager.snapshot
    assert json.loads(body)["active_resource"] == "GPU shape"


@pytest.mark.parametrize("action", ["pause", "resume", "cancel"])
def test_commands_require_json_origin_and_session_capability(dashboard, action):
    server, manager = dashboard
    headers = control_headers(server)
    status, _, body = request(
        server, f"/api/batches/batch-1/{action}", method="POST", body="{}", headers=headers
    )
    assert status == 200
    assert json.loads(body) == {"ok": True, "batch_id": "batch-1", "action": action}
    assert manager.actions == [(action, "batch-1")]


@pytest.mark.parametrize(
    "change",
    [
        {"Origin": None},
        {"Origin": "https://attacker.example"},
        {"Origin": "null"},
        {"X-Queue-Token": None},
        {"X-Queue-Token": "wrong"},
        {"X-Queue-Token": "é"},
        {"Host": "attacker.example"},
    ],
)
def test_cross_origin_or_unauthorized_controls_do_not_mutate(dashboard, change):
    server, manager = dashboard
    headers = control_headers(server)
    for key, value in change.items():
        if value is None:
            headers.pop(key, None)
        else:
            headers[key] = value
    status, _, _ = request(
        server, "/api/batches/batch-1/cancel", method="POST", body="{}", headers=headers
    )
    assert status == 403
    assert manager.actions == []


def test_unauthorized_control_body_is_consumed_before_sending_the_error():
    handler = object.__new__(_DashboardHandler)
    handler.command = "POST"
    handler.headers = Message()
    handler.headers["Content-Length"] = "2"
    handler.headers["Origin"] = "http://127.0.0.1:1234"
    handler.owner = SimpleNamespace(url="http://127.0.0.1:1234", _token="required-token")
    handler.rfile = io.BytesIO(b"{}")
    timeouts = []
    handler.connection = SimpleNamespace(gettimeout=lambda: 10, settimeout=timeouts.append)
    sent = []
    handler._json = lambda status, value: sent.append((status, value, handler.rfile.tell()))
    assert handler._authorized(require_origin=True) is False
    assert sent == [(403, {"error": "Dashboard authorization required"}, 2)]
    assert len(timeouts) == 2 and 0 < timeouts[0] <= 1.0 and timeouts[1] == 10


def test_rejected_drip_fed_body_cannot_extend_absolute_discard_deadline(monkeypatch):
    handler = object.__new__(_DashboardHandler)
    handler.command = "POST"
    handler.headers = Message()
    handler.headers["Content-Length"] = "10"
    clock = {"now": 0.0, "timeout": 10.0, "delivered": 0}
    timeouts = []

    def settimeout(value):
        clock["timeout"] = value
        timeouts.append(value)

    def read1(length):
        assert 0 < length <= 8192
        # A sender produces one byte every 0.4 s. Model socket timeout behavior
        # without sleeping or opening a connection in this deterministic check.
        clock["now"] += min(0.4, clock["timeout"])
        if clock["timeout"] < 0.4:
            raise TimeoutError("Next byte arrives after the remaining deadline")
        clock["delivered"] += 1
        return b"x"

    monkeypatch.setattr(dashboard_module, "monotonic", lambda: clock["now"])
    handler.connection = SimpleNamespace(gettimeout=lambda: 10.0, settimeout=settimeout)
    handler.rfile = SimpleNamespace(read1=read1)
    sent = []
    handler._json = lambda status, value: sent.append((status, clock["now"]))
    handler._error(403, "Dashboard authorization required")
    assert clock["delivered"] == 2
    assert clock["now"] == pytest.approx(1.0)
    assert timeouts == pytest.approx([1.0, 0.6, 0.2, 10.0])
    assert sent == [(403, pytest.approx(1.0))]
    assert handler.close_connection is True


@pytest.mark.parametrize("lengths,transfer", [(["2", "2"], None), (["2"], "chunked"),
                                              (["65537"], None), (["invalid"], None)])
def test_rejected_body_discard_never_reads_ambiguous_or_unbounded_framing(lengths, transfer):
    handler = object.__new__(_DashboardHandler)
    handler.command = "POST"
    handler.headers = Message()
    for length in lengths:
        handler.headers["Content-Length"] = length
    if transfer:
        handler.headers["Transfer-Encoding"] = transfer
    handler.rfile = SimpleNamespace(read1=lambda length: pytest.fail("Unsafe rejected-body drain"))
    sent = []
    handler._json = lambda status, value: sent.append(status)
    handler._error(400, "Rejected framing")
    assert sent == [400]


def test_json_error_does_not_attempt_to_read_an_already_consumed_body():
    handler = object.__new__(_DashboardHandler)
    handler.command = "POST"
    handler._control_body_consumed = True
    handler.rfile = SimpleNamespace(read1=lambda length: pytest.fail("Body was already consumed"))
    sent = []
    handler._json = lambda status, value: sent.append(status)
    handler._error(400, "Send an empty JSON object")
    assert sent == [400]


def test_rebinding_and_cross_origin_reads_are_rejected(dashboard):
    server, _ = dashboard
    for path in ("/", "/api/queue", "/artifacts/batch-1/asset-1/game.glb",
                 "/models/batch-1/asset-1/original-shape.glb", "/viewer.js"):
        assert request(server, path, headers={"Host": "attacker.example"})[0] == 403
        assert request(server, path, headers={"Origin": "https://attacker.example"})[0] == 403


@pytest.mark.parametrize("site", ["cross-site", "same-site", "unexpected"])
def test_fetch_metadata_rejects_browser_reads_without_origin(dashboard, site):
    server, manager = dashboard
    for path in ("/", "/artifacts/batch-1/asset-1/game.glb", "/viewer.js"):
        assert request(server, path, headers={"Sec-Fetch-Site": site})[0] == 403
    assert manager.resolutions == []


@pytest.mark.parametrize("site", ["same-origin", "none"])
def test_fetch_metadata_allows_local_browser_requests(dashboard, site):
    server, _ = dashboard
    assert request(server, "/", headers={"Sec-Fetch-Site": site})[0] == 200


@pytest.mark.parametrize(
    ("body", "content_type", "expected"),
    [
        ("{}", "text/plain", 415),
        ("[]", "application/json", 400),
        ('{"path":"C:/private"}', "application/json", 400),
        ("not json", "application/json", 400),
        ("x" * 5000, "application/json", 413),
    ],
)
def test_bad_command_bodies_do_not_mutate(dashboard, body, content_type, expected):
    server, manager = dashboard
    headers = control_headers(server)
    headers["Content-Type"] = content_type
    assert (
        request(server, "/api/batches/batch-1/pause", method="POST", body=body, headers=headers)[0]
        == expected
    )
    assert manager.actions == []


@pytest.mark.parametrize(
    "extra_headers",
    [
        [("Content-Length", "+2")],
        [("Content-Length", "-2")],
        [("Content-Length", "2, 2")],
        [("Content-Length", "2"), ("Content-Length", "2")],
        [("Content-Length", "2"), ("Transfer-Encoding", "")],
        [("Content-Length", "2"), ("Transfer-Encoding", ""),
         ("Transfer-Encoding", "chunked")],
    ],
)
def test_ambiguous_command_framing_does_not_mutate(dashboard, extra_headers):
    server, manager = dashboard
    status, body = raw_control_request(
        server, [("Content-Type", "application/json"), *extra_headers]
    )
    assert status == 400
    assert b"Traceback" not in body
    assert manager.actions == []


def test_duplicate_command_content_types_do_not_mutate(dashboard):
    server, manager = dashboard
    status, _ = raw_control_request(server, [
        ("Content-Type", "application/json"), ("Content-Type", "text/plain"),
        ("Content-Length", "2"),
    ])
    assert status == 415
    assert manager.actions == []


@pytest.mark.parametrize("body", [b"[" * 1500 + b"]" * 1500, b"\xff\xff"])
def test_malformed_or_deep_command_json_fails_cleanly(dashboard, body):
    server, manager = dashboard
    status, payload = raw_control_request(server, [
        ("Content-Type", "application/json"), ("Content-Length", str(len(body))),
    ], body)
    assert status == 400
    assert json.loads(payload) == {"error": "Send an empty JSON object"}
    assert manager.actions == []


def test_incomplete_command_body_does_not_mutate(dashboard):
    server, manager = dashboard
    status, _ = raw_control_request(server, [
        ("Content-Type", "application/json"), ("Content-Length", "3"),
    ], finish_body=True)
    assert status == 400
    assert manager.actions == []


def test_stalled_command_body_times_out_cleanly(dashboard, monkeypatch):
    from codex_3d_mcp.hunyuan_mv import dashboard as module

    server, manager = dashboard
    setup = module._DashboardHandler.setup

    def fast_timeout(handler):
        setup(handler)
        handler.connection.settimeout(0.1)

    monkeypatch.setattr(module._DashboardHandler, "setup", fast_timeout)
    status, body = raw_control_request(server, [
        ("Content-Type", "application/json"), ("Content-Length", "3"),
    ])
    assert status == 408
    assert json.loads(body) == {"error": "Command body timed out"}
    assert manager.actions == []


@pytest.mark.parametrize(
    "path",
    [
        "/artifacts/batch-1/asset-1/../private.txt",
        "/artifacts/batch-1/asset-1/%2e%2e/private.txt",
        "/artifacts/batch-1/asset-1/..%20/private.txt",
        "/artifacts/batch-1/asset-1/%252e%252e/private.txt",
        "/artifacts/batch-1/asset-1/%2fprivate.txt",
        "/artifacts/batch-1/asset-1/..%5cprivate.txt",
        "/artifacts/batch-1/asset-1/C%3a/private.txt",
        "/artifacts/batch-1/asset-1/file%00.png",
        "/artifacts/batch-1/asset-1/previews//front.png",
        "/artifacts/batch-1/asset-1/file%xy.png",
        "/artifacts/batch%2fother/asset-1/game.glb",
    ],
)
def test_traversal_is_rejected_before_manager_resolution(dashboard, path):
    server, manager = dashboard
    assert request(server, path)[0] == 404
    assert manager.resolutions == []


def test_scoped_artifacts_stream_and_active_content_downloads(dashboard):
    server, manager = dashboard
    status, headers, body = request(server, "/artifacts/batch-1/asset-1/previews/front.png")
    assert status == 200 and body.startswith(b"\x89PNG")
    assert headers["Content-Type"] == "image/png"
    assert headers["Cross-Origin-Resource-Policy"] == "same-origin"
    status, headers, body = request(server, "/artifacts/batch-1/asset-1/game.glb")
    assert status == 200 and body == b"glTFfixture"
    assert headers["Content-Disposition"] == "attachment"
    status, headers, body = request(server, "/artifacts/batch-1/asset-1/unsafe.html")
    assert status == 200
    assert headers["Content-Type"] == "application/octet-stream"
    assert headers["Content-Disposition"] == "attachment"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert request(server, "/artifacts/batch-1/asset-1/game.glb", method="HEAD")[2] == b""
    assert manager.resolutions[0] == ("batch-1", "asset-1", "previews/front.png")


def test_unknown_routes_and_assets_do_not_expose_filesystem(dashboard):
    server, _ = dashboard
    for path in (
        "/dashboard.py",
        "/web/../dashboard.py",
        "/.env",
        "/artifacts/batch-1/asset-2/game.glb",
    ):
        status, _, body = request(server, path)
        assert status == 404
        assert b"Traceback" not in body


def test_invalid_snapshots_fail_cleanly(dashboard):
    server, manager = dashboard
    manager.snapshot["bad_value"] = float("nan")
    status, _, body = request(
        server, "/api/queue", headers={"X-Queue-Token": session_token(server)}
    )
    assert status == 500
    assert json.loads(body) == {"error": "Queue snapshot is unavailable"}


def test_static_app_has_no_remote_dependencies_or_unsafe_html_sink(dashboard):
    server, _ = dashboard
    status, _, html = request(server, "/")
    assert status == 200
    assert b"Cognito-3D-mcp" in html
    assert b"__QUEUE_SESSION_TOKEN__" not in html
    assert b"http://" not in html and b"https://" not in html
    status, _, script = request(server, "/app.js")
    assert status == 200
    assert b"innerHTML" not in script
    assert b"document.write" not in script
    assert b"textContent" in script
    assert request(server, "/style.css")[0] == 200


def test_missing_packaged_assets_fail_cleanly(dashboard, tmp_path, monkeypatch):
    from codex_3d_mcp.hunyuan_mv import dashboard as module

    server, _ = dashboard
    monkeypatch.setattr(module, "_WEB", tmp_path / "missing-web-assets")
    assert request(server, "/")[0] == 500
    assert request(server, "/app.js")[0] == 404


def test_all_packaged_viewer_dependencies_are_served(dashboard):
    from codex_3d_mcp.hunyuan_mv import dashboard as module

    server, _ = dashboard
    for route in module._STATIC:
        status, _, content = request(server, route)
        assert status == 200, route
        assert content, route


def test_original_shape_alias_allows_only_fixed_scoped_read(dashboard):
    server, manager = dashboard
    path = "/models/batch-1/asset-1/original-shape.glb"
    status, headers, body = request(server, path)
    assert status == 200 and body == b"glTFfixture"
    assert headers["Content-Disposition"] == "attachment"
    assert request(server, path, method="HEAD")[2] == b""
    assert manager.model_resolutions == [("batch-1", "asset-1", "original-shape.glb")] * 2
    for invalid in (
        "/models/batch-1/asset-1/../original-shape.glb",
        "/models/batch-1/asset-1/%2e%2e/original-shape.glb",
        "/models/batch-1/asset-1/other.glb",
        "/models/batch-1/asset-1/sub/original-shape.glb",
        "/models/batch%2fother/asset-1/original-shape.glb",
    ):
        assert request(server, invalid)[0] == 404
    assert len(manager.model_resolutions) == 2
    assert request(server, "/models/batch-1/asset-2/original-shape.glb")[0] == 404


def test_viewer_static_allowlist_and_minimal_blob_image_csp(dashboard, tmp_path, monkeypatch):
    from codex_3d_mcp.hunyuan_mv import dashboard as module

    server, _ = dashboard
    static_root = tmp_path / "viewer-static"
    (static_root / "vendor").mkdir(parents=True)
    monkeypatch.setattr(module, "_WEB", static_root)
    routes = ["/viewer.js", "/viewer.css", *(
        f"/vendor/{name}" for name in (
            "three.module.js", "three.core.js", "OrbitControls.js", "GLTFLoader.js",
            "BufferGeometryUtils.js", "RoomEnvironment.js", "LICENSE.txt",
        )
    )]
    for route in routes:
        (static_root / route.lstrip("/")).write_text("fixture", encoding="utf-8")
        status, headers, body = request(server, route)
        assert status == 200 and body == b"fixture"
        expected_type = (
            "text/plain" if route.endswith(".txt") else
            "text/css" if route.endswith(".css") else "text/javascript"
        )
        assert headers["Content-Type"] == expected_type + "; charset=utf-8"
        policy = headers["Content-Security-Policy"]
        assert "img-src 'self' blob:" in policy
        assert "script-src 'self';" in policy and "connect-src 'self' blob:;" in policy
        assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    for route in ("/vendor/arbitrary.js", "/vendor/../dashboard.py", "/vendor/%2e%2e/app.js"):
        assert request(server, route)[0] == 404
