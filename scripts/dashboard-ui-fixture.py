"""Serve or exercise the real dashboard against a synthetic queue (no GPU jobs).

Run ``python scripts/dashboard-ui-fixture.py --serve`` for visual iteration, or
``python scripts/dashboard-ui-fixture.py --test --node /path/to/node`` for the
Playwright smoke suite. Set PLAYWRIGHT_MODULE and CHROME_EXECUTABLE if needed.
All generated fixtures and screenshots stay under .tools/dashboard-ui/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codex_3d_mcp.hunyuan_mv.dashboard import QueueDashboard  # noqa: E402

OUTPUT = ROOT / ".tools" / "dashboard-ui"
ARTIFACTS = OUTPUT / "artifacts"
VIEWS = ("front", "back", "left", "right")


def write_fixtures() -> None:
    """Make tiny known-good PNG and embedded GLB fixtures without model libraries."""
    (ARTIFACTS / "previews").mkdir(parents=True, exist_ok=True)
    for view, shade in zip(VIEWS, ("#8b909b", "#767c89", "#9299a6", "#69707d")):
        image = Image.new("RGB", (512, 384), "#252628")
        draw = ImageDraw.Draw(image)
        draw.ellipse((155, 287, 376, 326), fill="#1d1e20")
        draw.polygon([(185, 108), (274, 71), (353, 115), (353, 280),
                      (263, 319), (185, 277)], fill=shade)
        draw.polygon([(185, 108), (274, 71), (353, 115), (263, 155)], fill="#b8bcc4")
        draw.polygon([(263, 155), (353, 115), (353, 280), (263, 319)], fill="#626875")
        draw.rounded_rectangle((210, 165, 243, 220), radius=14, fill="#343840")
        image.save(ARTIFACTS / "previews" / f"{view}.png")

    positions = (-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1,
                 -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1)
    indices = (0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
               2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5)
    binary = struct.pack("<24f36H", *positions, *indices)
    document = {
        "asset": {"version": "2.0", "generator": "dashboard-ui-fixture"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0},
                                    "indices": 1, "material": 0}]}],
        "materials": [{"doubleSided": True, "pbrMetallicRoughness": {
            "baseColorFactor": [0.54, 0.57, 0.62, 1], "metallicFactor": 0,
            "roughnessFactor": 0.8}}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 96},
                        {"buffer": 0, "byteOffset": 96, "byteLength": 72}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 8,
                       "type": "VEC3", "min": [-1, -1, -1], "max": [1, 1, 1]},
                      {"bufferView": 1, "componentType": 5123, "count": 36,
                       "type": "SCALAR"}],
    }
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    glb = (struct.pack("<5I", 0x46546C67, 2, 28 + len(encoded) + len(binary),
                       len(encoded), 0x4E4F534A) + encoded
           + struct.pack("<2I", len(binary), 0x004E4942) + binary)
    (ARTIFACTS / "game.glb").write_bytes(glb)
    (ARTIFACTS / "review.json").write_text(
        json.dumps({"fixture_only": True, "review": "Inspect the rear pillar."}),
        encoding="utf-8",
    )


class FixtureManager:
    """Only dashboard protocol methods; no job creation or worker dependency."""

    def __init__(self) -> None:
        self.state = "awaiting_agent"
        self.actions: list[str] = []
        self.revision = 0
        self.lock = threading.Lock()

    def queue_snapshot(self) -> dict:
        with self.lock:
            self.revision += 1
            stamp = datetime(2026, 9, 12, 13, 42, tzinfo=timezone.utc)
            updated = (stamp + timedelta(seconds=self.revision)).isoformat()
            profiles = {
                key: {"state": "approved", "target_quads": target,
                      "actual_quads": actual, "actual_triangles": actual * 2,
                      "triangle_budget": target * 3}
                for key, target, actual in (("full_game", 20000, 19264),
                                           ("mobile", 6000, 5812), ("browser", 3000, 2876))
            }
            tower = {
                "asset_id": "tower", "name": "Sanctuary watchtower",
                "state": "awaiting_review", "stage": "paint", "attempt": 2,
                "previews": {v: f"/artifacts/ui-fixture/tower/previews/{v}.png"
                             for v in VIEWS},
                "profiles": profiles,
                "models": [{"id": key, "label": label, "textured": False,
                            "stage": "remesh", "approved": True,
                            "quad_count": profiles[key]["actual_quads"],
                            "target_quads": profiles[key]["target_quads"],
                            "triangle_count": 12,
                            "url": "/artifacts/ui-fixture/tower/game.glb"}
                           for key, label in (("full_game", "Full game"),
                                              ("mobile", "Mobile"), ("browser", "Browser"))],
                "artifacts": [{"name": "game.glb", "relative_path": "game.glb"},
                              {"name": "review.json", "relative_path": "review.json"}],
                "evidence": {"gate": {"passed": True},
                             "review": "Inspect the rear pillar before accepting this phase.",
                             "history": [{"attempt": 1, "state": "needs_repair"},
                                         {"attempt": 2, "state": "awaiting_review"}]},
            }
            return {
                "active_batch_id": "ui-fixture", "active_asset_id": None,
                "active_resource": "Idle · agent review",
                "batches": [
                    {"batch_id": "ui-fixture", "name": "Sanctuary collection",
                     "state": self.state, "stage": "paint", "created_at": stamp.isoformat(),
                     "updated_at": updated,
                     "counts": {"references": 3, "shape": 3, "remesh": 3,
                                "paint": 1, "finish": 0},
                     "assets": [tower,
                                {"asset_id": "archway", "name": "Eastern archway",
                                 "state": "needs_repair", "stage": "paint", "attempt": 1,
                                 "error": "A color seam on the rear pillar needs another paint attempt.",
                                 "previews": {}, "artifacts": []},
                                {"asset_id": "vessel", "name": "Ceremonial vessel",
                                 "state": "approved", "stage": "paint", "attempt": 1,
                                 "previews": {"front": "/artifacts/ui-fixture/vessel/previews/front.png"},
                                 "artifacts": []}]},
                    {"batch_id": "completed-fixture", "name": "Coastal props",
                     "state": "completed", "stage": "finish", "created_at": stamp.isoformat(),
                     "updated_at": stamp.isoformat(),
                     "counts": {key: 1 for key in ("references", "shape", "remesh", "paint", "finish")},
                     "assets": [{"asset_id": "buoy", "name": "Harbour buoy",
                                 "state": "completed", "stage": "finish", "attempt": 1,
                                 "previews": {}, "artifacts": []}]},
                ],
            }

    def resolve_artifact(self, batch_id: str, asset_id: str, relative_path: str) -> Path:
        if batch_id != "ui-fixture" or asset_id not in {"tower", "vessel", "archway"}:
            raise KeyError("Unknown fixture asset")
        target = (ARTIFACTS / relative_path).resolve()
        target.relative_to(ARTIFACTS.resolve())
        return target

    def _control(self, batch_id: str, action: str, state: str) -> None:
        if batch_id != "ui-fixture":
            raise KeyError("Unknown fixture batch")
        with self.lock:
            self.actions.append(action)
            self.state = state

    def pause_batch(self, batch_id: str) -> None:
        self._control(batch_id, "pause", "paused")

    def resume_batch(self, batch_id: str) -> None:
        self._control(batch_id, "resume", "awaiting_agent")

    def cancel_batch(self, batch_id: str) -> None:
        self._control(batch_id, "cancel", "cancelled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--test", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--node", default=shutil.which("node"))
    args = parser.parse_args()
    write_fixtures()
    manager = FixtureManager()
    dashboard = QueueDashboard(manager)
    try:
        url = dashboard.start(port=args.port)
        (OUTPUT / "url.txt").write_text(url, encoding="utf-8")
        print(json.dumps({"url": url, "fixture_only": True, "output": str(OUTPUT)}), flush=True)
        if args.test:
            if not args.node:
                parser.error("Pass --node with the Node.js executable path")
            result = subprocess.run(
                [args.node, str(ROOT / "tests" / "dashboard-ui-smoke.cjs"), url, str(OUTPUT)],
                cwd=ROOT, check=False, timeout=180,
            )
            if result.returncode:
                raise SystemExit(result.returncode)
            assert manager.actions == ["pause", "resume", "cancel"], manager.actions
            print(json.dumps({"manager_actions": manager.actions, "fixture_only": True}))
        else:
            threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()


if __name__ == "__main__":
    main()
