from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageDraw

from codex_3d_mcp.trellis.config import TrellisSettings
from codex_3d_mcp.trellis.pipeline import BidirectionalJobManager
from codex_3d_mcp.trellis.store import TrellisJobStore


class FakeLease:
    def acquire(self, **_kwargs) -> None:
        return None

    def release(self) -> None:
        return None


class FakeRuntime:
    def __init__(self) -> None:
        self.gpu_lease = FakeLease()

    @contextmanager
    def session(self, _log_path, _cancel_event):
        yield self

    def generate(self, _image: Path, output: Path, _seed: int) -> None:
        output.write_bytes(b"glTF" + b"\0" * 32)

    def stop(self) -> None:
        return None

    def start(self, _log_path: Path) -> None:
        return None

    def close(self) -> None:
        return None


class FakeSemantic:
    def similarity(self, _front: Path, _back: Path) -> float:
        return 0.92


def _settings(tmp_path: Path) -> TrellisSettings:
    runtime = tmp_path / "runtime"
    models = tmp_path / "models"
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"fake")
    return TrellisSettings(
        base_dir=tmp_path,
        server_name="test",
        server_version="test",
        output_dir=tmp_path / "outputs",
        database_path=tmp_path / "outputs" / "jobs.sqlite3",
        runtime_dir=runtime,
        model_dir=models,
        trellis_server=runtime / "server.exe",
        trellis_cli=runtime / "cli.exe",
        realesrgan_exe=runtime / "realesrgan.exe",
        blender_exe=blender,
        gltf_validator=runtime / "validator.exe",
        allowed_input_roots=(tmp_path,),
        gpu_lock_path=runtime / "gpu.lock",
        texture_resolution=64,
        master_faces=100,
        game_faces=60,
        lod_faces=(40, 20),
        minimum_image_size=128,
        minimum_free_bytes=1,
        test_mode=True,
    )


def _image(path: Path, color: str) -> Path:
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 16, 96, 112), fill=color)
    image.save(path, "PNG")
    return path


def _fake_blender(job_dir: Path, config: dict, label: str) -> None:
    result_path = Path(config["result_path"])
    texture_dir = job_dir / "textures"
    texture_dir.mkdir(exist_ok=True)
    geometry = {
        "master": {"faces": 100, "degenerate_faces": 0, "watertight": True},
        "game": {"faces": 60, "degenerate_faces": 0, "watertight": True},
        "lod1": {"faces": 40, "degenerate_faces": 0, "watertight": True},
        "lod2": {"faces": 20, "degenerate_faces": 0, "watertight": True},
    }
    if label == "analyze":
        value = {
            "passed": True,
            "profiles": {
                "front": {"passed": True, "components": 1},
                "back": {"passed": True, "components": 1},
            },
        }
    elif label == "build":
        for name in geometry:
            (job_dir / f"{name}.glb").write_bytes(b"glTF" + b"\0" * 32)
        Image.new("RGBA", (64, 64), (200, 50, 40, 255)).save(texture_dir / "baseColor.png")
        Image.new("L", (64, 64), 220).save(texture_dir / "ao.png")
        Image.new("RGB", (64, 64), (128, 128, 255)).save(texture_dir / "normal.png")
        (job_dir / "working.blend").write_bytes(b"blend")
        value = {
            "geometry": geometry,
            "alignment": {"side_agreement": 0.95},
            "uv": {"occupancy": 0.72},
        }
    else:
        (job_dir / "previews").mkdir(exist_ok=True)
        value = {"geometry": geometry, "previews": str(job_dir / "previews")}
    result_path.write_text(json.dumps(value), encoding="utf-8")


def test_complete_job_is_durable_and_produces_all_deliverables(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manager = BidirectionalJobManager(
        settings,
        store=TrellisJobStore(settings.database_path),
        runtime=FakeRuntime(),
        semantic=FakeSemantic(),
    )
    manager._run_blender = _fake_blender  # type: ignore[method-assign]
    front = _image(tmp_path / "front.png", "red")
    back = _image(tmp_path / "back.png", "blue")
    job = manager.submit("test asset", str(front), str(back), "painted metal", 42)
    deadline = time.time() + 10
    while time.time() < deadline:
        result = manager.get(job["job_id"])
        if result["state"] in {"completed", "failed", "failed_quality"}:
            break
        time.sleep(0.05)
    assert result["state"] == "completed", result
    assert {Path(item["path"]).name for item in result["artifacts"]} >= {
        "master.glb",
        "game.glb",
        "lod1.glb",
        "lod2.glb",
        "manifest.json",
    }
    manager.close()


def test_fragmented_raw_sources_fail_before_fusion(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manager = BidirectionalJobManager(
        settings,
        store=TrellisJobStore(settings.database_path),
        runtime=FakeRuntime(),
        semantic=FakeSemantic(),
    )
    calls = []

    def fragmented(_job_dir: Path, config: dict, label: str) -> None:
        calls.append(label)
        assert label == "analyze"
        Path(config["result_path"]).write_text(
            json.dumps(
                {
                    "passed": False,
                    "profiles": {
                        "front": {"passed": False, "components": 2000},
                        "back": {"passed": False, "components": 2200},
                    },
                }
            ),
            encoding="utf-8",
        )

    manager._run_blender = fragmented  # type: ignore[method-assign]
    front = _image(tmp_path / "front.png", "red")
    back = _image(tmp_path / "back.png", "blue")
    job = manager.submit("fragmented asset", str(front), str(back), "stone", 42)
    deadline = time.time() + 10
    while time.time() < deadline:
        result = manager.get(job["job_id"])
        if result["state"] in {"failed", "failed_quality"}:
            break
        time.sleep(0.05)

    assert result["state"] == "failed_quality"
    assert result["error"]["code"] == "SOURCE_FRAGMENTATION"
    assert calls == ["analyze"]
    assert not (manager.job_dir(job["job_id"]) / "master.glb").is_file()
    manager.close()


def test_low_uv_occupancy_is_not_published(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manager = BidirectionalJobManager(
        settings,
        store=TrellisJobStore(settings.database_path),
        runtime=FakeRuntime(),
        semantic=FakeSemantic(),
    )

    def poor_uv(job_dir: Path, config: dict, label: str) -> None:
        _fake_blender(job_dir, config, label)
        if label == "build":
            result_path = Path(config["result_path"])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["uv"] = {"occupancy": 0.20}
            result_path.write_text(json.dumps(result), encoding="utf-8")

    manager._run_blender = poor_uv  # type: ignore[method-assign]
    front = _image(tmp_path / "front.png", "red")
    back = _image(tmp_path / "back.png", "blue")
    job = manager.submit("low occupancy", str(front), str(back), "painted stone", 42)
    deadline = time.time() + 10
    while time.time() < deadline:
        result = manager.get(job["job_id"])
        if result["state"] in {"failed", "failed_quality"}:
            break
        time.sleep(0.05)

    assert result["state"] == "failed_quality"
    report = json.loads(
        (manager.job_dir(job["job_id"]) / "qa" / "report.json").read_text(encoding="utf-8")
    )
    occupancy = next(check for check in report["checks"] if check["name"] == "uv_occupancy")
    assert occupancy["passed"] is False
    manager.close()
