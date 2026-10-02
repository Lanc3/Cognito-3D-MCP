from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from codex_3d_mcp.errors import InvalidImageError
from codex_3d_mcp.hunyuan_mv.config import HunyuanMVSettings
from codex_3d_mcp.hunyuan_mv.pipeline import HunyuanMVJobManager
from codex_3d_mcp.trellis.store import TrellisJobStore


class FakeRuntime:
    def __init__(self) -> None:
        self.shape_views: list[str] = []
        self.texture_image: Path | None = None

    def generate_shape(self, views, output, _params, _log, _cancel):
        self.shape_views = list(views)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"glTF" + b"\0" * 32)
        return {"views": self.shape_views, "faces": 120_000}

    def generate_texture(self, _mesh, front, output, _log, _cancel):
        self.texture_image = front
        output.write_bytes(b"glTF" + b"\0" * 32)
        return {"conditioning_view": "front", "faces": 100_000}

    def stop(self) -> None:
        return None

    def close(self) -> None:
        return None


def _settings(tmp_path: Path) -> HunyuanMVSettings:
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"fake")
    return HunyuanMVSettings(
        base_dir=tmp_path,
        server_name="test-hunyuan",
        server_version="test",
        output_dir=tmp_path / "outputs",
        database_path=tmp_path / "outputs" / "jobs.sqlite3",
        python_exe=tmp_path / "venv" / "python.exe",
        upstream_dir=tmp_path / "upstream",
        model_cache_dir=tmp_path / "models",
        blender_exe=blender,
        gltf_validator=tmp_path / "validator.exe",
        allowed_input_roots=(tmp_path,),
        gpu_lock_path=tmp_path / "gpu.lock",
        minimum_image_size=64,
        minimum_free_bytes=1,
        test_mode=True,
    )


def _image(path: Path, color: str, marker: str) -> Path:
    image = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 16, 104, 112), fill=color)
    draw.text((32, 32), marker, fill="black")
    image.save(path, "PNG")
    return path


def _fake_blender(job_dir: Path, _source: Path, result: Path) -> None:
    geometry = {
        "master": {
            "faces": 300_000,
            "degenerate_faces": 0,
            "non_manifold_edges": 0,
            "watertight": True,
        },
        "game": {"faces": 100_000},
        "lod1": {"faces": 50_000},
        "lod2": {"faces": 20_000},
    }
    geometry["master"].update(
        {"location": [0, 0, 0], "rotation_euler": [0, 0, 0], "scale": [1, 1, 1]}
    )
    for name in geometry:
        geometry[name].setdefault("watertight", True)
        geometry[name].setdefault("non_manifold_edges", 0)
        (job_dir / f"{name}.glb").write_bytes(b"glTF" + b"\0" * 32)
    previews = job_dir / "previews"
    previews.mkdir(exist_ok=True)
    for name in ("front", "back", "left", "right"):
        Image.new("RGB", (32, 32), "gray").save(previews / f"{name}.png")
    result.write_text(json.dumps({"worker_version": 3, "geometry": geometry}), encoding="utf-8")


def _wait(manager: HunyuanMVJobManager, job_id: str) -> dict:
    deadline = time.time() + 10
    while time.time() < deadline:
        record = manager.get(job_id)
        if record["state"] in {"completed", "failed", "failed_quality"}:
            return record
        time.sleep(0.02)
    raise AssertionError(manager.get(job_id))


def test_four_views_make_one_shape_and_no_fusion_stage(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    runtime = FakeRuntime()
    manager = HunyuanMVJobManager(
        settings, store=TrellisJobStore(settings.database_path), runtime=runtime
    )
    manager._run_blender = _fake_blender  # type: ignore[method-assign]
    paths = {
        name: _image(tmp_path / f"{name}.png", color, name)
        for name, color in zip(
            ("front", "back", "left", "right"),
            ("red", "blue", "green", "orange"),
            strict=True,
        )
    }
    job = manager.submit(
        "asymmetric asset",
        {name: str(path) for name, path in paths.items()},
        "painted stone",
        42,
        "standard",
        True,
    )
    result = _wait(manager, job["job_id"])
    assert result["state"] == "completed", result
    assert runtime.shape_views == ["front", "back", "left", "right"]
    assert runtime.texture_image is not None and runtime.texture_image.name == "front.png"
    assert "fusing" not in result["checkpoints"]
    assert {Path(item["path"]).name for item in result["artifacts"]} >= {
        "master.glb",
        "game.glb",
        "lod1.glb",
        "lod2.glb",
        "manifest.json",
    }
    manager.close()


def test_duplicate_views_fail_before_shape_generation(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    runtime = FakeRuntime()
    manager = HunyuanMVJobManager(
        settings, store=TrellisJobStore(settings.database_path), runtime=runtime
    )
    front = _image(tmp_path / "front.png", "red", "same")
    duplicate = tmp_path / "back.png"
    duplicate.write_bytes(front.read_bytes())
    job = manager.submit(
        "duplicate",
        {"front": str(front), "back": str(duplicate)},
        "",
        42,
        "draft",
        False,
    )
    result = _wait(manager, job["job_id"])
    assert result["state"] == "failed"
    assert result["error"]["code"] == InvalidImageError.code
    assert runtime.shape_views == []
    manager.close()


def test_front_view_is_required(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manager = HunyuanMVJobManager(
        settings, store=TrellisJobStore(settings.database_path), runtime=FakeRuntime()
    )
    back = _image(tmp_path / "back.png", "blue", "back")
    with pytest.raises(ValueError, match="front_image_path"):
        manager.submit("missing front", {"back": str(back)}, "", 42, "draft", False)
    manager.close()


@pytest.mark.parametrize("variant", ["game", "lod1", "lod2"])
def test_export_gate_rejects_cracked_variant_with_closed_master(tmp_path: Path, variant: str):
    settings = _settings(tmp_path)
    manager = HunyuanMVJobManager(
        settings, store=TrellisJobStore(settings.database_path), runtime=FakeRuntime()
    )
    try:
        job_dir = tmp_path / "final"
        job_dir.mkdir()
        result_path = job_dir / "blender-result.json"
        _fake_blender(job_dir, job_dir / "input.glb", result_path)
        metrics = json.loads(result_path.read_text())
        assert manager._qa(job_dir, metrics)["passed"]
        metrics["geometry"][variant].update(watertight=False, non_manifold_edges=12)
        report = manager._qa(job_dir, metrics)
        assert not report["passed"]
        assert not next(c for c in report["checks"] if c["name"] == f"{variant}_watertight")[
            "passed"
        ]
        assert next(c for c in report["checks"] if c["name"] == "master_watertight")["passed"]
    finally:
        manager.close()


@pytest.mark.parametrize("failure", ["coverage", "cpu", "triangle_ceiling", "topology", "preview"])
def test_profile_exports_require_complete_baking_and_geometry_evidence(tmp_path, failure):
    manager = HunyuanMVJobManager(_settings(tmp_path), runtime=FakeRuntime())
    try:
        output = tmp_path / "profiles"
        output.mkdir()
        result = output / "blender-result.json"
        _fake_blender(output, output / "source.glb", result)
        metrics = json.loads(result.read_text())
        metrics["prepared_profiles"] = True
        metrics["profile_triangle_budgets"] = {
            "full_game": 100000,
            "mobile": 50000,
            "browser": 20000,
        }
        metrics["gpu"] = {"devices": ["GPU"], "cpu_enabled": False}
        metrics["bakes"] = {
            name: {"passed": True, "coverage": 1.0} for name in ("mobile", "browser")
        }
        metrics["geometry"]["master"]["faces"] = 100000
        for name, alias in (("full_game", "game"), ("mobile", "lod1"), ("browser", "lod2")):
            metrics["geometry"][name] = dict(metrics["geometry"][alias])
            (output / f"{name}.glb").write_bytes((output / f"{alias}.glb").read_bytes())
            previews = output / "previews" / name
            previews.mkdir()
            for view in ("front", "back", "left", "right"):
                (previews / f"{view}.png").write_bytes(
                    (output / "previews" / f"{view}.png").read_bytes()
                )
        assert manager._qa(output, metrics)["passed"]
        if failure == "coverage":
            metrics["bakes"]["mobile"]["coverage"] = 0.9
        elif failure == "cpu":
            metrics["gpu"]["cpu_enabled"] = True
        elif failure == "triangle_ceiling":
            metrics["geometry"]["mobile"]["faces"] = 50001
        elif failure == "topology":
            metrics["geometry"]["browser"].update(watertight=False, non_manifold_edges=2)
        else:
            (output / "previews" / "browser" / "left.png").unlink()
        assert not manager._qa(output, metrics)["passed"]
    finally:
        manager.close()
