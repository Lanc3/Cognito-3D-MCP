from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from codex_3d_mcp.trellis.config import TrellisSettings
from codex_3d_mcp.trellis.textures import TextureProcessor

BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
RUN = os.getenv("RUN_BLENDER_INTEGRATION") == "1"


@pytest.mark.skipif(not RUN or not BLENDER.is_file(), reason="explicit Blender smoke test")
def test_world_yaw_applies_to_quaternion_import(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    fixture = Path(__file__).with_name("blender_rotation_fixture.py")
    worker = project / "src" / "codex_3d_mcp" / "trellis" / "blender_worker.py"
    result = tmp_path / "rotation.json"
    subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--factory-startup",
            "--python",
            str(fixture),
            "--",
            str(worker),
            str(result),
        ],
        check=True,
        timeout=60,
    )
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["point"] == pytest.approx([-1.0, -2.0, 3.0], abs=1e-6)
    assert payload["rotation_mode"] == "XYZ"
    assert payload["rotation_euler"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


@pytest.mark.skipif(not RUN or not BLENDER.is_file(), reason="explicit Blender smoke test")
def test_real_blender_build_and_finalize(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    fixture = Path(__file__).with_name("blender_fixture.py")
    worker = project / "src" / "codex_3d_mcp" / "trellis" / "blender_worker.py"
    subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--factory-startup",
            "--python",
            str(fixture),
            "--",
            str(tmp_path),
        ],
        check=True,
        timeout=120,
    )
    for name, color in (("front", (220, 45, 30)), ("back", (35, 80, 220))):
        image = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((30, 10, 98, 118), fill=color)
        image.save(tmp_path / f"{name}.png")

    build_result = tmp_path / "build.json"
    config = {
        "mode": "build",
        "job_dir": str(tmp_path),
        "front_glb": str(tmp_path / "front.glb"),
        "back_glb": str(tmp_path / "back.glb"),
        "front_image": str(tmp_path / "front.png"),
        "back_image": str(tmp_path / "back.png"),
        "texture_resolution": 128,
        "master_faces": 1000,
        "game_faces": 500,
        "lod_faces": [250, 100],
        "voxel_size": 0.05,
        "result_path": str(build_result),
    }
    config_path = tmp_path / "build-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--factory-startup",
            "--python",
            str(worker),
            "--",
            str(config_path),
        ],
        check=True,
        timeout=300,
    )
    assert build_result.is_file()
    build_metrics = json.loads(build_result.read_text(encoding="utf-8"))
    targets = {"master": 1000, "game": 500, "lod1": 250, "lod2": 100}
    for name, target in targets.items():
        geometry = build_metrics["geometry"][name]
        assert geometry["watertight"] is True
        assert geometry["degenerate_faces"] == 0
        assert target * 0.95 <= geometry["faces"] <= target * 1.05
    assert (tmp_path / "textures" / "baseColor.png").is_file()
    assert (tmp_path / "textures" / "normal.png").is_file()

    runtime = tmp_path / "runtime"
    settings = TrellisSettings(
        base_dir=tmp_path,
        server_name="test",
        server_version="test",
        output_dir=tmp_path,
        database_path=tmp_path / "jobs.sqlite3",
        runtime_dir=runtime,
        model_dir=runtime,
        trellis_server=runtime / "server.exe",
        trellis_cli=runtime / "cli.exe",
        realesrgan_exe=runtime / "real.exe",
        blender_exe=BLENDER,
        gltf_validator=runtime / "validator.exe",
        allowed_input_roots=(tmp_path,),
        gpu_lock_path=runtime / "gpu.lock",
        texture_resolution=128,
        test_mode=True,
    )
    maps = TextureProcessor(settings).assemble_maps(tmp_path / "textures", "painted metal")
    final_result = tmp_path / "final.json"
    config.update({"mode": "finalize", "maps": maps, "result_path": str(final_result)})
    config_path.write_text(json.dumps(config), encoding="utf-8")
    subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--factory-startup",
            "--python",
            str(worker),
            "--",
            str(config_path),
        ],
        check=True,
        timeout=300,
    )
    assert final_result.is_file()
    assert (tmp_path / "previews" / "front.png").is_file()
    assert (tmp_path / "game.glb").is_file()
