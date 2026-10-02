from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from codex_3d_mcp.errors import Codex3DError
from codex_3d_mcp.hunyuan_mv.config import HunyuanMVSettings
from codex_3d_mcp.hunyuan_mv.runtime import (
    PINNED_HF_STACK,
    PINNED_TORCH_STACK,
    HunyuanMVRuntime,
)
from codex_3d_mcp.hunyuan_mv.safe_loader import SAFE_LOADER_ID


def _settings(tmp_path: Path) -> HunyuanMVSettings:
    runtime = tmp_path / "runtime"
    return HunyuanMVSettings(
        base_dir=tmp_path,
        server_name="test",
        server_version="test",
        output_dir=tmp_path / "outputs",
        database_path=tmp_path / "outputs" / "jobs.sqlite3",
        python_exe=runtime / "venv" / "Scripts" / "python.exe",
        upstream_dir=runtime / "Hunyuan3D-2",
        model_cache_dir=tmp_path / "models",
        blender_exe=tmp_path / "blender.exe",
        gltf_validator=tmp_path / "validator.exe",
        allowed_input_roots=(tmp_path,),
        gpu_lock_path=tmp_path / "gpu.lock",
    )


def test_install_manifest_must_match_all_pins(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manifest = settings.python_exe.parents[2] / "install-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "upstream_commit": settings.upstream_commit,
                "shape_revision": settings.shape_revision,
                "texture_revision": settings.texture_revision,
                "shape_loader": SAFE_LOADER_ID,
                **PINNED_TORCH_STACK,
                **PINNED_HF_STACK,
                "cuda_verified": True,
            }
        ),
        encoding="utf-8",
    )
    runtime = HunyuanMVRuntime(settings)
    assert runtime.verify_install_manifest() == (True, None)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["shape_revision"] = "wrong"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    verified, error = runtime.verify_install_manifest()
    assert verified is False
    assert error == "shape_revision does not match the pinned runtime"


def test_texture_worker_uses_exact_local_snapshot_and_supported_subfolder(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    runtime = HunyuanMVRuntime(settings)
    captured = {}

    def fake_run(request, _output, _log, _cancel):
        captured.update(request)
        return {"conditioning_view": "front"}

    monkeypatch.setattr(runtime, "_run_worker", fake_run)
    runtime.generate_texture(
        tmp_path / "shape.glb",
        tmp_path / "front.png",
        tmp_path / "textured.glb",
        tmp_path / "texture.log",
        Event(),
    )
    assert captured["model_path"].endswith(settings.texture_revision)
    assert captured["subfolder"] == "hunyuan3d-paint-v2-0"
    assert "revision" not in captured


def test_preflight_requires_owned_checkpoint_not_safetensors(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    snapshot = (
        settings.model_cache_dir
        / "hub"
        / "models--tencent--Hunyuan3D-2mv"
        / "snapshots"
        / settings.shape_revision
        / settings.shape_subfolder
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (snapshot / "model.fp16.safetensors").write_bytes(b"legacy")
    runtime = HunyuanMVRuntime(settings)
    assert runtime.preflight()["shape_cache_present"] is False

    with zipfile.ZipFile(snapshot / "model.fp16.ckpt", "w") as archive:
        archive.writestr("data.pkl", b"owned")
    status = runtime.preflight()
    assert status["shape_cache_present"] is True
    assert status["shape_safe_loader"] == "codex-hunyuan-windows-owned-v1"


def test_dependency_probe_inspects_configured_worker_interpreter(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    settings.python_exe.parent.mkdir(parents=True)
    settings.python_exe.write_bytes(b"fake executable")
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps({"torch": "2.9.1+cu130", "hy3dgen": True}))

    monkeypatch.setattr("codex_3d_mcp.hunyuan_mv.runtime.subprocess.run", probe)
    runtime = HunyuanMVRuntime(settings)
    assert calls[0][0] == str(settings.python_exe)
    assert str(settings.upstream_dir) in calls[0]
    assert runtime._dependency_probe["hy3dgen"] is True


def test_manifest_with_non_object_json_fails_closed(tmp_path):
    settings = _settings(tmp_path)
    manifest = settings.python_exe.parents[2] / "install-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("[]", encoding="utf-8")
    assert HunyuanMVRuntime(settings).verify_install_manifest()[0] is False


def test_worker_cannot_succeed_using_stale_attempt_artifacts(tmp_path, monkeypatch):
    settings = replace(_settings(tmp_path), test_mode=True)
    runtime = HunyuanMVRuntime(settings)
    output = tmp_path / "textured.glb"
    output.write_bytes(b"glTF" + b"\0" * 32)
    result = output.with_suffix(".texture.result.json")
    result.write_text("{}", encoding="utf-8")
    process = SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr("codex_3d_mcp.hunyuan_mv.runtime.subprocess.Popen", lambda *a, **kw: process)
    with pytest.raises(Codex3DError, match="no valid GLB"):
        runtime._run_worker({"action": "texture"}, output, tmp_path / "worker.log", Event())
    assert not result.exists()
    assert runtime.gpu_lease._handle is None


def test_failed_worker_termination_retains_gpu_lease(tmp_path, monkeypatch):
    settings = replace(_settings(tmp_path), test_mode=True)
    runtime = HunyuanMVRuntime(settings)
    cancel = Event()
    process = SimpleNamespace(poll=lambda: None, returncode=None)

    def launch(*args, **kwargs):
        cancel.set()
        return process

    def cannot_stop(process):
        raise OSError("Fixture termination failure")

    monkeypatch.setattr("codex_3d_mcp.hunyuan_mv.runtime.subprocess.Popen", launch)
    monkeypatch.setattr("codex_3d_mcp.hunyuan_mv.runtime._terminate_process_tree", cannot_stop)
    try:
        with pytest.raises(OSError, match="termination failure"):
            runtime._run_worker(
                {"action": "texture"}, tmp_path / "output.glb", tmp_path / "worker.log", cancel
            )
        assert runtime._process is process
        assert runtime.gpu_lease._handle is not None
    finally:
        runtime.gpu_lease.release()
