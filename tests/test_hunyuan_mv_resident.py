from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from codex_3d_mcp.errors import Codex3DError
from codex_3d_mcp.hunyuan_mv import paint_loader, safe_loader, worker
from codex_3d_mcp.hunyuan_mv.config import HunyuanMVSettings
from codex_3d_mcp.hunyuan_mv.runtime import ResidentStageRuntime


def _module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _cutout(tmp_path, name):
    path = tmp_path / name
    image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    image.putpixel((1, 1), (200, 100, 50, 255))
    image.save(path)
    return str(path)


class Mesh:
    faces = SimpleNamespace(shape=(12, 3))
    vertices = SimpleNamespace(shape=(8, 3))

    def __init__(self, source=""):
        self.source = source

    def export(self, path, **kwargs):
        Path(path).write_bytes(b"glTF" + b"0" * 28)


def test_shape_model_loads_once_but_each_asset_has_fresh_seed_and_parameters(tmp_path, monkeypatch):
    calls, loads = [], []

    class Generator:
        def manual_seed(self, seed):
            self.seed = seed
            return self

    def pipeline(**kwargs):
        calls.append(kwargs)
        return object()

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return pipeline, {"checkpoint_loads": 1}

    monkeypatch.setattr(worker, "_SHAPE_CACHE", {})
    monkeypatch.setattr(worker, "_activate_upstream", lambda: None)
    monkeypatch.setattr(safe_loader, "load_shape_pipeline", load)
    _module(monkeypatch, "torch", Generator=Generator)
    _module(monkeypatch, "hy3dgen.shapegen.pipelines", export_to_trimesh=lambda output: [Mesh()])
    base = {
        "views": {"front": _cutout(tmp_path, "front.png")},
        "model_path": "pinned-shape",
        "subfolder": "mv",
        "shape_loader_id": safe_loader.SAFE_LOADER_ID,
        "seed": 42,
        "steps": 20,
        "guidance_scale": 5,
        "octree_resolution": 196,
        "num_chunks": 8000,
        "output": str(tmp_path / "first.glb"),
    }
    worker._shape(base)
    worker._shape(
        {
            **base,
            "seed": 83,
            "steps": 30,
            "octree_resolution": 256,
            "output": str(tmp_path / "second.glb"),
        }
    )
    assert len(loads) == 1
    assert loads[0][1]["device"] == "cuda"
    assert [call["generator"].seed for call in calls] == [42, 83]
    assert calls[0]["generator"] is not calls[1]["generator"]
    assert [call["num_inference_steps"] for call in calls] == [20, 30]
    assert [call["octree_resolution"] for call in calls] == [196, 256]


def test_paint_model_is_resident_and_updates_both_atlas_setters_without_offload(
    tmp_path, monkeypatch
):
    calls, loads, texture_sizes, render_sizes = [], [], [], []

    class Paint:
        def __init__(self):
            self.config = SimpleNamespace()
            self.render = SimpleNamespace(
                set_default_texture_resolution=texture_sizes.append,
                set_default_render_resolution=render_sizes.append,
            )

        def enable_model_cpu_offload(self):
            pytest.fail("Default resident Paint must stay on GPU")

        def __call__(self, mesh, image):
            calls.append((mesh.source, image, self.config.texture_size, self.config.render_size))
            return mesh

    monkeypatch.setattr(worker, "_PAINT_CACHE", {})
    monkeypatch.setattr(worker, "_activate_upstream", lambda: None)
    _module(monkeypatch, "trimesh", load=lambda path, **kwargs: Mesh(path))
    proof = {"ready": True, "safe_loader": paint_loader.PAINT_LOADER_ID, "device": "cuda"}

    def load_pipeline(*args, **kwargs):
        loads.append((args, kwargs))
        return Paint(), proof

    monkeypatch.setattr(paint_loader, "load_paint_pipeline", load_pipeline)
    base = {
        "model_path": "pinned-paint",
        "subfolder": "base",
        "mesh": "first.glb",
        "image": _cutout(tmp_path, "first.png"),
        "output": str(tmp_path / "painted1.glb"),
        "texture_resolution": 1024,
    }
    first = worker._texture(base)
    second = worker._texture(
        {
            **base,
            "mesh": "second.glb",
            "image": _cutout(tmp_path, "second.png"),
            "texture_resolution": 4096,
            "output": str(tmp_path / "painted2.glb"),
        }
    )
    assert len(loads) == 1
    assert [call[0] for call in calls] == ["first.glb", "second.glb"]
    assert calls[0][1] != calls[1][1]
    assert texture_sizes == render_sizes == [1024, 4096]
    assert [call[2:] for call in calls] == [(1024, 1024), (4096, 4096)]
    assert first["runtime"] == second["runtime"] == proof
    with pytest.raises(RuntimeError, match="CPU offload is disabled"):
        worker._texture({**base, "low_vram_mode": True})
    assert len(loads) == 1


def _runtime(tmp_path, monkeypatch):
    settings = replace(
        HunyuanMVSettings.from_env(tmp_path),
        python_exe=tmp_path / "nonexistent-python.exe",
        model_cache_dir=tmp_path / "cache",
        gpu_lock_path=tmp_path / "gpu.lock",
    )
    runtime = ResidentStageRuntime(settings)
    events = []
    runtime.gpu_lease = SimpleNamespace(
        acquire=lambda **kwargs: events.append("acquire"), release=lambda: events.append("release")
    )
    processes = []

    class Process:
        def __init__(self, command, **kwargs):
            self.stage = command[-1]
            self.alive = True
            self.stdin = SimpleNamespace(close=lambda: events.append(f"{self.stage}:stdin-close"))
            events.append(f"{self.stage}:spawn")
            processes.append(self)

        def poll(self):
            return None if self.alive else 0

        def wait(self, timeout):
            events.append(f"{self.stage}:joined")
            self.alive = False
            return 0

    monkeypatch.setattr("codex_3d_mcp.hunyuan_mv.runtime.subprocess.Popen", Process)
    return runtime, events, processes


def test_phase_reuses_worker_and_joins_it_before_cpu_or_next_gpu_phase(tmp_path, monkeypatch):
    runtime, events, processes = _runtime(tmp_path, monkeypatch)
    cancel = threading.Event()
    runtime.begin_stage("shape", tmp_path / "shape.log", cancel)
    runtime.begin_stage("shape", tmp_path / "shape.log", cancel)
    assert len(processes) == 1
    events.clear()
    runtime.begin_stage("remesh", tmp_path / "remesh.log", cancel)
    assert events == ["shape:stdin-close", "shape:joined", "release", "acquire"]
    assert runtime._process is None and len(processes) == 1
    events.clear()
    runtime.begin_stage("texture", tmp_path / "paint.log", cancel)
    assert events == ["release", "acquire", "texture:spawn"]
    runtime.end_stage()
    assert not any(process.alive for process in processes)


def test_failed_worker_termination_must_retain_resource_ownership(tmp_path, monkeypatch):
    runtime, events, processes = _runtime(tmp_path, monkeypatch)
    runtime.begin_stage("shape", tmp_path / "shape.log", threading.Event())
    process = processes[0]

    def never_exits(timeout):
        raise subprocess.TimeoutExpired("shape", timeout)

    process.wait = never_exits
    monkeypatch.setattr(
        "codex_3d_mcp.hunyuan_mv.runtime._terminate_process_tree", lambda proc: None
    )
    events.clear()
    with pytest.raises(Codex3DError, match="still alive"):
        runtime.begin_stage("texture", tmp_path / "paint.log", threading.Event())
    assert "release" not in events
    assert "texture:spawn" not in events
    assert runtime._process is process
    assert runtime.active_stage == "shape"
    process.alive = False
    process.wait = lambda timeout: 0
    runtime.end_stage()


def test_memory_pressure_during_inference_stops_worker_before_releasing_lease(
    tmp_path, monkeypatch
):
    runtime, events, processes = _runtime(tmp_path, monkeypatch)
    runtime.begin_stage("shape", tmp_path / "shape.log", threading.Event())
    process = processes[0]
    process.stdin.write = lambda value: None
    process.stdin.flush = lambda: None

    def memory_pressure(*args, **kwargs):
        raise Codex3DError("Host commit reserve exhausted", code="REMESH_RESOURCE_GUARD")

    monkeypatch.setattr("codex_3d_mcp.hunyuan_mv.remesh._check_memory", memory_pressure)
    events.clear()
    with pytest.raises(Codex3DError, match="commit reserve"):
        runtime._run_worker(
            {"action": "shape"}, tmp_path / "shape.glb", tmp_path / "shape.log", threading.Event()
        )
    assert events == ["shape:stdin-close", "shape:joined", "release"]
    assert runtime._process is None and not process.alive


def test_batch_forces_gpu_even_with_legacy_cpu_offload_configuration(tmp_path, monkeypatch):
    from codex_3d_mcp.hunyuan_mv.stage_processor import BatchStageProcessor, _sha256

    settings = replace(HunyuanMVSettings.from_env(tmp_path), low_vram_mode=True)
    monkeypatch.setattr(
        "codex_3d_mcp.hunyuan_mv.stage_processor.ResidentStageRuntime",
        lambda configured: SimpleNamespace(settings=configured),
    )
    monkeypatch.setattr(
        "codex_3d_mcp.hunyuan_mv.stage_processor.AutoRemesherRuntime",
        lambda configured: SimpleNamespace(settings=configured),
    )
    processor = BatchStageProcessor(settings, SimpleNamespace())
    remeshed = tmp_path / "remeshed.glb"
    Mesh().export(remeshed)
    monkeypatch.setattr(
        processor, "_accepted_profiles",
        lambda asset: {"profiles": {"full_game": {
            "output": str(remeshed), "output_sha256": _sha256(remeshed), "triangle_budget": 100000,
        }}},
    )
    monkeypatch.setattr(
        "codex_3d_mcp.hunyuan_mv.stage_processor._glb_triangle_count", lambda path: 12
    )
    assert processor.runtime.settings.low_vram_mode is False
    processor.runtime.active_stage = "texture"
    calls = []

    def generate(mesh, front, output, log, cancel):
        calls.append(processor.runtime.settings)
        Mesh().export(output)
        return {"faces": 12}

    processor.runtime.generate_texture = generate
    monkeypatch.setattr(
        "codex_3d_mcp.hunyuan_mv.stage_processor._has_color_texture", lambda path: True
    )
    directory = tmp_path / "batch" / "asset" / "attempt"
    directory.mkdir(parents=True)
    processor.run(
        "paint",
        {
            "params": {"texture_resolution": 4096},
            "stages": {
                "references": {"prepared_views": {"front": "front.png"}},
                "remesh": {"output": "remeshed.glb"},
            },
        },
        directory,
        threading.Event(),
    )
    assert calls[0].low_vram_mode is False
    assert calls[0].texture_resolution == 4096
