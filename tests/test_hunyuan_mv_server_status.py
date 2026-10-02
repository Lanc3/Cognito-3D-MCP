"""Status remains inspectable when the Windows generation runtime is unsupported."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_3d_mcp.errors import Codex3DError
from codex_3d_mcp.hunyuan_mv import remesh, stage_processor


def test_non_windows_server_status_reports_unsupported_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(remesh, "os", SimpleNamespace(name="posix"))
    for key, value in {
        "CODEX_3D_DATA_ROOT": tmp_path / "data",
        "CODEX_HUNYUAN_MV_OUTPUT_DIR": tmp_path / "outputs",
        "CODEX_HUNYUAN_MV_PYTHON": tmp_path / "absent-python",
        "CODEX_HUNYUAN_MV_RUNTIME_ROOT": tmp_path / "absent-runtime",
        "CODEX_AUTOREMESHER_EXE": tmp_path / "autoremesher.exe",
        "CODEX_3D_GPU_LOCK": tmp_path / "gpu.lock",
    }.items():
        monkeypatch.setenv(key, str(value))

    def forbidden_probe(*args, **kwargs):
        pytest.fail("Unsupported-platform status must not probe Windows APIs or launch a worker")

    monkeypatch.setattr(stage_processor, "_memory_status", forbidden_probe)
    monkeypatch.setattr(remesh.AutoRemesherRuntime, "_run", forbidden_probe)
    monkeypatch.setattr(
        stage_processor.ResidentStageRuntime, "preflight",
        lambda self: {"ready_for_generation": True},
    )
    path = Path(stage_processor.__file__).with_name("server.py")
    spec = importlib.util.spec_from_file_location(
        "codex_3d_mcp.hunyuan_mv._platform_status_test", path
    )
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    try:
        status = server.server_status()
        assert status["runtime"]["ready_for_generation"] is False
        assert status["runtime"]["host_memory"]["ready"] is False
        assert status["remesher"]["ready"] is False
        assert status["remesher"]["platform_supported"] is False
        assert status["remesher"]["error"]["code"] == "REMESH_PLATFORM"
        assert "Windows" in status["remesher"]["error"]["message"]
        with pytest.raises(Codex3DError, match="requires Windows") as caught:
            server.batches.processor.remesher.remesh(
                tmp_path / "source.glb", tmp_path / "output.glb", {}, tmp_path / "log"
            )
        assert caught.value.code == "REMESH_PLATFORM"
    finally:
        server.batches.close()
        server.jobs.close()
