"""SPAR3D supervision tests without importing torch or running a model."""

import io
import subprocess
from types import SimpleNamespace

import pytest

from codex_3d_mcp.trellis.config import TrellisSettings
from codex_3d_mcp.trellis.spar3d_runtime import Spar3DRuntime


def test_failed_termination_retains_worker_and_gpu_lease(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_3D_DATA_ROOT", str(tmp_path / "data"))
    runtime = Spar3DRuntime(TrellisSettings.from_env(tmp_path))
    process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(), pid=123,
                              poll=lambda: None)

    def cannot_exit(timeout):
        raise subprocess.TimeoutExpired("fixture", timeout)

    def cannot_kill():
        raise OSError("Fixture termination failed")

    process.wait = cannot_exit
    process.kill = cannot_kill
    process.terminate = lambda: None
    runtime._process = process
    monkeypatch.setattr("codex_3d_mcp.trellis.spar3d_runtime.subprocess.run", lambda *a, **kw: None)
    try:
        with pytest.raises(OSError, match="termination failed"):
            with runtime.session(tmp_path / "worker.log"):
                pass
    finally:
        assert runtime._process is process
        assert runtime.gpu_lease._handle is not None
        runtime.gpu_lease.release()


def test_model_cache_requires_config_and_weights_in_same_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_3D_DATA_ROOT", str(tmp_path / "data"))
    runtime = Spar3DRuntime(TrellisSettings.from_env(tmp_path))
    runtime.settings.spar3d_python.parent.mkdir(parents=True)
    runtime.settings.spar3d_python.write_bytes(b"fake")
    snapshots = (runtime.settings.spar3d_model_cache_dir / "hub"
                 / "models--stabilityai--stable-point-aware-3d" / "snapshots")
    for name in ("config-only", "weights-only"):
        (snapshots / name).mkdir(parents=True)
    (snapshots / "config-only" / "config.yaml").write_text("model: {}")
    (snapshots / "weights-only" / "model.safetensors").write_bytes(b"weights")
    assert runtime.verify_runtime_hash()[0] is False
    (snapshots / "config-only" / "model.safetensors").write_bytes(b"weights")
    assert runtime.verify_runtime_hash() == (True, None)
