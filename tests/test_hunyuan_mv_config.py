from __future__ import annotations

from pathlib import Path

from codex_3d_mcp import __version__
from codex_3d_mcp.hunyuan_mv.config import HunyuanMVSettings


def test_defaults_are_isolated_and_use_named_multiview_model(tmp_path: Path) -> None:
    settings = HunyuanMVSettings.from_env(tmp_path)
    assert settings.shape_model_id == "tencent/Hunyuan3D-2mv"
    assert settings.shape_subfolder == "hunyuan3d-dit-v2-mv"
    assert settings.texture_model_id == "tencent/Hunyuan3D-2"
    assert settings.output_dir == (tmp_path / "outputs" / "hunyuan-mv").resolve()
    assert settings.python_exe != tmp_path / ".venv-spar3d" / "Scripts" / "python.exe"


def test_allowed_roots_do_not_accept_sibling_prefixes(tmp_path: Path) -> None:
    allowed = tmp_path / "inputs"
    sibling = tmp_path / "inputs-evil" / "view.png"
    settings = HunyuanMVSettings.from_env(tmp_path)
    object.__setattr__(settings, "allowed_input_roots", (allowed,))
    assert settings.is_allowed_input(allowed / "view.png")
    assert not settings.is_allowed_input(sibling)


def test_cpu_configuration_cannot_exceed_two_logical_processors(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_AUTOREMESHER_THREADS", "128")
    assert HunyuanMVSettings.from_env(tmp_path).remesh_threads == 2


def test_portable_data_root_defaults_and_explicit_runtime_override(tmp_path, monkeypatch):
    for name in (
        "CODEX_HUNYUAN_MV_RUNTIME_ROOT", "CODEX_HUNYUAN_MV_PYTHON",
        "CODEX_HUNYUAN_MV_MODEL_CACHE", "CODEX_HUNYUAN_MV_REPAIR_PACKAGES",
        "CODEX_AUTOREMESHER_EXE", "CODEX_3D_GPU_LOCK",
    ):
        monkeypatch.delenv(name, raising=False)
    data = tmp_path / "user data"
    monkeypatch.setenv("CODEX_3D_DATA_ROOT", str(data))
    settings = HunyuanMVSettings.from_env(tmp_path)
    assert settings.python_exe.is_relative_to(data / "runtime" / "hunyuan3d-2mv")
    assert settings.model_cache_dir == data / "models" / "hunyuan3d-2mv"
    assert settings.repair_package_dir == data / "runtime" / "shape-repair-packages"
    assert settings.autoremesher_exe == data / "runtime" / "autoremesher" / "autoremesher.exe"
    assert settings.gpu_lock_path == data / "runtime" / "gpu.lock"
    assert settings.server_version == __version__ == "0.4.0"
    override = tmp_path / "custom runtime"
    monkeypatch.setenv("CODEX_HUNYUAN_MV_RUNTIME_ROOT", str(override))
    assert HunyuanMVSettings.from_env(tmp_path).python_exe.is_relative_to(override)
