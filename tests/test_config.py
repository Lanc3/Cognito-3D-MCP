from pathlib import Path

from codex_3d_mcp.config import Settings, data_root
from codex_3d_mcp.trellis.config import TrellisSettings


def test_spar3d_backend_selects_matching_default_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_3D_BACKEND", "spar3d")
    monkeypatch.delenv("CODEX_3D_MODEL_ID", raising=False)

    settings = Settings.from_env(tmp_path)

    assert settings.backend == "spar3d"
    assert settings.model_id == "stabilityai/stable-point-aware-3d"


def test_bidirectional_backend_selects_spar3d_runtime(tmp_path: Path, monkeypatch) -> None:
    python = tmp_path / "spar3d-python.exe"
    cache = tmp_path / "spar3d-cache"
    monkeypatch.setenv("CODEX_BIDIRECTIONAL_RECONSTRUCTION_BACKEND", "spar3d")
    monkeypatch.setenv("CODEX_BIDIRECTIONAL_SPAR3D_PYTHON", str(python))
    monkeypatch.setenv("CODEX_BIDIRECTIONAL_SPAR3D_MODEL_CACHE", str(cache))

    settings = TrellisSettings.from_env(tmp_path)

    assert settings.reconstruction_backend == "spar3d"
    assert settings.spar3d_python == python
    assert settings.spar3d_model_cache_dir == cache


def test_default_data_root_uses_current_users_local_appdata(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_3D_DATA_ROOT", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local app data"))
    assert data_root(tmp_path) == tmp_path / "local app data" / "Cognito-3D-mcp"


def test_bidirectional_and_hunyuan_share_portable_gpu_lock(tmp_path, monkeypatch):
    from codex_3d_mcp.hunyuan_mv.config import HunyuanMVSettings

    monkeypatch.setenv("CODEX_3D_DATA_ROOT", str(tmp_path / "user data"))
    monkeypatch.delenv("CODEX_3D_GPU_LOCK", raising=False)
    lock = TrellisSettings.from_env(tmp_path).gpu_lock_path
    assert lock == HunyuanMVSettings.from_env(tmp_path).gpu_lock_path
    assert lock == Settings.from_env(tmp_path).gpu_lock_path
