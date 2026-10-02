from __future__ import annotations

from pathlib import Path

import huggingface_hub

from codex_3d_mcp.config import Settings
from codex_3d_mcp.model import StableFast3DAdapter


def _adapter(tmp_path: Path) -> StableFast3DAdapter:
    return StableFast3DAdapter(
        Settings(
            model_cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "outputs",
            allowed_input_roots=(tmp_path,),
        )
    )


def test_partial_model_cache_is_not_complete(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    snapshot = (
        adapter.settings.model_cache_dir
        / "hub"
        / "models--stabilityai--stable-fast-3d"
        / "snapshots"
        / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.yaml").write_text("model: config", encoding="utf-8")

    assert adapter._model_cache_complete() is False

    (snapshot / "model.safetensors").write_bytes(b"weights")

    assert adapter._model_cache_complete() is True


def test_gated_model_access_is_reported_without_exposing_token(tmp_path: Path, monkeypatch) -> None:
    adapter = _adapter(tmp_path)
    secret_token = "hf_do-not-expose-this"

    gated_error = type("GatedRepoError", (Exception,), {})

    def deny_download(**_kwargs) -> None:
        raise gated_error("request contained a secret")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", deny_download)

    access, error = adapter._status_model_access(secret_token)

    assert access is False
    assert error is not None
    assert "has not been granted access" in error
    assert secret_token not in error


def test_spar3d_uses_its_own_vendor_and_cache_identity(tmp_path: Path) -> None:
    adapter = StableFast3DAdapter(
        Settings(
            backend="spar3d",
            model_id="stabilityai/stable-point-aware-3d",
            model_cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "outputs",
            allowed_input_roots=(tmp_path,),
        )
    )
    snapshot = (
        adapter.settings.model_cache_dir
        / "hub"
        / "models--stabilityai--stable-point-aware-3d"
        / "snapshots"
        / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.yaml").write_text("model: config", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    assert adapter.model_name == "SPAR3D"
    assert adapter._model_cache_complete() is True


def test_shared_huggingface_token_path_precedes_home_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    shared = tmp_path / "shared-token"
    shared.write_text("hf_shared-fixture", encoding="utf-8")
    home = tmp_path / "huggingface-home"
    home.mkdir()
    (home / "token").write_text("hf_home-fixture", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(shared))
    monkeypatch.setenv("HF_HOME", str(home))
    adapter = _adapter(tmp_path)
    assert adapter._configured_hf_token() == "hf_shared-fixture"
    monkeypatch.delenv("HF_TOKEN_PATH")
    assert adapter._configured_hf_token() == "hf_home-fixture"
