from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from codex_3d_mcp.errors import Codex3DError
from codex_3d_mcp.hunyuan_mv.config import HunyuanMVSettings
from codex_3d_mcp.hunyuan_mv.runtime import _validate_safe_loader_proof
from codex_3d_mcp.hunyuan_mv.safe_loader import SAFE_LOADER_ID, load_shape_pipeline


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


def _proof(settings: HunyuanMVSettings) -> dict:
    checkpoint = (
        settings.model_cache_dir
        / "hub"
        / "models--tencent--Hunyuan3D-2mv"
        / "snapshots"
        / settings.shape_revision
        / settings.shape_subfolder
        / "model.fp16.ckpt"
    )
    return {
        "runtime": {
            "ready": True,
            "safe_loader": SAFE_LOADER_ID,
            "checkpoint": str(checkpoint),
            "checkpoint_format": "torch_zip",
            "checkpoint_loads": 1,
            "checkpoint_mmap": False,
            "vae_preallocated": True,
            "vae_owned_copy": True,
            "meta_tensor_count": 0,
            "cuda_synchronized": True,
        }
    }


def test_safe_loader_source_keeps_checkpoint_storage_owned() -> None:
    source = inspect.getsource(load_shape_pipeline)
    assert "mmap=False" in source
    assert "assign=False" in source
    assert "with torch.device(\"meta\")" in source
    assert ".from_pretrained(" not in source
    assert source.index('vae = instantiate_from_config(config["vae"])') < source.index(
        "checkpoint = torch.load("
    )


def test_shape_result_requires_complete_runtime_proof(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _validate_safe_loader_proof(_proof(settings), settings)

    invalid = _proof(settings)
    invalid["runtime"]["vae_owned_copy"] = False
    with pytest.raises(Codex3DError) as caught:
        _validate_safe_loader_proof(invalid, settings)
    assert caught.value.code == "HUNYUAN_SAFE_LOADER_UNPROVEN"


def test_shape_result_without_runtime_proof_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(Codex3DError) as caught:
        _validate_safe_loader_proof({}, _settings(tmp_path))
    assert caught.value.code == "HUNYUAN_SAFE_LOADER_UNPROVEN"
