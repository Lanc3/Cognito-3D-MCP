"""Load the pinned Paint UNet without duplicate CPU weight allocations."""

from __future__ import annotations

import gc
import importlib
import inspect
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .remesh import _check_memory

PAINT_LOADER_ID = "codex-hunyuan-paint-owned-meta-v1"


def _record_stage(proof: dict[str, Any], stage: str) -> None:
    memory = _check_memory(
        {"memory_limit_mb": 0, "min_available_mb": 2048}, starting=False
    )
    entry = {"stage": stage, "monotonic_seconds": time.monotonic(), "memory": memory}
    proof.setdefault("loading_stages", []).append(entry)
    print("[paint-loader] " + json.dumps(entry), flush=True)


def _module_tensors(module):
    yield from module.named_parameters()
    yield from module.named_buffers()


def _load_unet(unet_class, directory: Path, torch_dtype, proof: dict[str, Any]):
    import torch

    module = importlib.import_module(unet_class.__module__)
    with (directory / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint_path = directory / "diffusion_pytorch_model.bin"
    _record_stage(proof, "before_meta_construction")
    with torch.device("meta"):
        unet = unet_class(module.UNet2DConditionModel(**config))
    proof["architecture_constructed_on_meta"] = True
    _record_stage(proof, "architecture_on_meta")
    # Non-mapped storage is required: Windows CUDA adoption of mapped tensors
    # previously crashed the shape loader. This owns exactly one CPU checkpoint.
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=False
    )
    proof["checkpoint_loads"] += 1
    proof["checkpoint_dtypes"] = sorted({str(value.dtype) for value in checkpoint.values()})
    _record_stage(proof, "checkpoint_loaded")
    unet.load_state_dict(checkpoint, strict=True, assign=True)
    proof["strict_assignment"] = True
    del checkpoint
    gc.collect()
    _record_stage(proof, "checkpoint_dictionary_released")
    # This matches the upstream loader's requested dtype; the pinned weights
    # already use float16, so no full precision intermediate is needed.
    unet.to(dtype=torch_dtype)
    remaining = [name for name, tensor in _module_tensors(unet) if tensor.is_meta]
    proof["remaining_meta_tensors"] = len(remaining)
    if remaining:
        raise RuntimeError(f"Paint UNet contains unmaterialized tensors: {remaining[:8]}")
    proof["loaded_dtype"] = str(torch_dtype)
    _record_stage(proof, "unet_materialized")
    return unet


@contextmanager
def _patched_unet_loader(unet_class, directory: Path, proof: dict[str, Any]):
    original = inspect.getattr_static(unet_class, "from_pretrained")

    def from_pretrained(path, **kwargs):
        import torch

        if Path(path).resolve() != directory:
            raise RuntimeError("Paint UNet loader received an unpinned checkpoint directory")
        if proof["checkpoint_loads"]:
            raise RuntimeError("Paint UNet checkpoint must be loaded exactly once")
        dtype = kwargs.pop("torch_dtype", torch.float32)
        return _load_unet(unet_class, directory, dtype, proof)

    unet_class.from_pretrained = staticmethod(from_pretrained)
    try:
        yield
    finally:
        unet_class.from_pretrained = original
        proof["loader_method_restored"] = (
            inspect.getattr_static(unet_class, "from_pretrained") is original
        )


def load_paint_pipeline(model_path: str | Path, *, subfolder: str):
    import torch
    from diffusers.utils.dynamic_modules_utils import get_class_from_dynamic_module
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("Paint requires CUDA; CPU inference/offload is unsupported")
    root = Path(model_path).resolve()
    directory = root / subfolder / "unet"
    module_path = directory / "modules.py"
    checkpoint_path = directory / "diffusion_pytorch_model.bin"
    for path in (module_path, checkpoint_path, directory / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    # Diffusers resolves this custom class from the snapshot, which differs
    # from the similarly named upstream Python source. Reuse its cached class.
    unet_class = get_class_from_dynamic_module(
        str(directory), module_file="modules.py", class_name="UNet2p5DConditionModel"
    )
    proof: dict[str, Any] = {
        "ready": False,
        "safe_loader": PAINT_LOADER_ID,
        "module_path": str(module_path),
        "dynamic_class_module": unet_class.__module__,
        "checkpoint": str(checkpoint_path),
        "checkpoint_mmap": False,
        "checkpoint_loads": 0,
        "remaining_meta_tensors": -1,
        "loader_method_restored": False,
        "cpu_offload": False,
    }
    _record_stage(proof, "before_pipeline_load")
    with _patched_unet_loader(unet_class, directory, proof):
        pipeline = Hunyuan3DPaintPipeline.from_pretrained(str(root), subfolder=subfolder)
    if proof["checkpoint_loads"] != 1:
        raise RuntimeError("Diffusers did not use the required safe Paint UNet loader")
    devices = set()
    for name in ("delight_model", "multiview_model"):
        component_pipeline = pipeline.models[name].pipeline
        for component in ("unet", "vae", "text_encoder"):
            for tensor_name, tensor in _module_tensors(getattr(component_pipeline, component)):
                if tensor.is_meta or tensor.device.type != "cuda":
                    raise RuntimeError(
                        f"Paint {name}.{component}.{tensor_name} is not materialized on CUDA"
                    )
                devices.add(str(tensor.device))
    torch.cuda.synchronize()
    proof["cuda_synchronized"] = True
    proof["devices"] = sorted(devices)
    proof["device"] = "cuda"
    proof["ready"] = True
    _record_stage(proof, "pipeline_on_gpu")
    print("[paint-loader] proof " + json.dumps(proof), flush=True)
    return pipeline, proof
