"""Windows-safe Hunyuan3D-2mv shape checkpoint loading.

The upstream safetensors/direct-CUDA path can terminate the interpreter on
Windows while CUDA adopts file-backed storage.  This loader keeps every
transfer source in process-owned memory and publishes an auditable proof that
the guarded path actually ran.
"""

from __future__ import annotations

import gc
import zipfile
from pathlib import Path
from typing import Any

SAFE_LOADER_ID = "codex-hunyuan-windows-owned-v1"


def shape_checkpoint_path(
    model_path: str | Path, subfolder: str, variant: str = "fp16"
) -> Path:
    return Path(model_path).resolve() / subfolder / f"model.{variant}.ckpt"


def is_torch_zip_checkpoint(path: Path) -> bool:
    """Reject missing, truncated, and non-ZIP checkpoint inputs."""

    return path.is_file() and zipfile.is_zipfile(path)


def load_shape_pipeline(
    model_path: str | Path,
    *,
    subfolder: str,
    device: str = "cuda",
    variant: str = "fp16",
) -> tuple[Any, dict[str, Any]]:
    """Load Hunyuan with non-mapped CPU storage before the CUDA transfer."""

    import torch
    import yaml
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.pipelines import instantiate_from_config

    root = Path(model_path).resolve()
    config_path = root / subfolder / "config.yaml"
    checkpoint_path = shape_checkpoint_path(root, subfolder, variant)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not is_torch_zip_checkpoint(checkpoint_path):
        raise RuntimeError(
            f"{checkpoint_path} is missing or is not a valid Torch ZIP checkpoint. "
            "Rerun setup-hunyuan-mv.ps1; safetensors are intentionally unsupported."
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Hunyuan requires CUDA, but torch.cuda.is_available() is false.")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required_config = {"model", "vae", "conditioner", "image_processor", "scheduler"}
    missing_config = sorted(required_config - set(config or {}))
    if missing_config:
        raise RuntimeError(f"Hunyuan config is missing: {', '.join(missing_config)}")

    proof: dict[str, Any] = {
        "ready": False,
        "safe_loader": SAFE_LOADER_ID,
        "checkpoint": str(checkpoint_path),
        "checkpoint_format": "torch_zip",
        "checkpoint_loads": 0,
        "checkpoint_mmap": False,
        "vae_preallocated": False,
        "vae_owned_copy": False,
        "meta_tensor_count": -1,
        "cuda_synchronized": False,
    }

    checkpoint: dict[str, Any] | None = None
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        # Allocate the VAE before the 4.9 GiB checkpoint. Its weights are copied
        # into this owned storage instead of adopting checkpoint-backed tensors.
        vae = instantiate_from_config(config["vae"])
        proof["vae_preallocated"] = True
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=False,
        )
        proof["checkpoint_loads"] = 1
        missing_components = sorted({"model", "vae", "conditioner"} - set(checkpoint))
        if missing_components:
            raise RuntimeError(
                "Hunyuan checkpoint is missing: " + ", ".join(missing_components)
            )

        with torch.device("meta"):
            model = instantiate_from_config(config["model"])
            conditioner = instantiate_from_config(config["conditioner"])
        model.load_state_dict(checkpoint["model"], assign=True)
        conditioner.load_state_dict(checkpoint["conditioner"], assign=True)
        vae.load_state_dict(checkpoint["vae"], strict=False, assign=False)
        proof["vae_owned_copy"] = True
    finally:
        torch.set_default_dtype(previous_dtype)

    image_processor = instantiate_from_config(config["image_processor"])
    scheduler = instantiate_from_config(config["scheduler"])
    pipeline = Hunyuan3DDiTFlowMatchingPipeline(
        vae=vae,
        model=model,
        scheduler=scheduler,
        conditioner=conditioner,
        image_processor=image_processor,
        device="cpu",
        dtype=torch.float16,
        from_pretrained_kwargs={
            "model_path": str(root),
            "subfolder": subfolder,
            "use_safetensors": False,
            "variant": variant,
            "dtype": torch.float16,
            "device": device,
        },
    )

    checkpoint = None
    gc.collect()

    meta_tensors = _meta_tensors(pipeline)
    proof["meta_tensor_count"] = len(meta_tensors)
    if meta_tensors:
        preview = ", ".join(meta_tensors[:8])
        raise RuntimeError(
            "The safe loader left tensors on the meta device: "
            f"{preview}{' ...' if len(meta_tensors) > 8 else ''}"
        )

    pipeline.to(device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        proof["cuda_synchronized"] = True
        proof["cuda_allocated_gib"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
    proof["device"] = str(device)
    proof["ready"] = True
    pipeline._codex_safe_loader_status = proof
    return pipeline, proof


def _meta_tensors(pipeline: Any) -> list[str]:
    values: list[str] = []
    for component_name in ("model", "vae", "conditioner"):
        component = getattr(pipeline, component_name, None)
        if component is None:
            continue
        for tensor_name, tensor in component.named_parameters():
            if tensor.is_meta:
                values.append(f"{component_name}.{tensor_name}")
        for tensor_name, tensor in component.named_buffers():
            if tensor.is_meta:
                values.append(f"{component_name}.{tensor_name}")
    return values
