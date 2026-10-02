"""One-shot worker loaded by the isolated Hunyuan virtual environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_SHAPE_CACHE: dict[tuple[str, str], Any] = {}
_PAINT_CACHE: dict[tuple[str, str, bool], Any] = {}


def _has_flat_magenta_backdrop(image: Any) -> bool:
    rgba = image.convert("RGBA")
    width, height = rgba.size
    corners = (
        rgba.getpixel((0, 0)),
        rgba.getpixel((width - 1, 0)),
        rgba.getpixel((0, height - 1)),
        rgba.getpixel((width - 1, height - 1)),
    )
    return all(red >= 245 and green <= 12 and blue >= 245 for red, green, blue, _ in corners)


def _remove_flat_magenta_backdrop(image: Any) -> Any:
    import numpy as np
    from PIL import Image

    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32).copy()
    distance = np.sqrt(
        np.square(255.0 - rgba[..., 0]) + np.square(rgba[..., 1]) + np.square(255.0 - rgba[..., 2])
    )
    coverage = np.clip((distance - 12.0) / 68.0, 0.0, 1.0)
    coverage = np.minimum(coverage, rgba[..., 3] / 255.0)
    recoverable = coverage > 0.05
    background = np.zeros_like(rgba[..., :3])
    background[..., 0] = 255.0
    background[..., 2] = 255.0
    foreground = np.zeros_like(rgba[..., :3])
    foreground[recoverable] = np.clip(
        (rgba[..., :3][recoverable] - (1.0 - coverage[recoverable, None]) * background[recoverable])
        / coverage[recoverable, None],
        0.0,
        255.0,
    )
    rgba[..., :3] = foreground
    rgba[..., 3] = coverage * 255.0
    return Image.fromarray(np.uint8(np.rint(rgba)), "RGBA")


def _activate_upstream() -> None:
    upstream = os.getenv("HUNYUAN3D_UPSTREAM")
    if upstream and upstream not in sys.path:
        sys.path.insert(0, upstream)


def _shape(request: dict[str, Any]) -> dict[str, Any]:
    import torch

    _activate_upstream()
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    from PIL import Image

    from .safe_loader import SAFE_LOADER_ID, load_shape_pipeline

    if request.get("shape_loader_id") != SAFE_LOADER_ID:
        raise RuntimeError("Shape request does not declare the required safe-loader contract.")

    started = time.monotonic()
    views = {name: Image.open(path).convert("RGBA") for name, path in request["views"].items()}
    # Prepared cutouts are shared by shape and Paint. Do not silently run a
    # different background-removal path inside either GPU stage.
    for name, image in views.items():
        alpha = image.getchannel("A")
        if alpha.getextrema()[0] == 255 or alpha.getbbox() is None:
            raise ValueError(f"{name}: prepared reference has no usable transparent background")
    cache_key = (request["model_path"], request["subfolder"])
    if cache_key not in _SHAPE_CACHE:
        _SHAPE_CACHE[cache_key] = load_shape_pipeline(
            request["model_path"], subfolder=request["subfolder"], device="cuda"
        )
    pipeline, runtime_proof = _SHAPE_CACHE[cache_key]
    generator = torch.Generator().manual_seed(int(request["seed"]))
    output = pipeline(
        image=views,
        num_inference_steps=int(request["steps"]),
        guidance_scale=float(request["guidance_scale"]),
        generator=generator,
        octree_resolution=int(request["octree_resolution"]),
        num_chunks=int(request["num_chunks"]),
        output_type="mesh",
    )
    mesh = export_to_trimesh(output)[0]
    destination = Path(request["output"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(destination, include_normals=True)
    return {
        "action": "shape",
        "views": list(views),
        "faces": int(mesh.faces.shape[0]),
        "vertices": int(mesh.vertices.shape[0]),
        "seconds": time.monotonic() - started,
        "runtime": runtime_proof,
    }


def _texture(request: dict[str, Any]) -> dict[str, Any]:
    _activate_upstream()
    import trimesh

    from .paint_loader import load_paint_pipeline

    started = time.monotonic()
    mesh = trimesh.load(request["mesh"], force="mesh")
    # Hunyuan's custom Paint loader is not a Diffusers loader and does not
    # accept Hugging Face's ``revision`` keyword. Runtime resolution pins the
    # snapshot first, then this worker loads that exact local directory.
    from PIL import Image

    with Image.open(request["image"]) as reference:
        if reference.mode != "RGBA" or reference.getchannel("A").getextrema()[0] == 255:
            raise ValueError("Paint requires the same gated transparent reference as shape")
    cache_key = (request["model_path"], request["subfolder"], request.get("low_vram_mode", False))
    if cache_key[2]:
        raise RuntimeError("Paint requires GPU inference; CPU offload is disabled")
    if cache_key not in _PAINT_CACHE:
        _PAINT_CACHE[cache_key] = load_paint_pipeline(
            request["model_path"], subfolder=request["subfolder"]
        )
    pipeline, runtime_proof = _PAINT_CACHE[cache_key]
    # The upstream atlas/render setters must both be used; metadata alone did
    # not previously change the texture dimensions.
    resolution = int(request.get("texture_resolution", 2048))
    pipeline.config.texture_size = resolution
    pipeline.config.render_size = resolution
    pipeline.render.set_default_texture_resolution(resolution)
    pipeline.render.set_default_render_resolution(resolution)
    textured = pipeline(mesh, image=request["image"])
    destination = Path(request["output"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    textured.export(destination, include_normals=True)
    return {
        "action": "texture",
        "faces": int(textured.faces.shape[0]),
        "vertices": int(textured.vertices.shape[0]),
        "conditioning_view": "front",
        "texture_resolution": resolution,
        "seconds": time.monotonic() - started,
        "runtime": runtime_proof,
    }


def _execute(request: dict[str, Any]) -> dict[str, Any]:
    action = request.get("action")
    if action == "shape":
        return _shape(request)
    if action == "texture":
        return _texture(request)
    raise ValueError(f"Unknown worker action: {action!r}")


def _write_result(request: dict[str, Any], result: dict[str, Any]) -> None:
    from ..trellis.artifacts import atomic_json

    atomic_json(Path(request["result"]), result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--serve-stage", choices=("shape", "texture"))
    args = parser.parse_args()
    if args.serve_stage:
        # Requests travel over stdin; stdout/stderr belong to the stage log.
        # Atomic response files prevent library progress output corrupting IPC.
        for line in sys.stdin:
            request = json.loads(Path(json.loads(line)["request"]).read_text(encoding="utf-8"))
            try:
                if request.get("action") != args.serve_stage:
                    raise ValueError("Resident worker cannot switch model stages")
                _write_result(request, {"ok": True, "result": _execute(request)})
            except Exception as exc:
                import traceback

                traceback.print_exc()
                _write_result(request, {"ok": False, "error": str(exc)})
                # A failed CUDA call can poison process state. Reload safely on retry.
                return
        return
    if args.request is None:
        parser.error("--request or --serve-stage is required")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    _write_result(request, _execute(request))


if __name__ == "__main__":
    main()
