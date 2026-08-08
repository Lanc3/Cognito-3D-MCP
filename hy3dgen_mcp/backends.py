"""Backend adapters used by the MCP tool surface."""

from __future__ import annotations

import shutil
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .config import Settings

BackendName = Literal["hunyuan3d", "sf3d", "spar3d"]
DEFAULT_BACKEND: BackendName = "hunyuan3d"
SUPPORTED_BACKENDS: tuple[BackendName, ...] = ("hunyuan3d", "sf3d", "spar3d")
VIEW_NAMES = ("front", "left", "back", "right")
_GPU_LOCK = threading.Lock()


@dataclass(frozen=True)
class GenerationRequest:
    front_image: Path
    output_dir: Path | None = None
    backend: BackendName = DEFAULT_BACKEND
    left_image: Path | None = None
    back_image: Path | None = None
    right_image: Path | None = None
    seed: int = 12345
    steps: int = 50
    guidance_scale: float = 5.0
    octree_resolution: int = 384
    texture_resolution: int = 1024


@dataclass(frozen=True)
class GenerationResult:
    backend: BackendName
    output_path: str
    input_views: tuple[str, ...]
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validated_image(path: Path | None, name: str) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{name} image does not exist or is not a file: {resolved}")
    if resolved.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError(f"{name} image must be PNG, JPEG, or WebP: {resolved}")
    return resolved


def validate_request(request: GenerationRequest) -> GenerationRequest:
    if request.backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend {request.backend!r}; choose one of {', '.join(SUPPORTED_BACKENDS)}"
        )
    if not 1 <= request.steps <= 200:
        raise ValueError("steps must be between 1 and 200")
    if not 32 <= request.octree_resolution <= 1024:
        raise ValueError("octree_resolution must be between 32 and 1024")
    if not 256 <= request.texture_resolution <= 4096:
        raise ValueError("texture_resolution must be between 256 and 4096")
    if request.backend != "hunyuan3d" and any(
        (request.left_image, request.back_image, request.right_image)
    ):
        raise ValueError(f"{request.backend} accepts a single front image only")

    return GenerationRequest(
        front_image=_validated_image(request.front_image, "front"),  # type: ignore[arg-type]
        output_dir=request.output_dir.expanduser().resolve() if request.output_dir else None,
        backend=request.backend,
        left_image=_validated_image(request.left_image, "left"),
        back_image=_validated_image(request.back_image, "back"),
        right_image=_validated_image(request.right_image, "right"),
        seed=request.seed,
        steps=request.steps,
        guidance_scale=request.guidance_scale,
        octree_resolution=request.octree_resolution,
        texture_resolution=request.texture_resolution,
    )


def backend_status(settings: Settings) -> list[dict[str, object]]:
    return [
        {
            "name": "hunyuan3d",
            "preferred": True,
            "available": True,
            "detail": f"model={settings.hunyuan_model}, device={settings.hunyuan_device}",
        },
        _cli_status("sf3d", settings.sf3d_root, settings.sf3d_python),
        _cli_status("spar3d", settings.spar3d_root, settings.spar3d_python),
    ]


def _cli_status(name: BackendName, root: Path | None, python: Path) -> dict[str, object]:
    available = root is not None and (root / "run.py").is_file() and python.is_file()
    return {
        "name": name,
        "preferred": False,
        "available": available,
        "detail": f"root={root or 'not configured'}, python={python}",
    }


class BackendRegistry:
    """Routes generation requests while serializing GPU-heavy jobs."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self._hunyuan_pipeline = None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request = validate_request(request)
        output_dir = request.output_dir or (
            self.settings.output_root / f"{request.backend}-{uuid.uuid4().hex[:12]}"
        )
        output_dir.mkdir(parents=True, exist_ok=False)

        with _GPU_LOCK:
            if request.backend == "hunyuan3d":
                output_path = self._run_hunyuan(request, output_dir)
            elif request.backend == "sf3d":
                output_path = self._run_external(
                    request, output_dir, self.settings.sf3d_root, self.settings.sf3d_python
                )
            else:
                output_path = self._run_external(
                    request, output_dir, self.settings.spar3d_root, self.settings.spar3d_python
                )

        views = tuple(
            name
            for name in VIEW_NAMES
            if getattr(request, f"{name}_image", None) is not None
        )
        return GenerationResult(
            backend=request.backend,
            output_path=str(output_path),
            input_views=views,
            message=f"Generated a GLB with {request.backend}",
        )

    def _run_hunyuan(self, request: GenerationRequest, output_dir: Path) -> Path:
        import torch
        from PIL import Image

        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        if self._hunyuan_pipeline is None:
            self._hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.settings.hunyuan_model,
                subfolder=self.settings.hunyuan_subfolder,
                variant=self.settings.hunyuan_variant,
                device=self.settings.hunyuan_device,
            )

        images = {
            name: Image.open(path).convert("RGBA")
            for name in VIEW_NAMES
            if (path := getattr(request, f"{name}_image", None)) is not None
        }
        mesh = self._hunyuan_pipeline(
            image=images,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            octree_resolution=request.octree_resolution,
            generator=torch.manual_seed(request.seed),
            output_type="trimesh",
        )[0]
        output_path = output_dir / "mesh.glb"
        mesh.export(output_path)
        return output_path

    def _run_external(
        self,
        request: GenerationRequest,
        output_dir: Path,
        root: Path | None,
        python: Path,
    ) -> Path:
        if root is None:
            raise RuntimeError(
                f"{request.backend} is not configured; set {request.backend.upper()}_ROOT and "
                f"{request.backend.upper()}_PYTHON"
            )
        script = root / "run.py"
        if not script.is_file() or not python.is_file():
            raise RuntimeError(f"Invalid {request.backend} installation: {script}, {python}")
        command = [
            str(python),
            str(script),
            str(request.front_image),
            "--output-dir",
            str(output_dir),
            "--texture-resolution",
            str(request.texture_resolution),
        ]
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=self.settings.job_timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(f"{request.backend} failed ({completed.returncode}): {detail}")

        generated = next(output_dir.rglob("*.glb"), None)
        if generated is None:
            raise RuntimeError(f"{request.backend} completed without producing a GLB")
        final_path = output_dir / "mesh.glb"
        if generated != final_path:
            shutil.move(str(generated), final_path)
        return final_path
