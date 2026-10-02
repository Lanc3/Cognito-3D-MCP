"""Environment-backed configuration for the local MCP server."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import __version__


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def data_root(base_dir: Path) -> Path:
    """Use a per-user data directory unless the operator specifies another root."""
    default = Path(os.getenv("LOCALAPPDATA", str(Path.home() / ".cache"))) / "Cognito-3D-mcp"
    return _resolve_path(os.getenv("CODEX_3D_DATA_ROOT", str(default)), base_dir)


def blender_path(root: Path) -> Path:
    """Discover Blender installed on PATH or by the standard Windows installer."""
    executable = shutil.which("blender")
    if executable:
        return Path(executable).resolve()
    program_files = Path(os.getenv("ProgramFiles", "C:/Program Files"))
    candidates = list((program_files / "Blender Foundation").glob("Blender */blender.exe"))
    if candidates:
        def version(path: Path) -> tuple[int, ...]:
            parts = path.parent.name.removeprefix("Blender ").split(".")
            return tuple(int(part) for part in parts if part.isdigit())

        return max(candidates, key=version).resolve()
    return root / "runtime" / "blender" / ("blender.exe" if os.name == "nt" else "blender")


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Paths are absolute after construction."""

    server_name: str = "Cognito-3D-mcp-single-view"
    server_version: str = __version__
    backend: str = "sf3d"
    model_id: str = "stabilityai/stable-fast-3d"
    low_vram_mode: bool = False
    device: str = "auto"
    model_cache_dir: Path = Path(".cache/huggingface")
    output_dir: Path = Path("outputs")
    allowed_input_roots: tuple[Path, ...] = (Path("."),)
    max_image_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 32 * 1024 * 1024
    job_timeout_seconds: int = 1800
    poll_seconds: float = 2.0
    max_queued_jobs: int = 8
    gpu_lock_path: Path = Path(".cache/gpu.lock")
    gpu_wait_seconds: float = 30.0
    default_texture_resolution: int = 1024
    default_foreground_ratio: float = 0.85
    default_remesh: str = "none"

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> Settings:
        base = (base_dir or Path.cwd()).resolve()
        backend = os.getenv("CODEX_3D_BACKEND", cls.backend).strip().lower()
        default_model_id = {
            "sf3d": "stabilityai/stable-fast-3d",
            "spar3d": "stabilityai/stable-point-aware-3d",
        }.get(backend, cls.model_id)
        roots_value = os.getenv("CODEX_3D_ALLOWED_INPUT_ROOTS", ".")
        roots = tuple(
            _resolve_path(item.strip(), base)
            for item in roots_value.split(os.pathsep)
            if item.strip()
        ) or (base,)
        poll_value = os.getenv("CODEX_3D_POLL_SECONDS", "2")
        try:
            poll_seconds = max(0.1, float(poll_value))
        except ValueError:
            poll_seconds = 2.0
        return cls(
            server_name=os.getenv("CODEX_3D_SERVER_NAME", cls.server_name),
            backend=backend,
            model_id=os.getenv("CODEX_3D_MODEL_ID", default_model_id),
            low_vram_mode=_env_bool("CODEX_3D_LOW_VRAM_MODE", False),
            device=os.getenv("CODEX_3D_DEVICE", cls.device).lower(),
            model_cache_dir=_resolve_path(
                os.getenv("CODEX_3D_MODEL_CACHE_DIR", str(cls.model_cache_dir)), base
            ),
            output_dir=_resolve_path(os.getenv("CODEX_3D_OUTPUT_DIR", str(cls.output_dir)), base),
            allowed_input_roots=roots,
            max_image_bytes=max(1, _env_int("CODEX_3D_MAX_IMAGE_BYTES", cls.max_image_bytes)),
            max_image_pixels=max(1, _env_int("CODEX_3D_MAX_IMAGE_PIXELS", cls.max_image_pixels)),
            job_timeout_seconds=max(
                1, _env_int("CODEX_3D_JOB_TIMEOUT_SECONDS", cls.job_timeout_seconds)
            ),
            poll_seconds=poll_seconds,
            max_queued_jobs=max(1, _env_int("CODEX_3D_MAX_QUEUED_JOBS", cls.max_queued_jobs)),
            gpu_lock_path=_resolve_path(
                os.getenv("CODEX_3D_GPU_LOCK", str(data_root(base) / "runtime" / "gpu.lock")), base
            ),
            gpu_wait_seconds=max(1, min(600, _env_int("CODEX_3D_GPU_WAIT_SECONDS", 30))),
        )

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

    def is_allowed_input(self, path: Path) -> bool:
        candidate = path.resolve()
        for root in self.allowed_input_roots:
            try:
                candidate.relative_to(root.resolve())
                return True
            except ValueError:
                continue
        return False
