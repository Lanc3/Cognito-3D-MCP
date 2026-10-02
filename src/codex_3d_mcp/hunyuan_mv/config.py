"""Configuration for the isolated Hunyuan3D-2mv server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .. import __version__
from ..config import blender_path, data_root


def _path(value: str | Path, base: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _integer(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _boolean(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class HunyuanMVSettings:
    """Absolute, environment-backed multiview runtime settings."""

    base_dir: Path
    server_name: str
    server_version: str
    output_dir: Path
    database_path: Path
    python_exe: Path
    upstream_dir: Path
    model_cache_dir: Path
    blender_exe: Path
    gltf_validator: Path
    allowed_input_roots: tuple[Path, ...]
    gpu_lock_path: Path
    shape_model_id: str = "tencent/Hunyuan3D-2mv"
    shape_subfolder: str = "hunyuan3d-dit-v2-mv"
    texture_model_id: str = "tencent/Hunyuan3D-2"
    texture_subfolder: str = "hunyuan3d-paint-v2-0"
    shape_revision: str = "3a761b539b29fe4ff64714813aa9560fd66f5de0"
    texture_revision: str = "9cd649ba6913f7a852e3286bad86bfa9a2d83dcf"
    upstream_commit: str = "f8db63096c8282cb27354314d896feba5ba6ff8a"
    default_seed: int = 42
    default_quality: str = "standard"
    texture_resolution: int = 2048
    game_faces: int = 100_000
    lod_faces: tuple[int, int] = (50_000, 20_000)
    minimum_image_size: int = 512
    minimum_free_bytes: int = 20 * 1024**3
    install_free_bytes: int = 45 * 1024**3
    request_timeout_seconds: int = 2_700
    max_queued_jobs: int = 4
    low_vram_mode: bool = False
    test_mode: bool = False
    autoremesher_exe: Path = Path("autoremesher.exe")
    remesh_target_quads: int = 100_000
    remesh_timeout_seconds: int = 1800
    remesh_threads: int = 2
    dashboard_auto_open: bool = True
    repair_package_dir: Path = Path("shape-repair-packages")
    repair_timeout_seconds: int = 300

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> HunyuanMVSettings:
        base = (base_dir or Path.cwd()).resolve()
        root = data_root(base)
        output = _path(os.getenv("CODEX_HUNYUAN_MV_OUTPUT_DIR", "outputs/hunyuan-mv"), base)
        runtime_root = _path(
            os.getenv("CODEX_HUNYUAN_MV_RUNTIME_ROOT", str(root / "runtime" / "hunyuan3d-2mv")),
            base,
        )
        cache = _path(
            os.getenv("CODEX_HUNYUAN_MV_MODEL_CACHE", str(root / "models" / "hunyuan3d-2mv")),
            base,
        )
        roots_value = os.getenv(
            "CODEX_HUNYUAN_MV_ALLOWED_INPUT_ROOTS",
            os.pathsep.join(
                [str(base / "inputs"), str(Path.home() / ".codex" / "generated_images")]
            ),
        )
        roots = tuple(
            _path(item.strip(), base) for item in roots_value.split(os.pathsep) if item.strip()
        )
        blender_default = blender_path(root)
        return cls(
            base_dir=base,
            server_name=os.getenv("CODEX_HUNYUAN_MV_SERVER_NAME", "Cognito-3D-mcp"),
            server_version=__version__,
            output_dir=output,
            database_path=_path(
                os.getenv("CODEX_HUNYUAN_MV_DATABASE", str(output / "jobs.sqlite3")), base
            ),
            python_exe=_path(
                os.getenv(
                    "CODEX_HUNYUAN_MV_PYTHON",
                    str(runtime_root / "venv" / "Scripts" / "python.exe"),
                ),
                base,
            ),
            upstream_dir=_path(
                os.getenv("CODEX_HUNYUAN_MV_UPSTREAM", str(runtime_root / "Hunyuan3D-2")),
                base,
            ),
            model_cache_dir=cache,
            blender_exe=_path(os.getenv("CODEX_BLENDER_EXE", str(blender_default)), base),
            gltf_validator=_path(
                os.getenv(
                    "CODEX_GLTF_VALIDATOR",
                    str(root / "runtime" / "gltf-validator" / "gltf_validator.exe"),
                ),
                base,
            ),
            allowed_input_roots=roots,
            gpu_lock_path=_path(
                os.getenv("CODEX_3D_GPU_LOCK", str(root / "runtime" / "gpu.lock")), base
            ),
            shape_model_id=os.getenv("CODEX_HUNYUAN_MV_SHAPE_MODEL", cls.shape_model_id),
            shape_subfolder=os.getenv("CODEX_HUNYUAN_MV_SHAPE_SUBFOLDER", cls.shape_subfolder),
            texture_model_id=os.getenv("CODEX_HUNYUAN_MV_TEXTURE_MODEL", cls.texture_model_id),
            texture_subfolder=os.getenv(
                "CODEX_HUNYUAN_MV_TEXTURE_SUBFOLDER", cls.texture_subfolder
            ),
            shape_revision=os.getenv("CODEX_HUNYUAN_MV_SHAPE_REVISION", cls.shape_revision),
            texture_revision=os.getenv("CODEX_HUNYUAN_MV_TEXTURE_REVISION", cls.texture_revision),
            upstream_commit=os.getenv("CODEX_HUNYUAN_MV_UPSTREAM_COMMIT", cls.upstream_commit),
            texture_resolution=max(
                1024, min(4096, _integer("CODEX_HUNYUAN_MV_TEXTURE_RESOLUTION", 2048))
            ),
            request_timeout_seconds=max(300, _integer("CODEX_HUNYUAN_MV_TIMEOUT_SECONDS", 2_700)),
            max_queued_jobs=max(1, _integer("CODEX_HUNYUAN_MV_MAX_QUEUED_JOBS", 4)),
            low_vram_mode=_boolean("CODEX_HUNYUAN_MV_LOW_VRAM", False),
            test_mode=_boolean("CODEX_HUNYUAN_MV_TEST_MODE"),
            autoremesher_exe=_path(
                os.getenv(
                    "CODEX_AUTOREMESHER_EXE",
                    str(root / "runtime" / "autoremesher" / "autoremesher.exe"),
                ),
                base,
            ),
            remesh_target_quads=max(1000, _integer("CODEX_AUTOREMESHER_TARGET_QUADS", 100_000)),
            remesh_timeout_seconds=max(60, _integer("CODEX_AUTOREMESHER_TIMEOUT", 1800)),
            remesh_threads=max(1, min(2, _integer("CODEX_AUTOREMESHER_THREADS", 2))),
            dashboard_auto_open=_boolean("CODEX_HUNYUAN_MV_DASHBOARD_AUTO_OPEN", True),
            repair_package_dir=_path(
                os.getenv(
                    "CODEX_HUNYUAN_MV_REPAIR_PACKAGES",
                    str(root / "runtime" / "shape-repair-packages"),
                ),
                base,
            ),
            repair_timeout_seconds=max(30, min(900, _integer("CODEX_HUNYUAN_MV_REPAIR_TIMEOUT", 300))),
        )

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)

    def is_allowed_input(self, path: Path) -> bool:
        candidate = path.resolve()
        return any(_is_relative_to(candidate, root.resolve()) for root in self.allowed_input_roots)


def _is_relative_to(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False
