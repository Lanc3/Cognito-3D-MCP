"""Configuration for the isolated bidirectional production server."""

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


def _floating(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _boolean(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TrellisSettings:
    """Absolute, environment-backed production settings."""

    base_dir: Path
    server_name: str
    server_version: str
    output_dir: Path
    database_path: Path
    runtime_dir: Path
    model_dir: Path
    trellis_server: Path
    trellis_cli: Path
    realesrgan_exe: Path
    blender_exe: Path
    gltf_validator: Path
    allowed_input_roots: tuple[Path, ...]
    gpu_lock_path: Path
    reconstruction_backend: str = "spar3d"
    spar3d_python: Path = Path(".venv-spar3d/Scripts/python.exe")
    spar3d_model_cache_dir: Path = Path(".cache/huggingface-spar3d")
    spar3d_model_id: str = "stabilityai/stable-point-aware-3d"
    spar3d_low_vram_mode: bool = True
    host: str = "127.0.0.1"
    port: int = 18082
    resolution: int = 1024
    atlas_resolution: int = 2048
    texture_resolution: int = 4096
    master_faces: int = 300_000
    game_faces: int = 100_000
    lod_faces: tuple[int, ...] = (50_000, 20_000)
    default_seed: int = 42
    minimum_image_size: int = 1024
    minimum_free_bytes: int = 20 * 1024**3
    max_job_bytes: int = 15 * 1024**3
    request_timeout_seconds: int = 1_800
    max_queued_jobs: int = 8
    pair_semantic_threshold: float = 0.78
    test_mode: bool = False

    runtime_version: str = "v0.5.4"
    runtime_commit: str = "ae1a63757264eec5bfba84b94cf59ddcc161e537"
    runtime_archive_sha256: str = (
        "f7d2912b064bf1520f03e025c5eb344df6b347ad04831ac7aa04d847581bd7ad"
    )
    weights_revision: str = "a57397bd3d351599d9729fc144b3f87c3f87d65b"

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> TrellisSettings:
        base = (base_dir or Path.cwd()).resolve()
        root = data_root(base)
        reconstruction_backend = os.getenv(
            "CODEX_BIDIRECTIONAL_RECONSTRUCTION_BACKEND", cls.reconstruction_backend
        ).strip().lower()
        if reconstruction_backend not in {"spar3d", "trellis"}:
            reconstruction_backend = cls.reconstruction_backend
        runtime = _path(
            os.getenv("CODEX_TRELLIS_RUNTIME_DIR", str(root / "runtime" / "trellis-v0.5.4")),
            base,
        )
        models = _path(
            os.getenv("CODEX_TRELLIS_MODEL_DIR", str(root / "models" / "trellis2-q8")), base
        )
        output = _path(os.getenv("CODEX_TRELLIS_OUTPUT_DIR", "outputs/trellis"), base)
        roots_value = os.getenv(
            "CODEX_TRELLIS_ALLOWED_INPUT_ROOTS",
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
            server_name=os.getenv("CODEX_TRELLIS_SERVER_NAME", "Cognito-3D-mcp-bidirectional"),
            server_version=__version__,
            output_dir=output,
            database_path=_path(
                os.getenv("CODEX_TRELLIS_DATABASE", str(output / "jobs.sqlite3")), base
            ),
            runtime_dir=runtime,
            model_dir=models,
            trellis_server=_path(
                os.getenv("CODEX_TRELLIS_SERVER_EXE", str(runtime / "trellis-server.exe")), base
            ),
            trellis_cli=_path(
                os.getenv("CODEX_TRELLIS_CLI_EXE", str(runtime / "trellis-cli.exe")), base
            ),
            realesrgan_exe=_path(
                os.getenv(
                    "CODEX_REALESRGAN_EXE",
                    str(root / "runtime" / "realesrgan-v0.2.5.0" / "realesrgan-ncnn-vulkan.exe"),
                ),
                base,
            ),
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
            reconstruction_backend=reconstruction_backend,
            spar3d_python=_path(
                os.getenv(
                    "CODEX_BIDIRECTIONAL_SPAR3D_PYTHON",
                    str(base / ".venv-spar3d" / "Scripts" / "python.exe"),
                ),
                base,
            ),
            spar3d_model_cache_dir=_path(
                os.getenv(
                    "CODEX_BIDIRECTIONAL_SPAR3D_MODEL_CACHE",
                    str(base / ".cache" / "huggingface-spar3d"),
                ),
                base,
            ),
            spar3d_model_id=os.getenv(
                "CODEX_BIDIRECTIONAL_SPAR3D_MODEL_ID", cls.spar3d_model_id
            ),
            spar3d_low_vram_mode=_boolean(
                "CODEX_BIDIRECTIONAL_SPAR3D_LOW_VRAM_MODE", True
            ),
            port=max(1024, min(65535, _integer("CODEX_TRELLIS_PORT", 18082))),
            request_timeout_seconds=max(
                60, _integer("CODEX_TRELLIS_TIMEOUT_SECONDS", 1_800)
            ),
            max_queued_jobs=max(1, _integer("CODEX_TRELLIS_MAX_QUEUED_JOBS", 8)),
            pair_semantic_threshold=max(
                0.0, min(1.0, _floating("CODEX_TRELLIS_SEMANTIC_THRESHOLD", 0.78))
            ),
            test_mode=_boolean("CODEX_TRELLIS_TEST_MODE"),
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
