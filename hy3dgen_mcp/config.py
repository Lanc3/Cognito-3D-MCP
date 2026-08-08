"""Environment-backed configuration for the MCP server."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _optional_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


@dataclass(frozen=True)
class Settings:
    """Runtime paths and model settings.

    SF3D and SPAR3D are intentionally run in separate Python environments. Their
    native dependencies can otherwise conflict with Hunyuan3D's environment.
    """

    output_root: Path
    hunyuan_model: str
    hunyuan_subfolder: str
    hunyuan_device: str
    hunyuan_variant: str
    sf3d_root: Path | None
    sf3d_python: Path
    spar3d_root: Path | None
    spar3d_python: Path
    job_timeout_seconds: int

    @classmethod
    def from_env(cls) -> Settings:
        output_root = Path(os.environ.get("HY3D_MCP_OUTPUT_ROOT", "outputs/mcp"))
        return cls(
            output_root=output_root.expanduser().resolve(),
            hunyuan_model=os.environ.get("HUNYUAN3D_MODEL_PATH", "tencent/Hunyuan3D-2mv"),
            hunyuan_subfolder=os.environ.get(
                "HUNYUAN3D_SUBFOLDER", "hunyuan3d-dit-v2-mv"
            ),
            hunyuan_device=os.environ.get("HUNYUAN3D_DEVICE", "cuda"),
            hunyuan_variant=os.environ.get("HUNYUAN3D_VARIANT", "fp16"),
            sf3d_root=_optional_path("SF3D_ROOT"),
            sf3d_python=Path(os.environ.get("SF3D_PYTHON", sys.executable)).resolve(),
            spar3d_root=_optional_path("SPAR3D_ROOT"),
            spar3d_python=Path(os.environ.get("SPAR3D_PYTHON", sys.executable)).resolve(),
            job_timeout_seconds=int(os.environ.get("HY3D_MCP_JOB_TIMEOUT", "1800")),
        )
