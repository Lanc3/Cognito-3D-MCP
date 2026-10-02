"""MCP entrypoint for Codex 3D generation."""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from .config import Settings
from .inputs import cleanup_staged_image, stage_image_input
from .jobs import JobManager
from .model import ImageTo3DAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("codex-3d-mcp")

load_dotenv(Path.cwd() / ".env", override=False)
settings = Settings.from_env()
settings.ensure_directories()
adapter = ImageTo3DAdapter(settings)
jobs = JobManager(settings, adapter)

mcp = FastMCP(
    settings.server_name,
    instructions=(
        "Use generate_3d_model with the user-provided prompt and exactly one image input. "
        "It returns a job_id; poll get_generation_status until completed, then use the "
        "absolute GLB artifact_path. "
        "Generation is local, single-worker, textured GLB only, and may take several minutes."
        f" The selected backend is {settings.backend}."
    ),
)


@mcp.tool()
def generate_3d_model(
    prompt: str,
    image_data_url: str | None = None,
    image_base64: str | None = None,
    image_path: str | None = None,
    texture_resolution: int = 1024,
    foreground_ratio: float = 0.85,
    remesh: Literal["none", "triangle", "quad"] = "none",
    target_vertex_count: int | None = None,
) -> dict:
    """Start a local textured GLB generation job with the selected backend."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if len(prompt) > 10_000:
        raise ValueError("prompt must be 10,000 characters or fewer")
    if texture_resolution not in {512, 1024, 2048, 4096}:
        raise ValueError("texture_resolution must be one of 512, 1024, 2048, or 4096")
    if not 0.5 <= foreground_ratio <= 1.0:
        raise ValueError("foreground_ratio must be between 0.5 and 1.0")
    if target_vertex_count is not None and not 100 <= target_vertex_count <= 10_000_000:
        raise ValueError("target_vertex_count must be between 100 and 10,000,000")

    staged = stage_image_input(
        settings,
        image_data_url=image_data_url,
        image_base64=image_base64,
        image_path=image_path,
    )
    params = {
        "backend": settings.backend,
        "texture_resolution": texture_resolution,
        "foreground_ratio": foreground_ratio,
        "remesh": remesh,
        "target_vertex_count": target_vertex_count,
    }
    try:
        record = jobs.submit(prompt.strip(), staged, params)
    except Exception:
        cleanup_staged_image(staged)
        raise
    return {
        "job_id": record.job_id,
        "status": record.status,
        "params": params,
        "stages": ["loading model", "preparing image", "generating textured mesh", "exporting GLB"],
        "artifact_format": "glb",
    }


@mcp.tool()
def get_generation_status(job_id: str) -> dict:
    """Return current state, progress, errors, or the completed GLB path for a job."""
    return jobs.get(job_id).as_dict()


@mcp.tool()
def cancel_generation(job_id: str) -> dict:
    """Request cancellation of a queued or running generation job."""
    return jobs.cancel(job_id)


@mcp.tool()
def server_status() -> dict:
    """Report runtime, dependency, model-cache, device, and queue readiness."""
    return {
        "server_name": settings.server_name,
        "server_version": settings.server_version,
        "runtime": {"python": __import__("sys").version.split()[0]},
        "model": adapter.status(),
        "queue": jobs.summary(),
        "gpu_arbitration": {
            "lock_path": str(settings.gpu_lock_path),
            "wait_timeout_seconds": settings.gpu_wait_seconds,
            "model_unloaded_between_jobs": True,
            "resource_error": jobs._resource_error,
        },
        "output_dir": str(settings.output_dir),
        "allowed_input_roots": [str(path) for path in settings.allowed_input_roots],
    }


def main() -> None:
    logger.info("Starting %s over STDIO", settings.server_name)
    mcp.run(transport="stdio")


atexit.register(jobs.shutdown)


if __name__ == "__main__":
    main()
