"""MCP entrypoint for production bidirectional 3D generation."""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from .config import TrellisSettings
from .pipeline import BidirectionalJobManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("codex-3d-bidirectional-mcp")

load_dotenv(Path.cwd() / ".env", override=False)
settings = TrellisSettings.from_env()
settings.ensure_directories()
jobs = BidirectionalJobManager(settings)

mcp = FastMCP(
    settings.server_name,
    instructions=(
        "Generate production GLBs from a matched front/back PNG pair. Call server_status first. "
        "Use generate_bidirectional_3d_model, poll get_generation_status, and respond to "
        "awaiting_review or awaiting_side_inputs. A completed job exposes master, game, and LOD "
        "GLBs plus QA evidence. Failed jobs preserve diagnostics and never publish a fallback."
    ),
)


@mcp.tool()
def generate_bidirectional_3d_model(
    prompt: str,
    front_image_path: str,
    back_image_path: str,
    material_hints: str = "",
    seed: int = 42,
) -> dict:
    """Start a durable front/back production reconstruction job."""
    prompt = prompt.strip()
    material_hints = material_hints.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    if len(prompt) > 10_000:
        raise ValueError("prompt must be 10,000 characters or fewer")
    if len(material_hints) > 4_000:
        raise ValueError("material_hints must be 4,000 characters or fewer")
    if not 1 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 1 and 4294967295")
    return jobs.submit(prompt, front_image_path, back_image_path, material_hints, seed)


@mcp.tool()
def review_seed_pair(
    job_id: str,
    decision: Literal["approve", "reject"],
    notes: str = "",
) -> dict:
    """Approve a warning-only pair or reject it without deleting evidence."""
    return jobs.review(job_id, decision, notes.strip())


@mcp.tool()
def add_side_references(job_id: str, left_image_path: str, right_image_path: str) -> dict:
    """Add approved left/right PNG references to a job awaiting side inputs."""
    return jobs.add_sides(job_id, left_image_path, right_image_path)


@mcp.tool()
def get_generation_status(job_id: str) -> dict:
    """Return durable state, stage, metrics, required action, and available artifacts."""
    return jobs.get(job_id)


@mcp.tool()
def get_generation_artifacts(job_id: str) -> dict:
    """Return paths and SHA-256 hashes for all currently available job artifacts."""
    return jobs.artifacts(job_id)


@mcp.tool()
def resume_generation(job_id: str) -> dict:
    """Resume an interrupted, failed, or explicitly repairable quality-failed job."""
    return jobs.resume(job_id)


@mcp.tool()
def cancel_generation(job_id: str) -> dict:
    """Cancel queued or running work while retaining completed checkpoints."""
    return jobs.cancel(job_id)


@mcp.tool()
def cleanup_generation_job(job_id: str, keep_final: bool = True) -> dict:
    """Explicitly delete job intermediates; failed jobs are never cleaned automatically."""
    return jobs.cleanup(job_id, keep_final)


@mcp.tool()
def server_status() -> dict:
    """Report reconstruction runtime, local resources, dependencies, and readiness."""
    preflight = jobs.runtime.preflight()
    runtime_verified, runtime_error = jobs.runtime.verify_runtime_hash()
    semantic_ready, semantic_error = jobs.semantic.available()
    blender_version = _command_version(settings.blender_exe, ["--version"])
    gpu = _command_version(
        Path("nvidia-smi"),
        ["--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
    )
    ready = bool(preflight["ready_for_generation"] and runtime_verified and semantic_ready)
    return {
        "server_name": settings.server_name,
        "server_version": settings.server_version,
        "reconstruction_backend": settings.reconstruction_backend,
        "ready_for_generation": ready,
        "runtime": preflight,
        "runtime_verified": runtime_verified,
        "runtime_verification_error": runtime_error,
        "semantic_validation_ready": semantic_ready,
        "semantic_validation_error": semantic_error,
        "blender_version": blender_version,
        "gpu": gpu,
        "queue": jobs.summary(),
        "allowed_input_roots": [str(path) for path in settings.allowed_input_roots],
        "output_dir": str(settings.output_dir),
        "production_defaults": {
            "seed": settings.default_seed,
            "resolution": settings.resolution,
            "texture_resolution": settings.texture_resolution,
            "master_faces": settings.master_faces,
            "game_faces": settings.game_faces,
            "lod_faces": list(settings.lod_faces),
        },
    }


def _command_version(executable: Path, arguments: list[str]) -> dict:
    try:
        result = subprocess.run(
            [str(executable), *arguments],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "detail": str(exc)}
    return {
        "available": result.returncode == 0,
        "detail": (result.stdout or result.stderr).strip()[:1000],
    }


def main() -> None:
    logger.info("Starting %s", settings.server_name)
    mcp.run(transport="stdio")


atexit.register(jobs.close)


if __name__ == "__main__":
    main()
