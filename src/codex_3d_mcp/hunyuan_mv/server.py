"""MCP entrypoint for coherent 1-4 view Hunyuan3D-2mv generation."""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from .batch import HunyuanBatchManager
from .config import HunyuanMVSettings
from .dashboard import QueueDashboard
from .pipeline import HunyuanMVJobManager
from .stage_processor import BatchStageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("codex-3d-hunyuan-mv-mcp")

load_dotenv(Path.cwd() / ".env", override=False)
settings = HunyuanMVSettings.from_env()
settings.ensure_directories()
jobs = HunyuanMVJobManager(settings)
batches = HunyuanBatchManager(settings, BatchStageProcessor(settings, jobs))
batches.dashboard = QueueDashboard(batches)

mcp = FastMCP(
    settings.server_name,
    instructions=(
        "Generate stage-serial Hunyuan asset batches. The calling Codex agent owns the complete "
        "repair loop: create_generation_batch, get_agent_work, generate or repair references, "
        "submit_asset_references, inspect evidence, create_shape_repair_attempt, "
        "record_asset_review, retry_asset_stage. "
        "Continue until every asset passes each machine gate AND agent visual review. Never waive "
        "a gate, silently skip an asset, or stop because an attempt count was reached. A reference "
        "set requires four consistent front/back/left/right cutouts. Create ALL reference images "
        "and pass ALL reference gates before GPU shape starts. Then ALL raw shapes, "
        "ALL shape diagnosis/repair and accepted-master reviews, ALL remeshes, "
        "ALL Paint, ALL finishing, with a strict phase barrier and one asset at a time. Hunyuan "
        "and Paint use GPU; AutoRemesher is CPU-only with enforced limits. Do not run unrelated "
        "heavy tools or image work for the next batch during compute. Use pause/cancel on request. "
        "When a gate fails, inspect its evidence, change inputs/settings or repair the cause, "
        "and submit another attempt; successful assets remain checkpointed. Runtime failures "
        "stay visible for repair and must never cause an automatic CPU fallback."
        " Raw generation is not production approval. A repair candidate needs all seven views, "
        "its report, unchanged source/output hashes, current machine gates and agent approval. "
        "Choose a typed recipe from measured defects; never repeat an unchanged quality failure."
    ),
)


@mcp.tool()
def generate_multiview_3d_model(
    prompt: str,
    front_image_path: str,
    back_image_path: str = "",
    left_image_path: str = "",
    right_image_path: str = "",
    material_hints: str = "",
    seed: int = 42,
    quality: Literal["draft", "standard", "high"] = "standard",
    generate_texture: bool = True,
) -> dict:
    """Start a durable Hunyuan3D-2mv job from named canonical views."""
    prompt = prompt.strip()
    material_hints = material_hints.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    if len(prompt) > 10_000 or len(material_hints) > 4_000:
        raise ValueError("prompt or material_hints is too long")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 4294967295")
    if not generate_texture:
        raise ValueError(
            "The gated batch pathway produces textured assets; shape-only batches are unsupported"
        )
    views = {
        "front": front_image_path,
        "back": back_image_path,
        "left": left_image_path,
        "right": right_image_path,
    }
    result = batches.create_batch(
        prompt[:80],
        [
            {
                "name": prompt[:80],
                "prompt": prompt,
                "material_hints": material_hints,
                "seed": seed,
                "quality": quality,
                "views": views,
            }
        ],
    )
    return result


@mcp.tool()
def create_generation_batch(name: str, assets: list[dict]) -> dict:
    """Queue up to 1000 named assets before generating references; launch queue UI.

    Each asset requires prompt and may include name, seed, quality, material_hints,
    and views {front,back,left,right}. Image creation is performed by the calling
    Codex agent. Read get_agent_work and continue the agent repair/review loop.
    """
    return batches.create_batch(name, assets)


@mcp.tool()
def submit_asset_references(
    batch_id: str,
    asset_id: str,
    front_image_path: str,
    back_image_path: str,
    left_image_path: str,
    right_image_path: str,
    background_mode: Literal["auto", "transparent", "chroma"] = "auto",
    key_color: list[int] | None = None,
) -> dict:
    """Prepare immutable RGBA cutouts and run pixel gates; agent visual review is next."""
    return batches.submit_references(
        batch_id,
        asset_id,
        {
            "front": front_image_path,
            "back": back_image_path,
            "left": left_image_path,
            "right": right_image_path,
        },
        background_mode=background_mode,
        key_color=key_color or (255, 0, 255),
    )


@mcp.tool()
def get_agent_work(batch_id: str = "", limit: int = 20) -> dict:
    """Get active-phase image generation, visual review or repair tasks for the Codex agent."""
    return batches.get_agent_work(batch_id or None, limit)


@mcp.tool()
def record_asset_review(
    batch_id: str,
    asset_id: str,
    stage: Literal["references", "shape", "shape_repair", "remesh", "paint", "finish"],
    attempt: int,
    approved: bool,
    notes: str,
    inspected_paths: list[str],
) -> dict:
    """Record the agent's inspection of current-attempt evidence; cannot override a failed gate."""
    return batches.review_asset(
        batch_id, asset_id, stage, attempt, approved, notes, inspected_paths
    )


@mcp.tool()
def retry_asset_stage(
    batch_id: str,
    asset_id: str,
    stage: Literal["references", "shape", "shape_repair", "remesh", "paint", "finish"],
    changes: dict,
    repair_note: str,
) -> dict:
    """Resubmit an agent-repaired stage or rewind its cause, retaining successful other assets."""
    return batches.retry_asset(batch_id, asset_id, stage, changes, repair_note)


@mcp.tool()
def get_shape_repair_capabilities() -> dict:
    """Read typed methods and gate policy; discovery is not runtime qualification."""
    return batches.get_shape_repair_capabilities()


@mcp.tool()
def create_shape_repair_attempt(
    batch_id: str,
    asset_id: str,
    recipe: dict,
    source_sha256: str,
    diagnosis: str,
    idempotency_key: str,
    source_attempt: int = 0,
    expected_shape_attempt: int | None = None,
    inspected_paths: list[str] | None = None,
) -> dict:
    """Queue one typed repair under serial limits; zero source_attempt selects raw shape.

    Positive source_attempt selects a retained repair candidate from the current raw shape.
    A paused batch stays paused. Candidate creation never approves or replaces the master.
    Mutations require an explicit recipe policy region. Use submission for imported edits.
    """
    return batches.create_shape_repair_attempt(
        batch_id, asset_id, recipe, source_sha256, diagnosis, idempotency_key,
        source_attempt=source_attempt, expected_shape_attempt=expected_shape_attempt,
        inspected_paths=inspected_paths,
    )


@mcp.tool()
def submit_shape_repair_candidate(
    batch_id: str,
    asset_id: str,
    candidate_path: str,
    candidate_sha256: str,
    source_sha256: str,
    recipe: dict,
    diagnosis: str,
    idempotency_key: str,
    source_attempt: int = 0,
    expected_shape_attempt: int | None = None,
    inspected_paths: list[str] | None = None,
) -> dict:
    """Stage an allowed-root edited GLB immutably, then run the same repair gates.

    Requires recipe.method=import_candidate, an explicit region, and exact candidate/source
    SHA-256 values. It queues validation and evidence; it does not accept the candidate.
    """
    return batches.submit_shape_repair_candidate(
        batch_id, asset_id, candidate_path, candidate_sha256, source_sha256, recipe,
        diagnosis, idempotency_key, source_attempt=source_attempt,
        expected_shape_attempt=expected_shape_attempt, inspected_paths=inspected_paths,
    )


@mcp.tool()
def get_batch_status(batch_id: str) -> dict:
    """Read durable phase, assets, attempts, gates and pending agent actions."""
    return batches.get_batch_status(batch_id)


@mcp.tool()
def get_queue_status() -> dict:
    """Read lightweight queue and resource state without hashing model artifacts."""
    return batches.queue_snapshot()


@mcp.tool()
def pause_generation_batch(batch_id: str) -> dict:
    """Pause at the current asset boundary and release the worker before more work starts."""
    return batches.pause_batch(batch_id)


@mcp.tool()
def open_generation_queue() -> dict:
    """Open the local queue dashboard with previews and pause/resume/cancel controls."""
    return {"url": batches.dashboard.open()}


@mcp.tool()
def get_generation_status(job_id: str) -> dict:
    """Return durable state, stage, progress, warnings, and available artifacts."""
    return batches.get_batch_status(job_id) if batches.has_batch(job_id) else jobs.get(job_id)


@mcp.tool()
def get_generation_artifacts(job_id: str) -> dict:
    """Return paths and SHA-256 hashes for every available job artifact."""
    if batches.has_batch(job_id):
        batch = batches.get_batch_status(job_id)
        return {
            "job_id": job_id,
            "state": batch["state"],
            "assets": [
                {
                    "asset_id": asset["asset_id"],
                    "name": asset["name"],
                    "artifacts": [
                        item for record in asset["stages"].values() for item in record["artifacts"]
                    ],
                }
                for asset in batch["assets"]
            ],
        }
    return jobs.artifacts(job_id)


@mcp.tool()
def resume_generation(job_id: str) -> dict:
    """Resume an interrupted or repairable failed job from verified outputs."""
    if batches.has_batch(job_id):
        return batches.resume_batch(job_id)
    raise ValueError(
        "Legacy jobs remain readable. Create a new batch using their saved references to "
        "resume processing under the serial resource and review gates."
    )


@mcp.tool()
def cancel_generation(job_id: str) -> dict:
    """Cancel queued or active work while retaining completed stages."""
    return batches.cancel_batch(job_id) if batches.has_batch(job_id) else jobs.cancel(job_id)


@mcp.tool()
def cleanup_generation_job(job_id: str, keep_final: bool = True) -> dict:
    """Explicitly remove job data; preserve final deliverables by default."""
    if batches.has_batch(job_id):
        raise ValueError("Batch attempts are retained for agent review; cleanup is not automatic")
    return jobs.cleanup(job_id, keep_final)


@mcp.tool()
def server_status() -> dict:
    """Report Hunyuan runtime, model cache, GPU dependencies, disk, and queue readiness."""
    batch_preflight = batches.processor.preflight()
    runtime_status = batch_preflight["runtime"]
    runtime_status["ready_for_generation"] = batch_preflight["ready"]
    runtime_status["host_memory"] = batch_preflight["host_memory"]
    return {
        "server_name": settings.server_name,
        "server_version": settings.server_version,
        "backend": "hunyuan3d-2mv",
        "runtime": runtime_status,
        "models": {
            "shape": settings.shape_model_id,
            "shape_subfolder": settings.shape_subfolder,
            "texture": settings.texture_model_id,
            "texture_subfolder": settings.texture_subfolder,
            "shape_revision": settings.shape_revision,
            "texture_revision": settings.texture_revision,
        },
        "queue": jobs.summary(),
        "batches": batches.queue_snapshot(),
        "remesher": batch_preflight["remesher"],
        "execution_policy": {
            "assets_in_parallel": 1,
            "phases_in_parallel": 1,
            "gpu_preferred": True,
            "agent_review_required": True,
            "retry_limit": None,
            "gate_bypass": False,
        },
        "allowed_input_roots": [str(path) for path in settings.allowed_input_roots],
        "output_dir": str(settings.output_dir),
        "supported_views": ["front", "back", "left", "right"],
        "production_defaults": {
            "seed": settings.default_seed,
            "quality": settings.default_quality,
            "texture_resolution": settings.texture_resolution,
            "game_faces": settings.game_faces,
            "lod_faces": list(settings.lod_faces),
            "low_vram_mode": False,
        },
        "texture_conditioning": {
            "native_hunyuan_paint_view": "front",
            "note": "All supplied views constrain geometry; upstream Paint v2 uses the front view.",
        },
    }


def main() -> None:
    logger.info("Starting %s", settings.server_name)
    mcp.run(transport="stdio")


atexit.register(jobs.close)
atexit.register(batches.close)


if __name__ == "__main__":
    main()
