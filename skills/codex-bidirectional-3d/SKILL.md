---
name: codex-bidirectional-3d
description: Create, resume, review, and deliver production 3D assets from matched front and back PNG references through the local codex_3d_models_trellis MCP server. Use for sequential SPAR3D reconstruction, automatic back-side improvement, front/back or four-view fusion, 4K PBR texture reprojection, a watertight master GLB, a game-ready GLB, LODs, Blender previews, or quality evidence. Do not use for the existing one-image SF3D/SPAR3D comparison workflow.
---

# Cognito-3D-mcp: Bidirectional 3D

Create a consistent reference pair, submit one durable local job, handle its review states, and deliver only QA-approved GLBs.

Call `server_status` to discover `allowed_input_roots` and the configured
`reconstruction_backend`. For references in another project, copy only the
selected PNGs into a unique job subdirectory of the reported allowed input
directory, preserve the originals, and submit the staged absolute paths. Do not
broaden access to an entire drive. Return accepted deliverables to the requesting
project after the quality gates pass.

## Prepare references

When the user provides a description instead of two images:

1. Generate the front view with the image-generation tool.
2. Create the back as an edit/reference of that front image. Specify a strict 180-degree turn with identical object identity, proportions, scale, camera elevation, crop, lighting, and material placement. Do not generate an unrelated second image.
3. Save both as square PNG files at least 1024x1024 beneath an allowed input root.

Require one centered upright object, a transparent or plain background, even neutral light, at least 2% clear border, and no floor, stand, text, hands, cast shadow, or extra objects. Never use a mirrored front as the back.

Extract concise material hints from the request: name metal, rough, glossy, transparent, cutout, cloth, organic, painted, and glowing regions. Read [quality-contract.md](references/quality-contract.md) when a pair is rejected or a completed job fails QA.

## Submit the production job

1. Call `codex_3d_models_trellis.server_status` before the first job. Require `ready_for_generation=true`. Report the exact missing runtime condition without asking for tokens or secrets.
2. Call `generate_bidirectional_3d_model` with the prompt, absolute front/back paths, material hints, and seed 42 unless reproducibility requires another seed.
3. Preserve the returned job ID. Poll `get_generation_status` about every five seconds; do not submit duplicate jobs while one is active.

The prompt is provenance. Geometry is conditioned by the images.

## Handle durable states

- `awaiting_review`: inspect `metadata.pair_validation`. Show both references and the warnings. Call `review_seed_pair(job_id, decision, notes)` to approve only warning-level pairs that still depict the same object, or reject them. A structurally rejected pair must be regenerated and submitted as a new job.
- `awaiting_side_inputs`: explain the reported side disagreement and ask before generating left/right images. If approved, derive them from the same canonical front reference and call `add_side_references`.
- `interrupted` or operational `failed`: inspect the error and available checkpoints. Call `resume_generation` after correcting the missing runtime condition.
- `failed_quality`: do not claim success and do not substitute the front mesh. Link the QA report, previews, raw meshes, and working Blender file for repair.
- `completed`: call `get_generation_artifacts` and deliver `master.glb`, `game.glb`, `lod1.glb`, `lod2.glb`, and the QA report with clear labels.

Call `cancel_generation` when the user cancels. Call `cleanup_generation_job` only after explicit instruction; cleanup is irreversible and failed jobs are intentionally preserved.

## Protect production guarantees

Do not silently lower the configured reconstruction profile or 4K texture profile. Do not publish artifacts before the hard QA gates pass. Use the `reconstruction_backend` reported by `server_status`: TRELLIS-only installations use TRELLIS, and installations including SPAR3D can use sequential SPAR3D. With SPAR3D, keep one resident child alive for all job views. Do not run SF3D, SPAR3D, TRELLIS, Real-ESRGAN, or Blender GPU work concurrently; the servers share a machine-wide GPU lease.

Image generation may be remote. All reconstruction, fusion, upscaling, texture baking, optimization, and QA are local.
