---
name: codex-3d-models
description: Run legacy or comparison Stable Fast 3D and SPAR3D single-image jobs through the codex_3d_models MCP servers. Use only when the user explicitly requests SF3D or SPAR3D, asks for an A/B backend comparison, needs their backend-specific remeshing controls, or is diagnosing an existing SF3D/SPAR3D job. For normal image-to-3D generation, use codex-hunyuan-multiview-3d and codex_3d_models_hunyuan_mv instead.
---

# Cognito-3D-mcp: Single-image 3D

Generate legacy or comparison textured GLB assets from one reference image. Use `codex_3d_models` for Stable Fast 3D and `codex_3d_models_spar3d` for SPAR3D. Do not select these for a normal generation request; Hunyuan3D-2mv is the primary route.

## Choose a backend

Use Stable Fast 3D only when the user explicitly wants the fastest legacy draft. Use SPAR3D when the user explicitly asks for its single-image reconstruction or material prediction. For an A/B comparison, submit the same image and parameters to both servers, keep both job IDs, poll each server independently, and label both artifacts clearly.

## Prepare the request

Require exactly one image. Prefer `image_path` for a local PNG, JPEG, or WebP already beneath one of the server's `allowed_input_roots`. Use `image_data_url` or `image_base64` when the image is attached but has no permitted local path. Never send more than one image field.

Ask the user to attach or identify a reference image if none is available. Do not treat a text prompt as a substitute: Stable Fast 3D is image-conditioned. The required `prompt` is job metadata and does not alter geometry or texture.

Favor a clean, centered object image with a plain or removable background. Explain that unseen surfaces are inferred from the single view.

## Check readiness

Call `server_status` before the first generation in a task. Inspect:

- `model.ready_for_generation`
- `model.backend` and `model.model_name`
- `model.dependencies`
- `model.device` and `model.device_error`
- `model.model_cache_present`
- `model.huggingface_authenticated`
- `model.huggingface_model_access` and `model.huggingface_access_error`
- `allowed_input_roots`
- `gpu_arbitration.resource_error`: if present, restart the server after resolving
  the failed model unload; submitting another job cannot repair lost ownership.

Do not submit a generation when readiness is false. Report the missing condition. A configured token is not proof of gated model access: require `huggingface_model_access` or a complete local cache. For absent model access, ask the user to request access to the reported `model_id` (`stabilityai/stable-fast-3d` or `stabilityai/stable-point-aware-3d`), accept its terms with the same Hugging Face account, and authenticate the server process with a read token or CLI login. Never request that a token be pasted into chat.

## Choose parameters

Use conservative defaults unless the user has a reason to change them:

- `texture_resolution`: `1024`. Use `512` for a fast draft; use `2048` or `4096` only when extra texture detail justifies more time and VRAM.
- `foreground_ratio`: `0.85`. Raise it when the object is too small in frame; lower it when the crop is tight.
- `remesh`: `none`. Use `triangle` or `quad` only when topology requirements matter.
- `target_vertex_count`: omit unless the user asks for a mesh budget. Valid range is 100–10,000,000.

## Run the job

1. Call `generate_3d_model` with the prompt, exactly one image field, and chosen parameters.
2. Preserve the returned `job_id`.
3. Poll `get_generation_status` about every two seconds. Do not flood the single-worker server.
4. Continue until `status` is `completed`, `failed`, or `cancelled`.
5. On completion, use the absolute `artifact_path`. Confirm the file exists when filesystem access is available, then link or present the GLB to the user.
6. On failure, surface `error.code` and `error.message` with the relevant remedy. Do not claim an artifact was produced.

Call `cancel_generation` when the user cancels or the result is no longer needed. Cancellation is cooperative for running model work and immediate for queued work.

## Handle failures

- `MODEL_UNAVAILABLE`: call `server_status`; check Hugging Face access/cache, PyTorch/CUDA, and compiled dependencies.
- `INVALID_PATH`: move or re-express the input beneath an allowed root, or send data/base64 instead.
- `INVALID_IMAGE` or `IMAGE_TOO_LARGE`: use a valid PNG/JPEG/WebP within the configured byte limit.
- `QUEUE_FULL`: wait and retry later; do not create a rapid retry loop.
- `JOB_NOT_FOUND`: re-check the exact `job_id`; jobs are process-local and disappear after server restart.

## Deliver the result

State that the output is a textured GLB and provide its absolute artifact link/path. Mention material limitations only when relevant. For redistribution or commercial use, direct the user to the repository `NOTICE` and Stable Fast 3D's Stability AI Community License requirements.
