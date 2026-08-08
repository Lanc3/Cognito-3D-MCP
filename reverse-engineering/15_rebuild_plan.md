# Training pipeline implementation plan

## Confidence

High for sequencing; medium for final scale hyperparameters.

Evidence uses `CONFIRMED`, `INFERRED`, and `UNKNOWN` labels.

## Goal

Create reproducible, versioned Hunyuan3D-2mv derivatives while preserving compatibility with the public inference pipeline.

## Invariants

- Preserve canonical view indices and preprocessing.
- Preserve latent shape `[3072,64]` and checkpoint namespaces.
- Freeze the VAE and DINO conditioner for the initial pipeline.
- Record base code commit, model revision/hash, dataset manifest hash, config, and software versions in every output version.
- Use safetensors for model artifacts.

## Rebuild Order

1. Pin source and checkpoint provenance.
2. Define/validate a JSONL manifest pairing surfaces/meshes with canonical views.
3. Cache VAE latents and per-view DINO tokens to move frozen models out of the training memory path.
4. Implement the linear flow velocity objective with classifier-free condition dropout.
5. Add LoRA-first training, gradient accumulation/checkpointing, mixed precision, resumable state, and deterministic logging.
6. Add adapter export and optional full checkpoint merge preserving `model/vae/conditioner` namespaces.
7. Add fixed-seed latent and mesh evaluation.
8. Prove tiny-set overfitting, then held-out quality, then multi-GPU scaling.

## Parity Tests

- Base checkpoint SHA-256 and strict component loading.
- Cached vs live conditioner token equality.
- Cached vs live VAE latent statistical check (posterior sampling is stochastic; mode should match exactly).
- Flow loss target matches `x1-x0` for the linear path.
- Zero-step/zero-adapter inference matches the base model.
- Merged checkpoint reloads through `Hunyuan3DDiTFlowMatchingPipeline.from_single_file`.

## Migration Concerns

Training state (optimizer/scaler/RNG) and deployable model artifacts should be separate. LoRA adapters must declare their target module patterns and base revision. A future full-rank run will require distributed sharding and cannot reuse local-only memory assumptions unchanged.

## Source Files Referenced

`hy3dgen/shapegen/pipelines.py`, `hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`, `hy3dgen/shapegen/models/conditioner.py`, `hy3dgen/shapegen/models/autoencoders/model.py`, `hy3dgen/shapegen/schedulers.py`.
