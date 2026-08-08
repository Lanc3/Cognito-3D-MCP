# Test and quality map

## Confidence

High.

Evidence uses `CONFIRMED`, `INFERRED`, and `UNKNOWN` labels.

## Test Commands

No upstream automated test command or CI workflow is present. Project-specific tests must be added with the training pipeline.

## Existing Tests

The `examples/` scripts are manual smoke tests, not assertions. They download models, run CUDA inference, and write GLB files.

## Critical Regression Tests

- Checkpoint namespaces and strict DiT loading.
- Canonical view sorting and subset handling.
- Flow interpolation endpoints and velocity target sign.
- Frozen VAE/conditioner parameters during DiT fine-tuning.
- Adapter save/load round trip and base-revision manifest.
- One optimizer step on a tiny synthetic model without NaNs.
- Fixed-seed generation comparison against the unmodified base model before training.

## Golden Master Candidates

- DINO token tensor for a fixed canonical input set.
- DiT velocity tensor for fixed latents/time/conditions.
- Final latent after a short fixed Euler schedule.
- Mesh vertex/face summary and rendered silhouettes for a fixed seed.

## Manual QA

Inspect front/left/back/right alignment, floaters, holes, symmetry, thin structures, and output diversity. Evaluation should compare both single-view and multiview subsets and include held-out object identities.

## Unknowns

Upstream numerical tolerances and official evaluation thresholds are not published in this repository.

## Source Files Referenced

`examples/shape_gen_multiview.py`, `hy3dgen/shapegen/pipelines.py`, `hy3dgen/shapegen/schedulers.py`.
