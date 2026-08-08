# Risks, unknowns, and questions

## Confidence

High that these are the current primary risks.

Evidence uses `CONFIRMED`, `INFERRED`, and `UNKNOWN` labels.

## High-Risk Unknowns

| Unknown | Impact | Next investigation |
|---|---|---|
| Exact original render/data recipe | Domain shift can dominate fine-tuning | Establish camera/background/scale invariants with a small controlled set |
| 2mv timestep sampler and weighting | Can change convergence and output sharpness | Begin with official 2.1 linear velocity objective; ablate later |
| Required GPU memory per strategy | Determines feasible local workflow | Measure cached-condition LoRA at 1, 2, then 3 views |
| Sharp-edge preprocessing robustness | Bad or empty edge samples corrupt VAE targets | Validate counts/normals and provide uniform fallback |
| Checkpoint merge parity | A malformed export can silently break inference | Strict load plus fixed-seed golden test |

## Questions for Maintainer

1. What kind of new version is desired first: a style/domain LoRA, category specialist, stronger multiview consistency model, or full foundation retrain?
2. How many licensed paired meshes and canonical renders are available?
3. Is cloud/multi-GPU training available after local smoke tests?
4. Must output remain a small adapter, or should the pipeline also produce a standalone merged safetensors checkpoint?

## Suggested Investigation

Start with cached latents/conditions and a LoRA over DiT linear projections. Prove an overfit on 8–32 objects, then scale only after deterministic validation and memory telemetry pass.

## Source Files Referenced

`models/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/config.yaml`, `hy3dgen/shapegen/models/conditioner.py`, `hy3dgen/shapegen/models/autoencoders/attention_blocks.py`, `hy3dgen/shapegen/pipelines.py`.
