# Hunyuan3D-2mv reverse-engineering dossier

## Confidence

High for the multiview shape inference path and checkpoint structure; medium for the original 2mv training recipe because Tencent did not publish a 2.0 trainer in this repository.

Evidence uses `CONFIRMED`, `INFERRED`, and `UNKNOWN` labels as defined by the dossier schema.

## Application Identity

This is a Python/PyTorch inference library and set of local applications for image-conditioned 3D shape generation and optional texture synthesis. The target `Hunyuan3D-2mv` variant accepts one to four canonical views and generates a shape latent that the bundled shape VAE decodes to a mesh.

## Most Important Files

| File | Responsibility |
|---|---|
| `models/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/config.yaml` | Exact 2mv component graph and dimensions |
| `hy3dgen/shapegen/pipelines.py` | Checkpoint loading and end-to-end sampling |
| `hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py` | 1.1B flow-matching transformer |
| `hy3dgen/shapegen/models/conditioner.py` | Frozen DINOv2 encoder and canonical-view embeddings |
| `hy3dgen/shapegen/models/autoencoders/model.py` | Surface-to-latent VAE and latent-to-mesh decoder |
| `hy3dgen/shapegen/preprocessors.py` | Image recentering and canonical view ordering |
| `hy3dgen/shapegen/schedulers.py` | Reversed flow-matching Euler integration |
| `models/MODEL_MANIFEST.json` | Pinned checkpoint revision, byte size, and SHA-256 |

## Main Execution Path

`MVImageProcessorV2` -> `DinoImageEncoderMV` -> Gaussian shape latent -> `Hunyuan3DDiT` Euler flow -> `ShapeVAE.latents2mesh` -> trimesh export.

See `diagrams/runtime_flow.mmd` for the data shapes and component boundaries.

## Document Map

- [System overview](01_system_overview.md)
- [Runtime and setup](02_runtime_and_setup.md)
- [Architecture map](03_architecture_map.md)
- [Module inventory](04_module_inventory.md)
- [Entrypoints and execution flow](05_entrypoints_and_execution_flow.md)
- [Data model](06_data_model.md)
- [API surface](07_api_surface.md)
- [UI surface](08_ui_surface.md)
- [Business rules](09_business_rules.md)
- [External integrations](10_external_integrations.md)
- [State and side effects](11_state_and_side_effects.md)
- [Security and permissions](12_security_and_permissions.md)
- [Test and quality map](13_test_and_quality_map.md)
- [Risks and unknowns](14_risks_unknowns_and_questions.md)
- [Training/rebuild plan](15_rebuild_plan.md)

## Critical Unknowns

- The exact private 2mv dataset composition, filtering, camera/rendering recipe, timestep distribution, and optimizer schedule are not present in the 2.0 source.
- The base VAE's encoder defaults require 5,120 uniform plus 5,120 sharp-edge samples; the public helper defaults do not exactly match this, so preprocessing must make the sizes explicit.
- Full-rank training hardware requirements are not published for 2mv. The local 16 GB GPU is a LoRA/smoke-test target, not a credible full-rank training target.

## Source Files Referenced

`hy3dgen/shapegen/pipelines.py`, `hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`, `hy3dgen/shapegen/models/conditioner.py`, `hy3dgen/shapegen/models/autoencoders/model.py`, `hy3dgen/shapegen/models/autoencoders/attention_blocks.py`, `hy3dgen/shapegen/preprocessors.py`, `hy3dgen/shapegen/schedulers.py`, `hy3dgen/shapegen/surface_loaders.py`, `examples/shape_gen_multiview.py`.
