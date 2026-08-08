# Architecture map

## Confidence

High.

Evidence uses `CONFIRMED`, `INFERRED`, and `UNKNOWN` labels.

## Layered View

1. Inputs: canonical object images and, for training, a matching normalized mesh/surface.
2. Preprocessing: alpha-derived foreground crop, white compositing, 512px resize, canonical view sort.
3. Conditioning: frozen 40-layer DINOv2-Giant at 518px plus fixed per-view sinusoidal embeddings.
4. Shape representation: VAE maps 10,240 point+normal surface samples to 3,072 vectors of width 64.
5. Generative core: flow-matching DiT maps interpolated/noisy shape latents and condition tokens to velocity.
6. Output: VAE transformer and cross-attention geometry decoder evaluate occupancy/SDF-like logits; marching cubes returns a mesh.

## Major Directories

| Directory | Responsibility | Risk |
|---|---|---|
| `hy3dgen/shapegen/models` | DINO conditioner, DiT, VAE | High: numerical and checkpoint compatibility |
| `hy3dgen/shapegen` | Loading, scheduling, preprocess/postprocess | High: defines public behavior |
| `hy3dgen/texgen` | Separate texture pipeline and CUDA extensions | Medium for this project |
| `examples` | Executable usage examples | Low |
| `assets` | Example inputs, reports, viewer templates | Low |
| `reverse-engineering` | This evidence dossier | Low |

## Core Modules

- `Hunyuan3DDiT`: joint-attention double-stream blocks followed by concatenated single-stream blocks and an AdaLN final projection.
- `DinoImageEncoderMV`: independently encodes images, adds view embeddings, and concatenates tokens.
- `ShapeVAE`: point cross-attention encoder plus Gaussian bottleneck and geometry decoder.
- `FlowMatchEulerDiscreteScheduler`: linear interpolation noise process and first-order ODE integration.

## Dependency Direction

Applications depend on pipelines; pipelines instantiate models/preprocessors/schedulers from YAML; model modules depend on PyTorch/Transformers; only postprocessing depends on mesh libraries. The training layer should depend on model modules, not on Gradio/API modules.

## Architectural Risks

- Single-stream self-attention sees conditioning and 3D latent tokens together; activation memory grows quadratically with view count.
- `DinoImageEncoderMV.unconditional_embedding` assumes `view_idxs` is supplied.
- The checkpoint loader splits tensor names on the first dot; exported versions must preserve `model.`, `vae.`, and `conditioner.` prefixes.
- Several requirements are unpinned, which makes binary compatibility and reproducibility fragile.

## Source Files Referenced

`hy3dgen/shapegen/pipelines.py`, `hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`, `hy3dgen/shapegen/models/conditioner.py`, `hy3dgen/shapegen/models/autoencoders/model.py`, `hy3dgen/shapegen/models/autoencoders/attention_blocks.py`.
