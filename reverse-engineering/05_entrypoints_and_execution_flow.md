# Entrypoints and execution flow

## Confidence

High.

Evidence uses `CONFIRMED`, `INFERRED`, and `UNKNOWN` labels.

## Primary Entrypoints

| Entrypoint | Purpose |
|---|---|
| `examples/shape_gen_multiview.py` | Minimal multiview shape generation |
| `minimal_demo.py` | Minimal single-view shape generation |
| `gradio_app.py` | Interactive browser application |
| `api_server.py` | FastAPI synchronous/asynchronous generation service |
| `blender_addon.py` | Blender-side client for the API |

## Execution Flows

### Load

`from_pretrained` resolves a local/Hugging Face snapshot, reads YAML, splits the safetensors file by its first namespace component, instantiates each component, loads its state dict, then moves all three neural components to the requested device/dtype.

### Multiview inference

The processor validates canonical view keys, sorts them by index, produces `[B,V,3,512,512]`, and passes `view_idxs`. The conditioner produces `[B,V*1370,1536]`. Sampling starts from `[B,3072,64]` Gaussian noise and integrates predicted velocity over `[0,1]`. The VAE decodes final latents and the surface extractor creates a mesh.

### Training candidate

For each paired example: encode the mesh surface to `x1`, sample `x0 ~ N(0,I)` and `t ~ U(0,1)`, form `xt=(1-t)x0+t*x1`, condition on selected views, predict velocity, and minimize mean squared error against `x1-x0`. This is confirmed by Tencent's later official 2.1 transport trainer and is consistent with this repository's Euler sampler; its application to 2mv is an evidence-backed port, not original 2mv training code.

## Unknowns

The original 2mv trainer's timestep distribution and any loss reweighting are not published here.

## Source Files Referenced

`examples/shape_gen_multiview.py`, `hy3dgen/shapegen/pipelines.py`, `hy3dgen/shapegen/preprocessors.py`, `hy3dgen/shapegen/schedulers.py`.
