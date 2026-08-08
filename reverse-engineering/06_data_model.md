# Data model

## Confidence

High for runtime tensors; medium for training dataset conventions.

Evidence uses `CONFIRMED`, `INFERRED`, and `UNKNOWN` labels.

## Entities

| Entity | Shape/content | Lifecycle |
|---|---|---|
| Canonical views | Mapping of 1–4 tags to images | Input -> normalized tensors |
| View indices | `front=0,left=1,back=2,right=3` | Added to DINO tokens |
| Surface | 5,120 uniform + 5,120 sharp-edge point/normal rows | Mesh -> VAE encoder input |
| Shape latent | `[B,3072,64]` | VAE data target / DiT state |
| Condition tokens | `[B,V*1370,1536]` | Frozen DINO output |
| Flow time | `[B]`, normalized to `[0,1]` | Sampled during training / stepped in inference |
| Velocity | `[B,3072,64]` | DiT output and flow target |
| Mesh | vertices/faces in trimesh | VAE decode -> export |

## Relationships

Every training example must pair one surface/mesh with images of the same object. View tags are semantic and must remain aligned with the rendered camera convention.

## Constraints

- Only the four canonical tags are accepted.
- Empty alpha masks are rejected by recentering.
- The VAE point encoder splits its input at configured uniform/sharp sizes; dataset tensors must match those sizes exactly.
- Conditioning token count varies with the selected number of views, so all examples in one batch need compatible padding or the same selected views. The initial pipeline uses batch size one to avoid ambiguous masking.

## State Lifecycle

Raw mesh/images -> validated manifest -> cached VAE latent and per-view DINO tokens -> stochastic flow training batches -> versioned adapter/full checkpoint -> inference evaluation.

## Source Files Referenced

`hy3dgen/shapegen/preprocessors.py`, `hy3dgen/shapegen/models/conditioner.py`, `hy3dgen/shapegen/models/autoencoders/attention_blocks.py`, `hy3dgen/shapegen/models/autoencoders/model.py`.
