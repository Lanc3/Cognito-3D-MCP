# Runtime and Setup

Generated: 2026-08-03T16:08:21.124818+00:00

## Confidence
Medium

## Source Files Referenced
- `hy3dgen/shapegen/models/autoencoders/attention_blocks.py`
- `hy3dgen/shapegen/models/autoencoders/attention_processors.py`
- `hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`
- `hy3dgen/shapegen/utils.py`
- `hy3dgen/texgen/pipelines.py`
- `requirements.txt`

## Claim Labels
- CONFIRMED: directly supported by source files, manifests, tests, or configs.
- INFERRED: likely from framework conventions, naming, or partial static evidence.
- UNKNOWN: not enough evidence yet.

## Commands
_None detected._

## Dependencies
| Manager | Name | Version | Scope | Source |
|---|---|---|---|---|
| pip | ninja |  | runtime | requirements.txt |
| pip | pybind11 |  | runtime | requirements.txt |
| pip | diffusers |  | runtime | requirements.txt |
| pip | einops |  | runtime | requirements.txt |
| pip | opencv-python |  | runtime | requirements.txt |
| pip | numpy |  | runtime | requirements.txt |
| pip | torch |  | runtime | requirements.txt |
| pip | transformers |  | runtime | requirements.txt |
| pip | torchvision |  | runtime | requirements.txt |
| pip | omegaconf |  | runtime | requirements.txt |
| pip | tqdm |  | runtime | requirements.txt |
| pip | trimesh |  | runtime | requirements.txt |
| pip | pymeshlab |  | runtime | requirements.txt |
| pip | pygltflib |  | runtime | requirements.txt |
| pip | xatlas |  | runtime | requirements.txt |
| pip | accelerate |  | runtime | requirements.txt |
| pip | gradio |  | runtime | requirements.txt |
| pip | fastapi |  | runtime | requirements.txt |
| pip | uvicorn |  | runtime | requirements.txt |
| pip | rembg |  | runtime | requirements.txt |
| pip | onnxruntime |  | runtime | requirements.txt |

## Environment Variables
| Name | Kind | Source | Line |
|---|---|---|---|
| CA_USE_SAGEATTN | code-reference | hy3dgen/shapegen/models/autoencoders/attention_processors.py | 21 |
| HY3DGEN_DEBUG | code-reference | hy3dgen/shapegen/utils.py | 62 |
| HY3DGEN_MODELS | code-reference | hy3dgen/shapegen/utils.py | 97 |
| HY3DGEN_MODELS | code-reference | hy3dgen/texgen/pipelines.py | 59 |
| USE_SAGEATTN | code-reference | hy3dgen/shapegen/models/autoencoders/attention_blocks.py | 29 |
| USE_SAGEATTN | code-reference | hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py | 25 |

## UNKNOWN
- Required runtime versions are unknown unless pinned in manifests or config.
- Do not run install/build/test commands without user approval if the repo is untrusted.
