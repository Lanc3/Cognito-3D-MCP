# External Integrations

Generated: 2026-08-03T16:08:21.126845+00:00

## Confidence
Medium

## Source Files Referenced
- `hy3dgen/shapegen/models/autoencoders/attention_blocks.py`
- `hy3dgen/shapegen/models/autoencoders/attention_processors.py`
- `hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`
- `hy3dgen/shapegen/utils.py`
- `hy3dgen/texgen/pipelines.py`

## Claim Labels
- CONFIRMED: directly supported by source files, manifests, tests, or configs.
- INFERRED: likely from framework conventions, naming, or partial static evidence.
- UNKNOWN: not enough evidence yet.

## Integration Dependency Candidates
_None detected._

## Environment Variable Clues
| Name | Kind | Source | Line |
|---|---|---|---|
| CA_USE_SAGEATTN | code-reference | hy3dgen/shapegen/models/autoencoders/attention_processors.py | 21 |
| HY3DGEN_DEBUG | code-reference | hy3dgen/shapegen/utils.py | 62 |
| HY3DGEN_MODELS | code-reference | hy3dgen/shapegen/utils.py | 97 |
| HY3DGEN_MODELS | code-reference | hy3dgen/texgen/pipelines.py | 59 |
| USE_SAGEATTN | code-reference | hy3dgen/shapegen/models/autoencoders/attention_blocks.py | 29 |
| USE_SAGEATTN | code-reference | hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py | 25 |

## UNKNOWN
- Provider purpose, failure behavior, and mocking strategy require source-backed analysis.
