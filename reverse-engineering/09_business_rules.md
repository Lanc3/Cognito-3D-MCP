# Business Rules

Generated: 2026-08-03T16:08:21.126845+00:00

## Confidence
Low to Medium

## Source Files Referenced
- `hy3dgen/rembg.py`

## Claim Labels
- CONFIRMED: directly supported by source files, manifests, tests, or configs.
- INFERRED: likely from framework conventions, naming, or partial static evidence.
- UNKNOWN: not enough evidence yet.

## Candidate Rule Evidence
| Area | File | Line | Evidence |
|---|---|---|---|
| auth | hy3dgen/rembg.py | 21 | self.session = new_session() |
| auth | hy3dgen/rembg.py | 24 | output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0]) |

## Agent Instructions
Promote candidates into rules only after reading the surrounding function/test/schema and recording evidence in `.facts/claims.jsonl`.

## UNKNOWN
- Domain rules are not confirmed by keyword matches alone.
