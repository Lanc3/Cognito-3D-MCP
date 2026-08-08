# API Surface

Generated: 2026-08-03T16:08:21.126845+00:00

## Confidence
Medium

## Source Files Referenced
- `api_server.py`

## Claim Labels
- CONFIRMED: directly supported by source files, manifests, tests, or configs.
- INFERRED: likely from framework conventions, naming, or partial static evidence.
- UNKNOWN: not enough evidence yet.

## Routes and Endpoint Candidates
| Method | Path | Framework | File | Line |
|---|---|---|---|---|
| POST | /generate | express-like | api_server.py | 244 |
|  | post | fastapi-like | api_server.py | 244 |
| POST | /send | express-like | api_server.py | 277 |
| GET | /status/{uid} | express-like | api_server.py | 287 |
|  | get | fastapi-like | api_server.py | 287 |

## CLI/Task Commands
_None detected._

## UNKNOWN
- Request/response schemas, auth requirements, and side effects for each endpoint require source-backed expansion.
