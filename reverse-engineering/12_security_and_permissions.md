# Security and Permissions

Generated: 2026-08-03T16:08:21.126845+00:00

## Confidence
Medium

## Source Files Referenced
- `api_server.py`
- `hy3dgen/rembg.py`
- `hy3dgen/shapegen/pipelines.py`
- `hy3dgen/texgen/custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer.cpp`

## Claim Labels
- CONFIRMED: directly supported by source files, manifests, tests, or configs.
- INFERRED: likely from framework conventions, naming, or partial static evidence.
- UNKNOWN: not enough evidence yet.

## Security Boundary Candidates
| Area | File | Line | Evidence |
|---|---|---|---|
| web-security | api_server.py | 233 | from fastapi.middleware.cors import CORSMiddleware |
| auth | hy3dgen/rembg.py | 21 | self.session = new_session() |
| auth | hy3dgen/rembg.py | 24 | output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0]) |
| web-security | hy3dgen/shapegen/pipelines.py | 69 | accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys()) |
| web-security | hy3dgen/shapegen/pipelines.py | 79 | accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys()) |
| web-security | hy3dgen/shapegen/pipelines.py | 462 | accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()) |
| web-security | hy3dgen/shapegen/pipelines.py | 468 | accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys()) |
| secret-handling | hy3dgen/texgen/custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer.cpp | 32 | INT64 token = (INT64)z_quantize * MAXINT + (INT64)(idx + 1); |
| secret-handling | hy3dgen/texgen/custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer.cpp | 35 | zbuffer[pixel] = std::min(zbuffer[pixel], token); |

## UNKNOWN
- Authentication model, role model, secret storage, upload validation, CORS/CSRF, and rate limiting are unknown until confirmed from source.
