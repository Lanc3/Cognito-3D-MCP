# Contributing to Cognito-3D-mcp

Use Python 3.11 or 3.12 for development. Install the lightweight development
dependencies in an isolated environment; GPU runtimes are unnecessary for the
unit suite:

```sh
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` on Windows or
`source .venv/bin/activate` on Linux), then run:

```sh
python -m pip install -e . -r requirements-dev.txt build
python -m pytest
python -m ruff check --select E9,F src tests scripts
python scripts/verify-release.py
python -m build
```

Real Blender integration tests are skipped
when Blender is unavailable. GPU smoke checks require a separately installed,
licensed runtime and should be reported separately from CPU tests.

Open a focused pull request describing the failing behavior, the fix, and the
checks you ran. Add regression coverage for changes to job state, cancellation,
input validation, quality gates, resource leases, or HTTP access. Preserve
serialized GPU execution and explicit approval gates. Do not silently relax a
gate to make a test pass.

Keep the agent workflows in `skills/` aligned with the MCP API. Existing
`codex_3d_models*` registration names and `CODEX_*` environment variables remain
compatibility identifiers; changing them needs a migration plan.

Do not commit credentials, generated models, local reference images, caches,
environments, job databases, or build output. Third-party materials require their
original notices. By contributing original code you agree to license that
contribution under this repository's MIT license.
