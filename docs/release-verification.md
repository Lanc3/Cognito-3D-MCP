# Release verification: 0.4.0

This release was reviewed in multiple passes by independent runtime, dashboard,
and installation agents, followed by integration and package checks on
2 October 2026. A second pass located the current 0.4 repair implementation and
reconciled its API with the shipped skills before publication.

## Verified locally

- Full Windows CPU regression suite: 375 passed, 8 optional integration checks
  skipped, including native CPU geometry fixtures.
- Job cancellation isolation; queue saturation and shutdown ordering; database
  lifetime; stale worker output; shared GPU-lock ownership and unload failures.
  Legacy models unload between jobs; failure retains the lease and requires a
  server restart. These tests simulate ownership; non-PyTorch native allocations
  and real GPU unload behavior still need hardware checks.
- Bounded image decoding and input-root validation; immutable repair candidates,
  hash-bound selections, topology/preservation gates and accepted-master state.
- HTTP access and framing tests, plus 12 Chromium software-WebGL browser checks:
  mobile layout/focus, repair UI, candidate/master history, viewer resources,
  malicious metadata, invalid paths and external GLB subresources. No browser
  errors or external resource requests occurred in the final fixture run.
- Offline installer helper checks in Windows PowerShell 5.1 and PowerShell 7:
  compiler-environment quoting, checksum validation, corrupted native files,
  exit codes and harmless native stderr. Dry-run plans perform no installs.
- Config backup/preservation, duplicate server aliases, backend paths and skill
  reference copying. Skills were validated with the skill-creator validator and
  compared against the actual tool APIs.
- Mocked gated model access, credential reuse, pinned download revisions and
  safe error messages; manual launch selects an installed engine.
- Real STDIO initialization, tool discovery and `server_status` for all three
  servers, from source and a wheel installed into a clean environment with only
  the declared runtime dependencies. No model inference was invoked.
- Non-Windows status reports the Windows-only remesher limitation without
  probing Windows memory APIs; unsupported generation remains disabled.
- Python syntax/Pyflakes checks, wheel and source builds, packaged viewer assets,
  skill references, licensing notices, and exclusion of private/generated files.
- A credential-pattern scan of release source. This is a release hygiene check,
  not a guarantee that every possible secret format can be detected.

GitHub Actions repeats CPU tests, lint, MCP startup and package verification on
Windows and Ubuntu with Python 3.11 and 3.12. Blender and Torch-specific tests
require separately provisioned tools and may be skipped in CPU environments.

## Hardware validation still required

The heavyweight installer was not run on a clean GPU machine during this
release review. No drivers, toolkits, model weights or native GPU builds were
downloaded or installed on the development host. The installer is supplied and
its offline behavior is tested; a complete fresh-machine CUDA installation,
real generation, and optional CuMesh build/probe need hardware verification.

Dependency pins, CUDA versions, drivers, model access and upstream download
availability can affect installation. The installer fails on a missing
prerequisite, rejected licence/access check, command failure or checksum mismatch
and retains useful diagnostics. It cannot obtain gated model approval for a user.

The pinned Hunyuan licence excludes the EU, UK and South Korea. Users must select
an engine they have rights to use; the integration's MIT licence does not grant
model rights. See THIRD_PARTY_NOTICES.md and each engine's full terms.

## Release scope

The current source contains only the 3D integration and its supporting files.
The 2D sprite studio, generated/private assets, local credentials, environments
and experimental reports are excluded. Heavy engines are fetched into isolated
runtimes. Original GitHub history is preserved; the old embedded engine API is
superseded as described in UPSTREAM.md.
