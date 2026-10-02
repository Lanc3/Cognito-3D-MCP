# Shape repair candidates and acceptance

The 0.4.0 pipeline adds a separate shape-repair phase between raw Hunyuan shape
generation and AutoRemesher. Raw shapes remain immutable. Every accepted master
must refer to a particular candidate, source hash, machine gate and agent review.
This document describes the implemented repair contract. CPU regression fixtures
exercise geometry and state transitions; native GPU qualification and visual
acceptance of a real asset are separate checks. A passing topology gate does not
establish that a shape has the intended solid volume or that it is visually correct.
Read the per-method capabilities: a successful build alone does not qualify an
operation, and every candidate still requires geometry gates and visual review.

## Agent operating flow

1. Read `server_status`, `get_queue_status`, `get_shape_repair_capabilities` and
   `get_agent_work`. Package discovery is not runtime qualification. Preserve the
   active batch's pause state and do not resume another project's work. Prefer a
   qualified GPU recipe when it addresses the measured defect; select it
   explicitly. An unavailable GPU method never triggers automatic CPU fallback.
2. Finish all references and then all raw shapes serially. Repair starts after
   the entire raw-shape phase finishes and its model is unloaded.
3. Inspect the current raw shape and its four reference images. Start with an
   `analyze` recipe to obtain source-bound topology, component and boundary-loop
   diagnostics. Analysis copies the source unchanged into a new candidate attempt.
4. Diagnose the failure. Submit a bounded recipe with
   `create_shape_repair_attempt`, or stage an authored edit with
   `submit_shape_repair_candidate`. Include the exact source SHA-256, current
   shape attempt, a concrete diagnosis, asset-local inspected evidence and an
   idempotency key. `source_attempt=0` means the immutable raw shape; a positive
   value means a retained repair candidate from that same current raw shape.
5. The scheduler runs the candidate operation and common gates under the shared
   compute lock. Inspect the actual exported/reloaded GLB, diagnostics, repair
   report and fixed previews. A failed operation or gate stays awaiting repair.
6. Record review for `stage="shape_repair"` and the exact attempt only after
   inspecting its evidence. A machine failure cannot be overridden by a visual
   approval. A successful review binds the accepted master to that candidate.
7. After every asset passes repair and review, AutoRemesher consumes the accepted
   master independently for the three profiles. Paint and finish follow their
   existing serial phase barriers and gates.

Reusing an idempotency key with the identical request does not create duplicate
work; using it with changed content is an error. Source hashes and expected raw
shape attempts prevent applying stale diagnoses. A paused batch remains paused
when a repair is queued. A rejected result requires a causal change or an authored
repair; resource failure must not become an unchanged retry loop.

## Available typed recipes

| Method | Intended use | Required restriction |
|---|---|---|
| `analyze` | Produce diagnostics and validate an unchanged source copy | Does not claim that a closed-looking object has the intended solid interior |
| `conservative_cleanup` | Explicit exact/tolerance weld, collapsed/duplicate face removal, optional winding correction | World-space region, displacement/removal limits; no blanket simplification |
| `patch_selected_holes` | Fill explicitly identified boundary loops | Exact diagnostic loop IDs, diameter/edge/area limits and a region; triangle or supported PyMeshLab backend |
| `remove_selected_components` | Remove diagnosed unwanted fragments | Exact component IDs, face/area ceilings and protected-component policy |
| `foundation_union` | Union an authored box, footprint or loft into the intended foundation | Manifold3D, explicit world-space edit region and occupied-volume foundation policy |
| `import_candidate` | Validate an agent-authored external GLB through the same gates | Allowed-root source, exact source/candidate hashes, immutable staging and edit region |
| `cumesh_diagnose` | Optional CUDA topology and loop diagnostics | Verified pinned CUDA installation; common gates still judge the exported candidate |
| `cumesh_cleanup` | Optional CUDA removal of explicitly selected duplicate/degenerate face classes | Declared edit region, preservation checks and per-method qualification |
| `cumesh_fill_selected_holes` | Optional CUDA patch of selected planar convex loops | Exact edge selection, input geometry hash and perimeter bound; new faces oriented against the unchanged source boundary, then all common gates |

All mutation/import recipes require `policy.roi` as world-coordinate `min` and
`max` vectors. Use bounds measured from the actual mesh, never guessed normalized
coordinates. The worker bakes GLB scene transforms and keeps that common frame;
it does not recenter or rescale each candidate independently.

The following checks cannot be disabled in gate v1: closed edges, vertex manifold
status, self-intersection freedom and preservation outside the edit region.
Protected component IDs come from the source diagnostics. The foundation policy
also specifies its occupied region, the up-axis, a bounded sample-grid size and
surface tolerance. A foundation union is not an automatically chosen primitive;
the agent must author it from reference and geometry evidence.

Minimal diagnostic recipe:

```json
{"method": "analyze", "parameters": {}, "policy": {}}
```

For an actual repair, obtain the permitted fields and current gate version from
`get_shape_repair_capabilities` and the validated recipe contract. Do not submit
arbitrary scripts, executable code or parameters for unsupported global repair.

## What the machine gate establishes

The gate inspects the actual serialized GLB after reloading it, including float32
export rounding. It checks edge use and direction, collapsed/duplicate faces,
vertex-manifold status, self-intersections, protected components and preservation
outside the declared edit region. The source remains hash-verified and unchanged.
Unavailable or failed checks keep the gate closed; missing measurements are not
interpreted as zero defects.

If an occupied foundation region is declared, sampled intervals test that region
only after the solid-topology prerequisites pass. This is finite sampling, not
a proof of every point in the volume. A watertight bowl can still have the wrong
foundation. The agent must inspect underside/peripheral cavities, intentional
openings, silhouettes, small details and reference identity separately.

Attempt artifacts include durable `repair-attempt.json`, resolved
`repair-worker-input.json`, the candidate output, `repair-report.json`,
`diagnostics.json`, `machine-gate.json`, stage `gate.json`, previews and
resource-control evidence. Source/candidate SHA-256 values bind the result to those specific files.
Failed attempts remain available for diagnosis; passing counts from another
attempt or an in-memory mesh cannot approve this candidate.

## CPU/GPU and resource boundaries

Core topology and preservation validation runs in an isolated worker using
bounded CPU checks; explicitly selected CuMesh candidate operations use CUDA.
The caller holds the same cross-process compute lease used by Hunyuan, Paint, rendering and
remeshing. It enforces at most two logical CPUs, below-normal priority, at most
2048 MiB per process and at least 2048 MiB free-memory reserve. The configured
timeout is bounded. This is not permission to run repair concurrently with GPU
shape generation: heavy phases remain serial even when they use different devices.

Optional packages load from a separate repair package directory configured by
`CODEX_HUNYUAN_MV_REPAIR_PACKAGES`. The worker uses the existing Hunyuan Python
runtime without installing over its Torch or other dependencies. No hidden
unrestricted CPU fallback, implicit runtime resource-ceiling increase, decimation
or global voxel remesh is part of this repair contract. The separately authorized
native build cap increases described below do not change normal repair limits.

## Verification scope

CPU regression tests exercise the contracts, source immutability, selection validation,
quality gates and accepted-master state transitions. Native GPU repair requires an
installed and currently qualified runtime; see [CuMesh repair](cumesh-repair.md).
Unit tests and a successful native probe do not constitute visual asset approval.
