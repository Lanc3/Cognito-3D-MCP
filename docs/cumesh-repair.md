# Optional CuMesh candidate repair

CuMesh is an optional GPU adapter for diagnosis, exact cleanup, and narrowly
selected planar-hole repair. It exports a candidate in the original world frame
for the same common geometry and preservation gates as CPU repair. It does not
approve shapes, infer intended foundations, or start downstream stages.

The normal installer includes CPU repair dependencies. CuMesh needs a separate
native build and qualification; its absence must be reported as an unavailable
optional capability rather than silently substituted for requested GPU repair.

## Setup

Use the configured Hunyuan Python runtime and a separate repair package directory.
Review `scripts/setup-cumesh-repair.ps1` parameters for explicit locations. Run
metadata preflight, `-Build`, and `-Probe` separately. Do not run other GPU jobs
while building or probing. A matching PyTorch/CUDA compiler stack and Visual
Studio 2022 C++ environment are required.

Source pins are CuMesh `12289e1062f0603f2f0d0771b02e1395d247f26f`, cubvh
`ce92267a24ef6ad7d2c8ccbc2ae2c021a6597e70`, and Eigen
`e63d9f6ccb7f6f29f31241b87c542f3f0ab3112b`. Preserve their licence files.
The build records source, wheel and native-extension hashes; the Python package
version alone is insufficient provenance.

Build and probe use the configured shared GPU lock, bounded CPU affinity and
memory, and retained logs. The default process memory cap is 2048 MiB. A
`-QualificationBuildMemoryMiB` override is build-only, bounded to the script's
supported values, and requires a unique run name; it never changes the runtime
cap. Check the actual failure before increasing a cap. Do not treat a timeout
as evidence that more memory is required.

## Candidate operations

- `diagnose` preserves geometry and reports native defect measurements.
- `cleanup` removes exact duplicate and zero-area faces with deterministic
  comparisons against the CPU reference. Original world coordinates are retained.
- `fill_selected_planar_holes` binds selection to the source hash and actual edge
  sets. It accepts simple, planar, strictly convex loops only when the requested
  set matches the native threshold selection. Original vertices and faces remain
  unchanged; new patch faces must have consistent boundary winding.

Ambiguous selections, stale hashes, non-planar or concave loops, unintended
holes, and changed source geometry are rejected. UDF remeshing is disabled:
reconstruction of intended solid volume requires an authored repair contract.

## Readiness and delivery

Inspect `get_shape_repair_capabilities`, the current source/gate report, and
code-bound qualification evidence before requesting this adapter. A successful
build or tiny CUDA probe establishes only the operations it exercised. Every
candidate still needs serialized export/reload checks, topology and preservation
gates, any authored foundation requirement, and agent visual approval.

CPU tests can validate selection and state contracts without loading CUDA.
Native qualification and real-asset visual acceptance are separate checks.
The immutable raw GLB and rejected candidates remain available for comparison.

Sources: [CuMesh](https://github.com/JeffreyXiang/CuMesh),
[pinned native cleanup](https://github.com/JeffreyXiang/CuMesh/blob/12289e1062f0603f2f0d0771b02e1395d247f26f/src/clean_up.cu),
[pinned build configuration](https://github.com/JeffreyXiang/CuMesh/blob/12289e1062f0603f2f0d0771b02e1395d247f26f/setup.py).
