# Bidirectional production operations

## Runtime identity

- Default reconstruction: `stabilityai/stable-point-aware-3d` in low-VRAM mode
- trellis.cpp v0.5.4, commit `ae1a63757264eec5bfba84b94cf59ddcc161e537`
- Windows CUDA archive SHA-256 `f7d2912b064bf1520f03e025c5eb344df6b347ad04831ac7aa04d847581bd7ad`
- `ilintar/trellis2-gguf` Q8 weights at revision `a57397bd3d351599d9729fc144b3f87c3f87d65b`
- DINOv2-small revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`
- Real-ESRGAN NCNN/Vulkan v0.2.5.0
- Khronos glTF Validator 2.0.0-dev.3.10
- Blender installed and reported by `server_status`

`server_status` verifies the selected reconstruction backend, complete local model cache, DINO cache, executables, Blender, and free disk space. Generation readiness is false if any production dependency is absent.

## Job storage and recovery

Jobs live under `outputs/trellis/<job_id>` and SQLite state lives at `outputs/trellis/jobs.sqlite3`. Every expensive stage writes a checkpoint with artifact hashes. A restart marks queued/running jobs as `interrupted`; resume reuses existing raw GLBs and build results.

Failed jobs are never pruned automatically. `cleanup_generation_job` is the only cleanup path and reports every irreversibly removed path.

## GPU and disk safety

All local 3D servers share the configured `CODEX_3D_GPU_LOCK` path. The installer
places it under the selected runtime root. The resident SPAR3D child keeps the
lock for its sequential view session; Real-ESRGAN takes it only for its
operations. Blender baking in this bidirectional pipeline uses CPU Cycles.

Setup and generation require at least 20GB free on the output drive. The production server refuses silent 512 fallback or 1536 generation on the 16GB hardware profile.

## Failure repair

Raw reconstruction GLBs are welded at coincident glTF seam vertices and measured before
upscaling or fusion. A source with more than 250 connected components and less
than 5% of both surface area and faces in its largest component fails as
`SOURCE_FRAGMENTATION`. The raw reconstructions and neutral diagnostic renders
remain available; no downstream mesh is published.

Seed references are background-matted in their original camera frame before
Real-ESRGAN. The foreground mask is retained separately and unknown pixels use a
neutral projection fallback so removable background colours cannot enter the 4K
atlas.

UV layout uses exact pixel margins rather than Blender's scaled-margin mode. QA
measures normalized UV triangle occupancy and rejects layouts below 65%, even
when Blender baking and glTF validation otherwise succeed.

- Pair rejection: regenerate the inconsistent reference and submit a new job.
- Pair warning: inspect metrics and approve only if both images depict the same asset.
- Side disagreement: generate left/right views only after user approval and call `add_side_references`.
- Operational failure: correct the runtime condition and call `resume_generation`.
- Quality failure: inspect `qa/report.html`, previews, alignment metrics, raw meshes, and `working.blend`; do not distribute the failed GLBs as production assets.

Run the complete software and Blender verification suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$env:RUN_BLENDER_INTEGRATION = "1"
.\.venv\Scripts\python.exe -m pytest -q tests\test_blender_worker_integration.py
.\.venv\Scripts\python.exe -m ruff check --select E9,F src tests scripts
```
