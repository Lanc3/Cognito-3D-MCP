---
name: codex-hunyuan-multiview-3d
description: Default local image-to-3D workflow using Hunyuan3D-2mv, three AutoRemesher output profiles, Paint, and GPU texture baking. Create serial asset batches, gate four-view cutouts, review and repair every profile, and deliver textured models through codex_3d_models_hunyuan_mv.
---

# Cognito-3D-mcp: Hunyuan asset batches

Use `codex_3d_models_hunyuan_mv` for normal image-to-3D work. The calling Codex
agent owns the entire generation, inspection, repair, and retry loop. The local
server runs deterministic checks and compute; it does not call an image generator
or an independent LLM. Keep working through `get_agent_work` until the batch is
completed or the user pauses/cancels it. Resource or installation faults require
repair, not bypassing gates or repeatedly executing unchanged failed work.

## Starting in another project

Call `server_status` before creating a batch. This quad-profile workflow requires
server version 0.4.0 or newer, runtime and remesher readiness, and the serial
execution policy. If the connection reports an older version after installation,
reload the MCP connection or restart Codex before starting work. Do not use the
older connection for a new batch. Check `get_queue_status` for earlier active
work; do not resume or cancel another project's batch without authorization.

The MCP is configured globally and keeps outputs in its configured shared output
directory. Reference input paths must be under a root reported by `server_status`.
For a project outside those roots, copy only its selected reference PNGs into a
unique batch/asset subdirectory of the already allowed
input directory reported by `server_status` (normally `inputs` in the configured data directory or the current user's `.codex/generated_images` directory), preserving the originals,
then submit those absolute paths. Do not broaden access to an entire drive.
Generate or stage all references before compute begins. After all gates pass,
copy the accepted deliverables into the requesting project's asset directory.

The queue's **View in 3D** selector shows each available textured profile and the
original untextured Hunyuan shape, repair attempts, and accepted master. Previous versions are labeled explicitly. It reports measured cleaned OBJ quad counts
separately from targets and exported GLB triangles. Drag to rotate, scroll to
zoom, or right-drag to pan. Use this for inspection alongside the fixed views.

## Strict phase order

1. Create a batch with `create_generation_batch(name, assets)`. Each asset has
   `name`, `prompt`, and optional `seed`, `quality`, `material_hints`, and named
   source `views`. This opens the local queue UI. Register the whole asset list
   before beginning image creation, including a task with 100 assets.
2. Finish **all** reference generation and reference gates for the active batch.
3. Generate **all** shapes sequentially with one resident GPU shape model.
4. Unload shape. Run shape_repair diagnosis, create typed repair candidates as needed, inspect all seven views and the repair report, and accept a master only after machine gates and agent review pass for every asset.
5. For each asset, independently remesh its accepted shape into **full game,
   mobile, and browser** meshes. Run every profile of every asset sequentially.
   Inspect all three profiles and their four views; repair until all pass.
6. Paint **all full-game meshes** sequentially with one resident GPU Paint model;
   unload it before preview rendering and review. Do not run a separate generative
   Paint pass for mobile or browser.
7. After every Paint gate/review passes, finish each asset sequentially: unwrap
   mobile, GPU-bake the full-game appearance to it, then do the same for browser.
   Apply shared final placement, export, and inspect all twelve profile previews
   and QA evidence. Preserve the accepted topology; do not decimate at finish.

Never run assets or compute phases in parallel, including other 3D servers or a
later batch's image preparation. Hunyuan and Paint use the GPU without CPU offload.
Texture baking also requires the GPU with CPU devices disabled. AutoRemesher is
CPU-only: the runtime enforces at most two logical CPUs, reduced
priority, a memory cap, and free-memory reserves. Do not remove these guards or
raise resource ceilings to make a gate pass. A live previous worker must exit
before its resource lease is released. Pause/cancel on the user's request.

## Generate and gate references

Read `get_agent_work` for the active batch. For every asset create front, back,
left, and right views of the same object: matching square dimensions, scale,
center, camera elevation, pose, lighting, and object identity. Prefer 1024px or
larger, orthographic appearance, complete silhouette, and generous empty margins.
No floor, stand, cast shadow, labels, extra objects, or three-quarter camera view.
Generate one consistent turnaround or derive additional views with referenced
edits. Inspect orientation before splitting a sheet into lossless PNGs.

Request genuine transparent alpha from image generation. If a reserved chroma
background is used, make it a perfectly uniform high-contrast color absent from
the asset, normally magenta `#FF00FF`; pass `background_mode=chroma` and its RGB
`key_color`. Do not mistake a painted checkerboard for transparency. Never pass
an opaque image directly to geometry generation.

Submit four paths with `submit_asset_references`. The script preserves originals,
removes a declared key deterministically, and checks the full transparent margin,
foreground area, cropping, dimensions, and duplicate views. Inspect its prepared
RGBA PNGs and alpha masks. Pixel checks cannot establish object identity, correct
view orientation, or absence of background painted inside the silhouette.

If a reference fails, inspect the report and regenerate or edit the offending
image, then resubmit the four paths. Reuse good references. Do not keep submitting
the same unchanged failing image. Shape uses all four prepared cutouts; Paint
uses that same prepared front PNG, avoiding a second background-removal path.

## Shape repair candidates (0.4)

Raw shape completion is not production approval. `shape_repair` first analyzes
an immutable copy; it does not silently clean the raw mesh. Call
`get_shape_repair_capabilities` and read the current gate, report, source hashes,
component IDs and boundary-loop IDs before selecting an operation.

Use `create_shape_repair_attempt` with the measured source SHA256, expected raw
shape attempt, diagnosis, a unique idempotency key, and a typed recipe. To build
on a prior candidate, supply its `source_attempt` and SHA256. Methods are
`analyze`, `conservative_cleanup`, `patch_selected_holes`,
`remove_selected_components`, and `foundation_union`. Imported externally
repaired GLBs use `submit_shape_repair_candidate` and `import_candidate`.
Stage an external candidate beneath a reported allowed input root before
submission, preserve the original, and provide its exact `candidate_sha256`.
Every edit/import declares `policy.roi` in baked GLB world coordinates and any
protected component IDs. Do not infer these coordinates from normalized previews.

Cleanup removes exact duplicate/zero-area faces by default; tolerance welding
requires explicit bounded displacement. Patch only diagnosed accidental loops
with physical size/area limits. Remove only identified debris; small antennas,
barrels and vents can be legitimate components. A foundation union needs valid
closed manifold input plus an authored overlapping solid and an occupied-volume
policy. Contradictory references or malformed semantic parts need corrected
references/regeneration or an authored replacement; a topology repair cannot
resolve their identity.

The gate reloads the exported candidate and checks closed edges, winding,
degenerate/duplicate faces, vertex manifoldness, self-intersections, outside-ROI
preservation and protected parts. For a required solid foundation, define its
region and height interval: watertightness alone permits a hollow shell or bowl.
Foundation samples are evidence, not a proof of every point in the volume.
Unavailable required checks fail closed. Read CuMesh's current capability proof;
package discovery or a successful build does not establish CUDA qualification.
Prefer an applicable GPU repair method for routine work only when that method
reports `production_ready` from successful common-worker qualification. An
experimental CUDA-qualified method can produce a test candidate, but still needs
every common gate and separate review. GPU failure must not silently fall back to
the CPU. Use the discovered recipe contract for supported parameters and limits.

Inspect front/back/left/right, bottom, underside_front_left and
underside_back_right, plus the repair report. Call `record_asset_review` for the
exact `shape_repair` attempt and these inspected paths. Only this acceptance
creates the master consumed by remeshing. Failed candidates and originals remain
available. A green geometry gate is not visual approval. Continue with a changed,
applicable repair until accepted; an unresolved defect stays awaiting repair.
Keep normal repairs within two CPUs, a 2 GiB process cap and 2 GiB free-memory
reserves. Resource changes require explicit user direction; a separately
authorized higher compilation cap does not change normal repair limits. Never
overlap compute phases.

When a lower profile target changes, matching approved full-game Paint can be
retained; finish/bakes are invalidated. Unchanged remesh profiles can be reused
only with matching source, settings, implementation and output hashes. Legacy
batches retain outputs but need the new master certificate before new processing.
## Three remesh profiles

Build each profile independently from the same hash-bound accepted master. The raw Hunyuan shape remains immutable. Do not
derive mobile from the full-game mesh or browser from mobile. Preserve the source
shape hash, native quad OBJ, measured geometry, and gates for each profile.

The alien fighter test uses these starting values:

| Profile | Target quads | Maximum exported triangles | Texture |
| --- | ---: | ---: | --- |
| Full game | 50,000 | 100,000 | Paint, 2048px |
| Mobile | 25,000 | 50,000 | GPU bake, 1024px |
| Browser | 10,000 | 20,000 | GPU bake, 1024px |

Use `full_game_target_quads`, `mobile_target_quads`, and `browser_target_quads`
with corresponding `*_triangle_budget` parameters. The legacy `target_quads`
parameter aliases full game. A target guides density; it does not guarantee the
native quad count or exported triangle count. Report the requested target, actual
cleaned quads, raw native quads when different, and actual exported triangles.
Triangle ceilings are upper limits, not minimum densities or universal platform
requirements. If a profile exceeds its triangle ceiling, adjust that profile's
remesh target and retry; do not introduce a hidden decimation fallback.

Lower targets can remove thin wings, small parts, or important curvature. Treat
better results than decimation as something to demonstrate through preservation
gates and visual evidence. Do not claim that a quad-dominant result automatically
has better game geometry.

## Agent review and repair loop

Use `get_agent_work` or `get_batch_status` to inspect the current stage/attempt.
For `inspect_and_review`, open the provided previews and reports. Check all four
directions against the references, looking for background slabs, missing parts,
silhouette changes, bad seams, wrong colors, or lost details. For remesh and
finish, inspect all three profiles from all four directions, not only full game.
For remesh, also inspect each profile's shape-preservation/component gates,
actual triangle count, and native quad OBJ when needed. For finish, inspect the
GPU bake coverage and texture-transfer evidence as well as all exported meshes.

Call `record_asset_review` with the exact stage and attempt, `approved`, concrete
notes, and absolute `inspected_paths` belonging to that attempt. Never approve
without inspection. A machine gate failure cannot be overridden by agent review.

For `inspect_repair_retry`, diagnose the causal stage. Call `retry_asset_stage`
with a repair rationale and supported changes (for example a new shape seed,
quality preset, a profile's quad target/sharp-edge/adaptivity settings, or atlas
resolution).
For references, resubmit corrected images after rewinding when necessary. For a
runtime fault, correct the reported dependency/resource cause before retrying.
An upstream repair invalidates that asset's downstream work while preserving
successful other assets. Old attempts and their evidence remain available.

There is no attempt-count escape: continue repair and review until the asset
passes, rather than waiving a check or delivering a failed candidate. Do not
turn resource exhaustion into a blind retry loop; expose the blocking condition
and keep the candidate awaiting repair until resources or inputs change.

## Outputs and limitations

The path is four reference cutouts → raw shape GLB → repair candidates → accepted master → three independent AutoRemesher
meshes → Paint full game once → GPU-bake mobile/browser → export and review. Every
batch completes the whole current phase before starting the next. Native OBJs
preserve quad topology; GLB triangulates. AutoRemesher owns topology cleanup and
profile density. Blender unwraps and bakes lower profiles, applies the same final
placement transform to all profiles, and renders each variant in isolation. No
final Blender decimation is part of this pathway.

The configured Paint adapter still uses the prepared front image for appearance;
all four views constrain geometry. Paint returns a color texture, not a full
generated metallic/roughness/normal-map set. Prompts/material hints are metadata.
GPU baking transfers the full-game base-color appearance to each lower-density
mesh without new AI painting. The coverage gate requires at least 99.5% coverage
of expected UV texels using projected/self white masks, so genuinely black paint
does not count as missing texture. Visual review must still catch projection
errors and inconsistent details that a coverage percentage cannot detect.

Deliver completed `full_game.glb`, `mobile.glb`, and `browser.glb`, their native
remesh OBJs, all twelve previews, `working.blend`, profile manifest, bake evidence,
and QA report. Compatibility exports `master.glb`/`game.glb`, `lod1.glb`, and
`lod2.glb` are byte copies of full game, mobile, and browser respectively. Use
`get_generation_artifacts`. Legacy jobs remain readable; new processing uses
the batch scheduler so an old queue cannot overlap the new one.
