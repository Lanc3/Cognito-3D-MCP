# Hunyuan batch pipeline: three remeshed profiles

The agent creates and repairs images; the server executes gated, strictly serial
phases. Register every asset first. No shape work starts until all four cutouts
for every asset have passed machine checks and agent inspection.

Version 0.4 adds [immutable shape repair candidates](shape-repair.md), seven-view
review, and a hash-bound accepted master. Original raw shapes remain preserved.
The three output profiles are built independently from the same accepted master,
before painting. Paint creates one full-game appearance; GPU baking transfers
that appearance to mobile and browser meshes. Finishing does not decimate meshes.

```mermaid
sequenceDiagram
    participant Agent as Codex agent
    participant Queue as Durable batch queue
    participant Shape as GPU Shape
    participant Repair as Bounded repair worker
    participant Remesh as Limited CPU AutoRemesher
    participant Paint as GPU Paint
    participant Finish as GPU bake / export / QA
    Agent->>Queue: Create batch with all assets (UI opens)
    loop Each asset, repair until references pass
        Agent->>Agent: Generate four views, inspect background
        Agent->>Queue: Prepare cutouts and check pixel gates
        Queue-->>Agent: Prepared PNGs / masks / gate evidence
        Agent->>Queue: Review or submit corrected references
    end
    Note over Queue,Shape: Barrier: ALL reference gates and reviews pass
    Queue->>Shape: Load one resident GPU model
    loop Every asset, one at a time
        Queue->>Shape: Generate one raw shape
        Shape-->>Queue: Saved shape and metrics
    end
    Queue->>Shape: Shutdown and join worker; release memory
    loop Each asset, diagnose and repair until accepted
        Queue->>Repair: Immutable source hash and typed recipe
        Repair-->>Queue: Candidate, reloaded geometry gates, report
        Queue-->>Agent: Seven views, including bottom and underside
        Agent->>Queue: New applicable repair or accept exact candidate
    end
    Note over Queue,Remesh: Barrier: ALL masters pass geometry AND agent review
    loop Every asset, one at a time
        loop Full game, mobile, browser — sequentially
            Queue->>Remesh: Same accepted master, profile quad target
            Remesh-->>Queue: Native quad OBJ, triangle GLB, preservation gates
        end
    end
    Queue-->>Agent: All three meshes, actual counts, twelve views; repair/review
    Note over Queue,Paint: Barrier: ALL three profiles for ALL assets pass
    Queue->>Paint: Load one resident GPU Paint model
    loop Every asset, one at a time
        Queue->>Paint: Full-game remesh plus prepared front PNG
        Paint-->>Queue: Textured full-game mesh
    end
    Queue->>Paint: Shutdown and join worker; release memory
    Queue-->>Agent: Paint evidence; repair/review loop
    Note over Queue,Finish: Barrier: ALL Paint gates and reviews pass
    loop Every asset, one at a time
        Queue->>Finish: GPU bake full-game appearance onto mobile mesh
        Queue->>Finish: GPU bake full-game appearance onto browser mesh
        Queue->>Finish: Shared placement, export, serialized geometry checks, twelve previews, QA
        Finish-->>Agent: Three textured profiles and evidence for final review
        Agent->>Queue: Repair/retry or approve
    end
    Queue-->>Agent: Complete only when every asset passes
```

## Ownership and restart behavior

- `batch.py` owns batch state, stage barriers, reviews, invalidation, attempt
  history, and the single worker thread. Its atomic `batch.json` survives restart.
- `backgrounds.py` prepares shared RGBA cutouts and fails unknown opaque images.
- `runtime.py` / `worker.py` retain one Hunyuan model for a compute phase. A failed
  shutdown retains the resource lease. Batches force Paint GPU mode.
- `remesh.py` runs the pinned native tool, enforces Windows CPU affinity/job
  memory limits, and validates fresh outputs. It never trusts exit code alone.
- `stage_processor.py` runs the stages and unloads models before evidence renders.
- `blender_worker.py` unwraps and bakes the accepted mobile/browser meshes on the
  GPU, applies shared final placement, and exports them without decimation.
- `dashboard.py` serves an ephemeral loopback UI with queue state and local
  artifacts. It never loads models or launches an LLM.

Restarted work is paused/interrupted. The agent reviews retained attempts and
explicitly retries incomplete work; completed stage approvals are preserved.
Changing a reference or shape invalidates that asset's later stages. Other
assets' passing stages stay checkpointed. Batches retain their submission order.

The active Codex task is the repair agent. The MCP does not secretly launch a
second agent or call an image-generation API. If that task is stopped, the queue
waits for it to resume; saved MCP instructions and the updated skill tell the
agent to keep driving `get_agent_work` until completion.

## AutoRemesher provenance

The linked [Lanc3 repository](https://github.com/Lanc3/autoremesher) at commit
`d9ef96bd72f0b134dd7e51acf5904f32a5679704` is identical to the upstream 1.2.0 tag.
The portable [Windows 1.2.0 release](https://github.com/huxingyi/autoremesher/releases/tag/1.2.0)
can therefore supply the native executable without a Qt/compiler build.

- Archive SHA-256: `f6184622cef84f0bcf032a0474df44e2076f4c10c159f1af0cb7230259af7476`
- Executable SHA-256: `103bd621bfe828d4f1fbc62ace095483eeb184e29e647e739894c3b9cc93e877`
- License: MIT; preserve the release's bundled notices and license files.
- Native interchange: OBJ → OBJ. Keep native quad OBJ; GLB is triangulated.
- CPU only: at most two logical CPUs, below-normal priority, per-process memory
  cap, job-tree termination, memory/commit reserves, timeout and cancellation.
- Fighter test defaults: 50k / 25k / 10k target quads for full game / mobile /
  browser, with 100k / 50k / 20k actual triangle limits respectively.
- Resource defaults: 2GiB CPU process cap, at least 2GiB available physical/commit
  reserve. A smaller cap is appropriate for small test fixtures. Each profile
  remains a separate serial native invocation under the same guards.

| Profile | Requested target quads | Exported triangle ceiling | Appearance |
| --- | ---: | ---: | --- |
| Full game | 50,000 | 100,000 | Paint once, 2048px atlas |
| Mobile | 25,000 | 50,000 | Bake from full game, 1024px atlas |
| Browser | 10,000 | 20,000 | Bake from full game, 1024px atlas |

These are the user-approved fighter test budgets, not universal platform limits.
`--target-quads` is a density request, not a guaranteed output count. A quad
usually triangulates into two triangles, but mixed faces and cleanup affect the
result. Gate the measured exported triangle count. The dashboard reports requested
quads, measured cleaned quads, raw native quads when different, and actual export
triangles separately. For repairs, change the relevant profile target rather than
silently decimating an over-budget result.

Profile parameters are `full_game_target_quads`, `mobile_target_quads`, and
`browser_target_quads`, with corresponding `*_triangle_budget` parameters. The
legacy `target_quads` field aliases the full-game request. The profile manifest
records each output and the shared source shape hash. Avoid chained reductions
from full game to mobile to browser; their geometry errors would accumulate.

Remesh gates check finite/nonempty mesh, topology, quad ratio, face budget,
significant components, bounds, area, and bidirectional sampled surface deviation.
The distance estimator uses nearby candidate triangles; it is not an exact
Hausdorff certificate. Agent visual review remains mandatory.

## Meaning of the gates

The pixel gate prevents a full opaque canvas from reaching Hunyuan. Genuine alpha
or a declared, uniformly keyed border is required. It cannot prove that an opaque
rectangle inside an otherwise transparent image is part of the object; the agent
must inspect all prepared views.

Geometry gates establish structural and preservation constraints, not subjective
likeness. The agent supplies the visual decision. A failure remains repairable
without a fixed retry count. Memory pressure, missing dependencies or failed
worker termination block execution instead of triggering CPU fallback or overlap.

Paint receives the prepared front reference. The current adapter generates a
color atlas; generated PBR material maps and multi-reference Paint are separate
future changes. Atlas resolution is now forwarded to the actual Paint renderer;
its internal diffusion view resolution remains upstream's 512px.

The pinned custom Paint UNet uses an owned-memory loader. It constructs the
identical architecture on the meta device, loads the half-precision checkpoint
once with `mmap=False`, and strictly assigns all weights before CUDA transfer.
This avoids the upstream loader's temporary float32 dual-model allocations.
The loader restores its scoped method replacement and records memory stages,
checkpoint identity, zero residual meta tensors, and CUDA placement. It never
enables CPU offload or weakens the physical-memory/commit reserve.

Paint runs once per asset on its accepted full-game mesh. After every asset's
Paint worker has exited and the Paint gates/reviews pass, finishing unwraps each
lower-density mesh and bakes the same base-color appearance onto it sequentially.
The bake requires a CUDA/OptiX GPU device with CPU devices disabled. It transfers
appearance without another generative Paint pass, keeping profile colors and
markings consistent. It does not generate a full PBR texture set.
The lower profiles also retain the source material's constant metallic,
roughness, IOR, alpha and backface-culling settings. Linked material maps require
an explicit material-map transfer instead of silently substituting constants.

Bake coverage must reach 99.5% of expected UV texels. Coverage is measured using a
projected white source mask against a white target self-mask; legitimate black
paint is not mistaken for a missing projection. The agent still inspects all
twelve profile previews for projection artifacts, texture seams, lost details,
and silhouette damage. A structurally valid bake can still fail visual review.

Blender preserves each accepted remesh topology, applies shared final placement,
and exports `full_game.glb`, `mobile.glb`, and `browser.glb`. Compatibility files
`master.glb` and `game.glb` are byte copies of full game; `lod1.glb` and `lod2.glb`
are byte copies of mobile and browser. They do not create additional geometry or
run a second reduction. Retain the three native quad OBJs, profile manifest,
texture-bake evidence, `working.blend`, previews, and QA reports. Triangle budgets
are upper limits, not required minimum densities.

The UI consumes scoped `full_game_front/back/left/right`, `mobile_*`, and
`browser_*` preview entries and displays every available safe preview key. All
three profile previews and measured counts are reviewable before the queue can
advance. A missing or failed profile keeps the asset at the current gate.

## Interactive 3D review

Choose **View in 3D** on an asset's queue card. The selector offers **Full game**,
**Mobile**, **Browser**, and **Original Hunyuan** as those models become available.
Finished profiles display their saved textures and materials. Before finishing,
full game can display its Paint output while lower profiles remain explicitly
untextured. Original Hunyuan shows the untextured shape before remeshing or Paint.

The displayed quad count is the measured count in the cleaned AutoRemesher OBJ,
before GLB triangulation; it is separate from the requested quad target. The
triangle count describes the loaded GLB. Original Hunyuan is a triangle source,
so its native quad count is shown as an em dash, never inferred by halving its
triangle count.

Left-drag to rotate, use the mouse wheel to zoom, and right-drag to pan.
**Reset view** restores the starting camera. The optional wireframe overlay helps
inspect geometry without replacing the saved materials.

The viewer uses locally bundled, MIT-licensed Three.js; it requires no CDN.
Rendering happens on demand when the view changes. Model switches load serially,
and only one model stays loaded; closing the viewer releases its graphics
resources. Viewing does not regenerate an asset or alter its gate approval.
