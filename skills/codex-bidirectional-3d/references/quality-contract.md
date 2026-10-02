# Production quality contract

Use this reference to explain validation failures and choose the correct recovery action.

## Seed pair

- Square PNG, minimum 1024x1024.
- Aspect difference at most 1%.
- Object-center difference at most 3% of the frame.
- Object-height ratio 0.90-1.10.
- Foreground-area ratio 0.75-1.25.
- At least 2% clear border.
- Local DINOv2 cosine similarity at least 0.78.
- No duplicates, mirrored backs, clipping, or corrupt images.

Warnings require review. Structural errors require regeneration.

## Production outputs

- `master.glb`: approximately 300,000 faces and watertight.
- `game.glb`: approximately 100,000 faces.
- `lod1.glb`: approximately 50,000 faces.
- `lod2.glb`: approximately 20,000 faces.
- `baseColor.png`, `normal.png`, and `orm.png`: 4096x4096 PNG.
- Optional `emissive.png` and `opacity.png` when detected or requested.
- Root transforms baked; glTF +Y up and object front toward +Z.

## Hard failure behavior

Any geometry, texture, export, or glTF Validator failure produces `failed_quality`. Preserve raw view meshes, seed images, masks, metrics, logs, `working.blend`, previews, checkpoints, and QA reports. Never replace a failed composite with a single-view fallback.

Use `resume_generation` for operational interruption. Use a new or repaired seed pair for structural mismatch. Add side views only after user approval.
