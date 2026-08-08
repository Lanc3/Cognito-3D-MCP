# State and Side Effects

Generated: 2026-08-03T16:08:21.126845+00:00

## Confidence
Medium

## Source Files Referenced
- `README.md`
- `api_server.py`
- `assets/modelviewer-textured-template.html`
- `blender_addon.py`
- `docs/source/conf.py`
- `docs/source/started/code.md`
- `gradio_app.py`
- `hy3dgen/rembg.py`
- `hy3dgen/shapegen/models/autoencoders/attention_blocks.py`
- `hy3dgen/shapegen/models/autoencoders/model.py`
- `hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`
- `hy3dgen/shapegen/pipelines.py`
- `hy3dgen/shapegen/postprocessors.py`
- `hy3dgen/texgen/custom_rasterizer/lib/custom_rasterizer_kernel/grid_neighbor.cpp`
- `hy3dgen/texgen/hunyuanpaint/pipeline.py`

## Claim Labels
- CONFIRMED: directly supported by source files, manifests, tests, or configs.
- INFERRED: likely from framework conventions, naming, or partial static evidence.
- UNKNOWN: not enough evidence yet.

## Side Effect Candidates
| Kind | File | Line | Evidence |
|---|---|---|---|
| database-write | README.md | 75 | The shape generative model, built on a scalable flow-based diffusion transformer, aims to create geometry that properly |
| database-write | README.md | 202 | The output mesh is a [trimesh object](https://trimesh.org/trimesh.html), which you could save to glb/obj (or other |
| database-write | api_server.py | 222 | with tempfile.NamedTemporaryFile(suffix=f'.{type}', delete=False) as temp_file: |
| database-write | assets/modelviewer-textured-template.html | 103 | appearanceButton.classList.remove('checked'); |
| database-write | assets/modelviewer-textured-template.html | 123 | geometryButton.classList.remove('checked'); |
| database-write | blender_addon.py | 146 | temp_glb_file = tempfile.NamedTemporaryFile(delete=False, suffix=".glb") |
| network-call | blender_addon.py | 196 | response = requests.post( |
| network-call | blender_addon.py | 209 | response = requests.post( |
| network-call | blender_addon.py | 232 | response = requests.post( |
| network-call | blender_addon.py | 244 | response = requests.post( |
| database-write | blender_addon.py | 264 | temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".glb") |
| database-write | docs/source/conf.py | 16 | sys.path.insert(0, os.path.abspath(".")) |
| database-write | docs/source/conf.py | 17 | sys.path.insert(0, os.path.abspath("../../")) |
| database-write | docs/source/started/code.md | 16 | The output mesh is a [trimesh object](https://trimesh.org/trimesh.html), which you could save to glb/obj (or other |
| file-write | gradio_app.py | 117 | with open(output_html_path, 'w', encoding='utf-8') as f: |
| database-write | gradio_app.py | 203 | time_meta['remove background'] = time.time() - start_time |
| database-write | gradio_app.py | 208 | time_meta['remove background'] = time.time() - start_time |
| database-write | gradio_app.py | 299 | gr.update(value=path), |
| database-write | gradio_app.py | 300 | gr.update(value=path_textured), |
| database-write | gradio_app.py | 346 | gr.update(value=path), |
| database-write | gradio_app.py | 440 | check_box_rembg = gr.Checkbox(value=True, label='Remove Background', min_width=100) |
| database-write | gradio_app.py | 521 | tab_ip.select(fn=lambda: gr.update(selected='tab_img_gallery'), outputs=gallery) |
| database-write | gradio_app.py | 523 | tab_tp.select(fn=lambda: gr.update(selected='tab_txt_gallery'), outputs=gallery) |
| database-write | gradio_app.py | 544 | lambda: (gr.update(visible=False, value=False), gr.update(interactive=True), gr.update(interactive=True), |
| database-write | gradio_app.py | 545 | gr.update(interactive=False)), |
| database-write | gradio_app.py | 548 | lambda: gr.update(selected='gen_mesh_panel'), |
| database-write | gradio_app.py | 571 | lambda: (gr.update(visible=True, value=True), gr.update(interactive=False), gr.update(interactive=True), |
| database-write | gradio_app.py | 572 | gr.update(interactive=False)), |
| database-write | gradio_app.py | 575 | lambda: gr.update(selected='gen_mesh_panel'), |
| database-write | gradio_app.py | 581 | return gr.update(value=5) |
| database-write | gradio_app.py | 583 | return gr.update(value=10) |
| database-write | gradio_app.py | 585 | return gr.update(value=30) |
| database-write | gradio_app.py | 591 | return gr.update(value=196) |
| database-write | gradio_app.py | 593 | return gr.update(value=256) |
| database-write | gradio_app.py | 595 | return gr.update(value=384) |
| database-write | gradio_app.py | 630 | return model_viewer_html, gr.update(value=path, interactive=True) |
| database-write | gradio_app.py | 633 | lambda: gr.update(selected='export_mesh_panel'), |
| database-write | hy3dgen/rembg.py | 16 | from rembg import remove, new_session |
| database-write | hy3dgen/rembg.py | 24 | output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0]) |
| database-write | hy3dgen/shapegen/models/autoencoders/attention_blocks.py | 255 | logger.info('Save kv cache,this should be called only once for one mesh') |
| database-write | hy3dgen/shapegen/models/autoencoders/model.py | 109 | model_kwargs.update(kwargs) |
| database-write | hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py | 41 | Create sinusoidal timestep embeddings. |
| database-write | hy3dgen/shapegen/pipelines.py | 130 | kwargs.update(params) |
| database-write | hy3dgen/shapegen/pipelines.py | 194 | model_kwargs.update(kwargs) |
| database-write | hy3dgen/shapegen/pipelines.py | 418 | hook.remove() |
| database-write | hy3dgen/shapegen/postprocessors.py | 63 | with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file: |
| database-write | hy3dgen/shapegen/postprocessors.py | 77 | with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file: |
| database-write | hy3dgen/shapegen/postprocessors.py | 151 | with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file: |
| database-write | hy3dgen/shapegen/postprocessors.py | 191 | with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_input: |
| database-write | hy3dgen/shapegen/postprocessors.py | 192 | with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_output: |
| process | hy3dgen/shapegen/postprocessors.py | 194 | os.system(f'{self.executable} {temp_input.name} {temp_output.name}') |
| database-write | hy3dgen/texgen/custom_rasterizer/lib/custom_rasterizer_kernel/grid_neighbor.cpp | 255 | visited_seq.insert(seq); |
| database-write | hy3dgen/texgen/hunyuanpaint/pipeline.py | 169 | Update target parameters to be closer to those of source parameters using |
| database-write | hy3dgen/texgen/hunyuanpaint/pipeline.py | 695 | progress_bar.update() |

## UNKNOWN
- Idempotency, transaction boundaries, rollback behavior, and concurrency safety require manual review.
