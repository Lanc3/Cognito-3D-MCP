# Third-party notices

The MIT licence in LICENSE covers original Cognito-3D-mcp integration code and
workflow skills. It does not relicense upstream engines, models, native tools,
or vendored modules. Source ZIPs and wheels exclude model weights and runtime
source trees. Installers download external materials separately and preserve
their upstream notices.

| Dependency | Source and terms | Distribution in this release |
| --- | --- | --- |
| Three.js 0.180.0 | [MIT](https://github.com/mrdoob/three.js/blob/r180/LICENSE); copyright Three.js authors | Browser modules vendored with `web/vendor/LICENSE.txt` |
| MCP Python SDK | [MIT](https://github.com/modelcontextprotocol/python-sdk/blob/main/LICENSE) | Python dependency |
| Pillow | [HPND-style licence](https://github.com/python-pillow/Pillow/blob/main/LICENSE) | Python dependency |
| python-dotenv | [BSD-3-Clause](https://github.com/theskumar/python-dotenv/blob/main/LICENSE) | Python dependency |
| Hunyuan3D-2 / 2mv and Paint | [Tencent Hunyuan 3D 2.0 Community Licence](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/f8db63096c8282cb27354314d896feba5ba6ff8a/LICENSE) | Separately downloaded source and weights |
| Stable Fast 3D | [Stability AI terms](https://github.com/Stability-AI/stable-fast-3d/blob/main/LICENSE.md), [model card](https://huggingface.co/stabilityai/stable-fast-3d) | Optional external runtime and gated model |
| SPAR3D | [Upstream source and licence](https://github.com/Stability-AI/stable-point-aware-3d), [model card](https://huggingface.co/stabilityai/stable-point-aware-3d) | Optional external runtime and gated model |
| AutoRemesher 1.2.0 | [MIT source](https://github.com/huxingyi/autoremesher), plus bundled Qt/dependency notices | Separately downloaded portable tool; retain LICENSE and ACKNOWLEDGEMENTS.html |
| Blender | [GPL licensing](https://www.blender.org/about/license/) | Separately installed executable |
| glTF Validator | [Apache-2.0](https://github.com/KhronosGroup/glTF-Validator) | Separately downloaded executable |
| trellis.cpp / TRELLIS.2 | [trellis.cpp](https://github.com/pwilkin/trellis.cpp), [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Optional external runtime; review each model licence |
| Real-ESRGAN | [BSD-3-Clause source](https://github.com/xinntao/Real-ESRGAN) | Optional external tool |
| DINOv2 | [Source and licensing](https://github.com/facebookresearch/dinov2) | Optional external model dependency |

The pinned Hunyuan licence excludes the European Union, United Kingdom and
South Korea, contains use restrictions, and includes additional commercial
conditions. This integration's licence grants no rights in those materials.
Review the full upstream terms before downloading, using, or redistributing an
engine. The installer requires a licence acknowledgement for Hunyuan; it cannot
grant an exception to the upstream territory restriction.

Stability AI model access must be obtained with the user's own Hugging Face
account. When distributing covered materials, preserve the attribution required
by the applicable upstream licence. Modly by Lightning Pixel informed workflow
concepts; its source code is not included.

The previous embedded Hunyuan tree remains in Git history under its original
licence; current source release archives package this integration only.
