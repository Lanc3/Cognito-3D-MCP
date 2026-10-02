# Cognito-3D-mcp

![Cognito-3D-mcp](assets/branding/cognito-3d-mcp-brand.png)

Local image-to-3D tools for MCP clients. Create textured GLBs, inspect them in a
local 3D viewer, repair geometry, and export assets through durable, reviewed
jobs. This repository ships the MCP integration, Windows setup scripts, tests,
and the agent skills needed to drive the workflows.

**3D only:** there is no sprite rig, spritesheet generator, or 2D editor.
Reference PNG preparation is part of the 3D reconstruction pipeline.

## Install on Windows

Download and extract the repository ZIP, then double-click **`Install.cmd`**
and choose a 3D engine from its guided menu. The installer prepares an isolated
runtime, downloads the selected engine's
models, installs support tools, and configures Codex and the bundled skills.
It checks each command and stops when a prerequisite or download fails.
Run it again after correcting the reported issue.

The default engine is Hunyuan3D-2mv. Its licence **excludes the EU, UK and South
Korea**. The installer requires acknowledgement of the upstream terms before
downloading it. Users need suitable rights for the engine they select; the MIT
licence of this integration does not change model terms. See
[third-party notices](THIRD_PARTY_NOTICES.md).

Generation requires a compatible NVIDIA GPU/driver, Python 3.11, Visual Studio
2022 C++ Build Tools, and the matching CUDA toolkit for native extension builds.
The installer reports missing prerequisites and can bootstrap supported tools
through Windows Package Manager. Allow at least 45 GB free for the Hunyuan
runtime/model install; additional environments and generated assets need more.
Windows is the supported generation setup. CPU tests and the lightweight MCP
package also run on Linux; a Linux GPU installer is not provided.

Installation is one launcher with guided prerequisite/licence steps, rather than
a promise that drivers, model access approvals, or hardware can be installed
without user involvement. Large model downloads can take substantial time.

Advanced installation and a preview with no downloads or configuration writes:

```powershell
.\install.ps1 -DryRun
.\install.ps1 -Backend Hunyuan
.\install.ps1 -Backend SPAR3D
.\install.ps1 -Backend SF3D
```

Runtimes and model caches default to `%LOCALAPPDATA%\Cognito-3D-mcp`.
The installer also defaults outputs to `%LOCALAPPDATA%\Cognito-3D-mcp\outputs`.
Use installer parameters or `.env` for alternate locations. The
example environment contains no account-specific paths or credentials.

Stability AI backends require access to gated Hugging Face models. Accept the
model terms with your own account. The installer reuses existing credentials or
guides you through local Hugging Face authentication before downloading weights.
Do not put a token in chat or source control. The installer preserves existing
MCP configuration and skill directories and reports conflicts for manual review.
Restart Codex after installing or updating MCP servers and skills.

## Choose a workflow

| Workflow | Registered server | Agent skill |
| --- | --- | --- |
| Named-view Hunyuan batches; repair, remesh, Paint, bake and inspect profiles | `codex_3d_models_hunyuan_mv` | [Multiview 3D](skills/codex-hunyuan-multiview-3d/SKILL.md) |
| Single-image SF3D and SPAR3D generation or backend comparisons | `codex_3d_models` / `codex_3d_models_spar3d` | [Single-image 3D](skills/codex-3d-models/SKILL.md) |
| Matched front/back reconstruction with fusion and QA | `codex_3d_models_trellis` | [Bidirectional 3D](skills/codex-bidirectional-3d/SKILL.md) |

For example, ask the connected agent: “Use Cognito-3D-mcp to create a textured
3D asset from these reference images.” The agent checks `server_status`, prepares
consistent images beneath an allowed input root, runs serial work, inspects the
quality evidence, and returns accepted GLBs. Text descriptions need an
image-generation capability in the MCP client; this server does not include an
image generator or make LLM API calls.

The Hunyuan batch workflow registers all assets, approves reference cutouts,
generates shapes, repairs and reviews them, independently remeshes three output
profiles, paints the full-game profile, and bakes that appearance to the lower
profiles. Worker processes and GPU leases are serial. Passing machine gates and
agent reviews are required before later stages or delivery.

The queue opens a loopback dashboard with progress, previews, quality reports,
and an interactive model viewer. Viewer modules are shipped locally. Failed
attempts and completed checkpoints remain available for repair and restart.

## Run and connect other clients

Codex starts the configured servers automatically when it connects. `Run.cmd`
starts the installed server manually; it waits for an MCP client on STDIO.
To select a particular installed engine, pass `-Backend SPAR3D` (or another
engine name).

```powershell
.\scripts\run-hunyuan-mv-server.ps1
```

MCP uses STDIO, so a terminal run waits for a client protocol session. Other MCP
hosts can start the same launcher with an absolute repository path:

```json
{
  "mcpServers": {
    "cognito-3d": {
      "command": "powershell.exe",
      "args": ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "C:\\path\\to\\Cognito-3D-mcp\\scripts\\run-hunyuan-mv-server.ps1"]
    }
  }
}
```

That execution-policy setting applies to the launched process. A non-Codex host
must supply the same workflow instructions or import the relevant `skills/`
content. Compatibility server names, the Python namespace `codex_3d_mcp`, and
`CODEX_*` variables remain stable despite the product rebrand.

## Documentation and development

- [Batch stages and quality gates](docs/hunyuan-batch-pipeline.md)
- [Dashboard and artifact viewer](docs/dashboard-ui.md)
- [Shape repair and accepted masters](docs/shape-repair.md)
- [Optional CuMesh repair adapter](docs/cumesh-repair.md)
- [Bidirectional pipeline](docs/production-trellis.md)
- [Contributing and CPU checks](CONTRIBUTING.md)
- [Security and private reporting](SECURITY.md)
- [Changes and migration](CHANGELOG.md), [upstream provenance](UPSTREAM.md)

The source ZIP contains the installer and skills. A Python wheel is the
lightweight integration package; heavy model runtimes are installed separately.
Automated checks verify source files, skill references, wheel browser assets,
HTTP access, job lifecycle, and CPU quality logic. Fresh-machine GPU installation
and real generation remain separate hardware checks; see
[release verification](docs/release-verification.md).

## Licensing

Original integration code and workflow skills use the [MIT licence](LICENSE).
External engines, model weights, and native tools retain their own terms.
Cognito-3D-mcp is maintained independently by Lanc3 and contributors and is not
affiliated with or endorsed by Tencent, Stability AI, Microsoft, or OpenAI.
