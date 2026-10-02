# Integration and upstream provenance

The current release packages Cognito-3D-mcp as an independent MCP integration.
Its source tree contains the local Python supervisors, job stores, review gates,
dashboard, setup scripts, tests, and agent workflow skills. Model engines and
weights are installed separately into isolated runtime directories.

The previous GitHub tree embedded a Hunyuan engine fork and a smaller MCP
adapter. That tree remains in Git history. The current package uses the
`codex_3d_mcp` import namespace and durable asynchronous APIs. Existing users of
`hy3dgen_mcp.server`, `list_3d_backends`, or `generate_3d` should install this
release and register the appropriate server as described in README.md; those
old synchronous APIs are not carried forward.

The local development workspace also contained a separate 2D sprite editor.
It is excluded from this release, along with private experimental assets,
generated outputs, old portable bundles, local caches, and workstation config.

The default installer pins Hunyuan3D-2 source to
`f8db63096c8282cb27354314d896feba5ba6ff8a`, shape weights to
`3a761b539b29fe4ff64714813aa9560fd66f5de0`, and Paint weights to
`9cd649ba6913f7a852e3286bad86bfa9a2d83dcf`. Windows native loader/build patches
are applied to downloaded runtime files by the setup scripts. The upstream
license and attribution are preserved in that runtime.

AutoRemesher downloads use archive and executable SHA-256 checks. Vendored
Three.js browser modules retain their bundled MIT license. See
THIRD_PARTY_NOTICES.md for all external project links and licensing distinctions.
