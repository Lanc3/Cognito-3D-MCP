# Security policy

Please report security issues privately through GitHub's **Report a
vulnerability** feature. Do not open a public issue for an unpatched
vulnerability.

The MCP server is intended for trusted local use. It accepts local image and
output paths and launches configured model runtimes. Do not expose the stdio
server through an unauthenticated network bridge.

Model checkpoints are executable supply-chain inputs. Download them only from
the official publishers and verify pinned hashes where this repository provides
them.
