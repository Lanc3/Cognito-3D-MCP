# Security

Report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/Lanc3/Cognito-3D-MCP/security/advisories/new).
If that feature is unavailable, open an issue asking for a private contact
without posting exploit details, credentials, or private artifacts.

The supported integration release is 0.4.x. The repository's earlier embedded
engine implementation is superseded by isolated upstream runtimes.

The MCP service is a local STDIO process. Its tools can read configured input
roots and create or remove artifacts under configured output directories.
Connect it only to trusted local clients. Keep allowed input roots narrow.
The dashboard binds to loopback, validates Host and Origin, and scopes artifact
paths; do not expose it with a public reverse proxy.

Model loaders and native tools execute external code in isolated environments.
Use trusted model snapshots and the pinned installer sources. Keep operating
system, GPU drivers, client, and dependencies maintained. Store Hugging Face
credentials in the official CLI credential store or process environment, never
in a committed `.env` file or chat message.
