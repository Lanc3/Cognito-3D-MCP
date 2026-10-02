"""Start an installed STDIO MCP server without writing installer output to stdout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from configure_codex import server_entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=["Hunyuan", "SF3D", "SPAR3D", "TRELLIS"], default="Hunyuan"
    )
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8-sig"))
    if args.backend not in settings["backends"]:
        parser.error(f"{args.backend} is not installed. Rerun install.ps1 -Backend {args.backend}.")
    name = {
        "Hunyuan": "codex_3d_models_hunyuan_mv",
        "SF3D": "codex_3d_models",
        "SPAR3D": "codex_3d_models_spar3d",
        "TRELLIS": "codex_3d_models_trellis",
    }[args.backend]
    entry = server_entries(settings)[name]
    environment = {**os.environ, **entry["env"]}
    # Explicit caller env vars remain useful for manual launch overrides.
    environment.update(
        {key: value for key, value in os.environ.items() if key.startswith("CODEX_")}
    )
    return subprocess.call([entry["command"], *entry["args"]], cwd=entry["cwd"], env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
