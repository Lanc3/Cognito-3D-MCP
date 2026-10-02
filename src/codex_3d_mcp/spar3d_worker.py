"""Long-lived SPAR3D worker used by the bidirectional production pipeline.

The worker speaks one JSON object per line on stdin/stdout. Heavy model state is
kept inside the dedicated SPAR3D virtual environment so the production MCP
process does not need to import its CUDA extensions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import Codex3DError
from .model import ImageTo3DAdapter


def _settings() -> Settings:
    return Settings.from_env(Path.cwd())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _generate(adapter: ImageTo3DAdapter, request: dict[str, Any]) -> dict[str, Any]:
    seed = int(request["seed"])
    _seed_everything(seed)
    output = Path(request["output"]).resolve()
    metadata = adapter.generate(
        Path(request["image"]).resolve(),
        output,
        texture_resolution=int(request.get("texture_resolution", 2048)),
        foreground_ratio=float(request.get("foreground_ratio", 0.85)),
        remesh=str(request.get("remesh", "none")),
        target_vertex_count=request.get("target_vertex_count"),
        progress=lambda _value, _stage: None,
        cancel_event=threading.Event(),
    )
    return {"ok": True, "metadata": metadata}


def serve() -> int:
    adapter = ImageTo3DAdapter(_settings())
    _write({"ok": True, "state": "ready", "protocol": 1})
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                action = request.get("action")
                if action == "generate":
                    response = _generate(adapter, request)
                elif action == "status":
                    response = {"ok": True, "status": adapter.status()}
                elif action == "shutdown":
                    _write({"ok": True, "state": "stopping"})
                    return 0
                else:
                    response = {"ok": False, "code": "INVALID_ACTION", "message": str(action)}
            except Exception as exc:
                response = {
                    "ok": False,
                    "code": exc.code if isinstance(exc, Codex3DError) else "SPAR3D_FAILED",
                    "message": exc.message if isinstance(exc, Codex3DError) else str(exc),
                    "exception": exc.__class__.__name__,
                }
            _write(response)
    finally:
        adapter.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--status", action="store_true")
    arguments = parser.parse_args()
    if arguments.status:
        _write(ImageTo3DAdapter(_settings()).status())
        return 0
    if arguments.serve:
        return serve()
    parser.error("choose --serve or --status")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
