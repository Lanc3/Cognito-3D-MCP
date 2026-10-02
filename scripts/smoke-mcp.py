"""Initialize each real STDIO server and inspect tools/status without GPU generation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVERS = {
    "codex_3d_mcp.hunyuan_mv.server": {
        "server_status", "create_generation_batch", "get_shape_repair_capabilities",
        "create_shape_repair_attempt", "submit_shape_repair_candidate",
    },
    "codex_3d_mcp.server": {
        "server_status", "generate_3d_model", "get_generation_status", "cancel_generation",
    },
    "codex_3d_mcp.trellis.server": {
        "server_status", "generate_bidirectional_3d_model", "review_seed_pair",
    },
}
REMOVED = {"create_sprite_animation", "replace_sprite_art", "get_sprite_project", "open_sprite_editor"}


async def smoke(module: str, expected: set[str], root: Path, installed: bool) -> dict:
    env = dict(os.environ)
    if not installed:
        source = Path(__file__).resolve().parents[1] / "src"
        env["PYTHONPATH"] = str(source) + os.pathsep + env.get("PYTHONPATH", "")
    env.update({
        "CODEX_3D_DATA_ROOT": str(root / "data"),
        "CODEX_3D_OUTPUT_DIR": str(root / "single"),
        "CODEX_HUNYUAN_MV_OUTPUT_DIR": str(root / "hunyuan"),
        "CODEX_HUNYUAN_MV_RUNTIME_ROOT": str(root / "absent-runtime"),
        "CODEX_HUNYUAN_MV_PYTHON": str(root / "absent-runtime" / "python"),
        "CODEX_HUNYUAN_MV_MODEL_CACHE": str(root / "absent-models"),
        "CODEX_HUNYUAN_MV_REPAIR_PACKAGES": str(root / "absent-repair"),
        "CODEX_HUNYUAN_MV_DASHBOARD_AUTO_OPEN": "false",
        "CODEX_TRELLIS_OUTPUT_DIR": str(root / "bidirectional"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
    })
    # Model credentials are irrelevant to this offline startup test.
    env.pop("HF_TOKEN", None)
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    params = StdioServerParameters(command=sys.executable, args=["-m", module], cwd=str(root), env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            if expected - names or names & REMOVED:
                raise RuntimeError(f"{module}: unexpected MCP tools: missing {expected - names}; 2D {names & REMOVED}")
            status = await session.call_tool("server_status")
            if status.isError:
                raise RuntimeError(f"{module}: server_status failed: {status.content}")
            payload = status.structuredContent
            if payload is None:
                payload = json.loads(next(item.text for item in status.content if item.type == "text"))
            version = payload.get("server_version")
            if version != "0.4.0":
                raise RuntimeError(f"{module}: unexpected integration version {version}")
            return {"module": module, "version": version, "tools": len(names), "status_ok": True}


async def run(installed: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="cognito-mcp-smoke-") as directory:
        for index, (module, expected) in enumerate(SERVERS.items()):
            root = Path(directory) / str(index)
            root.mkdir()
            result = await asyncio.wait_for(smoke(module, expected, root, installed), timeout=45)
            print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true", help="Use installed package instead of repository src")
    args = parser.parse_args()
    asyncio.run(run(args.installed))
