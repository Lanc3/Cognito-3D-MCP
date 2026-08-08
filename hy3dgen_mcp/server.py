"""Model Context Protocol server exposing three image-to-3D backends."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP

from .backends import BackendRegistry, GenerationRequest, backend_status

mcp = FastMCP(
    "Hunyuan3D multi-backend",
    instructions=(
        "Generate GLB assets from local images. Hunyuan3D-2mv is preferred and supports "
        "up to four canonical views; SF3D and SPAR3D are optional single-image fallbacks."
    ),
)
registry = BackendRegistry()


@mcp.tool()
def list_3d_backends() -> list[dict[str, object]]:
    """List the three model backends, their availability, and the preferred backend."""

    return backend_status(registry.settings)


@mcp.tool()
def generate_3d(
    front_image: str,
    backend: Literal["hunyuan3d", "sf3d", "spar3d"] = "hunyuan3d",
    left_image: str | None = None,
    back_image: str | None = None,
    right_image: str | None = None,
    output_dir: str | None = None,
    seed: int = 12345,
    steps: int = 50,
    guidance_scale: float = 5.0,
    octree_resolution: int = 384,
    texture_resolution: int = 1024,
) -> dict[str, object]:
    """Generate a GLB from one or more local images.

    Omit ``backend`` to use the preferred Hunyuan3D-2mv model. Only Hunyuan
    accepts the optional left, back, and right canonical views.
    """

    result = registry.generate(
        GenerationRequest(
            front_image=Path(front_image),
            backend=backend,
            left_image=Path(left_image) if left_image else None,
            back_image=Path(back_image) if back_image else None,
            right_image=Path(right_image) if right_image else None,
            output_dir=Path(output_dir) if output_dir else None,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
            texture_resolution=texture_resolution,
        )
    )
    return result.to_dict()


def main() -> None:
    """Run the MCP server over stdio."""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

