"""Model Context Protocol server exposing three image-to-3D backends."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP

from .backends import BackendRegistry, GenerationRequest, backend_status

mcp = FastMCP(
    "Cognito-3D-MCP",
    instructions=(
        "Generate local GLB assets from one or more reference images. The default engine "
        "supports up to four canonical views; optional engines accept a single image."
    ),
)
registry = BackendRegistry()


@mcp.tool()
def list_3d_backends() -> list[dict[str, object]]:
    """List the generation engines, their availability, and the default engine."""

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

    Omit ``backend`` to use the default multiview engine. Only the ``hunyuan3d``
    engine accepts the optional left, back, and right canonical views.
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
