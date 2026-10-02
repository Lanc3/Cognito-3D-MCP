"""Create a small asymmetric front/back GLB pair for the Blender smoke test."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy


def export_selected(path: Path) -> None:
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB", use_selection=True, export_yup=True
    )


def main() -> None:
    destination = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0)
    body = bpy.context.object
    body.scale = (0.65, 0.45, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.2, radius2=0.0, depth=0.5)
    spike = bpy.context.object
    spike.location = (0.45, 0.0, 0.55)
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    spike.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    body.name = "asymmetric_fixture"
    export_selected(destination / "front.glb")
    body.rotation_euler[2] = math.pi
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    export_selected(destination / "back.glb")


if __name__ == "__main__":
    main()
