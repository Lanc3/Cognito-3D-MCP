"""Render a SPAR3D back GLB before/after yaw from both camera directions."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def bounds(obj) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    minimum = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    return minimum, maximum


def look_at(obj, point: Vector) -> None:
    obj.rotation_euler = (point - obj.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    separator = sys.argv.index("--")
    source = Path(sys.argv[separator + 1]).resolve()
    destination = Path(sys.argv[separator + 2]).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    subject = bpy.context.view_layer.objects.active
    imported_parent = subject.parent.name if subject.parent else None
    imported_world_matrix = [list(row) for row in subject.matrix_world]
    world_matrix = subject.matrix_world.copy()
    subject.parent = None
    subject.matrix_world = world_matrix
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    minimum, maximum = bounds(subject)
    subject.location -= (minimum + maximum) * 0.5
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    minimum, maximum = bounds(subject)
    extent = maximum - minimum

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("Orientation diagnostic world")
    scene.world.color = (0.04, 0.04, 0.04)

    camera_data = bpy.data.cameras.new("Orientation camera")
    camera = bpy.data.objects.new("Orientation camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(extent.x, extent.z) * 1.18

    for location, energy, size in (
        ((-3.0, -4.0, 5.0), 900.0, 4.0),
        ((3.0, 2.0, 3.0), 500.0, 3.0),
    ):
        light_data = bpy.data.lights.new("Orientation area", "AREA")
        light_data.energy = energy
        light_data.size = size
        light = bpy.data.objects.new("Orientation area", light_data)
        light.location = location
        scene.collection.objects.link(light)
        look_at(light, Vector((0.0, 0.0, 0.0)))

    distance = max(extent) * 2.5
    variants = (
        ("yaw000_camera_pos_y", 0.0, 1.0),
        ("yaw000_camera_neg_y", 0.0, -1.0),
        ("yaw180_camera_pos_y", math.pi, 1.0),
        ("yaw180_camera_neg_y", math.pi, -1.0),
    )
    report = {
        "source": str(source),
        "imported_parent": imported_parent,
        "imported_world_matrix": imported_world_matrix,
        "parent_cleared_before_rotation": True,
        "extent": list(extent),
        "renders": {},
    }
    for name, yaw, camera_y_sign in variants:
        subject.rotation_mode = "XYZ"
        subject.rotation_euler = (0.0, 0.0, yaw)
        camera.location = Vector((0.0, camera_y_sign * distance, extent.z * 0.12))
        look_at(camera, Vector((0.0, 0.0, 0.0)))
        output = destination / f"{name}.png"
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        report["renders"][name] = {
            "yaw_degrees": math.degrees(yaw),
            "camera_y_sign": camera_y_sign,
            "path": str(output),
        }

    (destination / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
