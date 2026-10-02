"""Render one object from an existing Blender file with an opaque neutral material."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
from mathutils import Vector


def main() -> None:
    separator = sys.argv.index("--")
    object_name = sys.argv[separator + 1]
    destination = Path(sys.argv[separator + 2]).resolve()
    subject = bpy.data.objects.get(object_name)
    if subject is None or subject.type != "MESH":
        raise RuntimeError(f"Mesh object not found: {object_name}")

    for obj in bpy.context.scene.objects:
        obj.hide_render = obj != subject
    subject.hide_render = False
    subject.hide_set(False)
    points = [subject.matrix_world @ vertex.co for vertex in subject.data.vertices]
    minimum = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    center = (minimum + maximum) * 0.5
    extent = max(maximum - minimum)
    subject.location -= center

    material = bpy.data.materials.new("Opaque neutral diagnostic")
    material.diffuse_color = (0.32, 0.38, 0.48, 1.0)
    subject.data.materials.clear()
    subject.data.materials.append(material)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world.color = (0.025, 0.025, 0.025)
    camera_data = bpy.data.cameras.new("Diagnostic camera")
    camera = bpy.data.objects.new("Diagnostic camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = (0.0, -3.2, 0.45)
    camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = extent * 1.25
    light_data = bpy.data.lights.new("Diagnostic light", "AREA")
    light_data.energy = 1100
    light_data.size = 4
    light = bpy.data.objects.new("Diagnostic light", light_data)
    light.location = (-3.0, -4.0, 5.0)
    scene.collection.objects.link(light)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(destination)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
