"""Render a neutral fixed-camera diagnostic for a GLB in headless Blender."""

from __future__ import annotations

import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector


def main() -> None:
    separator = sys.argv.index("--")
    source = Path(sys.argv[separator + 1]).resolve()
    destination = Path(sys.argv[separator + 2]).resolve()
    recalculate_normals = len(sys.argv) > separator + 3 and sys.argv[separator + 3] == "recalculate"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    subject = bpy.context.view_layer.objects.active
    welded = bmesh.new()
    welded.from_mesh(subject.data)
    bmesh.ops.remove_doubles(welded, verts=list(welded.verts), dist=1e-6)
    if recalculate_normals:
        welded.faces.ensure_lookup_table()
        bmesh.ops.recalc_face_normals(welded, faces=list(welded.faces))
    welded.to_mesh(subject.data)
    welded.free()
    subject.data.update()
    points = [subject.matrix_world @ vertex.co for vertex in subject.data.vertices]
    minimum = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    center = (minimum + maximum) * 0.5
    subject.location -= center

    material = bpy.data.materials.new("Neutral diagnostic")
    material.diffuse_color = (0.18, 0.22, 0.28, 1.0)
    subject.data.materials.clear()
    subject.data.materials.append(material)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world = bpy.data.worlds.new("Diagnostic world")
    scene.world.color = (0.035, 0.035, 0.035)
    camera_data = bpy.data.cameras.new("Diagnostic camera")
    camera = bpy.data.objects.new("Diagnostic camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = (0.0, -3.2, 0.35)
    direction = -camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.5
    for position, energy, size in (
        ((-3.0, -4.0, 5.0), 1300.0, 4.0),
        ((3.0, -1.0, 2.0), 700.0, 3.0),
    ):
        light_data = bpy.data.lights.new("Area", "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new("Area", light_data)
        light.location = position
        scene.collection.objects.link(light)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(destination)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
