"""Compare topology-carrier and capped-Boolean front/back mesh fusion."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def arguments() -> tuple[Path, Path, Path]:
    separator = sys.argv.index("--")
    return (
        Path(sys.argv[separator + 1]).resolve(),
        Path(sys.argv[separator + 2]).resolve(),
        Path(sys.argv[separator + 3]).resolve(),
    )


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_join(path: Path, name: str):
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj not in before and obj.type == "MESH"
    ]
    if not meshes:
        raise RuntimeError(f"No mesh imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    result = bpy.context.view_layer.objects.active
    result.name = name
    world_matrix = result.matrix_world.copy()
    result.parent = None
    result.matrix_world = world_matrix
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return result


def world_bounds(obj) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    minimum = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    return minimum, maximum


def normalize_preserving_axes(obj) -> dict[str, list[float] | float]:
    minimum, maximum = world_bounds(obj)
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    scale = 2.0 / max(extent.z, 1e-9)
    obj.location -= center
    obj.scale = (scale, scale, scale)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    normalized_minimum, normalized_maximum = world_bounds(obj)
    return {
        "source_extent": list(extent),
        "normalized_extent": list(normalized_maximum - normalized_minimum),
        "scale": scale,
    }


def duplicate(obj, name: str):
    result = obj.copy()
    result.data = obj.data.copy()
    bpy.context.collection.objects.link(result)
    result.name = name
    return result


def activate(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def repair(obj) -> None:
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-9, edges=list(bm.edges))
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-7)
    bm.faces.ensure_lookup_table()
    if bm.faces:
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def recalculate_normals(obj) -> None:
    """Update normals without changing the carrier's closed topology."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    if bm.faces:
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def voxel_solidify(obj, voxel_size: float = 0.015) -> dict[str, object]:
    """Turn one SPAR3D surface into a closed carrier before combining views."""
    before = metrics(obj)
    activate(obj)
    obj.data.remesh_voxel_size = voxel_size
    obj.data.remesh_voxel_adaptivity = 0.0
    bpy.ops.object.voxel_remesh()
    repair(obj)
    return {
        "voxel_size": voxel_size,
        "before": before,
        "after": metrics(obj),
    }


def carrier_strategy(
    front,
    back,
    transition: float = 0.22,
    maximum_projection: float = 0.25,
):
    result = duplicate(front, "strategy_carrier")
    solidification = voxel_solidify(result)
    rear_group = result.vertex_groups.new(name="Rear confidence")
    seam_group = result.vertex_groups.new(name="Transition band")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    back_tree = BVHTree.FromObject(back, depsgraph)
    back_minimum, back_maximum = world_bounds(back)
    ray_origin_y = back_minimum.y - 0.1
    ray_distance = back_maximum.y - back_minimum.y + 0.2
    projected = 0
    rejected = 0
    for vertex in result.data.vertices:
        rear_weight = smoothstep((transition - vertex.co.y) / (2.0 * transition))
        seam_weight = max(0.0, 1.0 - abs(vertex.co.y) / transition)
        rear_group.add([vertex.index], rear_weight, "REPLACE")
        seam_group.add([vertex.index], seam_weight, "REPLACE")
        if rear_weight <= 0.0:
            continue
        hit = back_tree.ray_cast(
            Vector((vertex.co.x, ray_origin_y, vertex.co.z)),
            Vector((0.0, 1.0, 0.0)),
            ray_distance,
        )
        if hit[0] is None or abs(hit[0].y - vertex.co.y) > maximum_projection:
            rejected += 1
            continue
        vertex.co.y += (hit[0].y - vertex.co.y) * rear_weight
        projected += 1
    result.data.update()

    activate(result)
    smooth = result.modifiers.new("Side transition smoothing", "LAPLACIANSMOOTH")
    smooth.vertex_group = seam_group.name
    smooth.iterations = 3
    smooth.lambda_factor = 0.18
    smooth.use_volume_preserve = True
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    recalculate_normals(result)
    return result, {
        "solidification": solidification,
        "maximum_projection": maximum_projection,
        "projected_vertices": projected,
        "rejected_vertices": rejected,
    }


def bisect_and_cap(obj, plane_y: float, keep_positive: bool) -> dict[str, int | float]:
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    geometry = list(bm.verts) + list(bm.edges) + list(bm.faces)
    bmesh.ops.bisect_plane(
        bm,
        geom=geometry,
        dist=1e-6,
        plane_co=Vector((0.0, plane_y, 0.0)),
        plane_no=Vector((0.0, 1.0, 0.0)),
    )
    doomed = [
        vertex
        for vertex in bm.verts
        if (vertex.co.y < plane_y - 1e-6) == keep_positive
    ]
    bmesh.ops.delete(bm, geom=doomed, context="VERTS")
    boundary = [edge for edge in bm.edges if edge.is_boundary]
    cap_faces = bmesh.ops.holes_fill(bm, edges=boundary, sides=0).get("faces", [])
    bm.faces.ensure_lookup_table()
    if bm.faces:
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return {"cut_boundary_edges": len(boundary), "cap_faces": len(cap_faces)}


def boolean_strategy(front, back, overlap: float = 0.08):
    front_half = duplicate(front, "strategy_boolean_front_half")
    back_half = duplicate(back, "strategy_boolean_back_half")
    front_solidification = voxel_solidify(front_half)
    back_solidification = voxel_solidify(back_half)
    front_cut = bisect_and_cap(front_half, -overlap, keep_positive=True)
    back_cut = bisect_and_cap(back_half, overlap, keep_positive=False)
    front_cut["geometry"] = metrics(front_half)
    back_cut["geometry"] = metrics(back_half)
    activate(front_half)
    boolean = front_half.modifiers.new("Capped half union", "BOOLEAN")
    boolean.operation = "UNION"
    boolean.solver = "MANIFOLD"
    boolean.operand_type = "OBJECT"
    boolean.object = back_half
    bpy.ops.object.modifier_apply(modifier=boolean.name)
    repair(front_half)
    front_half.name = "strategy_boolean"
    back_half.hide_render = True
    back_half.hide_set(True)
    return front_half, {
        "front_solidification": front_solidification,
        "back_solidification": back_solidification,
        "front_cut": front_cut,
        "back_cut": back_cut,
    }


def component_count(bm) -> int:
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    visited: set[int] = set()
    components = 0
    for root in bm.faces:
        if root.index in visited:
            continue
        components += 1
        stack = [root]
        while stack:
            face = stack.pop()
            if face.index in visited:
                continue
            visited.add(face.index)
            for edge in face.edges:
                stack.extend(linked for linked in edge.link_faces if linked.index not in visited)
    return components


def metrics(obj) -> dict[str, object]:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    boundary = sum(1 for edge in bm.edges if edge.is_boundary)
    non_manifold = sum(1 for edge in bm.edges if not edge.is_manifold)
    signed_volume = bm.calc_volume(signed=True) if not boundary else 0.0
    minimum, maximum = world_bounds(obj)
    result = {
        "vertices": len(bm.verts),
        "faces": len(bm.faces),
        "components": component_count(bm),
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
        "watertight": boundary == 0 and non_manifold == 0,
        "signed_volume": signed_volume,
        "bounds_min": list(minimum),
        "bounds_max": list(maximum),
        "extent": list(maximum - minimum),
    }
    bm.free()
    return result


def surface_residual(obj, front, back, transition: float = 0.22) -> dict[str, float | int]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    front_tree = BVHTree.FromObject(front, depsgraph)
    back_tree = BVHTree.FromObject(back, depsgraph)
    vertices = obj.data.vertices
    stride = max(1, len(vertices) // 20_000)
    distances = []
    front_samples = 0
    back_samples = 0
    for vertex_index in range(0, len(vertices), stride):
        vertex = vertices[vertex_index]
        point = obj.matrix_world @ vertex.co
        if point.y >= transition:
            nearest = front_tree.find_nearest(point)
            front_samples += 1
        elif point.y <= -transition:
            nearest = back_tree.find_nearest(point)
            back_samples += 1
        else:
            continue
        if nearest is not None:
            distances.append(float(nearest[3]))
    distances.sort()
    if not distances:
        return {"samples": 0, "p50": math.inf, "p95": math.inf, "maximum": math.inf}
    return {
        "samples": len(distances),
        "front_samples": front_samples,
        "back_samples": back_samples,
        "p50": distances[len(distances) // 2],
        "p95": distances[min(len(distances) - 1, round(len(distances) * 0.95))],
        "maximum": distances[-1],
    }


def opaque_material(obj) -> None:
    material = bpy.data.materials.new(f"Opaque_{obj.name}")
    material.diffuse_color = (0.28, 0.38, 0.55, 1.0)
    obj.data.materials.clear()
    obj.data.materials.append(material)


def look_at(obj, point=(0.0, 0.0, 0.0)) -> None:
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def render_views(obj, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    opaque_material(obj)
    for candidate in bpy.context.scene.objects:
        candidate.hide_render = candidate != obj
    obj.hide_render = False
    obj.hide_set(False)
    minimum, maximum = world_bounds(obj)
    center = (minimum + maximum) * 0.5
    extent = max(maximum - minimum)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world = bpy.data.worlds.new(f"World_{obj.name}")
    scene.world.color = (0.025, 0.025, 0.025)
    camera_data = bpy.data.cameras.new(f"Camera_{obj.name}")
    camera = bpy.data.objects.new(f"Camera_{obj.name}", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = extent * 1.25
    light_data = bpy.data.lights.new(f"Light_{obj.name}", "AREA")
    light_data.energy = 1200
    light_data.size = 4
    light = bpy.data.objects.new(f"Light_{obj.name}", light_data)
    scene.collection.objects.link(light)
    light.location = center + Vector((-3.0, -4.0, 5.0))
    look_at(light, center)
    distance = extent * 2.2
    views = {
        "front": center + Vector((0.0, distance, extent * 0.15)),
        "back": center + Vector((0.0, -distance, extent * 0.15)),
        "left": center + Vector((-distance, 0.0, extent * 0.15)),
        "right": center + Vector((distance, 0.0, extent * 0.15)),
    }
    for name, position in views.items():
        camera.location = position
        look_at(camera, center)
        scene.render.filepath = str(directory / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(camera, do_unlink=True)
    bpy.data.objects.remove(light, do_unlink=True)


def export_glb(obj, path: Path) -> None:
    activate(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
    )


def main() -> None:
    front_path, back_path, output_dir = arguments()
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_scene()
    front = import_join(front_path, "aligned_front_source")
    front_alignment = normalize_preserving_axes(front)
    back = import_join(back_path, "aligned_back_source")
    back_alignment = normalize_preserving_axes(back)
    back.rotation_mode = "XYZ"
    back.rotation_euler[2] = math.pi
    activate(back)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    back_alignment["canonical_yaw_degrees"] = 180.0
    back_alignment["canonical_yaw_axis"] = "world_z"

    source_metrics = {"front": metrics(front), "back": metrics(back)}
    carrier, carrier_build = carrier_strategy(front, back)
    boolean, cut_metrics = boolean_strategy(front, back)
    report = {
        "alignment": {"front": front_alignment, "back": back_alignment},
        "sources": source_metrics,
        "carrier": {
            "geometry": metrics(carrier),
            "surface_residual": surface_residual(carrier, front, back),
            "build": carrier_build,
        },
        "boolean": {
            "geometry": metrics(boolean),
            "surface_residual": surface_residual(boolean, front, back),
            "cuts": cut_metrics,
        },
    }

    for name, obj in (("carrier", carrier), ("boolean", boolean)):
        render_views(obj, output_dir / "renders" / name)
        export_glb(obj, output_dir / f"{name}.glb")
    front.hide_render = True
    back.hide_render = True
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "fusion-strategies.blend"))
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
