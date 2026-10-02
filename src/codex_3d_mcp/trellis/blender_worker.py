"""Headless Blender geometry fusion, UV baking, export, preview, and QA worker.

Run only through Blender:: ``blender --background --python blender_worker.py -- config.json``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _bpy():
    import bpy

    return bpy


def _arguments() -> dict:
    if "--" not in sys.argv:
        raise RuntimeError("Expected a JSON configuration path after --")
    path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def _clear() -> None:
    bpy = _bpy()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _import_join(path: str, name: str):
    bpy = _bpy()
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [obj for obj in bpy.context.scene.objects if obj not in before and obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh was imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.hide_set(False)
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    # glTF imports meshes below a coordinate-conversion parent.  Clear that
    # parent while preserving the world matrix so subsequent yaw values are
    # applied around Blender's world-space Z up axis.
    world_matrix = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = world_matrix
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def _world_points(obj, limit: int = 30_000):
    vertices = obj.data.vertices
    stride = max(1, len(vertices) // limit)
    return [obj.matrix_world @ vertices[index].co for index in range(0, len(vertices), stride)]


def _jacobi_axes(points):
    from mathutils import Matrix, Vector

    center = sum(points, Vector()) / len(points)
    covariance = [[0.0] * 3 for _ in range(3)]
    for point in points:
        delta = point - center
        for row in range(3):
            for column in range(3):
                covariance[row][column] += delta[row] * delta[column]
    matrix = Matrix(covariance)
    vectors = Matrix.Identity(3)
    for _ in range(24):
        p, q = max(((0, 1), (0, 2), (1, 2)), key=lambda pair: abs(matrix[pair[0]][pair[1]]))
        if abs(matrix[p][q]) < 1e-10:
            break
        angle = 0.5 * math.atan2(2 * matrix[p][q], matrix[q][q] - matrix[p][p])
        rotation = Matrix.Identity(3)
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation[p][p] = cosine
        rotation[q][q] = cosine
        rotation[p][q] = sine
        rotation[q][p] = -sine
        matrix = rotation.transposed() @ matrix @ rotation
        vectors = vectors @ rotation
    return [vectors.col[index].normalized() for index in range(3)]


def _level_and_normalize(obj) -> dict:
    bpy = _bpy()
    from mathutils import Vector

    points = _world_points(obj)
    axes = _jacobi_axes(points)
    up = max(axes, key=lambda axis: abs(axis.dot(Vector((0, 0, 1)))))
    if up.z < 0:
        up.negate()
    angle = math.degrees(up.angle(Vector((0, 0, 1))))
    if angle <= 20.0:
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = up.rotation_difference(Vector((0, 0, 1)))
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    points = _world_points(obj)
    minimum = Vector(
        (
            min(point.x for point in points),
            min(point.y for point in points),
            min(point.z for point in points),
        )
    )
    maximum = Vector(
        (
            max(point.x for point in points),
            max(point.y for point in points),
            max(point.z for point in points),
        )
    )
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    scale = 2.0 / max(extent)
    obj.location -= center
    obj.scale = (scale, scale, scale)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return {"level_correction_degrees": angle, "source_extent": list(extent)}


def _trim(obj, *, axis: str, positive: bool) -> None:
    import bmesh

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    threshold = -0.12 if positive else 0.12
    doomed = []
    for face in bm.faces:
        center = face.calc_center_median()
        coordinate = center.y if axis == "y" else center.x
        if (positive and coordinate < threshold) or (not positive and coordinate > threshold):
            doomed.append(face)
    bmesh.ops.delete(bm, geom=doomed, context="FACES")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def _apply_world_yaw(obj, angle: float) -> None:
    """Bake a yaw around Blender's world-space Z-up axis."""

    bpy = _bpy()
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, angle)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)


def _join(objects, name: str):
    bpy = _bpy()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.hide_set(False)
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    result = bpy.context.view_layer.objects.active
    result.name = name
    return result


def _remove_small_components(obj, minimum_area_ratio: float = 0.0005) -> dict:
    """Remove loose surface components below the production area threshold."""

    import bmesh

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    total_area = sum(face.calc_area() for face in bm.faces)
    threshold = total_area * minimum_area_ratio
    visited = bytearray(len(bm.faces))
    small_components = []
    component_count = 0
    removed_components = 0
    for root in bm.faces:
        if visited[root.index]:
            continue
        component_count += 1
        stack = [root]
        component = []
        area = 0.0
        while stack:
            face = stack.pop()
            if visited[face.index]:
                continue
            visited[face.index] = 1
            area += face.calc_area()
            if area <= threshold:
                component.append(face)
            else:
                component = []
            for edge in face.edges:
                for linked in edge.link_faces:
                    if not visited[linked.index]:
                        stack.append(linked)
        if area < threshold:
            small_components.append((area, component))
    if len(small_components) == component_count and small_components:
        retained_area = 0.0
        keep_until = total_area * 0.995
        doomed_components = []
        for area, component in sorted(small_components, key=lambda item: item[0], reverse=True):
            if retained_area < keep_until:
                retained_area += area
            else:
                doomed_components.append((area, component))
    else:
        doomed_components = small_components
    doomed = [face for _area, component in doomed_components for face in component]
    removed_components = len(doomed_components)
    removed_faces = len(doomed)
    if doomed:
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        loose = [vertex for vertex in bm.verts if not vertex.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return {
        "components_before": component_count,
        "removed_components": removed_components,
        "removed_faces": removed_faces,
        "minimum_area_ratio": minimum_area_ratio,
    }


def _component_profile(obj) -> dict:
    """Measure connected surface fragmentation without mutating the mesh."""

    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    visited = bytearray(len(bm.faces))
    areas = []
    face_counts = []
    for root in bm.faces:
        if visited[root.index]:
            continue
        stack = [root]
        area = 0.0
        faces = 0
        while stack:
            face = stack.pop()
            if visited[face.index]:
                continue
            visited[face.index] = 1
            area += face.calc_area()
            faces += 1
            for edge in face.edges:
                for linked in edge.link_faces:
                    if not visited[linked.index]:
                        stack.append(linked)
        areas.append(area)
        face_counts.append(faces)
    bm.free()
    total_area = sum(areas)
    largest_ratio = max(areas, default=0.0) / max(total_area, 1e-12)
    largest_face_ratio = max(face_counts, default=0) / max(sum(face_counts), 1)
    fragmented = len(areas) > 250 and largest_ratio < 0.05 and largest_face_ratio < 0.05
    return {
        "components": len(areas),
        "largest_area_ratio": largest_ratio,
        "largest_face_ratio": largest_face_ratio,
        "faces": sum(face_counts),
        "passed": not fragmented,
    }


def _weld_vertices(obj, distance: float = 1e-6) -> int:
    """Weld coincident glTF seam vertices before topology analysis or fusion."""

    import bmesh

    mesh = obj.data
    before = len(mesh.vertices)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=distance)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return before - len(mesh.vertices)


def _repair_mesh(obj) -> None:
    """Remove collapsed elements and close remesh boundaries in-place."""

    import bmesh

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-9, edges=list(bm.edges))
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-7)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    boundary = [edge for edge in bm.edges if edge.is_boundary]
    if boundary:
        bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
    bm.faces.ensure_lookup_table()
    if bm.faces:
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.faces.ensure_lookup_table()
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate(verbose=False, clean_customdata=False)
    mesh.update()


def _target_faces(obj, target: int, source_voxel_size: float) -> float:
    bpy = _bpy()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    _repair_mesh(obj)
    current = len(obj.data.polygons)
    if int(target * 0.95) <= current <= int(target * 1.05):
        return source_voxel_size
    source_mesh = obj.data.copy()
    voxel_size = source_voxel_size * math.sqrt(max(1.0, current / target))
    for _ in range(4):
        obj.data = source_mesh.copy()
        obj.data.remesh_voxel_size = voxel_size
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.voxel_remesh()
        _remove_small_components(obj)
        _repair_mesh(obj)
        actual = len(obj.data.polygons)
        if int(target * 0.95) <= actual <= int(target * 1.05):
            return voxel_size
        voxel_size *= math.sqrt(max(actual, 1) / target)
    actual = len(obj.data.polygons)
    if actual > int(target * 1.05):
        modifier = obj.modifiers.new("Voxel target correction", "DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = min(1.0, (target * 1.02) / actual)
        modifier.use_collapse_triangulate = False
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    elif actual < int(target * 0.95):
        import bmesh

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        missing = target - actual
        faces = sorted(bm.faces, key=lambda face: face.calc_area(), reverse=True)
        bmesh.ops.poke(
            bm,
            faces=faces[: max(0, missing // 2)],
            offset=0.0,
            center_mode="MEAN_WEIGHTED",
        )
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
    return voxel_size


def _transfer_uv(source, target) -> None:
    bpy = _bpy()
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    modifier = target.modifiers.new("Transfer production UV", "DATA_TRANSFER")
    modifier.object = source
    modifier.use_loop_data = True
    modifier.data_types_loops = {"UV"}
    modifier.loop_mapping = "POLYINTERP_NEAREST"
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def _smart_uv(obj, margin: float) -> None:
    bpy = _bpy()
    if not obj.data.polygons:
        raise RuntimeError("Cannot unwrap an empty production mesh")
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.hide_viewport = False
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=math.radians(66),
        margin_method="ADD",
        island_margin=margin,
        scale_to_bounds=True,
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def _uv_metrics(obj) -> dict[str, float]:
    """Measure non-overlapping UV triangle area in the normalized atlas."""

    layer = obj.data.uv_layers.active
    if layer is None:
        return {"occupancy": 0.0}
    area = 0.0
    for polygon in obj.data.polygons:
        points = [layer.data[index].uv for index in polygon.loop_indices]
        origin = points[0]
        for index in range(1, len(points) - 1):
            first = points[index] - origin
            second = points[index + 1] - origin
            area += abs(first.x * second.y - first.y * second.x) * 0.5
    return {"occupancy": min(1.0, area)}


def _projection_material(front_image: str, back_image: str):
    bpy = _bpy()
    material = bpy.data.materials.new("BidirectionalSeedProjection")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    texcoord = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    front_combine = nodes.new("ShaderNodeCombineXYZ")
    back_combine = nodes.new("ShaderNodeCombineXYZ")
    invert_x = nodes.new("ShaderNodeMath")
    invert_x.operation = "SUBTRACT"
    invert_x.inputs[0].default_value = 1.0
    front = nodes.new("ShaderNodeTexImage")
    back = nodes.new("ShaderNodeTexImage")
    front.image = bpy.data.images.load(front_image, check_existing=True)
    back.image = bpy.data.images.load(back_image, check_existing=True)
    front.extension = back.extension = "CLIP"
    geometry = nodes.new("ShaderNodeNewGeometry")
    dot = nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = (0.0, 1.0, 0.0)
    map_range = nodes.new("ShaderNodeMapRange")
    map_range.inputs[1].default_value = -0.35
    map_range.inputs[2].default_value = 0.35
    map_range.inputs[3].default_value = 0.0
    map_range.inputs[4].default_value = 1.0
    map_range.clamp = True
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    links.new(texcoord.outputs["Generated"], separate.inputs[0])
    links.new(separate.outputs["X"], front_combine.inputs["X"])
    links.new(separate.outputs["Z"], front_combine.inputs["Y"])
    links.new(separate.outputs["X"], invert_x.inputs[1])
    links.new(invert_x.outputs[0], back_combine.inputs["X"])
    links.new(separate.outputs["Z"], back_combine.inputs["Y"])
    links.new(front_combine.outputs[0], front.inputs["Vector"])
    links.new(back_combine.outputs[0], back.inputs["Vector"])
    links.new(geometry.outputs["Normal"], dot.inputs[0])
    links.new(dot.outputs["Value"], map_range.inputs[0])
    links.new(map_range.outputs["Result"], mix.inputs[0])
    links.new(back.outputs["Color"], mix.inputs[1])
    links.new(front.outputs["Color"], mix.inputs[2])
    links.new(mix.outputs[0], shader.inputs["Base Color"])
    links.new(shader.outputs[0], output.inputs[0])
    return material


def _bake(obj, material, path: Path, resolution: int, bake_type: str) -> None:
    bpy = _bpy()
    image = bpy.data.images.new(path.stem, width=resolution, height=resolution, alpha=True)
    target = material.node_tree.nodes.new("ShaderNodeTexImage")
    target.name = f"BAKE_TARGET_{path.stem}"
    target.image = image
    material.node_tree.nodes.active = target
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.render.image_settings.file_format = "PNG"
    scene.render.bake.margin = 16
    if bake_type == "DIFFUSE":
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})
    else:
        bpy.ops.object.bake(type=bake_type)
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()


def _normal_bake(master, game, path: Path, resolution: int) -> None:
    bpy = _bpy()
    material = game.data.materials[0]
    image = bpy.data.images.new(path.stem, width=resolution, height=resolution, alpha=False)
    target = material.node_tree.nodes.new("ShaderNodeTexImage")
    target.name = "BAKE_TARGET_normal"
    target.image = image
    material.node_tree.nodes.active = target
    bpy.ops.object.select_all(action="DESELECT")
    master.hide_set(False)
    master.select_set(True)
    game.select_set(True)
    bpy.context.view_layer.objects.active = game
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "CPU"
    bpy.context.scene.render.bake.use_selected_to_active = True
    bpy.context.scene.render.bake.cage_extrusion = 0.03
    bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT")
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.context.scene.render.bake.use_selected_to_active = False


def _export(obj, path: Path) -> None:
    bpy = _bpy()
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
    )


def _mesh_metrics(obj) -> dict:
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    boundary = sum(1 for edge in bm.edges if edge.is_boundary)
    non_manifold = sum(1 for edge in bm.edges if not edge.is_manifold)
    degenerate = sum(1 for face in bm.faces if face.calc_area() <= 1e-12)
    area_bands = {
        f"le_{threshold:.0e}": sum(1 for face in bm.faces if face.calc_area() <= threshold)
        for threshold in (1e-10, 1e-8, 1e-6)
    }
    metrics = {
        "vertices": len(bm.verts),
        "faces": len(bm.faces),
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
        "degenerate_faces": degenerate,
        "area_bands": area_bands,
        "watertight": boundary == 0 and non_manifold == 0,
    }
    bm.free()
    return metrics


def _configure_final_material(obj, texture_dir: Path, maps: dict):
    bpy = _bpy()
    material = bpy.data.materials.new(f"PBR_{obj.name}")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    base = nodes.new("ShaderNodeTexImage")
    base.image = bpy.data.images.load(str(texture_dir / "baseColor.png"), check_existing=True)
    orm = nodes.new("ShaderNodeTexImage")
    orm.image = bpy.data.images.load(str(texture_dir / "orm.png"), check_existing=True)
    orm.image.colorspace_settings.name = "Non-Color"
    separate = nodes.new("ShaderNodeSeparateColor")
    normal_tex = nodes.new("ShaderNodeTexImage")
    normal_tex.image = bpy.data.images.load(str(texture_dir / "normal.png"), check_existing=True)
    normal_tex.image.colorspace_settings.name = "Non-Color"
    normal = nodes.new("ShaderNodeNormalMap")
    links.new(base.outputs["Color"], shader.inputs["Base Color"])
    links.new(base.outputs["Alpha"], shader.inputs["Alpha"])
    links.new(orm.outputs["Color"], separate.inputs["Color"])
    links.new(separate.outputs["Green"], shader.inputs["Roughness"])
    links.new(separate.outputs["Blue"], shader.inputs["Metallic"])
    links.new(normal_tex.outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], shader.inputs["Normal"])
    if maps.get("emissive"):
        emission = nodes.new("ShaderNodeTexImage")
        emission.image = bpy.data.images.load(
            str(texture_dir / "emissive.png"), check_existing=True
        )
        links.new(emission.outputs["Color"], shader.inputs["Emission Color"])
        shader.inputs["Emission Strength"].default_value = 1.0
    links.new(shader.outputs[0], output.inputs[0])
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return material


def _look_at(camera, point=(0, 0, 0)) -> None:
    from mathutils import Vector

    camera.rotation_euler = (Vector(point) - camera.location).to_track_quat("-Z", "Y").to_euler()


def _render_previews(obj, directory: Path) -> None:
    bpy = _bpy()
    directory.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world.color = (0.035, 0.035, 0.035)
    camera_data = bpy.data.cameras.new("QA Camera")
    camera = bpy.data.objects.new("QA Camera", camera_data)
    scene.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.5
    scene.camera = camera
    light_data = bpy.data.lights.new("QA Key", "AREA")
    light_data.energy = 900
    light_data.shape = "DISK"
    light_data.size = 5
    light = bpy.data.objects.new("QA Key", light_data)
    scene.collection.objects.link(light)
    light.location = (4, 4, 6)
    _look_at(light)
    views = {
        "front": (0, 4, 0.3),
        "back": (0, -4, 0.3),
        "left": (-4, 0, 0.3),
        "right": (4, 0, 0.3),
    }
    for name, position in views.items():
        camera.location = position
        _look_at(camera)
        scene.render.filepath = str(directory / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    turntable = directory / "turntable"
    turntable.mkdir(exist_ok=True)
    for index in range(12):
        angle = math.radians(index * 30)
        camera.location = (4 * math.sin(angle), 4 * math.cos(angle), 0.3)
        _look_at(camera)
        scene.render.filepath = str(turntable / f"frame_{index:03d}.png")
        bpy.ops.render.render(write_still=True)


def _build(config: dict) -> dict:
    bpy = _bpy()
    job = Path(config["job_dir"])
    texture_dir = job / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    _clear()
    front = _import_join(config["front_glb"], "front_source")
    front_alignment = _level_and_normalize(front)
    front_alignment["welded_vertices"] = _weld_vertices(front)
    back = _import_join(config["back_glb"], "back_source")
    back_alignment = _level_and_normalize(back)
    back_alignment["welded_vertices"] = _weld_vertices(back)
    _apply_world_yaw(back, math.pi)
    back_alignment["canonical_yaw_degrees"] = 180.0
    back_alignment["canonical_yaw_axis"] = "world_z"
    _trim(front, axis="y", positive=True)
    _trim(back, axis="y", positive=False)
    sources = [front, back]
    side_alignment = {}
    for view, angle, positive in (("left", math.pi / 2, False), ("right", -math.pi / 2, True)):
        path = config.get(f"{view}_glb")
        if not path:
            continue
        side = _import_join(path, f"{view}_source")
        side_alignment[view] = _level_and_normalize(side)
        _apply_world_yaw(side, angle)
        side_alignment[view]["canonical_yaw_degrees"] = math.degrees(angle)
        side_alignment[view]["canonical_yaw_axis"] = "world_z"
        _trim(side, axis="x", positive=positive)
        sources.append(side)
    source_shell = _join(sources, "source_shell")
    source_cleanup = {"skipped_until_after_volume_fusion": True}
    fused = source_shell.copy()
    fused.data = source_shell.data.copy()
    bpy.context.collection.objects.link(fused)
    fused.name = "master"
    source_shell.hide_render = True
    source_shell.hide_set(True)
    bpy.context.view_layer.objects.active = fused
    fused.select_set(True)
    fused.data.remesh_voxel_size = float(config.get("voxel_size", 0.006))
    bpy.ops.object.voxel_remesh()
    voxel_cleanup = _remove_small_components(fused)
    topology_stages = {"post_voxel_remesh": _mesh_metrics(fused)}
    _repair_mesh(fused)
    topology_stages["post_voxel_repair"] = _mesh_metrics(fused)
    shrinkwrap = fused.modifiers.new("Recover trusted detail", "SHRINKWRAP")
    shrinkwrap.target = source_shell
    shrinkwrap.wrap_method = "NEAREST_SURFACEPOINT"
    shrinkwrap.wrap_mode = "ON_SURFACE"
    detail_group = fused.vertex_groups.new(name="Trusted detail influence")
    detail_group.add(list(range(len(fused.data.vertices))), 0.25, "REPLACE")
    shrinkwrap.vertex_group = detail_group.name
    bpy.ops.object.modifier_apply(modifier=shrinkwrap.name)
    remaining_group = fused.vertex_groups.get("Trusted detail influence")
    if remaining_group is not None:
        fused.vertex_groups.remove(remaining_group)
    topology_stages["post_detail_recovery"] = _mesh_metrics(fused)
    _repair_mesh(fused)
    topology_stages["post_detail_repair"] = _mesh_metrics(fused)
    source_voxel_size = float(config.get("voxel_size", 0.006))
    master_voxel_size = _target_faces(
        fused, int(config["master_faces"]), source_voxel_size
    )
    topology_stages["post_master_target"] = _mesh_metrics(fused)
    _smart_uv(fused, 16 / int(config["texture_resolution"]))
    material = _projection_material(config["front_image"], config["back_image"])
    fused.data.materials.clear()
    fused.data.materials.append(material)
    resolution = int(config["texture_resolution"])
    _bake(fused, material, texture_dir / "baseColor.png", resolution, "DIFFUSE")
    _bake(fused, material, texture_dir / "ao.png", resolution, "AO")

    objects = {"master": fused}
    for name, target in (
        ("game", int(config["game_faces"])),
        ("lod1", int(config["lod_faces"][0])),
        ("lod2", int(config["lod_faces"][1])),
    ):
        duplicate = fused.copy()
        duplicate.data = fused.data.copy()
        bpy.context.collection.objects.link(duplicate)
        duplicate.name = name
        _target_faces(duplicate, target, master_voxel_size)
        _transfer_uv(fused, duplicate)
        objects[name] = duplicate
    _normal_bake(fused, objects["game"], texture_dir / "normal.png", resolution)
    for name, obj in objects.items():
        _export(obj, job / f"{name}.glb")
    blend_path = job / "working.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    front_extent = front_alignment["source_extent"]
    back_extent = back_alignment["source_extent"]
    extent_error = max(
        abs(float(a) - float(b)) / max(float(a), float(b), 1e-9)
        for a, b in zip(front_extent, back_extent, strict=True)
    )
    return {
        "alignment": {
            "front": front_alignment,
            "back": back_alignment,
            "sides": side_alignment,
            "normalized_extent_error": extent_error,
            "side_agreement": max(0.0, 1.0 - extent_error),
        },
        "geometry": {name: _mesh_metrics(obj) for name, obj in objects.items()},
        "uv": _uv_metrics(fused),
        "topology_stages": topology_stages,
        "fragment_cleanup": {"source": source_cleanup, "voxel": voxel_cleanup},
        "working_blend": str(blend_path),
    }


def _analyze(config: dict) -> dict:
    _clear()
    profiles = {}
    for view in ("front", "back", "left", "right"):
        path = config.get(f"{view}_glb")
        if not path:
            continue
        obj = _import_join(path, f"{view}_analysis")
        unwelded = _component_profile(obj)
        welded_vertices = _weld_vertices(obj)
        profiles[view] = {
            **_component_profile(obj),
            "unwelded_components": unwelded["components"],
            "welded_vertices": welded_vertices,
        }
        _bpy().data.objects.remove(obj, do_unlink=True)
    return {
        "passed": bool(profiles) and all(profile["passed"] for profile in profiles.values()),
        "profiles": profiles,
    }


def _finalize(config: dict) -> dict:
    bpy = _bpy()
    job = Path(config["job_dir"])
    bpy.ops.wm.open_mainfile(filepath=str(job / "working.blend"))
    maps = config["maps"]
    geometry = {}
    for name in ("master", "game", "lod1", "lod2"):
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise RuntimeError(f"Missing {name} object in working.blend")
        _configure_final_material(obj, job / "textures", maps)
        _export(obj, job / f"{name}.glb")
        geometry[name] = _mesh_metrics(obj)
    game = bpy.data.objects["game"]
    _render_previews(game, job / "previews")
    bpy.ops.wm.save_as_mainfile(filepath=str(job / "working.blend"))
    return {"geometry": geometry, "previews": str((job / "previews").resolve())}


def main() -> None:
    config = _arguments()
    if config["mode"] == "analyze":
        result = _analyze(config)
    elif config["mode"] == "build":
        result = _build(config)
    else:
        result = _finalize(config)
    output = Path(config["result_path"])
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
