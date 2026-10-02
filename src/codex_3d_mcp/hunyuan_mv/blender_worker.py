"""Headless Blender cleanup, identity-transform export, LODs, and mesh metrics."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path


def _args() -> tuple[Path, Path]:
    marker = sys.argv.index("--")
    return Path(sys.argv[marker + 1]), Path(sys.argv[marker + 2])


def _clear(bpy) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _import_join(bpy, path: Path):
    existing = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [
        obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj not in existing
    ]
    if not meshes:
        raise RuntimeError("The input GLB contains no mesh objects.")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.hide_set(False)
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    if obj.parent:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return obj


def _normalize(bpy, obj, *, validate=True) -> None:
    # Blender 5.1 exposes bound-box corners as ``bpy_prop_array`` values,
    # which no longer participate directly in Matrix multiplication.
    from mathutils import Vector

    corners = [obj.matrix_world @ Vector(vector) for vector in obj.bound_box]
    low = [min(point[index] for point in corners) for index in range(3)]
    high = [max(point[index] for point in corners) for index in range(3)]
    center = [(low[index] + high[index]) / 2 for index in range(3)]
    obj.location.x -= center[0]
    obj.location.y -= center[1]
    obj.location.z -= low[2]
    largest = max(high[index] - low[index] for index in range(3))
    if largest > 0:
        obj.scale = (2 / largest,) * 3
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if validate:
        obj.data.validate(verbose=False, clean_customdata=False)
    obj.data.update()


def _weld_geometry(obj) -> None:
    """Weld glTF attribute-split vertices without flattening loop-domain UVs."""
    import bmesh

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-6)
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate(verbose=False, clean_customdata=False)
    mesh.update()


def _decimate_to_target(bpy, obj, target: int):
    faces = len(obj.data.polygons)
    if faces > target:
        modifier = obj.modifiers.new("Production decimation", "DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = max(0.001, target / faces)
        modifier.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.data.validate(verbose=False, clean_customdata=False)
    obj.data.update()
    return obj


def _duplicate_lod(bpy, source, name: str, target: int):
    duplicate = source.copy()
    duplicate.data = source.data.copy()
    duplicate.name = name
    bpy.context.collection.objects.link(duplicate)
    if len(duplicate.data.polygons) > target:
        # Collapse a connected geometric shell, not disconnected glTF seam
        # vertices. Welding this copy retains loop UVs and leaves master intact.
        _weld_geometry(duplicate)
    _decimate_to_target(bpy, duplicate, target)
    return duplicate


def _metrics(obj) -> dict:
    import bmesh

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    attribute_split_edges = sum(1 for edge in bm.edges if len(edge.link_faces) != 2)
    # glTF import duplicates vertices at UV/material seams. Those edges are
    # topologically open in Blender's attribute-split mesh even when the
    # geometric shell is closed. Weld only the metrics copy at a microscopic
    # tolerance so QA measures the surface, without destroying production UVs.
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-6)
    degenerate = sum(1 for face in bm.faces if face.calc_area() <= 1e-12)
    boundary = sum(1 for edge in bm.edges if len(edge.link_faces) != 2)
    metrics = {
        "vertices": len(mesh.vertices),
        "faces": sum(len(face.vertices) - 2 for face in mesh.polygons),
        "degenerate_faces": degenerate,
        "non_manifold_edges": boundary,
        "attribute_split_non_manifold_edges": attribute_split_edges,
        "watertight": boundary == 0,
        "location": list(obj.location),
        "rotation_euler": list(obj.rotation_euler),
        "scale": list(obj.scale),
    }
    bm.free()
    return metrics


def _require_closed_variants(geometry: dict) -> None:
    for name in ("game", "lod1", "lod2"):
        metrics = geometry.get(name, {})
        if not metrics.get("watertight", False) or metrics.get("non_manifold_edges") != 0:
            raise ValueError(
                f"{name} export is not a closed mesh: "
                f"watertight={metrics.get('watertight')}, "
                f"non_manifold_edges={metrics.get('non_manifold_edges')}. "
                "Inspect the retained variant and repair it before accepting final exports."
            )


def _export_one(bpy, objects: list, active, path: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(obj == active)
    bpy.context.view_layer.objects.active = active
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
    )


def _require_gpu_baking(bpy) -> dict:
    """Fail closed if Cycles cannot select an actual GPU; never enable CPU render."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    failures = []
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            devices = []
            for device in prefs.devices:
                device.use = device.type == backend
                if device.use:
                    devices.append(device.name)
            if devices:
                scene = bpy.context.scene
                scene.render.engine = "CYCLES"
                scene.cycles.device = "GPU"
                scene.cycles.samples = 1
                scene.render.threads_mode = "FIXED"
                scene.render.threads = 2
                return {"backend": backend, "devices": devices, "cpu_enabled": False}
        except (RuntimeError, TypeError, ValueError) as exc:
            failures.append(f"{backend}: {exc}")
    raise RuntimeError(f"A supported GPU is required for texture baking. {failures}")


def _emission_material(bpy, name, *, original=None, white=False):
    material = original.copy() if original else bpy.data.materials.new(name)
    material.name = name
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    emission = nodes.new("ShaderNodeEmission")
    emission.name = "Base Color Bake Emission"
    emission.inputs["Color"].default_value = (1, 1, 1, 1)
    if shader and not white:
        color = shader.inputs["Base Color"]
        if color.is_linked:
            links.new(color.links[0].from_socket, emission.inputs["Color"])
        else:
            emission.inputs["Color"].default_value = color.default_value
    output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if output is None:
        output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs[0], output.inputs["Surface"])
    return material


def _bake_call(
    bpy, source, target, image, *, node, selected_to_active, extrusion, distance, margin=8
):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.hide_render = obj not in (source, target)
    target.hide_set(False)
    target.select_set(True)
    source.hide_set(False)
    source.select_set(True)
    bpy.context.view_layer.objects.active = target
    node.image = image
    target.active_material.node_tree.nodes.active = node
    bpy.ops.object.bake(
        type="EMIT",
        use_selected_to_active=selected_to_active,
        use_clear=True,
        margin=margin,
        use_cage=False,
        cage_extrusion=extrusion,
        max_ray_distance=distance,
    )


def _bake_profile(bpy, source, target, directory: Path, resolution: int) -> dict:
    """Bake base color and measure ray coverage against the target's own UV mask."""
    import numpy as np
    from mathutils import Vector

    source_materials = list(source.data.materials)
    if not source_materials or any(mat is None for mat in source_materials):
        raise RuntimeError("Painted source is missing a material")
    # Color baking does not transfer the other material factors. Preserve the
    # Paint material's constants so an LOD switch does not change its shading.
    factors = None
    for source_material in source_materials:
        if not source_material.use_nodes:
            raise RuntimeError("Painted source requires a node material")
        source_shader = next(
            (n for n in source_material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None
        )
        if source_shader is None:
            raise RuntimeError("Painted source requires a Principled material")
        current = {}
        for key in ("Metallic", "Roughness", "IOR", "Alpha"):
            socket = source_shader.inputs[key]
            if socket.is_linked:
                raise RuntimeError(f"Texture transfer requires constant source {key}")
            current[key] = float(socket.default_value)
        if factors is not None and current != factors:
            raise RuntimeError("Source material factors differ; a material-map bake is required")
        factors = current
    print(f"[profiles] unwrap and bake {target.name} at {resolution}px", flush=True)
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    # Topology cleanup precedes UV generation and is never applied after baking.
    _weld_geometry(target)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.015)
    bpy.ops.object.mode_set(mode="OBJECT")
    if not target.data.uv_layers.active:
        raise RuntimeError(f"{target.name} UV unwrap produced no UV map")
    material = bpy.data.materials.new(f"{target.name} base color")
    material.use_nodes = True
    material.use_backface_culling = source_materials[0].use_backface_culling
    target.data.materials.clear()
    target.data.materials.append(material)
    shader = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    for key, value in factors.items():
        shader.inputs[key].default_value = value
    image = bpy.data.images.new(f"{target.name}_base_color", resolution, resolution, alpha=True)
    image.generated_color = (0, 0, 0, 0)
    image.colorspace_settings.name = "sRGB"
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    corners = [source.matrix_world @ Vector(p) for p in source.bound_box]
    diag = Vector(
        [max(p[i] for p in corners) - min(p[i] for p in corners) for i in range(3)]
    ).length
    extrusion, distance = diag * 0.01, diag * 0.04
    color_materials = [
        _emission_material(bpy, f"bake color {i}", original=mat)
        for i, mat in enumerate(source_materials)
    ]
    masks = []
    try:
        for i, mat in enumerate(color_materials):
            source.data.materials[i] = mat
        _bake_call(
            bpy,
            source,
            target,
            image,
            node=node,
            selected_to_active=True,
            extrusion=extrusion,
            distance=distance,
        )
        directory.mkdir(parents=True, exist_ok=True)
        image.filepath_raw = str(directory / f"{target.name}_base_color.png")
        image.file_format = "PNG"
        image.save()
        image.pack()
        # Legitimate black paint must not be confused with a failed ray. Bake
        # white source and white target masks as independent coverage evidence.
        for mat in color_materials:
            emission = mat.node_tree.nodes["Base Color Bake Emission"]
            for link in list(emission.inputs["Color"].links):
                mat.node_tree.links.remove(link)
            emission.inputs["Color"].default_value = (1, 1, 1, 1)
        projected = bpy.data.images.new(
            f"{target.name}_projected_mask", resolution, resolution, alpha=True
        )
        projected.generated_color = (0, 0, 0, 0)
        masks.append(projected)
        _bake_call(
            bpy,
            source,
            target,
            projected,
            node=node,
            selected_to_active=True,
            extrusion=extrusion,
            distance=distance,
            margin=0,
        )
        white_target = _emission_material(bpy, "target UV coverage", white=True)
        target.data.materials[0] = white_target
        mask_node = white_target.node_tree.nodes.new("ShaderNodeTexImage")
        expected = bpy.data.images.new(f"{target.name}_uv_mask", resolution, resolution, alpha=True)
        expected.generated_color = (0, 0, 0, 0)
        masks.append(expected)
        _bake_call(
            bpy,
            target,
            target,
            expected,
            node=mask_node,
            selected_to_active=False,
            extrusion=0,
            distance=0,
            margin=0,
        )
        arrays = []
        for mask in masks:
            values = np.empty(resolution * resolution * 4, dtype=np.float32)
            mask.pixels.foreach_get(values)
            arrays.append(values.reshape((-1, 4)))
            mask.filepath_raw = str(directory / f"{mask.name}.png")
            mask.file_format = "PNG"
            mask.save()
        wanted = arrays[1][:, 0] > 0.9
        covered = arrays[0][:, 0] > 0.9
        total = int(wanted.sum())
        misses = int(np.count_nonzero(wanted & ~covered))
        coverage = 1 - misses / max(1, total)
        result = {
            "resolution": resolution,
            "expected_texels": total,
            "missed_texels": misses,
            "coverage": coverage,
            "passed": total > 0 and coverage >= 0.995,
            "cage_extrusion": extrusion,
            "max_ray_distance": distance,
            "source": source.name,
            "material_factors": factors,
            "texture": image.filepath_raw,
            "method": "GPU selected-to-active emission/base-color, no scene lighting",
        }
        (directory / f"{target.name}_bake.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        if not result["passed"]:
            raise RuntimeError(f"{target.name} texture projection coverage failed: {result}")
        return result
    finally:
        for i, mat in enumerate(source_materials):
            source.data.materials[i] = mat
        target.data.materials[0] = material
        node.image = image
        material.node_tree.links.new(node.outputs["Color"], shader.inputs["Base Color"])
        material.node_tree.nodes.active = node


def _finish_profiles(bpy, config, result_path):
    from mathutils import Matrix, Vector

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = config["profile_meshes"]
    if set(paths) != {"full_game", "mobile", "browser"}:
        raise RuntimeError("All three accepted remesh profile paths are required")
    full = _import_join(bpy, Path(config["input_glb"]))
    full.name = "full_game"
    objects = [full]
    for name in ("mobile", "browser"):
        obj = _import_join(bpy, Path(paths[name]))
        obj.name = name
        objects.append(obj)
    gpu = _require_gpu_baking(bpy)
    bakes = {}
    for obj in objects[1:]:
        bakes[obj.name] = _bake_profile(bpy, full, obj, output_dir / "textures", 1024)
    # One common affine transform preserves correspondence between profiles.
    corners = [full.matrix_world @ Vector(p) for p in full.bound_box]
    low = [min(p[i] for p in corners) for i in range(3)]
    high = [max(p[i] for p in corners) for i in range(3)]
    factor = 2 / max(high[i] - low[i] for i in range(3))
    offset = Vector((-(low[0] + high[0]) / 2, -(low[1] + high[1]) / 2, -low[2]))
    transform = Matrix.Scale(factor, 4) @ Matrix.Translation(offset)
    for obj in objects:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        obj.matrix_world = transform @ obj.matrix_world
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    geometry = {obj.name: _metrics(obj) for obj in objects}
    for name, value in geometry.items():
        if not value["watertight"] or value["non_manifold_edges"] or value["degenerate_faces"]:
            raise RuntimeError(f"{name} has invalid geometry after texture transfer: {value}")
        if value["faces"] > int(config["profile_triangle_budgets"][name]):
            raise RuntimeError(f"{name} exceeds its exported triangle budget")
    for obj in objects:
        _export_one(bpy, objects, obj, output_dir / f"{obj.name}.glb")
        _render_previews(bpy, obj, output_dir / "previews" / obj.name)
    # Alias files preserve existing clients without introducing another mesh.
    aliases = {"master": "full_game", "game": "full_game", "lod1": "mobile", "lod2": "browser"}
    for alias, name in aliases.items():
        shutil.copyfile(output_dir / f"{name}.glb", output_dir / f"{alias}.glb")
        geometry[alias] = dict(geometry[name])
    for view in ("front", "back", "left", "right"):
        shutil.copyfile(
            output_dir / "previews" / "full_game" / f"{view}.png",
            output_dir / "previews" / f"{view}.png",
        )
    for obj in objects:
        obj.hide_render = obj != full
        obj.hide_set(obj != full)
        obj.select_set(obj == full)
    bpy.context.view_layer.objects.active = full
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "working.blend"))
    result_path.write_text(
        json.dumps(
            {
                "worker_version": 5,
                "prepared_profiles": True,
                "geometry": geometry,
                "bakes": bakes,
                "gpu": gpu,
                "profile_triangle_budgets": config["profile_triangle_budgets"],
                "shared_transform": [list(row) for row in transform],
                "profiles_manifest": config["profiles_manifest"],
                "orientation": {"up": "+Y glTF", "front": "+Z glTF", "root_identity": True},
                "previews": str(output_dir / "previews"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _preview_camera_positions(world_corners):
    """Frame the actual bounds and match Hunyuan's conditioning view names."""
    x, y, z = (
        (min(point[axis] for point in world_corners) + max(point[axis] for point in world_corners))
        / 2
        for axis in range(3)
    )
    # Blender looks down local -Z with +Y as camera up. Hunyuan's left
    # reference presents a forward-facing feature toward image-left.
    return {
        "front": ((x, y - 4, z), (math.radians(90), 0, 0)),
        "back": ((x, y + 4, z), (math.radians(90), 0, math.radians(180))),
        "left": ((x + 4, y, z), (math.radians(90), 0, math.radians(90))),
        "right": ((x - 4, y, z), (math.radians(90), 0, math.radians(-90))),
    }


def _render_previews(bpy, obj, directory: Path, *, include_underside=False) -> None:
    from mathutils import Vector

    directory.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    # Export selection does not affect rendering. Only this variant is visible.
    for candidate in scene.objects:
        if candidate.type == "MESH":
            candidate.hide_render = candidate != obj
    # Review previews use the lightweight GPU Workbench renderer. Eevee's
    # light-probe shader compilation can exhaust a capped preview process.
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.show_shadows = False
    for slot in obj.material_slots:
        material = slot.material
        if material and material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    material.node_tree.nodes.active = node
                    break
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world.color = (0.04, 0.04, 0.04)
    camera_data = bpy.data.cameras.new("QA Camera")
    camera = bpy.data.objects.new("QA Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.5
    light_data = bpy.data.lights.new("QA Key", "AREA")
    light_data.energy = 1000
    light_data.shape = "DISK"
    light_data.size = 5
    light = bpy.data.objects.new("QA Key", light_data)
    light.location = (3, -4, 5)
    scene.collection.objects.link(light)
    world_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    positions = _preview_camera_positions(world_corners)
    if include_underside:
        center = sum(world_corners, Vector()) / len(world_corners)
        for name, offset in {
            "bottom": (0, 0, -4),
            "underside_front_left": (3, -4, -3),
            "underside_back_right": (-3, 4, -3),
        }.items():
            location = center + Vector(offset)
            rotation = (center - location).to_track_quat("-Z", "Y").to_euler()
            positions[name] = (location, rotation)
    framing = {}
    for name, (location, rotation) in positions.items():
        camera.location = location
        camera.rotation_euler = rotation
        # Oblique underside views span the diagonal of the bounding box. Fit
        # projected corners separately, keeping a measurable empty border.
        inverse_rotation = camera.rotation_euler.to_matrix().transposed()
        projected = [inverse_rotation @ (corner - camera.location) for corner in world_corners]
        width = 2 * max(abs(point.x) for point in projected)
        height = 2 * max(abs(point.y) for point in projected)
        aspect = scene.render.resolution_x / scene.render.resolution_y
        camera_data.ortho_scale = max(height, width / aspect, 1e-6) * 1.12
        coverage = max(height / camera_data.ortho_scale, width / (camera_data.ortho_scale * aspect))
        framing[name] = {"passed": coverage <= 0.9, "bounding_box_frame_fraction": coverage,
                         "ortho_scale": camera_data.ortho_scale}
        scene.render.filepath = str(directory / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    return framing


def main() -> None:
    import bpy

    config_path, result_path = _args()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _clear(bpy)
    if config.get("prepared_profiles") and not config.get("preview_only"):
        _finish_profiles(bpy, config, result_path)
        return
    master = _import_join(bpy, Path(config["input_glb"]))
    master.name = "master"
    # Remeshing owns topology. Final placement still grounds the centered
    # remesh and bakes export transforms without changing UVs or connectivity.
    exact_preview = bool(config.get("exact_candidate_preview"))
    _normalize(bpy, master, validate=not exact_preview)
    if config.get("preview_only", False):
        output_dir = Path(config["output_dir"])
        framing = _render_previews(
            bpy, master, output_dir / "previews", include_underside=exact_preview
        )
        result_path.write_text(
            json.dumps({
                "geometry": _metrics(master), "previews": str(output_dir / "previews"),
                "preview_only": True, "implicit_topology_cleanup": not exact_preview,
                "geometry_metrics_are_acceptance_gate": False,
                "framing": framing,
            }),
            encoding="utf-8",
        )
        return
    if config.get("prepared_master", False):
        raise RuntimeError(
            "This batch requires three remeshed profiles before final export. "
            "Reload the updated MCP server and retry the remesh stage."
        )
    if not config.get("prepared_master", False):
        _weld_geometry(master)
        _decimate_to_target(bpy, master, int(config.get("master_faces", 300_000)))
    game = _duplicate_lod(bpy, master, "game", int(config["game_faces"]))
    lod1 = _duplicate_lod(bpy, master, "lod1", int(config["lod_faces"][0]))
    lod2 = _duplicate_lod(bpy, master, "lod2", int(config["lod_faces"][1]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = [master, game, lod1, lod2]
    for obj in objects:
        _export_one(bpy, objects, obj, output_dir / f"{obj.name}.glb")
    geometry = {obj.name: _metrics(obj) for obj in objects}
    _require_closed_variants(geometry)
    for obj in (master, lod1, lod2):
        _render_previews(bpy, obj, output_dir / "previews" / obj.name)
    # Keep the established game preview paths and save game as the visible variant.
    _render_previews(bpy, game, output_dir / "previews")
    for obj in objects:
        obj.hide_set(obj != game)
        obj.select_set(obj == game)
    bpy.context.view_layer.objects.active = game
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "working.blend"))
    result = {
        "worker_version": 4,
        "geometry": geometry,
        "orientation": {"up": "+Y glTF", "front": "+Z glTF", "root_identity": True},
        "previews": str((output_dir / "previews").resolve()),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
