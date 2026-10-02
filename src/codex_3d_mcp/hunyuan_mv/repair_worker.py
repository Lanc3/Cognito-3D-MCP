"""File-driven immutable repair worker. The caller owns lease and CPU/memory limits.

Run: python -m codex_3d_mcp.hunyuan_mv.repair_worker --request request.json
Every candidate needs separate visual/semantic approval, even when this gate passes.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .repair_contract import REPAIR_GATE_VERSION, get_capabilities, validate_recipe


class RepairCheckError(RuntimeError):
    """Preserve a backend's typed failure while propagating a required check."""

    def __init__(self, message: str, check: dict[str, Any]):
        super().__init__(message)
        self.failure_kind = check.get("failure_kind") or "geometry_rejected"
        self.code = check.get("code") or "REPAIR_QUALITY_GATES_FAILED"
        self.check = check


def classify_failure(exc: Exception) -> dict[str, str]:
    """Machine retry category independent of exception-message spelling."""
    if isinstance(exc, RepairCheckError):
        kind, code = exc.failure_kind, exc.code
    elif type(exc).__module__ == __package__ + ".cumesh_repair" and getattr(
            exc, "failure_kind", None) in {"runtime_unavailable", "resource"}:
        kind, code = exc.failure_kind, exc.code
    elif isinstance(exc, ImportError):
        kind, code = "runtime_unavailable", "REPAIR_BACKEND_UNAVAILABLE"
    elif isinstance(exc, MemoryError) or (isinstance(exc, OSError) and (
            exc.errno in {errno.ENOMEM, errno.ENOSPC, errno.EAGAIN} or
            getattr(exc, "winerror", None) in {8, 14, 1455})):
        kind, code = "resource", "REPAIR_RESOURCE_EXHAUSTED"
    elif isinstance(exc, ValueError):
        kind, code = "geometry_rejected", "REPAIR_INVALID_GEOMETRY"
    else:
        kind, code = "runtime_failure", "REPAIR_RUNTIME_FAILURE"
    return {"failure_kind": kind, "code": code}


def _require_check(check: dict[str, Any], message: str) -> None:
    if not (check.get("evaluated") and check.get("passed")):
        raise RepairCheckError(message, check)


def _finalize_failure_kind(gate: dict[str, Any]) -> None:
    """Keep operational blockers retryable even if other quality checks also fail."""
    failures = []
    for check in gate["checks"]:
        if not check.get("passed") or not check.get("evaluated"):
            check.setdefault("failure_kind", "geometry_rejected")
            check.setdefault("code", "REPAIR_QUALITY_GATES_FAILED")
            failures.append(check)
    gate["failure_kind"], gate["failure_code"] = None, None
    for kind in ("resource", "runtime_unavailable", "runtime_failure", "geometry_rejected"):
        matching = next((check for check in failures if check["failure_kind"] == kind), None)
        if matching is not None:
            gate["failure_kind"], gate["failure_code"] = kind, matching["code"]
            break


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _copy_new(source: Path, destination: Path) -> None:
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def _load(path: Path) -> tuple[Any, Any]:
    import numpy as np
    import trimesh
    from .repair_geometry import checked_arrays

    if path.suffix.lower() != ".glb" or path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("Only GLB inputs up to 256 MiB are accepted")
    with path.open("rb") as stream:
        if stream.read(4) != b"glTF":
            raise ValueError("Input does not have a GLB header")
    # force=mesh bakes scene instances/transforms. No welding or normalization.
    mesh = trimesh.load(path, force="mesh", process=False, skip_materials=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected triangle mesh geometry")
    vertices, faces = checked_arrays(mesh.vertices.copy(), mesh.faces.copy())
    with np.errstate(over="ignore"):
        if not np.isfinite(vertices.astype(np.float32)).all():
            raise ValueError("World coordinates cannot be represented by GLB float32")
    return vertices, faces


def _export(path: Path, vertices: Any, faces: Any) -> None:
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    payload = mesh.export(file_type="glb")
    with path.open("xb") as stream:
        stream.write(payload)


def _cleanup(vertices: Any, faces: Any, parameters: dict[str, Any], roi: dict[str, Any],
             steps: list[dict[str, Any]]) -> tuple[Any, Any]:
    import numpy as np
    from .repair_geometry import areas, inside

    vertices, faces = vertices.copy(), faces.copy()
    original_faces = len(faces)
    if parameters["exact_weld"]:
        before = len(vertices)
        vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        faces = inverse[faces]
        steps.append({"operation": "exact_coordinate_weld", "vertices_before": before,
                      "vertices_after": len(vertices), "maximum_displacement": 0.0})
    tolerance = parameters["tolerance_weld"]
    if tolerance is not None:
        distance = tolerance["distance"]
        diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        if distance > diagonal * 0.001:
            raise ValueError("Tolerance weld distance exceeds 0.1% of asset diagonal")
        selected = np.flatnonzero(inside(vertices, roi))
        # Grid bins bound every merged pair's Euclidean separation by distance;
        # this deliberately avoids transitive radius clustering across thin parts.
        width = distance / np.sqrt(3.0)
        bins = np.floor((vertices[selected] - np.asarray(roi["min"])) / width)
        if not np.isfinite(bins).all() or np.abs(bins).max(initial=0) > 2**52:
            raise ValueError("Weld grid exceeds safe integer precision")
        _, first, inverse = np.unique(bins.astype(np.int64), axis=0,
                                      return_index=True, return_inverse=True)
        replacements = vertices[selected[first[inverse]]]
        displacement = np.linalg.norm(replacements - vertices[selected], axis=1)
        maximum = float(displacement.max(initial=0))
        if maximum > tolerance["max_displacement"]:
            raise ValueError("Measured weld displacement exceeds authored maximum")
        vertices[selected] = replacements
        before = len(vertices)
        vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        faces = inverse[faces]
        steps.append({"operation": "explicit_tolerance_grid_weld", "distance": distance,
                      "maximum_displacement": maximum, "vertices_before": before,
                      "vertices_after": len(vertices), "region": roi,
                      "method": "same grid bin, first coordinate retained; no transitive radius union"})
    if parameters["remove_duplicate_faces"]:
        _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
        duplicates = np.ones(len(faces), dtype=bool)
        duplicates[first] = False
        remove = duplicates & inside(vertices, roi)[faces].all(axis=1)
        faces = faces[~remove]
        steps.append({"operation": "remove_exact_duplicate_faces_in_roi",
                      "removed_faces": int(remove.sum())})
    if parameters["remove_collapsed_faces"]:
        remove = (areas(vertices, faces) == 0) & inside(vertices, roi)[faces].all(axis=1)
        faces = faces[~remove]
        steps.append({"operation": "remove_exact_zero_area_faces_in_roi",
                      "removed_faces": int(remove.sum()), "near_degenerate_faces_removed": False})
    removed = original_faces - len(faces)
    if removed > parameters["max_removed_faces"] or removed / original_faces > parameters["max_removed_fraction"]:
        raise ValueError("Cleanup exceeded authored face-removal bound")
    if len(faces) < 4:
        raise ValueError("Cleanup left fewer than four faces")
    if parameters["fix_winding"]:
        import trimesh
        if not inside(vertices, roi)[faces].all():
            raise ValueError("Winding correction requires the whole corrected mesh inside ROI")
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        mesh.fix_normals(multibody=True)
        changed = int(np.any(mesh.faces != faces, axis=1).sum())
        faces = mesh.faces.copy()
        steps.append({"operation": "fix_winding_multibody", "changed_faces": changed})
    used, inverse = np.unique(faces, return_inverse=True)
    steps.append({"operation": "remove_unreferenced_vertices", "removed_vertices": len(vertices) - len(used)})
    return vertices[used], inverse.reshape((-1, 3))


def _patch(vertices: Any, faces: Any, diagnostics: dict[str, Any], info: dict[str, Any],
           parameters: dict[str, Any], roi: dict[str, Any], steps: list[dict[str, Any]]) -> tuple[Any, Any]:
    import numpy as np
    from .repair_geometry import areas, inside

    if diagnostics["topology"]["boundary_loop_error"]:
        raise ValueError("Boundary graph is not an enumerable collection of simple loops")
    lookup = {loop["id"]: loop for loop in diagnostics["loops"]}
    chosen = []
    for identifier in parameters["loop_ids"]:
        if identifier not in lookup:
            raise ValueError(f"Unknown source loop ID: {identifier}")
        loop = lookup[identifier]
        if loop["edge_count"] > parameters["max_edges"] or loop["diameter_upper_bound"] > parameters["max_diameter"]:
            raise ValueError("Selected loop exceeds authored edge or diameter bound")
        if not inside(np.asarray(loop["vertices"]), roi).all():
            raise ValueError("Selected loop extends outside repair ROI")
        chosen.append(loop)
    vertices = info["canonical_vertices"].copy()
    faces = info["canonical_faces"].copy()
    original_faces = len(faces)
    boundary = {tuple(edge) for edge in info["boundary_edges"].tolist()}
    if parameters["backend"] == "triangle":
        patches = []
        for loop in chosen:
            if loop["edge_count"] != 3:
                raise ValueError("Triangle patch backend accepts exactly three-edge loops")
            triangle = list(loop["vertex_indices"])
            if (triangle[0], triangle[1]) in boundary:
                triangle.reverse()
            if any((triangle[i], triangle[(i + 1) % 3]) in boundary or
                   (triangle[(i + 1) % 3], triangle[i]) not in boundary for i in range(3)):
                raise ValueError("Selected boundary winding is inconsistent")
            patch_area = float(areas(vertices, np.asarray([triangle]))[0])
            if patch_area < parameters["min_triangle_area"]:
                raise ValueError("Patch would recreate a near-degenerate triangle; retriangulation required")
            patches.append(triangle)
        faces = np.vstack([faces, np.asarray(patches, dtype=np.int64)])
    else:
        import pymeshlab

        # Select adjacent faces, but first ensure no other hole touches that set.
        chosen_edges = {tuple(sorted((loop["vertex_indices"][i],
                         loop["vertex_indices"][(i + 1) % loop["edge_count"]])))
                        for loop in chosen for i in range(loop["edge_count"])}
        selection = np.zeros(len(faces), dtype=bool)
        for a, b in chosen_edges:
            selection |= (faces == a).any(axis=1) & (faces == b).any(axis=1)
        for loop in diagnostics["loops"]:
            if loop["id"] in parameters["loop_ids"]:
                continue
            for i, a in enumerate(loop["vertex_indices"]):
                b = loop["vertex_indices"][(i + 1) % loop["edge_count"]]
                if (selection & (faces == a).any(axis=1) & (faces == b).any(axis=1)).any():
                    raise ValueError("Selected boundary faces also touch an unapproved loop")
        ms = pymeshlab.MeshSet()
        if not all(callable(getattr(ms, method, None)) for method in (
                "compute_selection_by_condition_per_face", "meshing_close_holes")):
            raise ImportError("PyMeshLab selected-hole API unavailable")
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces.astype(np.int32),
                                  f_scalar_array=selection.astype(np.float64)))
        ms.compute_selection_by_condition_per_face(condselect="fq > 0.5")
        result = ms.meshing_close_holes(maxholesize=parameters["max_edges"], selected=True,
            newfaceselected=True, selfintersection=True, refinehole=False)
        mesh = ms.current_mesh()
        vertices, faces = mesh.vertex_matrix().copy(), mesh.face_matrix().astype(np.int64)
        if not np.array_equal(vertices, info["canonical_vertices"]) or not np.array_equal(
                faces[:original_faces], info["canonical_faces"]):
            raise ValueError("Hole backend changed pre-existing vertices/faces")
        steps.append({"operation": "pymeshlab_selected_hole_backend", "backend_result": result,
                      "refinement": False, "adjacent_intersection_heuristic": True})
    patches = faces[original_faces:]
    patch_areas = areas(vertices, patches)
    if not len(patches) or patch_areas.min() < parameters["min_triangle_area"]:
        raise ValueError("Empty or near-degenerate patch")
    if patch_areas.sum() > parameters["max_patch_area"] or not inside(vertices, roi)[patches].all():
        raise ValueError("Patch exceeds authored area or ROI bound")
    steps.append({"operation": "patch_selected_holes", "loop_ids": parameters["loop_ids"],
                  "added_faces": len(patches), "patch_area": float(patch_areas.sum()),
                  "minimum_triangle_area": float(patch_areas.min()),
                  "existing_vertex_displacement": 0.0})
    return vertices, faces


def _orient_new_patch_faces(patch: Any, source_boundary: Any) -> tuple[Any, dict[str, Any]]:
    """Orient only new faces, anchored to the unchanged directed source boundary.

    A patch edge must be either one selected boundary edge or an internal edge
    incident to two patch faces. Every patch component needs a boundary anchor;
    contradictory parity constraints are rejected rather than repairing source.
    """
    import numpy as np

    if not 0 < len(patch) <= 32768 or not 0 < len(source_boundary) <= 16384:
        raise ValueError("Patch orientation exceeds bounded selected-hole counts")
    boundary = {}
    for a, b in source_boundary:
        a, b = int(a), int(b)
        key = (min(a, b), max(a, b))
        if a == b or key in boundary:
            raise ValueError("Ambiguous directed source patch boundary")
        boundary[key] = 1 if a < b else -1
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for index, triangle in enumerate(patch):
        if len(set(map(int, triangle))) != 3:
            raise ValueError("Patch contains collapsed connectivity")
        for a, b in zip(triangle, np.roll(triangle, -1)):
            a, b = int(a), int(b)
            key = (min(a, b), max(a, b))
            edges.setdefault(key, []).append((index, 1 if a < b else -1))
    flips: dict[int, int] = {}
    neighbors: list[list[tuple[int, int]]] = [[] for _ in patch]
    for key, incident in edges.items():
        if key in boundary:
            if len(incident) != 1:
                raise ValueError("Selected boundary must meet exactly one new patch face")
            index, direction = incident[0]
            value = int(direction == boundary[key])
            if index in flips and flips[index] != value:
                raise ValueError("Contradictory patch orientation anchors")
            flips[index] = value
        else:
            if len(incident) != 2:
                raise ValueError("New patch has an open or nonmanifold internal edge")
            (a, direction_a), (b, direction_b) = incident
            parity = int(direction_a == direction_b)
            neighbors[a].append((b, parity))
            neighbors[b].append((a, parity))
    if not boundary.keys() <= edges.keys():
        raise ValueError("Patch omits selected source boundary edges")
    pending = list(flips)
    while pending:
        index = pending.pop()
        for other, parity in neighbors[index]:
            value = flips[index] ^ parity
            if other in flips:
                if flips[other] != value:
                    raise ValueError("Contradictory patch orientation constraints")
            else:
                flips[other] = value
                pending.append(other)
    if len(flips) != len(patch):
        raise ValueError("New patch contains a component without a source boundary anchor")
    oriented = np.asarray(patch, dtype=np.int64).copy()
    changed = [index for index, flip in flips.items() if flip]
    oriented[changed] = oriented[changed, ::-1]
    return oriented, {"method": "new_face_parity_anchored_to_directed_source_boundary_v1",
        "patch_faces": len(patch), "flipped_new_faces": len(changed),
        "anchored_boundary_edges": len(boundary), "existing_faces_changed": 0,
        "vertices_changed": 0, "blanket_normal_fix": False,
        "native_patch_indices_sha256": hashlib.sha256(np.asarray(patch, dtype="<i8").tobytes()).hexdigest(),
        "oriented_patch_indices_sha256": hashlib.sha256(oriented.astype("<i8", copy=False).tobytes()).hexdigest()}


def _cumesh_candidate(diagnostics: dict[str, Any], info: dict[str, Any], recipe: dict[str, Any],
                      report: dict[str, Any]) -> tuple[Any, Any]:
    """Translate hash-bound stable IDs into current canonical edges; keep all gates."""
    import numpy as np
    from . import cumesh_repair as adapter
    from .repair_cumesh_qualification import discover
    from .repair_geometry import areas, checked_arrays, geometry_digest, inside

    qualification = discover(list(sys.path), gate_version=REPAIR_GATE_VERSION, fresh=True)
    if not qualification["cuda_qualified"]:
        raise adapter.CuMeshUnavailable(qualification["reason"])
    report["gpu_qualification_binding"] = qualification["qualification_binding"]
    method, parameters, policy = recipe["method"], recipe["parameters"], recipe["policy"]
    # GLB positions are serialized as float32. Canonicalize that exact frame once,
    # preserving source diagnostic index correspondence through the inverse map.
    source_vertices = info["canonical_vertices"]
    vertices, inverse = np.unique(np.asarray(source_vertices, dtype=np.float32),
                                  axis=0, return_inverse=True)
    if len(vertices) != len(source_vertices):
        raise ValueError("Float32 canonicalization merges distinct source points; GPU recipe rejected")
    vertices = vertices.astype(np.float64)
    faces = np.asarray(inverse[info["canonical_faces"]], dtype=np.int64)
    checked_arrays(vertices, faces)
    native = {"method": method, "parameters": {}, "policy": policy}
    selected_edges = []
    if method == "cumesh_cleanup":
        native["parameters"] = {k: parameters[k] for k in (
            "remove_duplicate_faces", "remove_degenerate_faces")}
    elif method == "cumesh_fill_selected_holes":
        if diagnostics["topology"]["boundary_loop_error"]:
            raise ValueError("Source boundary graph does not enumerate simple loops")
        lookup = {loop["id"]: loop for loop in diagnostics["loops"]}
        for identifier in parameters["loop_ids"]:
            if identifier not in lookup:
                raise ValueError(f"Unknown source loop ID: {identifier}")
            loop = lookup[identifier]
            if (loop["edge_count"] > parameters["max_edges"] or
                    loop["diameter_upper_bound"] > parameters["max_diameter"]):
                raise ValueError("Selected loop exceeds authored edge or diameter bound")
            indices = inverse[np.asarray(loop["vertex_indices"], dtype=np.int64)]
            if not inside(vertices[indices], policy["roi"]).all():
                raise ValueError("Selected loop extends outside repair ROI")
            selected_edges.append([[int(indices[i]), int(indices[(i + 1) % len(indices)])]
                                   for i in range(len(indices))])
        native["parameters"] = {"selected_loop_edges": selected_edges,
            "input_geometry_sha256": adapter.geometry_sha256(vertices, faces),
            "max_hole_perimeter": parameters["max_hole_perimeter"]}
    input_hash = adapter.geometry_sha256(vertices, faces)
    cv, cf, native_report = adapter.repair(vertices, faces, native)
    cv, cf = checked_arrays(cv, cf)
    if (native_report.get("input_geometry_sha256") != input_hash or
            native_report.get("output_geometry_sha256") != adapter.geometry_sha256(cv, cf)
            or native_report.get("method") != method):
        raise ValueError("Native operation report is not bound to the dispatched arrays")
    operation = {"operation": "cumesh_common_candidate", "method": method,
        "native_dispatch_completed": True, "native_report": native_report,
        "canonical_exact_weld": True, "float32_world_frame": True,
        "coordinate_transform": "identity", "cpu_fallback": False,
        "all_common_export_gates_required": True}
    if method == "cumesh_diagnose":
        if input_hash != adapter.geometry_sha256(cv, cf):
            raise ValueError("CuMesh diagnosis changed geometry")
        operation["original_glb_bytes_preserved"] = True
    elif method == "cumesh_cleanup":
        removed = len(faces) - len(cf)
        if removed < 0 or removed > parameters["max_removed_faces"] or (
                removed / len(faces) > parameters["max_removed_fraction"]):
            raise ValueError("GPU cleanup exceeded authored face-removal bound")
        # Cleanup may remove only requested exact duplicates/zero-area faces.
        # Comparing the expected geometric multiset prevents displacement,
        # unrelated deletion, re-triangulation, and accidental near-zero deletion.
        expected = faces
        if parameters["remove_duplicate_faces"]:
            _, first = np.unique(np.sort(expected, axis=1), axis=0, return_index=True)
            expected = expected[np.sort(first)]
        if parameters["remove_degenerate_faces"]:
            expected = expected[areas(vertices, expected) != 0]
        if geometry_digest(vertices, expected) != geometry_digest(cv, cf):
            raise ValueError("GPU cleanup changed geometry beyond the exact requested face removals")
        operation.update(removed_faces=removed, near_degenerate_faces_removed=False)
    elif method == "cumesh_fill_selected_holes":
        if not np.array_equal(cv[:len(vertices)], vertices) or not np.array_equal(cf[:len(faces)], faces):
            raise ValueError("GPU patch changed pre-existing vertices or faces")
        patch = cf[len(faces):]
        patch_area = areas(cv, patch)
        if not len(patch) or patch_area.min() < parameters["min_triangle_area"]:
            raise ValueError("GPU patch is empty or introduces near-degenerate triangles")
        if patch_area.sum() > parameters["max_patch_area"] or not inside(cv, policy["roi"])[patch].all():
            raise ValueError("GPU patch exceeds authored area or ROI")
        selected_keys = {tuple(sorted(edge)) for loop in selected_edges for edge in loop}
        directed_boundary = [list(map(int, inverse[edge])) for edge in info["boundary_edges"]
                             if tuple(sorted(map(int, inverse[edge]))) in selected_keys]
        oriented, orientation = _orient_new_patch_faces(patch, directed_boundary)
        cf[len(faces):] = oriented
        if not np.array_equal(cf[:len(faces)], faces):
            raise ValueError("Patch orientation unexpectedly changed pre-existing faces")
        operation.update(loop_ids=parameters["loop_ids"], canonical_selected_edges=selected_edges,
            added_faces=len(patch), patch_area=float(patch_area.sum()), patch_orientation=orientation,
            post_orientation_geometry_sha256=adapter.geometry_sha256(cv, cf))
    report["steps"].append(operation)
    return cv, cf


def _remove_components(vertices: Any, faces: Any, diagnostics: dict[str, Any],
                       info: dict[str, Any], parameters: dict[str, Any], policy: dict[str, Any],
                       steps: list[dict[str, Any]]) -> tuple[Any, Any]:
    import numpy as np
    from .repair_geometry import inside

    remove = np.zeros(len(faces), dtype=bool)
    for identifier in parameters["component_ids"]:
        if identifier in policy["protected_component_ids"]:
            raise ValueError("Cannot delete a protected component")
        if identifier not in info["component_faces"]:
            raise ValueError(f"Unknown source component ID: {identifier}")
        remove[info["component_faces"][identifier]] = True
    total_area = float(info["face_areas"][remove].sum())
    if remove.sum() > parameters["max_total_faces"] or total_area > parameters["max_total_area"]:
        raise ValueError("Selected components exceed authored face/area bound")
    if not inside(vertices, policy["roi"])[faces[remove]].all():
        raise ValueError("Selected component extends outside ROI")
    if (~remove).sum() < 4:
        raise ValueError("Component removal would leave fewer than four faces")
    steps.append({"operation": "remove_selected_components", "ids": parameters["component_ids"],
                  "removed_faces": int(remove.sum()), "removed_area": total_area})
    return vertices.copy(), faces[~remove].copy()


def _polygon_triangulation(points: Any) -> list[list[int]]:
    """Ear-clip a simple CCW polygon; reject degeneracy instead of guessing."""
    import numpy as np
    points = np.asarray(points, dtype=np.float64)
    scale = float(np.ptp(points, axis=0).max())
    epsilon = max(scale * scale * 1e-12, 1e-20)

    def cross(a: Any, b: Any, c: Any) -> float:
        u, v = b - a, c - a
        return float(u[0] * v[1] - u[1] * v[0])

    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Primitive ring has repeated points")
    area = sum(cross(np.zeros(2), points[i], points[(i + 1) % len(points)]) for i in range(len(points)))
    if area <= epsilon:
        raise ValueError("Primitive rings must be nondegenerate and counterclockwise")
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        for j in range(i + 1, len(points)):
            if j in {i, (i + 1) % len(points)} or (j + 1) % len(points) == i:
                continue
            c, d = points[j], points[(j + 1) % len(points)]
            x1, x2, x3, x4 = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
            if x1 * x2 <= epsilon * epsilon and x3 * x4 <= epsilon * epsilon:
                raise ValueError("Primitive ring self-intersects or has ambiguous collinear edges")
    remaining, triangles = list(range(len(points))), []
    while len(remaining) > 3:
        found = False
        for index, b in enumerate(remaining):
            a, c = remaining[index - 1], remaining[(index + 1) % len(remaining)]
            if cross(points[a], points[b], points[c]) <= epsilon:
                continue
            if any(min(cross(points[a], points[b], points[p]),
                       cross(points[b], points[c], points[p]),
                       cross(points[c], points[a], points[p])) >= -epsilon
                   for p in remaining if p not in {a, b, c}):
                continue
            triangles.append([a, b, c])
            remaining.pop(index)
            found = True
            break
        if not found:
            raise ValueError("No stable triangulation for authored primitive")
    triangles.append(remaining)
    return triangles


def _primitive_mesh(primitive: dict[str, Any]) -> tuple[Any, Any]:
    import numpy as np
    import trimesh

    if primitive["kind"] == "box":
        lo, hi = np.asarray(primitive["min"]), np.asarray(primitive["max"])
        mesh = trimesh.creation.box(extents=hi - lo)
        mesh.apply_translation((lo + hi) * 0.5)
        return mesh.vertices.copy(), mesh.faces.copy()
    up = primitive["up_axis"]
    horizontal = [a for a in range(3) if a != up]
    rings = primitive["rings"]
    count = len(rings[0]["points"])
    vertices = np.zeros((len(rings) * count, 3), dtype=np.float64)
    cap_triangles = []
    for index, ring in enumerate(rings):
        cap_triangles.append(_polygon_triangulation(ring["points"]))
        vertices[index * count:(index + 1) * count, up] = ring["height"]
        for coordinate, axis in enumerate(horizontal):
            vertices[index * count:(index + 1) * count, axis] = np.asarray(ring["points"])[:, coordinate]
    faces = [list(reversed(triangle)) for triangle in cap_triangles[0]]
    faces.extend([[i + (len(rings) - 1) * count for i in triangle] for triangle in cap_triangles[-1]])
    for ring in range(len(rings) - 1):
        for index in range(count):
            a, b = ring * count + index, ring * count + (index + 1) % count
            c, d = a + count, b + count
            faces.extend([[a, b, d], [a, d, c]])
    faces = np.asarray(faces, dtype=np.int64)
    # [horizontal0,horizontal1,up] is left-handed only when up=1.
    if up == 1:
        faces = faces[:, ::-1]
    return vertices, faces


def _restore_outside_faces(source_vertices: Any, source_faces: Any, candidate_vertices: Any,
                           candidate_faces: Any, roi: dict[str, Any],
                           steps: list[dict[str, Any]]) -> tuple[Any, Any]:
    """Restore immutable exterior triangulation without fitting or tolerance.

    Crossing faces belong to the protected exterior. A mismatch at the join is
    deliberately left visible to the unchanged exported topology/intersection
    gates. This operation does not claim that the two regions form a valid mesh.
    """
    import numpy as np
    from .repair_geometry import MAX_FACES, checked_arrays, geometry_digest, inside

    source_outside = ~inside(source_vertices, roi)[source_faces].all(axis=1)
    candidate_inside = inside(candidate_vertices, roi)[candidate_faces].all(axis=1)
    source_kept = source_faces[source_outside]
    candidate_kept = candidate_faces[candidate_inside]
    count = len(source_kept) + len(candidate_kept)
    if not 0 < count <= MAX_FACES:
        raise ValueError("Outside-face restoration exceeds the existing triangle cap")
    source_digest = geometry_digest(source_vertices, source_kept)
    removed_digest = geometry_digest(candidate_vertices, candidate_faces[~candidate_inside])
    retained_inside_digest = geometry_digest(candidate_vertices, candidate_kept)
    vertices = np.vstack([source_vertices, candidate_vertices])
    faces = np.vstack([source_kept, candidate_kept + len(source_vertices)])
    before_weld = len(vertices)
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse[faces]
    welded = before_weld - len(vertices)
    used, inverse = np.unique(faces, return_inverse=True)
    unreferenced = len(vertices) - len(used)
    vertices, faces = vertices[used], inverse.reshape((-1, 3))
    checked_arrays(vertices, faces)
    restored_outside_digest = geometry_digest(vertices, faces[:len(source_kept)])
    restored_inside_digest = geometry_digest(vertices, faces[len(source_kept):])
    if restored_outside_digest != source_digest or restored_inside_digest != retained_inside_digest:
        raise ValueError("Exact welding unexpectedly changed restored region geometry")
    steps.append({"operation": "restore_original_outside_faces", "explicitly_requested": True,
        "roi": roi, "source_outside_faces_retained": len(source_kept),
        "candidate_outside_faces_discarded": int((~candidate_inside).sum()),
        "candidate_inside_faces_retained": len(candidate_kept), "result_faces": len(faces),
        "source_outside_sha256": source_digest, "discarded_candidate_outside_sha256": removed_digest,
        "restored_outside_sha256": restored_outside_digest,
        "candidate_inside_sha256": retained_inside_digest, "restored_inside_sha256": restored_inside_digest,
        "exact_coordinate_vertices_welded": welded, "unreferenced_vertices_removed": unreferenced,
        "maximum_coordinate_displacement": 0.0, "tolerance_weld": False,
        "gate_waiver": False, "seam_validated_here": False,
        "note": "Every exported gate still applies; incompatible boundary connectivity rejects candidate"})
    return vertices, faces


def _union(vertices: Any, faces: Any, diagnostics: dict[str, Any], info: dict[str, Any],
           parameters: dict[str, Any], policy: dict[str, Any], steps: list[dict[str, Any]]) -> tuple[Any, Any]:
    import numpy as np
    import manifold3d
    from .repair_geometry import inside, inspect, self_intersections

    if not all(hasattr(manifold3d, name) for name in ("Mesh", "Manifold", "Error")):
        raise ImportError("Manifold3D solid Boolean API unavailable")

    topo = diagnostics["topology"]
    if not topo["all_edges_have_two_faces"] or topo["inconsistent_edge_directions"] or not (
            topo["vertex_manifold_evaluated"] and topo["nonmanifold_vertices"] == 0) or (
            topo["collapsed_faces"] or topo["zero_area_faces"] or topo["duplicate_faces"]):
        raise ValueError("Foundation Boolean requires closed oriented manifold source")
    intersections = self_intersections(vertices, faces)
    _require_check(intersections, "Foundation source self-intersection check failed or unavailable")
    pv, pf = _primitive_mesh(parameters["primitive"])
    if not inside(pv, policy["roi"]).all():
        raise ValueError("Authored primitive extends outside edit ROI")
    primitive_diagnostics, _ = inspect(pv, pf)
    primitive_intersections = self_intersections(pv, pf)
    if not primitive_diagnostics["topology"]["all_edges_have_two_faces"]:
        raise ValueError("Authored primitive is not a valid closed solid")
    _require_check(primitive_intersections, "Foundation primitive self-intersection check failed or unavailable")

    def manifold(v: Any, f: Any) -> Any:
        mesh = manifold3d.Mesh(np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.uint32))
        solid = manifold3d.Manifold(mesh)
        if solid.status() != manifold3d.Error.NoError or solid.is_empty() or solid.volume() <= 0:
            raise ValueError(f"Manifold rejected solid: {solid.status()}")
        return solid

    source = manifold(info["canonical_vertices"], info["canonical_faces"])
    plug = manifold(pv, pf)
    overlap = source ^ plug
    if overlap.status() != manifold3d.Error.NoError or overlap.is_empty() or overlap.volume() <= 0:
        raise ValueError("Foundation plug does not overlap source volume")
    result = source + plug
    if result.status() != manifold3d.Error.NoError or result.is_empty() or result.volume() <= 0:
        raise ValueError(f"Manifold union failed: {result.status()}")
    exported = result.to_mesh()
    steps.append({"operation": "authored_foundation_union", "backend": "manifold3d",
                  "primitive": parameters["primitive"], "overlap_volume": float(overlap.volume()),
                  "source_volume": float(source.volume()), "union_volume": float(result.volume()),
                  "frame": "identity baked GLB world frame; no normalization"})
    candidate_vertices = np.asarray(exported.vert_properties[:, :3], dtype=np.float64)
    candidate_faces = np.asarray(exported.tri_verts, dtype=np.int64)
    if parameters.get("restore_outside_faces", False):
        candidate_vertices, candidate_faces = _restore_outside_faces(
            vertices, faces, candidate_vertices, candidate_faces, policy["roi"], steps)
    return candidate_vertices, candidate_faces


def run(request: dict[str, Any]) -> dict[str, Any]:
    allowed = {"schema_version", "source_path", "source_sha256", "recipe", "output_dir", "result_path",
               "import_path", "import_sha256"}
    if not isinstance(request, dict) or set(request) - allowed or request.get("schema_version", 1) != 1:
        raise ValueError("Unknown request keys/schema")
    recipe = validate_recipe(request.get("recipe"))
    source = Path(request["source_path"]).resolve(strict=True)
    source_hash = request.get("source_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("source_sha256 must be lowercase SHA256")
    if _hash(source) != source_hash:
        raise ValueError("Source hash mismatch before repair")
    output_dir = Path(request["output_dir"]).resolve()
    result_path = Path(request.get("result_path", output_dir / "result.json")).resolve()
    if result_path.parent != output_dir:
        raise ValueError("result_path must be directly inside output_dir")
    paths = {"output": output_dir / "candidate.glb", "repair_report": output_dir / "repair-report.json",
             "diagnostics": output_dir / "diagnostics.json", "result": result_path}
    if len(set(paths.values())) != len(paths) or source in paths.values():
        raise ValueError("Artifact paths alias one another or the raw source")
    if any(path.exists() for path in paths.values()):
        raise ValueError("Repair artifacts are immutable; select a fresh attempt directory")
    imported = None
    if recipe["method"] == "import_candidate":
        imported = Path(request["import_path"]).resolve(strict=True)
        if not isinstance(request.get("import_sha256"), str) or _hash(imported) != request["import_sha256"]:
            raise ValueError("Imported candidate hash mismatch")
        if imported in paths.values():
            raise ValueError("Import artifact aliases a new output")
    elif "import_path" in request or "import_sha256" in request:
        raise ValueError("Import fields only valid for import_candidate")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Reserve the attempt before any compute; failures keep this marker and reports.
    _json_new(output_dir / "repair-started.json", {"source_path": str(source), "source_sha256": source_hash,
        "recipe": recipe, "gate_version": REPAIR_GATE_VERSION,
        "started_utc": datetime.now(timezone.utc).isoformat()})
    report: dict[str, Any] = {"gate_version": REPAIR_GATE_VERSION, "recipe": recipe,
        "source_path": str(source), "source_sha256": source_hash, "steps": [],
        "capabilities": get_capabilities(), "frame": "baked GLB world coordinates; identity transform",
        "semantic_approved": False, "production_approved": False, "error": None}
    diagnostics: dict[str, Any] = {"gate_version": REPAIR_GATE_VERSION, "source_sha256": source_hash,
                                    "source": None, "candidate": None}
    gate: dict[str, Any] = {"passed": False, "gate_version": REPAIR_GATE_VERSION,
        "source_sha256": source_hash, "output_sha256": None, "candidate_sha256": None, "checks": [],
        "semantic_approved": False, "requires_agent_review": True,
        "limitations": ["Numerical gates do not establish object identity or design intent",
                        "Foundation intervals sample the authored ROI; no whole-volume proof"]}
    try:
        from .repair_geometry import (checked_arrays, foundation_intervals, inspect,
                                      preservation, self_intersections)
        vertices, faces = _load(source)
        diagnostics["source"], info = inspect(vertices, faces)
        for identifier in recipe["policy"]["protected_component_ids"]:
            if identifier not in info["component_faces"]:
                raise ValueError(f"Unknown protected component: {identifier}")
        method, parameters, policy = recipe["method"], recipe["parameters"], recipe["policy"]
        if method in {"analyze", "cumesh_diagnose"}:
            if method == "cumesh_diagnose":
                cv, cf = _cumesh_candidate(diagnostics["source"], info, recipe, report)
                del cv, cf
            _copy_new(source, paths["output"])
            report["steps"].append({"operation": "analyze_identity_copy", "geometry_edited": False})
        elif method == "import_candidate":
            _copy_new(imported, paths["output"])
            report["steps"].append({"operation": "import_candidate", "import_path": str(imported),
                                    "import_sha256": request["import_sha256"]})
        else:
            if method == "conservative_cleanup":
                cv, cf = _cleanup(vertices, faces, parameters, policy["roi"], report["steps"])
            elif method == "patch_selected_holes":
                cv, cf = _patch(vertices, faces, diagnostics["source"], info, parameters,
                                policy["roi"], report["steps"])
            elif method == "remove_selected_components":
                cv, cf = _remove_components(vertices, faces, diagnostics["source"], info,
                                            parameters, policy, report["steps"])
            elif method == "foundation_union":
                cv, cf = _union(vertices, faces, diagnostics["source"], info, parameters,
                                policy, report["steps"])
            elif method in {"cumesh_cleanup", "cumesh_fill_selected_holes"}:
                cv, cf = _cumesh_candidate(diagnostics["source"], info, recipe, report)
            else:
                raise ValueError("Unsupported method")
            checked_arrays(cv, cf)
            _export(paths["output"], cv, cf)
            del cv, cf
        # This is the actual serialized output, including float32 rounding.
        cv, cf = _load(paths["output"])
        diagnostics["candidate"], candidate_info = inspect(cv, cf)
        del candidate_info
        topo = diagnostics["candidate"]["topology"]
        intersection = self_intersections(cv, cf)
        retained = preservation(vertices, faces, cv, cf, info, diagnostics["candidate"], policy)
        foundation = foundation_intervals(cv, cf, policy["foundation"])
        diagnostics["candidate"]["self_intersections"] = intersection
        diagnostics["candidate"]["foundation"] = foundation
        report["preservation"] = retained
        checks = [
            {"name": "export_closed_edges", "evaluated": True, "passed": topo["all_edges_have_two_faces"]},
            {"name": "export_consistent_winding", "evaluated": True,
             "passed": topo["inconsistent_edge_directions"] == 0},
            {"name": "export_no_collapsed_or_duplicate_faces", "evaluated": True,
             "passed": topo["collapsed_faces"] == topo["zero_area_faces"] == topo["duplicate_faces"] == 0},
            {"name": "export_vertex_manifold", "evaluated": topo["vertex_manifold_evaluated"],
             "passed": topo["vertex_manifold_evaluated"] and topo["nonmanifold_vertices"] == 0},
            {"name": "export_self_intersections", **intersection},
            {"name": "outside_roi_preserved", **retained["outside_roi"]},
            {"name": "protected_components_preserved", **retained["protected_components"]},
        ]
        if foundation["required"]:
            prerequisite = all(check["passed"] for check in checks[:5])
            checks.append({"name": "authored_foundation_intervals", "evaluated": prerequisite,
                           "passed": prerequisite and foundation["passed"],
                           "failed_samples": foundation["failed_samples"], "sampling_only": True,
                           "reason": None if prerequisite else "invalid_solid_preconditions"})
        gate["checks"] = checks
        gate["passed"] = all(check["passed"] and check["evaluated"] for check in checks)
    except Exception as exc:
        failure = classify_failure(exc)
        report["error"] = {"type": type(exc).__name__, "message": str(exc), **failure}
        if isinstance(exc, RepairCheckError):
            report["error"]["required_check"] = exc.check
        gate["checks"].append({"name": "worker_operation_completed", "evaluated": False,
                               "passed": False, "reason": str(exc), **failure})
        gate["passed"] = False
    finally:
        unchanged = source.is_file() and _hash(source) == source_hash
        gate["checks"].append({"name": "raw_source_unchanged", "evaluated": True, "passed": unchanged,
            **({"failure_kind": "runtime_failure", "code": "REPAIR_SOURCE_CHANGED"} if not unchanged else {})})
        if imported is not None:
            import_unchanged = imported.is_file() and _hash(imported) == request["import_sha256"]
            gate["checks"].append({"name": "import_source_unchanged", "evaluated": True,
                "passed": import_unchanged, **({"failure_kind": "runtime_failure",
                    "code": "REPAIR_IMPORT_SOURCE_CHANGED"} if not import_unchanged else {})})
            gate["passed"] = gate["passed"] and import_unchanged
        gate["passed"] = gate["passed"] and unchanged
        report["raw_source_unchanged"] = unchanged
        if paths["output"].is_file():
            gate["output_sha256"] = _hash(paths["output"])
            gate["candidate_sha256"] = gate["output_sha256"]
        diagnostics["output_sha256"] = gate["output_sha256"]
        _finalize_failure_kind(gate)
        report["failure_kind"], report["failure_code"] = gate["failure_kind"], gate["failure_code"]
        diagnostics["failure_kind"] = gate["failure_kind"]
        report["gate"] = gate
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _json_new(paths["diagnostics"], diagnostics)
        _json_new(paths["repair_report"], report)
        result = {"output": str(paths["output"]) if paths["output"].is_file() else None,
                  "gate": gate, "repair_report": str(paths["repair_report"]),
                  "diagnostics": str(paths["diagnostics"]), "failure_kind": gate["failure_kind"],
                  "failure_code": gate["failure_code"]}
        _json_new(paths["result"], result)
    return result


def main() -> None:
    # Isolate stdin before importing NumPy/OpenBLAS through any geometry helper.
    from .remesh_worker import _isolate_stdin
    _isolate_stdin()
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    if args.request.stat().st_size > 256 * 1024:
        raise ValueError("Repair request exceeds 256 KiB")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = run(request)
    print(json.dumps({"result": str(request.get("result_path", "result.json")),
                      "machine_gate_passed": result["gate"]["passed"]}), flush=True)
    # A completed diagnostic failure is a normal result requiring agent repair.
    # Exceptions before a result exists remain nonzero process failures.


if __name__ == "__main__":
    main()
