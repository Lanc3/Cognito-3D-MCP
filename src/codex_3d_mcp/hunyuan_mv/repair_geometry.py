"""Bounded-array geometry diagnostics. Imported only by the resource-limited child."""
from __future__ import annotations

import hashlib
import errno
from typing import Any

import numpy as np

MAX_FACES = 1_500_000
MAX_COMPONENTS = 4096
MAX_BOUNDARY_EDGES = 100_000


def checked_arrays(vertices: Any, faces: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Expected nonempty 3D vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not 0 < len(faces) <= MAX_FACES:
        raise ValueError(f"Expected 1..{MAX_FACES} triangles")
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Nonfinite coordinates or invalid face indices")
    if np.ptp(vertices, axis=0).max() <= 0:
        raise ValueError("Degenerate mesh bounds")
    return vertices, faces


def areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    result = np.empty(len(faces), dtype=np.float64)
    for start in range(0, len(faces), 65536):
        t = vertices[faces[start:start + 65536]]
        result[start:start + len(t)] = np.linalg.norm(
            np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) * 0.5
    return result


def geometry_digest(vertices: np.ndarray, faces: np.ndarray) -> str:
    """Canonical geometric face multiset at GLB float32 precision; winding independent.

    Bakes neither normalization nor tolerance welding into the comparison. Exact
    float32 conversion is explicit because that is the candidate export precision.
    """
    h = hashlib.sha256(b"glb-f32-geometric-face-multiset-v1\0")
    h.update(str(len(faces)).encode("ascii"))
    if len(faces):
        used = np.unique(faces)
        local_faces = np.searchsorted(used, faces)
        unique, inverse = np.unique(np.asarray(vertices[used], dtype="<f4"), axis=0,
                                     return_inverse=True)
        canon = np.sort(inverse[local_faces], axis=1)
        order = np.lexsort((canon[:, 2], canon[:, 1], canon[:, 0]))
        for start in range(0, len(order), 8192):
            h.update(np.asarray(unique[canon[order[start:start + 8192]]], dtype="<f4").tobytes())
    return h.hexdigest()


def inside(vertices: np.ndarray, roi: dict[str, Any]) -> np.ndarray:
    # No user-defined fuzzy tolerance: an edit must stay inside the declared box.
    return ((vertices >= np.asarray(roi["min"])) &
            (vertices <= np.asarray(roi["max"]))).all(axis=1)


def _loops(vertices: np.ndarray, directed: np.ndarray) -> tuple[list[dict[str, Any]], str | None]:
    if len(directed) > MAX_BOUNDARY_EDGES:
        return [], "boundary_edge_limit_exceeded"
    neighbors: dict[int, list[int]] = {}
    for a, b in directed:
        neighbors.setdefault(int(a), []).append(int(b))
        neighbors.setdefault(int(b), []).append(int(a))
    if any(len(values) != 2 for values in neighbors.values()):
        return [], "boundary_graph_not_disjoint_simple_cycles"
    remaining = set(neighbors)
    output = []
    while remaining:
        first = min(remaining)
        path, visited, previous, current = [first], {first}, first, min(neighbors[first])
        while current != first:
            if current in visited or current not in remaining or len(path) > 8192:
                return output, "boundary_cycle_invalid_or_exceeds_8192_vertices"
            path.append(current)
            visited.add(current)
            adjacent = neighbors[current]
            previous, current = current, adjacent[0] if adjacent[0] != previous else adjacent[1]
        remaining.difference_update(path)
        points = vertices[path]
        lo, hi = points.min(axis=0), points.max(axis=0)
        # The AABB diagonal is a safe upper bound even for a long complicated loop.
        diameter = float(np.linalg.norm(hi - lo))
        projected_area = float(np.linalg.norm(np.cross(points, np.roll(points, -1, axis=0))
                                             .sum(axis=0)) * 0.5)
        reverse = [path[0], *reversed(path[1:])]
        canonical = min(path, reverse)
        identifier = hashlib.sha256(b"boundary-loop-f32-v1\0" +
            np.asarray(vertices[canonical], dtype="<f4").tobytes()).hexdigest()
        output.append({"id": "loop:" + identifier, "vertex_indices": path,
                       "vertices": points.tolist(), "edge_count": len(path),
                       "bounds": {"min": lo.tolist(), "max": hi.tolist()},
                       "diameter_upper_bound": diameter, "projected_area": projected_area})
        if len(output) >= 4096 and remaining:
            return output, "boundary_loop_count_limit_exceeded"
    return output, None


def inspect(vertices: np.ndarray, faces: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exact-coordinate edge incidence, vertex links and edge-connected components.

    Vertex manifoldness is checked by connectedness of each vertex's face fan and
    a boundary degree of zero or two. It catches two closed shells touching at a
    vertex, which an edge-incidence-only test misses. No repair is performed.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    vertices, faces = checked_arrays(vertices, faces)
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    canonical = inverse[faces]
    nfaces = len(faces)
    directed = np.concatenate([canonical[:, [0, 1]], canonical[:, [1, 2]], canonical[:, [2, 0]]])
    signs = directed[:, 0] < directed[:, 1]
    edges = np.sort(directed, axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    starts = np.r_[0, np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(edges)])
    one, two, over = counts == 1, counts == 2, counts > 2
    boundary = directed[order[starts[one]]]
    loops, loop_error = _loops(unique, boundary)
    pair_starts = starts[two]
    face_a, face_b = order[pair_starts] % nfaces, order[pair_starts + 1] % nfaces
    bad_winding = int((signs[order[pair_starts]] == signs[order[pair_starts + 1]]).sum())

    # Component graph connects adjacent faces; overconnected edges are diagnosed,
    # but still connect those faces so a fragment cannot masquerade as a new ID.
    rows, cols = [face_a], [face_b]
    for start, count in zip(starts[over], counts[over]):
        group = order[start:start + count] % nfaces
        rows.append(np.repeat(group[0], len(group) - 1))
        cols.append(group[1:])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                       shape=(nfaces, nfaces)).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    del graph, rows, cols
    face_areas = areas(vertices, faces)
    components, component_faces = [], {}
    if component_count > MAX_COMPONENTS:
        raise ValueError(f"Component count {component_count} exceeds diagnostic cap {MAX_COMPONENTS}")
    sorted_faces = np.argsort(labels, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(labels[sorted_faces])) + 1, nfaces]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        selected = sorted_faces[start:end]
        digest = geometry_digest(vertices, faces[selected])
        identifier = "component:" + digest
        used = np.unique(faces[selected])
        bounds = {"min": vertices[used].min(axis=0).tolist(),
                  "max": vertices[used].max(axis=0).tolist()}
        components.append({"id": identifier, "geometry_sha256": digest,
                           "faces": len(selected), "vertices": len(used), "bounds": bounds,
                           "area": float(face_areas[selected].sum())})
        if identifier in component_faces:
            raise ValueError("Ambiguous duplicate component geometry IDs")
        component_faces[identifier] = selected

    collapsed = ((canonical[:, 0] == canonical[:, 1]) |
                 (canonical[:, 1] == canonical[:, 2]) |
                 (canonical[:, 2] == canonical[:, 0]))
    unordered_faces = np.sort(canonical, axis=1)
    duplicates = nfaces - len(np.unique(unordered_faces, axis=0))
    bad_vertices = np.empty(0, dtype=np.int64)
    vertex_evaluated = not over.any() and not collapsed.any()
    if vertex_evaluated:
        # One node for each face corner, linked across shared edges at that vertex.
        endpoints = edges[pair_starts]
        corner_rows, corner_cols = [], []
        for endpoint in (0, 1):
            values = endpoints[:, endpoint]
            ca = face_a * 3 + (canonical[face_a] == values[:, None]).argmax(axis=1)
            cb = face_b * 3 + (canonical[face_b] == values[:, None]).argmax(axis=1)
            corner_rows.append(ca)
            corner_cols.append(cb)
        cr, cc = np.concatenate(corner_rows), np.concatenate(corner_cols)
        link_graph = coo_matrix((np.ones(len(cr), dtype=np.uint8), (cr, cc)),
                                shape=(3 * nfaces, 3 * nfaces)).tocsr()
        _, link_labels = connected_components(link_graph, directed=False)
        flat = canonical.ravel()
        vertex_order = np.argsort(flat, kind="stable")
        same_vertex = flat[vertex_order[1:]] == flat[vertex_order[:-1]]
        separate_fans = link_labels[vertex_order[1:]] != link_labels[vertex_order[:-1]]
        fan_bad = flat[vertex_order[1:][same_vertex & separate_fans]]
        boundary_degree = np.bincount(boundary.ravel(), minlength=len(unique))
        bad_vertices = np.unique(np.r_[fan_bad, np.flatnonzero(
            (boundary_degree != 0) & (boundary_degree != 2))])
        del link_graph, cr, cc, link_labels, vertex_order
    topology = {
        "method": "exact_geometric_edges_and_vertex_links_v1",
        "triangles": nfaces, "geometric_vertices": len(unique),
        "boundary_edges": int(one.sum()), "overconnected_edges": int(over.sum()),
        "inconsistent_edge_directions": bad_winding,
        "collapsed_faces": int(collapsed.sum()), "duplicate_faces": int(duplicates),
        "zero_area_faces": int((face_areas == 0).sum()),
        "near_zero_area_faces_1e_12": int((face_areas <= 1e-12).sum()),
        "all_edges_have_two_faces": bool(two.all()),
        "vertex_manifold_evaluated": vertex_evaluated,
        "nonmanifold_vertices": int(len(bad_vertices)) if vertex_evaluated else None,
        "nonmanifold_vertex_samples": unique[bad_vertices[:32]].tolist(),
        "vertex_manifold_reason": None if vertex_evaluated else "overconnected_or_collapsed_faces",
        "component_count": int(component_count), "boundary_loop_error": loop_error,
        "area": float(face_areas.sum()),
        "bounds": {"min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
        "area_units": "baked GLB world coordinate units squared",
    }
    return {"topology": topology, "loops": loops, "components": components}, {
        "canonical_vertices": unique, "canonical_faces": canonical,
        "component_faces": component_faces, "face_areas": face_areas,
        "boundary_edges": boundary,
    }


def self_intersections(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    try:
        import importlib.metadata
        import pymeshlab

        if not callable(getattr(pymeshlab, "MeshSet", None)) or not callable(getattr(pymeshlab, "Mesh", None)):
            raise ImportError("PyMeshLab Mesh/MeshSet API unavailable")
        ms = pymeshlab.MeshSet()
        if not callable(getattr(ms, "compute_selection_by_self_intersections_per_face", None)):
            raise ImportError("PyMeshLab self-intersection filter unavailable")
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces.astype(np.int32)))
        ms.compute_selection_by_self_intersections_per_face()
        selected = int(ms.current_mesh().selected_face_number())
        return {"evaluated": True, "passed": selected == 0, "selected_faces": selected,
                "backend": "pymeshlab.compute_selection_by_self_intersections_per_face",
                "version": importlib.metadata.version("pymeshlab"),
                "failure_kind": None if selected == 0 else "geometry_rejected",
                "code": None if selected == 0 else "REPAIR_SELF_INTERSECTIONS_FOUND",
                "scope": "backend intersection detector; no formal exact-arithmetic certificate"}
    except Exception as exc:
        # Classify actual exception types/codes, never fragments of human messages.
        if isinstance(exc, ImportError):
            failure_kind, code = "runtime_unavailable", "REPAIR_BACKEND_UNAVAILABLE"
        elif isinstance(exc, MemoryError) or (isinstance(exc, OSError) and (
                exc.errno in {errno.ENOMEM, errno.ENOSPC, errno.EAGAIN} or
                getattr(exc, "winerror", None) in {8, 14, 1455})):
            failure_kind, code = "resource", "REPAIR_RESOURCE_EXHAUSTED"
        elif isinstance(exc, ValueError):
            failure_kind, code = "geometry_rejected", "REPAIR_INVALID_GEOMETRY"
        else:
            failure_kind, code = "runtime_failure", "REPAIR_BACKEND_RUNTIME_FAILURE"
        return {"evaluated": False, "passed": False, "selected_faces": None,
                "reason": f"{type(exc).__name__}: {exc}", "backend": "pymeshlab",
                "failure_kind": failure_kind, "code": code,
                "error_type": type(exc).__name__}


def preservation(source_v: np.ndarray, source_f: np.ndarray, candidate_v: np.ndarray,
                 candidate_f: np.ndarray, source_info: dict[str, Any],
                 candidate_diagnostics: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    roi = policy["roi"]
    source_out = np.ones(len(source_f), dtype=bool) if roi is None else (
        ~inside(source_v, roi)[source_f].all(axis=1))
    candidate_out = np.ones(len(candidate_f), dtype=bool) if roi is None else (
        ~inside(candidate_v, roi)[candidate_f].all(axis=1))
    before = geometry_digest(source_v, source_f[source_out])
    after = geometry_digest(candidate_v, candidate_f[candidate_out])
    available = {part["id"] for part in candidate_diagnostics["components"]}
    protected = []
    for identifier in policy["protected_component_ids"]:
        if identifier not in source_info["component_faces"]:
            raise ValueError(f"Unknown protected source component: {identifier}")
        protected.append({"id": identifier, "passed": identifier in available})
    return {"outside_roi": {"evaluated": True, "passed": before == after,
                "source_sha256": before, "candidate_sha256": after,
                "source_faces": int(source_out.sum()), "candidate_faces": int(candidate_out.sum()),
                "method": "exact float32 geometric face multiset; crossing faces protected",
                "winding_and_materials_included": False},
            "protected_components": {"evaluated": True,
                "passed": all(item["passed"] for item in protected), "components": protected,
                "note": "Only explicitly protected components are certified unchanged"}}


def foundation_intervals(vertices: np.ndarray, faces: np.ndarray,
                         policy: dict[str, Any] | None) -> dict[str, Any]:
    if policy is None:
        return {"evaluated": False, "required": False, "passed": None,
                "reason": "No authored foundation volume; semantic foundation unassessed"}
    up = policy["up_axis"]
    horizontal = [axis for axis in range(3) if axis != up]
    lo, hi = np.asarray(policy["roi"]["min"]), np.asarray(policy["roi"]["max"])
    size, tolerance = policy["grid_size"], policy["surface_tolerance"]
    positions = [lo[a] + (np.arange(size) + 0.5) * (hi[a] - lo[a]) / size for a in horizontal]
    samples = []
    for x in positions[0]:
        for y in positions[1]:
            heights = []
            for start in range(0, len(faces), 65536):
                triangle = vertices[faces[start:start + 65536]]
                p = triangle[:, :, horizontal]
                a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
                delta = np.array([x, y]) - p[:, 0]
                determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
                valid = np.abs(determinant) > 1e-20
                alpha = np.zeros(len(p))
                beta = np.zeros(len(p))
                alpha[valid] = (delta[valid, 0] * b[valid, 1] - delta[valid, 1] * b[valid, 0]) / determinant[valid]
                beta[valid] = (a[valid, 0] * delta[valid, 1] - a[valid, 1] * delta[valid, 0]) / determinant[valid]
                hit = valid & (alpha >= -1e-10) & (beta >= -1e-10) & (alpha + beta <= 1 + 1e-10)
                heights.extend((triangle[hit, 0, up] + alpha[hit] *
                    (triangle[hit, 1, up] - triangle[hit, 0, up]) + beta[hit] *
                    (triangle[hit, 2, up] - triangle[hit, 0, up])).tolist())
            heights.sort()
            crossings = []
            merge_tolerance = min(tolerance * 0.1, (hi[up] - lo[up]) * 1e-7)
            for height in heights:
                if not crossings or height - crossings[-1] > merge_tolerance:
                    crossings.append(height)
            intervals = [[crossings[i], crossings[i + 1]] for i in range(0, len(crossings) - 1, 2)]
            passed = len(crossings) % 2 == 0 and any(
                bottom <= lo[up] + tolerance and top >= hi[up] - tolerance
                for bottom, top in intervals)
            samples.append({"horizontal_position": [float(x), float(y)], "intervals": intervals,
                            "crossings": len(crossings), "passed": bool(passed)})
    return {"evaluated": True, "required": True, "passed": all(s["passed"] for s in samples),
            "failed_samples": sum(not s["passed"] for s in samples), "samples": samples,
            "policy": policy, "method": "vertical parity intervals at cell-center sample grid",
            "sampling_only": True, "whole_volume_proven": False,
            "precondition": "Closed, consistently oriented, non-self-intersecting geometry"}
