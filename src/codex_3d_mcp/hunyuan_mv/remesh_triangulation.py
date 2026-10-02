"""Fail-closed polygon OBJ conversion; never weld, drop, fill or add vertices.

Native boundary edges and previously chosen diagonals are globally reserved.
This is conversion, not mesh repair. A native hole or unresolved triangulation
returns evidence and must remain a failed remesh attempt.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class PolygonConversionError(ValueError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(report["reason"])


def _edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _cross(a: Any, b: Any, c: Any) -> float:
    ab, ac = b - a, c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _segments_intersect(a: Any, b: Any, c: Any, d: Any, eps: float) -> bool:
    abc, abd, cda, cdb = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if ((abc > eps and abd < -eps) or (abc < -eps and abd > eps)) and (
        (cda > eps and cdb < -eps) or (cda < -eps and cdb > eps)
    ):
        return True
    # Collinear contact between non-neighboring boundary edges is not simple.
    distance_eps = np.sqrt(eps) * 1e-3
    for value, point, first, second in ((abc, c, a, b), (abd, d, a, b), (cda, a, c, d), (cdb, b, c, d)):
        if abs(value) <= eps and np.all(point >= np.minimum(first, second) - distance_eps) and np.all(point <= np.maximum(first, second) + distance_eps):
            return True
    return False


def _project(points: Any) -> tuple[Any, float, float]:
    """Use the dominant Newell-normal projection; reject ambiguous loops."""
    centered = points - points.mean(axis=0)
    normal = np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0)
    scale = float(np.ptp(points, axis=0).max())
    if scale <= 0 or not np.isfinite(normal).all():
        raise ValueError("zero extent or non-finite polygon normal")
    eps = scale * scale * 1e-12
    if float(np.abs(normal).max()) <= eps:
        raise ValueError("zero-area or self-cancelling polygon normal")
    projected = np.delete(centered, int(np.abs(normal).argmax()), axis=1)
    twice_area = sum(_cross(np.zeros(2), projected[i], projected[(i + 1) % len(points)]) for i in range(len(points)))
    if abs(twice_area) <= eps:
        raise ValueError("zero projected polygon area")
    sign = 1.0 if twice_area > 0 else -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if j == i + 1 or (i == 0 and j == len(points) - 1):
                continue
            if _segments_intersect(projected[i], projected[(i + 1) % len(points)], projected[j], projected[(j + 1) % len(points)], eps):
                raise ValueError("self-intersecting or touching projected polygon boundary")
    return projected, sign, eps


def _valid_triangles(local: list[tuple[int, int, int]], projected: Any, sign: float, eps: float) -> bool:
    # The caller supplies a boundary-preserving split of a simple polygon.
    # Positive area in the same projection rejects outside/folded diagonals.
    return all(sign * _cross(projected[a], projected[b], projected[c]) > eps for a, b, c in local)


def _internal_edges(triangles: list[tuple[int, int, int]], boundary: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {_edge(t[i], t[(i + 1) % 3]) for t in triangles for i in range(3)} - boundary


def _ear_clip(loop: tuple[int, ...], projected: Any, sign: float, eps: float, forbidden: set[tuple[int, int]], max_states: int) -> list[tuple[int, int, int]] | None:
    """Bounded backtracking within one simple polygon, using only its corners."""
    states = 0
    native = {_edge(loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))}

    def search(remaining: tuple[int, ...]) -> list[tuple[int, int, int]] | None:
        nonlocal states
        states += 1
        if states > max_states:
            return None
        if len(remaining) == 3:
            tri = tuple(remaining)
            return [tri] if _valid_triangles([tri], projected, sign, eps) else None
        for slot, curr in enumerate(remaining):
            prev, nxt = remaining[slot - 1], remaining[(slot + 1) % len(remaining)]
            tri = (prev, curr, nxt)
            if not _valid_triangles([tri], projected, sign, eps):
                continue
            diagonal = _edge(loop[prev], loop[nxt])
            if diagonal not in native and diagonal in forbidden:
                continue
            if any(
                all(sign * _cross(projected[a], projected[b], projected[other]) >= -eps for a, b in ((prev, curr), (curr, nxt), (nxt, prev)))
                for other in remaining if other not in tri
            ):
                continue
            tail = search(remaining[:slot] + remaining[slot + 1:])
            if tail is not None:
                return [tri] + tail
        return None

    local = search(tuple(range(len(loop))))
    return None if local is None else [tuple(loop[i] for i in tri) for tri in local]


def triangulate_polygons(vertices: Any, polygons: list[tuple[int, ...]], *, max_ear_states: int = 4096) -> tuple[Any, dict[str, Any]]:
    """Return triangle indices and evidence; fail if any original face is unsafe.

    Native closure uses vertex indices. The worker retains its existing weld
    and independent post-weld manifold, winding and degeneracy checks.
    Greedy global reservations may reject a solvable mesh; never guess on failure.
    """
    points = np.asarray(vertices, dtype=np.float64)
    report: dict[str, Any] = {"method": "reserved-edge simple-polygon triangulation v1", "passed": False, "native_vertices": len(points), "native_polygons": len(polygons), "added_vertices": 0, "dropped_polygons": 0, "max_ear_states": max_ear_states}

    def fail(reason: str, **evidence: Any) -> None:
        report.update(reason=reason, **evidence)
        raise PolygonConversionError(report)

    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        fail("invalid or non-finite vertex array")
    if not polygons:
        fail("no polygons")
    incidence: dict[tuple[int, int], list[int]] = defaultdict(list)
    prepared = []
    for face_id, values in enumerate(polygons):
        loop = tuple(int(v) for v in values)
        if len(loop) < 3 or len(loop) > 128 or len(set(loop)) != len(loop) or min(loop) < 0 or max(loop) >= len(points):
            fail("invalid, repeated, or unsupported polygon indices", face_index=face_id, loop=list(loop))
        try:
            projected, sign, eps = _project(points[list(loop)])
        except ValueError as exc:
            fail(str(exc), face_index=face_id, loop=list(loop))
        for i, a in enumerate(loop):
            incidence[_edge(a, loop[(i + 1) % len(loop)])].append(face_id)
        prepared.append((loop, projected, sign, eps))
    boundary = [e for e, faces in incidence.items() if len(faces) == 1]
    nonmanifold = [e for e, faces in incidence.items() if len(faces) > 2]
    report["native_topology"] = {"passed": not boundary and not nonmanifold, "boundary_edges": len(boundary), "nonmanifold_edges": len(nonmanifold), "examples": [{"edge": list(e), "faces": incidence[e]} for e in (boundary + nonmanifold)[:32]]}
    if boundary or nonmanifold:
        fail("native polygon topology is open or nonmanifold; conversion cannot repair it")
    all_native = set(incidence)
    reserved: dict[tuple[int, int], int] = {}
    accepted: dict[int, list[tuple[int, int, int]]] = {}
    alternatives = []
    changed = []
    for face_id, (loop, projected, sign, eps) in enumerate(prepared):
        own = {_edge(loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))}
        if len(loop) == 3:
            accepted[face_id] = [loop]
            continue
        if len(loop) != 4:
            alternatives.append((3, face_id, None))
            continue
        candidates = []
        rejected = []
        for diagonal, local in (([0, 2], [(0, 1, 2), (2, 3, 0)]), ([1, 3], [(0, 1, 3), (1, 2, 3)])):
            tris = [tuple(loop[i] for i in tri) for tri in local]
            internal = _internal_edges(tris, own)
            collisions = internal & all_native
            valid = _valid_triangles(local, projected, sign, eps)
            if valid and not collisions:
                candidates.append((diagonal, tris, internal))
            else:
                rejected.append({"diagonal": [loop[i] for i in diagonal], "geometry_valid": valid, "native_collisions": [{"edge": list(e), "faces": incidence[e]} for e in sorted(collisions)]})
        if not candidates:
            fail("no valid unoccupied quad diagonal", face_index=face_id, loop=list(loop), rejected_diagonals=rejected)
        alternatives.append((len(candidates), face_id, candidates))
    # Forced quads first avoids wasting their only valid diagonal on a flexible
    # neighbor. Global choices are deterministic and explicitly fail closed.
    for _, face_id, candidates in sorted(alternatives):
        loop, projected, sign, eps = prepared[face_id]
        own = {_edge(loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))}
        if candidates is None:
            tris = _ear_clip(loop, projected, sign, eps, all_native | set(reserved), max_ear_states)
            if tris is None:
                fail("no admissible n-gon triangulation within search bound", face_index=face_id, loop=list(loop), reserved_conflicts=[{"edge": list(e), "face": owner} for e, owner in reserved.items() if e[0] in loop and e[1] in loop])
            internal = _internal_edges(tris, own)
        else:
            chosen = next((candidate for candidate in candidates if not candidate[2].intersection(reserved)), None)
            if chosen is None:
                fail("quad diagonals conflict with previously reserved internal edges", face_index=face_id, loop=list(loop), reserved_conflicts=[{"edge": list(e), "face": reserved[e]} for _, _, edges in candidates for e in edges if e in reserved])
            diagonal, tris, internal = chosen
            if diagonal == [1, 3]:
                changed.append(face_id)
        if internal & (all_native | set(reserved)):
            fail("internal reservation invariant failed", face_index=face_id)
        for edge in internal:
            reserved[edge] = face_id
        accepted[face_id] = tris
    triangles = np.asarray([tri for face_id in range(len(polygons)) for tri in accepted[face_id]], dtype=np.int64)
    tri_incidence: dict[tuple[int, int], int] = defaultdict(int)
    for tri in triangles:
        for i in range(3):
            tri_incidence[_edge(int(tri[i]), int(tri[(i + 1) % 3]))] += 1
    bad_edges = [(edge, count) for edge, count in tri_incidence.items() if count != 2]
    expected = sum(len(loop) - 2 for loop in polygons)
    if bad_edges or len(triangles) != expected:
        fail("triangulated topology invariant failed", triangle_count=len(triangles), expected_triangles=expected, invalid_edges=[{"edge": list(e), "incidence": count} for e, count in bad_edges[:32]])
    report.update(passed=True, triangles=len(triangles), internal_edges=len(reserved), alternate_quad_faces=changed, alternate_quad_count=len(changed), triangulated_edge_incidence=2)
    return triangles, report


def load_polygon_obj(path: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Read only geometric OBJ data; corner UVs/normals are irrelevant pre-Paint."""
    data = path.read_bytes()
    vertices, polygons = [], []
    try:
        for line in data.decode("utf-8-sig").splitlines():
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                if len(fields) != 4:
                    raise ValueError("only xyz geometric vertices are supported")
                vertices.append(tuple(float(v) for v in fields[1:]))
            elif fields[0] == "f":
                loop = []
                for token in fields[1:]:
                    value = int(token.split("/", 1)[0])
                    if value == 0:
                        raise ValueError("zero OBJ vertex index")
                    index = value - 1 if value > 0 else len(vertices) + value
                    if index < 0:
                        raise ValueError("invalid relative OBJ vertex index")
                    loop.append(index)
                polygons.append(tuple(loop))
        faces, report = triangulate_polygons(vertices, polygons)
    except PolygonConversionError as exc:
        exc.report.update(input_path=str(path), input_sha256=hashlib.sha256(data).hexdigest())
        raise
    except (UnicodeError, ValueError) as exc:
        raise PolygonConversionError({"passed": False, "reason": f"OBJ parse failed: {exc}", "input_path": str(path), "input_sha256": hashlib.sha256(data).hexdigest()}) from exc
    report.update(input_path=str(path), input_sha256=hashlib.sha256(data).hexdigest())
    return np.asarray(vertices, dtype=np.float64), faces, report
