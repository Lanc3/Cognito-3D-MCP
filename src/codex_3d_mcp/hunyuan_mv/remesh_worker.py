"""CPU-only geometry conversion and gates; run inside the Hunyuan Python runtime."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _progress(message: str) -> None:
    print(f"[geometry {time.monotonic():.3f}] {message}", flush=True)


def _isolate_stdin() -> None:
    """File-driven workers must not inherit MCP's actively read input pipe.

    Defend at entry as well as Popen so already-running server processes can
    launch an updated helper without inheriting the Windows OpenBLAS pipe hang.
    """
    descriptor = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(descriptor, 0)
    finally:
        if descriptor != 0:
            os.close(descriptor)
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
        kernel.SetStdHandle.restype = wintypes.BOOL
        if not kernel.SetStdHandle(0xFFFFFFF6, msvcrt.get_osfhandle(0)):
            raise ctypes.WinError(ctypes.get_last_error())


def _load(path: Path) -> Any:
    _progress("Importing trimesh")
    import trimesh

    _progress(f"Loading mesh: {path}")
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Empty or unsupported mesh: {path}")
    _progress(f"Loaded {len(mesh.vertices)} vertices and {len(mesh.faces)} triangles")
    return mesh


def prepare(request: dict[str, Any]) -> None:
    _progress("Prepare: importing numpy")
    import numpy as np

    _progress("Prepare: numpy imported; loading source")
    attempt = Path(request["attempt"])
    mesh = _load(Path(request["source"]))
    _progress("Prepare: checking bounds and finite geometry")
    if not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.area):
        raise ValueError("Shape has non-finite vertices or surface area")
    center, longest = mesh.bounds.mean(axis=0), float(mesh.extents.max())
    if longest <= 1e-10:
        raise ValueError("Shape bounds are degenerate")
    _progress("Prepare: normalizing and cleaning geometry")
    mesh.vertices = (mesh.vertices - center) * (2.0 / longest)
    mesh.merge_vertices(digits_vertex=6)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) < 4:
        raise ValueError("Shape has too few usable triangles after cleanup")
    mesh.fix_normals(multibody=True)
    _progress(f"Prepare: cleanup complete; exporting {len(mesh.faces)} triangles")
    mesh.export(attempt / "normalized.obj", include_texture=False)
    _progress("Prepare: OBJ exported; writing normalization report")
    (attempt / "normalization.json").write_text(
        json.dumps(
            {
                "original_center": center.tolist(),
                "uniform_scale": 2.0 / longest,
                "weld_precision": 1e-6,
                "source_faces": len(mesh.faces),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _progress("Prepare: complete")


def _components(mesh: Any) -> list[dict[str, Any]]:
    return [
        {"area": float(part.area), "bounds": part.bounds.tolist(), "faces": len(part.faces)}
        for part in mesh.split(only_watertight=False)
        if len(part.faces) > 0
    ]


def _points(mesh: Any, count: int = 4096) -> Any:
    import numpy as np

    rng = np.random.default_rng(42)
    areas = mesh.area_faces
    indices = rng.choice(len(areas), size=count, p=areas / areas.sum())
    triangles = mesh.triangles[indices]
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1.0] = 1.0 - uv[uv.sum(axis=1) > 1.0]
    surface = triangles[:, 0] + uv[:, :1] * (triangles[:, 1] - triangles[:, 0])
    surface += uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    # Include vertices so small tips and boundary artifacts are not hidden by area sampling.
    indices = np.linspace(0, len(mesh.vertices) - 1, min(count, len(mesh.vertices)), dtype=int)
    return np.concatenate([surface, mesh.vertices[indices]])


def _surface_distances(points: Any, mesh: Any) -> Any:
    """Bounded-memory deterministic nearest-candidate-triangle approximation."""
    import numpy as np
    from scipy.spatial import cKDTree
    from trimesh.triangles import closest_point

    triangles = mesh.triangles
    tree = cKDTree(triangles.mean(axis=1))
    k = min(32, len(triangles))
    _, candidates = tree.query(points, k=k, workers=1)
    candidates = candidates.reshape((len(points), k))
    values = []
    for start in range(0, len(points), 128):
        subset = points[start : start + 128]
        chosen = triangles[candidates[start : start + 128]].reshape((-1, 3, 3))
        repeated = np.repeat(subset, k, axis=0)
        nearest = closest_point(chosen, repeated)
        values.append(np.linalg.norm(nearest - repeated, axis=1).reshape((-1, k)).min(axis=1))
    return np.concatenate(values)


def _distance_summary(values: Any) -> dict[str, Any]:
    """Preserve failed measurements as JSON-safe diagnostics, never drop them."""
    import numpy as np

    samples = np.asarray(values, dtype=float)
    nonfinite = int((~np.isfinite(samples)).sum())
    summary = {"evaluated": False, "p95_fraction": None, "max_fraction": None,
               "sample_count": int(samples.size), "nonfinite_samples": nonfinite}
    if not samples.size or nonfinite:
        summary["reason"] = "nonfinite_distance_samples" if nonfinite else "empty_distance_samples"
        return summary
    p95, maximum = float(np.quantile(samples, 0.95)), float(samples.max())
    if not np.isfinite([p95, maximum]).all():
        summary["reason"] = "nonfinite_distance_statistics"
        return summary
    summary.update(evaluated=True, p95_fraction=p95, max_fraction=maximum)
    return summary


def _preservation_checks(source: Any, mesh: Any, diagonal: float, threshold: float) -> list[dict[str, Any]]:
    """Keep both preservation gates failed when proximity is undefined."""
    import numpy as np

    invalid = {"source_degenerate_faces": int((~source.nondegenerate_faces()).sum()),
               "remeshed_degenerate_faces": int((~mesh.nondegenerate_faces()).sum())}
    if any(invalid.values()):
        reason = {"evaluated": False, "reason": "degenerate_geometry_proximity_skipped", **invalid}
        return [
            {"name": "surface_preserved", "passed": False, "p95_fraction": None, "max_fraction": None,
             "p95_limit": 0.025, "max_limit": 0.075, **reason},
            {"name": "component_surfaces_preserved", "passed": False,
             "component_p95_fractions": None, "maximum": 0.025, **reason},
        ]
    _progress("Validate: sampling bidirectional surface distances")
    forward = _surface_distances(_points(source), mesh) / diagonal
    backward = _surface_distances(_points(mesh), source) / diagonal
    summary = _distance_summary(np.concatenate([forward, backward]))
    checks = [{
        "name": "surface_preserved",
        "passed": bool(summary["evaluated"] and summary["p95_fraction"] <= 0.025 and summary["max_fraction"] <= 0.075),
        **summary, "p95_limit": 0.025, "max_limit": 0.075,
        "method": "bidirectional deterministic samples, nearest of 32 centroid-neighbor triangles",
    }]
    _progress("Validate: sampling significant component surfaces")
    losses, diagnostics = [], []
    for index, part in enumerate(source.split(only_watertight=False)):
        if part.area < threshold:
            continue
        measured = _distance_summary(_surface_distances(_points(part, 256), mesh) / diagonal)
        losses.append(measured["p95_fraction"])
        if not measured["evaluated"]:
            diagnostics.append({"component_index": index, **measured})
    checks.append({
        "name": "component_surfaces_preserved",
        "passed": bool(losses) and all(error is not None and error <= 0.025 for error in losses),
        "evaluated": bool(losses) and not diagnostics,
        "component_p95_fractions": losses, "maximum": 0.025,
        "invalid_components": diagnostics,
        **({"reason": "no_significant_components"} if not losses else {}),
    })
    return checks


def _remove_duplicate_polygons(native: Path, cleaned: Path) -> dict[str, Any]:
    """Keep the first exact vertex-index loop, ignoring its start and direction.

    Preserve every other OBJ line byte-for-byte; do not weld, fill holes, alter
    vertices, or change the retained polygon's winding or corner attributes.
    """
    if native.resolve() == cleaned.resolve():
        raise ValueError("Cleaned OBJ must not overwrite the native artifact")
    seen: set[tuple[int, ...]] = set()
    vertices = native_faces = native_quads = removed = cleaned_quads = 0
    with native.open("rb") as source, cleaned.open("wb") as destination:
        for line in source:
            fields = line.split(b"#", 1)[0].split()
            if fields and fields[0] == b"v":
                vertices += 1
            if fields and fields[0] == b"f":
                indices = tuple(int(token.split(b"/", 1)[0]) for token in fields[1:])
                if any(index == 0 for index in indices):
                    raise ValueError("Native OBJ contains an invalid zero vertex index")
                indices = tuple(index if index > 0 else vertices + index + 1 for index in indices)
                if len(indices) < 3 or any(index <= 0 for index in indices):
                    raise ValueError("Native OBJ contains an invalid polygon index loop")
                native_faces += 1
                native_quads += len(indices) == 4
                reverse = indices[::-1]
                key = min(
                    ring[start:] + ring[:start]
                    for ring in (indices, reverse)
                    for start in range(len(ring))
                )
                if key in seen:
                    removed += 1
                    continue
                seen.add(key)
                cleaned_quads += len(indices) == 4
            destination.write(line)
    return {
        "method": "exact vertex-index polygon loops modulo cyclic start and reversal",
        "native_path": str(native),
        "cleaned_path": str(cleaned),
        "removed_duplicate_polygons": removed,
        "native_faces": native_faces,
        "native_quads": native_quads,
        "cleaned_faces": native_faces - removed,
        "cleaned_quads": cleaned_quads,
    }


def validate(request: dict[str, Any]) -> dict[str, Any]:
    _progress("Validate: importing numpy")
    import numpy as np

    _progress("Validate: numpy imported; checking native output")
    attempt, options = Path(request["attempt"]), request["params"]
    native = attempt / "quads.obj"
    cleaned = attempt / "quads.cleaned.obj"
    checks = []

    def gate(name: str, passed: bool, **evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **evidence})
        _progress(f"Validate: {name}: {'passed' if passed else 'failed'}")

    artifacts = {
        "normalized_obj": str(attempt / "normalized.obj"),
        "quad_obj": str(cleaned),
        "native_quad_obj": str(native),
        "native_report": str(attempt / "native-report.txt"),
        "normalization": str(attempt / "normalization.json"),
        "validation_report": str(attempt / "result.json"),
        "log": str(attempt / "process.log"),
    }
    result = {"passed": False, "checks": checks, "artifacts": artifacts, "geometry": {}}
    gate("native_output_exists", native.is_file() and native.stat().st_size > 0)
    if not checks[-1]["passed"]:
        return result
    _progress("Validate: removing exact duplicate polygon loops; retaining native OBJ")
    cleanup = _remove_duplicate_polygons(native, cleaned)
    result["cleanup"] = cleanup
    cleanup_path = attempt / "quads.cleanup.json"
    cleanup_path.write_text(json.dumps(cleanup, indent=2), encoding="utf-8")
    artifacts["duplicate_cleanup"] = str(cleanup_path)
    _progress(f"Validate: removed {cleanup['removed_duplicate_polygons']} duplicate polygons")
    _progress("Validate: scanning cleaned face sizes")
    face_sizes = []
    with cleaned.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split("#", 1)[0].split()
            if fields and fields[0] == "f":
                face_sizes.append(len(fields) - 1)
    gate("native_faces_present", len(face_sizes) > 0)
    if not face_sizes:
        return result
    _progress("Validate: triangulating cleaned polygons with global edge reservations")
    import trimesh
    if __package__:
        from .remesh_triangulation import PolygonConversionError, load_polygon_obj
    else:
        from remesh_triangulation import PolygonConversionError, load_polygon_obj

    conversion_path = attempt / "quads.triangulation.json"
    artifacts["polygon_triangulation"] = str(conversion_path)
    try:
        vertices, triangles, conversion = load_polygon_obj(cleaned)
    except PolygonConversionError as exc:
        conversion = exc.report
        conversion_path.write_text(json.dumps(conversion, indent=2, allow_nan=False), encoding="utf-8")
        native_topology = conversion.get("native_topology")
        if native_topology is not None:
            gate("native_polygon_topology", **native_topology)
        gate("polygon_triangulation", False, **{key: value for key, value in conversion.items() if key != "passed"})
        result["triangulation"] = conversion
        return result
    conversion_path.write_text(json.dumps(conversion, indent=2, allow_nan=False), encoding="utf-8")
    result["triangulation"] = conversion
    gate("native_polygon_topology", **conversion["native_topology"])
    gate("polygon_triangulation", True, report=str(conversion_path))
    source = _load(attempt / "normalized.obj")
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    diagonal = float(np.linalg.norm(source.extents))
    source_valid = bool(np.isfinite(source.vertices).all() and np.isfinite(source.area_faces).all()
                        and np.isfinite(source.area) and source.area > 0
                        and np.isfinite(diagonal) and diagonal > 0)
    gate("source_geometry_valid", source_valid)
    if not source_valid:
        return result
    gate("source_no_degenerate_faces", bool(source.nondegenerate_faces().all()))
    _progress("Validate: welding attribute splits and checking topology")
    mesh.merge_vertices(digits_vertex=6)
    mesh.remove_unreferenced_vertices()
    finite = bool(np.isfinite(mesh.vertices).all() and np.isfinite(mesh.area_faces).all()
                  and np.isfinite(mesh.area) and mesh.area > 0)
    gate("finite_geometry", finite)
    if not finite:
        return result
    quad_fraction = face_sizes.count(4) / len(face_sizes)
    # Quads are retained in OBJ; GLB intentionally stores their triangulated equivalent.
    gate("quad_dominant", quad_fraction >= 0.85, fraction=quad_fraction, minimum=0.85)
    gate(
        "face_budget",
        len(mesh.faces) <= int(options.get("triangle_budget", options["target_quads"] * 3)),
        triangles=len(mesh.faces),
        maximum=int(options.get("triangle_budget", options["target_quads"] * 3)),
        target_quads=int(options["target_quads"]),
    )
    gate("watertight", mesh.is_watertight)
    gate("consistent_winding", mesh.is_winding_consistent)
    gate("no_degenerate_faces", bool(mesh.nondegenerate_faces().all()))
    _progress("Validate: inspecting connected components")
    source_parts, parts = _components(source), _components(mesh)
    threshold = float(source.area) * 0.005
    significant_before = sum(part["area"] >= threshold for part in source_parts)
    significant_after = sum(part["area"] >= threshold for part in parts)
    gate(
        "significant_components_preserved",
        significant_before == significant_after,
        source=significant_before,
        remeshed=significant_after,
        area_fraction_threshold=0.005,
    )
    bound_error = float(np.abs(source.bounds - mesh.bounds).max() / diagonal)
    gate("bounds_preserved", bound_error <= 0.05, error_fraction=bound_error, maximum=0.05)
    for check in _preservation_checks(source, mesh, diagonal, threshold):
        gate(**check)
    candidate = attempt / "candidate.glb"
    _progress("Validate: exporting candidate GLB")
    mesh.export(candidate)
    _progress("Validate: candidate exported; collecting result")
    artifacts["candidate_glb"] = str(candidate)
    result["geometry"] = {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "native_quads": cleanup["native_quads"],
        "native_faces": cleanup["native_faces"],
        "cleaned_quads": face_sizes.count(4),
        "cleaned_faces": len(face_sizes),
        "removed_duplicate_polygons": cleanup["removed_duplicate_polygons"],
        "triangle_budget": int(options.get("triangle_budget", options["target_quads"] * 3)),
        "source_components": source_parts,
        "components": parts,
        "bounds": mesh.bounds.tolist(),
        "area": float(mesh.area),
    }
    result["passed"] = all(item["passed"] for item in checks)
    _progress(f"Validate: complete; passed={result['passed']}")
    return result


def main() -> None:
    import faulthandler

    _isolate_stdin()
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        operation, request_path = sys.argv[1:]
        _progress(f"Starting {operation}; reading request: {request_path}")
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        if operation == "prepare":
            prepare(request)
        elif operation == "validate":
            result = validate(request)
            _progress("Validate: writing result report")
            (Path(request["attempt"]) / "result.json").write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
            )
            _progress("Validate: result report written")
        else:
            raise ValueError(f"Unknown geometry operation: {operation}")
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
