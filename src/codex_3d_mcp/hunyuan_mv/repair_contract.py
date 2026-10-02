"""Strict, dependency-light contract for immutable shape repair candidates.

Coordinates are baked GLB world coordinates. Nothing here normalizes geometry.
Capability discovery is informational; worker import/operation errors fail closed.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import importlib.machinery
import math
import re
import sys
from typing import Any

REPAIR_GATE_VERSION = "shape-repair-v1"
REPAIR_METHODS = (
    "analyze", "conservative_cleanup", "patch_selected_holes",
    "remove_selected_components", "foundation_union", "import_candidate",
    "cumesh_diagnose", "cumesh_cleanup", "cumesh_fill_selected_holes",
)


def _object(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - keys:
        raise ValueError(f"{name}: expected object with keys {sorted(keys)}")
    return dict(value)


def _number(value: Any, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name}: expected finite {'positive ' if positive else ''}number")
    return value


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: expected integer in [{low}, {high}]")
    return value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name}: expected boolean")
    return value


def _vector(value: Any, name: str, length: int = 3) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name}: expected {length} coordinates")
    return [_number(item, name, positive=False) for item in value]


def _bounds(value: Any, name: str) -> dict[str, list[float]]:
    data = _object(value, {"min", "max"}, name)
    lo, hi = _vector(data.get("min"), name), _vector(data.get("max"), name)
    if any(a >= b for a, b in zip(lo, hi)):
        raise ValueError(f"{name}: each minimum must be below maximum")
    return {"min": lo, "max": hi}


def _ids(value: Any, prefix: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or len(value) > 4096 or (required and not value):
        raise ValueError(f"{prefix} ids: expected bounded nonempty list" if required
                         else f"{prefix} ids: expected bounded list")
    if any(not isinstance(item, str) or not re.fullmatch(prefix + r":[0-9a-f]{64}", item)
           for item in value) or len(set(value)) != len(value):
        raise ValueError(f"{prefix} ids: expected unique diagnostic identifiers")
    return list(value)


def _primitive(value: Any) -> dict[str, Any]:
    data = _object(value, {"kind", "min", "max", "up_axis", "rings"}, "primitive")
    kind = data.get("kind")
    if kind == "box":
        if set(data) - {"kind", "min", "max"}:
            raise ValueError("box primitive: unknown keys")
        return {"kind": kind, **_bounds({k: data.get(k) for k in ("min", "max")}, "box")}
    if kind not in {"loft", "footprint"} or set(data) - {"kind", "up_axis", "rings"}:
        raise ValueError("primitive: use authored box, footprint or loft")
    axis = _integer(data.get("up_axis"), "primitive.up_axis", 0, 2)
    rings = data.get("rings")
    if not isinstance(rings, list) or not 2 <= len(rings) <= 64:
        raise ValueError("primitive.rings: expected 2..64 rings")
    result = []
    for ring in rings:
        ring = _object(ring, {"height", "points"}, "ring")
        points = ring.get("points")
        if not isinstance(points, list) or not 3 <= len(points) <= 256:
            raise ValueError("ring.points: expected 3..256 corresponding perimeter points")
        result.append({"height": _number(ring.get("height"), "height", positive=False),
                       "points": [_vector(p, "perimeter point", 2) for p in points]})
    if any(a["height"] >= b["height"] or len(a["points"]) != len(b["points"])
           for a, b in zip(result, result[1:])):
        raise ValueError("loft: strictly increasing heights and corresponding point counts required")
    if kind == "footprint" and (len(result) != 2 or result[0]["points"] != result[1]["points"]):
        raise ValueError("footprint: exactly two identical perimeter rings required")
    return {"kind": kind, "up_axis": axis, "rings": result}


def validate_recipe(recipe: Any) -> dict[str, Any]:
    data = _object(recipe, {"method", "parameters", "policy"}, "recipe")
    method = data.get("method")
    if method not in REPAIR_METHODS:
        raise ValueError(f"Unknown repair method: {method!r}")
    raw = data.get("parameters", {})
    parameters: dict[str, Any] = {}
    if method in {"analyze", "import_candidate", "cumesh_diagnose"}:
        _object(raw, set(), "parameters")
    elif method == "cumesh_cleanup":
        raw = _object(raw, {"remove_duplicate_faces", "remove_degenerate_faces",
                           "max_removed_faces", "max_removed_fraction"}, "parameters")
        for key in ("remove_duplicate_faces", "remove_degenerate_faces"):
            parameters[key] = _bool(raw.get(key, False), key)
        if not any(parameters.values()):
            raise ValueError("CuMesh cleanup requires explicit removal operations")
        parameters["max_removed_faces"] = _integer(raw.get("max_removed_faces", 1000),
                                                    "max_removed_faces", 0, 100000)
        parameters["max_removed_fraction"] = _number(raw.get("max_removed_fraction", 0.001),
                                                      "max_removed_fraction")
        if parameters["max_removed_fraction"] > 0.05:
            raise ValueError("CuMesh cleanup cannot remove over 5% of faces")
    elif method == "conservative_cleanup":
        raw = _object(raw, {"exact_weld", "tolerance_weld", "remove_collapsed_faces",
                           "remove_duplicate_faces", "fix_winding", "max_removed_faces",
                           "max_removed_fraction"}, "parameters")
        for key, default in (("exact_weld", True), ("remove_collapsed_faces", True),
                             ("remove_duplicate_faces", True), ("fix_winding", False)):
            parameters[key] = _bool(raw.get(key, default), key)
        tolerance = raw.get("tolerance_weld")
        if tolerance is not None:
            tolerance = _object(tolerance, {"distance", "max_displacement"}, "tolerance_weld")
            tolerance = {key: _number(tolerance.get(key), key)
                         for key in ("distance", "max_displacement")}
            if tolerance["max_displacement"] > tolerance["distance"]:
                raise ValueError("max_displacement must not exceed weld distance")
        parameters["tolerance_weld"] = tolerance
        parameters["max_removed_faces"] = _integer(raw.get("max_removed_faces", 1000),
                                                   "max_removed_faces", 0, 100000)
        fraction = _number(raw.get("max_removed_fraction", 0.001), "max_removed_fraction")
        if fraction > 0.05:
            raise ValueError("conservative cleanup cannot remove over 5% of faces")
        parameters["max_removed_fraction"] = fraction
    elif method in {"patch_selected_holes", "cumesh_fill_selected_holes"}:
        gpu = method == "cumesh_fill_selected_holes"
        raw = _object(raw, {"loop_ids", "max_diameter", "max_edges", "max_patch_area",
                           "min_triangle_area"} | ({"max_hole_perimeter"} if gpu else {"backend"}), "parameters")
        parameters["loop_ids"] = _ids(raw.get("loop_ids"), "loop", required=True)
        if len(parameters["loop_ids"]) > 64:
            raise ValueError("At most 64 loops per repair")
        parameters["max_edges"] = _integer(raw.get("max_edges", 3), "max_edges", 3, 256)
        for key in ("max_diameter", "max_patch_area"):
            parameters[key] = _number(raw.get(key), key)
        parameters["min_triangle_area"] = _number(raw.get("min_triangle_area", 1e-12),
                                                    "min_triangle_area")
        if gpu:
            parameters["max_hole_perimeter"] = _number(raw.get("max_hole_perimeter"), "max_hole_perimeter")
        else:
            parameters["backend"] = raw.get("backend", "triangle")
            if parameters["backend"] not in {"triangle", "pymeshlab"}:
                raise ValueError("hole backend: triangle or pymeshlab required")
    elif method == "remove_selected_components":
        raw = _object(raw, {"component_ids", "max_total_faces", "max_total_area"}, "parameters")
        parameters["component_ids"] = _ids(raw.get("component_ids"), "component", required=True)
        parameters["max_total_faces"] = _integer(raw.get("max_total_faces"),
                                                 "max_total_faces", 1, 1000000)
        parameters["max_total_area"] = _number(raw.get("max_total_area"), "max_total_area")
    elif method == "foundation_union":
        raw = _object(raw, {"primitive", "backend", "restore_outside_faces"}, "parameters")
        if raw.get("backend", "manifold3d") != "manifold3d":
            raise ValueError("foundation backend: manifold3d required")
        parameters = {"primitive": _primitive(raw.get("primitive")), "backend": "manifold3d",
            "restore_outside_faces": _bool(raw.get("restore_outside_faces", False), "restore_outside_faces")}

    policy = _object(data.get("policy", {}), {"roi", "protected_component_ids", "require_closed",
        "require_vertex_manifold", "require_self_intersection_free",
        "require_outside_roi_preserved", "foundation"}, "policy")
    normalized: dict[str, Any] = {"roi": None}
    if policy.get("roi") is not None:
        normalized["roi"] = _bounds(policy["roi"], "policy.roi")
    if method not in {"analyze", "cumesh_diagnose"} and normalized["roi"] is None:
        raise ValueError("Every mutation/import requires an explicit world-coordinate policy.roi")
    normalized["protected_component_ids"] = _ids(policy.get("protected_component_ids", []),
                                                  "component")
    for key in ("require_closed", "require_vertex_manifold", "require_self_intersection_free",
                "require_outside_roi_preserved"):
        normalized[key] = _bool(policy.get(key, True), key)
        if not normalized[key]:
            raise ValueError(f"{key} cannot be disabled in repair gate v1")
    foundation = policy.get("foundation")
    if foundation is not None:
        foundation = _object(foundation, {"roi", "up_axis", "grid_size", "surface_tolerance"},
                             "foundation")
        foundation = {
            "roi": _bounds(foundation.get("roi"), "foundation.roi"),
            "up_axis": _integer(foundation.get("up_axis"), "foundation.up_axis", 0, 2),
            "grid_size": _integer(foundation.get("grid_size", 5), "foundation.grid_size", 3, 9),
            "surface_tolerance": _number(foundation.get("surface_tolerance", 1e-6),
                                          "foundation.surface_tolerance"),
        }
        shortest = min(b - a for a, b in zip(foundation["roi"]["min"], foundation["roi"]["max"]))
        if foundation["surface_tolerance"] > shortest * 0.01:
            raise ValueError("foundation tolerance must not exceed 1% of the shortest ROI dimension")
    if method == "foundation_union" and foundation is None:
        raise ValueError("foundation_union requires an explicit occupied-volume foundation policy")
    normalized["foundation"] = foundation
    return {"method": method, "parameters": parameters, "policy": normalized}


def _recipe_contract() -> dict[str, Any]:
    """Discoverable recipe fields; validation remains authoritative and fail closed."""
    return {
        "shape": {"method": "one of methods below", "parameters": "method-specific object",
                  "policy": "shared policy object"},
        "unknown_fields": "rejected", "numeric_values": "finite, JSON numbers; booleans are not numbers",
        "coordinate_frame": "baked input GLB world coordinates; no automatic normalization or orientation",
        "source_binding": "loop/component IDs must come from this source artifact's hash-bound diagnostics",
        "bounds_type": {"min": "3 finite coordinates", "max": "3 finite coordinates, each greater than min"},
        "methods": {
            "analyze": {"parameters": {}, "note": "Byte-copy and validate; no geometry edits"},
            "conservative_cleanup": {"parameters": {
                "exact_weld": {"type": "boolean", "default": True},
                "tolerance_weld": {"default": None, "type": "null or object",
                    "fields": {"distance": "positive world distance, at most 0.1% of measured diagonal",
                               "max_displacement": "positive, at most distance"},
                    "note": "Same-bin exact representative; does not perform transitive radius clustering"},
                "remove_collapsed_faces": {"type": "boolean", "default": True,
                    "note": "Only exact zero-area faces within ROI; near-degenerates retained"},
                "remove_duplicate_faces": {"type": "boolean", "default": True},
                "fix_winding": {"type": "boolean", "default": False,
                    "note": "Requires whole corrected mesh inside ROI"},
                "max_removed_faces": {"type": "integer", "minimum": 0, "maximum": 100000, "default": 1000},
                "max_removed_fraction": {"type": "number", "exclusive_minimum": 0, "maximum": 0.05, "default": 0.001}}},
            "patch_selected_holes": {"parameters": {
                "loop_ids": {"type": "unique array of loop:<64 lowercase hex> IDs", "required": True,
                             "minimum_items": 1, "maximum_items": 64},
                "max_diameter": {"type": "positive world distance", "required": True,
                    "note": "Applied to measured AABB-diagonal upper bound"},
                "max_edges": {"type": "integer", "minimum": 3, "maximum": 256, "default": 3},
                "max_patch_area": {"type": "positive world units squared", "required": True},
                "min_triangle_area": {"type": "positive world units squared", "default": 1e-12},
                "backend": {"enum": ["triangle", "pymeshlab"], "default": "triangle",
                            "note": "triangle requires exactly three edges; PyMeshLab never refines the patch"}}},
            "remove_selected_components": {"parameters": {
                "component_ids": {"type": "unique array of component:<64 lowercase hex> IDs", "required": True,
                                  "minimum_items": 1, "maximum_items": 4096},
                "max_total_faces": {"type": "integer", "minimum": 1, "maximum": 1000000, "required": True},
                "max_total_area": {"type": "positive world units squared", "required": True}},
                "note": "Cannot remove protected components or any selected geometry outside ROI"},
            "foundation_union": {"parameters": {
                "primitive": {"type": "authored primitive object described below", "required": True},
                "backend": {"enum": ["manifold3d"], "default": "manifold3d"},
                "restore_outside_faces": {"type": "boolean", "default": False,
                    "note": "Retain original crossing/outside faces plus Boolean fully-inside faces; exact weld only; all export gates still required"}},
                "note": "Valid closed input solids, positive overlap, and foundation policy required"},
            "import_candidate": {"parameters": {},
                "note": "Imported GLB path/hash supplied via submission request; identical export/preservation gates apply"},
            "cumesh_diagnose": {"parameters": {}, "backend": "cumesh",
                "note": "Explicit CUDA experiment after verified probe; original GLB byte-copy plus all common gates"},
            "cumesh_cleanup": {"backend": "cumesh", "parameters": {
                "remove_duplicate_faces": {"type": "boolean", "default": False},
                "remove_degenerate_faces": {"type": "boolean", "default": False,
                    "note": "Exact zero cross product only; no near-degenerate deletion"},
                "max_removed_faces": {"type": "integer", "minimum": 0, "maximum": 100000, "default": 1000},
                "max_removed_fraction": {"type": "number", "exclusive_minimum": 0, "maximum": 0.05, "default": 0.001}},
                "note": "At least one removal flag true. Global native cleanup must preserve every outside-ROI face; otherwise rejected. No CPU fallback."},
            "cumesh_fill_selected_holes": {"backend": "cumesh", "parameters": {
                "loop_ids": {"type": "unique source diagnostic loop:<64 lowercase hex> IDs", "required": True,
                    "minimum_items": 1, "maximum_items": 64},
                "max_diameter": {"type": "positive world distance", "required": True},
                "max_edges": {"type": "integer", "minimum": 3, "maximum": 256, "default": 3},
                "max_patch_area": {"type": "positive world units squared", "required": True},
                "min_triangle_area": {"type": "positive world units squared", "default": 1e-12},
                "max_hole_perimeter": {"type": "positive world distance", "required": True}},
                "note": "Worker recomputes canonical edges and input hash. Threshold must select precisely these planar convex loops. All export gates remain mandatory."},
        },
        "primitive": {
            "box": {"kind": "box", "min": "3 finite coordinates", "max": "3 finite coordinates greater than min"},
            "footprint_or_loft": {"kind": ["footprint", "loft"], "up_axis": [0, 1, 2],
                "rings": "2..64 objects {height: finite number, points: [[u,v], ...]}",
                "requirements": ["3..256 unique perimeter points per ring, corresponding counts",
                    "counterclockwise simple nondegenerate rings; no holes",
                    "strictly increasing ring heights", "footprint requires exactly two identical perimeters",
                    "all primitive vertices inside edit ROI"]}},
        "policy": {
            "roi": {"type": "bounds or null", "default": None,
                    "required_for": [method for method in REPAIR_METHODS if method not in {"analyze", "cumesh_diagnose"}],
                    "note": "Any face crossing outside this box is protected in full"},
            "protected_component_ids": {"type": "unique array of component:<64 lowercase hex> IDs",
                                        "default": [], "maximum_items": 4096},
            "require_closed": {"type": "boolean", "default": True, "allowed": [True]},
            "require_vertex_manifold": {"type": "boolean", "default": True, "allowed": [True]},
            "require_self_intersection_free": {"type": "boolean", "default": True, "allowed": [True]},
            "require_outside_roi_preserved": {"type": "boolean", "default": True, "allowed": [True]},
            "foundation": {"type": "null or object", "default": None, "required_for": ["foundation_union"],
                "fields": {"roi": "required bounds of volume expected continuously occupied",
                           "up_axis": "required integer 0, 1 or 2",
                           "grid_size": {"type": "integer", "minimum": 3, "maximum": 9, "default": 5},
                           "surface_tolerance": {"type": "positive world distance", "default": 1e-6,
                               "maximum": "1% of shortest foundation ROI dimension"}},
                "note": "Cell-center vertical intervals are sampled evidence, not a whole-volume guarantee"}},
        "semantic_approval": "Always separate; a passed numerical gate is not production approval",
    }


def get_capabilities(*, package_paths: list[str] | None = None,
                     worker_python: str | None = None) -> dict[str, Any]:
    from .repair_cumesh_qualification import METHODS as GPU_METHODS, discover
    roots = list(sys.path) if package_paths is None else list(package_paths)
    packages = {}
    versions = {}
    for dist in importlib.metadata.distributions(path=roots):
        name = (dist.metadata.get("Name") or "").lower().replace("-", "_")
        versions.setdefault(name, dist.version)
    for name in ("numpy", "scipy", "trimesh", "pymeshlab", "manifold3d"):
        try:
            available = importlib.machinery.PathFinder.find_spec(name, roots) is not None
            version = versions.get(name) if available else None
        except (ImportError, ValueError):
            available, version = False, None
        packages[name] = {"discoverable": available, "version": version,
                          "runtime_qualified": False}
    core = all(packages[k]["discoverable"] for k in ("numpy", "scipy", "trimesh"))
    experimental_cumesh = discover(roots, gate_version=REPAIR_GATE_VERSION)
    full_gate = core and packages["pymeshlab"]["discoverable"]
    methods = {}
    for method in REPAIR_METHODS:
        gpu = method in GPU_METHODS
        methods[method] = {"backend": "cumesh" if gpu else "cpu",
            "discoverable": (core and experimental_cumesh["package_present"] if gpu else core and (
                packages["manifold3d"]["discoverable"] if method == "foundation_union" else True)),
            "runtime_qualified": full_gate and experimental_cumesh["cuda_qualified"] if gpu else False,
            "experimental_runnable": full_gate and experimental_cumesh["cuda_qualified"] if gpu else False,
            "production_ready": full_gate and experimental_cumesh["method_production_ready"][method] if gpu else False}
    return {"gate_version": REPAIR_GATE_VERSION, "packages": packages,
            "recipe_contract": _recipe_contract(),
            "methods": methods, "worker_python": worker_python,
            "experimental_backends": {"cumesh": experimental_cumesh},
            "required_self_intersection_backend": "pymeshlab",
            "full_gate_discoverable": full_gate,
            "execution": "Serial bounded worker; explicit optional CUDA recipes; caller must hold shared GPU lease and enforce 2 CPU / 2048 MiB limits",
            "semantic_approval": "separate agent review always required"}
