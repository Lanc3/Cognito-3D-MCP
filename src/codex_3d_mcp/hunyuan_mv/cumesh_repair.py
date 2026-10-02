"""Optional CUDA candidate repair. Importing this module never loads Torch/CUDA.

The caller owns the serial GPU lease and process resource guards. Arrays use the
unchanged world frame; this adapter does not center, scale, weld, or decimate.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "12289e1062f0603f2f0d0771b02e1395d247f26f"
SOURCE_URL = "https://github.com/JeffreyXiang/CuMesh"
ADAPTER_VERSION = "cumesh-candidate-v1"
METHODS = ("cumesh_diagnose", "cumesh_cleanup", "cumesh_fill_selected_holes")


class CuMeshUnavailable(RuntimeError):
    """The optional pinned CUDA backend cannot run; there is no CPU fallback."""
    failure_kind = "runtime_unavailable"
    code = "CUMESH_BACKEND_UNAVAILABLE"


class CuMeshResourceError(CuMeshUnavailable):
    """The bounded CUDA preflight cannot retain its required reserve."""
    failure_kind = "resource"
    code = "CUMESH_RESOURCE_EXHAUSTED"


@dataclass(frozen=True)
class RepairRecipe:
    method: str
    parameters: dict[str, Any]
    policy: dict[str, Any]

    @classmethod
    def parse(cls, value: dict[str, Any]) -> RepairRecipe:
        if set(value) - {"method", "parameters", "policy"}:
            raise ValueError("Unknown CuMesh recipe fields")
        method = value.get("method")
        if method not in METHODS:
            raise ValueError(f"Unsupported CuMesh method: {method}")
        params, policy = value.get("parameters", {}), value.get("policy", {})
        if not isinstance(params, dict) or not isinstance(policy, dict):
            raise ValueError("parameters and policy must be objects")
        allowed = {
            "cumesh_diagnose": set(),
            "cumesh_cleanup": {"remove_duplicate_faces", "remove_degenerate_faces"},
            "cumesh_fill_selected_holes": {
                "selected_loop_edges", "input_geometry_sha256", "max_hole_perimeter"
            },
        }[method]
        if set(params) - allowed:
            raise ValueError("Unsupported CuMesh parameter; no blanket repair is available")
        if method == "cumesh_cleanup":
            if not params or any(type(v) is not bool for v in params.values()):
                raise ValueError("Cleanup requires explicit boolean operations")
            if not any(params.values()):
                raise ValueError("Select at least one cleanup operation")
        if method == "cumesh_fill_selected_holes":
            if set(params) != allowed or not params["selected_loop_edges"]:
                raise ValueError("Selected holes require exact edges, input hash and perimeter")
            perimeter = params["max_hole_perimeter"]
            if isinstance(perimeter, bool) or not isinstance(perimeter, (int, float)):
                raise ValueError("Perimeter must be a positive world-space number")
            if not math.isfinite(perimeter) or perimeter <= 0:
                raise ValueError("Perimeter must be positive and finite")
        return cls(method, dict(params), dict(policy))


def capabilities(*, package_paths: list[str] | None = None) -> dict[str, Any]:
    """Hash-bound metadata only; never initialize Torch/CUDA."""
    from .repair_contract import REPAIR_GATE_VERSION
    from .repair_cumesh_qualification import discover
    result = discover(list(sys.path) if package_paths is None else package_paths,
                      gate_version=REPAIR_GATE_VERSION)
    return {**result, "backend": "cumesh", "adapter_version": ADAPTER_VERSION,
        "license": "MIT", "production_qualified": result["production_ready"],
        "automatic_cpu_fallback": False, "global_remesh_enabled": False,
        "semantic_foundation_repair": False}


def _backend() -> tuple[Any, Any]:
    spec = importlib.util.find_spec("cumesh")
    if spec is None or not spec.origin:
        raise CuMeshUnavailable("CuMesh is absent from the isolated repair package target")
    package_root = Path(spec.origin).parent.parent
    try:
        manifest = json.loads((package_root / "cumesh-install-manifest.json").read_text())
        if manifest["source_commit"] != SOURCE_COMMIT:
            raise ValueError("source commit mismatch")
        native_hashes = manifest["native_file_sha256"]
        if not native_hashes:
            raise ValueError("missing native binary hashes")
        for filename, expected in native_hashes.items():
            path = (package_root / filename).resolve()
            if not path.is_relative_to(package_root.resolve()):
                raise ValueError("invalid native binary path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("native binary hash mismatch")
    except (OSError, KeyError, ValueError) as exc:
        raise CuMeshUnavailable(f"CuMesh pinned installation not verified: {exc}") from exc
    try:
        import torch
        from cumesh import CuMesh
    except (ImportError, OSError) as exc:
        raise CuMeshUnavailable(f"CuMesh CUDA extension unavailable: {exc}") from exc
    if torch.__version__.split("+")[0] != "2.9.1" or torch.version.cuda != "13.0":
        raise CuMeshUnavailable("CuMesh adapter requires the qualified Torch 2.9.1/cu130 ABI")
    if not torch.cuda.is_available():
        raise CuMeshUnavailable("CUDA is required; CPU fallback is disabled")
    return torch, CuMesh


def _arrays(vertices: Any, faces: Any) -> tuple[Any, Any]:
    import numpy as np

    v, f = np.asarray(vertices), np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("Expected [V,3] positions and [F,3] triangle indices")
    if not len(v) or not len(f) or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("Empty geometry or non-integer indices")
    if len(v) > 2_000_000 or len(f) > 2_000_000:
        raise ValueError("CuMesh candidate input exceeds the bounded two-million element limit")
    if not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
        raise ValueError("Nonfinite positions or out-of-range indices")
    vf = np.ascontiguousarray(v, dtype=np.float32)
    if not np.isfinite(vf).all():
        raise ValueError("Positions cannot be represented as float32")
    return vf, np.ascontiguousarray(f, dtype=np.int32)


def geometry_sha256(vertices: Any, faces: Any) -> str:
    """Hash the exact float32/int32 arrays passed to native CUDA, including shapes."""
    v, f = _arrays(vertices, faces)
    digest = hashlib.sha256()
    digest.update(json.dumps([list(v.shape), list(f.shape)]).encode())
    digest.update(v.astype("<f4", copy=False).tobytes())
    digest.update(f.astype("<i4", copy=False).tobytes())
    return digest.hexdigest()


def _edge_key(edges: Any) -> tuple[tuple[int, int], ...]:
    result = []
    for edge in edges:
        if len(edge) != 2 or any(type(x) is not int for x in edge):
            raise ValueError("Each selected boundary edge must contain two integer vertex IDs")
        a, b = edge
        if a < 0 or b < 0 or a == b:
            raise ValueError("Invalid boundary edge")
        result.append(tuple(sorted((a, b))))
    if len(result) < 3 or len(result) != len(set(result)):
        raise ValueError("A selected loop requires at least three distinct edges")
    return tuple(sorted(result))


def _loops(mesh: Any, vertices: Any) -> list[dict[str, Any]]:
    import numpy as np

    mesh.get_boundary_loops()
    if mesh.num_boundary_loops == 0:
        if mesh.num_boundaries:
            raise ValueError("Native boundary graph has open edges but no simple loops")
        return []
    count, indices, offsets = mesh.read_boundary_loops()
    if count > 256:
        raise ValueError("More than 256 boundary loops; bounded diagnosis required")
    # Native loop indices address the complete edge array, not read_boundaries().
    edges = mesh.read_edges().cpu().numpy()
    index, offset = indices.cpu().numpy(), offsets.cpu().numpy()
    if len(index) != mesh.num_boundaries:
        raise ValueError("Not all boundary edges belong to diagnosed simple loops")
    if len(offset) != count + 1:
        raise ValueError("Unexpected CuMesh loop offset layout")
    result = []
    for i in range(count):
        selected = edges[index[offset[i]:offset[i + 1]]]
        if len(selected) > 1024:
            raise ValueError("Loop exceeds bounded 1024-edge diagnostic limit")
        key = _edge_key(selected.astype(int).tolist())
        delta = vertices[selected[:, 0]] - vertices[selected[:, 1]]
        perimeter = float(np.linalg.norm(delta.astype(np.float64), axis=1).sum())
        result.append({"edges": [list(e) for e in key], "perimeter": perimeter,
                       "edge_count": len(key)})
    return result


def _planar_convex(edges: tuple[tuple[int, int], ...], vertices: Any) -> None:
    """CuMesh uses a midpoint fan: reject loops whose fan is not a safe planar cap."""
    import numpy as np

    neighbors: dict[int, list[int]] = {}
    for a, b in edges:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in neighbors.values()):
        raise ValueError("Selected boundary is not a simple degree-two loop")
    ordered, previous, current = [], None, min(neighbors)
    while current not in ordered:
        ordered.append(current)
        choices = neighbors[current]
        nxt = choices[0] if choices[0] != previous else choices[1]
        previous, current = current, nxt
    if current != ordered[0] or len(ordered) != len(neighbors):
        raise ValueError("Selected boundary contains disconnected cycles")
    points = vertices[ordered].astype(np.float64)
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    size = float(np.linalg.norm(np.ptp(points, axis=0)))
    if size <= 0 or np.abs(centered @ vh[-1]).max() > size * 1e-5:
        raise ValueError("Selected hole is not planar; midpoint fan is not accepted")
    xy = centered @ vh[:2].T
    e1, e2 = np.roll(xy, -1, axis=0) - xy, np.roll(xy, -2, axis=0) - np.roll(xy, -1, axis=0)
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    if not ((cross > size * size * 1e-10).all() or (cross < -size * size * 1e-10).all()):
        raise ValueError("Selected hole is not strictly convex; midpoint fan is not accepted")
    # Consistent local turns alone can accept a self-crossing star polygon.
    for i, edge in enumerate(e1):
        other = xy - xy[i]
        side = edge[0] * other[:, 1] - edge[1] * other[:, 0]
        if (side * np.sign(cross[0]) < -size * size * 1e-10).any():
            raise ValueError("Selected loop self-crosses or is not a convex polygon")


def _validate_selection(loops: list[dict[str, Any]], params: dict[str, Any], vertices: Any) -> set:
    selected = {_edge_key(loop) for loop in params["selected_loop_edges"]}
    if len(selected) != len(params["selected_loop_edges"]):
        raise ValueError("Selected loop list contains duplicates")
    actual = {_edge_key(loop["edges"]): loop for loop in loops}
    if not selected <= actual.keys():
        raise ValueError("Selected edges do not match exact CuMesh boundary loops")
    threshold = float(params["max_hole_perimeter"])
    # Public native API selects all loops below a threshold. Require that exact
    # selection, with generous float32 reduction margin, before invoking it.
    for key, loop in actual.items():
        margin = max(threshold, loop["perimeter"]) * 1e-4
        if abs(loop["perimeter"] - threshold) <= margin:
            raise ValueError("Perimeter is too close to the native float32 threshold")
        if (loop["perimeter"] < threshold) != (key in selected):
            raise ValueError("CuMesh perimeter would fill an unselected loop or miss a selected one")
    for key in selected:
        _planar_convex(key, vertices)
    return selected


def repair(vertices: Any, faces: Any, recipe: dict[str, Any], *,
           _qualification_probe: bool = False) -> tuple[Any, Any, dict[str, Any]]:
    """Run only inside an already leased, resource-bounded GPU worker process."""
    import numpy as np

    parsed = RepairRecipe.parse(recipe)
    v, f = _arrays(vertices, faces)
    input_hash = geometry_sha256(v, f)
    if parsed.method == "cumesh_fill_selected_holes":
        if parsed.parameters["input_geometry_sha256"] != input_hash:
            raise ValueError("Selected holes refer to different input geometry")
    if not _qualification_probe:
        from .repair_contract import REPAIR_GATE_VERSION
        from .repair_cumesh_qualification import discover
        qualification = discover(list(sys.path), gate_version=REPAIR_GATE_VERSION, fresh=True)
        if not qualification["cuda_qualified"]:
            raise CuMeshUnavailable(qualification["reason"])
    torch, cls = _backend()
    torch.set_num_threads(2)
    free, _ = torch.cuda.mem_get_info()
    # Conservative bounded preflight. Native allocations bypass Torch's allocator;
    # this is an input ceiling/reserve check, not a claimed hard VRAM cap.
    estimated = int((len(v) + len(f)) * 512 + 256 * 1024**2)
    if free < estimated + 2 * 1024**3:
        raise CuMeshResourceError("Insufficient free VRAM for conservative repair estimate and reserve")
    mesh = cls()
    mesh.init(torch.from_numpy(v).cuda(), torch.from_numpy(f).cuda())
    # Corrupt degenerate connectivity need not enter loop construction before
    # its explicitly requested cleanup. CPU/source diagnostics remain authoritative.
    before = _loops(mesh, v) if parsed.method != "cumesh_cleanup" else None
    operations = []
    if parsed.method == "cumesh_cleanup":
        if parsed.parameters.get("remove_duplicate_faces"):
            mesh.remove_duplicate_faces()
            operations.append("remove_duplicate_faces")
        if parsed.parameters.get("remove_degenerate_faces"):
            # Native threshold is strictly '<', so (0,0) would retain collinear
            # faces. Evaluate exact zero cross product on GPU in float64 instead.
            current_v, current_f = mesh.read()
            triangles = current_v.double()[current_f.long()]
            cross = torch.linalg.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0])
            keep = (cross != 0).any(dim=1)
            if not bool(keep.any()):
                raise ValueError("Exact degenerate cleanup would leave an empty candidate")
            mesh = cls()
            mesh.init(current_v.contiguous(), current_f[keep].contiguous())
            operations.append("remove_exact_degenerate_faces_cuda_float64")
    elif parsed.method == "cumesh_fill_selected_holes":
        selected = _validate_selection(before, parsed.parameters, v)
        mesh.fill_holes(float(parsed.parameters["max_hole_perimeter"]))
        operations.append("fill_exact_selected_planar_convex_loops")
    out_v, out_f = mesh.read()
    torch.cuda.synchronize()
    result_v, result_f = out_v.cpu().numpy(), out_f.cpu().numpy()
    _arrays(result_v, result_f)
    # Reinitialize so cached connectivity cannot refer to pre-mutation geometry.
    inspected = cls()
    inspected.init(out_v.contiguous(), out_f.contiguous())
    after = _loops(inspected, result_v)
    if parsed.method == "cumesh_fill_selected_holes":
        expected = {_edge_key(x["edges"]) for x in before} - selected
        if {_edge_key(x["edges"]) for x in after} != expected:
            raise ValueError("Native fill changed unexpected boundaries; reject candidate")
        if not np.array_equal(result_v[:len(v)], v) or not np.array_equal(result_f[:len(f)], f):
            raise ValueError("Native hole fill changed pre-existing geometry; reject candidate")
    report = {"backend": "cumesh", "source_commit": SOURCE_COMMIT,
              "adapter_version": ADAPTER_VERSION, "device": torch.cuda.get_device_name(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "input_geometry_sha256": input_hash,
              "output_geometry_sha256": geometry_sha256(result_v, result_f),
              "method": parsed.method, "operations": operations,
              "input_vertices": len(v), "input_faces": len(f),
              "output_vertices": len(result_v), "output_faces": len(result_f),
              "boundaries_before": before, "boundaries_after": after,
              "policy": parsed.policy, "policy_validation": "external_geometry_gate_required",
              "coordinate_transform": "identity", "native_dtype": "float32/int32",
              "candidate_only": True, "watertight_verified": False,
              "semantic_foundation_verified": False, "estimated_workspace_bytes": estimated}
    return result_v.astype(np.float64), result_f.astype(np.int64), report


def probe(output: Path) -> dict[str, Any]:
    """Tiny empirical CUDA checks; the external caller must hold the same guards."""
    import numpy as np

    vertices = np.array([[10, 20, 30], [11, 20, 30], [10, 21, 30], [10, 20, 31]])
    closed = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    recipe = {"method": "cumesh_cleanup", "parameters": {"remove_duplicate_faces": True}}
    v, f, cleaned = repair(vertices, np.concatenate([closed, closed[:1]]), recipe,
                           _qualification_probe=True)
    if len(f) != 4 or sorted(map(tuple, v)) != sorted(map(tuple, vertices)):
        raise RuntimeError("CUDA duplicate cleanup qualification failed")
    opened = closed[:-1]
    _, _, diagnostic = repair(vertices, opened, {"method": "cumesh_diagnose"}, _qualification_probe=True)
    loops = diagnostic["boundaries_before"]
    if len(loops) != 1:
        raise RuntimeError("CUDA boundary-loop qualification failed")
    _, f, filled = repair(vertices, opened, {
        "method": "cumesh_fill_selected_holes", "parameters": {
            "input_geometry_sha256": geometry_sha256(vertices, opened),
            "selected_loop_edges": [loops[0]["edges"]],
            "max_hole_perimeter": loops[0]["perimeter"] * 1.1,
        }}, _qualification_probe=True)
    if filled["boundaries_after"] or len(f) != 6:
        raise RuntimeError("CUDA selected hole qualification failed")
    report = {"cuda_verified": True, "production_qualified": False,
              "source_commit": SOURCE_COMMIT, "adapter_version": ADAPTER_VERSION,
              "checks": {"duplicate_cleanup": cleaned, "loop_diagnostic": diagnostic,
                         "selected_planar_hole": filled}}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
