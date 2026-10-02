"""Read-only, dependency-light verification of portable CuMesh qualification.

Manifest booleans are never qualification. Verify the retained native probe and
actual common-worker result artifacts against current code and native binaries.
This module imports neither Torch nor the native package.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "12289e1062f0603f2f0d0771b02e1395d247f26f"
METHODS = ("cumesh_diagnose", "cumesh_cleanup", "cumesh_fill_selected_holes")
CODE_FILES = ("cumesh_repair.py", "repair_worker.py", "repair_geometry.py",
              "repair_contract.py", "repair_cumesh_qualification.py")
REQUIRED_CHECKS = {"export_closed_edges", "export_consistent_winding",
    "export_no_collapsed_or_duplicate_faces", "export_vertex_manifold",
    "export_self_intersections", "outside_roi_preserved",
    "protected_components_preserved", "raw_source_unchanged"}


@lru_cache(maxsize=128)
def _cached_hash(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hash(path: Path, *, fresh: bool = False) -> str:
    path = path.resolve(strict=True)
    state = path.stat()
    if fresh:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    return _cached_hash(str(path), state.st_size, state.st_mtime_ns, state.st_ctime_ns)


def _json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Qualification JSON exceeds 4 MiB")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Qualification evidence must be an object")
    return data


def _local(root: Path, path: Any) -> Path:
    if not isinstance(path, str) or not path:
        raise ValueError("Missing qualification artifact path")
    candidate = (root / path).resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError("Qualification artifact must be a file under package root")
    return candidate


def _artifact(root: Path, entry: Any, *, fresh: bool) -> Path:
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        raise ValueError("Expected hash-bound artifact {path, sha256}")
    path = _local(root, entry["path"])
    if file_hash(path, fresh=fresh) != entry["sha256"]:
        raise ValueError("Qualification artifact hash mismatch")
    return path


def _resources(proof: Any) -> bool:
    return (isinstance(proof, dict) and proof.get("memory_limit_mb") == 2048
        and type(proof.get("logical_cpu_limit")) is int
        and 1 <= proof["logical_cpu_limit"] <= 2
        and proof.get("priority") == "below_normal"
        and proof.get("kill_tree_on_close") is True)


def code_binding(*, fresh: bool = False) -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {name: file_hash(directory / name, fresh=fresh) for name in CODE_FILES}


def _common(root: Path, binding: dict[str, Any], *, fresh: bool) -> dict[str, bool]:
    ready = dict.fromkeys(METHODS, False)
    path = root / "cumesh-common-worker-qualification.json"
    if not path.is_file():
        return ready
    proof = _json(path)
    if proof.get("schema_version") != 1 or proof.get("qualification_binding") != binding:
        return ready
    for method in METHODS:
        try:
            entry = proof.get("methods", {}).get(method)
            if not isinstance(entry, dict) or set(entry) != {
                    "result", "report", "resource_controls", "source", "candidate"}:
                continue
            paths = {name: _artifact(root, ref, fresh=fresh) for name, ref in entry.items()}
            result, report = _json(paths["result"]), _json(paths["report"])
            gate = result.get("gate", {})
            checks = gate.get("checks", [])
            passed = (gate.get("passed") is True and gate.get("gate_version") == binding["gate_version"]
                and gate.get("source_sha256") == entry["source"]["sha256"]
                and gate.get("candidate_sha256") == gate.get("output_sha256") == entry["candidate"]["sha256"]
                and report.get("gate") == gate and report.get("source_sha256") == gate["source_sha256"]
                and report.get("recipe", {}).get("method") == method
                and report.get("gpu_qualification_binding") == binding
                and report.get("raw_source_unchanged") is True
                and REQUIRED_CHECKS <= {c.get("name") for c in checks}
                and all(c.get("passed") is True and c.get("evaluated") is True for c in checks)
                and _resources(_json(paths["resource_controls"]))
                and any(s.get("operation") == "cumesh_common_candidate"
                        and s.get("method") == method and s.get("native_dispatch_completed") is True
                        for s in report.get("steps", [])))
            ready[method] = bool(passed)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            ready[method] = False
    return ready


def discover(package_paths: list[str], *, gate_version: str, fresh: bool = False) -> dict[str, Any]:
    """Ordered worker roots only. A shadowing unqualified package fails closed."""
    base: dict[str, Any] = {"package_present": False, "metadata_available": False,
        "pinned_native_verified": False, "cuda_verified": False, "cuda_qualified": False,
        "common_gate_qualified": False, "production_ready": False, "runtime_qualified": False,
        "runnable": False, "experimental_runnable": False, "source_commit": SOURCE_COMMIT,
        "methods": list(METHODS), "method_production_ready": dict.fromkeys(METHODS, False),
        "reason_code": "CUMESH_PACKAGE_UNAVAILABLE", "reason": "No package in configured worker paths",
        "cpu_fallback": False, "runtime_memory_limit_mb": 2048, "requires_gpu_lease": True}
    roots = [Path(p).resolve() for p in package_paths if p]
    root = next((p for p in roots if (p / "cumesh").is_dir() or (p / "cumesh.py").is_file()), None)
    if root is None:
        return base
    base.update(package_present=True, package_root=str(root), metadata_available=True)
    try:
        manifest = _json(root / "cumesh-install-manifest.json")
        native = manifest.get("native_file_sha256")
        if manifest.get("source_commit") != SOURCE_COMMIT or not isinstance(native, dict) or not native:
            raise ValueError("Missing pinned commit and native binary hashes")
        actual_native = {p.relative_to(root).as_posix() for p in (root / "cumesh").rglob("*.pyd")}
        if {Path(name).as_posix() for name in native} != actual_native:
            raise ValueError("Native binary inventory differs from pinned manifest")
        for name, digest in native.items():
            if file_hash(_local(root, name), fresh=fresh) != digest:
                raise ValueError("Pinned native binary hash mismatch")
        base["pinned_native_verified"] = True
        if manifest.get("cuda_verified") is not True:
            raise ValueError("Pinned build exists; actual CUDA qualification not completed")
        artifacts = {name: _artifact(root, {"path": manifest.get(f"cuda_probe_{name}path"),
                    "sha256": manifest.get(f"cuda_probe_{name}sha256")}, fresh=fresh)
                    for name in ("", "result_", "adapter_")}
        probe, guard = _json(artifacts[""]), _json(artifacts["result_"])
        code = code_binding(fresh=fresh)
        expected = {"source_commit": SOURCE_COMMIT, "native_file_sha256": native,
            "adapter_sha256": code["cumesh_repair.py"], "memory_limit_mb": 2048, "logical_cpu_limit": 2}
        if (file_hash(artifacts["adapter_"], fresh=fresh) != code["cumesh_repair.py"]
                or probe.get("qualification_binding") != expected
                or probe.get("source_commit") != SOURCE_COMMIT or probe.get("cuda_verified") is not True
                or not all(isinstance(probe.get("checks", {}).get(k), dict)
                    and probe["checks"][k].get("method") == method
                    and probe["checks"][k].get("source_commit") == SOURCE_COMMIT
                    and probe["checks"][k].get("candidate_only") is True for k, method in (
                        ("duplicate_cleanup", "cumesh_cleanup"), ("loop_diagnostic", "cumesh_diagnose"),
                        ("selected_planar_hole", "cumesh_fill_selected_holes")))
                or guard.get("mode") != "probe" or guard.get("succeeded") is not True
                or guard.get("source_commit") != SOURCE_COMMIT or guard.get("memory_limit_mb") != 2048
                or guard.get("logical_cpu_limit") != 2
                or not _resources(guard.get("resource_controls"))):
            raise ValueError("CUDA probe, code binding or bounded execution proof is invalid")
        binding = {"source_commit": SOURCE_COMMIT, "native_file_sha256": native,
            "code_sha256": code, "cuda_probe_sha256": manifest["cuda_probe_sha256"],
            "cuda_probe_result_sha256": manifest["cuda_probe_result_sha256"],
            "gate_version": gate_version, "memory_limit_mb": 2048, "logical_cpu_limit": 2,
            "methods": list(METHODS)}
        base.update(cuda_verified=True, cuda_qualified=True, runtime_qualified=True,
            runnable=True, experimental_runnable=True, qualification_binding=binding,
            reason_code="CUMESH_COMMON_WORKER_UNQUALIFIED",
            reason="CUDA probe qualified; explicit experimental candidates still require all common gates and review")
        common = _common(root, binding, fresh=fresh)
        base.update(method_production_ready=common, common_gate_qualified=all(common.values()),
                    production_ready=all(common.values()))
        if base["production_ready"]:
            base.update(reason_code=None, reason="All three methods have hash-bound common-worker qualification")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        base.update(reason_code="CUMESH_QUALIFICATION_INVALID", reason=str(exc))
    return base
