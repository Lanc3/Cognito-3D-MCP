"""Metadata and mocked dispatch tests only; no native CUDA execution."""
import json
import shutil
import sys

import pytest

from codex_3d_mcp.hunyuan_mv import repair_cumesh_qualification as proof
from codex_3d_mcp.hunyuan_mv.repair_contract import REPAIR_GATE_VERSION, validate_recipe


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def reference(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": proof.file_hash(path, fresh=True)}


def resources():
    return {"memory_limit_mb": 2048, "logical_cpu_limit": 2,
            "priority": "below_normal", "kill_tree_on_close": True}


def cuda_fixture(root):
    native = root / "cumesh" / "_C.pyd"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"test-only fake native bytes; no executable code")
    hashes = {"cumesh/_C.pyd": proof.file_hash(native)}
    directory = root / "cumesh-qualification" / "probe"
    directory.mkdir(parents=True)
    adapter = directory / "adapter-source.py"
    shutil.copyfile(proof.Path(proof.__file__).with_name("cumesh_repair.py"), adapter)
    probe = directory / "cuda-probe.json"
    write(probe, {"cuda_verified": True, "source_commit": proof.SOURCE_COMMIT,
        "qualification_binding": {"source_commit": proof.SOURCE_COMMIT,
            "native_file_sha256": hashes, "adapter_sha256": proof.file_hash(adapter),
            "memory_limit_mb": 2048, "logical_cpu_limit": 2},
        "checks": {name: {"method": method, "source_commit": proof.SOURCE_COMMIT, "candidate_only": True}
            for name, method in (("duplicate_cleanup", "cumesh_cleanup"),
                ("loop_diagnostic", "cumesh_diagnose"), ("selected_planar_hole", "cumesh_fill_selected_holes"))}})
    guard = directory / "probe-result.json"
    write(guard, {"mode": "probe", "succeeded": True, "source_commit": proof.SOURCE_COMMIT,
        "memory_limit_mb": 2048, "logical_cpu_limit": 2, "resource_controls": resources()})
    manifest = {"source_commit": proof.SOURCE_COMMIT, "native_file_sha256": hashes, "cuda_verified": True}
    for prefix, path in (("", probe), ("result_", guard), ("adapter_", adapter)):
        ref = reference(root, path)
        manifest[f"cuda_probe_{prefix}path"] = ref["path"]
        manifest[f"cuda_probe_{prefix}sha256"] = ref["sha256"]
    write(root / "cumesh-install-manifest.json", manifest)
    return manifest


def discover(root):
    return proof.discover([str(root)], gate_version=REPAIR_GATE_VERSION, fresh=True)


def test_cuda_boolean_without_evidence_is_not_qualification(tmp_path):
    cuda_fixture(tmp_path)
    write(tmp_path / "cumesh-install-manifest.json", {"cuda_verified": True})
    assert not discover(tmp_path)["cuda_qualified"]


def test_cuda_fixture_is_experimentally_runnable_without_circular_common_proof(tmp_path):
    before = "torch" in sys.modules
    cuda_fixture(tmp_path)
    data = discover(tmp_path)
    assert data["cuda_qualified"] and data["experimental_runnable"]
    assert not data["production_ready"]
    assert data["qualification_binding"]["code_sha256"] == proof.code_binding()
    assert ("torch" in sys.modules) == before


@pytest.mark.parametrize("tamper", ["native", "adapter", "resource", "extra_binary", "path_escape"])
def test_hash_cap_and_path_changes_fail_closed(tmp_path, tamper):
    manifest = cuda_fixture(tmp_path)
    if tamper == "native":
        (tmp_path / "cumesh" / "_C.pyd").write_bytes(b"changed")
    elif tamper == "adapter":
        (tmp_path / manifest["cuda_probe_adapter_path"]).write_text("changed", encoding="utf-8")
    elif tamper == "resource":
        path = tmp_path / manifest["cuda_probe_result_path"]
        data = json.loads(path.read_text())
        data["resource_controls"]["memory_limit_mb"] = 3072
        write(path, data)
        manifest["cuda_probe_result_sha256"] = proof.file_hash(path, fresh=True)
    elif tamper == "extra_binary":
        (tmp_path / "cumesh" / "unlisted.pyd").write_bytes(b"unlisted")
    else:
        manifest["cuda_probe_path"] = "../outside.json"
    write(tmp_path / "cumesh-install-manifest.json", manifest)
    assert not discover(tmp_path)["cuda_qualified"]


def common_fixture(root):
    cuda_fixture(root)
    binding = discover(root)["qualification_binding"]
    records = {}
    for method in proof.METHODS:
        directory = root / "common-proof" / method
        directory.mkdir(parents=True)
        source, candidate = directory / "source.glb", directory / "candidate.glb"
        source.write_bytes(b"metadata-only source test bytes")
        candidate.write_bytes(b"metadata-only candidate test bytes")
        gate = {"passed": True, "gate_version": REPAIR_GATE_VERSION,
            "source_sha256": proof.file_hash(source), "candidate_sha256": proof.file_hash(candidate),
            "output_sha256": proof.file_hash(candidate),
            "checks": [{"name": name, "evaluated": True, "passed": True} for name in proof.REQUIRED_CHECKS]}
        report = directory / "repair-report.json"
        write(report, {"recipe": {"method": method}, "source_sha256": gate["source_sha256"],
            "gate": gate, "raw_source_unchanged": True, "gpu_qualification_binding": binding,
            "steps": [{"operation": "cumesh_common_candidate", "method": method, "native_dispatch_completed": True}]})
        result = directory / "result.json"
        write(result, {"gate": gate, "repair_report": "old-machine-path", "output": "old-machine-path"})
        controls = directory / "resource-controls.json"
        write(controls, resources())
        records[method] = {name: reference(root, path) for name, path in (
            ("source", source), ("candidate", candidate), ("report", report),
            ("result", result), ("resource_controls", controls))}
    document = {"schema_version": 1, "qualification_binding": binding, "methods": records}
    write(root / "cumesh-common-worker-qualification.json", document)
    return document


def test_common_gate_qualification_is_artifact_bound_and_portable(tmp_path):
    original = tmp_path / "original"
    common_fixture(original)
    assert discover(original)["production_ready"]
    copy = tmp_path / "deployed"
    shutil.copytree(original, copy)
    assert discover(copy)["production_ready"]
    (copy / "common-proof" / "cumesh_cleanup" / "candidate.glb").write_bytes(b"tampered")
    data = discover(copy)
    assert data["cuda_qualified"] and not data["production_ready"]
    assert data["method_production_ready"]["cumesh_diagnose"]
    assert not data["method_production_ready"]["cumesh_cleanup"]


def test_unknown_shadowing_package_does_not_use_later_qualified_install(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    (first / "cumesh").mkdir(parents=True)
    cuda_fixture(second)
    data = proof.discover([str(first), str(second)], gate_version=REPAIR_GATE_VERSION)
    assert not data["cuda_qualified"]


def test_gpu_contract_requires_authoring_and_does_not_accept_native_indices():
    roi = {"min": [-2, -2, -2], "max": [2, 2, 2]}
    assert validate_recipe({"method": "cumesh_diagnose"})["policy"]["roi"] is None
    with pytest.raises(ValueError):
        validate_recipe({"method": "cumesh_cleanup", "policy": {"roi": roi}})
    with pytest.raises(ValueError):
        validate_recipe({"method": "cumesh_fill_selected_holes", "parameters": {
            "selected_loop_edges": [[[0, 1], [1, 2], [2, 0]]]}, "policy": {"roi": roi}})
    cleanup = validate_recipe({"method": "cumesh_cleanup", "parameters": {
        "remove_degenerate_faces": True}, "policy": {"roi": roi}})
    assert cleanup["parameters"]["max_removed_fraction"] == 0.001


def test_gpu_unavailable_and_resource_are_typed_operational_failures():
    from codex_3d_mcp.hunyuan_mv import cumesh_repair as adapter, repair_worker as worker
    assert worker.classify_failure(adapter.CuMeshUnavailable("missing"))["failure_kind"] == "runtime_unavailable"
    assert worker.classify_failure(adapter.CuMeshResourceError("reserve"))["failure_kind"] == "resource"
