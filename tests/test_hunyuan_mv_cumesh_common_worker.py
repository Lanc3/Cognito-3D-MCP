"""Common-worker integration with a simulated adapter, never native CUDA."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
trimesh = pytest.importorskip("trimesh")

from codex_3d_mcp.hunyuan_mv import cumesh_repair as adapter
from codex_3d_mcp.hunyuan_mv import repair_cumesh_qualification as qualification
from codex_3d_mcp.hunyuan_mv import repair_geometry as geometry
from codex_3d_mcp.hunyuan_mv import repair_worker as worker
from codex_3d_mcp.hunyuan_mv.repair_contract import validate_recipe


def tetra():
    return (np.array([[10, 20, 30], [11, 20, 30], [10, 21, 30], [10, 20, 31]], dtype=float),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64))


def ready(monkeypatch):
    data = qualification.discover([], gate_version="shape-repair-v1")
    data.update(cuda_qualified=True, package_present=True,
                qualification_binding={"test_only_mock": True})
    monkeypatch.setattr(qualification, "discover", lambda *a, **k: dict(data))


def native_result(v, f, recipe, cv=None, cf=None):
    cv, cf = v.copy() if cv is None else cv, f.copy() if cf is None else cf
    return cv, cf, {"method": recipe["method"], "input_geometry_sha256": adapter.geometry_sha256(v, f),
                   "output_geometry_sha256": adapter.geometry_sha256(cv, cf)}


def test_unqualified_gpu_never_dispatches(monkeypatch):
    monkeypatch.setattr(qualification, "discover", lambda *a, **k: {
        "cuda_qualified": False, "reason": "missing retained CUDA probe"})
    monkeypatch.setattr(adapter, "repair", lambda *a, **k: pytest.fail("must not dispatch"))
    v, f = tetra()
    diagnostics, info = geometry.inspect(v, f)
    with pytest.raises(adapter.CuMeshUnavailable):
        worker._cumesh_candidate(diagnostics, info, validate_recipe({"method": "cumesh_diagnose"}), {"steps": []})


def test_gpu_diagnose_preserves_original_glb_bytes_and_runs_common_checks(tmp_path, monkeypatch):
    ready(monkeypatch)
    monkeypatch.setattr(adapter, "repair", native_result)
    monkeypatch.setattr(geometry, "self_intersections", lambda *a: {
        "evaluated": True, "passed": True, "backend": "test-only intersection stub"})
    v, f = tetra()
    source = tmp_path / "source.glb"
    source.write_bytes(trimesh.Trimesh(v, f, process=False).export(file_type="glb"))
    before = source.read_bytes()
    result = worker.run({"source_path": str(source), "source_sha256": worker._hash(source),
        "recipe": {"method": "cumesh_diagnose"}, "output_dir": str(tmp_path / "attempt")})
    assert result["gate"]["passed"]
    assert source.read_bytes() == before == worker.Path(result["output"]).read_bytes()
    assert qualification.REQUIRED_CHECKS <= {c["name"] for c in result["gate"]["checks"]}


def test_source_loop_ids_map_to_recomputed_canonical_edges_after_reordering(monkeypatch):
    ready(monkeypatch)
    v, f = tetra()
    order = np.array([3, 1, 0, 2])
    v, f = v[order], np.argsort(order)[f[:-1]]
    diagnostics, info = geometry.inspect(v, f)
    loop = diagnostics["loops"][0]
    recipe = validate_recipe({"method": "cumesh_fill_selected_holes", "parameters": {
        "loop_ids": [loop["id"]], "max_diameter": 2, "max_edges": 3,
        "max_patch_area": 1, "max_hole_perimeter": 5},
        "policy": {"roi": {"min": [9, 19, 29], "max": [12, 22, 32]}}})
    called = []

    def fill(cv, cf, native):
        called.append(native)
        edges = native["parameters"]["selected_loop_edges"][0]
        actual = {tuple(cv[index]) for edge in edges for index in edge}
        assert actual == set(map(tuple, loop["vertices"]))
        assert native["parameters"]["input_geometry_sha256"] == adapter.geometry_sha256(cv, cf)
        patch = [edge[0] for edge in edges][::-1]
        return native_result(cv, cf, native, cf=np.vstack([cf, patch]))

    monkeypatch.setattr(adapter, "repair", fill)
    cv, cf = worker._cumesh_candidate(diagnostics, info, recipe, {"steps": []})
    assert len(called) == 1 and len(cf) == len(f) + 1
    candidate, _ = geometry.inspect(cv, cf)
    assert candidate["topology"]["inconsistent_edge_directions"] == 0
    recipe["parameters"]["loop_ids"] = ["loop:" + "a" * 64]
    with pytest.raises(ValueError, match="Unknown source loop"):
        worker._cumesh_candidate(diagnostics, info, recipe, {"steps": []})
    assert len(called) == 1


def test_gpu_cleanup_outside_roi_still_fails_common_preservation(tmp_path, monkeypatch):
    ready(monkeypatch)
    monkeypatch.setattr(geometry, "self_intersections", lambda *a: {"evaluated": True, "passed": True})
    v, f = tetra()
    v = np.vstack([v + [i * 3, 0, 0] for i in range(10)])
    f = np.vstack([f + i * 4 for i in range(10)])
    f = np.vstack([f, f[:1]])

    def cleanup(cv, cf, recipe):
        _, first = np.unique(np.sort(cf, axis=1), axis=0, return_index=True)
        return native_result(cv, cf, recipe, cf=cf[np.sort(first)])

    monkeypatch.setattr(adapter, "repair", cleanup)
    source = tmp_path / "source.glb"
    source.write_bytes(trimesh.Trimesh(v, f, process=False).export(file_type="glb"))
    result = worker.run({"source_path": str(source), "source_sha256": worker._hash(source),
        "recipe": {"method": "cumesh_cleanup", "parameters": {
            "remove_duplicate_faces": True, "max_removed_fraction": 0.05},
            "policy": {"roi": {"min": [36, 19, 29], "max": [39, 22, 32]}}},
        "output_dir": str(tmp_path / "attempt")})
    assert result["output"]
    assert not result["gate"]["passed"]
    checks = {c["name"]: c for c in result["gate"]["checks"]}
    assert checks["export_closed_edges"]["passed"]
    assert not checks["outside_roi_preserved"]["passed"]
    assert checks["raw_source_unchanged"]["passed"]


def test_gpu_cleanup_cannot_remove_a_nonzero_face_even_within_removal_budget(monkeypatch):
    ready(monkeypatch)
    v, f = tetra()
    v = np.vstack([v + [i * 3, 0, 0] for i in range(10)])
    f = np.vstack([f + i * 4 for i in range(10)])
    diagnostics, info = geometry.inspect(v, f)
    monkeypatch.setattr(adapter, "repair", lambda cv, cf, recipe: native_result(cv, cf, recipe, cf=cf[:-1]))
    recipe = validate_recipe({"method": "cumesh_cleanup", "parameters": {
        "remove_degenerate_faces": True, "max_removed_fraction": 0.05},
        "policy": {"roi": {"min": [0, 0, 0], "max": [50, 50, 50]}}})
    with pytest.raises(ValueError, match="beyond the exact"):
        worker._cumesh_candidate(diagnostics, info, recipe, {"steps": []})


@pytest.mark.parametrize("wrong_faces", [1, 3])
def test_orient_only_new_fan_faces_uses_unchanged_source_boundary(wrong_faces):
    v, f = tetra()
    _, info = geometry.inspect(v, f[:-1])
    v, f = info["canonical_vertices"], info["canonical_faces"]
    boundary = info["boundary_edges"]
    center = len(v)
    cv = np.vstack([v, v[np.unique(boundary)].mean(axis=0)])
    patch = np.array([[int(b), int(a), center] for a, b in boundary], dtype=np.int64)
    patch[:wrong_faces] = patch[:wrong_faces, ::-1]
    before_patch, before_vertices = patch.copy(), cv.copy()
    before, _ = geometry.inspect(cv, np.vstack([f, patch]))
    assert before["topology"]["inconsistent_edge_directions"] > 0
    oriented, report = worker._orient_new_patch_faces(patch, boundary)
    cf = np.vstack([f, oriented])
    after, _ = geometry.inspect(cv, cf)
    assert after["topology"]["inconsistent_edge_directions"] == 0
    assert after["topology"]["all_edges_have_two_faces"]
    assert np.array_equal(cf[:len(f)], f)
    assert np.array_equal(cv, before_vertices)
    assert np.array_equal(patch, before_patch)
    assert report["flipped_new_faces"] == wrong_faces
    assert report["existing_faces_changed"] == report["vertices_changed"] == 0
    assert not report["blanket_normal_fix"]


def test_patch_orientation_rejects_contradictory_source_anchors():
    v, f = tetra()
    _, info = geometry.inspect(v, f[:-1])
    boundary = info["boundary_edges"].copy()
    patch = np.array([[int(a), int(b), 4] for a, b in boundary], dtype=np.int64)
    boundary[0] = boundary[0, ::-1]
    with pytest.raises(ValueError, match="Contradictory patch orientation"):
        worker._orient_new_patch_faces(patch, boundary)


def test_patch_orientation_rejects_unanchored_extra_closed_component():
    v, f = tetra()
    _, info = geometry.inspect(v, f[:-1])
    boundary = info["boundary_edges"]
    patch = np.array([[int(a), int(b), 4] for a, b in boundary], dtype=np.int64)
    patch = np.vstack([patch, f + 5])
    with pytest.raises(ValueError, match="without a source boundary anchor"):
        worker._orient_new_patch_faces(patch, boundary)


def test_patch_orientation_refuses_nonmanifold_new_faces():
    v, f = tetra()
    _, info = geometry.inspect(v, f[:-1])
    boundary = info["boundary_edges"]
    patch = np.array([[int(a), int(b), 4] for a, b in boundary], dtype=np.int64)
    patch = np.vstack([patch, patch[:1]])
    with pytest.raises(ValueError, match="exactly one new patch face|nonmanifold"):
        worker._orient_new_patch_faces(patch, boundary)
