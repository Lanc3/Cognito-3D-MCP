"""Small synthetic fixtures only; run in the same bounded CPU worker environment."""
import json
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
trimesh = pytest.importorskip("trimesh")

from codex_3d_mcp.hunyuan_mv import repair_geometry as geometry
from codex_3d_mcp.hunyuan_mv import repair_worker as worker
from codex_3d_mcp.hunyuan_mv.repair_contract import validate_recipe


def tetra():
    return (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
            np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64))


def roi():
    return {"min": [-2, -2, -2], "max": [3, 3, 3]}


def test_exact_vertex_links_catch_two_closed_shells_touching_at_one_vertex():
    vertices, faces = tetra()
    vertices = np.vstack([vertices, -vertices])
    faces = np.vstack([faces, faces[:, ::-1] + 4])
    report, _ = geometry.inspect(vertices, faces)
    assert report["topology"]["all_edges_have_two_faces"]
    assert report["topology"]["component_count"] == 2
    assert report["topology"]["nonmanifold_vertices"] == 1


def test_component_and_loop_ids_survive_vertex_face_reordering():
    vertices, faces = tetra()
    faces = faces[:-1]
    original, _ = geometry.inspect(vertices, faces)
    order = np.array([2, 0, 3, 1])
    inverse = np.argsort(order)
    permuted, _ = geometry.inspect(vertices[order], inverse[faces[::-1]])
    assert original["components"][0]["id"] == permuted["components"][0]["id"]
    assert original["loops"][0]["id"] == permuted["loops"][0]["id"]


def test_selected_triangle_patch_closes_exactly_one_missing_face_without_displacement():
    vertices, faces = tetra()
    source, info = geometry.inspect(vertices, faces[:-1])
    recipe = validate_recipe({"method": "patch_selected_holes", "parameters": {
        "loop_ids": [source["loops"][0]["id"]], "max_diameter": 2,
        "max_patch_area": 1, "min_triangle_area": 1e-12}, "policy": {"roi": roi()}})
    steps = []
    cv, cf = worker._patch(vertices, faces[:-1], source, info, recipe["parameters"], roi(), steps)
    output, _ = geometry.inspect(cv, cf)
    assert output["topology"]["all_edges_have_two_faces"]
    assert output["topology"]["inconsistent_edge_directions"] == 0
    assert geometry.geometry_digest(vertices, faces) == geometry.geometry_digest(cv, cf)
    assert steps[-1]["added_faces"] == 1
    assert steps[-1]["existing_vertex_displacement"] == 0


def test_triangle_patch_refuses_to_recreate_near_degenerate_face():
    vertices, faces = tetra()
    vertices[2, 1] = 1e-15
    source, info = geometry.inspect(vertices, faces[1:])
    recipe = validate_recipe({"method": "patch_selected_holes", "parameters": {
        "loop_ids": [source["loops"][0]["id"]], "max_diameter": 2,
        "max_patch_area": 1, "min_triangle_area": 1e-12}, "policy": {"roi": roi()}})
    with pytest.raises(ValueError, match="near-degenerate"):
        worker._patch(vertices, faces[1:], source, info, recipe["parameters"], roi(), [])


def test_cleanup_removes_exact_degeneracy_but_preserves_small_nonzero_face():
    vertices, faces = tetra()
    all_v = np.vstack([vertices + [i * 2, 0, 0] for i in range(10)])
    all_f = np.vstack([faces + i * 4 for i in range(10)])
    extra = np.array([[30, 0, 0], [31, 0, 0], [30, 1e-14, 0]])
    all_v = np.vstack([all_v, extra])
    all_f = np.vstack([all_f, [40, 41, 42], [0, 0, 1]])
    region = {"min": [-1, -1, -1], "max": [32, 2, 2]}
    recipe = validate_recipe({"method": "conservative_cleanup", "parameters": {
        "max_removed_fraction": 0.05}, "policy": {"roi": region}})
    steps = []
    cv, cf = worker._cleanup(all_v, all_f, recipe["parameters"], region, steps)
    assert len(cf) == len(all_f) - 1
    assert np.sum(geometry.areas(cv, cf) <= 1e-12) == 1
    assert next(s for s in steps if s["operation"] == "remove_exact_zero_area_faces_in_roi")["removed_faces"] == 1


def test_preservation_rejects_changed_outside_faces_and_protected_component():
    vertices, faces = tetra()
    source, info = geometry.inspect(vertices, faces)
    moved = vertices.copy()
    moved[1] += [0.1, 0, 0]
    candidate, _ = geometry.inspect(moved, faces)
    policy = validate_recipe({"method": "analyze", "policy": {
        "roi": {"min": [-0.1, -0.1, -0.1], "max": [0.1, 0.1, 0.1]},
        "protected_component_ids": [source["components"][0]["id"]]}})["policy"]
    result = geometry.preservation(vertices, faces, moved, faces, info, candidate, policy)
    assert not result["outside_roi"]["passed"]
    assert not result["protected_components"]["passed"]


def test_foundation_intervals_detect_closed_empty_cavity():
    outer = trimesh.creation.box(extents=[1, 1, 1])
    outer.apply_translation([0.5, 0.5, 0.5])
    inner = trimesh.creation.box(extents=[0.8, 0.8, 0.8])
    inner.apply_translation([0.5, 0.5, 0.5])
    vertices = np.vstack([outer.vertices, inner.vertices])
    faces = np.vstack([outer.faces, inner.faces[:, ::-1] + len(outer.vertices)])
    diagnostics, _ = geometry.inspect(vertices, faces)
    assert diagnostics["topology"]["all_edges_have_two_faces"]
    policy = {"roi": {"min": [0.2, 0.2, 0.2], "max": [0.8, 0.8, 0.8]},
              "up_axis": 1, "grid_size": 3, "surface_tolerance": 1e-6}
    assert geometry.foundation_intervals(outer.vertices, outer.faces, policy)["passed"]
    cavity = geometry.foundation_intervals(vertices, faces, policy)
    assert cavity["failed_samples"] == 9
    assert not cavity["whole_volume_proven"]


def test_missing_intersection_backend_fails_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymeshlab", None)
    result = geometry.self_intersections(*tetra())
    assert result["evaluated"] is False
    assert result["passed"] is False
    assert result["failure_kind"] == "runtime_unavailable"
    assert result["code"] == "REPAIR_BACKEND_UNAVAILABLE"


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_authored_footprint_builds_oriented_closed_solid_for_each_up_axis(axis):
    primitive = {"kind": "footprint", "up_axis": axis, "rings": [
        {"height": h, "points": [[0, 0], [1, 0], [1, 1], [0, 1]]} for h in [0, 1]]}
    vertices, faces = worker._primitive_mesh(primitive)
    diagnostics, _ = geometry.inspect(vertices, faces)
    assert diagnostics["topology"]["all_edges_have_two_faces"]
    assert diagnostics["topology"]["inconsistent_edge_directions"] == 0
    assert trimesh.Trimesh(vertices, faces, process=False).volume == pytest.approx(1)


def test_import_is_hash_bound_immutable_and_uses_source_preservation_gate(tmp_path, monkeypatch):
    # Isolate import/hash/preservation behavior; real PyMeshLab is qualified by
    # the coordinating agent's bounded integration run, not this test double.
    monkeypatch.setattr(geometry, "self_intersections", lambda *_: {
        "evaluated": True, "passed": True, "backend": "test double", "selected_faces": 0})
    original = trimesh.Trimesh(*tetra(), process=False)
    source, imported = tmp_path / "raw.glb", tmp_path / "import.glb"
    original.export(source)
    modified = original.copy()
    modified.vertices[1] += [0.2, 0, 0]
    modified.export(imported)
    request = {"source_path": str(source), "source_sha256": worker._hash(source),
        "import_path": str(imported), "import_sha256": worker._hash(imported),
        "output_dir": str(tmp_path / "attempt"), "recipe": {"method": "import_candidate",
            "policy": {"roi": {"min": [-0.1, -0.1, -0.1], "max": [0.1, 0.1, 0.1]}}}}
    result = worker.run(request)
    assert not result["gate"]["passed"]
    assert result["gate"]["candidate_sha256"] == worker._hash(imported)
    assert worker._hash(source) == request["source_sha256"]
    diagnostics = json.loads((tmp_path / "attempt" / "diagnostics.json").read_text())
    assert diagnostics["source"]["components"]
    assert diagnostics["candidate"]["topology"]["all_edges_have_two_faces"]
    with pytest.raises(ValueError, match="immutable"):
        worker.run(request)


def test_source_hash_mismatch_produces_no_candidate(tmp_path):
    source = tmp_path / "raw.glb"
    trimesh.Trimesh(*tetra(), process=False).export(source)
    with pytest.raises(ValueError, match="Source hash mismatch"):
        worker.run({"source_path": str(source), "source_sha256": "0" * 64,
                    "output_dir": str(tmp_path / "attempt"), "recipe": {"method": "analyze"}})
    assert not (tmp_path / "attempt").exists()


@pytest.mark.parametrize("exception,kind,code", [
    (ModuleNotFoundError("dependency absent"), "runtime_unavailable", "REPAIR_BACKEND_UNAVAILABLE"),
    (MemoryError("allocation failed"), "resource", "REPAIR_RESOURCE_EXHAUSTED"),
    (RuntimeError("backend crashed"), "runtime_failure", "REPAIR_RUNTIME_FAILURE"),
    # Message spelling must never turn a real geometry rejection into an operational retry.
    (ValueError("No module named fake_backend"), "geometry_rejected", "REPAIR_INVALID_GEOMETRY"),
])
def test_worker_exception_categories_are_typed_and_persisted(tmp_path, monkeypatch, exception, kind, code):
    source = tmp_path / "raw.glb"
    trimesh.Trimesh(*tetra(), process=False).export(source)

    def fail(_path):
        raise exception

    monkeypatch.setattr(worker, "_load", fail)
    result = worker.run({"source_path": str(source), "source_sha256": worker._hash(source),
                         "output_dir": str(tmp_path / "attempt"), "recipe": {"method": "analyze"}})
    report = json.loads((tmp_path / "attempt" / "repair-report.json").read_text())
    assert result["failure_kind"] == result["gate"]["failure_kind"] == report["failure_kind"] == kind
    assert result["failure_code"] == report["error"]["code"] == code
    assert report["error"]["failure_kind"] == kind
    assert not result["gate"]["passed"]
    assert result["output"] is None


def test_non_evaluated_required_backend_is_retryable_even_with_other_quality_failures(tmp_path, monkeypatch):
    vertices, faces = tetra()
    source = tmp_path / "open.glb"
    trimesh.Trimesh(vertices, faces[:-1], process=False).export(source)
    monkeypatch.setattr(geometry, "self_intersections", lambda *_: {
        "evaluated": False, "passed": False, "backend": "pymeshlab", "selected_faces": None,
        "failure_kind": "runtime_unavailable", "code": "REPAIR_BACKEND_UNAVAILABLE"})
    result = worker.run({"source_path": str(source), "source_sha256": worker._hash(source),
                         "output_dir": str(tmp_path / "attempt"), "recipe": {"method": "analyze"}})
    checks = {check["name"]: check for check in result["gate"]["checks"]}
    assert checks["export_closed_edges"]["failure_kind"] == "geometry_rejected"
    assert checks["export_self_intersections"]["failure_kind"] == "runtime_unavailable"
    assert result["failure_kind"] == "runtime_unavailable"
    assert result["gate"]["candidate_sha256"] == worker._hash(source)


def test_completed_geometry_rejection_is_quality_failure_not_operational(tmp_path, monkeypatch):
    vertices, faces = tetra()
    source = tmp_path / "open.glb"
    trimesh.Trimesh(vertices, faces[:-1], process=False).export(source)
    monkeypatch.setattr(geometry, "self_intersections", lambda *_: {
        "evaluated": True, "passed": True, "backend": "test double", "selected_faces": 0})
    result = worker.run({"source_path": str(source), "source_sha256": worker._hash(source),
                         "output_dir": str(tmp_path / "attempt"), "recipe": {"method": "analyze"}})
    assert result["failure_kind"] == "geometry_rejected"
    assert result["failure_code"] == "REPAIR_QUALITY_GATES_FAILED"


def test_required_boolean_precheck_retains_operational_failure_type():
    check = {"evaluated": False, "passed": False, "failure_kind": "resource",
             "code": "REPAIR_RESOURCE_EXHAUSTED"}
    with pytest.raises(worker.RepairCheckError) as failure:
        worker._require_check(check, "required source test unavailable")
    assert worker.classify_failure(failure.value) == {
        "failure_kind": "resource", "code": "REPAIR_RESOURCE_EXHAUSTED"}


def restoration_fixture(compatible):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int64)
    if compatible:
        candidate_v = np.vstack([vertices, [0.5, 0.5, 1.1]])
        candidate_f = np.vstack([[[0, 3, 1], [1, 3, 2]],
                                [[4, 5, 8], [5, 6, 8], [6, 7, 8], [7, 4, 8]], faces[4:]])
    else:
        candidate_v, candidate_f = vertices.copy(), faces.copy()
        candidate_v[4:, :2] = candidate_v[4:, :2] * 0.8 + 0.1
        candidate_f[:2] = [[0, 3, 1], [1, 3, 2]]
    region = {"min": [-0.1, -0.1, 0.9], "max": [1.1, 1.1, 1.2]}
    return vertices, faces, candidate_v, candidate_f, region


def test_restore_outside_faces_keeps_new_inside_surface_and_original_diagonal():
    sv, sf, cv, cf, region = restoration_fixture(True)
    steps = []
    vertices, faces = worker._restore_outside_faces(sv, sf, cv, cf, region, steps)
    source_outside = ~geometry.inside(sv, region)[sf].all(axis=1)
    restored_outside = ~geometry.inside(vertices, region)[faces].all(axis=1)
    assert geometry.geometry_digest(sv, sf[source_outside]) == geometry.geometry_digest(vertices, faces[restored_outside])
    assert len(faces) == 14
    assert vertices[:, 2].max() == 1.1
    diagnostics, _ = geometry.inspect(vertices, faces)
    assert diagnostics["topology"]["all_edges_have_two_faces"]
    assert diagnostics["topology"]["inconsistent_edge_directions"] == 0
    assert steps[-1]["source_outside_faces_retained"] == 10
    assert steps[-1]["candidate_inside_faces_retained"] == 4
    assert steps[-1]["restored_outside_sha256"] == steps[-1]["source_outside_sha256"]
    assert steps[-1]["restored_inside_sha256"] == steps[-1]["candidate_inside_sha256"]
    assert steps[-1]["maximum_coordinate_displacement"] == 0
    assert steps[-1]["gate_waiver"] is False


@pytest.mark.parametrize("compatible", [True, False])
def test_restored_export_still_uses_all_gates_and_rejects_incompatible_boundary(tmp_path, compatible):
    pytest.importorskip("pymeshlab")
    sv, sf, cv, cf, region = restoration_fixture(compatible)
    rv, rf = worker._restore_outside_faces(sv, sf, cv, cf, region, [])
    source, imported = tmp_path / "source.glb", tmp_path / "restored.glb"
    worker._export(source, sv, sf)
    worker._export(imported, rv, rf)
    result = worker.run({"source_path": str(source), "source_sha256": worker._hash(source),
        "import_path": str(imported), "import_sha256": worker._hash(imported),
        "output_dir": str(tmp_path / "attempt"), "recipe": {"method": "import_candidate",
            "policy": {"roi": region}}})
    checks = {check["name"]: check for check in result["gate"]["checks"]}
    assert checks["outside_roi_preserved"]["passed"]
    assert checks["export_self_intersections"]["evaluated"]
    assert result["gate"]["passed"] is compatible
    if not compatible:
        assert not checks["export_closed_edges"]["passed"]
        assert result["failure_kind"] == "geometry_rejected"


def test_outside_restoration_retains_existing_face_cap(monkeypatch):
    sv, sf, cv, cf, region = restoration_fixture(True)
    monkeypatch.setattr(geometry, "MAX_FACES", 13)
    with pytest.raises(ValueError, match="triangle cap"):
        worker._restore_outside_faces(sv, sf, cv, cf, region, [])


@pytest.mark.parametrize("duplicate_face", [False, True], ids=["closed_uv_seams", "duplicate_coincident_face"])
def test_native_textured_uv_seam_glb_analyze_preserves_bytes_and_rejects_duplicate(tmp_path, duplicate_face):
    """Real PyMeshLab, no detector mock; every face has its own UV/vertex corners."""
    pytest.importorskip("pymeshlab")
    image_module = pytest.importorskip("PIL.Image")
    vertices, faces = tetra()
    if duplicate_face:
        faces = np.vstack([faces, faces[0]])
    split_vertices = vertices[faces.ravel()]
    split_faces = np.arange(len(split_vertices), dtype=np.int64).reshape((-1, 3))
    uv = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]), (len(faces), 1))
    texture = image_module.new("RGBA", (2, 2), (61, 129, 207, 255))
    material = trimesh.visual.material.SimpleMaterial(image=texture)
    mesh = trimesh.Trimesh(vertices=split_vertices, faces=split_faces, process=False)
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    source = tmp_path / "textured-uv-seams.glb"
    source.write_bytes(mesh.export(file_type="glb"))
    raw = source.read_bytes()
    # Establish that this is an embedded textured GLB, not merely a split mesh.
    json_length = int.from_bytes(raw[12:16], "little")
    document = json.loads(raw[20:20 + json_length].decode("utf-8"))
    assert document["images"] and document["textures"]
    assert "TEXCOORD_0" in document["meshes"][0]["primitives"][0]["attributes"]
    loaded_vertices, loaded_faces = worker._load(source)
    assert len(loaded_vertices) > len(np.unique(loaded_vertices, axis=0))
    assert len(loaded_faces) == (5 if duplicate_face else 4)
    original_hash = worker._hash(source)
    result = worker.run({"source_path": str(source), "source_sha256": original_hash,
        "output_dir": str(tmp_path / "attempt"), "recipe": {"method": "analyze"}})
    assert source.read_bytes() == raw
    assert (tmp_path / "attempt" / "candidate.glb").read_bytes() == raw
    assert result["gate"]["candidate_sha256"] == original_hash
    checks = {check["name"]: check for check in result["gate"]["checks"]}
    assert checks["export_self_intersections"]["evaluated"]
    assert checks["export_self_intersections"]["backend"] == "pymeshlab.compute_selection_by_self_intersections_per_face"
    diagnostics = json.loads((tmp_path / "attempt" / "diagnostics.json").read_text())
    if duplicate_face:
        assert diagnostics["candidate"]["topology"]["duplicate_faces"] == 1
        assert not checks["export_no_collapsed_or_duplicate_faces"]["passed"]
        assert not result["gate"]["passed"]
        assert result["failure_kind"] == "geometry_rejected"
    else:
        assert diagnostics["candidate"]["topology"]["all_edges_have_two_faces"]
        assert checks["export_self_intersections"]["passed"]
        assert result["gate"]["passed"]
