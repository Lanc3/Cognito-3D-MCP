"""Safety-contract tests; actual CUDA qualification is a separate guarded probe."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from codex_3d_mcp.hunyuan_mv import cumesh_repair as adapter


def test_import_and_capabilities_do_not_import_torch(monkeypatch):
    before = "torch" in sys.modules
    assert adapter.capabilities(package_paths=[])["cuda_verified"] is False
    assert ("torch" in sys.modules) == before


def test_absent_cumesh_is_discoverable_in_contract_but_not_runnable():
    from codex_3d_mcp.hunyuan_mv.repair_contract import REPAIR_METHODS, get_capabilities

    data = get_capabilities(package_paths=[])
    backend = data["experimental_backends"]["cumesh"]
    assert backend["runnable"] is False
    assert backend["runtime_qualified"] is False
    assert backend["reason_code"] == "CUMESH_PACKAGE_UNAVAILABLE"
    assert all(method in REPAIR_METHODS for method in adapter.METHODS)
    assert all(not data["methods"][method]["production_ready"] for method in adapter.METHODS)


@pytest.mark.parametrize("recipe", [
    {"method": "simplify"},
    {"method": "cumesh_cleanup", "parameters": {"remove_small_components": True}},
    {"method": "cumesh_cleanup", "parameters": {"remove_duplicate_faces": "true"}},
    {"method": "cumesh_cleanup", "parameters": {"remove_duplicate_faces": False}},
    {"method": "cumesh_fill_selected_holes", "parameters": {"max_hole_perimeter": 2}},
])
def test_unsafe_or_ambiguous_recipes_rejected(recipe):
    with pytest.raises(ValueError):
        adapter.RepairRecipe.parse(recipe)


def test_canonical_frame_hash_tracks_positions_and_connectivity():
    vertices = np.array([[10, 20, 30], [11, 20, 30], [10, 21, 30]], dtype=float)
    faces = np.array([[0, 1, 2]])
    result, _ = adapter._arrays(vertices, faces)
    assert np.array_equal(result, vertices)
    value = adapter.geometry_sha256(vertices, faces)
    assert value != adapter.geometry_sha256(vertices + 1, faces)
    assert value != adapter.geometry_sha256(vertices, faces[:, ::-1])


def test_invalid_index_is_rejected_before_backend_import(monkeypatch):
    monkeypatch.setattr(adapter, "_backend", lambda: pytest.fail("must not load backend"))
    with pytest.raises(ValueError, match="out-of-range"):
        adapter.repair([[0, 0, 0]] * 3, [[0, 1, 5]], {"method": "cumesh_diagnose"})


def test_stale_selected_loop_hash_rejected_before_backend(monkeypatch):
    monkeypatch.setattr(adapter, "_backend", lambda: pytest.fail("must not load backend"))
    with pytest.raises(ValueError, match="different input"):
        adapter.repair([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], {
            "method": "cumesh_fill_selected_holes", "parameters": {
                "selected_loop_edges": [[[0, 1], [1, 2], [2, 0]]],
                "input_geometry_sha256": "stale", "max_hole_perimeter": 4}})


def test_perimeter_cannot_close_unselected_smaller_loop():
    first = [[0, 1], [1, 2], [2, 0]]
    second = [[3, 4], [4, 5], [5, 3]]
    loops = [{"edges": first, "perimeter": 3}, {"edges": second, "perimeter": 2}]
    with pytest.raises(ValueError, match="unselected"):
        adapter._validate_selection(loops, {
            "selected_loop_edges": [first], "max_hole_perimeter": 4}, np.zeros((6, 3)))


def test_perimeter_uncertainty_rejected():
    edges = [[0, 1], [1, 2], [2, 0]]
    with pytest.raises(ValueError, match="too close"):
        adapter._validate_selection([{"edges": edges, "perimeter": 3}], {
            "selected_loop_edges": [edges], "max_hole_perimeter": 3.00001}, np.zeros((3, 3)))


def test_only_exact_planar_convex_selected_loop_is_accepted():
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    v = np.array([[10, 20, 30], [11, 20, 30], [11, 21, 30], [10, 21, 30]])
    selected = adapter._validate_selection([{"edges": edges, "perimeter": 4}], {
        "selected_loop_edges": [edges], "max_hole_perimeter": 5}, v)
    assert selected == {adapter._edge_key(edges)}
    v[3, 2] += 1
    with pytest.raises(ValueError, match="not planar"):
        adapter._planar_convex(adapter._edge_key(edges), v)


def test_concave_loop_rejected():
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    v = np.array([[0, 0, 0], [2, 0, 0], [1, 0.5, 0], [2, 2, 0], [0, 2, 0]])
    with pytest.raises(ValueError, match="not strictly convex"):
        adapter._planar_convex(adapter._edge_key(edges), v)


def test_zero_boundaries_does_not_read_uninitialized_native_loop_buffers():
    mesh = SimpleNamespace(get_boundary_loops=lambda: None,
                           num_boundary_loops=0, num_boundaries=0)
    assert adapter._loops(mesh, np.empty((0, 3))) == []
    mesh.num_boundaries = 2
    with pytest.raises(ValueError, match="no simple loops"):
        adapter._loops(mesh, np.empty((0, 3)))


def test_self_crossing_star_does_not_pass_consistent_local_turns():
    angle = np.arange(5) * 2 * np.pi / 5
    points = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(5)])
    sequence = [0, 2, 4, 1, 3]
    edges = [[sequence[i], sequence[(i + 1) % 5]] for i in range(5)]
    with pytest.raises(ValueError, match="self-crosses"):
        adapter._planar_convex(adapter._edge_key(edges), points)
