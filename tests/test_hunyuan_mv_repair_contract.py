import pytest

from codex_3d_mcp.hunyuan_mv.repair_contract import validate_recipe


def test_recipe_rejects_unknown_fields_and_gate_bypass():
    for recipe in ({"method": "analyze", "surprise": True},
                   {"method": "analyze", "parameters": {"voxel_size": 0.1}},
                   {"method": "analyze", "policy": {"require_closed": False}},
                   {"method": "analyze", "policy": {"require_self_intersection_free": False}},
                   {"method": "conservative_cleanup"}):
        with pytest.raises(ValueError):
            validate_recipe(recipe)


def test_recipe_separates_exact_weld_from_explicit_tolerance():
    recipe = validate_recipe({"method": "conservative_cleanup", "policy": {
        "roi": {"min": [-1, -1, -1], "max": [1, 1, 1]}},
        "parameters": {"exact_weld": True}})
    assert recipe["parameters"]["exact_weld"] is True
    assert recipe["parameters"]["tolerance_weld"] is None
    assert recipe["parameters"]["fix_winding"] is False


def test_import_requires_explicit_roi_and_foundation_requires_volume_policy():
    with pytest.raises(ValueError):
        validate_recipe({"method": "import_candidate"})
    with pytest.raises(ValueError):
        validate_recipe({"method": "foundation_union", "parameters": {
            "primitive": {"kind": "box", "min": [0, 0, 0], "max": [1, 1, 1]}},
            "policy": {"roi": {"min": [-1, -1, -1], "max": [2, 2, 2]}}})


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), "0.01", -1])
def test_hole_measurements_are_strict_finite_numbers(invalid):
    with pytest.raises(ValueError):
        validate_recipe({"method": "patch_selected_holes", "parameters": {
            "loop_ids": ["loop:" + "a" * 64], "max_diameter": invalid, "max_patch_area": 1},
            "policy": {"roi": {"min": [-1, -1, -1], "max": [2, 2, 2]}}})


def test_outside_face_restoration_is_explicit_foundation_boolean_option_only():
    recipe = {"method": "foundation_union", "parameters": {
        "primitive": {"kind": "box", "min": [0, 0, 0], "max": [1, 1, 1]}},
        "policy": {"roi": {"min": [-1, -1, -1], "max": [2, 2, 2]},
                   "foundation": {"roi": {"min": [0.1, 0.1, 0.1], "max": [0.9, 0.9, 0.9]},
                                  "up_axis": 1}}}
    assert validate_recipe(recipe)["parameters"]["restore_outside_faces"] is False
    recipe["parameters"]["restore_outside_faces"] = True
    assert validate_recipe(recipe)["parameters"]["restore_outside_faces"] is True
    recipe["parameters"]["restore_outside_faces"] = "true"
    with pytest.raises(ValueError):
        validate_recipe(recipe)
    with pytest.raises(ValueError):
        validate_recipe({"method": "analyze", "parameters": {"restore_outside_faces": True}})


def test_capabilities_describe_restoration_and_required_guards_without_native_imports():
    import json
    from codex_3d_mcp.hunyuan_mv.repair_contract import get_capabilities

    capabilities = get_capabilities()
    contract = capabilities["recipe_contract"]
    assert contract["methods"]["foundation_union"]["parameters"]["restore_outside_faces"]["default"] is False
    assert contract["policy"]["require_outside_roi_preserved"]["allowed"] == [True]
    assert "foundation_union" in contract["policy"]["foundation"]["required_for"]
    assert contract["methods"]["patch_selected_holes"]["parameters"]["loop_ids"]["required"] is True
    json.dumps(capabilities, allow_nan=False)
