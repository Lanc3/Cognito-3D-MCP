import itertools
import math

import pytest

from codex_3d_mcp.hunyuan_mv.blender_worker import _preview_camera_positions


def test_flat_translated_asset_is_centered_with_conditioning_view_orientation():
    corners = list(itertools.product((9.0, 11.0), (18.0, 22.0), (3.0, 3.8)))
    center = (10.0, 20.0, 3.4)
    cameras = _preview_camera_positions(corners)

    for location, rotation in cameras.values():
        # After Blender's X=90 degree rotation, local -Z looks horizontally
        # toward the bounds center, including the actual thin asset's height.
        yaw = rotation[2]
        forward = (-math.sin(yaw), math.cos(yaw), 0.0)
        to_center = tuple(center[axis] - location[axis] for axis in range(3))
        assert to_center == pytest.approx(tuple(4 * value for value in forward))

    # A nose facing Blender -Y must project image-left in the left reference
    # and image-right in the right reference, matching Hunyuan's samples.
    nose = (0.0, -1.0, 0.0)
    projected_nose = {}
    for name in ("left", "right"):
        yaw = cameras[name][1][2]
        camera_right = (math.cos(yaw), math.sin(yaw), 0.0)
        projected_nose[name] = sum(a * b for a, b in zip(nose, camera_right))
    assert projected_nose["left"] < 0 < projected_nose["right"]
