"""Blender-side regression fixture for imported quaternion rotations."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import bpy


def main() -> None:
    separator = sys.argv.index("--")
    worker_path = Path(sys.argv[separator + 1]).resolve()
    result_path = Path(sys.argv[separator + 2]).resolve()
    spec = importlib.util.spec_from_file_location("blender_worker_under_test", worker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {worker_path}")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    mesh = bpy.data.meshes.new("asymmetric_rotation_fixture")
    mesh.from_pydata([(1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], [], [(0, 1, 2)])
    subject = bpy.data.objects.new("quaternion_import", mesh)
    bpy.context.collection.objects.link(subject)
    subject.rotation_mode = "QUATERNION"
    bpy.context.view_layer.objects.active = subject
    subject.select_set(True)

    worker._apply_world_yaw(subject, math.pi)
    point = subject.matrix_world @ subject.data.vertices[0].co
    result_path.write_text(
        json.dumps(
            {
                "point": list(point),
                "rotation_mode": subject.rotation_mode,
                "rotation_euler": list(subject.rotation_euler),
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
