from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_3d_mcp.errors import Codex3DError
from codex_3d_mcp.hunyuan_mv.remesh import AutoRemesherRuntime, _check_memory, _parameters


def test_file_worker_disconnects_inherited_protocol_input():
    import subprocess
    import sys

    source_root = str(Path(__file__).resolve().parents[1] / "src")
    script = (
        f"import sys; sys.path.insert(0, {source_root!r}); "
        "from codex_3d_mcp.hunyuan_mv.remesh_worker import _isolate_stdin; "
        "_isolate_stdin(); print(repr(sys.stdin.buffer.read()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=b"protocol data that must not reach numerical libraries",
        capture_output=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.strip() == b"b''"


def test_cli_numeric_options_reject_invalid_and_unsafe_values():
    settings = SimpleNamespace(remesh_target_quads=100000, remesh_threads=2)
    for params in (
        {"target_quads": -1},
        {"target_quads": 1500.5},
        {"threads": 32},
        {"memory_limit_mb": 0},
        {"adaptivity": float("nan")},
        {"edge_scaling": "--output surprise.obj"},
    ):
        with pytest.raises(Codex3DError, match="AutoRemesher"):
            _parameters(params, settings)
    assert _parameters({}, settings)["threads"] == 2


def test_resource_parameters_cannot_exceed_operator_limits():
    settings = SimpleNamespace(remesh_target_quads=100000, remesh_threads=1)
    for params in ({"threads": 2}, {"memory_limit_mb": 8192}, {"min_available_mb": 1024}):
        with pytest.raises(Codex3DError):
            _parameters(params, settings)


def test_resource_guard_checks_commit_even_when_ram_is_available(monkeypatch):
    monkeypatch.setattr(
        "codex_3d_mcp.hunyuan_mv.remesh._memory_status",
        lambda: {
            "available_physical": 12 * 1024**3,
            "available_commit": 1 * 1024**3,
        },
    )
    with pytest.raises(Codex3DError, match="memory pressure"):
        _check_memory({"memory_limit_mb": 1024, "min_available_mb": 2048}, starting=False)


@pytest.mark.parametrize("executable,current_failure", [
    ("autoremesher.exe", True), ("autoremesher.exe", False), ("python.exe", True),
])
def test_native_failed_load_stops_owned_process_without_matching_earlier_commands(
    tmp_path: Path, monkeypatch, executable: str, current_failure: bool,
):
    from codex_3d_mcp.hunyuan_mv import remesh

    log = tmp_path / "process.log"
    log.write_bytes(b"Earlier command: Error: Failed to load old.obj\n")

    class Process:
        running = True
        killed = False
        joined = False
        returncode = 0

        def poll(self):
            return None if self.running else self.returncode

        def kill(self):
            self.running = False
            self.killed = True

        def wait(self, timeout):
            self.joined = True
            return self.returncode

    process = Process()

    def launch(command, **kwargs):
        if current_failure:
            kwargs["stdout"].write(b"Error: Failed to load current.obj\n")
            kwargs["stdout"].flush()
        return process

    monkeypatch.setattr(remesh.subprocess, "Popen", launch)
    monkeypatch.setattr(remesh.subprocess, "CREATE_NO_WINDOW", 0, raising=False)
    monkeypatch.setattr(remesh.subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0, raising=False)
    monkeypatch.setattr(remesh, "_check_memory", lambda *args, **kwargs: {})
    monkeypatch.setattr(remesh, "_WindowsLimits", lambda *args: SimpleNamespace(
        proof={"logical_cpu_limit": 2}, close=lambda: None,
    ))
    monkeypatch.setattr(remesh.time, "sleep", lambda seconds: setattr(process, "running", False))
    params = {"threads": 2, "memory_limit_mb": 2048}
    if executable == "autoremesher.exe" and current_failure:
        with pytest.raises(Codex3DError, match="failed to load its input"):
            AutoRemesherRuntime._run([executable], log, None, params, float("inf"))
        assert process.killed
    else:
        AutoRemesherRuntime._run([executable], log, None, params, float("inf"))
        assert not process.killed
    assert process.joined


@pytest.mark.parametrize("native_fails", [False, True])
def test_native_uses_short_paths_and_preserves_outputs_in_long_attempt(
    tmp_path: Path, monkeypatch, native_fails: bool,
):
    temporary_root = tmp_path / "native-temp"
    temporary_root.mkdir()
    monkeypatch.setattr(
        "codex_3d_mcp.hunyuan_mv.remesh.tempfile.gettempdir", lambda: str(temporary_root),
    )
    output = tmp_path / ("a" * 80) / ("b" * 80) / ("c" * 80) / "remeshed.glb"
    source = tmp_path / "shape.glb"
    source.write_bytes(b"immutable shape")
    settings = SimpleNamespace(
        autoremesher_exe=Path("native.exe"), python_exe=Path("python.exe"),
        remesh_target_quads=1000, remesh_threads=2, remesh_timeout_seconds=60,
    )
    runtime = AutoRemesherRuntime(settings)
    monkeypatch.setattr(runtime, "preflight", lambda: {"ready": True})
    stages, observed = [], {}

    def runner(command, log, cancel_event, params, deadline):
        assert params["threads"] == 2 and params["memory_limit_mb"] == 2048
        if command[0] == str(settings.autoremesher_exe):
            stages.append("native")
            paths = {flag: Path(command[command.index(flag) + 1])
                     for flag in ("--input", "--output", "--report")}
            scratch = paths["--input"].parent
            observed["scratch"] = scratch
            assert scratch.parent == temporary_root.resolve()
            assert all(path.parent == scratch and len(str(path)) < 240 for path in paths.values())
            assert paths["--input"].read_bytes() == b"prepared mesh"
            paths["--output"].write_bytes(b"native quads")
            paths["--report"].write_text("native report", encoding="utf-8")
            if native_fails:
                raise Codex3DError("simulated native failure", code="REMESH_PROCESS_FAILED")
        else:
            stage = command[2]
            stages.append(stage)
            request = json.loads(Path(command[3]).read_text(encoding="utf-8"))
            attempt = Path(request["attempt"])
            observed["attempt"] = attempt
            assert len(str(attempt / "normalized.obj")) > 300
            if stage == "prepare":
                (attempt / "normalized.obj").write_bytes(b"prepared mesh")
            else:
                assert stage == "validate"
                assert not observed["scratch"].exists()
                assert (attempt / "quads.obj").read_bytes() == b"native quads"
                assert (attempt / "native-report.txt").read_text() == "native report"
                candidate = attempt / "candidate.glb"
                candidate.write_bytes(b"validated GLB")
                (attempt / "result.json").write_text(json.dumps({
                    "passed": True, "artifacts": {"candidate_glb": str(candidate)},
                }), encoding="utf-8")
        return {"logical_cpu_limit": 2, "memory_limit_mb": 2048}

    monkeypatch.setattr(runtime, "_run", runner)
    params = {"memory_limit_mb": 2048}
    if native_fails:
        with pytest.raises(Codex3DError, match="simulated native failure"):
            runtime.remesh(source, output, params, tmp_path / "aggregate.log")
        assert stages == ["prepare", "native"]
        assert not output.exists()
    else:
        result = runtime.remesh(source, output, params, tmp_path / "aggregate.log")
        assert result["passed"] and stages == ["prepare", "native", "validate"]
        assert output.read_bytes() == b"validated GLB"
        assert Path(result["artifacts"]["candidate_glb"]).parent == observed["attempt"]
    assert not observed["scratch"].exists()
    assert (observed["attempt"] / "quads.obj").read_bytes() == b"native quads"
    assert (observed["attempt"] / "native-report.txt").read_text() == "native report"
    assert source.read_bytes() == b"immutable shape"


def _cube_obj(path: Path, *, open_top: bool = False, offset: float = 0.0):
    vertices = [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ]
    faces = [(1, 4, 3, 2), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8)]
    if open_top:
        faces.pop(1)
    path.write_text(
        "".join(f"v {x + offset} {y} {z}\n" for x, y, z in vertices)
        + "".join("f " + " ".join(map(str, face)) + "\n" for face in faces)
    )


def _worker():
    pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    from codex_3d_mcp.hunyuan_mv import remesh_worker

    return remesh_worker


def test_missing_output_is_a_failed_gate_not_success_exit(tmp_path: Path):
    worker = _worker()
    result = worker.validate({"attempt": str(tmp_path), "params": {"target_quads": 1000}})
    assert result["passed"] is False
    assert result["checks"][0]["name"] == "native_output_exists"


def test_validation_rejects_open_mesh_even_when_native_has_quads(tmp_path: Path):
    worker = _worker()
    _cube_obj(tmp_path / "normalized.obj")
    _cube_obj(tmp_path / "quads.obj", open_top=True)
    result = worker.validate({"attempt": str(tmp_path), "params": {"target_quads": 1000}})
    assert not result["passed"]
    # The missing top is rejected before triangulation/export can create a mesh.
    checks = {item["name"]: item for item in result["checks"]}
    topology = checks["native_polygon_topology"]
    assert topology["passed"] is False
    assert topology["boundary_edges"] == 4
    assert topology["nonmanifold_edges"] == 0
    assert {tuple(item["edge"]) for item in topology["examples"]} == {
        (4, 5), (5, 6), (6, 7), (4, 7),
    }
    assert checks["polygon_triangulation"]["passed"] is False
    assert "native polygon topology is open" in checks["polygon_triangulation"]["reason"]
    assert not (tmp_path / "candidate.glb").exists()
    assert result["cleanup"]["native_quads"] == 5
    assert result["cleanup"]["removed_duplicate_polygons"] == 0


def test_validation_rejects_displaced_closed_mesh(tmp_path: Path):
    worker = _worker()
    _cube_obj(tmp_path / "normalized.obj")
    _cube_obj(tmp_path / "quads.obj", offset=0.5)
    result = worker.validate({"attempt": str(tmp_path), "params": {"target_quads": 1000}})
    assert not result["passed"]
    assert not next(item for item in result["checks"] if item["name"] == "surface_preserved")[
        "passed"
    ]


def test_validation_accepts_matching_closed_quad_mesh(tmp_path: Path):
    worker = _worker()
    _cube_obj(tmp_path / "normalized.obj")
    _cube_obj(tmp_path / "quads.obj")
    result = worker.validate({"attempt": str(tmp_path), "params": {"target_quads": 1000}})
    assert result["passed"], json.dumps(result["checks"])
    assert result["geometry"]["native_quads"] == 6
    assert Path(result["artifacts"]["candidate_glb"]).is_file()


def test_triangle_budget_is_independent_of_quad_target(tmp_path: Path):
    worker = _worker()
    _cube_obj(tmp_path / "normalized.obj")
    _cube_obj(tmp_path / "quads.obj")
    result = worker.validate({
        "attempt": str(tmp_path), "params": {"target_quads": 1000, "triangle_budget": 11},
    })
    check = next(item for item in result["checks"] if item["name"] == "face_budget")
    assert check["triangles"] == 12 and check["maximum"] == 11
    assert not check["passed"] and not result["passed"]


def test_validation_removes_three_duplicate_quads_and_preserves_native(tmp_path: Path):
    worker = _worker()
    _cube_obj(tmp_path / "normalized.obj")
    native = tmp_path / "quads.obj"
    _cube_obj(native)
    expected_cleaned = native.read_bytes() + b"# Keep native metadata\r\n"
    # Exact, cyclically rotated, and reversed duplicates of three cube faces.
    raw = expected_cleaned + b"f 1 4 3 2\nf 6 7 8 5\nf 5 6 2 1\n"
    native.write_bytes(raw)

    result = worker.validate({"attempt": str(tmp_path), "params": {"target_quads": 1000}})

    assert result["passed"], json.dumps(result["checks"])
    assert result["geometry"]["triangles"] == 12
    assert result["cleanup"]["removed_duplicate_polygons"] == 3
    assert result["cleanup"]["native_quads"] == 9
    assert result["cleanup"]["cleaned_quads"] == 6
    assert Path(result["artifacts"]["native_quad_obj"]) == native
    assert native.read_bytes() == raw
    assert Path(result["artifacts"]["quad_obj"]).read_bytes() == expected_cleaned
    cleanup_report = Path(result["artifacts"]["duplicate_cleanup"])
    assert json.loads(cleanup_report.read_text()) == result["cleanup"]


def test_duplicate_cleanup_preserves_distinct_order_and_nonface_lines(tmp_path: Path):
    from codex_3d_mcp.hunyuan_mv.remesh_worker import _remove_duplicate_polygons

    native, cleaned = tmp_path / "quads.obj", tmp_path / "quads.cleaned.obj"
    original = (
        b"# Raw metadata\r\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        b"usemtl original\nf 1 2 3 4\nf 1 3 2 4\n"
    )
    # Equivalent relative indices duplicate the first loop; a different
    # connectivity with the same vertex set above must remain untouched.
    native.write_bytes(original + b"f -4 -3 -2 -1\n")
    report = _remove_duplicate_polygons(native, cleaned)
    assert report["removed_duplicate_polygons"] == 1
    assert cleaned.read_bytes() == original


def test_validation_detects_dropped_component_despite_nonempty_closed_output(tmp_path: Path):
    worker = _worker()
    import trimesh

    first = trimesh.creation.box()
    second = trimesh.creation.box()
    second.apply_translation((3, 0, 0))
    trimesh.util.concatenate([first, second]).export(tmp_path / "normalized.obj")
    _cube_obj(tmp_path / "quads.obj")
    result = worker.validate({"attempt": str(tmp_path), "params": {"target_quads": 1000}})
    assert not result["passed"]
    check = next(
        item for item in result["checks"] if item["name"] == "significant_components_preserved"
    )
    assert check["source"] == 2 and check["remeshed"] == 1
    assert not check["passed"]
