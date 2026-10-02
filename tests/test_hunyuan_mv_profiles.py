"""Profile orchestration and strict exported counts without native/GPU work."""

import json
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from codex_3d_mcp.errors import Codex3DError
from codex_3d_mcp.hunyuan_mv import stage_processor as module
from codex_3d_mcp.hunyuan_mv.config import HunyuanMVSettings
from codex_3d_mcp.hunyuan_mv.remesh import PROFILE_TARGETS, profile_parameters
from codex_3d_mcp.trellis.artifacts import artifact_record


def write_glb(path, triangles):
    document = {
        "asset": {"version": "2.0"},
        "accessors": [{"count": triangles * 3}],
        "meshes": [{"primitives": [{"indices": 0}]}],
    }
    content = json.dumps(document).encode()
    content += b" " * (-len(content) % 4)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 20 + len(content))
        + struct.pack("<II", len(content), 0x4E4F534A) + content
    )


@pytest.fixture
def setup(tmp_path):
    processor = object.__new__(module.BatchStageProcessor)
    processor.settings = HunyuanMVSettings.from_env(tmp_path)
    processor.runtime = SimpleNamespace(begin_stage=lambda *args: None)
    calls = []
    counts = {"full_game": 12, "mobile": 10, "browser": 8}

    def remesh(source, output, params, log, cancel):
        calls.append((source, output.parent.name, dict(params)))
        write_glb(output, counts[output.parent.name])
        return {
            "passed": True, "checks": [], "artifacts": {},
            "geometry": {"triangles": 12, "native_quads": 6, "cleaned_quads": 6},
        }

    processor.remesher = SimpleNamespace(remesh=remesh)
    source = tmp_path / "shape.glb"
    write_glb(source, 24)
    source_hash = module._sha256(source)
    request = {"request_id": "fixture-repair-1", "source_sha256": source_hash,
               "source_shape_sha256": source_hash, "source_shape_attempt": 1}
    asset = {
        "params": {
            **profile_parameters({}), "threads": 2, "memory_limit_mb": 2048,
            "min_available_mb": 2048,
        },
        "stages": {
            "references": {"prepared_views": {}}, "shape": {"output": str(source), "attempt": 1},
            "shape_repair": {
                "approved": True, "attempt": 1, "output": str(source), "repair_request": request,
                "gate": {"passed": True, "gate_version": "shape-repair-v1",
                         "source_sha256": source_hash, "output_sha256": source_hash},
            },
            "remesh": {},
        },
        "accepted_master": {"path": str(source), "sha256": source_hash, **request,
                            "gate_version": "shape-repair-v1", "attempt": 1},
    }
    directory = tmp_path / "remesh" / "attempt-0001"
    directory.mkdir(parents=True)
    return processor, asset, directory, calls, counts


def test_profiles_use_same_source_in_order_and_record_actual_exported_counts(setup):
    processor, asset, directory, calls, _ = setup
    result = processor.run("remesh", asset, directory, threading.Event())
    assert result["gate"]["passed"]
    assert [name for _, name, _ in calls] == list(PROFILE_TARGETS)
    assert {path for path, _, _ in calls} == {Path(asset["stages"]["shape"]["output"])}
    assert [params["triangle_budget"] for _, _, params in calls] == [100000, 50000, 20000]
    assert [params["target_quads"] for _, _, params in calls] == [50000, 25000, 10000]
    assert all(
        params["memory_limit_mb"] == 2048 and params["threads"] == 2 for _, _, params in calls
    )
    assert result["profiles"]["mobile"]["exported_triangles"] == 10
    asset["stages"]["remesh"] = {**result, "approved": True}
    assert processor._accepted_profiles(asset)["profiles"] == result["profiles"]


def test_retry_reuses_only_profiles_with_identical_geometry_inputs(setup):
    processor, asset, directory, calls, _ = setup
    first = processor.run("remesh", asset, directory, threading.Event())
    asset["stages"]["remesh"] = {"history": [first]}
    asset["params"]["mobile_target_quads"] = 28000
    next_dir = directory.parent / "attempt-0002"
    next_dir.mkdir()
    calls.clear()
    second = processor.run("remesh", asset, next_dir, threading.Event())
    assert [name for _, name, _ in calls] == ["mobile"]
    assert second["profiles"]["full_game"]["output"] == first["profiles"]["full_game"]["output"]
    assert second["profiles"]["browser"]["output"] == first["profiles"]["browser"]["output"]
    asset["stages"]["remesh"] = {**second, "approved": True}
    assert processor._accepted_profiles(asset)["passed"]


def test_tampered_previous_profile_is_recomputed(setup):
    processor, asset, directory, calls, _ = setup
    first = processor.run("remesh", asset, directory, threading.Event())
    asset["stages"]["remesh"] = {"history": [first]}
    write_glb(Path(first["profiles"]["browser"]["output"]), 16)
    next_dir = directory.parent / "attempt-0002"
    next_dir.mkdir()
    calls.clear()
    processor.run("remesh", asset, next_dir, threading.Event())
    assert [name for _, name, _ in calls] == ["browser"]


def test_raw_shape_without_master_review_cannot_remesh(setup):
    processor, asset, directory, calls, _ = setup
    del asset["accepted_master"]
    with pytest.raises(Codex3DError, match="Shape repair and review"):
        processor.run("remesh", asset, directory, threading.Event())
    assert calls == []


@pytest.mark.parametrize("tamper", ["pointer", "repair_output", "certificate_output", "certificate_source",
                                     "request_id", "request_source", "request_shape_attempt"])
def test_master_pointer_must_match_current_reviewed_candidate_and_certificate(setup, tamper):
    processor, asset, directory, calls, _ = setup
    other = directory.parent / "other.glb"
    write_glb(other, 18)
    repair = asset["stages"]["shape_repair"]
    if tamper == "pointer":
        asset["accepted_master"].update(path=str(other), sha256=module._sha256(other))
    elif tamper == "repair_output":
        repair["output"] = str(other)
    elif tamper == "certificate_output":
        repair["gate"]["output_sha256"] = "0" * 64
    elif tamper == "certificate_source":
        repair["gate"]["source_sha256"] = "0" * 64
    elif tamper == "request_id":
        repair["repair_request"]["request_id"] = "another-request"
    elif tamper == "request_source":
        repair["repair_request"]["source_sha256"] = "0" * 64
    else:
        repair["repair_request"]["source_shape_attempt"] = 2
    with pytest.raises(Codex3DError, match="reviewed repair certificate"):
        processor.run("remesh", asset, directory, threading.Event())
    assert calls == []


def _accepted_paint(processor, asset, directory):
    result = processor.run("remesh", asset, directory, threading.Event())
    asset["stages"]["remesh"] = {**result, "approved": True}
    paint_dir = directory.parent.parent / "paint" / "attempt-0001"
    paint_dir.mkdir(parents=True)
    painted = paint_dir / "textured.glb"
    write_glb(painted, 12)
    asset["stages"]["paint"] = {
        "approved": True, "output": str(painted),
        "source_profile_sha256": result["profiles"]["full_game"]["output_sha256"],
        "artifacts": [artifact_record(painted, paint_dir)],
    }
    return result, painted


@pytest.mark.parametrize("tamper", ["paint_bytes", "source_profile", "missing_artifact"])
def test_finish_checks_preserved_paint_against_current_full_game_and_own_artifact(setup, monkeypatch, tamper):
    processor, asset, directory, _, _ = setup
    _, painted = _accepted_paint(processor, asset, directory)
    if tamper == "paint_bytes":
        write_glb(painted, 9)
    elif tamper == "source_profile":
        asset["stages"]["paint"]["source_profile_sha256"] = "0" * 64
    else:
        asset["stages"]["paint"]["artifacts"] = []
    monkeypatch.setattr(processor, "_blender", lambda *args, **kwargs: pytest.fail("Stale Paint reached Blender"))
    with pytest.raises(Codex3DError) as raised:
        processor.run("finish", asset, directory, threading.Event())
    assert raised.value.code == ("PAINT_STALE" if tamper == "source_profile" else "PAINT_CHANGED")


@pytest.mark.parametrize("defect", [None, "master_alias", "over_budget", "diagnostic_count",
                                    "diagnostic_hash", "topology"])
def test_finish_uses_actual_serialized_exports_and_exact_master_alias(setup, monkeypatch, defect):
    processor, asset, directory, _, counts = setup
    _accepted_paint(processor, asset, directory)
    asset.update(asset_id="fixture", name="Serialized export fixture")
    finished = directory.parent.parent / "finish" / "attempt-0001"
    finished.mkdir(parents=True)
    seen = []

    def blender(source, destination, cancel, **kwargs):
        for name in PROFILE_TARGETS:
            triangles = 50001 if defect == "over_budget" and name == "mobile" else counts[name]
            write_glb(destination / f"{name}.glb", triangles)
        (destination / "master.glb").write_bytes((destination / "full_game.glb").read_bytes())
        if defect == "master_alias":
            write_glb(destination / "master.glb", 11)
        # Simulate successful pre-export metrics. Serialized files must still be checked.
        return {"geometry": {name: {"faces": counts[name]} for name in PROFILE_TARGETS}}

    def analyze(validation_asset, destination, cancel):
        source = Path(validation_asset["shape_repair_request"]["source_path"])
        seen.append(source.stem)
        digest = module._sha256(source)
        measured = module._glb_triangle_count(source)
        candidate = destination / "candidate.glb"
        candidate.write_bytes(source.read_bytes())
        if defect == "diagnostic_count" and source.stem == "mobile":
            measured += 1
        diagnostics = destination / "diagnostics.json"
        diagnostics.write_text(json.dumps({
            "source_sha256": "0" * 64 if defect == "diagnostic_hash" else digest,
            "output_sha256": digest, "candidate": {"topology": {"triangles": measured}},
        }))
        report = destination / "repair-report.json"
        report.write_text("{}")
        return {"output": str(candidate), "gate": {"passed": defect != "topology"},
                "diagnostics": str(diagnostics), "repair_report": str(report)}

    processor.qa_manager = SimpleNamespace(_qa=lambda *args: {"passed": True, "checks": []})
    monkeypatch.setattr(processor, "_blender", blender)
    monkeypatch.setattr(processor, "_repair_shape", analyze)
    monkeypatch.setattr(module, "_has_color_texture", lambda path: True)
    result = processor.run("finish", asset, finished, threading.Event())
    assert seen == list(PROFILE_TARGETS)
    assert result["gate"]["passed"] is (defect is None)
    checks = {item["name"]: item for item in result["gate"]["checks"]}
    if defect == "master_alias":
        assert checks["master_matches_validated_full_game"]["passed"] is False
    if defect == "over_budget":
        check = checks["mobile_serialized_triangle_budget"]
        assert check["triangles"] == 50001 and check["maximum"] == 50000


def test_actual_over_budget_export_fails_even_when_worker_reports_success(setup):
    processor, asset, directory, calls, counts = setup
    counts["mobile"] = 50001
    result = processor.run("remesh", asset, directory, threading.Event())
    assert not result["gate"]["passed"]
    assert [name for _, name, _ in calls] == ["full_game", "mobile"]
    assert result["profiles"]["mobile"]["exported_triangles"] == 50001
    assert not result["profiles"]["mobile"]["gate"]["checks"][-1]["passed"]
    asset["stages"]["remesh"] = {**result, "approved": True}
    with pytest.raises(Codex3DError, match="All three"):
        processor._accepted_profiles(asset)


def test_evidence_renders_every_profile_serially_and_exposes_twelve_paths(setup, monkeypatch):
    processor, asset, directory, _, counts = setup
    result = processor.run("remesh", asset, directory, threading.Event())
    rendered = []

    def blender(source, destination, cancel, **kwargs):
        name = destination.name
        rendered.append(name)
        previews = destination / "previews"
        previews.mkdir()
        for view in ("front", "back", "left", "right"):
            (previews / f"{view}.png").write_bytes(b"preview fixture")
        return {"geometry": {
            "faces": counts[name], "watertight": True, "non_manifold_edges": 0,
            "degenerate_faces": 0,
        }}

    monkeypatch.setattr(processor, "_blender", blender)
    evidence = processor.evidence("remesh", Path(result["output"]), directory, threading.Event())
    assert rendered == list(PROFILE_TARGETS)
    assert evidence["gate"]["passed"]
    assert len(evidence["previews"]) == 12
    assert set(evidence["previews"]) == {
        f"{name}_{view}" for name in PROFILE_TARGETS for view in ("front", "back", "left", "right")
    }


@pytest.mark.parametrize("mutate", ["source", "profile", "params", "missing_manifest"])
def test_reviewed_profile_inputs_are_reverified_before_paint(setup, mutate):
    processor, asset, directory, _, _ = setup
    result = processor.run("remesh", asset, directory, threading.Event())
    asset["stages"]["remesh"] = {**result, "approved": True}
    if mutate == "source":
        Path(asset["stages"]["shape"]["output"]).write_bytes(b"changed source")
    elif mutate == "profile":
        write_glb(Path(result["profiles"]["mobile"]["output"]), 14)
    elif mutate == "params":
        asset["params"]["mobile_target_quads"] = 20000
    else:
        asset["stages"]["remesh"].pop("profiles_manifest")
    with pytest.raises(Codex3DError):
        processor._accepted_profiles(asset)


def test_finish_config_requires_profiles_and_nonzero_python_failure_exit(setup, monkeypatch):
    processor, asset, directory, _, _ = setup
    result = processor.run("remesh", asset, directory, threading.Event())
    captured = {}

    def run(command, log, cancel, **kwargs):
        captured["command"] = command
        captured["config"] = json.loads(Path(command[-2]).read_text())
        Path(command[-1]).write_text("{}")

    monkeypatch.setattr(module, "run_cpu_process", run)
    with pytest.raises(Codex3DError, match="requires all remesh profiles"):
        processor._blender(Path(result["output"]), directory, threading.Event(), preview_only=False)
    processor._blender(
        Path(result["output"]), directory, threading.Event(), preview_only=False,
        profiles=result["profiles"], profiles_manifest=result["profiles_manifest"],
    )
    assert captured["config"]["prepared_profiles"] is True
    assert set(captured["config"]["profile_meshes"]) == set(PROFILE_TARGETS)
    assert captured["config"]["profile_triangle_budgets"] == {
        "full_game": 100000, "mobile": 50000, "browser": 20000,
    }
    command = captured["command"]
    assert command[command.index("--python-exit-code") + 1] == "1"
