"""Repair orchestration using tiny fake artifacts; no geometry/model execution."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from codex_3d_mcp.errors import Codex3DError
from codex_3d_mcp.hunyuan_mv import batch as batch_module
from codex_3d_mcp.hunyuan_mv.batch import REPAIR_VIEWS, HunyuanBatchManager
from codex_3d_mcp.hunyuan_mv.repair_contract import REPAIR_GATE_VERSION
from codex_3d_mcp.hunyuan_mv.repair_migration import migrate_repair_stage
from codex_3d_mcp.trellis.artifacts import artifact_record, sha256_file
from test_hunyuan_mv_batch import make_settings


class RepairProcessor:
    def __init__(self):
        self.passed = True
        self.seen = []
        self.failure_kind = None
        self.gate_extra = {}
        self.report_error = None

    def run(self, stage, asset, directory, cancel):
        assert stage == "shape_repair"
        request = copy.deepcopy(asset["shape_repair_request"])
        self.seen.append(request)
        source = Path(request.get("import_path") or request["source_path"])
        output = directory / "candidate.glb"
        output.write_bytes(source.read_bytes())
        gate = {
            "passed": self.passed, "gate_version": REPAIR_GATE_VERSION,
            "source_sha256": request["source_sha256"], "output_sha256": sha256_file(output),
            **self.gate_extra,
        }
        report = directory / "repair-report.json"
        report.write_text(json.dumps({"gate": gate, "error": self.report_error}))
        return {"output": str(output), "gate": gate, "repair_report": str(report),
                "failure_kind": self.failure_kind}

    def evidence(self, stage, output, directory, cancel):
        previews = {}
        for name in REPAIR_VIEWS:
            path = directory / f"{name}.png"
            path.write_bytes(b"synthetic preview evidence")
            previews[name] = str(path)
        return {"gate": {"passed": True}, "previews": previews}

    def end_stage(self):
        pass

    def close(self):
        pass


@pytest.fixture
def manager(tmp_path):
    value = HunyuanBatchManager(make_settings(tmp_path), RepairProcessor())
    # Tests drive transitions directly; the scheduler must never execute a model.
    value._shutdown.set()
    with value._condition:
        value._condition.notify_all()
    value._thread.join(timeout=5)
    assert not value._thread.is_alive()
    yield value
    value.close()


def seed_shapes(manager, count=1):
    status = manager.create_batch("Repair fixtures", [
        {"name": f"Asset {index}", "prompt": f"Fixture {index}"} for index in range(count)
    ])
    batch = manager._batch(status["batch_id"])
    batch.update(stage="shape_repair", state="paused", paused=True)
    for index, asset in enumerate(batch["assets"]):
        asset["stages"]["references"].update(approved=True, state="approved", gate={"passed": True})
        directory = manager._attempt_directory(batch, asset, "shape", 1)
        directory.mkdir(parents=True)
        path = directory / "shape.glb"
        path.write_bytes(f"immutable raw fixture {index}".encode())
        asset["stages"]["shape"].update(
            state="awaiting_review", attempt=1, output=str(path), approved=False,
            gate={"passed": True, "geometry": {"faces": 4}},
            artifacts=[artifact_record(path, directory)],
        )
    return batch, batch["assets"][0]


def queue_analyze(manager, batch, asset, key="analysis-1"):
    return manager.create_shape_repair_attempt(
        batch["batch_id"], asset["asset_id"], {"method": "analyze"},
        sha256_file(Path(asset["stages"]["shape"]["output"])),
        "Inspect topology and identity before master adoption", key, expected_shape_attempt=1,
    )


def run_attempt(manager, batch, asset):
    batch.update(paused=False, state="queued", stage="shape_repair")
    record, directory = manager._new_attempt(batch, asset, "shape_repair")
    manager._perform(batch, asset, "shape_repair", record, directory, False)
    if record["state"] == "needs_evidence":
        record["state"] = "running"
        manager._perform(batch, asset, "shape_repair", record, directory, True)
    return record, directory


def review_paths(record):
    return [*record["previews"].values(), record["repair_report"]]


def approve(manager, batch, asset, record, paths=None):
    return manager.review_asset(batch["batch_id"], asset["asset_id"], "shape_repair",
                                record["attempt"], True, "Inspected exact candidate and all evidence",
                                paths if paths is not None else review_paths(record))


def test_raw_quality_failure_can_enter_diagnosis_without_being_approved(manager):
    batch, asset = seed_shapes(manager)
    shape = asset["stages"]["shape"]
    shape.update(state="needs_repair", gate={"passed": False, "geometry": {"faces": 4}})
    assert manager._phase_complete(batch, "shape")
    assert not shape["approved"]
    assert not manager._phase_complete(batch, "shape_repair")


def test_failed_repair_keeps_evidence_and_cannot_be_approved(manager):
    batch, asset = seed_shapes(manager)
    queue_analyze(manager, batch, asset)
    manager.processor.passed = False
    record, _ = run_attempt(manager, batch, asset)
    assert record["state"] == "needs_repair"
    assert record["gate"]["compute"]["passed"] is False
    assert set(record["previews"]) == set(REPAIR_VIEWS)
    with pytest.raises(Codex3DError, match="machine gate"):
        approve(manager, batch, asset, record)
    assert "accepted_master" not in asset


def test_master_adoption_binds_hashes_policy_and_seven_views(manager):
    batch, asset = seed_shapes(manager)
    raw = Path(asset["stages"]["shape"]["output"])
    original = sha256_file(raw)
    queue_analyze(manager, batch, asset)
    record, directory = run_attempt(manager, batch, asset)
    with pytest.raises(ValueError, match="seven"):
        approve(manager, batch, asset, record, list(record["previews"].values()))
    assert not record["approved"]
    approve(manager, batch, asset, record)
    master = asset["accepted_master"]
    assert master["source_shape_sha256"] == original == sha256_file(raw)
    assert master["sha256"] == sha256_file(Path(record["output"]))
    assert master["gate_version"] == REPAIR_GATE_VERSION
    assert (directory / "accepted-master.json").is_file()
    assert manager._phase_complete(batch, "shape_repair")
    assert not asset["stages"]["shape"]["approved"]


def test_final_gate_certificate_hash_survives_evidence_and_can_be_reviewed(manager):
    batch, asset = seed_shapes(manager)
    queue_analyze(manager, batch, asset)
    record, directory = run_attempt(manager, batch, asset)
    certificate = directory / "gate.json"
    retained = next(item for item in record["artifacts"] if Path(item["path"]) == certificate)
    assert retained == artifact_record(certificate, directory)
    content = json.loads(certificate.read_text())
    assert content["finished_at"] == record["finished_at"]
    assert all(Path(item["path"]) != certificate for item in content["artifacts"])
    assert set(content["previews"]) == set(REPAIR_VIEWS)
    approve(manager, batch, asset, record, [*review_paths(record), str(certificate)])
    assert asset["accepted_master"]["sha256"] == sha256_file(Path(record["output"]))


@pytest.mark.parametrize("failure_kind,gate_extra,report_error,expected", [
    ("runtime_unavailable", {}, None, "runtime_unavailable"),
    ("resource", {}, None, "resource"),
    (None, {"failure_kind": "runtime_failure"}, None, "runtime_failure"),
    (None, {"checks": [{"evaluated": False, "failure_kind": "runtime_unavailable"}]},
     None, "runtime_unavailable"),
    (None, {"checks": [{"name": "export_self_intersections", "evaluated": False,
                       "passed": False, "backend": "pymeshlab"}]}, None, "runtime_unavailable"),
    (None, {}, {"type": "ModuleNotFoundError", "message": "Missing optional backend"}, "runtime_unavailable"),
    (None, {}, {"type": "RuntimeError", "message": "Backend operation failed"}, "runtime_failure"),
    (None, {}, {"type": "MemoryError", "message": "Bounded allocation failed"}, "resource"),
])
def test_operational_repair_failure_stays_unapproved_but_allows_retry_after_cause_fix(
    manager, failure_kind, gate_extra, report_error, expected,
):
    batch, asset = seed_shapes(manager)
    queue_analyze(manager, batch, asset)
    manager.processor.passed = False
    manager.processor.failure_kind = failure_kind
    manager.processor.gate_extra = gate_extra
    manager.processor.report_error = report_error
    record, _ = run_attempt(manager, batch, asset)
    # Classification must survive the subsequent evidence gate wrapping compute.
    assert record["error"]["code"] == "REPAIR_OPERATION_FAILED"
    assert record["error"]["failure_kind"] == expected
    assert record["gate"]["passed"] is False
    assert record["state"] == "needs_repair"
    assert "accepted_master" not in asset
    queue_analyze(manager, batch, asset, key="retry-after-backend-correction")
    assert asset["stages"]["shape_repair"]["state"] == "queued"


@pytest.mark.parametrize("gate_extra,report_error", [
    ({"failure_kind": "geometry_rejected"}, None),
    ({"checks": [{"name": "export_self_intersections", "evaluated": True,
                  "passed": False, "selected_faces": 4, "backend": "pymeshlab"}]}, None),
    ({"checks": [{"name": "authored_foundation_intervals", "evaluated": False,
                  "passed": False, "reason": "invalid_solid_preconditions"}]}, None),
    ({}, {"type": "ValueError", "message": "Foundation plug does not overlap source volume"}),
])
def test_geometric_rejection_does_not_become_retryable_runtime_failure(manager, gate_extra, report_error):
    batch, asset = seed_shapes(manager)
    queue_analyze(manager, batch, asset)
    manager.processor.passed = False
    manager.processor.gate_extra = gate_extra
    manager.processor.report_error = report_error
    record, _ = run_attempt(manager, batch, asset)
    assert record["error"]["code"] == "QUALITY_GATES_FAILED"
    with pytest.raises(Codex3DError, match="already exists"):
        queue_analyze(manager, batch, asset, key="blind-quality-repeat")


@pytest.mark.parametrize("tamper", ["candidate", "source", "report", "preview", "gate_version", "source_binding", "shape_attempt"])
def test_tampered_or_stale_candidate_never_adopts(manager, tamper):
    batch, asset = seed_shapes(manager)
    queue_analyze(manager, batch, asset)
    record, _ = run_attempt(manager, batch, asset)
    if tamper == "candidate":
        Path(record["output"]).write_bytes(b"edited candidate")
    elif tamper == "source":
        Path(asset["stages"]["shape"]["output"]).write_bytes(b"edited raw")
    elif tamper == "report":
        Path(record["repair_report"]).write_text("changed diagnostic report")
    elif tamper == "preview":
        Path(record["previews"]["bottom"]).write_bytes(b"changed evidence")
    elif tamper == "gate_version":
        record["gate"]["compute"]["gate_version"] = "obsolete"
    elif tamper == "source_binding":
        record["gate"]["compute"]["source_sha256"] = "0" * 64
    else:
        asset["stages"]["shape"]["attempt"] = 2
    with pytest.raises((ValueError, Codex3DError)):
        approve(manager, batch, asset, record)
    assert "accepted_master" not in asset
    assert record["approved"] is False


def test_idempotency_does_not_duplicate_or_allow_conflicting_requests(manager):
    batch, asset = seed_shapes(manager)
    queue_analyze(manager, batch, asset)
    request_id = asset["shape_repair_request"]["request_id"]
    queue_analyze(manager, batch, asset)
    assert asset["shape_repair_request"]["request_id"] == request_id
    assert asset["stages"]["shape_repair"]["attempt"] == 0
    asset["stages"]["shape_repair"]["state"] = "running"
    queue_analyze(manager, batch, asset)
    assert asset["shape_repair_request"]["request_id"] == request_id
    asset["stages"]["shape_repair"]["state"] = "queued"
    with pytest.raises(Codex3DError, match="Idempotency"):
        manager.create_shape_repair_attempt(
            batch["batch_id"], asset["asset_id"], {"method": "analyze"},
            asset["shape_repair_request"]["source_sha256"], "A different explanation",
            "analysis-1", expected_shape_attempt=1,
        )


def test_repeated_quality_recipe_requires_changed_cause(manager):
    batch, asset = seed_shapes(manager)
    queue_analyze(manager, batch, asset)
    record, _ = run_attempt(manager, batch, asset)
    manager.review_asset(batch["batch_id"], asset["asset_id"], "shape_repair", 1,
                         False, "Semantic design failure", review_paths(record))
    with pytest.raises(Codex3DError, match="already exists"):
        queue_analyze(manager, batch, asset, key="blind-repeat")


def test_stale_source_request_and_unknown_recipe_rejected_without_mutation(manager):
    batch, asset = seed_shapes(manager)
    with pytest.raises(Codex3DError, match="hash"):
        manager.create_shape_repair_attempt(batch["batch_id"], asset["asset_id"],
                                           {"method": "analyze"}, "0" * 64, "Inspect", "stale")
    with pytest.raises(ValueError, match="Unknown repair"):
        manager.create_shape_repair_attempt(batch["batch_id"], asset["asset_id"],
                                           {"method": "arbitrary_script"}, "0" * 64, "Inspect", "bad")
    assert "shape_repair_request" not in asset


def import_recipe():
    return {"method": "import_candidate", "policy": {"roi": {"min": [-1, -1, -1], "max": [1, 1, 1]}}}


def test_import_staged_immutably_before_worker_and_does_not_auto_approve(manager, tmp_path):
    batch, asset = seed_shapes(manager)
    candidate = tmp_path / "edited.glb"
    candidate.write_bytes(b"candidate edited outside managed artifacts")
    digest = sha256_file(candidate)
    manager.submit_shape_repair_candidate(
        batch["batch_id"], asset["asset_id"], str(candidate), digest,
        sha256_file(Path(asset["stages"]["shape"]["output"])), import_recipe(),
        "Targeted edit in selected region", "import-1",
    )
    assert batch["paused"] and batch["state"] == "paused"
    record, directory = run_attempt(manager, batch, asset)
    seen = manager.processor.seen[-1]
    staged = Path(seen["import_path"])
    assert staged.is_relative_to(directory) and staged != candidate
    assert sha256_file(staged) == digest
    assert record["state"] == "awaiting_review"
    assert "accepted_master" not in asset


def test_import_change_between_queue_and_execution_fails_before_worker(manager, tmp_path):
    batch, asset = seed_shapes(manager)
    path = tmp_path / "import.glb"
    path.write_bytes(b"expected candidate")
    manager.submit_shape_repair_candidate(
        batch["batch_id"], asset["asset_id"], str(path), sha256_file(path),
        sha256_file(Path(asset["stages"]["shape"]["output"])), import_recipe(),
        "Reviewed local correction", "import-change",
    )
    path.write_bytes(b"different candidate")
    record, _ = run_attempt(manager, batch, asset)
    assert record["state"] == "needs_repair"
    assert record["error"]["code"] == "STALE_REPAIR"
    assert manager.processor.seen == []


def test_import_outside_allowed_roots_rejected(manager, tmp_path):
    batch, asset = seed_shapes(manager)
    path = tmp_path / "outside.glb"
    path.write_bytes(b"candidate")
    manager.settings = replace(manager.settings, allowed_input_roots=(tmp_path / "allowed",))
    with pytest.raises(ValueError, match="allowed root"):
        manager.submit_shape_repair_candidate(
            batch["batch_id"], asset["asset_id"], str(path), sha256_file(path),
            sha256_file(Path(asset["stages"]["shape"]["output"])), import_recipe(), "Edit", "outside",
        )


def test_import_limit_matches_worker_256_mib_before_hashing_or_staging(manager, tmp_path, monkeypatch):
    batch, asset = seed_shapes(manager)
    path = tmp_path / "too-large.glb"
    path.write_bytes(b"size metadata fixture")
    original_stat = Path.stat

    def oversized_stat(value, *args, **kwargs):
        result = original_stat(value, *args, **kwargs)
        if value == path:
            fields = list(result)
            fields[6] = 256 * 1024**2 + 1
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(Path, "stat", oversized_stat)
    with pytest.raises(ValueError, match="at most 256 MiB"):
        manager.submit_shape_repair_candidate(
            batch["batch_id"], asset["asset_id"], str(path), "0" * 64,
            sha256_file(Path(asset["stages"]["shape"]["output"])), import_recipe(), "Edit", "oversize",
        )
    assert "shape_repair_request" not in asset
    assert manager.processor.seen == []


def test_reference_rewind_requeues_only_changed_asset_and_preserves_other_rejections(manager):
    batch, asset = seed_shapes(manager, count=10)
    batch.update(paused=False, state="awaiting_agent")
    for other in batch["assets"][1:]:
        other["stages"]["shape"]["state"] = "needs_repair"
    preserved = copy.deepcopy([other["stages"]["shape"] for other in batch["assets"][1:]])
    manager.retry_asset(batch["batch_id"], asset["asset_id"], "references", {}, "Correct fixed attachment view")
    asset["stages"]["references"].update(approved=True, state="approved")
    manager._queue_phase(batch, "shape")
    assert asset["stages"]["shape"]["state"] == "queued"
    assert [other["stages"]["shape"] for other in batch["assets"][1:]] == preserved
    assert "accepted_master" not in asset


def test_migration_preserves_paused_history_and_never_certifies_old_master(manager):
    batch, asset = seed_shapes(manager)
    batch.pop("schema_version")
    asset["stages"].pop("shape_repair")
    batch["stage"] = "paint"
    raw = copy.deepcopy(asset["stages"]["shape"])
    assert migrate_repair_stage(batch)
    assert batch["paused"] and batch["state"] == "paused"
    assert batch["stage"] == "shape_repair"
    assert asset["stages"]["shape"] == raw
    assert asset["stages"]["shape_repair"]["approved"] is False
    assert "accepted_master" not in asset
    assert not migrate_repair_stage(batch)


@pytest.mark.parametrize("change,has_source_hash,preserve", [
    ({"mobile_target_quads": 12000}, True, True),
    ({"browser_triangle_budget": 18000}, True, True),
    ({"full_game_target_quads": 40000}, True, False),
    ({"sharp_edge": 80}, True, False),
    ({"mobile_target_quads": 12000}, False, False),
])
def test_lower_profile_retry_preserves_only_proven_unchanged_full_game_paint(
    manager, change, has_source_hash, preserve,
):
    batch, asset = seed_shapes(manager)
    batch.update(stage="finish", paused=False, state="awaiting_agent")
    paint = asset["stages"]["paint"]
    paint.update(approved=True, state="approved", attempt=1, output="retained-paint.glb")
    if has_source_hash:
        paint["source_profile_sha256"] = "1" * 64
    before = copy.deepcopy(paint)
    asset["stages"]["finish"].update(approved=True, state="approved", attempt=1)
    manager.retry_asset(batch["batch_id"], asset["asset_id"], "remesh", change, "Revise profile geometry")
    if preserve:
        assert asset["stages"]["paint"] == before
    else:
        assert asset["stages"]["paint"]["approved"] is False
        assert asset["stages"]["paint"]["stale"]
    assert asset["stages"]["finish"]["state"] == "pending"
    assert asset["stages"]["finish"]["approved"] is False


def test_capabilities_report_configured_worker_metadata_without_import(manager, tmp_path):
    worker = tmp_path / "worker" / "Scripts" / "python.exe"
    worker.parent.mkdir(parents=True)
    worker.write_bytes(b"never executed")
    packages = worker.parent.parent / "Lib" / "site-packages"
    for name in ("numpy", "scipy", "trimesh", "pymeshlab"):
        package = packages / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("raise AssertionError('Capability discovery imported a worker package')\n")
        distribution = packages / f"{name}-1.0.dist-info"
        distribution.mkdir(parents=True)
        (distribution / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n")
    extra = tmp_path / "empty-repair-packages"
    extra.mkdir()
    manager.settings = replace(manager.settings, python_exe=worker, repair_package_dir=extra)
    status = manager.get_shape_repair_capabilities()
    assert status["full_gate_discoverable"]
    assert status["packages"]["trimesh"]["version"] == "1.0"
    assert status["packages"]["trimesh"]["runtime_qualified"] is False
    assert not status["methods"]["foundation_union"]["discoverable"]


def test_capabilities_resolve_ordered_worker_roots_once_and_preserve_gpu_qualification(
    manager, tmp_path, monkeypatch,
):
    worker = tmp_path / "isolated-worker" / "Scripts" / "python.exe"
    worker.parent.mkdir(parents=True)
    worker.write_bytes(b"worker executable metadata fixture")
    extra = tmp_path / "repair-native-packages"
    extra.mkdir()
    linux_packages = worker.parent.parent / "lib" / "python3.12" / "site-packages"
    linux_packages.mkdir(parents=True)
    manager.settings = replace(manager.settings, python_exe=worker, repair_package_dir=extra)
    calls = []
    gpu = {"backend": "cumesh", "discoverable": False, "runtime_qualified": False,
           "experimental_runnable": False, "production_ready": False,
           "reason": "Native CUDA and common worker proof are not present"}

    def resolve(*, package_paths, worker_python):
        calls.append((package_paths, worker_python))
        return {"packages": {"numpy": {"discoverable": True}}, "full_gate_discoverable": True,
                "methods": {"analyze": {"backend": "cpu", "discoverable": True},
                            "cumesh_cleanup": copy.deepcopy(gpu)}}

    monkeypatch.setattr(batch_module, "get_capabilities", resolve)
    status = manager.get_shape_repair_capabilities()
    assert calls == [([str(extra.resolve()), str((worker.parent.parent / "Lib" / "site-packages").resolve()),
                       str(linux_packages.resolve())], str(worker))]
    assert status["methods"]["cumesh_cleanup"] == gpu
    assert status["worker_python_present"]
    assert status["worker_package_paths"] == calls[0][0]
    assert "server_process_packages" not in status


def test_worker_distribution_metadata_without_package_files_is_not_discoverable(manager, tmp_path):
    worker = tmp_path / "metadata-only-worker" / "Scripts" / "python.exe"
    worker.parent.mkdir(parents=True)
    worker.write_bytes(b"worker executable metadata fixture")
    packages = worker.parent.parent / "Lib" / "site-packages"
    for name in ("numpy", "scipy", "trimesh", "pymeshlab"):
        distribution = packages / f"{name}-1.0.dist-info"
        distribution.mkdir(parents=True)
        (distribution / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n")
    manager.settings = replace(manager.settings, python_exe=worker,
                               repair_package_dir=tmp_path / "absent-extra-packages")
    status = manager.get_shape_repair_capabilities()
    assert status["full_gate_discoverable"] is False
    assert all(not item["discoverable"] for item in status["packages"].values())


@pytest.mark.parametrize("method_changes,full_gate,worker_present,recommended", [
    ({}, True, True, True),
    ({"production_ready": False, "experimental_runnable": True}, True, True, False),
    ({"runtime_qualified": False}, True, True, False),
    ({"discoverable": False}, True, True, False),
    ({"backend": "cpu"}, True, True, False),
    ({}, False, True, False),
    ({}, True, False, False),
])
def test_agent_gpu_recommendations_require_current_common_worker_qualification(
    manager, monkeypatch, method_changes, full_gate, worker_present, recommended,
):
    batch, asset = seed_shapes(manager, count=2)
    batch.update(paused=False, state="awaiting_agent")
    for current in batch["assets"]:
        current["stages"]["shape_repair"].update(state="needs_repair")
    method = {"backend": "cumesh", "discoverable": True, "runtime_qualified": True,
              "production_ready": True, **method_changes}
    calls = []

    def capabilities():
        calls.append(True)
        return {"full_gate_discoverable": full_gate, "worker_python_present": worker_present,
                "methods": {"cumesh_cleanup": method}}

    monkeypatch.setattr(manager, "get_shape_repair_capabilities", capabilities)
    before = copy.deepcopy(batch)
    work = manager.get_agent_work(batch["batch_id"])["work"]
    assert calls == [True]  # One metadata resolution for the whole actionable queue.
    assert len(work) == 2
    for item in work:
        assert item["automatic_cpu_fallback"] is False
        assert item.get("qualified_gpu_repair_options", []) == (["cumesh_cleanup"] if recommended else [])
        assert ("Qualified GPU repair options" in item["instruction"]) is recommended
    assert batch == before
    assert manager.processor.seen == []


def test_gpu_discovery_failure_does_not_hide_agent_work_or_schedule_a_fallback(manager, monkeypatch):
    batch, asset = seed_shapes(manager)
    batch.update(paused=False, state="awaiting_agent")
    asset["stages"]["shape_repair"]["state"] = "needs_repair"

    def unavailable():
        raise OSError("Qualification proof is unreadable")

    monkeypatch.setattr(manager, "get_shape_repair_capabilities", unavailable)
    item = manager.get_agent_work(batch["batch_id"])["work"][0]
    assert item["action"] == "inspect_repair_retry"
    assert "qualified_gpu_repair_options" not in item
    assert item["gpu_capability_discovery_error"]["type"] == "OSError"
    assert item["automatic_cpu_fallback"] is False
    assert manager.processor.seen == []


def test_review_work_does_not_probe_or_recommend_another_gpu_recipe(manager, monkeypatch):
    batch, asset = seed_shapes(manager)
    batch.update(paused=False, state="awaiting_agent")
    asset["stages"]["shape_repair"]["state"] = "awaiting_review"
    monkeypatch.setattr(manager, "get_shape_repair_capabilities",
                        lambda: pytest.fail("Review needs current evidence, not backend recommendations"))
    item = manager.get_agent_work(batch["batch_id"])["work"][0]
    assert item["action"] == "inspect_and_review"
    assert "qualified_gpu_repair_options" not in item


def test_stale_textured_export_does_not_win_current_profile_view(manager, monkeypatch):
    batch, asset = seed_shapes(manager)
    root = manager.root / batch["batch_id"] / asset["asset_id"]
    finish = root / "finish" / "attempt-0001"
    finish.mkdir(parents=True)
    for name in ("master", "full_game"):
        (finish / f"{name}.glb").write_bytes(b"old texture")
    remesh = root / "remesh" / "attempt-0002" / "profiles" / "full_game"
    remesh.mkdir(parents=True)
    fresh = remesh / "remeshed.glb"
    fresh.write_bytes(b"fresh geometry")
    asset["stages"]["finish"].update(output=str(finish / "master.glb"), stale=True, state="pending")
    asset["stages"]["remesh"].update(state="awaiting_review", stale=False)
    monkeypatch.setattr(manager, "_model_metadata", lambda path: {
        "triangle_count": 4, "vertex_count": 4, "textured": "finish" in path.parts,
    } if path.is_file() else None)
    models = manager._viewer_models(batch, asset, {"full_game": {"output": str(fresh)}})
    current = next(item for item in models if item["id"] == "full_game")
    assert current["current"] and not current["textured"]
    assert any(item["stale"] and item["textured"] for item in models)
