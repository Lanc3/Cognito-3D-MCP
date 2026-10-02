"""Batch barriers and repair/restart behavior without any model or GPU work."""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from codex_3d_mcp.errors import Codex3DError, GenerationCancelled
from codex_3d_mcp.hunyuan_mv import batch as batch_module
from codex_3d_mcp.hunyuan_mv.batch import REPAIR_VIEWS, STAGES, VIEWS, HunyuanBatchManager
from codex_3d_mcp.hunyuan_mv.config import HunyuanMVSettings
from codex_3d_mcp.hunyuan_mv.repair_contract import REPAIR_GATE_VERSION
from codex_3d_mcp.trellis.artifacts import sha256_file


class FakeProcessor:
    """Record execution, enforce unload boundaries and optionally block one phase."""

    def __init__(self):
        self.events = []
        self.calls = Counter()
        self.failures = set()
        self.loaded_stage = None
        self.active = 0
        self.max_active = 0
        self.violations = []
        self.block_stage = None
        self.blocked = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def _enter(self, kind, stage, asset_id):
        with self._lock:
            self.active += 1
            self.max_active = max(self.active, self.max_active)
            if self.active != 1:
                self.violations.append("overlapping assets or evidence")
            if kind == "run":
                if self.loaded_stage not in {None, stage}:
                    self.violations.append("phase started before prior phase unloaded")
                self.loaded_stage = stage
            elif self.loaded_stage is not None:
                self.violations.append("evidence started with a model phase still loaded")
            self.events.append((kind, stage, asset_id))

    def _leave(self):
        with self._lock:
            self.active -= 1

    def run(self, stage, asset, directory, cancel):
        asset_id = asset["asset_id"]
        self._enter("run", stage, asset_id)
        try:
            self.calls[stage, asset_id] += 1
            if self.block_stage == stage:
                self.blocked.set()
                while not self.release.wait(0.01):
                    if cancel.is_set():
                        raise GenerationCancelled("fake active operation observed cancellation")
            if cancel.is_set():
                raise GenerationCancelled("fake operation cancelled")
            output = directory / f"{stage}.glb"
            output.write_text(f"{stage}:{asset_id}:{self.calls[stage, asset_id]}")
            passed = (stage, asset_id, self.calls[stage, asset_id]) not in self.failures
            result = {"output": str(output), "gate": {"passed": passed}}
            if stage == "shape_repair":
                result["gate"].update(
                    gate_version=REPAIR_GATE_VERSION,
                    source_sha256=asset["shape_repair_request"]["source_sha256"],
                    output_sha256=sha256_file(output),
                )
                report = directory / "repair-report.json"
                report.write_text(json.dumps({"gate": result["gate"]}))
                result["repair_report"] = str(report)
            if stage == "finish":
                result["previews"] = self._preview(directory)
            return result
        finally:
            self._leave()

    def evidence(self, stage, output, directory, cancel):
        asset_id = output.read_text().split(":")[1]
        self._enter("evidence", stage, asset_id)
        try:
            if cancel.is_set():
                raise GenerationCancelled("fake evidence cancelled")
            return {"gate": {"passed": True}, "previews": self._preview(directory, stage)}
        finally:
            self._leave()

    @staticmethod
    def _preview(directory, stage=None):
        if stage == "shape_repair":
            previews = {}
            for view in REPAIR_VIEWS:
                path = directory / f"{view}.png"
                Image.new("RGB", (8, 8), "steelblue").save(path)
                previews[view] = str(path)
            return previews
        path = directory / "preview.png"
        Image.new("RGB", (8, 8), "steelblue").save(path)
        return {"front": str(path)}

    def end_stage(self):
        with self._lock:
            if self.active:
                self.violations.append("unload overlapped a running operation")
            if self.loaded_stage is not None:
                self.events.append(("end", self.loaded_stage, None))
            self.loaded_stage = None

    def close(self):
        self.release.set()


def make_settings(tmp_path):
    return HunyuanMVSettings(
        base_dir=tmp_path,
        server_name="batch-test",
        server_version="test",
        output_dir=tmp_path / "outputs",
        database_path=tmp_path / "outputs" / "jobs.sqlite3",
        python_exe=tmp_path / "python.exe",
        upstream_dir=tmp_path / "upstream",
        model_cache_dir=tmp_path / "models",
        blender_exe=tmp_path / "blender.exe",
        gltf_validator=tmp_path / "validator.exe",
        allowed_input_roots=(tmp_path,),
        gpu_lock_path=tmp_path / "gpu.lock",
        minimum_image_size=64,
        minimum_free_bytes=1,
        dashboard_auto_open=False,
        test_mode=True,
    )


@pytest.fixture
def manager_factory(tmp_path):
    managers = []

    def create(processor=None):
        manager = HunyuanBatchManager(make_settings(tmp_path), processor or FakeProcessor())
        managers.append(manager)
        return manager

    yield create
    for manager in reversed(managers):
        manager.close()


@pytest.fixture
def views(tmp_path):
    result = {}
    for index, name in enumerate(VIEWS):
        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((12, 8, 51, 55), fill=(50 + index * 30, 110, 160, 255))
        draw.rectangle((25 + index, 24, 29 + index, 29), fill=(220, 160, 60, 255))
        path = tmp_path / f"{name}.png"
        image.save(path)
        result[name] = str(path)
    return result


def create_batch(manager, count=2):
    return manager.create_batch(
        "Synthetic assets", [{"name": f"Asset {i}", "prompt": f"Object {i}"} for i in range(count)]
    )


def wait_status(manager, batch_id, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.get_batch_status(batch_id)
        if predicate(status):
            return status
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for batch state: {manager.get_batch_status(batch_id)}")


def stage_record(status, asset_id, stage):
    return next(a for a in status["assets"] if a["asset_id"] == asset_id)["stages"][stage]


def wait_phase_review(manager, batch_id, stage):
    return wait_status(
        manager, batch_id,
        lambda status: status["stage"] == stage and all(
            asset["stages"][stage]["state"] in {"awaiting_review", "approved", "needs_repair"}
            for asset in status["assets"]
        ),
    )


def review(manager, batch_id, asset_id, stage, approved=True):
    record = stage_record(manager.get_batch_status(batch_id), asset_id, stage)
    paths = list(record["previews"].values())
    if stage == "shape_repair":
        paths.append(record["repair_report"])
    return manager.review_asset(
        batch_id, asset_id, stage, record["attempt"], approved,
        "Inspected current candidate silhouettes, framing and surface evidence.",
        paths,
    )


def prepare_all(manager, status, views):
    for asset in status["assets"]:
        manager.submit_references(status["batch_id"], asset["asset_id"], views)
        review(manager, status["batch_id"], asset["asset_id"], "references")


def test_entire_batch_crosses_each_phase_barrier_with_one_active_asset(manager_factory, views):
    manager = manager_factory()
    processor = manager.processor
    initial = create_batch(manager, 3)
    batch_id = initial["batch_id"]
    ids = [asset["asset_id"] for asset in initial["assets"]]
    for asset_id in ids[:-1]:
        manager.submit_references(batch_id, asset_id, views)
        review(manager, batch_id, asset_id, "references")
    assert not processor.calls
    assert manager.get_batch_status(batch_id)["stage"] == "references"
    manager.submit_references(batch_id, ids[-1], views)
    review(manager, batch_id, ids[-1], "references")

    # Raw inference advances to diagnosis; it is never implicitly approved.
    for stage in STAGES[2:]:
        status = wait_phase_review(manager, batch_id, stage)
        assert all(
            asset["stages"][stage]["state"] == "awaiting_review" for asset in status["assets"]
        )
        assert [processor.calls[stage, asset_id] for asset_id in ids] == [1, 1, 1]
        for asset_id in ids[:-1]:
            review(manager, batch_id, asset_id, stage)
        assert manager.get_batch_status(batch_id)["stage"] == stage
        if stage != "finish":
            following = STAGES[STAGES.index(stage) + 1]
            assert sum(processor.calls[following, asset_id] for asset_id in ids) == 0
        review(manager, batch_id, ids[-1], stage)

    wait_status(manager, batch_id, lambda status: status["state"] == "completed")
    assert all(processor.calls["shape", asset_id] == 1 for asset_id in ids)
    phase_events = [event for event in processor.events if event[0] in {"run", "evidence"}]
    expected = []
    for stage in STAGES[1:]:
        expected.extend(("run", stage, asset_id) for asset_id in ids)
        if stage != "finish":
            expected.extend(("evidence", stage, asset_id) for asset_id in ids)
    assert phase_events == expected
    assert processor.max_active == 1
    assert processor.violations == []


def test_failed_machine_gate_awaits_agent_and_cannot_be_overridden(manager_factory, views):
    manager = manager_factory()
    initial = create_batch(manager)
    batch_id = initial["batch_id"]
    failed, successful = [asset["asset_id"] for asset in initial["assets"]]
    manager.processor.failures.add(("shape", failed, 1))
    prepare_all(manager, initial, views)
    status = wait_phase_review(manager, batch_id, "shape")
    record = stage_record(status, failed, "shape")
    assert record["state"] == "needs_repair"
    assert record["gate"]["passed"] is False
    assert manager.get_agent_work(batch_id)["work"][0]["action"] == "inspect_repair_retry"
    with pytest.raises(Codex3DError) as caught:
        manager.review_asset(
            batch_id, failed, "shape", 1, True, "Try to force approval", [record["output"]]
        )
    assert caught.value.code == "GATE_FAILED"
    review(manager, batch_id, successful, "shape")
    assert manager.get_batch_status(batch_id)["stage"] == "shape"
    assert not any(stage == "remesh" for stage, _ in manager.processor.calls)

    manager.retry_asset(batch_id, failed, "shape", {"seed": 43}, "Change seed for missing geometry")
    status = wait_phase_review(manager, batch_id, "shape_repair")
    assert stage_record(status, failed, "shape")["attempt"] == 2
    assert manager.processor.calls["shape", successful] == 1
    assert manager.processor.calls["shape", failed] == 2
    for asset_id in (failed, successful):
        review(manager, batch_id, asset_id, "shape_repair")
    wait_phase_review(manager, batch_id, "remesh")
    assert manager.processor.violations == []


def test_rejected_attempt_history_and_files_survive_retry_stale_reviews_fail(
    manager_factory, views,
):
    manager = manager_factory()
    initial = create_batch(manager, 1)
    batch_id, asset_id = initial["batch_id"], initial["assets"][0]["asset_id"]
    prepare_all(manager, initial, views)
    status = wait_phase_review(manager, batch_id, "shape_repair")
    old = stage_record(status, asset_id, "shape_repair")
    old_preview = old["previews"]["front"]
    old_bytes = Path(old["output"]).read_bytes()
    review(manager, batch_id, asset_id, "shape_repair", approved=False)
    manager.create_shape_repair_attempt(
        batch_id, asset_id,
        {"method": "analyze", "policy": {"roi": {"min": [-1, -1, -1], "max": [1, 1, 1]}}},
        sha256_file(Path(stage_record(status, asset_id, "shape")["output"])),
        "Reinspect diagnosed region", "region-review",
    )
    status = wait_phase_review(manager, batch_id, "shape_repair")
    record = stage_record(status, asset_id, "shape_repair")
    assert record["attempt"] == 2
    assert record["history"][0]["attempt"] == 1
    assert record["history"][0]["state"] == "needs_repair"
    assert record["history"][0]["error"]["code"] == "AGENT_REJECTED"
    assert Path(old["output"]).read_bytes() == old_bytes
    assert Path(old_preview).exists()
    with pytest.raises(Codex3DError) as caught:
        manager.review_asset(
            batch_id, asset_id, "shape_repair", 1, True, "Old candidate review", [old_preview]
        )
    assert caught.value.code == "STALE_REVIEW"
    with pytest.raises(ValueError, match="current immutable attempt"):
        manager.review_asset(
            batch_id, asset_id, "shape_repair", 2, True, "Old image mislabelled", [old_preview]
        )
    review(manager, batch_id, asset_id, "shape_repair")
    wait_phase_review(manager, batch_id, "remesh")


def test_earlier_phase_repair_preserves_successful_assets_downstream(manager_factory, views):
    manager = manager_factory()
    initial = create_batch(manager)
    batch_id = initial["batch_id"]
    repaired, successful = [asset["asset_id"] for asset in initial["assets"]]
    prepare_all(manager, initial, views)
    wait_phase_review(manager, batch_id, "shape_repair")
    for asset in initial["assets"]:
        review(manager, batch_id, asset["asset_id"], "shape_repair")
    wait_phase_review(manager, batch_id, "remesh")
    review(manager, batch_id, successful, "remesh")
    review(manager, batch_id, repaired, "remesh", approved=False)
    manager.retry_asset(batch_id, repaired, "shape", {"seed": 44}, "Topology issue starts in shape")
    wait_phase_review(manager, batch_id, "shape_repair")
    review(manager, batch_id, repaired, "shape_repair")
    status = wait_phase_review(manager, batch_id, "remesh")
    assert manager.processor.calls["shape", successful] == 1
    assert manager.processor.calls["remesh", successful] == 1
    assert manager.processor.calls["shape", repaired] == 2
    assert manager.processor.calls["remesh", repaired] == 2
    assert stage_record(status, successful, "remesh")["approved"] is True


def test_queued_batch_cannot_prepare_images_before_it_is_active(
    manager_factory, views, monkeypatch,
):
    manager = manager_factory()
    first = create_batch(manager, 1)
    second = create_batch(manager, 1)
    prepared = []
    original = batch_module.prepare_reference

    def track(*args, **kwargs):
        prepared.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(batch_module, "prepare_reference", track)
    second_id = second["assets"][0]["asset_id"]
    with pytest.raises(Codex3DError) as caught:
        manager.submit_references(second["batch_id"], second_id, views)
    assert caught.value.code == "BATCH_QUEUED"
    assert prepared == []
    assert manager.get_agent_work(second["batch_id"])["work"] == []
    assert not (manager.root / second["batch_id"] / second_id).exists()
    manager.cancel_batch(first["batch_id"])
    manager.submit_references(second["batch_id"], second_id, views)
    assert len(prepared) == 4


def test_pause_stops_at_asset_boundary_then_resume_continues(manager_factory, views):
    processor = FakeProcessor()
    processor.block_stage = "shape"
    manager = manager_factory(processor)
    initial = create_batch(manager)
    batch_id = initial["batch_id"]
    first, second = [asset["asset_id"] for asset in initial["assets"]]
    prepare_all(manager, initial, views)
    assert processor.blocked.wait(2)
    running = manager.get_batch_status(batch_id)
    assert running["active_asset_id"] == first
    assert running["active_resource"] == "GPU shape"
    manager.pause_batch(batch_id)
    processor.release.set()
    status = wait_status(
        manager, batch_id,
        lambda s: stage_record(s, first, "shape")["state"] == "needs_evidence",
    )
    assert status["paused"] is True
    assert processor.calls["shape", second] == 0
    assert manager.get_agent_work(batch_id)["work"] == []
    manager.resume_batch(batch_id)
    wait_phase_review(manager, batch_id, "shape_repair")
    assert processor.calls["shape", first] == processor.calls["shape", second] == 1
    assert processor.violations == []


def test_cancel_active_operation_retains_attempt_and_stays_cancelled_after_restart(
    manager_factory, views,
):
    processor = FakeProcessor()
    processor.block_stage = "shape"
    manager = manager_factory(processor)
    initial = create_batch(manager)
    batch_id = initial["batch_id"]
    first, second = [asset["asset_id"] for asset in initial["assets"]]
    prepare_all(manager, initial, views)
    assert processor.blocked.wait(2)
    manager.cancel_batch(batch_id)
    status = wait_status(
        manager, batch_id,
        lambda s: stage_record(s, first, "shape")["state"] == "needs_repair",
    )
    assert status["state"] == "cancelled"
    assert processor.calls["shape", second] == 0
    assert stage_record(status, first, "shape")["error"]["code"] == "CANCELLED"
    assert list((manager.root / batch_id / first / "shape").glob("attempt-0001/gate.json"))
    manager.close()
    restarted = manager_factory()
    assert restarted.get_batch_status(batch_id)["state"] == "cancelled"
    assert restarted.resume_batch(batch_id)["state"] == "cancelled"
    assert restarted.get_agent_work(batch_id)["work"] == []


def test_restart_preserves_approved_assets_and_requires_retry_for_interrupted_attempt(
    manager_factory, views,
):
    manager = manager_factory()
    initial = create_batch(manager)
    batch_id = initial["batch_id"]
    successful, interrupted = [asset["asset_id"] for asset in initial["assets"]]
    prepare_all(manager, initial, views)
    wait_phase_review(manager, batch_id, "shape_repair")
    review(manager, batch_id, successful, "shape_repair")
    manager.pause_batch(batch_id)
    manager.close()
    manifest = manager.root / batch_id / "batch.json"
    persisted = json.loads(manifest.read_text())
    persisted["state"] = "running"
    persisted["paused"] = False
    stage_record(persisted, interrupted, "shape_repair")["state"] = "running"
    manifest.write_text(json.dumps(persisted))

    restarted = manager_factory()
    status = restarted.get_batch_status(batch_id)
    assert status["paused"] is True and status["state"] == "interrupted"
    assert stage_record(status, successful, "shape_repair")["approved"] is True
    assert stage_record(status, interrupted, "shape_repair")["error"]["code"] == "INTERRUPTED"
    assert restarted.get_agent_work(batch_id)["work"] == []
    restarted.resume_batch(batch_id)
    assert restarted.get_agent_work(batch_id)["work"][0]["action"] == "inspect_repair_retry"
    restarted.create_shape_repair_attempt(
        batch_id, interrupted, {"method": "analyze"},
        sha256_file(Path(stage_record(status, interrupted, "shape")["output"])),
        "Repeat interrupted operation", "restart-operation",
    )
    status = wait_phase_review(restarted, batch_id, "shape_repair")
    assert stage_record(status, interrupted, "shape_repair")["attempt"] == 2
    assert restarted.processor.calls["shape_repair", successful] == 0
    assert restarted.processor.calls["shape_repair", interrupted] == 1


def test_restart_retains_batch_submission_order_not_uuid_sort(manager_factory, monkeypatch):
    ids = iter(["f" * 32, "1" * 32, "a" * 32, "2" * 32])
    monkeypatch.setattr(batch_module.uuid, "uuid4", lambda: SimpleNamespace(hex=next(ids)))
    manager = manager_factory()
    first = create_batch(manager, 1)
    second = create_batch(manager, 1)
    manager.close()
    restarted = manager_factory()
    assert restarted.queue_snapshot()["active_batch_id"] == first["batch_id"]
    restarted.cancel_batch(first["batch_id"])
    assert restarted.queue_snapshot()["active_batch_id"] == second["batch_id"]


def test_hundred_asset_manifest_does_not_inherit_four_job_limit(manager_factory):
    manager = manager_factory()
    status = create_batch(manager, 100)
    assert manager.settings.max_queued_jobs == 4
    assert len(status["assets"]) == 100
    assert len(manager.get_agent_work(status["batch_id"], limit=100)["work"]) == 100
    persisted = json.loads((manager.root / status["batch_id"] / "batch.json").read_text())
    assert len(persisted["assets"]) == 100
    assert manager.processor.calls == {}
    assert not any(path.is_dir() for path in (manager.root / status["batch_id"]).iterdir())


def test_profile_quad_retry_derives_budget_unless_explicitly_overridden(manager_factory):
    manager = manager_factory()
    status = create_batch(manager, 1)
    batch_id, asset_id = status["batch_id"], status["assets"][0]["asset_id"]
    assert status["assets"][0]["params"]["full_game_target_quads"] == 50000
    status = manager.retry_asset(
        batch_id, asset_id, "references", {"mobile_target_quads": 20000}, "Revise mobile detail"
    )
    assert status["assets"][0]["params"]["mobile_triangle_budget"] == 40000
    status = manager.retry_asset(
        batch_id, asset_id, "references",
        {"mobile_target_quads": 18000, "mobile_triangle_budget": 45000}, "Explicit export ceiling",
    )
    assert status["assets"][0]["params"]["mobile_triangle_budget"] == 45000
    snapshot = manager.queue_snapshot()["batches"][0]["assets"][0]["profiles"]
    assert snapshot["mobile"]["target_quads"] == 18000
    assert snapshot["mobile"]["actual_triangles"] is None


@pytest.mark.parametrize("changes", [
    {"mobile_target_quads": 1.5}, {"browser_target_quads": True},
    {"full_game_triangle_budget": float("nan")}, {"mobile_triangle_budget": 300001},
])
def test_profile_retry_rejects_invalid_counts_without_changing_params(manager_factory, changes):
    manager = manager_factory()
    status = create_batch(manager, 1)
    batch_id, asset_id = status["batch_id"], status["assets"][0]["asset_id"]
    before = status["assets"][0]["params"]
    with pytest.raises(ValueError):
        manager.retry_asset(batch_id, asset_id, "references", changes, "Invalid count test")
    assert manager.get_batch_status(batch_id)["assets"][0]["params"] == before


def test_failed_reference_set_must_be_repaired_before_any_model_phase(manager_factory, views):
    manager = manager_factory()
    initial = create_batch(manager, 1)
    batch_id, asset_id = initial["batch_id"], initial["assets"][0]["asset_id"]
    duplicated = dict.fromkeys(VIEWS, views["front"])
    status = manager.submit_references(batch_id, asset_id, duplicated)
    record = stage_record(status, asset_id, "references")
    assert record["state"] == "needs_repair"
    assert record["gate"]["passed"] is False
    assert "duplicates" in record["error"]["message"]
    assert manager.processor.calls == {}
    assert manager.get_agent_work(batch_id)["work"][0]["action"] == "inspect_repair_retry"
    status = manager.submit_references(batch_id, asset_id, views)
    repaired = stage_record(status, asset_id, "references")
    assert repaired["attempt"] == 2
    assert repaired["history"][0]["gate"]["passed"] is False
    assert len(repaired["prepared_views"]) == 4
    assert manager.processor.calls == {}
    review(manager, batch_id, asset_id, "references")
    wait_phase_review(manager, batch_id, "shape_repair")


def test_repeated_failures_never_auto_approve_or_exhaust_a_retry_counter(manager_factory, views):
    manager = manager_factory()
    initial = create_batch(manager, 1)
    batch_id, asset_id = initial["batch_id"], initial["assets"][0]["asset_id"]
    manager.processor.failures.update(("shape", asset_id, attempt) for attempt in range(1, 6))
    prepare_all(manager, initial, views)
    for attempt in range(1, 6):
        status = wait_phase_review(manager, batch_id, "shape")
        record = stage_record(status, asset_id, "shape")
        assert record["attempt"] == attempt
        assert record["state"] == "needs_repair"
        assert record["approved"] is False
        assert manager.processor.calls["remesh", asset_id] == 0
        manager.retry_asset(
            batch_id, asset_id, "shape", {"seed": 42 + attempt},
            f"Repair candidate {attempt} after inspecting failed gate",
        )
    status = wait_phase_review(manager, batch_id, "shape_repair")
    record = stage_record(status, asset_id, "shape")
    assert record["attempt"] == 6
    assert record["state"] == "awaiting_review"
    assert record["approved"] is False
    assert len(record["history"]) == 5
    review(manager, batch_id, asset_id, "shape_repair")
    wait_phase_review(manager, batch_id, "remesh")


def _viewer_glb(path, triangles, *, textured=False, unused_texture=False):
    import struct

    path.parent.mkdir(parents=True, exist_ok=True)
    primitive = {"indices": 0, "attributes": {"POSITION": 1}}
    document = {
        "asset": {"version": "2.0"},
        "accessors": [{"count": triangles * 3}, {"count": triangles + 2}],
        "meshes": [{"primitives": [primitive]}],
    }
    if textured or unused_texture:
        document.update(
            materials=[{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
            textures=[{"source": 0}], images=[{"bufferView": 0, "mimeType": "image/png"}],
            bufferViews=[{"buffer": 0, "byteLength": 4}],
        )
        if textured:
            primitive["material"] = 0
    content = json.dumps(document).encode()
    content += b" " * (-len(content) % 4)
    path.write_bytes(struct.pack("<4sIIII", b"glTF", 2, 20 + len(content), len(content), 0x4E4F534A)
                     + content)


@pytest.fixture
def viewer_asset(manager_factory):
    manager = manager_factory()
    initial = create_batch(manager, 1)
    manager.pause_batch(initial["batch_id"])
    batch = manager._batch(initial["batch_id"])
    asset = batch["assets"][0]
    root = manager.root / batch["batch_id"] / asset["asset_id"]
    return manager, batch, asset, root


def test_viewer_profiles_prefer_finished_textures_and_report_real_glb_counts(viewer_asset):
    manager, batch, asset, root = viewer_asset
    shape = root / "shape" / "shape.glb"
    _viewer_glb(shape, 120)
    asset["stages"]["shape"].update(output=str(shape), approved=True, state="approved")
    profiles = {}
    for index, name in enumerate(("full_game", "mobile", "browser")):
        output = root / "remesh" / name / "remeshed.glb"
        _viewer_glb(output, 80 - index * 20)
        profiles[name] = {
            "output": str(output), "target_quads": 50000 - index * 10000,
            "geometry": {"native_quads": 40 - index * 10, "cleaned_quads": 39 - index * 10},
        }
    asset["stages"]["remesh"]["profiles"] = profiles
    asset["stages"]["remesh"]["state"] = "awaiting_review"

    def models():
        return {item["id"]: item for item in manager.queue_snapshot()["batches"][0]["assets"][0][
            "models"
        ]}

    before = models()
    assert set(before) == {"shape", "full_game", "mobile", "browser"}
    assert before["shape"]["triangle_count"] == 120
    assert before["shape"]["quad_count"] is None
    assert before["shape"]["texture_state"] == "untextured"
    assert not any(model["textured"] for model in before.values())
    paint = root / "paint" / "textured.glb"
    _viewer_glb(paint, 76, textured=True)
    asset["stages"]["paint"].update(output=str(paint), approved=True, state="approved")
    painted = models()
    assert painted["full_game"]["stage"] == "paint" and painted["full_game"]["textured"]
    assert painted["full_game"]["triangle_count"] == 76
    assert not painted["mobile"]["textured"]
    finish = root / "finish" / "attempt-0001"
    for index, name in enumerate(("full_game", "mobile", "browser")):
        _viewer_glb(finish / f"{name}.glb", 70 - index * 20, textured=True)
    for alias in ("master", "game", "lod1", "lod2"):
        _viewer_glb(finish / f"{alias}.glb", 70, textured=True)
    asset["stages"]["finish"].update(output=str(finish / "master.glb"), approved=True, state="approved")
    finished = models()
    assert set(finished) == set(before)
    assert finished["full_game"]["stage"] == "finish"
    assert finished["full_game"]["triangle_count"] == 70
    assert finished["full_game"]["vertex_count"] == 72
    assert finished["full_game"]["quad_count"] == 39
    assert finished["full_game"]["native_quad_count"] == 40
    assert "before GLB triangulation" in finished["full_game"]["quad_count_source"]
    assert all(model["source_id"] == ("shape" if name == "shape" else "accepted_master")
               for name, model in finished.items())
    assert all(model["approved"] and model["textured"]
               for name, model in finished.items() if name != "shape")


def test_viewer_does_not_label_unused_texture_metadata_as_textured(viewer_asset):
    manager, _, asset, root = viewer_asset
    source = root / "shape.glb"
    _viewer_glb(source, 12, unused_texture=True)
    asset["stages"]["shape"]["output"] = str(source)
    model = manager.queue_snapshot()["batches"][0]["assets"][0]["models"][0]
    assert not model["textured"] and model["texture_state"] == "untextured"


def test_imported_raw_shape_alias_is_exact_and_hash_verified_without_approval(viewer_asset, tmp_path):
    import hashlib

    manager, batch, asset, root = viewer_asset
    source = tmp_path / "accepted-external" / "shape.glb"
    _viewer_glb(source, 120)
    asset["stages"]["shape"].update(
        output=str(source), approved=True, gate={"passed": True}, artifacts=[{
            "path": str(source), "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }],
    )
    model = manager.queue_snapshot()["batches"][0]["assets"][0]["models"][0]
    expected = f"/models/{batch['batch_id']}/{asset['asset_id']}/original-shape.glb"
    assert model["url"] == expected and not model["textured"]
    assert manager.resolve_model(batch["batch_id"], asset["asset_id"], "original-shape.glb") == source
    with pytest.raises(ValueError):
        manager.resolve_model(batch["batch_id"], asset["asset_id"], "../shape.glb")
    with pytest.raises(ValueError):
        manager.resolve_artifact(batch["batch_id"], asset["asset_id"], str(source))
    asset["stages"]["shape"]["approved"] = False
    assert manager.queue_snapshot()["batches"][0]["assets"][0]["models"]
    assert manager.resolve_model(batch["batch_id"], asset["asset_id"], "original-shape.glb") == source
    asset["stages"]["shape"]["approved"] = True
    original = source.read_bytes()
    source.write_bytes(original[:-1] + (b" " if original[-1:] != b" " else b"\n"))
    with pytest.raises(ValueError, match="changed after generation"):
        manager.resolve_model(batch["batch_id"], asset["asset_id"], "original-shape.glb")
    assert not source.is_relative_to(root)


def test_viewer_legacy_aliases_and_profile_files_stay_scoped(viewer_asset, tmp_path):
    manager, batch, asset, root = viewer_asset
    finish = root / "finish"
    for name in ("master", "game", "lod1", "lod2"):
        _viewer_glb(finish / f"{name}.glb", 12, textured=True)
    asset["stages"]["finish"]["output"] = str(finish / "master.glb")
    models = manager._viewer_models(batch, asset, {})
    assert {model["id"] for model in models} == {"master", "game", "lod1", "lod2"}
    outside = tmp_path / "outside.glb"
    _viewer_glb(outside, 12, textured=True)
    profiles = {"mobile": {"output": str(outside), "geometry": {"cleaned_quads": 6}}}
    assert not manager._viewer_models(batch, asset, profiles)


def test_close_cancels_active_operation_before_releasing_processor(manager_factory, views):
    processor = FakeProcessor()
    processor.block_stage = "shape"
    manager = manager_factory(processor)
    initial = create_batch(manager, 1)
    prepare_all(manager, initial, views)
    assert processor.blocked.wait(2)
    original_close = processor.close

    def close_after_join():
        assert not manager._thread.is_alive()
        assert processor.active == 0
        original_close()

    processor.close = close_after_join
    manager.close()
    assert processor.violations == []


def test_close_reports_failed_final_resource_release(manager_factory):
    manager = manager_factory()
    original_close = manager.processor.close

    def failed_release():
        raise Codex3DError("Fixture resident model remains live", code="WORKER_BUSY")

    manager.processor.close = failed_release
    try:
        with pytest.raises(Codex3DError, match="remains live"):
            manager.close()
    finally:
        manager.processor.close = original_close
