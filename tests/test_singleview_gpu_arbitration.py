"""Cross-manager GPU exclusion, unload ownership and cancellation without torch."""

import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from codex_3d_mcp.config import Settings
from codex_3d_mcp.errors import Codex3DError, GenerationCancelled
from codex_3d_mcp.inputs import StagedImage
from codex_3d_mcp.jobs import JobManager
from codex_3d_mcp.model import ImageTo3DAdapter
from codex_3d_mcp.trellis.runtime import GpuLease

PARAMS = {"texture_resolution": 1024, "foreground_ratio": 0.85,
          "remesh": "none", "target_vertex_count": None}


class BlockingAdapter:
    def __init__(self):
        self.started = Event()
        self.finish_generation = Event()
        self.unloading = Event()
        self.finish_unload = Event()
        self.finish_generation.set()
        self.finish_unload.set()
        self.unload_failure = False
        self.calls = 0

    def generate(self, image, output, **kwargs):
        self.calls += 1
        self.started.set()
        while not self.finish_generation.wait(0.01):
            if kwargs["cancel_event"].is_set():
                raise GenerationCancelled("cancelled fake generation")
        output.write_bytes(b"glTF" + b"\0" * 32)
        return {}

    def shutdown(self):
        self.unloading.set()
        assert self.finish_unload.wait(5)
        if self.unload_failure:
            raise RuntimeError("fixture GPU allocations remain")


def manager(tmp_path, adapter, *, wait=1):
    return JobManager(Settings(output_dir=tmp_path / str(id(adapter)),
                               gpu_lock_path=tmp_path / "shared-gpu.lock",
                               gpu_wait_seconds=wait), adapter)


def submit(value):
    return value.submit("asset", StagedImage(Path("fixture.png"), False, "test"), PARAMS)


def wait_terminal(value, record):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = value.get(record.job_id)
        if state.status in {"completed", "failed", "cancelled"}:
            return state
        time.sleep(0.01)
    raise AssertionError(record.as_dict())


def test_shared_lease_is_held_through_generation_and_unload(tmp_path):
    first, second = BlockingAdapter(), BlockingAdapter()
    first.finish_generation.clear()
    first.finish_unload.clear()
    owner, competitor = manager(tmp_path, first), manager(tmp_path, second, wait=0)
    try:
        original = submit(owner)
        assert first.started.wait(5)
        blocked = wait_terminal(competitor, submit(competitor))
        assert blocked.error["code"] == "GPU_BUSY"
        assert second.calls == 0
        first.finish_generation.set()
        assert first.unloading.wait(5)
        assert original.status == "running"
        blocked = wait_terminal(competitor, submit(competitor))
        assert blocked.error["code"] == "GPU_BUSY"
        first.finish_unload.set()
        assert wait_terminal(owner, original).status == "completed"
        assert wait_terminal(competitor, submit(competitor)).status == "completed"
        assert second.calls == 1
    finally:
        first.finish_generation.set()
        first.finish_unload.set()
        owner.shutdown()
        competitor.shutdown()


def test_gpu_wait_can_be_cancelled_without_loading_a_model(tmp_path):
    adapter = BlockingAdapter()
    value = manager(tmp_path, adapter, wait=30)
    lease = GpuLease(value.settings.gpu_lock_path)
    lease.acquire(timeout=1)
    try:
        job = submit(value)
        deadline = time.monotonic() + 5
        while job.stage != "waiting for GPU" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.stage == "waiting for GPU"
        value.cancel(job.job_id)
        assert wait_terminal(value, job).status == "cancelled"
        assert adapter.calls == 0
    finally:
        lease.release()
        value.shutdown()


def test_shutdown_cancels_gpu_wait_without_releasing_another_managers_lease(tmp_path):
    adapter = BlockingAdapter()
    value = manager(tmp_path, adapter, wait=30)
    lease = GpuLease(value.settings.gpu_lock_path)
    lease.acquire(timeout=1)
    try:
        job = submit(value)
        value.shutdown()
        assert not value._worker.is_alive()
        assert job.status == "cancelled"
        assert adapter.calls == 0
        challenger = GpuLease(lease.path)
        with pytest.raises(Codex3DError, match="Timed out"):
            challenger.acquire(timeout=0)
    finally:
        lease.release()
        value.shutdown()


def test_failed_unload_retains_lease_and_blocks_later_work(tmp_path):
    adapter = BlockingAdapter()
    adapter.unload_failure = True
    value = manager(tmp_path, adapter)
    try:
        job = wait_terminal(value, submit(value))
        assert job.status == "failed" and job.error["code"] == "GPU_UNLOAD_FAILED"
        challenger = GpuLease(value.settings.gpu_lock_path)
        with pytest.raises(Codex3DError, match="Timed out"):
            challenger.acquire(timeout=0)
        with pytest.raises(Codex3DError, match="unload failed"):
            submit(value)
        with pytest.raises(Codex3DError, match="unload failed"):
            value.shutdown()
        assert value._lease_owned
        assert adapter.calls == 1
    finally:
        # Explicit test-only recovery. Production requires process restart.
        adapter.unload_failure = False
        value._resource_error = None
        value.shutdown()


def test_cancel_and_shutdown_keep_lease_until_active_model_unload_finishes(tmp_path, monkeypatch):
    adapter = BlockingAdapter()
    adapter.finish_generation.clear()
    adapter.finish_unload.clear()
    value = manager(tmp_path, adapter)
    real_join = value._worker.join
    try:
        job = submit(value)
        assert adapter.started.wait(5)
        value.cancel(job.job_id)
        assert adapter.unloading.wait(5)
        challenger = GpuLease(value.settings.gpu_lock_path)
        with pytest.raises(Codex3DError, match="Timed out"):
            challenger.acquire(timeout=0)
        # Bound this test's wait; production allows the worker 30 seconds.
        monkeypatch.setattr(value._worker, "join", lambda timeout: real_join(timeout=0.01))
        with pytest.raises(Codex3DError, match="has not stopped"):
            value.shutdown()
        assert value._worker.is_alive() and value._lease_owned
        with pytest.raises(Codex3DError, match="Timed out"):
            challenger.acquire(timeout=0)
        adapter.finish_unload.set()
        real_join(timeout=5)
        assert not value._worker.is_alive()
        assert job.status == "cancelled"
        challenger.acquire(timeout=0)
        challenger.release()
    finally:
        adapter.finish_generation.set()
        adapter.finish_unload.set()
        real_join(timeout=5)
        value.shutdown()


@pytest.mark.parametrize("remaining", [(1, 0), (0, 1)])
def test_model_unload_requires_synchronized_empty_cuda_allocator(tmp_path, monkeypatch, remaining):
    events = []
    cuda = SimpleNamespace(
        synchronize=lambda: events.append("synchronize"),
        empty_cache=lambda: events.append("empty_cache"),
        memory_allocated=lambda: remaining[0], memory_reserved=lambda: remaining[1],
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", SimpleNamespace(cuda=cuda))
    adapter = ImageTo3DAdapter(Settings(model_cache_dir=tmp_path / "cache"))
    adapter._device = "cuda"
    adapter._model = object()
    with pytest.raises(Codex3DError, match="restart this server"):
        adapter.shutdown()
    assert events == ["synchronize", "empty_cache"]
    assert adapter._model is None
    assert adapter._device == "cuda"


def test_model_unload_clears_device_only_after_cuda_allocator_is_empty(tmp_path, monkeypatch):
    events = []
    cuda = SimpleNamespace(
        synchronize=lambda: events.append("synchronize"),
        empty_cache=lambda: events.append("empty_cache"),
        memory_allocated=lambda: 0, memory_reserved=lambda: 0,
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", SimpleNamespace(cuda=cuda))
    adapter = ImageTo3DAdapter(Settings(model_cache_dir=tmp_path / "cache"))
    adapter._device = "cuda"
    adapter._model = object()
    adapter.shutdown()
    assert events == ["synchronize", "empty_cache"]
    assert adapter._model is None and adapter._device is None
