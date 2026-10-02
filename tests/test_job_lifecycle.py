"""Real worker-thread cancellation and shutdown regressions without model work."""

from dataclasses import replace
from threading import Event

import pytest

from codex_3d_mcp.errors import Codex3DError, GenerationCancelled
from codex_3d_mcp.hunyuan_mv.config import HunyuanMVSettings
from codex_3d_mcp.hunyuan_mv.pipeline import HunyuanMVJobManager
from codex_3d_mcp.trellis.config import TrellisSettings
from codex_3d_mcp.trellis.pipeline import BidirectionalJobManager
from codex_3d_mcp.trellis.store import TrellisJobStore


class FakeRuntime:
    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1

    def close(self):
        pass


@pytest.fixture(params=["hunyuan", "bidirectional"])
def manager(request, tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_3D_DATA_ROOT", str(tmp_path / "data"))
    runtime = FakeRuntime()
    if request.param == "hunyuan":
        settings = replace(HunyuanMVSettings.from_env(tmp_path), max_queued_jobs=1)
        value = HunyuanMVJobManager(settings, runtime=runtime)
    else:
        settings = replace(TrellisSettings.from_env(tmp_path), max_queued_jobs=1)
        value = BidirectionalJobManager(settings, runtime=runtime, semantic=object())
    yield value
    value.close()


def enqueue(manager):
    record = manager.store.create("thread fixture", {})
    manager.job_dir(record["job_id"]).mkdir(parents=True)
    manager._enqueue(record["job_id"])
    return record["job_id"]


def test_cancelling_queued_job_does_not_stop_running_job(manager):
    started, release = Event(), Event()

    def process(job_id, cancel):
        started.set()
        assert release.wait(5)
        assert not cancel.is_set()
        manager.store.update(job_id, state="completed")

    manager._process = process
    first = enqueue(manager)
    assert started.wait(5)
    second = enqueue(manager)
    try:
        assert manager.cancel(second)["state"] == "cancelled"
        assert manager.runtime.stops == 0
        assert manager.store.get(first)["state"] == "running"
    finally:
        release.set()
        manager._queue.join()
    assert manager.store.get(first)["state"] == "completed"


def test_close_joins_worker_before_closing_store_even_when_queue_is_full(manager):
    started, wrote_checkpoint = Event(), Event()

    def process(job_id, cancel):
        started.set()
        assert cancel.wait(5)
        # This write must still be possible after shutdown was requested.
        manager.store.checkpoint(job_id, "shutdown", {"retained": True})
        wrote_checkpoint.set()
        raise GenerationCancelled("shutdown")

    manager._process = process
    first = enqueue(manager)
    assert started.wait(5)
    second = enqueue(manager)
    manager.close()
    assert wrote_checkpoint.is_set()
    assert not manager._worker.is_alive()
    with pytest.raises(Codex3DError, match="shutting down"):
        manager.resume(first)
    reopened = TrellisJobStore(manager.settings.database_path)
    try:
        assert reopened.get(first)["state"] == "cancelled"
        assert reopened.get(second)["state"] == "interrupted"
        assert reopened.checkpoints(first)["shutdown"] == {"retained": True}
    finally:
        reopened.close()


def test_cancelling_waiting_review_is_terminal(manager):
    record = manager.store.create("awaiting human review", {})
    manager.store.update(record["job_id"], state="awaiting_review")
    cancelled = manager.cancel(record["job_id"])
    assert cancelled["state"] == "cancelled"
    assert manager.runtime.stops == 0
