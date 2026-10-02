import threading
import time
from pathlib import Path

from codex_3d_mcp.config import Settings
from codex_3d_mcp.inputs import StagedImage
from codex_3d_mcp.jobs import JobManager


class FakeAdapter:
    def generate(self, image_path: Path, output_path: Path, **kwargs):
        progress = kwargs["progress"]
        cancel = kwargs["cancel_event"]
        progress(10, "fake")
        time.sleep(0.02)
        if cancel.is_set():
            from codex_3d_mcp.errors import GenerationCancelled

            raise GenerationCancelled("cancelled")
        output_path.write_bytes(b"glb")
        progress(100, "completed")
        return {"vertex_count": 3}

    def shutdown(self):
        pass


def test_job_completes(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path / "out", allowed_input_roots=(tmp_path,))
    manager = JobManager(settings, FakeAdapter())
    staged = StagedImage(tmp_path / "input.png", False, "path")
    record = manager.submit(
        "a chair",
        staged,
        {
            "texture_resolution": 1024,
            "foreground_ratio": 0.85,
            "remesh": "none",
            "target_vertex_count": None,
        },
    )
    deadline = time.time() + 2
    while time.time() < deadline and manager.get(record.job_id).status not in {
        "completed",
        "failed",
    }:
        time.sleep(0.01)
    result = manager.get(record.job_id).as_dict()
    assert result["status"] == "completed"
    assert result["artifact_path"].endswith(".glb")
    manager.shutdown()


def test_queued_job_can_be_cancelled(tmp_path: Path) -> None:
    settings = Settings(
        output_dir=tmp_path / "out", allowed_input_roots=(tmp_path,), max_queued_jobs=2
    )
    manager = JobManager(settings, FakeAdapter())
    blocker = threading.Event()
    blocker.set()
    first = manager.submit(
        "first",
        StagedImage(tmp_path / "a.png", False, "path"),
        {
            "texture_resolution": 1024,
            "foreground_ratio": 0.85,
            "remesh": "none",
            "target_vertex_count": None,
        },
    )
    second = manager.submit(
        "second",
        StagedImage(tmp_path / "b.png", False, "path"),
        {
            "texture_resolution": 1024,
            "foreground_ratio": 0.85,
            "remesh": "none",
            "target_vertex_count": None,
        },
    )
    cancelled = manager.cancel(second.job_id)
    assert cancelled["cancel_requested"]
    assert manager.get(second.job_id).status == "cancelled"
    deadline = time.time() + 2
    while time.time() < deadline and manager.get(first.job_id).status == "queued":
        time.sleep(0.01)
    manager.shutdown()
