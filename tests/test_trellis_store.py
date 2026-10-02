from __future__ import annotations

from pathlib import Path

from codex_3d_mcp.trellis.store import TrellisJobStore


def test_jobs_and_checkpoints_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    store = TrellisJobStore(path)
    job = store.create("asset", {"seed": 42})
    store.update(job["job_id"], state="running", stage="generating_front", progress=20)
    store.checkpoint(job["job_id"], "validating_pair", {"approved": True})
    store.close()

    reopened = TrellisJobStore(path)
    record = reopened.get(job["job_id"])
    assert record["state"] == "interrupted"
    assert reopened.checkpoints(job["job_id"])["validating_pair"] == {"approved": True}
    reopened.close()


def test_metadata_merge_preserves_existing_values(tmp_path: Path) -> None:
    store = TrellisJobStore(tmp_path / "jobs.sqlite3")
    job = store.create("asset", {})
    store.update(job["job_id"], metadata={"one": 1})
    result = store.merge_metadata(job["job_id"], {"two": 2})
    assert result["metadata"] == {"one": 1, "two": 2}
    store.close()
