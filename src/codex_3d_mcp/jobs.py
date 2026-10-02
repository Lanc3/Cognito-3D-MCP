"""Single-worker asynchronous generation jobs."""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import Codex3DError, GenerationCancelled, JobNotFoundError
from .inputs import StagedImage, cleanup_staged_image
from .lifecycle import manager_operation
from .model import ImageTo3DAdapter
from .trellis.runtime import GpuLease


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    job_id: str
    prompt: str
    image: StagedImage
    params: dict[str, Any]
    output_path: Path
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    status: str = "queued"
    progress: int = 0
    stage: str = "queued"
    metadata: dict[str, Any] = field(default_factory=dict)
    error: dict[str, str] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "prompt": self.prompt,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "artifact_path": str(self.output_path) if self.status == "completed" else None,
            "params": dict(self.params),
            "metadata": dict(self.metadata),
            "error": dict(self.error) if self.error else None,
        }


class JobManager:
    def __init__(self, settings: Settings, adapter: ImageTo3DAdapter) -> None:
        self.settings = settings
        self.adapter = adapter
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._closing = False
        self._closed = False
        self.gpu_lease = GpuLease(settings.gpu_lock_path)
        self._lease_owned = False
        self._resource_error: dict[str, str] | None = None
        self._queue: queue.Queue[JobRecord | None] = queue.Queue(maxsize=settings.max_queued_jobs)
        self._worker = threading.Thread(
            target=self._worker_loop, name="codex-3d-worker", daemon=True
        )
        self._worker.start()

    @manager_operation
    def submit(self, prompt: str, image: StagedImage, params: dict[str, Any]) -> JobRecord:
        if self._resource_error is not None:
            cleanup_staged_image(image)
            raise Codex3DError(self._resource_error["message"], code="GPU_UNLOAD_FAILED")
        job_id = uuid.uuid4().hex
        record = JobRecord(
            job_id=job_id,
            prompt=prompt,
            image=image,
            params=dict(params),
            output_path=self.settings.output_dir / f"{job_id}.glb",
        )
        with self._lock:
            self._jobs[job_id] = record
            try:
                self._queue.put_nowait(record)
            except queue.Full as exc:
                self._jobs.pop(job_id, None)
                cleanup_staged_image(image)
                raise Codex3DError(
                    "The generation queue is full; try again later.", code="QUEUE_FULL"
                ) from exc
        return record

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs.get(job_id)
        if record is None:
            raise JobNotFoundError(f"Unknown job_id: {job_id}")
        return record

    def cancel(self, job_id: str) -> dict[str, Any]:
        record = self.get(job_id)
        with self._lock:
            if record.status in {"completed", "failed", "cancelled"}:
                return {"cancel_requested": False, **record.as_dict()}
            record.cancel_event.set()
            if record.status == "queued":
                record.status = "cancelled"
                record.stage = "cancelled"
                record.finished_at = _now()
        return {"cancel_requested": True, **record.as_dict()}

    def summary(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for record in self._jobs.values():
                counts[record.status] = counts.get(record.status, 0) + 1
            return counts

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                record = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if record is None:
                self._queue.task_done()
                return
            metadata = None
            lease_acquired = False
            try:
                with self._lock:
                    if self._shutdown.is_set() or record.cancel_event.is_set():
                        continue
                    record.status = "running"
                    record.started_at = _now()
                    record.stage = "starting"
                    if self._resource_error is not None:
                        raise Codex3DError(self._resource_error["message"], code="GPU_UNLOAD_FAILED")

                def progress(value: int, stage: str) -> None:
                    with self._lock:
                        record.progress = max(0, min(100, int(value)))
                        record.stage = stage

                progress(1, "waiting for GPU")
                self.gpu_lease.acquire(
                    timeout=self.settings.gpu_wait_seconds, cancel_event=record.cancel_event
                )
                self._lease_owned = True
                lease_acquired = True
                metadata = self.adapter.generate(
                    record.image.path,
                    record.output_path,
                    texture_resolution=record.params["texture_resolution"],
                    foreground_ratio=record.params["foreground_ratio"],
                    remesh=record.params["remesh"],
                    target_vertex_count=record.params["target_vertex_count"],
                    progress=progress,
                    cancel_event=record.cancel_event,
                )
            except GenerationCancelled as exc:
                with self._lock:
                    record.status = "cancelled"
                    record.stage = "cancelled"
                    record.error = {"code": exc.code, "message": exc.message}
                    record.finished_at = _now()
                    record.output_path.unlink(missing_ok=True)
            except Exception as exc:
                with self._lock:
                    record.status = "failed"
                    record.stage = "failed"
                    if isinstance(exc, Codex3DError):
                        record.error = {"code": exc.code, "message": exc.message}
                    else:
                        record.error = {"code": "GENERATION_FAILED", "message": str(exc)}
                    record.finished_at = _now()
                    record.output_path.unlink(missing_ok=True)
            finally:
                if lease_acquired:
                    with self._lock:
                        record.stage = "unloading model"
                    try:
                        self._unload_and_release()
                    except Exception as exc:
                        with self._lock:
                            self._resource_error = {
                                "code": "GPU_UNLOAD_FAILED",
                                "message": f"Model unload failed; the GPU lease is retained. "
                                f"Restart this server before more work. {exc}",
                            }
                            record.status = "failed"
                            record.error = dict(self._resource_error)
                            metadata = None
                            record.output_path.unlink(missing_ok=True)
                    with self._lock:
                        if metadata is not None:
                            if record.cancel_event.is_set():
                                record.status = "cancelled"
                                record.output_path.unlink(missing_ok=True)
                            else:
                                record.status = "completed"
                                record.progress = 100
                                record.metadata.update(metadata)
                        record.stage = record.status
                        record.finished_at = _now()
                cleanup_staged_image(record.image)
                self._queue.task_done()

    def _unload_and_release(self) -> None:
        # The lease remains owned if model destruction, synchronization, or
        # allocator validation raises or hangs. A different server cannot overlap it.
        self.adapter.shutdown()
        if self._lease_owned:
            self.gpu_lease.release()
            self._lease_owned = False
        self._resource_error = None

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closing = True
            self._shutdown.set()
            for record in self._jobs.values():
                if record.status in {"queued", "running"}:
                    record.cancel_event.set()
                if record.status == "queued":
                    record.status = "cancelled"
                    record.stage = "cancelled"
                    record.finished_at = _now()
                    cleanup_staged_image(record.image)
        self._worker.join(timeout=30)
        if self._worker.is_alive():
            raise Codex3DError("Generation worker has not stopped.", code="WORKER_BUSY")
        if self._resource_error is not None:
            # Retrying a failed CUDA unload on the caller could hang shutdown.
            # Process exit is the final recovery boundary for an unproven unload.
            raise Codex3DError(self._resource_error["message"], code="GPU_UNLOAD_FAILED")
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
        self._unload_and_release()
        self._closed = True
