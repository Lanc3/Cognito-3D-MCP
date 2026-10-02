"""Durable bidirectional reconstruction state machine."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

from ..errors import Codex3DError, GenerationCancelled, InvalidPathError
from ..lifecycle import manager_operation
from ..inputs import validate_image_file
from .artifacts import artifact_record, atomic_json, stage_immutable
from .config import TrellisSettings
from .qa import QualityAssurance
from .runtime import TrellisRuntime
from .semantic import DinoSemanticScorer
from .spar3d_runtime import Spar3DRuntime
from .store import TERMINAL_STATES, TrellisJobStore, utc_now
from .textures import TextureProcessor
from .validation import PairValidator

STAGE_PROGRESS = {
    "validating_pair": 5,
    "generating_front": 15,
    "generating_back": 30,
    "canonicalizing": 43,
    "aligning": 50,
    "fusing": 60,
    "unwrapping": 70,
    "projecting_textures": 78,
    "baking_pbr": 85,
    "optimizing": 92,
    "validating": 97,
}


class BidirectionalJobManager:
    def __init__(
        self,
        settings: TrellisSettings,
        *,
        store: TrellisJobStore | None = None,
        runtime: TrellisRuntime | Spar3DRuntime | None = None,
        semantic: DinoSemanticScorer | None = None,
    ) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.store = store or TrellisJobStore(settings.database_path)
        self.runtime = runtime or self._reconstruction_runtime(settings)
        self.semantic = semantic or DinoSemanticScorer(settings.model_dir / "dino-cache")
        self.validator = PairValidator(
            settings.minimum_image_size, settings.pair_semantic_threshold
        )
        self.textures = TextureProcessor(settings)
        self.qa = QualityAssurance(settings)
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=settings.max_queued_jobs)
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._closing = False
        self._closed = False
        self._active_job_id: str | None = None
        self._worker = threading.Thread(
            target=self._worker_loop, name="bidirectional-3d-worker", daemon=True
        )
        self._worker.start()

    @manager_operation
    def submit(
        self,
        prompt: str,
        front_image_path: str,
        back_image_path: str,
        material_hints: str,
        seed: int,
    ) -> dict[str, Any]:
        front = self._source_path(front_image_path)
        back = self._source_path(back_image_path)
        params = {
            "reconstruction_backend": self.settings.reconstruction_backend,
            "seed": seed,
            "resolution": self.settings.resolution,
            "texture_resolution": self.settings.texture_resolution,
            "master_faces": self.settings.master_faces,
            "game_faces": self.settings.game_faces,
            "lod_faces": list(self.settings.lod_faces),
            "material_hints": material_hints,
        }
        record = self.store.create(prompt, params)
        job_dir = self.job_dir(record["job_id"])
        try:
            front_record = stage_immutable(front, job_dir / "staged" / "front.png")
            back_record = stage_immutable(back, job_dir / "staged" / "back.png")
            metadata = {
                "job_dir": str(job_dir),
                "inputs": {"front": front_record, "back": back_record},
                "required_action": None,
            }
            self.store.update(record["job_id"], metadata=metadata)
            atomic_json(
                job_dir / "request.json",
                {
                    "job_id": record["job_id"],
                    "prompt": prompt,
                    "params": params,
                    "inputs": metadata["inputs"],
                    "runtime": self._provenance(),
                },
            )
            self._enqueue(record["job_id"])
        except Exception as exc:
            self._fail(record["job_id"], exc)
            raise
        return self.get(record["job_id"])

    def get(self, job_id: str) -> dict[str, Any]:
        record = self.store.get(job_id)
        record["artifacts"] = self._available_artifacts(self.job_dir(job_id))
        record["checkpoints"] = sorted(self.store.checkpoints(job_id))
        return record

    def artifacts(self, job_id: str) -> dict[str, Any]:
        record = self.store.get(job_id)
        return {
            "job_id": job_id,
            "state": record["state"],
            "job_dir": str(self.job_dir(job_id)),
            "artifacts": self._available_artifacts(self.job_dir(job_id)),
        }

    @manager_operation
    def review(self, job_id: str, decision: str, notes: str = "") -> dict[str, Any]:
        record = self.store.get(job_id)
        if record["state"] != "awaiting_review":
            raise Codex3DError("The job is not awaiting seed review.", code="INVALID_STATE")
        validation = record["metadata"].get("pair_validation", {})
        if decision == "approve":
            if validation.get("outcome") == "rejected":
                raise Codex3DError(
                    "A structurally rejected pair cannot be approved; "
                    "submit regenerated references.",
                    code="PAIR_REJECTED",
                )
            self.store.merge_metadata(
                job_id,
                {"pair_approved": True, "review_notes": notes, "required_action": None},
            )
            self.store.update(job_id, state="queued", stage="generating_front")
            self._enqueue(job_id)
        elif decision == "reject":
            self.store.update(
                job_id,
                state="failed_quality",
                stage="validating_pair",
                finished_at=utc_now(),
                error={"code": "PAIR_REJECTED", "message": notes or "Seed pair rejected."},
            )
        else:
            raise ValueError("decision must be 'approve' or 'reject'")
        return self.get(job_id)

    @manager_operation
    def add_sides(self, job_id: str, left_path: str, right_path: str) -> dict[str, Any]:
        record = self.store.get(job_id)
        if record["state"] != "awaiting_side_inputs":
            raise Codex3DError("The job is not awaiting side references.", code="INVALID_STATE")
        left = self._source_path(left_path)
        right = self._source_path(right_path)
        job_dir = self.job_dir(job_id)
        inputs = dict(record["metadata"]["inputs"])
        inputs["left"] = stage_immutable(left, job_dir / "staged" / "left.png")
        inputs["right"] = stage_immutable(right, job_dir / "staged" / "right.png")
        self.store.merge_metadata(job_id, {"inputs": inputs, "required_action": None})
        self.store.update(job_id, state="queued", stage="generating_front")
        self._enqueue(job_id)
        return self.get(job_id)

    @manager_operation
    def resume(self, job_id: str) -> dict[str, Any]:
        record = self.store.get(job_id)
        if record["state"] not in {"interrupted", "failed", "failed_quality"}:
            raise Codex3DError(
                "Only interrupted, operationally failed, or repairable quality-failed "
                "jobs can resume.",
                code="INVALID_STATE",
            )
        if record["state"] == "failed_quality":
            job_dir = self.job_dir(job_id)
            required = (
                job_dir / "raw" / "front.glb",
                job_dir / "raw" / "back.glb",
            )
            if not all(path.is_file() for path in required):
                raise Codex3DError(
                    "The quality-failed job has no complete repair checkpoint.",
                    code="NOT_REPAIRABLE",
                )
        self.store.update(
            job_id, state="queued", error=None, cancel_requested=False, finished_at=None
        )
        self._enqueue(job_id)
        return self.get(job_id)

    @manager_operation
    def cancel(self, job_id: str) -> dict[str, Any]:
        record = self.store.get(job_id)
        if record["state"] in TERMINAL_STATES:
            return {"cancel_requested": False, **self.get(job_id)}
        with self._lock:
            event = self._cancel_events.setdefault(job_id, threading.Event())
            event.set()
        self.store.update(job_id, cancel_requested=True)
        if self._active_job_id == job_id:
            self.runtime.stop()
        else:
            self.store.update(
                job_id,
                state="cancelled",
                stage="cancelled",
                finished_at=utc_now(),
            )
        return {"cancel_requested": True, **self.get(job_id)}

    @manager_operation
    def cleanup(self, job_id: str, keep_final: bool = True) -> dict[str, Any]:
        record = self.store.get(job_id)
        if record["state"] not in TERMINAL_STATES:
            raise Codex3DError("Only terminal jobs may be cleaned up.", code="INVALID_STATE")
        job_dir = self.job_dir(job_id)
        preserved = {"master.glb", "game.glb", "lod1.glb", "lod2.glb", "request.json"}
        removed = []
        for path in list(job_dir.iterdir()):
            if keep_final and path.name in preserved:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            removed.append(str(path))
        return {"job_id": job_id, "removed": removed, "recoverable": False}

    def summary(self) -> dict[str, int]:
        states = (
            "queued",
            "running",
            "awaiting_review",
            "awaiting_side_inputs",
            "completed",
            "failed",
            "failed_quality",
            "cancelled",
            "interrupted",
        )
        return {state: len(self.store.list_by_state(state)) for state in states}

    def job_dir(self, job_id: str) -> Path:
        path = (self.settings.output_dir / job_id).resolve()
        path.relative_to(self.settings.output_dir.resolve())
        return path

    def _source_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.settings.base_dir / path
        path = path.resolve()
        if not path.is_file():
            raise InvalidPathError(f"Image path is not a file: {path}")
        if not self.settings.is_allowed_input(path):
            raise InvalidPathError(f"Image path is outside allowed input roots: {path}")
        if path.suffix.lower() != ".png":
            raise InvalidPathError("Bidirectional production references must be PNG files.")
        validate_image_file(path)
        return path

    def _enqueue(self, job_id: str) -> None:
        with self._lock:
            self._cancel_events[job_id] = threading.Event()
        try:
            self._queue.put_nowait(job_id)
        except queue.Full as exc:
            with self._lock:
                self._cancel_events.pop(job_id, None)
            self._fail(job_id, Codex3DError("3D reconstruction queue is full.", code="QUEUE_FULL"))
            raise Codex3DError("3D reconstruction queue is full.", code="QUEUE_FULL") from exc

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                job_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job_id is None:
                self._queue.task_done()
                return
            try:
                with self._lock:
                    if self._shutdown.is_set():
                        continue
                    record = self.store.get(job_id)
                    if record["state"] == "cancelled":
                        continue
                    self._active_job_id = job_id
                    self.store.update(
                        job_id,
                        state="running",
                        started_at=record["started_at"] or utc_now(),
                        cancel_requested=False,
                    )
                self._process(job_id, self._cancel_events[job_id])
            except GenerationCancelled as exc:
                self.store.update(
                    job_id,
                    state="cancelled",
                    stage="cancelled",
                    finished_at=utc_now(),
                    error={"code": exc.code, "message": exc.message},
                )
            except Exception as exc:
                if self._cancel_events[job_id].is_set():
                    self.store.update(
                        job_id, state="cancelled", stage="cancelled", finished_at=utc_now()
                    )
                else:
                    self._fail(job_id, exc)
            finally:
                with self._lock:
                    self._active_job_id = None
                    self._cancel_events.pop(job_id, None)
                self._queue.task_done()

    def _process(self, job_id: str, cancel_event: threading.Event) -> None:
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        record = self.store.get(job_id)
        metadata = record["metadata"]
        checkpoints = self.store.checkpoints(job_id)
        staged = job_dir / "staged"
        masks = job_dir / "masks"
        masks.mkdir(exist_ok=True)
        front = staged / "front.png"
        back = staged / "back.png"

        if "validating_pair" not in checkpoints:
            self._stage(job_id, "validating_pair")
            if self.settings.test_mode:
                self._test_mask(front, masks / "front.png")
                self._test_mask(back, masks / "back.png")
            else:
                self.runtime.gpu_lease.acquire(timeout=600, cancel_event=cancel_event)
                try:
                    mask_log = job_dir / "logs" / "mask.log"
                    self.runtime.extract_mask(front, masks / "front.png", mask_log)
                    self.runtime.extract_mask(back, masks / "back.png", mask_log)
                finally:
                    self.runtime.gpu_lease.release()
            validation = self.validator.validate(
                front,
                back,
                masks / "front.png",
                masks / "back.png",
                self.semantic.similarity,
            ).as_dict()
            self.store.merge_metadata(job_id, {"pair_validation": validation})
            self.store.checkpoint(job_id, "validating_pair", validation)
        else:
            validation = checkpoints["validating_pair"]
        if validation["outcome"] != "approved" and not metadata.get("pair_approved"):
            self.store.merge_metadata(
                job_id,
                {
                    "required_action": "review_seed_pair",
                    "pair_validation": validation,
                },
            )
            self.store.update(job_id, state="awaiting_review", stage="validating_pair")
            return
        self._check_cancel(cancel_event)

        raw_dir = job_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        views = ["front", "back"]
        current_inputs = self.store.get(job_id)["metadata"]["inputs"]
        if "left" in current_inputs and "right" in current_inputs:
            views.extend(["left", "right"])
        missing_views = [view for view in views if not (raw_dir / f"{view}.glb").is_file()]
        if missing_views:
            backend = self.settings.reconstruction_backend
            with self.runtime.session(job_dir / "logs" / f"{backend}.log", cancel_event):
                for index, view in enumerate(views):
                    output = raw_dir / f"{view}.glb"
                    if output.is_file():
                        continue
                    self._stage(
                        job_id,
                        "generating_front" if index == 0 else "generating_back",
                    )
                    self._check_cancel(cancel_event)
                    self._generate_with_retry(
                        staged / f"{view}.png", output, record["params"]["seed"]
                    )
                    self.store.checkpoint(
                        job_id, f"generated_{view}", artifact_record(output, job_dir)
                    )
        self._check_cancel(cancel_event)

        analysis_result = job_dir / "raw-analysis.json"
        if not analysis_result.is_file() or len(views) > 2:
            self._stage(job_id, "canonicalizing")
            config = self._blender_config(job_id, "analyze", analysis_result)
            self._run_blender(job_dir, config, "analyze")
        raw_analysis = json.loads(analysis_result.read_text(encoding="utf-8"))
        self.store.checkpoint(job_id, "raw_analysis", raw_analysis)
        if not raw_analysis.get("passed", False):
            self.store.merge_metadata(
                job_id,
                {"required_action": None, "raw_analysis": raw_analysis},
            )
            self.store.update(
                job_id,
                state="failed_quality",
                stage="canonicalizing",
                finished_at=utc_now(),
                error={
                    "code": "SOURCE_FRAGMENTATION",
                    "message": "Raw reconstruction fragmentation exceeds production limits.",
                },
            )
            return

        texture_dir = job_dir / "textures"
        texture_dir.mkdir(exist_ok=True)
        self._stage(job_id, "projecting_textures")
        prepared_front = texture_dir / "front-reference.png"
        prepared_back = texture_dir / "back-reference.png"
        references_missing = not prepared_front.is_file() or not prepared_back.is_file()
        if references_missing:
            self.textures.prepare_reference(front, prepared_front)
            self.textures.prepare_reference(back, prepared_back)
        if references_missing or not (texture_dir / "front-upscaled.png").is_file():
            self.runtime.gpu_lease.acquire(timeout=600, cancel_event=cancel_event)
            try:
                self.textures.upscale(prepared_front, texture_dir / "front-upscaled.png")
                self.textures.upscale(prepared_back, texture_dir / "back-upscaled.png")
            finally:
                self.runtime.gpu_lease.release()

        build_result = job_dir / "blender-build.json"
        if not build_result.is_file() or len(views) > 2:
            self._stage(job_id, "fusing")
            config = self._blender_config(job_id, "build", build_result)
            self._run_blender(job_dir, config, "build")
        build_metrics = json.loads(build_result.read_text(encoding="utf-8"))
        side_agreement = float(build_metrics.get("alignment", {}).get("side_agreement", 1.0))
        if side_agreement < 0.75 and len(views) == 2:
            self.store.merge_metadata(
                job_id,
                {
                    "required_action": "add_side_references",
                    "side_agreement": side_agreement,
                },
            )
            self.store.update(job_id, state="awaiting_side_inputs", stage="aligning", progress=50)
            return
        self.store.checkpoint(job_id, "fusing", build_metrics)
        self._check_cancel(cancel_event)

        self._stage(job_id, "baking_pbr")
        maps = self.textures.assemble_maps(texture_dir, record["params"]["material_hints"])
        final_result = job_dir / "blender-final.json"
        config = self._blender_config(job_id, "finalize", final_result)
        config["maps"] = maps
        self._run_blender(job_dir, config, "finalize")
        final_metrics = json.loads(final_result.read_text(encoding="utf-8"))
        combined_metrics = {**build_metrics, **final_metrics}
        self.store.checkpoint(job_id, "baking_pbr", {"maps": maps})

        self._stage(job_id, "validating")
        report = self.qa.validate(job_dir, combined_metrics)
        self.store.checkpoint(job_id, "validating", report)
        atomic_json(job_dir / "manifest.json", self._manifest(job_id, report, maps))
        if not report["passed"]:
            self.store.update(
                job_id,
                state="failed_quality",
                stage="validating",
                finished_at=utc_now(),
                error={"code": "QUALITY_GATES_FAILED", "message": "See qa/report.html."},
            )
            return
        self.store.merge_metadata(job_id, {"required_action": None, "qa": report})
        self.store.update(
            job_id,
            state="completed",
            stage="completed",
            progress=100,
            finished_at=utc_now(),
        )

    def _generate_with_retry(self, image: Path, output: Path, seed: int) -> None:
        try:
            self.runtime.generate(image, output, seed)
        except Exception:
            self.runtime.stop()
            time.sleep(1)
            backend = self.settings.reconstruction_backend
            self.runtime.start(output.parent.parent / "logs" / f"{backend}-retry.log")
            self.runtime.generate(image, output, seed)

    def _blender_config(self, job_id: str, mode: str, result_path: Path) -> dict[str, Any]:
        job_dir = self.job_dir(job_id)
        raw = job_dir / "raw"
        texture = job_dir / "textures"
        config = {
            "mode": mode,
            "job_dir": str(job_dir),
            "front_glb": str(raw / "front.glb"),
            "back_glb": str(raw / "back.glb"),
            "front_image": str(texture / "front-upscaled.png"),
            "back_image": str(texture / "back-upscaled.png"),
            "texture_resolution": self.settings.texture_resolution,
            "master_faces": self.settings.master_faces,
            "game_faces": self.settings.game_faces,
            "lod_faces": list(self.settings.lod_faces),
            "result_path": str(result_path),
        }
        if (raw / "left.glb").is_file():
            config["left_glb"] = str(raw / "left.glb")
            config["right_glb"] = str(raw / "right.glb")
        return config

    def _run_blender(self, job_dir: Path, config: dict[str, Any], label: str) -> None:
        if not self.settings.blender_exe.is_file():
            raise Codex3DError("Blender 5.1 is missing.", code="BLENDER_UNAVAILABLE")
        config_path = job_dir / f"blender-{label}-config.json"
        atomic_json(config_path, config)
        script = Path(__file__).with_name("blender_worker.py").resolve()
        command = [
            str(self.settings.blender_exe),
            "--background",
            "--factory-startup",
            "--python",
            str(script),
            "--",
            str(config_path),
        ]
        temp_dir = job_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        environment = os.environ.copy()
        environment.update({"TMP": str(temp_dir), "TEMP": str(temp_dir), "TMPDIR": str(temp_dir)})
        log_path = job_dir / "logs" / f"blender-{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log:
            result = subprocess.run(
                command,
                cwd=self.settings.base_dir,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=self.settings.request_timeout_seconds,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                check=False,
            )
        if result.returncode != 0 or not Path(config["result_path"]).is_file():
            detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise Codex3DError(f"Blender {label} failed: {detail}", code="BLENDER_FAILED")

    def _stage(self, job_id: str, stage: str) -> None:
        self.store.update(
            job_id,
            state="running",
            stage=stage,
            progress=STAGE_PROGRESS.get(stage, 0),
        )

    def _check_cancel(self, event: threading.Event) -> None:
        if event.is_set():
            raise GenerationCancelled("Generation was cancelled.")

    def _fail(self, job_id: str, exc: Exception) -> None:
        code = exc.code if isinstance(exc, Codex3DError) else "PIPELINE_FAILED"
        message = exc.message if isinstance(exc, Codex3DError) else str(exc)
        self.store.update(
            job_id,
            state="failed",
            stage=self.store.get(job_id)["stage"],
            finished_at=utc_now(),
            error={"code": code, "message": message},
        )

    def _test_mask(self, source: Path, destination: Path) -> None:
        with Image.open(source) as image:
            rgba = image.convert("RGBA")
            if rgba.getchannel("A").getextrema()[0] < 255:
                alpha = rgba.getchannel("A")
            else:
                background = Image.new("RGB", rgba.size, rgba.convert("RGB").getpixel((0, 0)))
                alpha = ImageChops.difference(rgba.convert("RGB"), background).convert("L")
                alpha = alpha.point(lambda value: 255 if value > 8 else 0)
            black = Image.new("RGB", rgba.size, (0, 0, 0))
            black.paste(rgba.convert("RGB"), mask=alpha)
            black.save(destination, "PNG")

    def _available_artifacts(self, job_dir: Path) -> list[dict[str, Any]]:
        if not job_dir.is_dir():
            return []
        candidates = [
            job_dir / "master.glb",
            job_dir / "game.glb",
            job_dir / "lod1.glb",
            job_dir / "lod2.glb",
            job_dir / "manifest.json",
            job_dir / "qa" / "report.json",
            job_dir / "qa" / "report.html",
            job_dir / "working.blend",
        ]
        return [artifact_record(path, job_dir) for path in candidates if path.is_file()]

    def _manifest(self, job_id: str, report: dict, maps: dict) -> dict[str, Any]:
        record = self.store.get(job_id)
        return {
            "schema_version": 1,
            "job": record,
            "runtime": self._provenance(),
            "maps": maps,
            "qa": report,
            "artifacts": self._available_artifacts(self.job_dir(job_id)),
        }

    def _provenance(self) -> dict[str, Any]:
        if self.settings.reconstruction_backend == "spar3d":
            return {
                "reconstruction_backend": "spar3d",
                "model_id": self.settings.spar3d_model_id,
                "model_cache_dir": str(self.settings.spar3d_model_cache_dir),
                "low_vram_mode": self.settings.spar3d_low_vram_mode,
            }
        return {
            "reconstruction_backend": "trellis",
            "trellis_version": self.settings.runtime_version,
            "trellis_commit": self.settings.runtime_commit,
            "archive_sha256": self.settings.runtime_archive_sha256,
            "weights_revision": self.settings.weights_revision,
        }

    @staticmethod
    def _reconstruction_runtime(
        settings: TrellisSettings,
    ) -> TrellisRuntime | Spar3DRuntime:
        if settings.reconstruction_backend == "spar3d":
            return Spar3DRuntime(settings)
        return TrellisRuntime(settings)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closing = True
            self._shutdown.set()
            for event in self._cancel_events.values():
                event.set()
        self.runtime.stop()
        self._worker.join(timeout=30)
        if self._worker.is_alive():
            raise Codex3DError("Generation worker has not stopped.", code="WORKER_BUSY")
        self.runtime.close()
        with self._lock:
            for record in self.store.list_by_state("queued"):
                self.store.update(record["job_id"], state="interrupted")
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()
            self._cancel_events.clear()
            self.store.close()
            self._closed = True
