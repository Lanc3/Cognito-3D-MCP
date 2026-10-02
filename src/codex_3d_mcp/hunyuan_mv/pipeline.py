"""Durable multiview-to-one-mesh Hunyuan3D-2mv state machine."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from ..errors import Codex3DError, GenerationCancelled, InvalidImageError, InvalidPathError
from ..lifecycle import manager_operation
from ..inputs import validate_image_file
from ..trellis.artifacts import artifact_record, atomic_json, stage_immutable
from ..trellis.store import TERMINAL_STATES, TrellisJobStore, utc_now
from .backgrounds import prepare_reference
from .config import HunyuanMVSettings
from .runtime import HunyuanMVRuntime

VIEW_NAMES = ("front", "back", "left", "right")
STAGE_PROGRESS = {
    "validating_views": 5,
    "generating_shape": 25,
    "texturing": 60,
    "postprocessing": 80,
    "validating": 95,
}
QUALITY_PRESETS = {
    "draft": {"steps": 20, "guidance_scale": 5.0, "octree_resolution": 196, "num_chunks": 8_000},
    "standard": {
        "steps": 30,
        "guidance_scale": 5.0,
        "octree_resolution": 256,
        "num_chunks": 20_000,
    },
    "high": {"steps": 50, "guidance_scale": 7.5, "octree_resolution": 384, "num_chunks": 40_000},
}


class HunyuanMVJobManager:
    def __init__(
        self,
        settings: HunyuanMVSettings,
        *,
        store: TrellisJobStore | None = None,
        runtime: HunyuanMVRuntime | None = None,
    ) -> None:
        self.settings = settings
        settings.ensure_directories()
        self.store = store or TrellisJobStore(settings.database_path)
        self.runtime = runtime or HunyuanMVRuntime(settings)
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=settings.max_queued_jobs)
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._closing = False
        self._closed = False
        self._active_job_id: str | None = None
        self._worker = threading.Thread(
            target=self._worker_loop, name="hunyuan-multiview-worker", daemon=True
        )
        self._worker.start()

    @manager_operation
    def submit(
        self,
        prompt: str,
        view_paths: dict[str, str],
        material_hints: str,
        seed: int,
        quality: str,
        generate_texture: bool,
    ) -> dict[str, Any]:
        if quality not in QUALITY_PRESETS:
            raise ValueError("quality must be 'draft', 'standard', or 'high'")
        supplied = {name: value for name, value in view_paths.items() if value.strip()}
        if "front" not in supplied:
            raise ValueError("front_image_path is required")
        if len(supplied) > 4 or set(supplied) - set(VIEW_NAMES):
            raise ValueError("Only front, back, left, and right views are supported")
        sources = {name: self._source_path(value) for name, value in supplied.items()}
        params = {
            "backend": "hunyuan3d-2mv",
            "views": list(sources),
            "seed": seed,
            "quality": quality,
            "generate_texture": generate_texture,
            "texture_resolution": self.settings.texture_resolution,
            "material_hints": material_hints,
            **QUALITY_PRESETS[quality],
        }
        record = self.store.create(prompt, params)
        job_id = record["job_id"]
        self.store.update(job_id, stage="validating_views")
        job_dir = self.job_dir(job_id)
        try:
            input_records = {}
            normalized = {}
            for name, source in sources.items():
                original = job_dir / "originals" / f"{name}{source.suffix.lower()}"
                input_records[name] = stage_immutable(source, original)
                normalized_path = job_dir / "views" / f"{name}.png"
                normalized_path.parent.mkdir(parents=True, exist_ok=True)
                prepare_reference(
                    original, normalized_path, minimum_size=self.settings.minimum_image_size
                )
                normalized[name] = artifact_record(normalized_path, job_dir)
            metadata = {
                "job_dir": str(job_dir),
                "inputs": input_records,
                "normalized_views": normalized,
                "required_action": None,
                "warnings": [],
            }
            self.store.update(job_id, metadata=metadata)
            atomic_json(
                job_dir / "request.json",
                {
                    "job_id": job_id,
                    "prompt": prompt,
                    "params": params,
                    "inputs": input_records,
                    "models": self._provenance(),
                },
            )
            self._enqueue(job_id)
        except Exception as exc:
            self._fail(job_id, exc)
            raise
        return self.get(job_id)

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
    def resume(self, job_id: str) -> dict[str, Any]:
        record = self.store.get(job_id)
        if record["state"] not in {"interrupted", "failed", "failed_quality"}:
            raise Codex3DError("This job is not resumable.", code="INVALID_STATE")
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
            self.store.update(job_id, state="cancelled", stage="cancelled", finished_at=utc_now())
        return {"cancel_requested": True, **self.get(job_id)}

    @manager_operation
    def cleanup(self, job_id: str, keep_final: bool = True) -> dict[str, Any]:
        record = self.store.get(job_id)
        if record["state"] not in TERMINAL_STATES:
            raise Codex3DError("Only terminal jobs may be cleaned up.", code="INVALID_STATE")
        job_dir = self.job_dir(job_id)
        preserved = {
            "master.glb",
            "game.glb",
            "lod1.glb",
            "lod2.glb",
            "manifest.json",
            "request.json",
            "qa",
            "previews",
        }
        removed = []
        for path in job_dir.iterdir():
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
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise InvalidImageError("Views must be PNG, JPEG, or WebP images.")
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
            self._fail(job_id, Codex3DError("Hunyuan queue is full.", code="QUEUE_FULL"))
            raise Codex3DError("Hunyuan queue is full.", code="QUEUE_FULL") from exc

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
        record = self.store.get(job_id)
        job_dir = self.job_dir(job_id)
        views = {name: job_dir / "views" / f"{name}.png" for name in record["params"]["views"]}
        checkpoints = self.store.checkpoints(job_id)
        if "validating_views" not in checkpoints:
            self._stage(job_id, "validating_views")
            validation = self._validate_views(views)
            self.store.merge_metadata(
                job_id, {"view_validation": validation, "warnings": validation["warnings"]}
            )
            self.store.checkpoint(job_id, "validating_views", validation)
        self._check_cancel(cancel_event)

        raw = job_dir / "raw" / "shape.glb"
        if "generating_shape" not in checkpoints or not _valid_glb(raw):
            self._stage(job_id, "generating_shape")
            shape_metrics = self.runtime.generate_shape(
                views, raw, record["params"], job_dir / "logs" / "shape.log", cancel_event
            )
            shape_metrics["artifact"] = artifact_record(raw, job_dir)
            self.store.checkpoint(job_id, "generating_shape", shape_metrics)
        self._check_cancel(cancel_event)

        source = raw
        textured = job_dir / "raw" / "textured.glb"
        if record["params"]["generate_texture"]:
            if "texturing" not in checkpoints or not _valid_glb(textured):
                self._stage(job_id, "texturing")
                texture_metrics = self.runtime.generate_texture(
                    raw,
                    views["front"],
                    textured,
                    job_dir / "logs" / "texture.log",
                    cancel_event,
                )
                texture_metrics["artifact"] = artifact_record(textured, job_dir)
                self.store.checkpoint(job_id, "texturing", texture_metrics)
            source = textured
        self._check_cancel(cancel_event)

        blender_result = job_dir / "blender-result.json"
        if (
            checkpoints.get("postprocessing", {}).get("worker_version") != 3
            or not (job_dir / "game.glb").is_file()
        ):
            self._stage(job_id, "postprocessing")
            self._run_blender(job_dir, source, blender_result)
            metrics = json.loads(blender_result.read_text(encoding="utf-8"))
            self.store.checkpoint(job_id, "postprocessing", metrics)
        else:
            metrics = checkpoints["postprocessing"]
        self._check_cancel(cancel_event)

        self._stage(job_id, "validating")
        report = self._qa(job_dir, metrics)
        self.store.checkpoint(job_id, "validating", report)
        atomic_json(job_dir / "manifest.json", self._manifest(job_id, report))
        if not report["passed"]:
            self.store.update(
                job_id,
                state="failed_quality",
                stage="validating",
                finished_at=utc_now(),
                error={"code": "QUALITY_GATES_FAILED", "message": "See qa/report.json."},
            )
            return
        self.store.merge_metadata(job_id, {"qa": report, "required_action": None})
        self.store.update(
            job_id, state="completed", stage="completed", progress=100, finished_at=utc_now()
        )

    def _validate_views(self, views: dict[str, Path]) -> dict[str, Any]:
        dimensions = {}
        aspect_ratios = []
        warnings = []
        hashes = set()
        for name, path in views.items():
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
            except (OSError, ValueError) as exc:
                raise InvalidImageError(f"Invalid {name} image: {exc}") from exc
            if min(width, height) < self.settings.minimum_image_size:
                raise InvalidImageError(
                    f"{name} must be at least {self.settings.minimum_image_size}px on both axes."
                )
            dimensions[name] = [width, height]
            aspect_ratios.append(width / height)
            digest = artifact_record(path, self.job_dir(path.parent.parent.name))["sha256"]
            if digest in hashes:
                raise InvalidImageError(f"{name} duplicates another supplied view.")
            hashes.add(digest)
        if max(aspect_ratios) - min(aspect_ratios) > 0.02:
            warnings.append("View aspect ratios differ by more than 2%.")
        if len(views) < 3:
            warnings.append("Fewer than three views reduces rear and side constraint quality.")
        return {
            "passed": True,
            "view_count": len(views),
            "dimensions": dimensions,
            "warnings": warnings,
        }

    def _run_blender(self, job_dir: Path, source: Path, result: Path) -> None:
        if not self.settings.blender_exe.is_file():
            raise Codex3DError("Blender 5.1 is unavailable.", code="BLENDER_UNAVAILABLE")
        config = {
            "input_glb": str(source),
            "output_dir": str(job_dir),
            "master_faces": 300_000,
            "game_faces": self.settings.game_faces,
            "lod_faces": list(self.settings.lod_faces),
        }
        config_path = job_dir / "blender-config.json"
        atomic_json(config_path, config)
        script = Path(__file__).with_name("blender_worker.py")
        command = [
            str(self.settings.blender_exe),
            "--background",
            "--factory-startup",
            "--python",
            str(script),
            "--",
            str(config_path),
            str(result),
        ]
        log_path = job_dir / "logs" / "blender.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                check=False,
            )
        if completed.returncode != 0 or not result.is_file():
            raise Codex3DError(
                "Blender post-processing failed; see logs/blender.log.", code="BLENDER_FAILED"
            )

    def _qa(self, job_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
        checks = []
        geometry = metrics.get("geometry", {})
        previous = None
        validator_reports: dict[str, dict[str, Any]] = {}
        face_targets = {"master": 300_000, "game": 100_000, "lod1": 50_000, "lod2": 20_000}
        profile_mode = bool(metrics.get("prepared_profiles"))
        if profile_mode:
            profile_targets = metrics.get("profile_triangle_budgets", {})
            face_targets = {
                "master": int(profile_targets.get("full_game", 0)),
                **{
                    name: int(profile_targets.get(name, 0))
                    for name in ("full_game", "mobile", "browser")
                },
            }
        for name in face_targets:
            path = job_dir / f"{name}.glb"
            faces = int(geometry.get(name, {}).get("faces", 0))
            checks.append(
                {"name": f"{name}_valid_glb", "passed": _valid_glb(path), "evidence": str(path)}
            )
            checks.append({"name": f"{name}_has_faces", "passed": faces > 0, "evidence": faces})
            if name != "master":
                variant = geometry.get(name, {})
                checks.append(
                    {
                        "name": f"{name}_watertight",
                        "passed": bool(variant.get("watertight"))
                        and variant.get("non_manifold_edges") == 0,
                        "evidence": {
                            "watertight": variant.get("watertight"),
                            "non_manifold_edges": variant.get("non_manifold_edges"),
                        },
                    }
                )
            if previous is not None:
                checks.append(
                    {
                        "name": f"{name}_not_denser",
                        "passed": faces <= previous,
                        "evidence": {"faces": faces, "previous": previous},
                    }
                )
            previous = faces
            target = face_targets[name]
            checks.append(
                {
                    "name": f"{name}_face_budget",
                    "passed": 0 < faces <= target * (1 if profile_mode else 1.05),
                    "evidence": {
                        "faces": faces,
                        "maximum": target,
                        "tolerance": "0%" if profile_mode else "5%",
                    },
                }
            )
            if path.is_file():
                validator = self._validate_glb(path, job_dir / "qa" / f"{name}-gltf.json")
                validator_reports[name] = validator
                checks.append(
                    {
                        "name": f"{name}_gltf_validator",
                        "passed": validator.get("errors") == 0,
                        "evidence": validator,
                    }
                )
        identity = geometry.get("master", {})
        checks.append(
            {
                "name": "master_identity_transform",
                "passed": _near(identity.get("location", []), [0, 0, 0])
                and _near(identity.get("rotation_euler", []), [0, 0, 0])
                and _near(identity.get("scale", []), [1, 1, 1]),
                "evidence": {
                    key: identity.get(key) for key in ("location", "rotation_euler", "scale")
                },
            }
        )
        master_faces = max(1, int(identity.get("faces", 0)))
        degenerate_faces = int(identity.get("degenerate_faces", 0))
        checks.extend(
            [
                {
                    "name": "master_watertight",
                    "passed": bool(identity.get("watertight")),
                    "evidence": {
                        "watertight": identity.get("watertight"),
                        "non_manifold_edges": identity.get("non_manifold_edges"),
                    },
                },
                {
                    "name": "master_degenerate_face_ratio",
                    "passed": degenerate_faces / master_faces < 0.0001,
                    "evidence": {
                        "degenerate_faces": degenerate_faces,
                        "faces": master_faces,
                        "ratio": degenerate_faces / master_faces,
                    },
                },
                {
                    "name": "fixed_previews",
                    "passed": all(
                        (job_dir / "previews" / f"{name}.png").is_file()
                        for name in ("front", "back", "left", "right")
                    ),
                    "evidence": str((job_dir / "previews").resolve()),
                },
            ]
        )
        if profile_mode:
            for profile in ("full_game", "mobile", "browser"):
                checks.append(
                    {
                        "name": f"{profile}_four_previews",
                        "passed": all(
                            (job_dir / "previews" / profile / f"{view}.png").is_file()
                            for view in ("front", "back", "left", "right")
                        ),
                    }
                )
            gpu = metrics.get("gpu", {})
            checks.append(
                {
                    "name": "baking_used_gpu",
                    "passed": bool(gpu.get("devices")) and gpu.get("cpu_enabled") is False,
                    "evidence": gpu,
                }
            )
            for profile in ("mobile", "browser"):
                bake = metrics.get("bakes", {}).get(profile, {})
                checks.append(
                    {
                        "name": f"{profile}_texture_coverage",
                        "passed": bool(bake.get("passed")) and bake.get("coverage", 0) >= 0.995,
                        "evidence": bake,
                    }
                )
        report = {
            "passed": all(check["passed"] for check in checks),
            "checks": checks,
            "geometry": geometry,
            "gltf_validator": validator_reports,
        }
        qa = job_dir / "qa"
        qa.mkdir(exist_ok=True)
        atomic_json(qa / "report.json", report)
        return report

    def _validate_glb(self, path: Path, report_path: Path) -> dict[str, Any]:
        if self.settings.test_mode:
            return {"errors": 0, "warnings": 0, "test_mode": True}
        if not self.settings.gltf_validator.is_file():
            return {"errors": 1, "warnings": 0, "message": "glTF Validator is missing"}
        command = [str(self.settings.gltf_validator), "--stdout", str(path)]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            check=False,
        )
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {
                "errors": int(completed.returncode != 0),
                "warnings": 0,
                "message": (completed.stderr or completed.stdout)[-2000:],
            }
        atomic_json(report_path, data)
        issues = data.get("issues", {})
        return {
            "errors": int(issues.get("numErrors", completed.returncode != 0)),
            "warnings": int(issues.get("numWarnings", 0)),
            "report_path": str(report_path.resolve()),
        }

    def _manifest(self, job_id: str, report: dict[str, Any]) -> dict[str, Any]:
        record = self.store.get(job_id)
        return {
            "job_id": job_id,
            "prompt": record["prompt"],
            "params": record["params"],
            "models": self._provenance(),
            "inputs": record["metadata"]["inputs"],
            "qa": report,
            "artifacts": self._available_artifacts(self.job_dir(job_id)),
        }

    def _provenance(self) -> dict[str, Any]:
        return {
            "shape_model": self.settings.shape_model_id,
            "shape_subfolder": self.settings.shape_subfolder,
            "texture_model": self.settings.texture_model_id,
            "shape_revision": self.settings.shape_revision,
            "texture_revision": self.settings.texture_revision,
            "upstream_commit": self.settings.upstream_commit,
        }

    def _stage(self, job_id: str, stage: str) -> None:
        self.store.update(job_id, state="running", stage=stage, progress=STAGE_PROGRESS[stage])

    @staticmethod
    def _check_cancel(event: threading.Event) -> None:
        if event.is_set():
            raise GenerationCancelled("Generation cancelled.")

    def _available_artifacts(self, job_dir: Path) -> list[dict[str, Any]]:
        if not job_dir.is_dir():
            return []
        records = []
        for path in sorted(job_dir.rglob("*")):
            if path.is_file() and not path.name.endswith((".partial", ".tmp")):
                records.append(artifact_record(path, job_dir))
        return records

    def _fail(self, job_id: str, exc: Exception) -> None:
        code = exc.code if isinstance(exc, Codex3DError) else "INTERNAL_ERROR"
        message = exc.message if isinstance(exc, Codex3DError) else str(exc)
        self.store.update(
            job_id,
            state="failed",
            finished_at=utc_now(),
            error={"code": code, "message": message},
        )

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


def _valid_glb(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 20:
        return False
    with path.open("rb") as handle:
        return handle.read(4) == b"glTF"


def _near(actual: list[Any], expected: list[float], tolerance: float = 1e-5) -> bool:
    return len(actual) == len(expected) and all(
        abs(float(left) - right) <= tolerance for left, right in zip(actual, expected, strict=True)
    )
