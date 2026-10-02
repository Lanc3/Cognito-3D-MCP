"""Durable, stage-serial asset batches with agent-owned quality repair loops.

No gate can be overridden by a retry counter. A failed candidate creates work
for the calling Codex agent; only a passing machine gate plus an explicit agent
review unlocks the next phase. Successful attempts remain immutable on disk.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from pathlib import Path
from typing import Any

from ..errors import Codex3DError, GenerationCancelled, InvalidImageError
from ..inputs import validate_image_file
from ..trellis.artifacts import artifact_record, atomic_json, sha256_file, stage_immutable
from ..trellis.store import utc_now
from .backgrounds import prepare_reference
from .config import HunyuanMVSettings
from .pipeline import QUALITY_PRESETS
from .remesh import PROFILE_PARAM_NAMES, PROFILE_TARGETS, profile_parameters
from .repair_contract import REPAIR_GATE_VERSION, get_capabilities, validate_recipe
from .repair_migration import BATCH_SCHEMA_VERSION, migrate_repair_stage

STAGES = ("references", "shape", "shape_repair", "remesh", "paint", "finish")
VIEWS = ("front", "back", "left", "right")
REPAIR_VIEWS = (*VIEWS, "bottom", "underside_front_left", "underside_back_right")
DONE = {"completed", "cancelled"}


def _stage_record(state: str = "pending") -> dict[str, Any]:
    return {
        "state": state,
        "attempt": 0,
        "approved": False,
        "gate": None,
        "error": None,
        "artifacts": [],
        "previews": {},
        "history": [],
    }


class HunyuanBatchManager:
    def __init__(self, settings: HunyuanMVSettings, processor: Any) -> None:
        self.settings = settings
        self.processor = processor
        self.root = settings.output_dir / "batches"
        self.root.mkdir(parents=True, exist_ok=True)
        self._condition = threading.Condition(threading.RLock())
        self._execution = threading.RLock()
        self._shutdown = threading.Event()
        self._cancel = threading.Event()
        self._batches: dict[str, dict[str, Any]] = {}
        self._model_metadata_cache: dict[str, tuple[Any, Any]] = {}
        self.active_resource = "idle"
        self.active_asset_id: str | None = None
        self.dashboard: Any = None
        saved = [
            json.loads(path.read_text(encoding="utf-8")) for path in self.root.glob("*/batch.json")
        ]
        for batch in sorted(saved, key=lambda item: item["created_at"]):
            migrate_repair_stage(batch)
            for asset in batch["assets"]:
                asset["params"].update(profile_parameters(asset["params"]))
            if batch["state"] not in DONE:
                if not batch.get("paused"):
                    batch["state"] = "interrupted"
                batch["paused"] = True
                for asset in batch["assets"]:
                    for record in asset["stages"].values():
                        if record["state"] in {"running", "queued"}:
                            record["state"] = "needs_repair"
                            record["error"] = {
                                "code": "INTERRUPTED",
                                "message": "Inspect retained attempt; retry this stage.",
                            }
            self._batches[batch["batch_id"]] = batch
            self._save(batch)
        self._thread = threading.Thread(target=self._loop, name="hunyuan-batch-serial", daemon=True)
        self._thread.start()

    def _save(self, batch: dict[str, Any]) -> None:
        batch["updated_at"] = utc_now()
        atomic_json(self.root / batch["batch_id"] / "batch.json", batch)

    def _batch(self, batch_id: str) -> dict[str, Any]:
        if batch_id not in self._batches:
            raise Codex3DError("Unknown batch ID", code="BATCH_NOT_FOUND")
        return self._batches[batch_id]

    def _asset(self, batch: dict[str, Any], asset_id: str) -> dict[str, Any]:
        for asset in batch["assets"]:
            if asset["asset_id"] == asset_id:
                return asset
        raise Codex3DError("Unknown asset ID", code="ASSET_NOT_FOUND")

    def _active(self) -> dict[str, Any] | None:
        return next((b for b in self._batches.values() if b["state"] not in DONE), None)

    def _require_active(self, batch: dict[str, Any]) -> None:
        if self._active() is not batch:
            raise Codex3DError("This batch is queued behind the active batch", code="BATCH_QUEUED")
        if batch["paused"] or batch["state"] in DONE:
            raise Codex3DError("Resume this batch before submitting work", code="BATCH_PAUSED")

    def create_batch(self, name: str, assets: list[dict[str, Any]]) -> dict[str, Any]:
        if not name.strip() or not 1 <= len(assets) <= 1000:
            raise ValueError("A named batch requires between 1 and 1000 assets")
        batch_id = uuid.uuid4().hex
        batch = {
            "batch_id": batch_id,
            "job_id": batch_id,
            "schema_version": BATCH_SCHEMA_VERSION,
            "name": name.strip(),
            "state": "awaiting_agent",
            "stage": "references",
            "paused": False,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "assets": [],
        }
        for index, spec in enumerate(assets):
            prompt = str(spec.get("prompt", "")).strip()
            if not prompt or len(prompt) > 10_000:
                raise ValueError("Every asset requires a prompt of 1–10000 characters")
            quality = spec.get("quality", "standard")
            if quality not in QUALITY_PRESETS:
                raise ValueError("quality must be draft, standard, or high")
            seed = int(spec.get("seed", 42))
            if not 0 <= seed <= 2**32 - 1:
                raise ValueError("seed must be an unsigned 32-bit integer")
            params = {
                "quality": quality,
                "seed": seed,
                **QUALITY_PRESETS[quality],
                "texture_resolution": self.settings.texture_resolution,
                "memory_limit_mb": 2048,
                "min_available_mb": 2048,
                "threads": self.settings.remesh_threads,
                **profile_parameters(spec),
            }
            params["target_quads"] = params["full_game_target_quads"]
            batch["assets"].append(
                {
                    "asset_id": uuid.uuid4().hex,
                    "name": str(spec.get("name", f"Asset {index + 1}")),
                    "prompt": prompt,
                    "material_hints": str(spec.get("material_hints", "")),
                    "params": params,
                    "repair_idempotency": {},
                    "stages": {
                        s: _stage_record("awaiting_images" if s == "references" else "pending")
                        for s in STAGES
                    },
                    "supplied_views": spec.get("views", {}),
                }
            )
        with self._condition:
            self._batches[batch_id] = batch
            self._save(batch)
            self._condition.notify_all()
        if self.dashboard is not None:
            self.dashboard.start(open_browser=False)
            if self.settings.dashboard_auto_open:
                self.dashboard.open()
        return self.get_batch_status(batch_id)

    def has_batch(self, batch_id: str) -> bool:
        with self._condition:
            return batch_id in self._batches

    def _attempt_directory(self, batch, asset, stage, attempt) -> Path:
        return self.root / batch["batch_id"] / asset["asset_id"] / stage / f"attempt-{attempt:04d}"

    def _new_attempt(self, batch, asset, stage) -> tuple[dict[str, Any], Path]:
        record = asset["stages"][stage]
        if record["attempt"]:
            previous = record.pop("previous_attempt", None)
            record["history"].append(
                previous or {k: copy.deepcopy(v) for k, v in record.items() if k != "history"}
            )
        history = record["history"]
        attempt = record["attempt"] + 1
        record.clear()
        record.update(_stage_record("running"))
        record.update(
            attempt=attempt,
            history=history,
            started_at=utc_now(),
            params_snapshot=copy.deepcopy(asset["params"]),
        )
        if stage == "shape_repair":
            self._ensure_repair_request(asset)
            record["repair_request"] = copy.deepcopy(asset["shape_repair_request"])
        directory = self._attempt_directory(batch, asset, stage, attempt)
        directory.mkdir(parents=True, exist_ok=False)
        return record, directory

    @staticmethod
    def _verified_output(record: dict[str, Any]) -> dict[str, Any]:
        path = Path(record.get("output") or "").resolve()
        artifact = next((value for value in record.get("artifacts", [])
                         if Path(value["path"]).resolve() == path), None)
        if (not artifact or not path.is_file() or path.suffix.lower() != ".glb"
                or path.stat().st_size != artifact.get("bytes")
                or sha256_file(path) != artifact.get("sha256")):
            raise Codex3DError("Source artifact is missing or changed", code="SHAPE_CHANGED")
        return {"path": str(path), "sha256": artifact["sha256"], "bytes": artifact["bytes"]}

    @staticmethod
    def _shape_generated(asset: dict[str, Any]) -> bool:
        """Generation completion is not approval of raw topology or identity."""
        record = asset["stages"]["shape"]
        if record["state"] in {"pending", "queued", "running", "needs_evidence"}:
            return False
        path = Path(record.get("output") or "")
        gate = record.get("gate") or {}
        compute = gate.get("compute", gate)
        faces = (compute.get("geometry") or {}).get("faces", 0)
        return bool(path.is_file() and path.stat().st_size > 0
                    and (compute.get("passed") is True or faces > 0)
                    and not record.get("stale"))

    @staticmethod
    def _invalidate_from(asset: dict[str, Any], stage: str) -> None:
        if STAGES.index(stage) <= STAGES.index("shape_repair"):
            prior = asset.pop("accepted_master", None)
            if prior:
                asset.setdefault("accepted_master_history", []).append(copy.deepcopy(prior))
        if STAGES.index(stage) < STAGES.index("shape_repair"):
            asset.pop("shape_repair_request", None)
        for downstream in STAGES[STAGES.index(stage):]:
            record = asset["stages"][downstream]
            if record["attempt"] and "previous_attempt" not in record:
                record["previous_attempt"] = {
                    key: copy.deepcopy(value) for key, value in record.items() if key != "history"
                }
            record.update(approved=False, stale=True, state="pending")

    def _ensure_repair_request(self, asset: dict[str, Any]) -> None:
        if asset.get("shape_repair_request"):
            return
        source = self._verified_output(asset["stages"]["shape"])
        asset["shape_repair_request"] = {
            "request_id": uuid.uuid4().hex,
            "source_path": source["path"], "source_sha256": source["sha256"],
            "source_shape_sha256": source["sha256"],
            "source_shape_attempt": asset["stages"]["shape"]["attempt"],
            "source_attempt": 0,
            "recipe": validate_recipe({"method": "analyze", "parameters": {}}),
            "gate_version": REPAIR_GATE_VERSION,
            "diagnosis": "Analyze the immutable raw shape before master acceptance",
            "idempotency_key": None,
        }

    def get_shape_repair_capabilities(self) -> dict[str, Any]:
        runtime_root = self.settings.python_exe.parent.parent
        package_paths = []
        extra = getattr(self.settings, "repair_package_dir", None)
        if extra:
            package_paths.append(Path(extra))
        package_paths.append(runtime_root / "Lib" / "site-packages")
        package_paths.extend((runtime_root / "lib").glob("python*/site-packages"))
        package_paths = list(dict.fromkeys(str(path.resolve()) for path in package_paths))
        # One resolver owns CPU and GPU readiness. In particular, core CPU
        # packages must never overwrite a CUDA method's qualification result.
        capabilities = get_capabilities(
            package_paths=package_paths, worker_python=str(self.settings.python_exe),
        )
        return {
            **capabilities, "gate_version": REPAIR_GATE_VERSION,
            "worker_python": str(self.settings.python_exe),
            "worker_python_present": self.settings.python_exe.is_file(),
            "worker_package_paths": package_paths,
            "discovery_scope": (
                "Configured worker roots only; package metadata, native hashes and recorded "
                "qualification evidence. No native imports or CUDA probes run in the MCP process."
            ),
            "serial_execution": True, "raw_shape_is_immutable": True,
            "import_requires_same_gates": True, "required_review_views": list(REPAIR_VIEWS),
        }

    def _require_repair_edit(self, batch: dict[str, Any]) -> None:
        if self._active() is not batch or batch["state"] in DONE:
            raise Codex3DError("This batch cannot accept repair work", code="BATCH_QUEUED")
        if STAGES.index(batch["stage"]) < STAGES.index("shape"):
            raise Codex3DError("Complete references and shape generation first", code="STAGE_BARRIER")
        if any(record["state"] == "running" for asset in batch["assets"]
               for record in asset["stages"].values()):
            raise Codex3DError("Wait for the active asset boundary", code="ASSET_BUSY")

    def _repair_source(self, asset, source_attempt: int) -> dict[str, Any]:
        raw = self._verified_output(asset["stages"]["shape"])
        if type(source_attempt) is not int or source_attempt < 0:
            raise ValueError("source_attempt must be zero for raw or a positive repair attempt")
        if source_attempt == 0:
            return raw
        stage = asset["stages"]["shape_repair"]
        record = next((item for item in [stage, *stage.get("history", [])]
                       if item.get("attempt") == source_attempt), None)
        if not record or (record.get("repair_request") or {}).get("source_shape_sha256") != raw["sha256"]:
            raise Codex3DError("Repair source belongs to a stale raw shape", code="STALE_REPAIR")
        return self._verified_output(record)

    def create_shape_repair_attempt(
        self, batch_id: str, asset_id: str, recipe: dict[str, Any], source_sha256: str,
        diagnosis: str, idempotency_key: str, *, source_attempt: int = 0,
        expected_shape_attempt: int | None = None, inspected_paths: list[str] | None = None,
        import_path: str | None = None, import_sha256: str | None = None,
    ) -> dict[str, Any]:
        recipe = validate_recipe(recipe)
        if (expected_shape_attempt is not None
                and (type(expected_shape_attempt) is not int or expected_shape_attempt < 1)):
            raise ValueError("expected_shape_attempt must be a positive integer")
        if not diagnosis.strip() or not idempotency_key.strip():
            raise ValueError("Repair requires diagnosis and an idempotency key")
        if len(idempotency_key) > 200 or len(diagnosis) > 10000:
            raise ValueError("Repair key or diagnosis is too long")
        if recipe["method"] == "import_candidate":
            if not import_path or not import_sha256:
                raise ValueError("Import repair requires the candidate path and expected SHA-256")
        elif import_path is not None or import_sha256 is not None:
            raise ValueError("Only import_candidate may supply an imported mesh")
        input_fingerprint = hashlib.sha256(json.dumps({
            "recipe": recipe, "source_sha256": source_sha256, "source_attempt": source_attempt,
            "expected_shape_attempt": expected_shape_attempt, "diagnosis": diagnosis,
            "inspected_paths": sorted(str(Path(value).resolve()) for value in inspected_paths or []),
            "import_path": str(Path(import_path).expanduser().resolve()) if import_path else None,
            "import_sha256": import_sha256,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        with self._condition:
            batch = self._batch(batch_id)
            asset = self._asset(batch, asset_id)
            known = asset.setdefault("repair_idempotency", {}).get(idempotency_key)
            if known and "input_fingerprint" in known:
                if known["input_fingerprint"] != input_fingerprint:
                    raise Codex3DError("Idempotency key already has another request", code="IDEMPOTENCY_CONFLICT")
                return self.get_batch_status(batch_id)
            self._require_repair_edit(batch)
            if not all(self._shape_generated(item) for item in batch["assets"]):
                raise Codex3DError("Complete the raw shape phase before repair", code="STAGE_BARRIER")
            raw_record = asset["stages"]["shape"]
            if expected_shape_attempt is not None and expected_shape_attempt != raw_record["attempt"]:
                raise Codex3DError("Raw shape attempt changed", code="STALE_REPAIR")
            raw = self._verified_output(raw_record)
            source = self._repair_source(asset, source_attempt)
            if source["sha256"] != source_sha256:
                raise Codex3DError("Repair source hash changed", code="STALE_REPAIR")
            inspected = []
            asset_root = (self.root / batch_id / asset_id).resolve()
            for value in inspected_paths or []:
                path = Path(value).resolve()
                if not path.is_relative_to(asset_root) or not path.is_file():
                    raise ValueError("Diagnosis evidence must belong to this asset")
                inspected.append(artifact_record(path, asset_root))
            request = {
                "source_path": source["path"], "source_sha256": source["sha256"],
                "source_shape_sha256": raw["sha256"],
                "source_shape_attempt": raw_record["attempt"], "source_attempt": source_attempt,
                "recipe": recipe, "gate_version": REPAIR_GATE_VERSION,
                "diagnosis": diagnosis, "diagnosis_evidence": inspected,
                "idempotency_key": idempotency_key,
            }
            if import_path:
                path = Path(import_path).expanduser().resolve()
                if (not self.settings.is_allowed_input(path) or not path.is_file()
                        or path.suffix.lower() != ".glb" or not 0 < path.stat().st_size <= 256 * 1024**2):
                    raise ValueError("Import a nonempty GLB of at most 256 MiB from an allowed root")
                if sha256_file(path) != import_sha256:
                    raise Codex3DError("Imported candidate hash changed", code="STALE_REPAIR")
                request.update(import_path=str(path), import_sha256=import_sha256)
            encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
            fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
            known = asset.setdefault("repair_idempotency", {}).get(idempotency_key)
            if known:
                if known["fingerprint"] != fingerprint:
                    raise Codex3DError("Idempotency key already has another request", code="IDEMPOTENCY_CONFLICT")
                return self.get_batch_status(batch_id)
            # Quality retries require different geometry inputs, recipe or policy.
            stage = asset["stages"]["shape_repair"]
            for previous in [stage, *stage.get("history", [])]:
                old = previous.get("repair_request") or {}
                same = all(old.get(key) == request.get(key) for key in
                           ("source_sha256", "source_shape_sha256", "recipe", "import_sha256", "gate_version"))
                operational = (previous.get("error") or {}).get("code") not in {
                    None, "AGENT_REJECTED", "QUALITY_GATES_FAILED"
                }
                if old and same and not operational:
                    raise Codex3DError("This quality attempt already exists; revise its cause", code="UNCHANGED_REPAIR")
            self._invalidate_from(asset, "shape_repair")
            request["request_id"] = uuid.uuid4().hex
            asset["shape_repair_request"] = request
            asset["repair_idempotency"][idempotency_key] = {
                "fingerprint": fingerprint, "input_fingerprint": input_fingerprint,
                "request_id": request["request_id"]
            }
            asset["stages"]["shape_repair"]["state"] = "queued"
            batch["stage"] = "shape_repair"
            batch["state"] = "paused" if batch["paused"] else "queued"
            self._save(batch)
            self._condition.notify_all()
        return self.get_batch_status(batch_id)

    def submit_shape_repair_candidate(
        self, batch_id: str, asset_id: str, candidate_path: str, candidate_sha256: str,
        source_sha256: str, recipe: dict[str, Any], diagnosis: str, idempotency_key: str,
        *, source_attempt: int = 0, expected_shape_attempt: int | None = None,
        inspected_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        if recipe.get("method") != "import_candidate":
            raise ValueError("Imported candidates require the import_candidate recipe")
        return self.create_shape_repair_attempt(
            batch_id, asset_id, recipe, source_sha256, diagnosis, idempotency_key,
            source_attempt=source_attempt, expected_shape_attempt=expected_shape_attempt,
            inspected_paths=inspected_paths, import_path=candidate_path, import_sha256=candidate_sha256,
        )

    def _prepare_repair_inputs(self, asset, record, directory) -> None:
        request = copy.deepcopy(record["repair_request"])
        raw = self._verified_output(asset["stages"]["shape"])
        if (raw["sha256"] != request["source_shape_sha256"]
                or asset["stages"]["shape"]["attempt"] != request["source_shape_attempt"]):
            raise Codex3DError("Raw shape changed before repair execution", code="STALE_REPAIR")
        if sha256_file(Path(request["source_path"])) != request["source_sha256"]:
            raise Codex3DError("Repair source changed before execution", code="STALE_REPAIR")
        if request["recipe"]["method"] == "import_candidate":
            path = Path(request["import_path"]).resolve()
            if not self.settings.is_allowed_input(path):
                raise ValueError("Imported source escaped allowed roots")
            staged = stage_immutable(path, directory / "inputs" / "imported.glb")
            if staged["sha256"] != request["import_sha256"]:
                raise Codex3DError("Imported candidate changed during staging", code="STALE_REPAIR")
            request["import_original_path"] = str(path)
            request["import_path"] = staged["path"]
        record["repair_request"] = request
        asset["shape_repair_request"] = copy.deepcopy(request)
        atomic_json(directory / "repair-attempt.json", request)

    def _repair_master(self, asset, record, directory, review) -> dict[str, Any]:
        gate = record.get("gate") or {}
        certificate = gate.get("compute", gate)
        request = record.get("repair_request") or {}
        raw = self._verified_output(asset["stages"]["shape"])
        output = self._verified_output(record)
        if not Path(output["path"]).is_relative_to(directory.resolve()):
            raise ValueError("Repair output must belong to its immutable attempt")
        if (not gate.get("passed") or certificate.get("gate_version") != REPAIR_GATE_VERSION
                or certificate.get("source_sha256") != request.get("source_sha256")
                or certificate.get("output_sha256") != output["sha256"]
                or request.get("source_shape_sha256") != raw["sha256"]
                or request.get("source_shape_attempt") != asset["stages"]["shape"]["attempt"]):
            raise Codex3DError("Repair certificate is stale or incomplete", code="STALE_REPAIR")
        report = Path(record.get("repair_report") or "").resolve()
        if not report.is_file() or not report.is_relative_to(directory.resolve()):
            raise ValueError("Repair report must belong to this attempt")
        return {
            **output, "source_shape_sha256": raw["sha256"],
            "source_shape_attempt": asset["stages"]["shape"]["attempt"],
            "source_sha256": request["source_sha256"], "attempt": record["attempt"],
            "request_id": request["request_id"], "gate_version": REPAIR_GATE_VERSION,
            "repair_report": str(report), "repair_report_sha256": sha256_file(report),
            "review": review, "accepted_at": utc_now(),
        }

    def _phase_complete(self, batch, stage) -> bool:
        if stage == "shape":
            return all(self._shape_generated(asset) for asset in batch["assets"])
        if stage == "shape_repair":
            for asset in batch["assets"]:
                record = asset["stages"][stage]
                master = asset.get("accepted_master") or {}
                raw = asset["stages"]["shape"]
                raw_artifact = next((item for item in raw.get("artifacts", [])
                                     if item.get("path") == raw.get("output")), {})
                if (not record.get("approved") or not (record.get("gate") or {}).get("passed")
                        or master.get("gate_version") != REPAIR_GATE_VERSION
                        or master.get("attempt") != record.get("attempt")
                        or master.get("source_shape_attempt") != raw.get("attempt")
                        or master.get("source_shape_sha256") != raw_artifact.get("sha256")
                        or not master.get("path") or not master.get("sha256")):
                    return False
            return True
        return all(asset["stages"][stage]["approved"] for asset in batch["assets"])

    @staticmethod
    def _queue_phase(batch, stage) -> None:
        batch.update(stage=stage, state="queued")
        for asset in batch["assets"]:
            record = asset["stages"][stage]
            # Unchanged reviews/rejections stay retained when a phase is revisited.
            if record["state"] == "pending":
                record["state"] = "queued"

    def submit_references(
        self,
        batch_id: str,
        asset_id: str,
        views: dict[str, str],
        *,
        background_mode: str = "auto",
        key_color=(255, 0, 255),
    ) -> dict[str, Any]:
        # Preparing later batches during another compute phase is deliberately refused.
        with self._condition:
            self._require_active(self._batch(batch_id))
            if self._batch(batch_id)["stage"] != "references":
                raise Codex3DError("Reference phase is not active", code="WRONG_STAGE")
        with self._execution, self._condition:
            batch = self._batch(batch_id)
            self._require_active(batch)
            if batch["stage"] != "references":
                raise Codex3DError(
                    "Rewind the asset to references before replacing images", code="WRONG_STAGE"
                )
            asset = self._asset(batch, asset_id)
            record, directory = self._new_attempt(batch, asset, "references")
            try:
                if set(views) != set(VIEWS) or not all(views.values()):
                    raise InvalidImageError(
                        "Every batch asset requires front, back, left and right"
                    )
                prepared = {}
                reports = {}
                hashes = set()
                sizes = []
                for view in VIEWS:
                    source = Path(views[view]).expanduser().resolve()
                    if not source.is_file() or not self.settings.is_allowed_input(source):
                        raise InvalidImageError(
                            f"{view}: use an image inside an allowed input root"
                        )
                    validate_image_file(source)
                    original = directory / "originals" / f"{view}{source.suffix.lower()}"
                    staged = stage_immutable(source, original)
                    if staged["sha256"] in hashes:
                        raise InvalidImageError(f"{view}: duplicates another supplied view")
                    hashes.add(staged["sha256"])
                    target = directory / "views" / f"{view}.png"
                    reports[view] = prepare_reference(
                        original,
                        target,
                        background_mode=background_mode,
                        key_color=tuple(key_color),
                        minimum_size=self.settings.minimum_image_size,
                    )
                    prepared[view] = str(target.resolve())
                    sizes.append((reports[view]["width"], reports[view]["height"]))
                if any(w != h for w, h in sizes) or len(set(sizes)) != 1:
                    raise InvalidImageError("All four views must use matching square dimensions")
                record.update(
                    state="awaiting_review",
                    gate={"passed": True, "views": reports},
                    prepared_views=prepared,
                    previews={v: prepared[v] for v in VIEWS},
                )
            except Exception as exc:
                record.update(
                    state="needs_repair",
                    gate={"passed": False},
                    error={"code": getattr(exc, "code", "REFERENCE_FAILED"), "message": str(exc)},
                )
            self._write_attempt_gate(record, directory)
            self._save(batch)
            self._condition.notify_all()
        return self.get_batch_status(batch_id)

    def review_asset(
        self,
        batch_id: str,
        asset_id: str,
        stage: str,
        attempt: int,
        approved: bool,
        notes: str,
        inspected_paths: list[str],
    ) -> dict[str, Any]:
        if stage not in STAGES or not notes.strip() or not inspected_paths:
            raise ValueError("Review needs a stage, specific notes, and inspected evidence paths")
        with self._condition:
            batch = self._batch(batch_id)
            self._require_active(batch)
            asset = self._asset(batch, asset_id)
            record = asset["stages"][stage]
            if batch["stage"] != stage or record["attempt"] != attempt:
                raise Codex3DError("Review refers to a stale stage or attempt", code="STALE_REVIEW")
            if record["state"] != "awaiting_review" or not record.get("gate", {}).get("passed"):
                raise Codex3DError(
                    "The machine gate must pass before agent approval", code="GATE_FAILED"
                )
            directory = self._attempt_directory(batch, asset, stage, attempt).resolve()
            evidence = []
            for value in inspected_paths:
                path = Path(value).resolve()
                if not path.is_relative_to(directory) or not path.is_file():
                    raise ValueError("Review evidence must belong to the current immutable attempt")
                observed = artifact_record(path, directory)
                retained = next((item for item in record.get("artifacts", [])
                                 if Path(item["path"]).resolve() == path), None)
                if (not retained or retained.get("sha256") != observed["sha256"]
                        or retained.get("bytes") != observed["bytes"]):
                    raise ValueError("Review evidence changed since this attempt was recorded")
                evidence.append(observed)
            if approved and stage == "remesh" and "profiles" in record:
                expected = {
                    f"{name}_{view}" for name in PROFILE_TARGETS for view in VIEWS
                }
                previews = record.get("previews", {})
                inspected = {str(Path(value).resolve()) for value in inspected_paths}
                if set(previews) != expected or not {
                    str(Path(path).resolve()) for path in previews.values()
                }.issubset(inspected):
                    raise ValueError(
                        "Inspect all four views of every remesh profile before approval"
                    )
            review = {"notes": notes, "inspected": evidence, "at": utc_now()}
            master = None
            if approved and stage == "shape_repair":
                previews = record.get("previews", {})
                inspected = {str(Path(value).resolve()) for value in inspected_paths}
                required = {str(Path(value).resolve()) for value in previews.values()}
                required.add(str(Path(record.get("repair_report") or "").resolve()))
                if not set(REPAIR_VIEWS).issubset(previews) or not required.issubset(inspected):
                    raise ValueError("Inspect all seven repair views and the repair report before approval")
                # Verify certificate, source and candidate before mutating either approval.
                master = self._repair_master(asset, record, directory, review)
            record.update(
                approved=bool(approved),
                state="approved" if approved else "needs_repair",
                review=review,
                error=None if approved else {"code": "AGENT_REJECTED", "message": notes},
            )
            if master is not None:
                asset["accepted_master"] = master
                record["stale"] = False
                atomic_json(directory / "accepted-master.json", master)
            atomic_json(directory / "agent-review.json", record["review"])
            if stage == "finish" and (directory / "manifest.json").is_file():
                manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                manifest.update(agent_approved=bool(approved), agent_review=record["review"])
                atomic_json(directory / "manifest.json", manifest)
            self._save(batch)
            self._condition.notify_all()
        return self.get_batch_status(batch_id)

    def retry_asset(
        self, batch_id: str, asset_id: str, stage: str, changes: dict[str, Any], repair_note: str
    ) -> dict[str, Any]:
        if stage not in STAGES or not repair_note.strip():
            raise ValueError("A retry requires a stage and the agent's repair rationale")
        if stage == "shape_repair":
            raise ValueError("Use create_shape_repair_attempt with a typed recipe and source hash")
        allowed = {
            "seed",
            "quality",
            "steps",
            "guidance_scale",
            "octree_resolution",
            "num_chunks",
            "target_quads",
            "edge_scaling",
            "sharp_edge",
            "smooth_normal",
            "adaptivity",
            "anisotropy",
            "texture_resolution",
            "threads",
            "memory_limit_mb",
            "min_available_mb",
        } | PROFILE_PARAM_NAMES
        if set(changes) - allowed:
            raise ValueError(f"Unsupported repair settings: {sorted(set(changes) - allowed)}")
        with self._condition:
            batch = self._batch(batch_id)
            self._require_active(batch)
            asset = self._asset(batch, asset_id)
            if STAGES.index(stage) > STAGES.index(batch["stage"]):
                raise Codex3DError("Cannot skip a batch phase", code="STAGE_BARRIER")
            if any(a["stages"][batch["stage"]]["state"] == "running" for a in batch["assets"]):
                raise Codex3DError(
                    "Wait for the active asset boundary before revising work", code="ASSET_BUSY"
                )
            if STAGES.index(stage) > STAGES.index("remesh") and set(changes) & (
                PROFILE_PARAM_NAMES | {"target_quads"}
            ):
                raise ValueError(
                    "Profile geometry changes require retrying remesh or an earlier stage"
                )
            params = dict(asset["params"])
            if "quality" in changes:
                if changes["quality"] not in QUALITY_PRESETS:
                    raise ValueError("Invalid quality preset")
                params.update(QUALITY_PRESETS[changes["quality"]])
            params.update(changes)
            if "target_quads" in changes and "full_game_target_quads" not in changes:
                params["full_game_target_quads"] = changes["target_quads"]
            for profile in PROFILE_TARGETS:
                target_key, budget_key = f"{profile}_target_quads", f"{profile}_triangle_budget"
                changed = target_key in changes or (
                    profile == "full_game" and "target_quads" in changes
                )
                if changed and budget_key not in changes:
                    params.pop(budget_key, None)
            params.update(profile_parameters(params))
            params["target_quads"] = params["full_game_target_quads"]
            if not 0 <= int(params["seed"]) <= 2**32 - 1:
                raise ValueError("Invalid seed")
            limits = {
                "steps": (1, 80),
                "guidance_scale": (0, 20),
                "octree_resolution": (64, 384),
                "num_chunks": (1000, 40000),
                "target_quads": (1000, 150000),
                "texture_resolution": (1024, 4096),
                "edge_scaling": (1, 4),
                "sharp_edge": (30, 180),
                "smooth_normal": (0, 180),
                "adaptivity": (0, 1),
                "anisotropy": (0, 1),
                "threads": (1, self.settings.remesh_threads),
                "memory_limit_mb": (512, 4096),
                "min_available_mb": (2048, 65536),
            }
            for key, (low, high) in limits.items():
                if key in params and not low <= float(params[key]) <= high:
                    raise ValueError(f"{key} must be between {low} and {high}")
            error = asset["stages"][stage].get("error") or {}
            if (stage != "references" and params == asset["params"]
                    and error.get("code") in {"AGENT_REJECTED", "QUALITY_GATES_FAILED"}):
                raise Codex3DError("Quality retry must change its cause", code="UNCHANGED_REPAIR")
            lower_profile_keys = {
                f"{profile}_{setting}" for profile in ("mobile", "browser")
                for setting in ("target_quads", "triangle_budget")
            }
            changed_keys = {key for key in set(params) | set(asset["params"])
                            if params.get(key) != asset["params"].get(key)}
            paint = asset["stages"]["paint"]
            preserved_paint = (
                copy.deepcopy(paint) if stage == "remesh" and changed_keys
                and changed_keys.issubset(lower_profile_keys) and paint.get("approved")
                and paint.get("source_profile_sha256") else None
            )
            asset["params"] = params
            self._invalidate_from(asset, stage)
            if preserved_paint is not None:
                asset["stages"]["paint"] = preserved_paint
            asset["stages"][stage]["state"] = "awaiting_images" if stage == "references" else "queued"
            asset["stages"][stage]["repair_note"] = repair_note
            batch.update(stage=stage, state="awaiting_agent" if stage == "references" else "queued")
            self._save(batch)
            self._condition.notify_all()
        return self.get_batch_status(batch_id)

    def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        with self._condition:
            batch = copy.deepcopy(self._batch(batch_id))
            batch["active_resource"] = (
                self.active_resource
                if self._active() and self._active()["batch_id"] == batch_id
                else "idle"
            )
            batch["active_asset_id"] = self.active_asset_id
            batch["dashboard_url"] = self.dashboard.url if self.dashboard else None
            batch["agent_work"] = self._agent_work(self._batch(batch_id), 20)
            return batch

    def _agent_work(self, batch, limit):
        if self._active() is not batch or batch["state"] in DONE or batch["paused"]:
            return []
        stage = batch["stage"]
        work = []
        qualified_gpu_methods = []
        capability_error = None
        if stage == "shape_repair" and any(
            asset["stages"][stage]["state"] == "needs_repair" for asset in batch["assets"]
        ):
            try:
                capabilities = self.get_shape_repair_capabilities()
                if capabilities.get("full_gate_discoverable") and capabilities.get("worker_python_present"):
                    qualified_gpu_methods = [
                        method for method, value in capabilities.get("methods", {}).items()
                        if value.get("backend") == "cumesh"
                        and value.get("discoverable") is True
                        and value.get("runtime_qualified") is True
                        and value.get("production_ready") is True
                    ]
            except Exception as exc:
                # Optional recommendation discovery cannot hide the actual
                # failed attempt or turn unavailable methods into recommendations.
                capability_error = {"type": type(exc).__name__, "message": str(exc)}
        for asset in batch["assets"]:
            record = asset["stages"][stage]
            state = record["state"]
            if state not in {"awaiting_images", "awaiting_review", "needs_repair"}:
                continue
            work.append(
                {
                    "batch_id": batch["batch_id"],
                    "asset_id": asset["asset_id"],
                    "name": asset["name"],
                    "stage": stage,
                    "attempt": record["attempt"],
                    "action": "generate_references"
                    if state == "awaiting_images"
                    else "inspect_and_review"
                    if state == "awaiting_review"
                    else "inspect_repair_retry",
                    "prompt": asset["prompt"],
                    "params": asset["params"],
                    "profiles": record.get("profiles", {}),
                    "supplied_views": asset.get("supplied_views", {}),
                    "error": record.get("error"),
                    "gate": record.get("gate"),
                    "evidence": record.get("artifacts", []),
                    "previews": record.get("previews", {}),
                    "repair_request": record.get("repair_request"),
                    "repair_report": record.get("repair_report"),
                    "accepted_master": asset.get("accepted_master"),
                    "instruction": (
                        "The Codex agent owns this loop. Inspect images and reports. Repair or "
                        "regenerate the failed stage, submit a NEW attempt, then review it. "
                        "Never bypass a gate or proceed to a later phase. Continue until every "
                        "asset passes. Rewind a causal earlier stage when needed; successful "
                        "assets stay checkpointed."
                    ),
                }
            )
            if stage == "shape_repair" and state == "needs_repair":
                work[-1]["automatic_cpu_fallback"] = False
                if qualified_gpu_methods:
                    work[-1]["qualified_gpu_repair_options"] = list(qualified_gpu_methods)
                    work[-1]["instruction"] += (
                        " Qualified GPU repair options are available: "
                        + ", ".join(qualified_gpu_methods)
                        + ". Prefer a GPU recipe only when it addresses the measured defect. "
                        "Choose and submit it explicitly; the queue never switches methods "
                        "or falls back to CPU automatically. All common gates and review still apply."
                    )
                if capability_error:
                    work[-1]["gpu_capability_discovery_error"] = capability_error
            if len(work) >= limit:
                break
        return copy.deepcopy(work)

    def get_agent_work(self, batch_id: str | None = None, limit: int = 20) -> dict[str, Any]:
        with self._condition:
            batch = self._batch(batch_id) if batch_id else self._active()
            return {
                "batch_id": batch["batch_id"] if batch else None,
                "state": batch["state"] if batch else "idle",
                "work": self._agent_work(batch, max(1, min(100, limit))) if batch else [],
            }

    def _records(self, directory: Path) -> list[dict[str, Any]]:
        return [
            artifact_record(p, directory)
            for p in sorted(directory.rglob("*"))
            if p.is_file() and not p.name.endswith((".tmp", ".partial"))
        ]

    def _write_attempt_gate(self, record: dict[str, Any], directory: Path) -> None:
        # The certificate cannot contain its own digest. Hash its final bytes only
        # in batch state, after writing its other artifact hashes and timestamps.
        certificate = directory / "gate.json"
        record["artifacts"] = [
            item for item in self._records(directory)
            if Path(item["path"]).resolve() != certificate.resolve()
        ]
        atomic_json(certificate, {k: v for k, v in record.items() if k != "history"})
        record["artifacts"].append(artifact_record(certificate, directory))

    @staticmethod
    def _repair_failure_kind(record: dict[str, Any], directory: Path) -> str | None:
        """Distinguish unevaluated operations from evaluated geometry rejection."""
        operational = {"runtime_unavailable", "runtime_failure", "resource", "operational"}

        def visit(value: Any) -> str | None:
            if not isinstance(value, dict):
                return None
            kind = value.get("failure_kind")
            if kind in operational:
                return kind
            for child in (value.get("compute"), value.get("error"), *value.get("checks", [])):
                found = visit(child)
                if found:
                    return found
            # Older worker reports use this precise detector result: an
            # unevaluated check means its backend never produced an answer.
            if (value.get("name") == "export_self_intersections"
                    and value.get("evaluated") is False
                    and value.get("backend") == "pymeshlab"):
                return "runtime_unavailable"
            return None

        found = visit(record) or visit(record.get("gate"))
        if found:
            return found
        report_path = Path(record.get("repair_report") or "").resolve()
        if report_path.is_relative_to(directory.resolve()) and report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            found = visit(report)
            if found:
                return found
            # Compatibility with reports written before typed failure_kind.
            # ValueError describes recipe/geometry rejection and stays quality.
            error_type = (report.get("error") or {}).get("type")
            return {
                "ImportError": "runtime_unavailable",
                "ModuleNotFoundError": "runtime_unavailable",
                "MemoryError": "resource",
                "RuntimeError": "runtime_failure",
            }.get(error_type)
        return None

    def _release_resources(self, batch) -> bool:
        try:
            self.processor.end_stage()
            if batch:
                batch.pop("resource_error", None)
            return True
        except Exception as exc:
            self.active_resource = "worker shutdown failed"
            if batch:
                batch.update(
                    paused=True,
                    state="resource_blocked",
                    resource_error={
                        "code": getattr(exc, "code", "WORKER_BUSY"),
                        "message": str(exc),
                    },
                )
                self._save(batch)
            return False

    def _loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                task = None
                with self._condition:
                    if self._shutdown.is_set():
                        break
                    batch = self._active()
                    if batch and not batch["paused"]:
                        stage = batch["stage"]
                        if self._phase_complete(batch, stage):
                            # Close and join the old process BEFORE advancing the phase.
                            if not self._release_resources(batch):
                                continue
                            self.active_resource = "idle"
                            if stage == "finish":
                                batch.update(state="completed", finished_at=utc_now())
                            else:
                                stage = STAGES[STAGES.index(stage) + 1]
                                self._queue_phase(batch, stage)
                            self._save(batch)
                            continue
                        if stage != "references":
                            # Compute the entire phase before unloading and rendering evidence.
                            asset = next(
                                (
                                    a
                                    for a in batch["assets"]
                                    if a["stages"][stage]["state"] == "queued"
                                ),
                                None,
                            )
                            evidence = False
                            if asset is None:
                                if not self._release_resources(batch):
                                    continue
                                self.active_resource = "idle"
                                asset = next(
                                    (
                                        a
                                        for a in batch["assets"]
                                        if a["stages"][stage]["state"] == "needs_evidence"
                                    ),
                                    None,
                                )
                                evidence = asset is not None
                            if asset is not None:
                                if evidence:
                                    record = asset["stages"][stage]
                                    directory = self._attempt_directory(
                                        batch, asset, stage, record["attempt"]
                                    )
                                    record["state"] = "running"
                                else:
                                    try:
                                        record, directory = self._new_attempt(batch, asset, stage)
                                    except Exception as exc:
                                        asset["stages"][stage].update(
                                            state="needs_repair", approved=False,
                                            error={"code": getattr(exc, "code", "STAGE_FAILED"),
                                                   "message": str(exc)},
                                        )
                                        self._save(batch)
                                        continue
                                self._cancel.clear()
                                self.active_asset_id = asset["asset_id"]
                                self.active_resource = (
                                    "GPU preview"
                                    if evidence
                                    else {
                                        "shape": "GPU shape",
                                        "shape_repair": "Shape diagnosis / bounded CPU repair",
                                        "remesh": "CPU remesh "
                                        f"({self.settings.remesh_threads} cores)",
                                        "paint": "GPU paint",
                                        "finish": "GPU preview / limited CPU export",
                                    }[stage]
                                )
                                batch["state"] = "running"
                                task = (batch, asset, stage, record, directory, evidence)
                                self._save(batch)
                    if task is None:
                        if not self._release_resources(batch):
                            self._condition.wait(timeout=1.0)
                            continue
                        self.active_resource = "idle"
                        self.active_asset_id = None
                        if batch and batch["state"] not in DONE and not batch["paused"]:
                            if batch["state"] != "awaiting_agent":
                                batch["state"] = "awaiting_agent"
                                self._save(batch)
                        self._condition.wait(timeout=1.0)
                        continue
                self._perform(*task)
        finally:
            with self._condition:
                self._release_resources(self._active())

    def _perform(self, batch, asset, stage, record, directory, evidence):
        try:
            with self._execution:
                if self._cancel.is_set():
                    raise GenerationCancelled("Batch cancelled")
                if evidence:
                    result = self.processor.evidence(
                        stage, Path(record["output"]), directory, self._cancel
                    )
                else:
                    if stage == "shape_repair":
                        self._prepare_repair_inputs(asset, record, directory)
                    result = self.processor.run(stage, asset, directory, self._cancel)
            with self._condition:
                if self._cancel.is_set() or batch["state"] == "cancelled":
                    raise GenerationCancelled("Batch cancelled")
                prior_gate = record.get("gate")
                record.update(result)
                gate = result.get("gate", record.get("gate")) or {
                    "passed": result.get("passed", False)
                }
                if evidence and prior_gate:
                    gate = {
                        "passed": bool(prior_gate.get("passed") and gate.get("passed")),
                        "compute": prior_gate,
                        "preview": gate,
                    }
                record["gate"] = gate
                if not gate.get("passed"):
                    failure_kind = (
                        self._repair_failure_kind(record, directory)
                        if stage == "shape_repair" else None
                    )
                    partial_profile_evidence = (
                        not evidence and stage == "remesh" and any(
                            profile.get("gate", {}).get("passed")
                            for profile in result.get("profiles", {}).values()
                        )
                    )
                    repair_evidence = (
                        not evidence and stage == "shape_repair"
                        and Path(record.get("output") or "").is_file()
                    )
                    record.update(
                        state="needs_evidence" if partial_profile_evidence or repair_evidence else "needs_repair",
                        error={
                            "code": "REPAIR_OPERATION_FAILED" if failure_kind else "QUALITY_GATES_FAILED",
                            "message": "Inspect this attempt's gate report and repair the cause",
                            **({"failure_kind": failure_kind} if failure_kind else {}),
                        },
                    )
                else:
                    record["state"] = (
                        "awaiting_review" if evidence or stage == "finish" else "needs_evidence"
                    )
        except Exception as exc:
            with self._condition:
                record.update(
                    state="needs_repair",
                    approved=False,
                    error={"code": getattr(exc, "code", "STAGE_FAILED"), "message": str(exc)},
                )
        finally:
            with self._condition:
                record["finished_at"] = utc_now()
                self._write_attempt_gate(record, directory)
                self.active_asset_id = None
                self._save(batch)
                self._condition.notify_all()

    def pause_batch(self, batch_id: str) -> dict[str, Any]:
        with self._condition:
            batch = self._batch(batch_id)
            if batch["state"] not in DONE:
                batch.update(paused=True, state="paused")
                self._save(batch)
                self._condition.notify_all()
        return self.get_batch_status(batch_id)

    def resume_batch(self, batch_id: str) -> dict[str, Any]:
        with self._condition:
            batch = self._batch(batch_id)
            if batch["state"] not in DONE:
                batch.update(paused=False, state="queued")
                self._save(batch)
                self._condition.notify_all()
        return self.get_batch_status(batch_id)

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        with self._condition:
            batch = self._batch(batch_id)
            if self._active() is batch:
                self._cancel.set()
            batch.update(state="cancelled", paused=True, finished_at=utc_now())
            self._save(batch)
            self._condition.notify_all()
        return self.get_batch_status(batch_id)

    def resolve_artifact(self, batch_id: str, asset_id: str, relative_path: str) -> Path:
        with self._condition:
            self._asset(self._batch(batch_id), asset_id)
        root = (self.root / batch_id / asset_id).resolve()
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Artifact path is outside the asset or does not exist")
        return path

    @staticmethod
    def _accepted_shape_path(asset: dict[str, Any], *, verify_hash: bool) -> Path:
        # Historical name retained for clients; raw inspection is not approval.
        record = asset["stages"]["shape"]
        path = Path(record["output"]).resolve()
        artifact = next((item for item in record.get("artifacts", [])
                         if Path(item["path"]).resolve() == path), None)
        if (
            not path.is_file() or path.suffix.lower() != ".glb" or not artifact
            or len(artifact.get("sha256", "")) != 64
            or path.stat().st_size != artifact.get("bytes")
        ):
            raise ValueError("Original shape is missing its immutable evidence")
        if verify_hash:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != artifact["sha256"]:
                raise ValueError("Original shape changed after generation")
        return path

    def resolve_model(self, batch_id: str, asset_id: str, model_name: str) -> Path:
        # This single read-only alias authorizes no parent directories or other
        # imported artifacts, even when the accepted checkpoint lives elsewhere.
        if model_name != "original-shape.glb":
            raise ValueError("Unknown model alias")
        with self._condition:
            asset = self._asset(self._batch(batch_id), asset_id)
            return self._accepted_shape_path(asset, verify_hash=True)

    def _model_metadata(self, path: Path) -> dict[str, Any] | None:
        from .stage_processor import _glb_document

        try:
            stat = path.stat()
            fingerprint = (stat.st_mtime_ns, stat.st_size)
            cached = self._model_metadata_cache.get(str(path))
            if cached and cached[0] == fingerprint:
                return cached[1]
            document = _glb_document(path)
            accessors = document.get("accessors", [])
            materials = document.get("materials", [])
            textures = document.get("textures", [])
            images = document.get("images", [])
            triangle_count, positions, textured = 0, set(), False
            for mesh in document.get("meshes", []):
                for primitive in mesh.get("primitives", []):
                    if primitive.get("mode", 4) != 4:
                        raise ValueError("Viewer models must contain triangles")
                    position = primitive.get("attributes", {}).get("POSITION")
                    index = primitive.get("indices", position)
                    if type(index) is not int or not 0 <= index < len(accessors):
                        raise ValueError("Invalid geometry accessor")
                    count = accessors[index]["count"]
                    if type(count) is not int or count <= 0 or count % 3:
                        raise ValueError("Invalid triangle count")
                    triangle_count += count // 3
                    if type(position) is int and 0 <= position < len(accessors):
                        positions.add(position)
                    material_index = primitive.get("material")
                    if type(material_index) is not int or not 0 <= material_index < len(materials):
                        continue
                    info = materials[material_index].get("pbrMetallicRoughness", {})
                    texture_index = info.get("baseColorTexture", {}).get("index")
                    if type(texture_index) is not int or not 0 <= texture_index < len(textures):
                        continue
                    source = textures[texture_index].get("source")
                    if type(source) is int and 0 <= source < len(images):
                        image = images[source]
                        view = image.get("bufferView")
                        embedded = type(view) is int and 0 <= view < len(
                            document.get("bufferViews", [])
                        )
                        textured |= embedded or str(image.get("uri", "")).startswith("data:image/")
            if triangle_count <= 0:
                raise ValueError("Empty model")
            metadata = {
                "triangle_count": triangle_count,
                "vertex_count": sum(accessors[index]["count"] for index in positions) or None,
                "textured": bool(textured),
            }
            if len(self._model_metadata_cache) >= 2048:
                self._model_metadata_cache.clear()
            self._model_metadata_cache[str(path)] = (fingerprint, metadata)
            return metadata
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return None

    def _viewer_models(self, batch, asset, profiles) -> list[dict[str, Any]]:
        root = (self.root / batch["batch_id"] / asset["asset_id"]).resolve()
        prefix = f"/artifacts/{batch['batch_id']}/{asset['asset_id']}/"
        models = []
        stages = asset["stages"]

        def descriptor(name, label, path, stage, profile=None, *, record=None, current=None):
            if not path:
                return None
            candidate = Path(path).resolve()
            if candidate.is_relative_to(root):
                url = prefix + candidate.relative_to(root).as_posix()
            elif name == "shape":
                try:
                    if candidate != self._accepted_shape_path(asset, verify_hash=False):
                        return None
                except (OSError, ValueError, KeyError):
                    return None
                url = f"/models/{batch['batch_id']}/{asset['asset_id']}/original-shape.glb"
            else:
                return None
            metadata = self._model_metadata(candidate)
            if not metadata:
                return None
            quad_geometry = (profile or {}).get("geometry", {})
            record = record if record is not None else stages[stage]
            if current is None:
                current = not record.get("stale") and record.get("state") not in {"pending", "queued"}
            return {
                "id": name, "label": label, "url": url, "profile": name,
                "source_id": "shape" if stage in {"shape", "shape_repair"} else "accepted_master",
                "stage": stage, **metadata,
                "texture_state": "textured" if metadata["textured"] else "untextured",
                "quad_count": quad_geometry.get("cleaned_quads"),
                "native_quad_count": quad_geometry.get("native_quads"),
                "quad_count_source": (
                    "cleaned AutoRemesher OBJ before GLB triangulation" if profile else None
                ),
                "target_quads": (profile or {}).get("target_quads"),
                "approved": bool(record.get("approved")) and bool(current),
                "current": bool(current), "stale": not current,
                "review_state": record.get("state"), "attempt": record.get("attempt"),
            }

        shape = descriptor("shape", "Original Hunyuan", stages["shape"].get("output"), "shape")
        if shape:
            models.append(shape)
        repair = stages["shape_repair"]
        for candidate_record in [repair, *reversed(repair.get("history", []))]:
            attempt = candidate_record.get("attempt", 0)
            if not attempt:
                continue
            value = descriptor(
                f"repair_attempt_{attempt}", f"Repair candidate {attempt}",
                candidate_record.get("output"), "shape_repair", record=candidate_record,
                current=(candidate_record is repair and not candidate_record.get("stale")
                         and candidate_record.get("state") not in {"pending", "queued"}),
            )
            if value:
                models.append(value)
        master = asset.get("accepted_master")
        if master:
            value = descriptor("accepted_master", "Accepted repair master", master.get("path"),
                               "shape_repair", record={"approved": True, "state": "approved",
                                                        "attempt": master.get("attempt")})
            if value:
                models.append(value)
        finish = stages["finish"]
        finish_directory = (
            Path(finish["output"]).parent if finish.get("output") else
            self._attempt_directory(batch, asset, "finish", finish["attempt"])
        )
        modern = bool(profiles) or (finish_directory / "full_game.glb").is_file()
        if modern:
            for name, label in (("full_game", "Full game"), ("mobile", "Mobile"),
                                ("browser", "Browser")):
                profile = profiles.get(name, {})
                candidates = [(finish_directory / f"{name}.glb", "finish")]
                if name == "full_game":
                    candidates.append((stages["paint"].get("output"), "paint"))
                candidates.append((profile.get("output"), "remesh"))
                available = [value for path, stage in candidates
                             if (value := descriptor(name, label, path, stage, profile))]
                current = [value for value in available if value["current"]]
                if current:
                    models.append(next((value for value in current if value["textured"]), current[0]))
                for value in available:
                    if value["stale"]:
                        value.update(id=f"stale_{name}_{value['stage']}",
                                     label=f"{label} — previous {value['stage']} (stale)")
                        models.append(value)
        else:
            for name in ("master", "game", "lod1", "lod2"):
                value = descriptor(name, name.upper(), finish_directory / f"{name}.glb", "finish")
                if value:
                    models.append(value)
        return models

    def queue_snapshot(self) -> dict[str, Any]:
        with self._condition:
            result = []
            for batch in self._batches.values():
                summary = {
                    k: batch[k]
                    for k in ("batch_id", "name", "state", "stage", "created_at", "updated_at")
                }
                summary["assets"] = []
                counts = {s: 0 for s in STAGES}
                for asset in batch["assets"]:
                    for stage in STAGES:
                        counts[stage] += int(self._shape_generated(asset) if stage == "shape"
                                             else asset["stages"][stage]["approved"])
                    record = asset["stages"][batch["stage"]]
                    root = self.root / batch["batch_id"] / asset["asset_id"]
                    previews = {}
                    for stage in reversed(STAGES):
                        if asset["stages"][stage].get("stale"):
                            continue
                        for view, path in asset["stages"][stage].get("previews", {}).items():
                            candidate = Path(path)
                            if candidate.is_relative_to(root):
                                previews.setdefault(
                                    view,
                                    f"/artifacts/{batch['batch_id']}/{asset['asset_id']}/{candidate.relative_to(root).as_posix()}",
                                )
                    artifacts = []
                    for stage in STAGES:
                        for item in asset["stages"][stage].get("artifacts", []):
                            path = Path(item["path"])
                            if path.suffix.lower() in {
                                ".glb",
                                ".obj",
                                ".blend",
                                ".json",
                            } and path.is_relative_to(root):
                                artifacts.append(
                                    {
                                        "name": f"{stage} / {path.name}",
                                        "relative_path": path.relative_to(root).as_posix(),
                                    }
                                )
                    remesh = asset["stages"]["remesh"]
                    profile_manifest = (
                        Path(remesh["profiles_manifest"]) if remesh.get("profiles_manifest") else
                        self._attempt_directory(batch, asset, "remesh", remesh["attempt"])
                        / "profiles.json"
                    )
                    current_profiles = remesh.get("profiles", {})
                    if remesh["state"] == "running":
                        try:
                            if (
                                profile_manifest.is_file()
                                and profile_manifest.stat().st_size <= 8 * 1024**2
                            ):
                                live = json.loads(profile_manifest.read_text(encoding="utf-8"))
                                current_profiles = live.get("profiles", current_profiles)
                        except (OSError, ValueError):
                            pass
                    profiles = {}
                    for name in PROFILE_TARGETS:
                        profile = current_profiles.get(name, {})
                        geometry = profile.get("geometry", {})
                        output = Path(profile["output"]) if profile.get("output") else None
                        profiles[name] = {
                            "target_quads": asset["params"][f"{name}_target_quads"],
                            "triangle_budget": asset["params"][f"{name}_triangle_budget"],
                            "native_quads": geometry.get("native_quads"),
                            "actual_quads": geometry.get("cleaned_quads"),
                            "actual_triangles": profile.get("exported_triangles"),
                            "geometry": geometry,
                            "state": profile.get("state", "pending"),
                            "stale": bool(remesh.get("stale")),
                            "gate": {"passed": profile.get("gate", {}).get("passed")},
                            "output": (
                                f"/artifacts/{batch['batch_id']}/{asset['asset_id']}/"
                                f"{output.relative_to(root).as_posix()}"
                                if output and output.is_relative_to(root) else None
                            ),
                        }
                        for view, path in profile.get("previews", {}).items():
                            candidate = Path(path)
                            if candidate.is_relative_to(root):
                                previews.setdefault(
                                    f"{name}_{view}",
                                    f"/artifacts/{batch['batch_id']}/{asset['asset_id']}/"
                                    f"{candidate.relative_to(root).as_posix()}",
                                )
                    summary["assets"].append(
                        {
                            "asset_id": asset["asset_id"],
                            "name": asset["name"],
                            "state": "completed"
                            if asset["stages"]["finish"]["approved"]
                            else record["state"],
                            "stage": batch["stage"],
                            "attempt": record["attempt"],
                            "error": record.get("error"),
                            "previews": previews,
                            "profiles": profiles,
                            "models": self._viewer_models(batch, asset, current_profiles),
                            "accepted_master": copy.deepcopy(asset.get("accepted_master")),
                            "shape_repair": {
                                key: copy.deepcopy(asset["stages"]["shape_repair"].get(key))
                                for key in ("state", "attempt", "approved", "stale", "gate",
                                            "repair_report", "repair_request", "review", "previews")
                            },
                            "profiles_manifest": (
                                f"/artifacts/{batch['batch_id']}/{asset['asset_id']}/"
                                f"{profile_manifest.relative_to(root).as_posix()}"
                                if profile_manifest and profile_manifest.is_relative_to(root)
                                else None
                            ),
                            "artifacts": artifacts,
                            "evidence": {
                                "gate": record.get("gate"),
                                "review": record.get("review"),
                                "history": [
                                    {k: h.get(k) for k in ("attempt", "state", "error", "review")}
                                    for h in record.get("history", [])[-20:]
                                ],
                            },
                        }
                    )
                summary["counts"] = counts
                result.append(summary)
            active = self._active()
            return {
                "batches": result,
                "active_batch_id": active["batch_id"] if active else None,
                "active_resource": self.active_resource,
                "active_asset_id": self.active_asset_id,
            }

    def close(self) -> None:
        with self._condition:
            self._shutdown.set()
            self._cancel.set()
            self._condition.notify_all()
        self._thread.join(timeout=15)
        if self._thread.is_alive():
            stop = getattr(self.processor, "stop", None)
            if stop is not None:
                stop()
            self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise Codex3DError("Batch worker has not stopped.", code="WORKER_BUSY")
        # The worker owns the lease until all CPU/GPU work has finished.
        # Retrying resource release here also exposes a failed final unload.
        self.processor.close()
        if self.dashboard:
            self.dashboard.close()
