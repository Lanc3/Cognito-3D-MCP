"""Execute one batch asset at a time, with exclusive resource ownership."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

from ..errors import Codex3DError, GenerationCancelled
from ..trellis.artifacts import atomic_json
from .remesh import (
    PROFILE_TARGETS,
    SOURCE_COMMIT,
    AutoRemesherRuntime,
    _check_memory,
    _memory_status,
    profile_parameters,
    run_cpu_process,
)
from .runtime import ResidentStageRuntime


class BatchStageProcessor:
    def __init__(self, settings, qa_manager):
        self.settings = settings
        self.runtime = ResidentStageRuntime(replace(settings, low_vram_mode=False))
        self.remesher = AutoRemesherRuntime(settings)
        self.qa_manager = qa_manager

    def end_stage(self):
        self.runtime.end_stage()

    def preflight(self):
        runtime = self.runtime.preflight()
        remesh = self.remesher.preflight()
        memory = _memory_status()
        checkpoint = (
            self.runtime._shape_snapshot() / self.settings.shape_subfolder / "model.fp16.ckpt"
        )
        loader_bytes = max(4096 * 1024**2, checkpoint.stat().st_size if checkpoint.is_file() else 0)
        required = loader_bytes + 2048 * 1024**2
        memory["required_before_shape_bytes"] = required
        memory["ready"] = min(memory["available_physical"], memory["available_commit"]) >= required
        return {
            "ready": runtime["ready_for_generation"] and remesh["ready"] and memory["ready"],
            "runtime": runtime,
            "remesher": remesh,
            "host_memory": memory,
        }

    def close(self):
        self.runtime.stop()
        self.runtime.end_stage()

    def stop(self):
        """Stop the owned model without releasing the batch's resource lease."""
        self.runtime.stop()

    def run(self, stage, asset, directory, cancel):
        params = asset["params"]
        refs = asset["stages"]["references"].get("prepared_views", {})
        if stage == "shape":
            if self.runtime.active_stage != "shape":
                checkpoint = (
                    self.runtime._shape_snapshot()
                    / self.settings.shape_subfolder
                    / "model.fp16.ckpt"
                )
                # Reserve host memory for the checkpoint loader before it starts.
                loader_mb = (
                    max(4096, int(checkpoint.stat().st_size / 1024**2))
                    if checkpoint.is_file()
                    else 4096
                )
                _check_memory(
                    {"memory_limit_mb": loader_mb, "min_available_mb": 2048}, starting=True
                )
            status = self.runtime.preflight()
            if not status["ready_for_generation"]:
                raise Codex3DError(
                    "Hunyuan preflight failed; inspect server_status before retry",
                    code="MODEL_UNAVAILABLE",
                )
            output = directory / "shape.glb"
            metrics = self.runtime.generate_shape(
                {name: Path(path) for name, path in refs.items()},
                output,
                params,
                directory.parents[2] / "logs" / "shape-phase.log",
                cancel,
            )
            gate = {"passed": metrics.get("faces", 0) > 0, "geometry": metrics}
        elif stage == "remesh":
            self.runtime.begin_stage("remesh", directory / "remesh.log", cancel)
            result = self._remesh_profiles(asset, directory, cancel)
            atomic_json(directory / "machine-gate.json", result["gate"])
            return result
        elif stage == "shape_repair":
            return self._repair_shape(asset, directory, cancel)
        elif stage == "paint":
            manifest = self._accepted_profiles(asset)
            if self.runtime.active_stage != "texture":
                _check_memory({"memory_limit_mb": 4096, "min_available_mb": 2048}, starting=True)
            output = directory / "textured.glb"
            # Per-attempt atlas resolution controls the renderer, not just metadata.
            self.runtime.settings = replace(
                self.settings,
                low_vram_mode=False,
                texture_resolution=int(params["texture_resolution"]),
            )
            metrics = self.runtime.generate_texture(
                Path(manifest["profiles"]["full_game"]["output"]),
                Path(refs["front"]),
                output,
                directory.parents[2] / "logs" / "paint-phase.log",
                cancel,
            )
            triangles = _glb_triangle_count(output)
            triangle_budget = manifest["profiles"]["full_game"]["triangle_budget"]
            gate = {
                "passed": metrics.get("faces", 0) > 0 and _has_color_texture(output)
                and 0 < triangles <= triangle_budget,
                "geometry": metrics,
                "has_base_color_texture": _has_color_texture(output),
                "exported_triangles": triangles,
                "triangle_budget": triangle_budget,
            }
            atomic_json(directory / "machine-gate.json", gate)
            return {"output": str(output), "gate": gate,
                    "source_profile_sha256": manifest["profiles"]["full_game"]["output_sha256"]}
        elif stage == "finish":
            manifest = self._accepted_profiles(asset)
            if asset["stages"]["paint"].get("source_profile_sha256") != manifest["profiles"]["full_game"]["output_sha256"]:
                raise Codex3DError("Paint belongs to a different full-game mesh; retry paint", code="PAINT_STALE")
            source = Path(asset["stages"]["paint"]["output"])
            paint_record = asset["stages"]["paint"]
            paint_artifact = next((item for item in paint_record.get("artifacts", [])
                                   if Path(item.get("path", "")).resolve() == source.resolve()), {})
            if not paint_record.get("approved") or not paint_artifact.get("sha256") or _sha256(source) != paint_artifact["sha256"]:
                raise Codex3DError("Approved Paint output changed; retry paint", code="PAINT_CHANGED")
            metrics = self._blender(
                source, directory, cancel, preview_only=False, limits=params,
                profiles=manifest["profiles"],
                profiles_manifest=asset["stages"]["remesh"]["profiles_manifest"],
            )
            gate = self.qa_manager._qa(directory, metrics)
            exported = self._validate_final_exports(asset, directory, cancel)
            gate["export_validation"] = exported
            for name, validation in exported.items():
                gate["checks"].append({"name": f"{name}_serialized_geometry",
                                       "passed": bool(validation["gate"].get("passed")),
                                       "report": validation["repair_report"]})
                gate["checks"].append({"name": f"{name}_serialized_triangle_budget",
                                       **validation["serialized_triangle_budget"]})
            master_alias = directory / "master.glb"
            full_game_export = directory / "full_game.glb"
            gate["checks"].append({
                "name": "master_matches_validated_full_game",
                "passed": master_alias.is_file() and full_game_export.is_file()
                and _sha256(master_alias) == _sha256(full_game_export),
            })
            for name in ("master", *PROFILE_TARGETS):
                present = _has_color_texture(directory / f"{name}.glb")
                gate["checks"].append({"name": f"{name}_has_color_texture", "passed": present})
            gate["passed"] = all(check["passed"] for check in gate["checks"])
            atomic_json(directory / "qa" / "report.json", gate)
            output = directory / "master.glb"
            atomic_json(
                directory / "manifest.json",
                {
                    "asset_id": asset["asset_id"],
                    "name": asset["name"],
                    "params": params,
                    "stages": asset["stages"],
                    "qa": gate,
                    "profiles": manifest["profiles"],
                    "note": (
                        "Profiles independently remeshed from one accepted Hunyuan shape. "
                        "Native and cleaned quad OBJs retained; GLB exports use triangles."
                    ),
                },
            )
            previews = {
                f"{name}_{view}": path
                for name in PROFILE_TARGETS
                for view, path in self._previews(
                    directory / "previews" / name, nested=False
                ).items()
            }
            return {"output": str(output), "gate": gate, "previews": previews}
        else:
            raise ValueError(f"Unknown batch stage: {stage}")
        atomic_json(directory / "machine-gate.json", gate)
        return {"output": str(output), "gate": gate}

    def evidence(self, stage, output, directory, cancel):
        # The scheduler has closed the model worker before this method starts.
        if stage == "shape_repair":
            repair = json.loads((directory / "repair-result.json").read_text(encoding="utf-8"))
            source_hash = _sha256(Path(output))
            metrics = self._blender(
                output, directory, cancel, preview_only=True, exact_candidate=True
            )
            previews = self._previews(directory, include_underside=True)
            framing = metrics.get("framing", {})
            gate = {
                "passed": bool(repair["gate"].get("passed"))
                and source_hash == _sha256(Path(output)) and len(previews) == 7
                and set(framing) == set(previews) and all(item.get("passed") for item in framing.values()),
                "repair": repair["gate"],
                "candidate_sha256": source_hash,
                "semantic_review_required": True,
                "preview_geometry_is_diagnostic_only": True,
                "geometry": metrics.get("geometry", {}),
                "framing": framing,
            }
            return {"gate": gate, "previews": previews, "repair_report": repair["repair_report"]}
        if stage == "remesh":
            manifest = _read_profile_manifest(directory / "profiles.json", require_complete=False)
            previews, gates = {}, {}
            for name in PROFILE_TARGETS:
                profile = manifest["profiles"].get(name)
                if not profile or not profile.get("gate", {}).get("passed"):
                    gates[name] = {
                        "passed": False, "reason": "Profile computation failed or is pending"
                    }
                    continue
                profile_dir = directory / "profiles" / name
                metrics = self._blender(
                    Path(profile["output"]), profile_dir, cancel, preview_only=True
                )
                geometry = metrics["geometry"]
                views = self._previews(profile_dir)
                gates[name] = {
                    "passed": bool(geometry.get("watertight"))
                    and geometry.get("non_manifold_edges") == 0
                    and 0 < geometry.get("faces", 0) <= profile["triangle_budget"]
                    and geometry.get("degenerate_faces", 0)
                    / max(1, geometry.get("faces", 0)) < 0.001
                    and len(views) == 4,
                    "geometry": geometry,
                    "semantic_review_required": True,
                }
                profile["previews"] = views
                profile["preview_gate"] = gates[name]
                previews.update({f"{name}_{view}": path for view, path in views.items()})
                atomic_json(directory / "profiles.json", manifest)
            atomic_json(directory / "profiles.json", manifest)
            return {
                "gate": {
                    "passed": all(gate["passed"] for gate in gates.values()), "profiles": gates
                },
                "profiles": manifest["profiles"],
                "previews": previews,
            }
        metrics = self._blender(output, directory, cancel, preview_only=True)
        geometry = metrics["geometry"]
        gate = {
            "passed": geometry.get("faces", 0) > 0
            and geometry.get("degenerate_faces", 0) / max(1, geometry.get("faces", 0)) < 0.001,
            "geometry": geometry,
            "semantic_review_required": True,
        }
        if stage == "remesh":
            gate["passed"] = gate["passed"] and bool(geometry.get("watertight"))
        if stage == "paint":
            gate["passed"] = gate["passed"] and _has_color_texture(output)
        previews = self._previews(directory)
        gate["passed"] = gate["passed"] and len(previews) == 4
        if stage == "paint":
            previews = {f"full_game_{name}": path for name, path in previews.items()}
        return {"gate": gate, "previews": previews}

    def _previews(self, directory, *, nested=True, include_underside=False):
        directory = directory / "previews" if nested else directory
        views = ("front", "back", "left", "right")
        if include_underside:
            views += ("bottom", "underside_front_left", "underside_back_right")
        return {
            name: str(directory / f"{name}.png")
            for name in views
            if (directory / f"{name}.png").is_file()
        }

    def _repair_shape(self, asset, directory, cancel):
        from .repair_contract import validate_recipe

        request = dict(asset.get("shape_repair_request") or {})
        source = Path(request.get("source_path") or asset["stages"]["shape"]["output"]).resolve()
        expected_hash = request.get("source_sha256") or _sha256(source)
        if _sha256(source) != expected_hash:
            raise Codex3DError("Repair source changed", code="REPAIR_SOURCE_CHANGED")
        recipe = validate_recipe(request.get("recipe") or {"method": "analyze"})
        self.runtime.begin_stage("shape_repair", directory / "repair.log", cancel)
        request.update(
            source_path=str(source), source_sha256=expected_hash, recipe=recipe,
            output_dir=str(directory.resolve()), result_path=str(directory / "repair-result.json"),
        )
        request = {key: value for key, value in request.items() if key in {
            "schema_version", "source_path", "source_sha256", "recipe", "output_dir", "result_path",
            "import_path", "import_sha256",
        }}
        request_path = directory / "repair-worker-input.json"
        atomic_json(request_path, request)
        # Explicit paths select the tested checkout and optional native packages.
        # Nothing here is interpreted by a shell or supplied as executable user code.
        paths = [str(Path(__file__).resolve().parents[2])]
        packages = getattr(self.settings, "repair_package_dir", None)
        if packages and Path(packages).is_dir():
            paths.append(str(Path(packages).resolve()))
        bootstrap = (
            "import sys,runpy;sys.path[:0]=" + repr(paths) + ";"
            "runpy.run_module('codex_3d_mcp.hunyuan_mv.repair_worker',run_name='__main__')"
        )
        limits = asset["params"]
        proof = run_cpu_process(
            [str(self.settings.python_exe), "-c", bootstrap, "--request", str(request_path)],
            directory / "repair.log", cancel,
            timeout_seconds=getattr(self.settings, "repair_timeout_seconds", 300),
            threads=min(2, int(limits.get("threads", 2))),
            memory_limit_mb=min(2048, int(limits.get("memory_limit_mb", 2048))),
            min_available_mb=max(2048, int(limits.get("min_available_mb", 2048))),
        )
        atomic_json(directory / "repair-resource-controls.json", proof)
        result_path = directory / "repair-result.json"
        if not result_path.is_file():
            raise Codex3DError("Repair worker omitted its result", code="REPAIR_RESULT_MISSING")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        output = Path(result["output"]).resolve() if result.get("output") else None
        if output is not None and (not output.is_relative_to(directory.resolve()) or not output.is_file()):
            raise Codex3DError("Repair output is outside its attempt", code="REPAIR_OUTPUT_PATH")
        if _sha256(source) != expected_hash:
            raise Codex3DError("Repair modified its immutable source", code="REPAIR_SOURCE_CHANGED")
        gate = result.get("gate", {})
        if gate.get("source_sha256") != expected_hash or gate.get("candidate_sha256") != (_sha256(output) if output else None):
            raise Codex3DError("Repair gate is not bound to these meshes", code="REPAIR_PROOF_CHANGED")
        atomic_json(directory / "machine-gate.json", result["gate"])
        return result

    @staticmethod
    def _accepted_master(asset):
        from .repair_contract import REPAIR_GATE_VERSION

        accepted = asset.get("accepted_master") or {}
        repair = asset["stages"].get("shape_repair") or {}
        if (not accepted.get("path") or not accepted.get("sha256")
                or accepted.get("gate_version") != REPAIR_GATE_VERSION
                or not repair.get("approved") or not (repair.get("gate") or {}).get("passed")
                or repair.get("stale")
                or accepted.get("attempt") != repair.get("attempt")):
            raise Codex3DError("Shape repair and review are required", code="MASTER_REVIEW_REQUIRED")
        source = Path(accepted["path"]).resolve()
        if not source.is_file() or _sha256(source) != accepted["sha256"]:
            raise Codex3DError("Accepted master changed", code="MASTER_CHANGED")
        certificate = repair["gate"].get("compute", repair["gate"])
        request = repair.get("repair_request") or {}
        if (not repair.get("output") or Path(repair["output"]).resolve() != source
                or not certificate.get("passed")
                or certificate.get("gate_version") != REPAIR_GATE_VERSION
                or certificate.get("output_sha256") != accepted["sha256"]
                or certificate.get("source_sha256") != accepted.get("source_sha256")
                or not request.get("request_id")
                or request["request_id"] != accepted.get("request_id")
                or request.get("source_sha256") != accepted.get("source_sha256")
                or request.get("source_shape_sha256") != accepted.get("source_shape_sha256")
                or request.get("source_shape_attempt") != accepted.get("source_shape_attempt")):
            raise Codex3DError("Accepted master does not match its reviewed repair certificate",
                               code="MASTER_CERTIFICATE_CHANGED")
        raw = Path(asset["stages"]["shape"]["output"])
        if (accepted.get("source_shape_sha256") != _sha256(raw)
                or accepted.get("source_shape_attempt") != asset["stages"]["shape"].get("attempt")):
            raise Codex3DError("Master belongs to an earlier shape", code="MASTER_STALE")
        return source

    def _validate_final_exports(self, asset, directory, cancel):
        """Check actual exported GLBs, independent of Blender's preview metrics."""
        results = {}
        plan = profile_parameters(asset["params"])
        for name in PROFILE_TARGETS:
            source = directory / f"{name}.glb"
            destination = directory / "qa" / f"export-{name}"
            destination.mkdir(parents=True, exist_ok=False)
            validation_asset = {
                "params": asset["params"],
                "stages": {"shape": {"output": str(source)}},
                "shape_repair_request": {"source_path": str(source), "source_sha256": _sha256(source),
                                         "recipe": {"method": "analyze"}},
            }
            result = self._repair_shape(validation_asset, destination, cancel)
            source_hash = _sha256(source)
            diagnostics_path = Path(result.get("diagnostics") or "").resolve()
            diagnostics = {}
            if diagnostics_path.is_relative_to(destination.resolve()) and diagnostics_path.is_file():
                diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            measured = (diagnostics.get("candidate") or {}).get("topology", {}).get("triangles")
            try:
                accessor_count = _glb_triangle_count(source)
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                accessor_count = None
            maximum = plan[f"{name}_triangle_budget"]
            result["serialized_triangle_budget"] = {
                "passed": type(measured) is int and 0 < measured <= maximum
                and measured == accessor_count
                and diagnostics.get("source_sha256") == source_hash
                and diagnostics.get("output_sha256") == source_hash,
                "triangles": measured, "accessor_triangles": accessor_count,
                "maximum": maximum, "diagnostics": str(diagnostics_path),
            }
            results[name] = result
        return results

    def _remesh_profiles(self, asset, directory, cancel):
        source = self._accepted_master(asset)
        source_hash = _sha256(source)
        plan = profile_parameters(asset["params"])
        manifest_path = directory / "profiles.json"
        manifest = {
            "schema_version": 2,
            "source_shape": {"path": str(source), "sha256": source_hash},
            "profiles": {},
            "passed": False,
        }
        atomic_json(manifest_path, manifest)
        for name in PROFILE_TARGETS:
            if cancel.is_set():
                raise GenerationCancelled("Profile remeshing cancelled at a profile boundary")
            output = directory / "profiles" / name / "remeshed.glb"
            output.parent.mkdir(parents=True, exist_ok=True)
            params = {
                **asset["params"],
                "target_quads": plan[f"{name}_target_quads"],
                "triangle_budget": plan[f"{name}_triangle_budget"],
            }
            fingerprint = self._profile_fingerprint(source_hash, params)
            reused = self._reusable_profile(asset, name, fingerprint)
            if reused:
                manifest["profiles"][name] = {
                    **reused, "reused_from": reused["output"], "state": "passed",
                }
                atomic_json(manifest_path, manifest)
                continue
            profile = {
                "output": str(output.resolve()),
                "target_quads": params["target_quads"],
                "triangle_budget": params["triangle_budget"],
                "source_shape_sha256": source_hash,
                "geometry_fingerprint": fingerprint,
                "state": "running",
            }
            manifest["profiles"][name] = profile
            atomic_json(manifest_path, manifest)
            try:
                if _sha256(source) != source_hash:
                    raise Codex3DError(
                        "Accepted shape changed during remeshing", code="SHAPE_CHANGED"
                    )
                result = self.remesher.remesh(
                    source, output, params, output.parent / "remesh.log", cancel
                )
                candidate = Path(result.get("artifacts", {}).get("candidate_glb", output))
                triangle_source = output if output.is_file() else candidate
                triangles = _glb_triangle_count(triangle_source) if triangle_source.is_file() else 0
                budget_gate = {
                    "name": "exported_triangle_budget",
                    "passed": 0 < triangles <= params["triangle_budget"],
                    "triangles": triangles,
                    "maximum": params["triangle_budget"],
                }
                result.setdefault("checks", []).append(budget_gate)
                result["passed"] = bool(result.get("passed") and budget_gate["passed"])
                profile.update(
                    gate=result, geometry=result.get("geometry", {}),
                    artifacts=result.get("artifacts", {}), exported_triangles=triangles,
                    state="passed" if result["passed"] else "needs_repair",
                    output_sha256=_sha256(output) if output.is_file() else None,
                )
                atomic_json(output.parent / "machine-gate.json", result)
            except GenerationCancelled:
                raise
            except Exception as exc:
                profile.update(
                    state="needs_repair", gate={"passed": False},
                    error={
                        "code": getattr(exc, "code", "PROFILE_REMESH_FAILED"), "message": str(exc)
                    },
                )
                atomic_json(manifest_path, manifest)
                break
            atomic_json(manifest_path, manifest)
            if not result["passed"]:
                break
        manifest["passed"] = set(manifest["profiles"]) == set(PROFILE_TARGETS) and all(
            profile.get("gate", {}).get("passed") for profile in manifest["profiles"].values()
        )
        atomic_json(manifest_path, manifest)
        return {
            "output": manifest["profiles"].get("full_game", {}).get(
                "output", str((directory / "profiles" / "full_game" / "remeshed.glb").resolve())
            ),
            "profiles_manifest": str(manifest_path.resolve()),
            "profiles": manifest["profiles"],
            "gate": {"passed": manifest["passed"], "profiles": manifest["profiles"]},
        }

    @staticmethod
    def _profile_fingerprint(source_hash, params):
        defaults = {
            "edge_scaling": 1.0, "sharp_edge": 90.0, "smooth_normal": 0.0,
            "adaptivity": 1.0, "anisotropy": 1.0,
        }
        contract = {
            "source": source_hash, "remesher": SOURCE_COMMIT,
            "target_quads": params["target_quads"],
            "triangle_budget": params["triangle_budget"],
            "geometry": {key: float(params.get(key, value)) for key, value in defaults.items()},
            "prepare_validate": _sha256(Path(__file__).with_name("remesh_worker.py")),
            "triangulation": _sha256(Path(__file__).with_name("remesh_triangulation.py")),
        }
        return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _reusable_profile(asset, name, fingerprint):
        record = asset["stages"]["remesh"]
        candidates = [record, record.get("previous_attempt") or {}, *reversed(record.get("history", []))]
        for previous in candidates:
            profile = previous.get("profiles", {}).get(name, {})
            if profile.get("geometry_fingerprint") != fingerprint or not profile.get("gate", {}).get("passed"):
                continue
            output = Path(profile.get("output", ""))
            if output.is_file() and profile.get("output_sha256") == _sha256(output):
                return dict(profile)
        return None

    def _accepted_profiles(self, asset):
        record = asset["stages"]["remesh"]
        if not all(record.get(key) for key in ("approved", "profiles_manifest", "profiles")):
            raise Codex3DError(
                "All three remesh profiles require gates and agent approval; retry remesh first",
                code="PROFILE_REVIEW_REQUIRED",
            )
        manifest = _read_profile_manifest(Path(record["profiles_manifest"]))
        source = self._accepted_master(asset)
        if Path(manifest["source_shape"]["path"]).resolve() != source:
            raise Codex3DError(
                "Profile source differs from the accepted shape", code="PROFILE_SOURCE"
            )
        plan = profile_parameters(asset["params"])
        for name, profile in manifest["profiles"].items():
            saved = record["profiles"].get(name, {})
            for field in (
                "output", "output_sha256", "source_shape_sha256", "target_quads", "triangle_budget",
                "geometry_fingerprint",
            ):
                if saved.get(field) != profile.get(field):
                    raise Codex3DError(
                        "Profile manifest differs from reviewed record", code="PROFILE_CHANGED"
                    )
            if any(
                profile[key] != plan[f"{name}_{key}"] for key in ("target_quads", "triangle_budget")
            ):
                raise Codex3DError("Profile settings changed; retry remesh", code="PROFILE_CHANGED")
            if manifest.get("schema_version") == 2 and profile.get("geometry_fingerprint") != self._profile_fingerprint(
                manifest["source_shape"]["sha256"],
                {**asset["params"], "target_quads": profile["target_quads"], "triangle_budget": profile["triangle_budget"]},
            ):
                raise Codex3DError("Profile geometry implementation changed; retry remesh", code="PROFILE_CHANGED")
        if record.get("output") != manifest["profiles"]["full_game"]["output"]:
            raise Codex3DError("Full-game profile output changed", code="PROFILE_CHANGED")
        return manifest

    def _blender(
        self, source, directory, cancel, *, preview_only, limits=None,
        profiles=None, profiles_manifest=None, exact_candidate=False,
    ):
        self.runtime.begin_stage(
            "evidence" if preview_only else "finish", directory / "blender.log", cancel
        )
        config = directory / "blender-config.json"
        result = directory / "blender-result.json"
        config_data = {
                "input_glb": str(source),
                "output_dir": str(directory),
                "prepared_master": True,
                "preview_only": preview_only,
                "exact_candidate_preview": exact_candidate,
                "game_faces": self.settings.game_faces,
                "lod_faces": list(self.settings.lod_faces),
        }
        if not preview_only:
            if not profiles or set(profiles) != set(PROFILE_TARGETS) or not profiles_manifest:
                raise Codex3DError(
                    "Final export requires all remesh profiles", code="PROFILES_REQUIRED"
                )
            config_data.update(
                prepared_profiles=True,
                profile_meshes={name: profile["output"] for name, profile in profiles.items()},
                profile_triangle_budgets={
                    name: profile["triangle_budget"] for name, profile in profiles.items()
                },
                profiles_manifest=str(profiles_manifest),
            )
        atomic_json(config, config_data)
        command = [
            str(self.settings.blender_exe),
            "--background",
            "--factory-startup",
            "--threads",
            str(self.settings.remesh_threads),
            "--python-exit-code",
            "1",
            "--python",
            str(Path(__file__).with_name("blender_worker.py")),
            "--",
            str(config),
            str(result),
        ]
        limits = limits or {
            "memory_limit_mb": 2048,
            "min_available_mb": 2048,
            "threads": self.settings.remesh_threads,
        }
        run_cpu_process(
            command,
            directory / "blender.log",
            cancel,
            timeout_seconds=900,
            threads=min(self.settings.remesh_threads, int(limits["threads"])),
            memory_limit_mb=min(2048, int(limits["memory_limit_mb"])),
            min_available_mb=max(2048, int(limits["min_available_mb"])),
        )
        if not result.is_file():
            raise Codex3DError(
                "Blender omitted evidence; inspect blender.log", code="BLENDER_FAILED"
            )
        return json.loads(result.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _glb_document(path: Path) -> dict:
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b"glTF":
            raise ValueError("Missing GLB header")
        _, version, total_length = struct.unpack("<4sII", header)
        if version != 2 or total_length != path.stat().st_size:
            raise ValueError("Invalid GLB version or length")
        chunk = stream.read(8)
        if len(chunk) != 8:
            raise ValueError("Missing GLB JSON chunk")
        length, kind = struct.unpack("<II", chunk)
        if kind != 0x4E4F534A or length > 32 * 1024**2:
            raise ValueError("Invalid or oversized GLB JSON chunk")
        content = stream.read(length)
        if len(content) != length:
            raise ValueError("Truncated GLB JSON chunk")
        return json.loads(content)


def _glb_triangle_count(path: Path) -> int:
    document = _glb_document(path)
    total = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                raise ValueError("Profile GLB must contain triangle primitives")
            index = primitive.get("indices", primitive.get("attributes", {}).get("POSITION"))
            count = document["accessors"][index]["count"]
            if type(count) is not int or count <= 0 or count % 3:
                raise ValueError("Invalid profile triangle accessor count")
            total += count // 3
    return total


def _read_profile_manifest(path: Path, *, require_complete=True) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        profiles = manifest["profiles"]
        if (
            manifest.get("schema_version") not in (1, 2) or not profiles
            or set(profiles) - set(PROFILE_TARGETS)
        ):
            raise ValueError("Invalid profile manifest")
        if require_complete and (
            manifest.get("passed") is not True or set(profiles) != set(PROFILE_TARGETS)
        ):
            raise ValueError("All three profile meshes are required")
        source = manifest["source_shape"]
        if _sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError("Accepted source shape changed")
        profile_parameters({
            f"{name}_{key}": profile[key]
            for name, profile in profiles.items()
            for key in ("target_quads", "triangle_budget")
        })
        for name, profile in profiles.items():
            if not profile.get("gate", {}).get("passed"):
                if require_complete:
                    raise ValueError(f"{name} has not passed its remesh gates")
                continue
            output = Path(profile["output"]).resolve()
            expected = (path.parent / "profiles" / name / "remeshed.glb").resolve()
            reused = False
            if manifest.get("schema_version") == 2 and profile.get("reused_from"):
                # Reuse can point only into a previous attempt of this asset's
                # remesh stage, never an arbitrary mesh or another asset.
                root = path.parent.parent.resolve()
                relative = output.relative_to(root)
                reused = (
                    len(relative.parts) == 4
                    and relative.parts[0].startswith("attempt-")
                    and relative.parts[0] < path.parent.name
                    and relative.parts[1:] == ("profiles", name, "remeshed.glb")
                    and Path(profile["reused_from"]).resolve() == output
                    and len(str(profile.get("geometry_fingerprint", ""))) == 64
                )
            if (output != expected and not reused) or profile["source_shape_sha256"] != source["sha256"]:
                raise ValueError(f"{name} does not match the shared source and attempt")
            if _sha256(output) != profile["output_sha256"]:
                raise ValueError(f"{name} mesh changed after validation")
            triangles = _glb_triangle_count(output)
            if not 0 < triangles <= profile["triangle_budget"]:
                raise ValueError(f"{name} exceeds its actual exported triangle budget")
            if triangles != profile["exported_triangles"]:
                raise ValueError(f"{name} triangle evidence does not match the exported GLB")
        return manifest
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise Codex3DError(
            f"Valid reviewed profile manifest required; retry remesh: {exc}",
            code="PROFILE_MANIFEST",
        ) from exc


def _has_color_texture(path: Path) -> bool:
    try:
        document = _glb_document(path)
        return bool(document.get("images")) and any(
            "baseColorTexture" in material.get("pbrMetallicRoughness", {})
            for material in document.get("materials", [])
        )
    except (OSError, ValueError, KeyError):
        return False
