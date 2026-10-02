"""Isolated one-shot Hunyuan workers and local hardware preflight."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from ..errors import Codex3DError, GenerationCancelled, ModelUnavailableError
from ..trellis.artifacts import atomic_json
from ..trellis.runtime import GpuLease
from .config import HunyuanMVSettings
from .safe_loader import SAFE_LOADER_ID, is_torch_zip_checkpoint, shape_checkpoint_path

PINNED_TORCH_STACK = {
    "torch": "2.9.1",
    "torchvision": "0.24.1",
}

PINNED_HF_STACK = {
    "diffusers": "0.31.0",
    "transformers": "4.48.3",
    "huggingface_hub": "0.26.5",
    "accelerate": "1.1.1",
    "gradio": "4.44.1",
    "rembg": "2.0.67",
}


class HunyuanMVRuntime:
    """Run shape and paint in separate processes so VRAM is released between stages."""

    def __init__(self, settings: HunyuanMVSettings) -> None:
        self.settings = settings
        self.gpu_lease = GpuLease(settings.gpu_lock_path)
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.RLock()
        self._dependency_probe = self._probe_python() if settings.python_exe.is_file() else {}

    def preflight(self) -> dict[str, Any]:
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.settings.output_dir).free
        python_present = self.settings.python_exe.is_file()
        upstream_present = (self.settings.upstream_dir / "hy3dgen").is_dir()
        shape_snapshot = self._shape_snapshot()
        texture_snapshot = self._texture_snapshot()
        shape_checkpoint = shape_checkpoint_path(shape_snapshot, self.settings.shape_subfolder)
        shape_config_present = (
            shape_snapshot / self.settings.shape_subfolder / "config.yaml"
        ).is_file()
        shape_checkpoint_present = shape_checkpoint.is_file()
        shape_checkpoint_format_verified = is_torch_zip_checkpoint(shape_checkpoint)
        shape_cache_present = bool(
            shape_config_present and shape_checkpoint_present and shape_checkpoint_format_verified
        )
        texture_cache_present = all(
            path.is_file()
            for path in (
                texture_snapshot
                / "hunyuan3d-paint-v2-0"
                / "unet"
                / "diffusion_pytorch_model.safetensors",
                texture_snapshot
                / "hunyuan3d-delight-v2-0"
                / "unet"
                / "diffusion_pytorch_model.safetensors",
            )
        )
        manifest_verified, manifest_error = self.verify_install_manifest()
        manifest = self._install_manifest()
        dependencies = self._dependency_probe
        ready = bool(
            self.settings.test_mode
            or (
                python_present
                and upstream_present
                and shape_cache_present
                and texture_cache_present
                and manifest_verified
                and all(
                    _base_version(dependencies.get(name)) == value
                    for name, value in PINNED_TORCH_STACK.items()
                )
                and dependencies.get("hy3dgen")
                and dependencies.get("custom_rasterizer_kernel")
                and dependencies.get("mesh_processor")
                and all(dependencies.get(name) == value for name, value in PINNED_HF_STACK.items())
                and manifest.get("cuda_verified") is True
                and self.settings.blender_exe.is_file()
                and free >= self.settings.minimum_free_bytes
            )
        )
        return {
            "ready_for_generation": ready,
            "python_present": python_present,
            "python_exe": str(self.settings.python_exe),
            "upstream_present": upstream_present,
            "upstream_dir": str(self.settings.upstream_dir),
            "shape_cache_present": shape_cache_present,
            "shape_snapshot": str(shape_snapshot),
            "shape_config_present": shape_config_present,
            "shape_checkpoint_present": shape_checkpoint_present,
            "shape_checkpoint_format_verified": shape_checkpoint_format_verified,
            "shape_checkpoint": str(shape_checkpoint),
            "shape_safe_loader": SAFE_LOADER_ID,
            "texture_cache_present": texture_cache_present,
            "texture_snapshot": str(texture_snapshot),
            "model_cache_dir": str(self.settings.model_cache_dir),
            "install_manifest_verified": manifest_verified,
            "install_manifest_error": manifest_error,
            "dependencies": dependencies,
            "gpu": manifest.get("gpu", {}),
            "blender_present": self.settings.blender_exe.is_file(),
            "gltf_validator_present": self.settings.gltf_validator.is_file(),
            "disk_free_bytes": free,
            "disk_ready": free >= self.settings.minimum_free_bytes,
            "install_disk_ready": free >= self.settings.install_free_bytes,
            "minimum_job_free_bytes": self.settings.minimum_free_bytes,
            "recommended_install_free_bytes": self.settings.install_free_bytes,
        }

    def verify_install_manifest(self) -> tuple[bool, str | None]:
        value = self._install_manifest()
        if not value:
            return False, "install-manifest.json is missing"
        expected = {
            "upstream_commit": self.settings.upstream_commit,
            "shape_revision": self.settings.shape_revision,
            "texture_revision": self.settings.texture_revision,
            "shape_loader": SAFE_LOADER_ID,
            **PINNED_TORCH_STACK,
            **PINNED_HF_STACK,
        }
        for key, wanted in expected.items():
            if value.get(key) != wanted:
                return False, f"{key} does not match the pinned runtime"
        if value.get("cuda_verified") is not True:
            return False, "the installer did not verify CUDA"
        return True, None

    def _install_manifest(self) -> dict[str, Any]:
        manifest = self.settings.python_exe.parents[2] / "install-manifest.json"
        if not manifest.is_file():
            return {}
        try:
            value = json.loads(manifest.read_text(encoding="utf-8-sig"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _shape_snapshot(self) -> Path:
        return (
            self.settings.model_cache_dir
            / "hub"
            / f"models--{self.settings.shape_model_id.replace('/', '--')}"
            / "snapshots"
            / self.settings.shape_revision
        )

    def _texture_snapshot(self) -> Path:
        return (
            self.settings.model_cache_dir
            / "hub"
            / f"models--{self.settings.texture_model_id.replace('/', '--')}"
            / "snapshots"
            / self.settings.texture_revision
        )

    def generate_shape(
        self,
        views: dict[str, Path],
        output: Path,
        params: dict[str, Any],
        log_path: Path,
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        request = {
            "action": "shape",
            "views": {name: str(path) for name, path in views.items()},
            "output": str(output),
            "model_path": str(self._shape_snapshot()),
            "subfolder": self.settings.shape_subfolder,
            "seed": params["seed"],
            "steps": params["steps"],
            "guidance_scale": params["guidance_scale"],
            "octree_resolution": params["octree_resolution"],
            "num_chunks": params["num_chunks"],
            "shape_loader_id": SAFE_LOADER_ID,
        }
        return self._run_worker(request, output, log_path, cancel_event)

    def generate_texture(
        self,
        mesh: Path,
        front: Path,
        output: Path,
        log_path: Path,
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        request = {
            "action": "texture",
            "mesh": str(mesh),
            "image": str(front),
            "output": str(output),
            "model_path": str(self._texture_snapshot()),
            "subfolder": self.settings.texture_subfolder,
            "low_vram_mode": self.settings.low_vram_mode,
            "texture_resolution": self.settings.texture_resolution,
        }
        return self._run_worker(request, output, log_path, cancel_event)

    def _run_worker(
        self,
        request: dict[str, Any],
        output: Path,
        log_path: Path,
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        if not self.settings.test_mode and not self.settings.python_exe.is_file():
            raise ModelUnavailableError(
                "Hunyuan runtime is not installed; run setup-hunyuan-mv.ps1."
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        request_path = output.with_suffix(f".{request['action']}.json")
        result_path = output.with_suffix(f".{request['action']}.result.json")
        # A resumed job must prove this attempt produced its artifacts.
        result_path.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        request["result"] = str(result_path)
        atomic_json(request_path, request)
        command = [
            str(self.settings.python_exe),
            "-m",
            "codex_3d_mcp.hunyuan_mv.worker",
            "--request",
            str(request_path),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HOME": str(self.settings.model_cache_dir),
                "HUGGINGFACE_HUB_CACHE": str(self.settings.model_cache_dir / "hub"),
                "TORCH_HOME": str(self.settings.model_cache_dir / "torch"),
                "U2NET_HOME": str(self.settings.model_cache_dir / "rembg"),
                "TEMP": str(self.settings.model_cache_dir / "temp"),
                "TMP": str(self.settings.model_cache_dir / "temp"),
                "HUNYUAN3D_UPSTREAM": str(self.settings.upstream_dir),
            }
        )
        Path(environment["TEMP"]).mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.gpu_lease.acquire(timeout=600, cancel_event=cancel_event)
        try:
            with log_path.open("ab") as log:
                with self._lock:
                    self._process = subprocess.Popen(
                        command,
                        cwd=self.settings.base_dir,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                    process = self._process
                deadline = time.monotonic() + self.settings.request_timeout_seconds
                while process.poll() is None:
                    if cancel_event.wait(0.25):
                        self.stop()
                        raise GenerationCancelled("Hunyuan generation was cancelled.")
                    if time.monotonic() >= deadline:
                        self.stop()
                        raise Codex3DError(
                            f"Hunyuan {request['action']} worker exceeded "
                            f"{self.settings.request_timeout_seconds} seconds.",
                            code="HUNYUAN_WORKER_TIMEOUT",
                        )
                if process.returncode != 0:
                    detail = _tail(log_path)
                    if process.returncode in {0xC0000005, -1073741819}:
                        raise Codex3DError(
                            "Hunyuan hit the Windows mapped-checkpoint access violation. "
                            "The safe-loader contract was bypassed or regressed. "
                            f"Worker log: {detail}",
                            code="HUNYUAN_MAPPED_CHECKPOINT_REGRESSION",
                        )
                    raise Codex3DError(
                        f"Hunyuan {request['action']} worker failed: {detail}",
                        code=f"HUNYUAN_{request['action'].upper()}_FAILED",
                    )
        finally:
            with self._lock:
                if self._process is not None and self._process.poll() is None:
                    self.stop()
                    if self._process.poll() is None:
                        raise Codex3DError("Model worker is still alive.", code="WORKER_BUSY")
                self._process = None
            self.gpu_lease.release()
        if not _valid_glb(output):
            raise Codex3DError("Hunyuan returned no valid GLB.", code="INVALID_GLB")
        if not result_path.is_file():
            raise Codex3DError("Hunyuan worker omitted its result record.", code="INVALID_RESULT")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if request["action"] == "shape":
            _validate_safe_loader_proof(result, self.settings)
        return result

    def stop(self) -> None:
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        _terminate_process_tree(process)

    def close(self) -> None:
        self.stop()

    def _probe_python(self) -> dict[str, Any]:
        # The lightweight MCP interpreter can differ from the model interpreter.
        # Inspect the configured environment without importing torch or CUDA.
        script = """
import importlib.metadata, importlib.util, json, sys
sys.path.insert(0, sys.argv[1])
result = {}
for name in json.loads(sys.argv[2]):
    try:
        result[name] = importlib.metadata.version(name.replace('_', '-'))
    except importlib.metadata.PackageNotFoundError:
        result[name] = None
for name in ('hy3dgen', 'custom_rasterizer_kernel', 'mesh_processor'):
    result[name] = importlib.util.find_spec(name) is not None
print(json.dumps(result))
"""
        try:
            completed = subprocess.run(
                [
                    str(self.settings.python_exe), "-c", script,
                    str(self.settings.upstream_dir),
                    json.dumps(list(PINNED_TORCH_STACK) + list(PINNED_HF_STACK)),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=20,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            value = json.loads(completed.stdout)
            return value if isinstance(value, dict) else {"error": "Invalid dependency probe"}
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            return {"error": str(exc)}


class ResidentStageRuntime(HunyuanMVRuntime):
    """One resident model for one sequential batch phase, never two models.

    The same machine-wide lease is also held during CPU-only phases so another
    local 3D server cannot begin GPU work while this batch is remeshing.
    """

    def __init__(self, settings: HunyuanMVSettings) -> None:
        super().__init__(settings)
        self.active_stage: str | None = None
        self._stage_log: Any = None

    def begin_stage(self, stage: str, log_path: Path, cancel_event: threading.Event) -> None:
        if self.active_stage == stage and (self._process is None or self._process.poll() is None):
            return
        self.end_stage()
        self.gpu_lease.acquire(timeout=600, cancel_event=cancel_event)
        try:
            self.active_stage = stage
            if stage not in {"shape", "texture"}:
                return
            environment = os.environ.copy()
            environment.update(
                {
                    "HF_HOME": str(self.settings.model_cache_dir),
                    "HUGGINGFACE_HUB_CACHE": str(self.settings.model_cache_dir / "hub"),
                    "TORCH_HOME": str(self.settings.model_cache_dir / "torch"),
                    "U2NET_HOME": str(self.settings.model_cache_dir / "rembg"),
                    "TEMP": str(self.settings.model_cache_dir / "temp"),
                    "TMP": str(self.settings.model_cache_dir / "temp"),
                    "HUNYUAN3D_UPSTREAM": str(self.settings.upstream_dir),
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2",
                }
            )
            Path(environment["TEMP"]).mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._stage_log = log_path.open("ab")
            self._process = subprocess.Popen(
                [
                    str(self.settings.python_exe),
                    "-m",
                    "codex_3d_mcp.hunyuan_mv.worker",
                    "--serve-stage",
                    stage,
                ],
                cwd=self.settings.base_dir,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=self._stage_log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except BaseException:
            self.end_stage()
            raise

    def end_stage(self) -> None:
        process = self._process
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process)
            if process.poll() is None:
                raise Codex3DError("Previous model worker is still alive", code="WORKER_BUSY")
        # Preserve process ownership and the lease if any shutdown operation
        # above fails. A live model must never overlap the next phase.
        self._process = None
        self.active_stage = None
        if self._stage_log is not None:
            self._stage_log.close()
            self._stage_log = None
        self.gpu_lease.release()

    def _run_worker(self, request, output, log_path, cancel_event):
        self.begin_stage(request["action"], log_path, cancel_event)
        process = self._process
        if process is None or process.stdin is None:
            raise Codex3DError("Resident worker is unavailable", code="WORKER_UNAVAILABLE")
        output.parent.mkdir(parents=True, exist_ok=True)
        request_path = output.with_suffix(f".{request['action']}.json")
        result_path = output.with_suffix(f".{request['action']}.result.json")
        if result_path.exists() or output.exists():
            raise Codex3DError("An attempt must use new output paths", code="ATTEMPT_EXISTS")
        request["result"] = str(result_path)
        atomic_json(request_path, request)
        try:
            process.stdin.write((json.dumps({"request": str(request_path)}) + "\n").encode())
            process.stdin.flush()
            deadline = time.monotonic() + self.settings.request_timeout_seconds
            while not result_path.is_file():
                if cancel_event.wait(0.2):
                    raise GenerationCancelled("Batch asset cancelled")
                from .remesh import _check_memory

                _check_memory({"memory_limit_mb": 0, "min_available_mb": 2048}, starting=False)
                if process.poll() is not None:
                    raise Codex3DError(
                        f"Resident {request['action']} worker stopped; see {log_path}",
                        code=f"HUNYUAN_{request['action'].upper()}_FAILED",
                    )
                if time.monotonic() > deadline:
                    raise Codex3DError("Resident worker timed out", code="HUNYUAN_WORKER_TIMEOUT")
            response = json.loads(result_path.read_text(encoding="utf-8"))
            if not response.get("ok"):
                raise Codex3DError(
                    response.get("error", "Worker failed"),
                    code=f"HUNYUAN_{request['action'].upper()}_FAILED",
                )
            with output.open("rb") as artifact:
                if artifact.read(4) != b"glTF" or output.stat().st_size < 20:
                    raise Codex3DError("Worker returned invalid GLB", code="INVALID_GLB")
            result = response["result"]
            if request["action"] == "shape":
                _validate_safe_loader_proof(result, self.settings)
            return result
        except BaseException:
            self.end_stage()
            raise

    def close(self) -> None:
        self.end_stage()


def _base_version(value: Any) -> str:
    return str(value or "").split("+", 1)[0]


def _tail(path: Path, limit: int = 3000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - limit * 4))
            return handle.read().decode("utf-8", "replace")[-limit:]
    except OSError:
        return "See the worker log."


def _validate_safe_loader_proof(result: dict[str, Any], settings: HunyuanMVSettings) -> None:
    proof = result.get("runtime")
    expected_checkpoint = (
        settings.model_cache_dir
        / "hub"
        / f"models--{settings.shape_model_id.replace('/', '--')}"
        / "snapshots"
        / settings.shape_revision
        / settings.shape_subfolder
        / "model.fp16.ckpt"
    ).resolve()
    required = {
        "ready": True,
        "safe_loader": SAFE_LOADER_ID,
        "checkpoint_format": "torch_zip",
        "checkpoint_loads": 1,
        "checkpoint_mmap": False,
        "vae_preallocated": True,
        "vae_owned_copy": True,
        "meta_tensor_count": 0,
        "cuda_synchronized": True,
    }
    if not isinstance(proof, dict):
        raise Codex3DError(
            "Hunyuan shape worker omitted its safe-loader proof.",
            code="HUNYUAN_SAFE_LOADER_UNPROVEN",
        )
    mismatches = [
        f"{key}={proof.get(key)!r} (expected {value!r})"
        for key, value in required.items()
        if proof.get(key) != value
    ]
    checkpoint = Path(str(proof.get("checkpoint", ""))).resolve()
    if checkpoint != expected_checkpoint:
        mismatches.append(f"checkpoint={checkpoint} (expected {expected_checkpoint})")
    if mismatches:
        raise Codex3DError(
            "Hunyuan safe-loader proof failed: " + "; ".join(mismatches),
            code="HUNYUAN_SAFE_LOADER_UNPROVEN",
        )


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate the exact worker and every descendant it spawned."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
            timeout=30,
        )
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _valid_glb(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 20:
        return False
    with path.open("rb") as handle:
        return handle.read(4) == b"glTF"
