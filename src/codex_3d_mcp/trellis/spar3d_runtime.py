"""SPAR3D reconstruction runtime for the bidirectional production pipeline."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import shutil
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from PIL import Image

from ..errors import Codex3DError, ModelUnavailableError
from .config import TrellisSettings
from .runtime import GpuLease
from .validation import frame_foreground_mask


class Spar3DRuntime:
    """Supervise a resident SPAR3D child and serialize it with other GPU work."""

    def __init__(self, settings: TrellisSettings) -> None:
        self.settings = settings
        self.gpu_lease = GpuLease(settings.gpu_lock_path)
        self._process: subprocess.Popen[str] | None = None
        self._log_handle: BinaryIO | None = None
        self._lock = threading.RLock()

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "CODEX_3D_BACKEND": "spar3d",
                "CODEX_3D_DEVICE": "auto",
                "CODEX_3D_MODEL_ID": self.settings.spar3d_model_id,
                "CODEX_3D_MODEL_CACHE_DIR": str(self.settings.spar3d_model_cache_dir),
                "CODEX_3D_LOW_VRAM_MODE": (
                    "true" if self.settings.spar3d_low_vram_mode else "false"
                ),
                "HF_HOME": str(self.settings.spar3d_model_cache_dir),
            }
        )
        return environment

    def preflight(self) -> dict[str, Any]:
        status: dict[str, Any] = {}
        status_error: str | None = None
        if self.settings.spar3d_python.is_file():
            try:
                result = subprocess.run(
                    [
                        str(self.settings.spar3d_python),
                        "-m",
                        "codex_3d_mcp.spar3d_worker",
                        "--status",
                    ],
                    cwd=self.settings.base_dir,
                    env=self._environment(),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    check=False,
                )
                if result.returncode == 0:
                    lines = [line for line in result.stdout.splitlines() if line.strip()]
                    status = json.loads(lines[-1]) if lines else {}
                else:
                    status_error = (result.stderr or result.stdout)[-2000:]
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
                status_error = str(exc)
        else:
            status_error = "The SPAR3D Python environment is missing."

        free = shutil.disk_usage(self.settings.output_dir).free
        supporting_tools = {
            "realesrgan_present": self.settings.realesrgan_exe.is_file(),
            "blender_present": self.settings.blender_exe.is_file(),
            "gltf_validator_present": self.settings.gltf_validator.is_file(),
        }
        model_ready = bool(status.get("ready_for_generation"))
        return {
            "backend": "spar3d",
            "model_id": self.settings.spar3d_model_id,
            "python": str(self.settings.spar3d_python),
            "python_present": self.settings.spar3d_python.is_file(),
            "model": status,
            "status_error": status_error,
            **supporting_tools,
            "disk_free_bytes": free,
            "disk_ready": free >= self.settings.minimum_free_bytes,
            "healthy": self.health(),
            "ready_for_generation": (
                self.settings.test_mode
                or (
                    model_ready
                    and all(supporting_tools.values())
                    and free >= self.settings.minimum_free_bytes
                )
            ),
        }

    def verify_runtime_hash(self) -> tuple[bool, str | None]:
        if not self.settings.spar3d_python.is_file():
            return False, "SPAR3D Python environment is missing"
        cache = self.settings.spar3d_model_cache_dir
        repository = f"models--{self.settings.spar3d_model_id.replace('/', '--')}"
        snapshot_root = cache / "hub" / repository / "snapshots"
        if not any(
            (snapshot / "config.yaml").is_file() and (snapshot / "model.safetensors").is_file()
            for snapshot in snapshot_root.glob("*") if snapshot.is_dir()
        ):
            return False, "The complete pinned SPAR3D checkpoint is not present locally"
        return True, None

    def health(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    @staticmethod
    def _readline(stream: TextIO, timeout: float) -> str:
        result: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

        def read() -> None:
            try:
                result.put(stream.readline())
            except BaseException as exc:
                result.put(exc)

        threading.Thread(target=read, name="spar3d-protocol-reader", daemon=True).start()
        try:
            value = result.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("SPAR3D worker response timed out") from exc
        if isinstance(value, BaseException):
            raise value
        return value

    def start(self, log_path: Path) -> None:
        with self._lock:
            if self.health():
                return
            if not self.settings.spar3d_python.is_file():
                raise ModelUnavailableError("SPAR3D environment is not installed.")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = log_path.open("ab")
            self._process = subprocess.Popen(
                [
                    str(self.settings.spar3d_python),
                    "-m",
                    "codex_3d_mcp.spar3d_worker",
                    "--serve",
                ],
                cwd=self.settings.base_dir,
                env=self._environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log_handle,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            process = self._process
        assert process.stdout is not None
        try:
            response = json.loads(self._readline(process.stdout, 30))
        except (TimeoutError, json.JSONDecodeError) as exc:
            self.stop()
            raise ModelUnavailableError("SPAR3D worker did not start correctly.") from exc
        if not response.get("ok") or response.get("state") != "ready":
            self.stop()
            raise ModelUnavailableError("SPAR3D worker rejected its startup handshake.")

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    if process.stdin is not None:
                        process.stdin.write('{"action":"shutdown"}\n')
                        process.stdin.flush()
                    process.wait(timeout=15)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            capture_output=True,
                            creationflags=subprocess.CREATE_NO_WINDOW,
                            timeout=15,
                            check=False,
                        )
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                if process.poll() is None:
                    raise Codex3DError("SPAR3D model worker is still alive.", code="WORKER_BUSY")
            # Preserve ownership if termination fails; session() must retain its lease.
            self._process = None
            if process is not None:
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

    @contextlib.contextmanager
    def session(
        self, log_path: Path, cancel_event: threading.Event | None = None
    ) -> Iterator[Spar3DRuntime]:
        self.gpu_lease.acquire(timeout=600, cancel_event=cancel_event)
        try:
            self.start(log_path)
            yield self
        finally:
            self.stop()
            self.gpu_lease.release()

    def generate(self, image: Path, output: Path, seed: int) -> None:
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise ModelUnavailableError("SPAR3D worker is not running.")
        if process.stdout is None:
            raise ModelUnavailableError("SPAR3D worker has no response channel.")

        temporary = output.with_suffix(".partial.glb")
        temporary.unlink(missing_ok=True)
        request = {
            "action": "generate",
            "image": str(image),
            "output": str(temporary),
            "seed": seed,
            "texture_resolution": self.settings.atlas_resolution,
            "foreground_ratio": 0.85,
            "remesh": "none",
            "target_vertex_count": None,
        }
        try:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
            line = self._readline(process.stdout, self.settings.request_timeout_seconds)
            response = json.loads(line)
        except (BrokenPipeError, OSError, TimeoutError, json.JSONDecodeError) as exc:
            self.stop()
            raise Codex3DError(
                "SPAR3D worker failed during generation.", code="SPAR3D_FAILED"
            ) from exc
        if not response.get("ok"):
            raise Codex3DError(
                response.get("message", "SPAR3D generation failed."),
                code=response.get("code", "SPAR3D_FAILED"),
            )
        if not temporary.is_file() or temporary.stat().st_size < 16:
            raise Codex3DError("SPAR3D returned no GLB artifact.", code="INVALID_GLB")
        with temporary.open("rb") as handle:
            if handle.read(4) != b"glTF":
                raise Codex3DError("SPAR3D returned an invalid GLB.", code="INVALID_GLB")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)

    def extract_mask(self, image: Path, output: Path, log_path: Path) -> Path:
        """Create a validation cutout without loading a second reconstruction model."""
        output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image) as source:
            rgb = source.convert("RGB")
            mask = frame_foreground_mask(source)
            cutout = Image.new("RGB", source.size, (0, 0, 0))
            cutout.paste(rgb, mask=mask)
            cutout.save(output, "PNG")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"SPAR3D validation mask: {image} -> {output}\n")
        return output

    def close(self) -> None:
        self.stop()
