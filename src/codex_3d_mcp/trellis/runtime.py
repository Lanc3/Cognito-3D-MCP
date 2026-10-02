"""Native TRELLIS runtime supervision, HTTP client, and GPU arbitration."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from ..errors import Codex3DError, GenerationCancelled, ModelUnavailableError
from .config import TrellisSettings

REQUIRED_Q8_MODELS = (
    "birefnet.gguf",
    "dinov3.gguf",
    "shape_dec.gguf",
    "shape_flow_1024.gguf",
    "shape_flow_512.gguf",
    "ss_dec.gguf",
    "ss_flow.gguf",
    "tex_dec.gguf",
    "tex_flow_1024.gguf",
    "tex_flow_512.gguf",
)


class GpuLease:
    """Cross-process exclusive lock backed by a one-byte file lock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: BinaryIO | None = None
        self._state_lock = threading.Lock()
        self._acquiring = False

    def acquire(self, timeout: float = 60.0, cancel_event: threading.Event | None = None) -> None:
        with self._state_lock:
            if self._handle is not None or self._acquiring:
                raise Codex3DError("This GPU lease is already in use.", code="GPU_BUSY")
            self._acquiring = True
        handle = None
        try:
            started = time.monotonic()
            handle = self.path.open("a+b")
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise GenerationCancelled("Generation cancelled while waiting for the GPU.")
                try:
                    _lock_file(handle)
                    with self._state_lock:
                        self._handle = handle
                    return
                except OSError:
                    if time.monotonic() - started >= timeout:
                        raise Codex3DError(
                            "Timed out waiting for the local GPU lease.", code="GPU_BUSY"
                        )
                    if cancel_event is not None:
                        cancel_event.wait(0.2)
                    else:
                        time.sleep(0.2)
        finally:
            with self._state_lock:
                self._acquiring = False
                if handle is not None and self._handle is not handle:
                    handle.close()

    def release(self) -> None:
        with self._state_lock:
            if self._handle is None:
                return
            try:
                _unlock_file(self._handle)
            finally:
                self._handle.close()
                self._handle = None

    def __enter__(self) -> GpuLease:
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


def _lock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class TrellisRuntime:
    def __init__(self, settings: TrellisSettings) -> None:
        self.settings = settings
        self.gpu_lease = GpuLease(settings.gpu_lock_path)
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: BinaryIO | None = None
        self._lock = threading.RLock()

    @property
    def base_url(self) -> str:
        return f"http://{self.settings.host}:{self.settings.port}"

    def preflight(self) -> dict:
        model_files = {
            name: (self.settings.model_dir / name).is_file() for name in REQUIRED_Q8_MODELS
        }
        free = shutil.disk_usage(self.settings.output_dir).free
        return {
            "runtime_version": self.settings.runtime_version,
            "runtime_commit": self.settings.runtime_commit,
            "weights_revision": self.settings.weights_revision,
            "trellis_server_present": self.settings.trellis_server.is_file(),
            "trellis_cli_present": self.settings.trellis_cli.is_file(),
            "q8_models": model_files,
            "q8_models_complete": all(model_files.values()),
            "realesrgan_present": self.settings.realesrgan_exe.is_file(),
            "blender_present": self.settings.blender_exe.is_file(),
            "gltf_validator_present": self.settings.gltf_validator.is_file(),
            "disk_free_bytes": free,
            "disk_ready": free >= self.settings.minimum_free_bytes,
            "healthy": self.health(),
            "ready_for_generation": (
                self.settings.test_mode
                or (
                    self.settings.trellis_server.is_file()
                    and self.settings.trellis_cli.is_file()
                    and all(model_files.values())
                    and self.settings.realesrgan_exe.is_file()
                    and self.settings.blender_exe.is_file()
                    and self.settings.gltf_validator.is_file()
                    and free >= self.settings.minimum_free_bytes
                )
            ),
        }

    def verify_runtime_hash(self) -> tuple[bool, str | None]:
        manifest = self.settings.runtime_dir / "install-manifest.json"
        if not manifest.is_file():
            return False, "install-manifest.json is missing"
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, str(exc)
        if data.get("archive_sha256") != self.settings.runtime_archive_sha256:
            return False, "runtime archive hash does not match the pinned release"
        if data.get("weights_revision") != self.settings.weights_revision:
            return False, "weights revision does not match the pinned release"
        return True, None

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=1.0) as response:
                return response.status == 200 and response.read().strip() == b"ok"
        except (OSError, urllib.error.URLError):
            return False

    def start(self, log_path: Path) -> None:
        with self._lock:
            if self.health():
                return
            if not self.settings.trellis_server.is_file():
                raise ModelUnavailableError(
                    "TRELLIS server is not installed; run scripts/setup-trellis.ps1."
                )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = log_path.open("ab")
            command = [
                str(self.settings.trellis_server),
                "--host",
                self.settings.host,
                "--port",
                str(self.settings.port),
                "--models",
                str(self.settings.model_dir),
                "--gpu",
                "0",
                "--res",
                str(self.settings.resolution),
                "--atlas",
                str(self.settings.atlas_resolution),
                "--webp",
                "off",
                "--require-gpu",
            ]
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._process = subprocess.Popen(
                command,
                cwd=self.settings.runtime_dir,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    self.stop()
                    raise ModelUnavailableError(
                        "TRELLIS server exited during startup; inspect its log."
                    )
                if self.health():
                    return
                time.sleep(0.25)
            self.stop()
            raise ModelUnavailableError("TRELLIS server did not become healthy within 30 seconds.")

    def stop(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            self._process = None
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

    @contextlib.contextmanager
    def session(
        self, log_path: Path, cancel_event: threading.Event | None = None
    ) -> Iterator[TrellisRuntime]:
        self.gpu_lease.acquire(timeout=600, cancel_event=cancel_event)
        try:
            self.start(log_path)
            yield self
        finally:
            self.stop()
            self.gpu_lease.release()

    def generate(self, image: Path, output: Path, seed: int) -> None:
        fields = {
            "seed": str(seed),
            "resolution": str(self.settings.resolution),
            "bg_removal": "birefnet",
            "uv": "xatlas",
            "band": "2",
            "webp": "off",
        }
        body, content_type = _multipart(image, fields)
        request = urllib.request.Request(
            f"{self.base_url}/generate",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.request_timeout_seconds
            ) as response:
                payload = response.read()
                if response.status != 200 or not payload.startswith(b"glTF"):
                    raise Codex3DError("TRELLIS returned an invalid GLB.", code="INVALID_GLB")
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", "replace")
            raise Codex3DError(
                f"TRELLIS generation failed with HTTP {exc.code}: {detail}",
                code="TRELLIS_FAILED",
            ) from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".partial")
        temporary.write_bytes(payload)
        os.replace(temporary, output)

    def extract_mask(self, image: Path, output: Path, log_path: Path) -> Path:
        if not self.settings.trellis_cli.is_file():
            raise ModelUnavailableError("TRELLIS CLI is missing; run scripts/setup-trellis.ps1.")
        dummy = output.with_suffix(".glb")
        command = [
            str(self.settings.trellis_cli),
            "--image",
            str(image),
            "--output",
            str(dummy),
            "--models",
            str(self.settings.model_dir),
            "--gpu",
            "0",
            "--bg-removal",
            "birefnet",
            "--bg-only",
            "--require-gpu",
        ]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with log_path.open("ab") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
                creationflags=creationflags,
                check=False,
            )
        generated = dummy.with_name(f"{dummy.stem}_cutout.png")
        if result.returncode != 0 or not generated.is_file():
            raise Codex3DError("BiRefNet mask extraction failed.", code="MASK_FAILED")
        generated.replace(output)
        return output

    def close(self) -> None:
        self.stop()


def _multipart(image: Path, fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = f"codex-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="image"; filename="{image.name}"\r\n'
            ).encode(),
            b"Content-Type: image/png\r\n\r\n",
            image.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
