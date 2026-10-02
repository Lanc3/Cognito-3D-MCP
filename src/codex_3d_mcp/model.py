"""Local image-to-3D model adapter.

The heavy model imports are intentionally deferred until generation/status is
requested so the MCP server can initialize and explain missing dependencies
without loading several gigabytes of weights.
"""

from __future__ import annotations

import contextlib
import gc
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import Codex3DError, GenerationCancelled, ModelUnavailableError

ProgressCallback = Callable[[int, str], None]


class ImageTo3DAdapter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any = None
        self._device: str | None = None
        self._load_lock = threading.Lock()
        self._model_access_checked = False
        self._model_access: bool | None = None
        self._model_access_error: str | None = None
        os.environ.setdefault("HF_HOME", str(self.settings.model_cache_dir))
        if self.settings.backend == "spar3d":
            # The checkpoint is a single 7.3 GB file. Give the resumable
            # downloader enough time for slow Windows/CDN connections.
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
            auxiliary_cache = self.settings.model_cache_dir.parent / "spar3d-aux"
            alpha_clip_cache = auxiliary_cache / "alpha-clip"
            temp_dir = auxiliary_cache / "temp"
            alpha_clip_cache.mkdir(parents=True, exist_ok=True)
            temp_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("ALPHA_CLIP_PATH", str(alpha_clip_cache))
            for variable in ("TEMP", "TMP", "TMPDIR"):
                os.environ[variable] = str(temp_dir)
        self._ensure_vendor_import_path()

    @property
    def model_name(self) -> str:
        return "SPAR3D" if self.settings.backend == "spar3d" else "Stable Fast 3D"

    def _ensure_vendor_import_path(self) -> None:
        """Make the selected pinned source checkout importable."""
        vendor_name = {
            "sf3d": "stable-fast-3d",
            "spar3d": "stable-point-aware-3d",
        }.get(self.settings.backend)
        if vendor_name is None:
            raise ModelUnavailableError(
                f"Unsupported CODEX_3D_BACKEND '{self.settings.backend}'. Use sf3d or spar3d."
            )
        vendor_root = Path(__file__).resolve().parents[2] / ".vendor" / vendor_name
        if vendor_root.is_dir() and str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))

    def _import_status(self) -> dict[str, Any]:
        backend_imports = (
            ("transparent_background", "spar3d")
            if self.settings.backend == "spar3d"
            else ("rembg", "sf3d")
        )
        return {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "PIL", *backend_imports)
        }

    def _resolve_device(self) -> str:
        requested = self.settings.device
        if requested not in {"auto", "cuda", "mps", "cpu"}:
            raise ModelUnavailableError(
                f"Unsupported CODEX_3D_DEVICE '{requested}'. Use auto, cuda, mps, or cpu."
            )
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailableError(
                "PyTorch is not installed. Install the matching PyTorch build first."
            ) from exc

        force_cpu_name = "SPAR3D_USE_CPU" if self.settings.backend == "spar3d" else "SF3D_USE_CPU"
        if os.getenv(force_cpu_name) == "1":
            return "cpu"
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise ModelUnavailableError("CUDA was requested but no CUDA device is available.")
            return "cuda"
        if requested == "mps":
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                raise ModelUnavailableError("MPS was requested but no MPS device is available.")
            return "mps"
        if requested == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch

                # Record the target before vendor imports/loading can allocate tensors.
                self._device = self._resolve_device()

                if self.settings.backend == "spar3d":
                    from spar3d.system import SPAR3D as ModelClass
                else:
                    from sf3d.system import SF3D as ModelClass
            except Exception as exc:
                raise ModelUnavailableError(
                    f"{self.model_name} dependencies are unavailable. Run the matching "
                    "setup script and inspect server_status."
                ) from exc
            device = self._device
            try:
                load_args: dict[str, Any] = {
                    "config_name": "config.yaml",
                    "weight_name": "model.safetensors",
                }
                if self.settings.backend == "spar3d":
                    load_args["low_vram_mode"] = self.settings.low_vram_mode
                with contextlib.redirect_stdout(sys.stderr):
                    model = ModelClass.from_pretrained(self.settings.model_id, **load_args)
                    model.to(device)
                    model.eval()
            except Exception as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if exc.__class__.__name__ == "GatedRepoError" or status_code == 403:
                    raise ModelUnavailableError(
                        f"The configured Hugging Face account has not been granted access to "
                        f"{self.settings.model_id}. Request access and accept the model terms, "
                        "then restart the server."
                    ) from exc
                raise ModelUnavailableError(
                    f"{self.model_name} could not be loaded. Confirm Hugging Face access, "
                    "the model cache, and the CUDA/compiled dependencies. "
                    f"Underlying error: {exc.__class__.__name__}: {exc}"
                ) from exc
            self._model = model
            self._device = device
            del torch

    def status(self) -> dict[str, Any]:
        imports = self._import_status()
        device, torch_version, device_error = self._status_device(imports["torch"])
        vram_gb = self._status_vram_gb() if device == "cuda" else None
        cache_present = self._model_cache_complete()
        token = self._configured_hf_token()
        authenticated = token is not None
        model_access: bool | None = True if cache_present else None
        access_error: str | None = None
        if not cache_present:
            model_access, access_error = self._status_model_access(token)
        return {
            "backend": self.settings.backend,
            "model_name": self.model_name,
            "model_id": self.settings.model_id,
            "low_vram_mode": self.settings.low_vram_mode,
            "model_loaded": self._model is not None,
            "model_cache_dir": str(self.settings.model_cache_dir),
            "model_cache_present": cache_present,
            "huggingface_authenticated": authenticated,
            "huggingface_model_access": model_access,
            "huggingface_access_error": access_error,
            "dependencies": imports,
            "torch_version": torch_version,
            "device": device,
            "vram_gb": vram_gb,
            "device_error": device_error,
            "ready_for_generation": (
                all(imports.values())
                and device is not None
                and (cache_present or model_access is True)
            ),
            "windows_support": "experimental",
        }

    def _model_cache_complete(self) -> bool:
        repo_cache_name = f"models--{self.settings.model_id.replace('/', '--')}"
        cache_roots = (
            self.settings.model_cache_dir / repo_cache_name,
            self.settings.model_cache_dir / "hub" / repo_cache_name,
        )
        for cache_root in cache_roots:
            snapshots = cache_root / "snapshots"
            if not snapshots.is_dir():
                continue
            for snapshot in snapshots.iterdir():
                if (
                    snapshot.is_dir()
                    and (snapshot / "config.yaml").is_file()
                    and (snapshot / "model.safetensors").is_file()
                ):
                    return True
        return False

    def _configured_hf_token(self) -> str | None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token and token.strip():
            return token.strip()
        token_paths = []
        if os.getenv("HF_TOKEN_PATH"):
            token_paths.append(Path(os.environ["HF_TOKEN_PATH"]).expanduser())
        if os.getenv("HF_HOME"):
            token_paths.append(Path(os.environ["HF_HOME"]).expanduser() / "token")
        token_paths.extend((
            self.settings.model_cache_dir / "token",
            Path.home() / ".cache" / "huggingface" / "token",
        ))
        for path in token_paths:
            try:
                token = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                continue
            if token:
                return token
        return None

    def _status_model_access(self, token: str | None) -> tuple[bool | None, str | None]:
        if self._model_access_checked:
            return self._model_access, self._model_access_error

        if token is None:
            result: tuple[bool | None, str | None] = (
                False,
                "No Hugging Face token is configured.",
            )
        else:
            try:
                from huggingface_hub import hf_hub_download

                hf_hub_download(
                    repo_id=self.settings.model_id,
                    filename="config.yaml",
                    cache_dir=self.settings.model_cache_dir / "hub",
                    token=token,
                    etag_timeout=5,
                )
            except Exception as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if exc.__class__.__name__ == "GatedRepoError" or status_code == 403:
                    result = (
                        False,
                        f"The configured Hugging Face account has not been granted access to "
                        f"{self.settings.model_id}.",
                    )
                elif status_code == 401:
                    result = (False, "The configured Hugging Face token was rejected.")
                else:
                    result = (
                        None,
                        f"Could not verify Hugging Face model access ({exc.__class__.__name__}).",
                    )
            else:
                result = (True, None)

        self._model_access, self._model_access_error = result
        self._model_access_checked = True
        return result

    def _status_device(self, torch_present: bool) -> tuple[str | None, str | None, str | None]:
        requested = self.settings.device
        if requested not in {"auto", "cuda", "mps", "cpu"}:
            return (
                None,
                None,
                (f"Unsupported CODEX_3D_DEVICE '{requested}'. Use auto, cuda, mps, or cpu."),
            )
        if not torch_present:
            return None, None, "PyTorch is not installed."
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            return None, None, "PyTorch package metadata is unavailable."
        if requested != "auto":
            return requested, torch_version, None
        if "+cu" in torch_version:
            return "cuda", torch_version, None
        if sys.platform == "darwin":
            return "mps", torch_version, None
        return "cpu", torch_version, None

    @staticmethod
    def _status_vram_gb() -> float | None:
        try:
            nvidia_smi = "nvidia-smi"
            if sys.platform == "win32" and os.getenv("SYSTEMROOT"):
                nvidia_smi = str(Path(os.environ["SYSTEMROOT"]) / "System32" / "nvidia-smi.exe")
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=3,
            )
            memory_mib = float(result.stdout.splitlines()[0].strip())
            return round(memory_mib / 1024, 2)
        except (OSError, ValueError, subprocess.SubprocessError, IndexError):
            return None

    def generate(
        self,
        image_path: Path,
        output_path: Path,
        *,
        texture_resolution: int,
        foreground_ratio: float,
        remesh: str,
        target_vertex_count: int | None,
        progress: ProgressCallback,
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        self._check_cancel(cancel_event)
        progress(5, "loading model")
        self._load()
        self._check_cancel(cancel_event)

        try:
            import torch
            from PIL import Image

            if self.settings.backend == "spar3d":
                from spar3d.utils import foreground_crop, remove_background
                from transparent_background import Remover
            else:
                import rembg
                from sf3d.utils import remove_background, resize_foreground
        except Exception as exc:
            raise ModelUnavailableError(
                f"{self.model_name} preprocessing dependencies are unavailable."
            ) from exc

        progress(15, "preparing image")
        try:
            image = Image.open(image_path).convert("RGBA")
            with contextlib.redirect_stdout(sys.stderr):
                if self.settings.backend == "spar3d":
                    remover = Remover(device=self._device)
                    image = remove_background(image, remover)
                    image = foreground_crop(image, crop_ratio=1.0 / foreground_ratio)
                else:
                    session = rembg.new_session()
                    image = remove_background(image, session)
                    image = resize_foreground(image, foreground_ratio)
        except Exception as exc:
            raise ModelUnavailableError(f"Could not preprocess the input image: {exc}") from exc
        self._check_cancel(cancel_event)

        progress(30, "generating textured mesh")
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._device and "cuda" in self._device
            else contextlib.nullcontext()
        )
        with contextlib.redirect_stdout(sys.stderr), torch.no_grad(), autocast:
            run_args: dict[str, Any] = {
                "bake_resolution": texture_resolution,
                "remesh": remesh,
                "vertex_count": (target_vertex_count if target_vertex_count is not None else -1),
            }
            if self.settings.backend == "spar3d":
                run_args["return_points"] = True
            mesh, _ = self._model.run_image([image], **run_args)
        self._check_cancel(cancel_event)
        if isinstance(mesh, (list, tuple)):
            if not mesh:
                raise ModelUnavailableError(f"{self.model_name} returned no mesh.")
            mesh = mesh[0]

        progress(90, "exporting GLB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            mesh.export(str(output_path), include_normals=True)
        except TypeError:
            mesh.export(str(output_path))
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise ModelUnavailableError(f"{self.model_name} did not produce a valid GLB artifact.")
        progress(100, "completed")

        vertices = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)
        return {
            "backend": self.settings.backend,
            "device": self._device,
            "model_id": self.settings.model_id,
            "vertex_count": int(len(vertices)) if vertices is not None else None,
            "face_count": int(len(faces)) if faces is not None else None,
            "bytes": output_path.stat().st_size,
        }

    @staticmethod
    def _check_cancel(cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            raise GenerationCancelled("Generation was cancelled.")

    def shutdown(self) -> None:
        """Release model tensors and prove the allocator is idle before lease release."""
        with self._load_lock:
            self._model = None
            gc.collect()
            device = self._device
            if device == "cuda":
                import torch

                # Synchronize before emptying the allocator: native inference may
                # otherwise still be using memory after Python references vanish.
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                if allocated or reserved:
                    raise Codex3DError(
                        f"Model unload left {allocated} allocated and {reserved} reserved CUDA bytes; "
                        "restart this server before more GPU work.",
                        code="GPU_UNLOAD_FAILED",
                    )
            elif device == "mps":
                import torch

                torch.mps.synchronize()
                torch.mps.empty_cache()
                if torch.mps.current_allocated_memory():
                    raise Codex3DError("Model unload left MPS tensors alive.", code="GPU_UNLOAD_FAILED")
            self._device = None
        sys.stderr.flush()


# Backward-compatible import for callers created before multi-backend support.
StableFast3DAdapter = ImageTo3DAdapter
