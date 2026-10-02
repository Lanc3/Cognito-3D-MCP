"""Resource-bounded, isolated AutoRemesher stage before Hunyuan Paint."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..errors import Codex3DError, GenerationCancelled

RELEASE = "1.2.0"
SOURCE_COMMIT = "d9ef96bd72f0b134dd7e51acf5904f32a5679704"
ARCHIVE_SHA256 = "f6184622cef84f0bcf032a0474df44e2076f4c10c159f1af0cb7230259af7476"
EXE_SHA256 = "103bd621bfe828d4f1fbc62ace095483eeb184e29e647e739894c3b9cc93e877"
PROFILE_TARGETS = {"full_game": 50_000, "mobile": 25_000, "browser": 10_000}
PROFILE_PARAM_NAMES = {
    f"{name}_{setting}"
    for name in PROFILE_TARGETS for setting in ("target_quads", "triangle_budget")
}


def profile_parameters(params: dict[str, Any]) -> dict[str, int]:
    """Resolve persisted quad hints and independent exported-triangle ceilings."""
    result = {}
    for name, default in PROFILE_TARGETS.items():
        target_key, budget_key = f"{name}_target_quads", f"{name}_triangle_budget"
        fallback = params.get("target_quads", default) if name == "full_game" else default
        target = params.get(target_key, fallback)
        for key, high in ((target_key, 150_000), (budget_key, 300_000)):
            value = target if key == target_key else params.get(budget_key, result[target_key] * 2)
            low = 1000 if key == target_key else 4
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer between {low} and {high}") from exc
            if isinstance(value, bool) or not math.isfinite(number) or not number.is_integer():
                raise ValueError(f"{key} must be an integer between {low} and {high}")
            if not low <= number <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
            result[key] = int(number)
    return result


def _parameters(params: dict[str, Any], settings: Any) -> dict[str, Any]:
    result = {
        "target_quads": params.get("target_quads", settings.remesh_target_quads),
        "edge_scaling": params.get("edge_scaling", 1.0),
        "sharp_edge": params.get("sharp_edge", 90.0),
        "smooth_normal": params.get("smooth_normal", 0.0),
        "adaptivity": params.get("adaptivity", 1.0),
        "anisotropy": params.get("anisotropy", 1.0),
        "threads": params.get("threads", getattr(settings, "remesh_threads", 2)),
        "memory_limit_mb": params.get("memory_limit_mb", 4096),
        "min_available_mb": params.get("min_available_mb", 2048),
    }
    ranges = {
        "target_quads": (1000, 1000000),
        "edge_scaling": (1, 4),
        "sharp_edge": (30, 180),
        "smooth_normal": (0, 180),
        "adaptivity": (0, 1),
        "anisotropy": (0, 1),
        "threads": (1, min(2, getattr(settings, "remesh_threads", 2))),
        "memory_limit_mb": (256, 4096),
        "min_available_mb": (2048, 65536),
    }
    integers = {"target_quads", "threads", "memory_limit_mb", "min_available_mb"}
    if "triangle_budget" in params:
        result["triangle_budget"] = params["triangle_budget"]
        ranges["triangle_budget"] = (4, 300_000)
        integers.add("triangle_budget")
    for key, (low, high) in ranges.items():
        try:
            value = float(result[key])
        except (ValueError, TypeError) as exc:
            raise Codex3DError(f"Invalid AutoRemesher {key}.", code="REMESH_PARAMETERS") from exc
        if not math.isfinite(value) or not low <= value <= high:
            raise Codex3DError(
                f"AutoRemesher {key} must be between {low} and {high}.",
                code="REMESH_PARAMETERS",
            )
        if key in integers and not value.is_integer():
            raise Codex3DError(f"AutoRemesher {key} must be an integer.", code="REMESH_PARAMETERS")
        result[key] = int(value) if key in integers else value
    return result


def _memory_status() -> dict[str, int]:
    if os.name != "nt":
        raise Codex3DError(
            "This bounded AutoRemesher runtime requires Windows.", code="REMESH_PLATFORM"
        )

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (key, ctypes.c_ulonglong)
            for key in (
                "total_physical",
                "available_physical",
                "total_commit",
                "available_commit",
                "total_virtual",
                "available_virtual",
                "extended_virtual",
            )
        ]

    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise Codex3DError("Cannot read Windows memory pressure.", code="REMESH_RESOURCE_GUARD")
    return {
        key: int(getattr(status, key))
        for key in (
            "total_physical",
            "available_physical",
            "total_commit",
            "available_commit",
        )
    }


def _check_memory(params: dict[str, Any], *, starting: bool) -> dict[str, int]:
    memory = _memory_status()
    reserve = int(params["min_available_mb"]) * 1024**2
    required = reserve + (int(params["memory_limit_mb"]) * 1024**2 if starting else 0)
    if min(memory["available_physical"], memory["available_commit"]) < required:
        raise Codex3DError(
            f"3D processing stopped for memory pressure: needs {required // 1024**2} MiB "
            f"headroom; available RAM={memory['available_physical'] // 1024**2} MiB, "
            f"commit={memory['available_commit'] // 1024**2} MiB. "
            "Inspect memory usage and repair the cause before retrying.",
            code="REMESH_RESOURCE_GUARD",
        )
    return memory


class _WindowsLimits:
    """Attach limits while the process is suspended, before TBB initializes."""

    def __init__(self, process: subprocess.Popen, threads: int, memory_mb: int) -> None:
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel = kernel
        self.job = None
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.GetProcessAffinityMask.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
        kernel.GetPriorityClass.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.GetCurrentProcess.restype = wintypes.HANDLE

        class Basic(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_longlong),
                ("job_time", ctypes.c_longlong),
                ("flags", wintypes.DWORD),
                ("min_working", ctypes.c_size_t),
                ("max_working", ctypes.c_size_t),
                ("active_processes", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class IO(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "read_ops",
                    "write_ops",
                    "other_ops",
                    "read_bytes",
                    "write_bytes",
                    "other_bytes",
                )
            ]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("basic", Basic),
                ("io", IO),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process", ctypes.c_size_t),
                ("peak_job", ctypes.c_size_t),
            ]

        try:
            available, system = ctypes.c_size_t(), ctypes.c_size_t()
            if not kernel.GetProcessAffinityMask(
                kernel.GetCurrentProcess(), ctypes.byref(available), ctypes.byref(system)
            ):
                raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask")
            cpus = [bit for bit in range(64) if available.value & (1 << bit)]
            affinity = sum(1 << bit for bit in cpus[-threads:])
            self.job = kernel.CreateJobObjectW(None, None)
            if not self.job:
                raise OSError(ctypes.get_last_error(), "CreateJobObjectW")
            limits = Extended()
            # KILL_ON_JOB_CLOSE | PROCESS_MEMORY | AFFINITY | PRIORITY_CLASS.
            limits.basic.flags = 0x2000 | 0x100 | 0x10 | 0x20
            limits.basic.affinity = affinity
            limits.basic.priority = 0x4000  # BELOW_NORMAL_PRIORITY_CLASS
            limits.process_memory = int(memory_mb) * 1024**2
            handle = wintypes.HANDLE(int(process._handle))
            if not kernel.SetInformationJobObject(
                self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                raise OSError(ctypes.get_last_error(), "SetInformationJobObject")
            if not kernel.AssignProcessToJobObject(self.job, handle):
                raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")
            if not kernel.SetProcessAffinityMask(handle, affinity):
                raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask")
            if kernel.GetPriorityClass(handle) != 0x4000:
                raise OSError("CPU process below-normal priority could not be verified")
            actual, ignored = ctypes.c_size_t(), ctypes.c_size_t()
            if (
                not kernel.GetProcessAffinityMask(
                    handle, ctypes.byref(actual), ctypes.byref(ignored)
                )
                or actual.value != affinity
            ):
                raise OSError("AutoRemesher CPU affinity could not be verified")
            self.proof = {
                "affinity_mask": affinity,
                "logical_cpu_limit": len(cpus[-threads:]),
                "memory_limit_mb": memory_mb,
                "priority": "below_normal",
                "kill_tree_on_close": True,
            }
            ntdll = ctypes.WinDLL("ntdll")
            ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
            ntdll.NtResumeProcess.restype = ctypes.c_long
            if ntdll.NtResumeProcess(handle) != 0:
                raise OSError("NtResumeProcess failed")
        except Exception:
            process.kill()
            self.close()
            raise

    def close(self) -> None:
        if self.job:
            self.kernel.CloseHandle(self.job)
            self.job = None


class AutoRemesherRuntime:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._serial = threading.Lock()

    def preflight(self) -> dict[str, Any]:
        exe = Path(self.settings.autoremesher_exe)
        verified = exe.is_file() and hashlib.sha256(exe.read_bytes()).hexdigest() == EXE_SHA256
        return {
            "ready": os.name == "nt" and verified and self.settings.python_exe.is_file(),
            "executable": str(exe),
            "release": RELEASE,
            "source_commit": SOURCE_COMMIT,
            "archive_sha256": ARCHIVE_SHA256,
            "executable_verified": verified,
            "device": "cpu",
            "threads": getattr(self.settings, "remesh_threads", 2),
            "resource_limits": "verified affinity, below-normal priority, memory-limited job",
        }

    @staticmethod
    def _run(
        command: list[str], log: Path, cancel_event: Any, params: dict[str, Any], deadline: float
    ) -> dict[str, Any]:
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("CPU stage cancelled before process launch.")
        if time.monotonic() > deadline:
            raise Codex3DError(
                "CPU stage deadline elapsed before process launch.", code="REMESH_TIMEOUT"
            )
        memory_before = _check_memory(params, starting=True)
        env = os.environ.copy()
        env.update(
            QT_OPENGL="software",
            QT_QPA_PLATFORM="offscreen",
            OMP_NUM_THREADS=str(params["threads"]),
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS | 0x4
        with log.open("ab") as stream:
            stream.write(("\nCOMMAND " + json.dumps(command) + "\n").encode())
            stream.flush()
            native_log_start = (
                stream.tell() if Path(command[0]).name.lower() == "autoremesher.exe" else None
            )
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, env=env, creationflags=flags
            )
            limits = None
            try:
                limits = _WindowsLimits(process, params["threads"], params["memory_limit_mb"])
                while process.poll() is None:
                    if native_log_start is not None:
                        with log.open("rb") as native_log:
                            native_log.seek(native_log_start)
                            native_start = native_log.read(64 * 1024)
                        if b"Error: Failed to load " in native_start:
                            raise Codex3DError(
                                f"AutoRemesher failed to load its input. Inspect {log}; "
                                "the native process was stopped and attempt retained.",
                                code="REMESH_NATIVE_INPUT",
                            )
                    if cancel_event is not None and cancel_event.is_set():
                        raise GenerationCancelled("AutoRemesher cancelled; attempt retained.")
                    if time.monotonic() > deadline:
                        raise Codex3DError(
                            "AutoRemesher timed out; inspect the retained log and "
                            "revise mesh/settings before retrying.",
                            code="REMESH_TIMEOUT",
                        )
                    _check_memory(params, starting=False)
                    time.sleep(0.25)
                if process.returncode:
                    raise Codex3DError(
                        f"AutoRemesher stage exited {process.returncode}. Inspect {log}; "
                        "a memory cap, malformed mesh, or native failure may be responsible.",
                        code="REMESH_PROCESS_FAILED",
                    )
                return {**limits.proof, "memory_before": memory_before}
            except (Codex3DError, GenerationCancelled):
                raise
            except Exception as exc:
                raise Codex3DError(
                    f"Cannot enforce remesher process limits: {exc}", code="REMESH_RESOURCE_GUARD"
                ) from exc
            finally:
                if limits:
                    limits.close()
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)

    def _run_native(
        self, attempt: Path, log: Path, cancel_event: Any,
        params: dict[str, Any], deadline: float,
    ) -> dict[str, Any]:
        # The native Windows OBJ loader does not support long paths. Python
        # preparation/validation and all retained evidence stay in the attempt.
        temporary_root = Path(tempfile.gettempdir()).resolve()
        with tempfile.TemporaryDirectory(prefix="hy-", dir=temporary_root) as directory:
            scratch = Path(directory).resolve()
            paths = {name: scratch / name for name in (
                "normalized.obj", "quads.obj", "native-report.txt",
            )}
            if scratch.parent != temporary_root or not scratch.is_relative_to(temporary_root):
                raise Codex3DError(
                    "Native scratch escaped the temporary root.", code="REMESH_INPUT",
                )
            if any(path.resolve().parent != scratch or len(str(path.resolve())) >= 240
                   for path in paths.values()):
                raise Codex3DError(
                    f"Native scratch paths must be shorter than 240 characters: {scratch}",
                    code="REMESH_INPUT",
                )
            shutil.copyfile(attempt / "normalized.obj", paths["normalized.obj"])
            command = [
                str(self.settings.autoremesher_exe),
                "--input", str(paths["normalized.obj"]),
                "--output", str(paths["quads.obj"]),
                "--report", str(paths["native-report.txt"]),
            ]
            for name in (
                "target_quads", "edge_scaling", "sharp_edge", "smooth_normal",
                "adaptivity", "anisotropy",
            ):
                command.extend(["--" + name.replace("_", "-"), str(params[name])])
            try:
                # _run kills and joins its process in finally before returning
                # or propagating a native failure, cancellation, or timeout.
                return self._run(command, log, cancel_event, params, deadline)
            finally:
                # Preserve partial native evidence too, before scratch cleanup.
                for name in ("quads.obj", "native-report.txt"):
                    if paths[name].is_file():
                        shutil.copyfile(paths[name], attempt / name)

    def remesh(
        self,
        source: Path,
        output: Path,
        params: dict[str, Any],
        log_path: Path,
        cancel_event: Any = None,
    ) -> dict[str, Any]:
        if not self.preflight()["ready"]:
            raise Codex3DError(
                "AutoRemesher is unavailable. Run scripts/setup-autoremesher.ps1 "
                "and configure CODEX_AUTOREMESHER_EXE.",
                code="REMESH_UNAVAILABLE",
            )
        if not source.is_file():
            raise Codex3DError(f"Shape mesh does not exist: {source}", code="REMESH_INPUT")
        if source.resolve() == output.resolve():
            raise Codex3DError(
                "Remesher output must not overwrite the source mesh.", code="REMESH_INPUT"
            )
        options = _parameters(params, self.settings)
        with self._serial:
            attempt = output.parent / "remesh-attempts" / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            local_log = attempt / "process.log"
            request_path, result_path = attempt / "request.json", attempt / "result.json"
            request = {
                "source": str(source.resolve()),
                "attempt": str(attempt.resolve()),
                "params": options,
                "source_commit": SOURCE_COMMIT,
                "archive_sha256": ARCHIVE_SHA256,
            }
            request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
            deadline = time.monotonic() + self.settings.remesh_timeout_seconds
            helper = Path(__file__).with_name("remesh_worker.py")
            python = str(self.settings.python_exe)
            try:
                controls = [
                    self._run(
                        [python, str(helper), "prepare", str(request_path)],
                        local_log,
                        cancel_event,
                        options,
                        deadline,
                    )
                ]
                controls.append(
                    self._run_native(attempt, local_log, cancel_event, options, deadline)
                )
                controls.append(
                    self._run(
                        [python, str(helper), "validate", str(request_path)],
                        local_log,
                        cancel_event,
                        options,
                        deadline,
                    )
                )
                if not result_path.is_file():
                    raise Codex3DError(
                        "Remesh validator did not write its report.",
                        code="REMESH_VALIDATION_FAILED",
                    )
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["resource_controls"] = controls
                result["provenance"] = {
                    "release": RELEASE,
                    "source_commit": SOURCE_COMMIT,
                    "archive_sha256": ARCHIVE_SHA256,
                }
                if result["passed"]:
                    temporary = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
                    shutil.copyfile(result["artifacts"]["candidate_glb"], temporary)
                    os.replace(temporary, output)
                    result["artifacts"]["remeshed_glb"] = str(output)
                result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                return result
            finally:
                with log_path.open("ab") as aggregate:
                    aggregate.write((f"\nREMESH ATTEMPT {attempt}\n").encode())
                    if local_log.exists():
                        aggregate.write(local_log.read_bytes())


def run_cpu_process(
    command: list[str],
    log_path: Path,
    cancel_event: Any = None,
    *,
    timeout_seconds: int = 1800,
    threads: int = 2,
    memory_limit_mb: int = 4096,
    min_available_mb: int = 2048,
) -> dict[str, Any]:
    """Shared bounded runner for Blender exports or other isolated CPU-only stages."""
    settings = SimpleNamespace(remesh_target_quads=1000, remesh_threads=2)
    params = _parameters(
        {
            "threads": threads,
            "memory_limit_mb": memory_limit_mb,
            "min_available_mb": min_available_mb,
        },
        settings,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return AutoRemesherRuntime._run(
        command, log_path, cancel_event, params, time.monotonic() + timeout_seconds
    )
