"""Guarded, isolated CuMesh wheel build/probe; no base-environment pip mutation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["metadata", "build", "probe", "build-child", "probe-child"],
        required=True,
    )
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--qualification-build-memory-mb",
        type=int,
        choices=[2048, 2560, 3072, 3584, 4096],
        default=2048,
    )
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()
    from codex_3d_mcp.hunyuan_mv.cumesh_repair import SOURCE_COMMIT

    is_build = args.mode in {"build", "build-child"}
    if not is_build and args.qualification_build_memory_mb != 2048:
        raise ValueError("Build-only memory override cannot change a probe/runtime cap")
    memory_limit_mb = args.qualification_build_memory_mb if is_build else 2048
    packages = args.packages.resolve()
    # Refuse a pip --target destination in any existing Python environment.
    for item in [packages, *packages.parents]:
        if item == Path(sys.prefix).resolve() or (item / "pyvenv.cfg").exists():
            raise ValueError("Repair target must be outside all Python environments")
    source = args.source.resolve()
    evidence = args.evidence.resolve()
    if args.run_name:
        if not all(c.isalnum() or c in "-_" for c in args.run_name):
            raise ValueError("Run name must be an alphanumeric evidence-directory name")
        evidence = evidence / args.run_name
        if args.mode in {"build", "probe", "metadata"} and evidence.exists():
            raise ValueError("Evidence directory already exists; choose a unique run name")
    elif memory_limit_mb != 2048:
        raise ValueError("An increased qualification cap requires a unique evidence run name")
    evidence.mkdir(parents=True, exist_ok=True)
    if args.mode == "probe-child":
        sys.path.insert(1, str(packages))
        from codex_3d_mcp.hunyuan_mv.cumesh_repair import probe

        print(json.dumps(probe(evidence / "cuda-probe.json")))
        return 0
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError("CuMesh checkout does not match the pinned commit")
    submodules = {
        "third_party/cubvh": "ce92267a24ef6ad7d2c8ccbc2ae2c021a6597e70",
        "third_party/cubvh/third_party/eigen": "e63d9f6ccb7f6f29f31241b87c542f3f0ab3112b",
    }
    for directory, expected in submodules.items():
        actual = subprocess.check_output(
            ["git", "-C", str(source / directory), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != expected:
            raise ValueError(f"Pinned submodule mismatch: {directory}")
    tracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).split(b"\0")
    hashes = {
        p.decode(): digest(source / p.decode())
        for p in tracked
        if p and (source / p.decode()).is_file()
    }
    metadata = {
        "source_commit": commit,
        "source_url": "https://github.com/JeffreyXiang/CuMesh",
        "license": "MIT",
        "source_file_sha256": hashes,
        "submodules": submodules,
        "python": sys.version,
        "torch_metadata": importlib.metadata.version("torch"),
        "cuda_toolkit": os.getenv("CUDA_HOME"),
        "architecture": os.getenv("TORCH_CUDA_ARCH_LIST"),
        "max_jobs": 1,
        "cpu_limit": 2,
        "memory_limit_mb": memory_limit_mb,
        "production_runtime_memory_limit_mb": 2048,
        "build_cap_override_authorized": memory_limit_mb > 2048,
        "min_available_mb": 2048,
        "package_target": str(packages),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_allows_unsupported_compiler": True,
        "cuda_verified": False,
        "production_qualified": False,
    }
    (evidence / "source-build-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if args.mode == "metadata":
        print(json.dumps({k: v for k, v in metadata.items() if k != "source_file_sha256"}))
        return 0
    if args.mode == "build-child":
        wheels = evidence / "wheels"
        wheels.mkdir(exist_ok=True)
        # nvcc's generated MSVC command did not quote a spaced TEMP path in the
        # first retained Windows attempt. Keep only temporary compiler files here.
        build_temp = Path(os.environ["LOCALAPPDATA"]) / "Temp" / ("codex-cumesh-" + commit[:12])
        if any(c.isspace() for c in str(build_temp)):
            raise ValueError("nvcc requires a space-free temporary directory on this host")
        build_temp.mkdir(parents=True, exist_ok=True)
        os.environ["TEMP"] = os.environ["TMP"] = str(build_temp)
        metadata["compiler_temp"] = str(build_temp)
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--no-cache-dir",
                "--wheel-dir",
                str(wheels),
                str(source),
            ],
            check=True,
        )
        wheel = sorted(wheels.glob("cumesh-*.whl"), key=lambda p: p.stat().st_mtime)[-1]
        if (packages / "cumesh").exists():
            raise ValueError("Repair package already exists; use a fresh isolated target")
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--target",
                str(packages),
                str(wheel),
            ],
            check=True,
        )
        metadata.update(
            wheel=str(wheel),
            wheel_sha256=digest(wheel),
            build_succeeded=True,
            native_file_sha256={
                str(p.relative_to(packages)): digest(p) for p in (packages / "cumesh").glob("*.pyd")
            },
        )
        (packages / "cumesh-install-manifest.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return 0
    from codex_3d_mcp.hunyuan_mv import cumesh_repair
    from codex_3d_mcp.hunyuan_mv.remesh import _check_memory, run_cpu_process
    from codex_3d_mcp.trellis.runtime import GpuLease

    adapter_path = Path(cumesh_repair.__file__).resolve()
    adapter_sha256 = digest(adapter_path)
    lease = GpuLease(args.gpu_lock)
    lease.acquire(timeout=1)
    started = time.monotonic()
    result = {
        "mode": args.mode,
        "succeeded": False,
        "source_commit": commit,
        "memory_limit_mb": memory_limit_mb,
        "resource_limits_raised": memory_limit_mb > 2048,
        "user_authorized_build_cap_override": memory_limit_mb > 2048,
        "production_runtime_limits_changed": False,
        "logical_cpu_limit": 2,
        "compiler_jobs": 1,
        "min_available_mb": 2048,
        "priority": "below_normal",
        "adapter_path": str(adapter_path),
        "adapter_sha256": adapter_sha256,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        result["memory_before"] = _check_memory(
            {"memory_limit_mb": memory_limit_mb, "min_available_mb": 2048}, starting=True
        )
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--mode",
            args.mode + "-child",
            "--packages",
            str(packages),
            "--gpu-lock",
            str(args.gpu_lock),
            "--source",
            str(source),
            "--evidence",
            str(args.evidence.resolve()),
            "--qualification-build-memory-mb",
            str(memory_limit_mb),
        ]
        if args.run_name:
            command.extend(["--run-name", args.run_name])
        result["command"] = command
        result["resource_controls"] = run_cpu_process(
            command,
            evidence / (args.mode + ".log"),
            timeout_seconds=900,
            threads=2,
            memory_limit_mb=memory_limit_mb,
            min_available_mb=2048,
        )
        result["succeeded"] = True
        if args.mode == "probe":
            if digest(adapter_path) != adapter_sha256:
                raise ValueError("Adapter changed while qualification probe was running")
            controls = result["resource_controls"]
            if controls["memory_limit_mb"] != 2048 or controls["logical_cpu_limit"] != 2:
                raise ValueError("Probe did not use the required production resource cap")
            manifest_path = packages / "cumesh-install-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            probe_path = evidence / "cuda-probe.json"
            probe_report = json.loads(probe_path.read_text(encoding="utf-8"))
            if probe_report.get("cuda_verified") is not True:
                raise ValueError("Probe did not complete empirical CUDA checks")
            probe_report["qualification_binding"] = {
                "source_commit": commit,
                "native_file_sha256": manifest["native_file_sha256"],
                "adapter_sha256": adapter_sha256,
                "memory_limit_mb": 2048,
                "logical_cpu_limit": 2,
            }
            probe_path.write_text(json.dumps(probe_report, indent=2), encoding="utf-8")
            result["cuda_probe_sha256"] = digest(probe_path)
    except Exception as exc:
        result["succeeded"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        lease.release()
        result["elapsed_seconds"] = time.monotonic() - started
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (evidence / (args.mode + "-result.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    if args.mode == "probe" and result["succeeded"]:
        # Result is finalized first so the install proof can bind both artifacts.
        proof_dir = packages / "cumesh-qualification" / (args.run_name or "probe-2048")
        proof_dir.mkdir(parents=True, exist_ok=False)
        portable_probe = proof_dir / "cuda-probe.json"
        portable_result = proof_dir / "probe-result.json"
        portable_adapter = proof_dir / "adapter-source.py"
        shutil.copyfile(probe_path, portable_probe)
        shutil.copyfile(evidence / "probe-result.json", portable_result)
        shutil.copyfile(adapter_path, portable_adapter)
        manifest.update(
            cuda_verified=True,
            cuda_probe_path=portable_probe.relative_to(packages).as_posix(),
            cuda_probe_sha256=digest(portable_probe),
            cuda_probe_result_path=portable_result.relative_to(packages).as_posix(),
            cuda_probe_result_sha256=digest(portable_result),
            cuda_probe_adapter_path=portable_adapter.relative_to(packages).as_posix(),
            cuda_probe_adapter_sha256=adapter_sha256,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
