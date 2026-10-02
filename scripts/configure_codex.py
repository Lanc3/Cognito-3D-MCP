"""Install bundled skills and append missing Codex MCP entries without replacing user settings.

Requires Python 3.11 for TOML validation. No model or runtime installation happens here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomllib


def quote(value: str | Path) -> str:
    """JSON string quoting is compatible with TOML basic strings for these path values."""
    return json.dumps(str(value), ensure_ascii=True)


def server_entries(settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    data = Path(settings["data_root"])
    runtime = Path(settings["runtime_root"])
    models = Path(settings["model_root"])
    outputs = Path(settings["output_root"])
    backends = settings["backends"]
    input_roots = os.pathsep.join(
        (str(data / "inputs"), str(Path.home() / ".codex" / "generated_images"))
    )
    shared = {
        "CODEX_3D_DATA_ROOT": str(data),
        "CODEX_BLENDER_EXE": settings["blender"],
        "CODEX_GLTF_VALIDATOR": str(runtime / "gltf-validator" / "gltf_validator.exe"),
        "CODEX_3D_GPU_LOCK": str(runtime / "gpu.lock"),
    }
    if settings.get("hf_token_path"):
        shared["HF_TOKEN_PATH"] = settings["hf_token_path"]
    entries: dict[str, dict[str, Any]] = {}

    def add(name: str, python: Path, module: str, env: dict[str, str]) -> None:
        entries[name] = {
            "command": str(python),
            "args": ["-m", module],
            "cwd": str(data),
            "startup_timeout_sec": 120,
            "tool_timeout_sec": 300,
            "env": {**shared, **env},
        }

    if "Hunyuan" in backends:
        base = runtime / "hunyuan3d-2mv"
        cache = models / "hunyuan3d-2mv"
        add(
            "codex_3d_models_hunyuan_mv",
            base / "venv" / "Scripts" / "python.exe",
            "codex_3d_mcp.hunyuan_mv.server",
            {
                "CODEX_HUNYUAN_MV_RUNTIME_ROOT": str(base),
                "CODEX_HUNYUAN_MV_PYTHON": str(base / "venv" / "Scripts" / "python.exe"),
                "CODEX_HUNYUAN_MV_UPSTREAM": str(base / "Hunyuan3D-2"),
                "CODEX_HUNYUAN_MV_MODEL_CACHE": str(cache),
                "CODEX_HUNYUAN_MV_OUTPUT_DIR": str(outputs / "hunyuan-mv"),
                "CODEX_HUNYUAN_MV_ALLOWED_INPUT_ROOTS": input_roots,
                "CODEX_AUTOREMESHER_EXE": str(runtime / "autoremesher" / "autoremesher.exe"),
                "CODEX_HUNYUAN_MV_REPAIR_PACKAGES": str(runtime / "shape-repair-packages"),
                "HF_HOME": str(cache),
                "HUGGINGFACE_HUB_CACHE": str(cache / "hub"),
                "TORCH_HOME": str(cache / "torch"),
                "TEMP": str(base / "temp"),
                "TMP": str(base / "temp"),
            },
        )
    for backend in ("SF3D", "SPAR3D"):
        if backend not in backends:
            continue
        suffix = backend.lower()
        vendor = "stable-fast-3d" if backend == "SF3D" else "stable-point-aware-3d"
        # Keep the original SF3D tool namespace; give SPAR3D its established comparison name.
        name = "codex_3d_models" if backend == "SF3D" else "codex_3d_models_spar3d"
        add(
            name,
            runtime / suffix / "venv" / "Scripts" / "python.exe",
            "codex_3d_mcp.server",
            {
                "CODEX_3D_SERVER_NAME": name,
                "CODEX_3D_BACKEND": suffix,
                "CODEX_3D_MODEL_CACHE_DIR": str(models / suffix),
                "CODEX_3D_OUTPUT_DIR": str(outputs / suffix),
                "CODEX_3D_ALLOWED_INPUT_ROOTS": input_roots,
                "PYTHONPATH": str(runtime / suffix / vendor),
                "HF_HOME": str(models / suffix),
                "HUGGINGFACE_HUB_CACHE": str(models / suffix / "hub"),
                **(
                    {"HF_TOKEN_PATH": settings["hf_token_paths"][suffix]}
                    if settings.get("hf_token_paths", {}).get(suffix)
                    else {}
                ),
            },
        )
    if "TRELLIS" in backends:
        # Installing TRELLIS alone must use its own backend instead of requiring SPAR3D.
        add(
            "codex_3d_models_trellis",
            runtime / "trellis-venv" / "Scripts" / "python.exe",
            "codex_3d_mcp.trellis.server",
            {
                "CODEX_TRELLIS_RUNTIME_DIR": str(runtime / "trellis-v0.5.4"),
                "CODEX_TRELLIS_MODEL_DIR": str(models / "trellis2-q8"),
                "CODEX_TRELLIS_OUTPUT_DIR": str(outputs / "trellis"),
                "CODEX_TRELLIS_ALLOWED_INPUT_ROOTS": input_roots,
                "CODEX_BIDIRECTIONAL_RECONSTRUCTION_BACKEND": "spar3d"
                if "SPAR3D" in backends
                else "trellis",
                "CODEX_BIDIRECTIONAL_SPAR3D_PYTHON": str(
                    runtime / "spar3d" / "venv" / "Scripts" / "python.exe"
                ),
                "CODEX_BIDIRECTIONAL_SPAR3D_MODEL_CACHE": str(models / "spar3d"),
                "CODEX_REALESRGAN_EXE": str(
                    runtime / "realesrgan-v0.2.5.0" / "realesrgan-ncnn-vulkan.exe"
                ),
                "HF_HOME": str(models / "trellis2-q8" / "dino-cache"),
                **(
                    {"HF_TOKEN_PATH": settings["hf_token_paths"]["spar3d"]}
                    if settings.get("hf_token_paths", {}).get("spar3d")
                    else {}
                ),
                "TEMP": str(runtime / "temp"),
                "TMP": str(runtime / "temp"),
            },
        )
    return entries


def render_entries(entries: dict[str, dict[str, Any]]) -> str:
    chunks = []
    for name, entry in entries.items():
        lines = [f"[mcp_servers.{quote(name)}]"]
        for key in ("command", "cwd"):
            lines.append(f"{key} = {quote(entry[key])}")
        lines.append("args = [" + ", ".join(quote(arg) for arg in entry["args"]) + "]")
        for key in ("startup_timeout_sec", "tool_timeout_sec"):
            lines.append(f"{key} = {entry[key]}")
        lines.append(f"[mcp_servers.{quote(name)}.env]")
        lines.extend(f"{key} = {quote(value)}" for key, value in entry["env"].items())
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) + "\n"


def register(config: Path, entries: dict[str, dict[str, Any]]) -> list[str]:
    original = config.read_bytes() if config.exists() else b""
    text = original.decode("utf-8-sig")
    parsed = tomllib.loads(text)
    existing = parsed.get("mcp_servers", {})
    if not isinstance(existing, dict):
        raise ValueError("Existing mcp_servers configuration is not a TOML table.")
    missing = {}
    for name, entry in entries.items():
        aliases = (name, name.replace("_", "-"))
        conflicts = [alias for alias in aliases if alias in existing]
        if conflicts:
            print(
                "Preserved existing MCP entry: "
                + ", ".join(conflicts)
                + ". Review cognito-mcp-config.toml to update it manually."
            )
        else:
            missing[name] = entry
    if not missing:
        return []
    combined = (
        text.rstrip()
        + "\n\n# Cognito-3D-mcp (installed without replacing existing entries)\n"
        + render_entries(missing)
    )
    tomllib.loads(combined)
    config.parent.mkdir(parents=True, exist_ok=True)
    if original:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = config.with_name(config.name + f".cognito-backup-{stamp}")
        backup.write_bytes(original)
    # Check for edits made by another program before committing this replacement.
    current = config.read_bytes() if config.exists() else b""
    if current != original:
        raise RuntimeError(
            "Codex config changed during installation; no changes were applied. Retry."
        )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=config.parent, delete=False
    ) as file:
        temporary = Path(file.name)
        file.write(combined)
    try:
        os.replace(temporary, config)
    finally:
        temporary.unlink(missing_ok=True)
    return list(missing)


def install_skills(source: Path, destination: Path, backends: list[str]) -> list[str]:
    wanted = set()
    if "Hunyuan" in backends:
        wanted.add("codex-hunyuan-multiview-3d")
    if "TRELLIS" in backends:
        wanted.add("codex-bidirectional-3d")
    if "SF3D" in backends or "SPAR3D" in backends:
        wanted.add("codex-3d-models")
    installed = []
    for name in sorted(wanted):
        skill = source / name
        if not (skill / "SKILL.md").is_file():
            raise FileNotFoundError(f"Bundled skill is missing: {skill}")
        target = destination / name
        if target.exists():
            print(f"Preserved existing skill: {target}. Updated instructions remain in {skill}.")
            continue
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(skill, target)
        installed.append(name)
    return installed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--no-skills", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8-sig"))
    entries = server_entries(settings)
    data = Path(settings["data_root"])
    snippet = data / "cognito-mcp-config.toml"
    snippet.write_text(render_entries(entries), encoding="utf-8")
    # Validate the generated config, including Windows backslashes and unusual path characters.
    tomllib.loads(snippet.read_text(encoding="utf-8"))
    if not args.no_skills:
        source = Path(settings["repository"]) / "skills"
        if not source.is_dir():
            source = Path(settings["repository"]) / "docs" / "skills"
        installed = install_skills(source, Path(settings["skills_root"]), settings["backends"])
        print("Installed skills: " + (", ".join(installed) or "none; existing skills preserved"))
    if not args.no_register:
        added = register(Path(settings["codex_config"]), entries)
        print("Registered MCP servers: " + (", ".join(added) or "none; existing entries preserved"))
    print(f"Complete MCP config for other clients or manual merging: {snippet}")


if __name__ == "__main__":
    main()
