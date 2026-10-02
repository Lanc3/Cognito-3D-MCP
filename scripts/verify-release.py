"""Check the source release and built wheel without loading a GPU runtime."""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = (
    "codex-hunyuan-multiview-3d",
    "codex-3d-models",
    "codex-bidirectional-3d",
)


def verify_source() -> list[str]:
    errors: list[str] = []
    required = (
        "LICENSE", "README.md", "NOTICE", "THIRD_PARTY_NOTICES.md",
        "SECURITY.md", "CONTRIBUTING.md", "pyproject.toml", "Install.cmd",
        "install.ps1", "scripts/configure_codex.py", "scripts/run-hunyuan-mv-server.ps1",
        "src/codex_3d_mcp/hunyuan_mv/web/vendor/LICENSE.txt",
        "requirements-repair.txt", "docs/shape-repair.md", "docs/cumesh-repair.md",
    )
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"Missing release file: {relative}")
    source = ROOT / "src"
    for path in source.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"Invalid Python: {path.relative_to(ROOT)}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in {"create_sprite_animation", "replace_sprite_art", "get_sprite_project", "open_sprite_editor"}:
                    errors.append(f"2D sprite tool remains: {path.relative_to(ROOT)}:{node.name}")
    for name in SKILLS:
        entry = ROOT / "skills" / name / "SKILL.md"
        if not entry.is_file():
            errors.append(f"Missing skill: {name}")
            continue
        content = entry.read_text(encoding="utf-8")
        if not content.startswith("---\n") or f"name: {name}\n" not in content:
            errors.append(f"Invalid skill frontmatter: {name}")
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if "://" not in target and not target.startswith("#"):
                if not (entry.parent / target.split("#", 1)[0]).is_file():
                    errors.append(f"Missing skill reference: {name}/{target}")
    return errors


def verify_wheels(directory: Path) -> list[str]:
    errors: list[str] = []
    wheels = sorted(directory.glob("*.whl"))
    if not wheels:
        return [f"No wheel found in {directory}"]
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            files = set(archive.namelist())
            for source in (ROOT / "src/codex_3d_mcp/hunyuan_mv/web").rglob("*"):
                if source.is_file():
                    package_path = source.relative_to(ROOT / "src").as_posix()
                    if package_path not in files:
                        errors.append(f"{wheel.name}: missing browser asset {package_path}")
            metadata_name = next((n for n in files if n.endswith(".dist-info/METADATA")), None)
            if metadata_name is None:
                errors.append(f"{wheel.name}: missing package metadata")
            elif "Name: cognito-3d-mcp" not in archive.read(metadata_name).decode():
                errors.append(f"{wheel.name}: distribution has the wrong brand name")
            if not any(n.endswith("/licenses/THIRD_PARTY_NOTICES.md") for n in files):
                errors.append(f"{wheel.name}: missing third-party licensing notices")
    return errors


def verify_archive(path: Path) -> list[str]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path) as archive:
            names = [member.name for member in archive.getmembers() if member.isfile()]
    # Source archives have one project-directory prefix.
    relative = {name.split("/", 1)[1] for name in names if "/" in name}
    errors: list[str] = []
    for required in (
        "Install.cmd", "install.ps1", "Run.cmd", "LICENSE", "README.md",
        "SECURITY.md", "CONTRIBUTING.md", "requirements-repair.txt",
        "assets/branding/cognito-3d-mcp-brand.png", "tests/dashboard-ui-smoke.cjs",
        "scripts/configure_codex.py", "scripts/launch-server.py",
        *(f"skills/{skill}/SKILL.md" for skill in SKILLS),
        "skills/codex-bidirectional-3d/references/quality-contract.md",
    ):
        if required not in relative:
            errors.append(f"{path.name}: missing source-release file {required}")
    forbidden = {
        ".git", ".local-archive", ".test-deps", ".test-env", ".tools", ".cache",
        ".vendor", ".venv", "node_modules", "__pycache__", ".env", "outputs",
    }
    for name in relative:
        if forbidden & set(name.split("/")) or name.endswith((".ckpt", ".safetensors", ".sqlite3")):
            errors.append(f"{path.name}: private/generated/runtime file included: {name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    errors = verify_source()
    if args.wheel_dir:
        errors.extend(verify_wheels(args.wheel_dir))
        for sdist in args.wheel_dir.glob("*.tar.gz"):
            errors.extend(verify_archive(sdist))
    if args.source_archive:
        errors.extend(verify_archive(args.source_archive))
    for message in errors:
        print(message, file=sys.stderr)
    if errors:
        return 1
    print("Cognito-3D-mcp release files, 3D tool scope, skills, and packaged assets verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
