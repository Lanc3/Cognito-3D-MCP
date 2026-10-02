"""Offline coverage for installer config preservation and portable client paths."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import tomllib

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "configure_codex.py"
_SPEC = importlib.util.spec_from_file_location("configure_codex", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
installer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(installer)


def settings(tmp_path: Path, backends: list[str]) -> dict:
    return {
        "data_root": str(tmp_path / "User's data Ã¼"),
        "runtime_root": str(tmp_path / "GPU runtime"),
        "model_root": str(tmp_path / "Models"),
        "output_root": str(tmp_path / "Output"),
        "blender": str(tmp_path / "Blender" / "blender.exe"),
        "backends": backends,
    }


def test_append_preserves_settings_and_creates_exact_backup(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    original = b'# keep this comment\r\nmodel = "user-choice"\r\n'
    config.write_bytes(original)
    entries = installer.server_entries(settings(tmp_path, ["Hunyuan"]))
    assert installer.register(config, entries) == ["codex_3d_models_hunyuan_mv"]
    parsed = tomllib.loads(config.read_text())
    assert parsed["model"] == "user-choice"
    assert (
        parsed["mcp_servers"]["codex_3d_models_hunyuan_mv"] == entries["codex_3d_models_hunyuan_mv"]
    )
    assert next(tmp_path.glob("config.toml.cognito-backup-*")).read_bytes() == original


def test_existing_server_is_never_replaced_and_reruns_are_noop(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    original = b'[mcp_servers."codex_3d_models_hunyuan_mv"]\ncommand = "custom-server"\n'
    config.write_bytes(original)
    entries = installer.server_entries(settings(tmp_path, ["Hunyuan"]))
    assert installer.register(config, entries) == []
    assert config.read_bytes() == original
    assert not list(tmp_path.glob("*.cognito-backup-*"))


def test_invalid_existing_config_is_left_untouched(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    original = b"[invalid TOML"
    config.write_bytes(original)
    with pytest.raises(tomllib.TOMLDecodeError):
        installer.register(config, installer.server_entries(settings(tmp_path, ["Hunyuan"])))
    assert config.read_bytes() == original


def test_hyphen_alias_prevents_duplicate_server_registration(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    original = b'[mcp_servers."codex-3d-models-hunyuan-mv"]\ncommand = "legacy-server"\n'
    config.write_bytes(original)
    entries = installer.server_entries(settings(tmp_path, ["Hunyuan"]))
    assert installer.register(config, entries) == []
    assert config.read_bytes() == original


def test_selected_credential_paths_are_registered_without_token_values(tmp_path: Path) -> None:
    chosen = settings(tmp_path, ["SPAR3D", "TRELLIS"])
    chosen["hf_token_path"] = str(tmp_path / "shared" / "token")
    chosen["hf_token_paths"] = {"spar3d": str(tmp_path / "models" / "spar3d" / "token")}
    entries = installer.server_entries(chosen)
    for name in ("codex_3d_models_spar3d", "codex_3d_models_trellis"):
        assert entries[name]["env"]["HF_TOKEN_PATH"] == chosen["hf_token_paths"]["spar3d"]
        assert "HF_TOKEN" not in entries[name]["env"]
    assert entries["codex_3d_models_spar3d"]["env"]["HUGGINGFACE_HUB_CACHE"] == str(
        Path(chosen["model_root"]) / "spar3d" / "hub"
    )


def test_all_backends_have_distinct_compatible_ids_and_selected_paths(tmp_path: Path) -> None:
    chosen = settings(tmp_path, ["Hunyuan", "SF3D", "SPAR3D", "TRELLIS"])
    entries = installer.server_entries(chosen)
    rendered = installer.render_entries(entries)
    assert tomllib.loads(rendered)["mcp_servers"] == entries
    assert set(entries) == {
        "codex_3d_models_hunyuan_mv",
        "codex_3d_models",
        "codex_3d_models_spar3d",
        "codex_3d_models_trellis",
    }
    for entry in entries.values():
        assert str(tmp_path) in entry["command"]
        assert entry["cwd"] == chosen["data_root"]
    assert (
        entries["codex_3d_models_trellis"]["env"]["CODEX_BIDIRECTIONAL_RECONSTRUCTION_BACKEND"]
        == "spar3d"
    )


def test_trellis_alone_uses_trellis_reconstruction(tmp_path: Path) -> None:
    entries = installer.server_entries(settings(tmp_path, ["TRELLIS"]))
    assert (
        entries["codex_3d_models_trellis"]["env"]["CODEX_BIDIRECTIONAL_RECONSTRUCTION_BACKEND"]
        == "trellis"
    )


def test_windows_paths_and_quotes_round_trip() -> None:
    path = 'C:\\Users\\O\'Brien\\A "quoted" path'
    assert tomllib.loads("path = " + installer.quote(path))["path"] == path
    assert json.loads(installer.quote(path)) == path


def test_skill_install_copies_references_and_preserves_existing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    skill = source / "codex-hunyuan-multiview-3d"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text("instructions")
    (skill / "references" / "setup.md").write_text("reference")
    destination = tmp_path / "installed"
    assert installer.install_skills(source, destination, ["Hunyuan"]) == [skill.name]
    target = destination / skill.name
    assert (target / "references" / "setup.md").read_text() == "reference"
    (target / "SKILL.md").write_text("user customization")
    assert installer.install_skills(source, destination, ["Hunyuan"]) == []
    assert (target / "SKILL.md").read_text() == "user customization"


@pytest.mark.parametrize(
    "backends, expected",
    [
        (["SF3D"], {"codex-3d-models"}),
        (["SPAR3D"], {"codex-3d-models"}),
        (
            ["Hunyuan", "SF3D", "SPAR3D", "TRELLIS"],
            {"codex-3d-models", "codex-hunyuan-multiview-3d", "codex-bidirectional-3d"},
        ),
    ],
)
def test_each_backend_installs_the_actual_bundled_skill_directories(
    tmp_path: Path, backends: list[str], expected: set[str]
) -> None:
    source = Path(__file__).resolve().parents[1] / "skills"
    destination = tmp_path / "installed"
    assert set(installer.install_skills(source, destination, backends)) == expected
    assert {path.name for path in destination.iterdir()} == expected
    assert all((destination / name / "SKILL.md").is_file() for name in expected)
