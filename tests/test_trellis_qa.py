from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from codex_3d_mcp.trellis.config import TrellisSettings
from codex_3d_mcp.trellis.qa import QualityAssurance


def test_validator_uses_positional_input_and_stdout(tmp_path: Path, monkeypatch) -> None:
    runtime = tmp_path / "runtime"
    validator = runtime / "gltf_validator.exe"
    validator.parent.mkdir(parents=True)
    validator.write_bytes(b"validator")
    settings = TrellisSettings(
        base_dir=tmp_path,
        server_name="test",
        server_version="test",
        output_dir=tmp_path,
        database_path=tmp_path / "jobs.sqlite3",
        runtime_dir=runtime,
        model_dir=runtime,
        trellis_server=runtime / "server.exe",
        trellis_cli=runtime / "cli.exe",
        realesrgan_exe=runtime / "real.exe",
        blender_exe=runtime / "blender.exe",
        gltf_validator=validator,
        allowed_input_roots=(tmp_path,),
        gpu_lock_path=runtime / "gpu.lock",
    )
    glb = tmp_path / "asset.glb"
    glb.write_bytes(b"glTF")
    called = {}

    def fake_run(command, **_kwargs):
        called["command"] = command
        payload = {"issues": {"numErrors": 0, "numWarnings": 2}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("codex_3d_mcp.trellis.qa.subprocess.run", fake_run)
    report_path = tmp_path / "report.json"
    result = QualityAssurance(settings)._validate_glb(glb, report_path)

    assert called["command"] == [str(validator), "--stdout", str(glb)]
    assert result["errors"] == 0
    assert result["warnings"] == 2
    assert report_path.is_file()
