"""Hard production gates and auditable QA reports."""

from __future__ import annotations

import html
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from .artifacts import artifact_record, atomic_json
from .config import TrellisSettings


class QualityAssurance:
    def __init__(self, settings: TrellisSettings) -> None:
        self.settings = settings

    def validate(self, job_dir: Path, blender_metrics: dict[str, Any]) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        targets = {
            "master": self.settings.master_faces,
            "game": self.settings.game_faces,
            "lod1": self.settings.lod_faces[0],
            "lod2": self.settings.lod_faces[1],
        }
        geometry = blender_metrics.get("geometry", {})
        for name, target in targets.items():
            metrics = geometry.get(name, {})
            faces = int(metrics.get("faces", 0))
            checks.append(
                _check(
                    f"{name}_face_budget",
                    target * 0.95 <= faces <= target * 1.05,
                    {"actual": faces, "target": target, "tolerance": "5%"},
                )
            )
            checks.append(
                _check(
                    f"{name}_degenerate_faces",
                    int(metrics.get("degenerate_faces", 1)) <= max(0, int(faces * 0.0001)),
                    metrics.get("degenerate_faces"),
                )
            )
        master_metrics = geometry.get("master", {})
        checks.append(
            _check("master_watertight", bool(master_metrics.get("watertight")), master_metrics)
        )
        uv_metrics = blender_metrics.get("uv", {})
        uv_occupancy = float(uv_metrics.get("occupancy", 0.0))
        checks.append(
            _check(
                "uv_occupancy",
                uv_occupancy >= 0.65,
                {"actual": uv_occupancy, "minimum": 0.65},
            )
        )

        texture_names = ["baseColor.png", "normal.png", "orm.png"]
        texture_dir = job_dir / "textures"
        for name in texture_names:
            path = texture_dir / name
            dimensions = None
            if path.is_file():
                with Image.open(path) as image:
                    dimensions = image.size
            checks.append(
                _check(
                    f"texture_{name}",
                    dimensions
                    == (self.settings.texture_resolution, self.settings.texture_resolution),
                    dimensions,
                )
            )
        jpeg_files = list(texture_dir.glob("*.jpg")) + list(texture_dir.glob("*.jpeg"))
        checks.append(
            _check("no_jpeg_textures", not jpeg_files, [str(path) for path in jpeg_files])
        )

        validator_reports = {}
        artifacts = []
        for name in targets:
            path = job_dir / f"{name}.glb"
            exists = path.is_file() and path.stat().st_size > 20
            checks.append(_check(f"{name}_glb_exists", exists, str(path)))
            if exists:
                artifacts.append(artifact_record(path, job_dir))
                report = self._validate_glb(path, job_dir / "qa" / f"{name}-gltf.json")
                validator_reports[name] = report
                checks.append(
                    _check(
                        f"{name}_gltf_validator",
                        report.get("errors") == 0,
                        report,
                    )
                )

        report = {
            "passed": all(check["passed"] for check in checks),
            "checks": checks,
            "geometry": geometry,
            "alignment": blender_metrics.get("alignment", {}),
            "gltf_validator": validator_reports,
            "artifacts": artifacts,
        }
        qa_dir = job_dir / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(qa_dir / "report.json", report)
        (qa_dir / "report.html").write_text(_html_report(report), encoding="utf-8")
        return report

    def _validate_glb(self, path: Path, report_path: Path) -> dict[str, Any]:
        if self.settings.test_mode:
            return {"errors": 0, "warnings": 0, "test_mode": True}
        if not self.settings.gltf_validator.is_file():
            return {"errors": 1, "warnings": 0, "message": "glTF Validator is missing"}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.settings.gltf_validator),
            "--stdout",
            str(path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            check=False,
        )
        try:
            data = json.loads(result.stdout)
            atomic_json(report_path, data)
            issues = data.get("issues", {})
            return {
                "errors": int(issues.get("numErrors", result.returncode != 0)),
                "warnings": int(issues.get("numWarnings", 0)),
                "report_path": str(report_path.resolve()),
            }
        except json.JSONDecodeError:
            pass
        return {
            "errors": int(result.returncode != 0),
            "warnings": 0,
            "message": (result.stderr or result.stdout)[-2000:],
        }


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _html_report(report: dict[str, Any]) -> str:
    rows = "\n".join(
        "<tr><td>{}</td><td>{}</td><td><pre>{}</pre></td></tr>".format(
            html.escape(check["name"]),
            "PASS" if check["passed"] else "FAIL",
            html.escape(json.dumps(check["evidence"], indent=2, default=str)),
        )
        for check in report["checks"]
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>TRELLIS QA</title>"
        "<style>body{font-family:system-ui;background:#111;color:#eee;padding:2rem}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #555;padding:.5rem}"
        "pre{white-space:pre-wrap}</style></head><body>"
        f"<h1>Production QA: {'PASS' if report['passed'] else 'FAIL'}</h1>"
        f"<table><tr><th>Check</th><th>Result</th><th>Evidence</th></tr>{rows}</table>"
        "</body></html>"
    )
