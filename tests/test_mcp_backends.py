from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from hy3dgen_mcp.backends import (
    DEFAULT_BACKEND,
    BackendRegistry,
    GenerationRequest,
    backend_status,
    validate_request,
)
from hy3dgen_mcp.config import Settings


def settings(root: Path) -> Settings:
    return Settings(
        output_root=root / "outputs",
        hunyuan_model="tencent/Hunyuan3D-2mv",
        hunyuan_subfolder="hunyuan3d-dit-v2-mv",
        hunyuan_device="cuda",
        hunyuan_variant="fp16",
        sf3d_root=None,
        sf3d_python=Path("missing-python"),
        spar3d_root=None,
        spar3d_python=Path("missing-python"),
        job_timeout_seconds=60,
    )


class BackendTests(unittest.TestCase):
    def test_hunyuan_is_the_default_and_preferred_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configured = settings(Path(directory))
            self.assertEqual(DEFAULT_BACKEND, "hunyuan3d")
            statuses = backend_status(configured)
            self.assertEqual(statuses[0]["name"], "hunyuan3d")
            self.assertTrue(statuses[0]["preferred"])

    def test_missing_image_is_rejected_before_model_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            validate_request(GenerationRequest(front_image=Path("missing.png")))

    def test_external_backend_requires_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            image.write_bytes(b"not decoded by this test")
            registry = BackendRegistry(settings(root))
            with self.assertRaisesRegex(RuntimeError, "SF3D_ROOT"):
                registry.generate(GenerationRequest(front_image=image, backend="sf3d"))

    def test_single_image_backends_reject_extra_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            image.write_bytes(b"not decoded by this test")
            configured = settings(root)
            fake_install = root / "sf3d"
            fake_install.mkdir()
            (fake_install / "run.py").write_text("", encoding="utf-8")
            configured = replace(configured, sf3d_root=fake_install, sf3d_python=image)
            registry = BackendRegistry(configured)
            with self.assertRaisesRegex(ValueError, "single front image"):
                registry.generate(
                    GenerationRequest(
                        front_image=image,
                        left_image=image,
                        backend="sf3d",
                    )
                )


if __name__ == "__main__":
    unittest.main()
