import base64
import io
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from codex_3d_mcp.config import Settings
from codex_3d_mcp.errors import InvalidImageError, InvalidPathError
from codex_3d_mcp.inputs import cleanup_staged_image, stage_image_input

_buffer = io.BytesIO()
Image.new("RGB", (8, 8), "red").save(_buffer, format="PNG")
PNG = _buffer.getvalue()


def settings(tmp_path: Path) -> Settings:
    return Settings(
        output_dir=tmp_path / "outputs",
        model_cache_dir=tmp_path / "cache",
        allowed_input_roots=(tmp_path.resolve(),),
    )


def test_stages_data_url_and_cleans_it(tmp_path: Path) -> None:
    staged = stage_image_input(
        settings(tmp_path), image_data_url="data:image/png;base64," + base64.b64encode(PNG).decode()
    )
    assert staged.path.is_file()
    cleanup_staged_image(staged)
    assert not staged.path.exists()


def test_requires_exactly_one_input(tmp_path: Path) -> None:
    with pytest.raises(InvalidImageError, match="exactly one"):
        stage_image_input(settings(tmp_path))


def test_rejects_path_outside_allowed_root(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG)
    restricted = settings(tmp_path)
    restricted = Settings(
        output_dir=restricted.output_dir,
        model_cache_dir=restricted.model_cache_dir,
        allowed_input_roots=(allowed.resolve(),),
    )
    with pytest.raises(InvalidPathError, match="outside allowed"):
        stage_image_input(restricted, image_path=str(outside))


def test_accepts_shared_path(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(PNG)
    staged = stage_image_input(settings(tmp_path), image_path=str(image))
    assert staged.path == image.resolve()
    assert not staged.cleanup


@pytest.mark.parametrize("payload", [b"\x89PNG\r\n\x1a\nminimal", b"RIFFfake wave file"])
def test_rejects_signature_only_payloads(tmp_path, payload):
    with pytest.raises(InvalidImageError, match="Invalid image"):
        stage_image_input(settings(tmp_path), image_base64=base64.b64encode(payload).decode())


def test_rejects_large_base64_before_decoding(tmp_path):
    restricted = replace(settings(tmp_path), max_image_bytes=3)
    with pytest.raises(InvalidImageError, match="Encoded image exceeds"):
        stage_image_input(restricted, image_base64="A" * 8)
    assert not restricted.output_dir.exists()


def test_rejects_excessive_decoded_dimensions(tmp_path):
    restricted = replace(settings(tmp_path), max_image_pixels=63)
    image = tmp_path / "input.png"
    image.write_bytes(PNG)
    with pytest.raises(InvalidImageError, match="64 pixels"):
        stage_image_input(restricted, image_path=str(image))
