from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from codex_3d_mcp.errors import InvalidImageError
from codex_3d_mcp.hunyuan_mv.backgrounds import prepare_reference


def save_reference(tmp_path: Path, image: Image.Image) -> Path:
    path = tmp_path / "source.png"
    image.save(path)
    return path


def foreground(background=(0, 0, 0, 0)) -> Image.Image:
    image = Image.new("RGBA", (512, 512), background)
    ImageDraw.Draw(image).ellipse((96, 64, 415, 447), fill=(32, 100, 180, 255))
    return image


def test_real_alpha_persists_rgba_mask_and_json(tmp_path):
    report = prepare_reference(save_reference(tmp_path, foreground()), tmp_path / "prepared.png")
    prepared = Image.open(report["prepared_path"])
    assert prepared.mode == "RGBA"
    assert prepared.getpixel((0, 0)) == (0, 0, 0, 0)
    assert prepared.getpixel((256, 256)) == (32, 100, 180, 255)
    assert Image.open(report["mask_path"]).mode == "L"
    assert json.loads(Path(report["report_path"]).read_text()) == report
    assert report["background_method"] == "existing_alpha"
    assert report["semantic_review_required"] is True


@pytest.mark.parametrize("background", [(255, 255, 255, 255), (0, 0, 0, 255)])
def test_opaque_background_square_regression_fails_closed(tmp_path, background):
    destination = tmp_path / "prepared.png"
    with pytest.raises(InvalidImageError, match="full margin"):
        prepare_reference(save_reference(tmp_path, foreground(background)), destination)
    assert not destination.exists()


def test_corner_pixels_do_not_establish_chroma_contract(tmp_path):
    image = foreground((255, 255, 255, 255))
    for corner in ((0, 0), (511, 0), (0, 511), (511, 511)):
        image.putpixel(corner, (255, 0, 255, 255))
    with pytest.raises(InvalidImageError, match="corner colors alone"):
        prepare_reference(save_reference(tmp_path, image), tmp_path / "prepared.png")


def test_chroma_removal_preserves_object_colors_and_unmixes_fringe(tmp_path):
    image = Image.new("RGB", (512, 512), (255, 0, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 411, 411), fill=(30, 70, 120))
    # Exact 50% mixture of the adjacent object's color and reserved magenta.
    draw.line((99, 100, 99, 411), fill=(142, 35, 188))
    draw.rectangle((200, 200, 220, 220), fill=(128, 0, 128))
    report = prepare_reference(save_reference(tmp_path, image), tmp_path / "prepared.png")
    prepared = Image.open(report["prepared_path"])
    assert prepared.getpixel((20, 20)) == (0, 0, 0, 0)
    assert prepared.getpixel((200, 200)) == (128, 0, 128, 255)
    assert prepared.getpixel((100, 250)) == (30, 70, 120, 255)
    red, green, blue, alpha = prepared.getpixel((99, 250))
    assert 120 <= alpha <= 135
    assert abs(red - 30) <= 2 and abs(green - 70) <= 2 and abs(blue - 120) <= 2
    assert report["edge_pixels_decontaminated"] > 0
    assert report["background_method"] == "chroma_key"


def test_explicit_custom_chroma_key(tmp_path):
    report = prepare_reference(
        save_reference(tmp_path, foreground((0, 255, 0, 255))),
        tmp_path / "prepared.png",
        background_mode="chroma",
        key_color=(0, 255, 0),
    )
    assert report["key_color"] == [0, 255, 0]
    assert Image.open(report["prepared_path"]).getpixel((0, 0))[3] == 0


def test_transparent_mode_does_not_silently_key_opaque_input(tmp_path):
    with pytest.raises(InvalidImageError, match="No useful alpha"):
        prepare_reference(
            save_reference(tmp_path, foreground((255, 0, 255, 255))),
            tmp_path / "prepared.png",
            background_mode="transparent",
        )


@pytest.mark.parametrize("empty", [(0, 0, 0, 0), (255, 0, 255, 255)])
def test_empty_foreground_fails(tmp_path, empty):
    with pytest.raises(InvalidImageError, match="Foreground is empty"):
        prepare_reference(
            save_reference(tmp_path, Image.new("RGBA", (512, 512), empty)),
            tmp_path / "prepared.png",
        )


def test_cropped_foreground_is_rejected(tmp_path):
    image = foreground()
    ImageDraw.Draw(image).rectangle((0, 240, 200, 270), fill=(30, 80, 100, 255))
    with pytest.raises(InvalidImageError, match="exterior margin"):
        prepare_reference(save_reference(tmp_path, image), tmp_path / "prepared.png")


@pytest.mark.parametrize("painted_alpha", [False, True])
def test_fake_checkerboard_transparency_is_rejected(tmp_path, painted_alpha):
    image = Image.new("RGBA", (512, 512), (245, 245, 245, 255))
    draw = ImageDraw.Draw(image)
    for y in range(0, 512, 32):
        for x in range(0, 512, 32):
            if ((x + y) // 32) % 2:
                draw.rectangle(
                    (x, y, x + 31, y + 31),
                    fill=(185, 185, 185, 0 if painted_alpha else 255),
                )
    draw.ellipse((150, 100, 360, 410), fill=(30, 90, 140, 255))
    with pytest.raises(InvalidImageError):
        prepare_reference(save_reference(tmp_path, image), tmp_path / "prepared.png")


def test_low_resolution_is_rejected(tmp_path):
    with pytest.raises(InvalidImageError, match="both dimensions"):
        prepare_reference(
            save_reference(tmp_path, foreground().resize((256, 256))),
            tmp_path / "prepared.png",
        )


def test_one_transparent_pixel_does_not_make_opaque_square_valid(tmp_path):
    image = foreground((255, 255, 255, 255))
    image.putpixel((0, 0), (0, 0, 0, 0))
    with pytest.raises(InvalidImageError, match="exterior margin"):
        prepare_reference(save_reference(tmp_path, image), tmp_path / "prepared.png")


def test_small_or_faint_foreground_is_rejected(tmp_path):
    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((250, 250, 252, 252), fill=(0, 0, 0, 255))
    with pytest.raises(InvalidImageError, match="too small"):
        prepare_reference(save_reference(tmp_path, image), tmp_path / "prepared.png")


def test_preparation_never_overwrites_source(tmp_path):
    source = save_reference(tmp_path, foreground())
    before = source.read_bytes()
    with pytest.raises(InvalidImageError, match="differ"):
        prepare_reference(source, source)
    assert source.read_bytes() == before


def test_mask_output_cannot_overwrite_input(tmp_path):
    source = tmp_path / "prepared.mask.png"
    foreground().save(source)
    before = source.read_bytes()
    with pytest.raises(InvalidImageError, match="differ"):
        prepare_reference(source, tmp_path / "prepared.png")
    assert source.read_bytes() == before


def test_background_square_with_token_transparent_border_is_rejected(tmp_path):
    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((12, 12, 499, 499), fill=(255, 255, 255, 255))
    with pytest.raises(InvalidImageError, match="over 85%"):
        prepare_reference(save_reference(tmp_path, image), tmp_path / "prepared.png")
