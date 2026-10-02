from PIL import Image

from codex_3d_mcp.hunyuan_mv.worker import (
    _has_flat_magenta_backdrop,
    _remove_flat_magenta_backdrop,
)


def test_flat_magenta_backdrop_is_keyed_without_erasing_foreground() -> None:
    image = Image.new("RGB", (8, 8), (255, 0, 255))
    for x in range(2, 6):
        for y in range(2, 6):
            image.putpixel((x, y), (32, 96, 128))

    assert _has_flat_magenta_backdrop(image)
    keyed = _remove_flat_magenta_backdrop(image)

    assert keyed.mode == "RGBA"
    assert keyed.getpixel((0, 0))[3] == 0
    assert keyed.getpixel((3, 3))[3] == 255
    assert keyed.getpixel((3, 3))[:3] == (32, 96, 128)


def test_non_magenta_backdrop_does_not_match_chroma_contract() -> None:
    image = Image.new("RGB", (8, 8), (127, 127, 127))
    assert not _has_flat_magenta_backdrop(image)
