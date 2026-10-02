from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from codex_3d_mcp.trellis.validation import PairValidator


def _view(path: Path, color: tuple[int, int, int], offset: int = 0) -> Path:
    image = Image.new("RGB", (128, 128), "black")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((32 + offset, 16, 96 + offset, 112), radius=12, fill=color)
    image.save(path, "PNG")
    return path


def test_clean_pair_is_approved(tmp_path: Path) -> None:
    front = _view(tmp_path / "front.png", (240, 40, 30))
    back = _view(tmp_path / "back.png", (30, 80, 240))
    validator = PairValidator(minimum_size=128)
    result = validator.validate(front, back, front, back, lambda _a, _b: 0.91)
    assert result.outcome == "approved"
    assert not result.errors


def test_center_mismatch_requires_review(tmp_path: Path) -> None:
    front = _view(tmp_path / "front.png", (240, 40, 30))
    back = _view(tmp_path / "back.png", (30, 80, 240), offset=6)
    validator = PairValidator(minimum_size=128)
    result = validator.validate(front, back, front, back, lambda _a, _b: 0.91)
    assert result.outcome == "review"
    assert any("centers" in warning for warning in result.warnings)


def test_semantically_different_pair_is_rejected(tmp_path: Path) -> None:
    front = _view(tmp_path / "front.png", (240, 40, 30))
    back = _view(tmp_path / "back.png", (30, 80, 240))
    validator = PairValidator(minimum_size=128)
    result = validator.validate(front, back, front, back, lambda _a, _b: 0.4)
    assert result.outcome == "rejected"
    assert any("DINOv2" in error for error in result.errors)


def test_cropped_birefnet_cutout_does_not_corrupt_frame_metrics(tmp_path: Path) -> None:
    front = _view(tmp_path / "front.png", (240, 40, 30))
    back = _view(tmp_path / "back.png", (30, 80, 240))
    _view(tmp_path / "front-mask.png", (240, 40, 30))
    _view(tmp_path / "back-mask.png", (30, 80, 240))
    with Image.open(tmp_path / "front-mask.png") as source:
        front_cutout = source.crop((24, 8, 105, 121))
    with Image.open(tmp_path / "back-mask.png") as source:
        back_cutout = source.crop((28, 12, 101, 117))
    front_cutout.save(tmp_path / "front-mask.png", "PNG")
    back_cutout.save(tmp_path / "back-mask.png", "PNG")

    result = PairValidator(minimum_size=128).validate(
        front,
        back,
        tmp_path / "front-mask.png",
        tmp_path / "back-mask.png",
        lambda _a, _b: 0.91,
    )

    assert result.outcome == "approved"
    assert result.metrics["center_difference"] == 0.0
    assert result.metrics["height_ratio"] == 1.0
