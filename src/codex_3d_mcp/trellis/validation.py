"""Seed-pair structural and semantic validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from PIL import Image, ImageChops, ImageStat

from ..errors import InvalidImageError


@dataclass(frozen=True)
class ViewMetrics:
    width: int
    height: int
    bbox: tuple[int, int, int, int]
    center_x: float
    center_y: float
    foreground_width: float
    foreground_height: float
    foreground_area: float
    minimum_margin: float


@dataclass(frozen=True)
class PairValidation:
    outcome: str
    metrics: dict[str, float | str | bool | dict]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def approved(self) -> bool:
        return self.outcome == "approved"

    def as_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "metrics": self.metrics,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "approved": self.approved,
        }


class PairValidator:
    def __init__(self, minimum_size: int = 1024, semantic_threshold: float = 0.78) -> None:
        self.minimum_size = minimum_size
        self.semantic_threshold = semantic_threshold

    def validate(
        self,
        front: Path,
        back: Path,
        front_mask: Path,
        back_mask: Path,
        semantic_similarity: Callable[[Path, Path], float],
    ) -> PairValidation:
        errors: list[str] = []
        warnings: list[str] = []
        front_view = _inspect(front, front_mask, self.minimum_size)
        back_view = _inspect(back, back_mask, self.minimum_size)

        aspect_difference = abs(
            front_view.width / front_view.height - back_view.width / back_view.height
        )
        center_difference = max(
            abs(front_view.center_x - back_view.center_x),
            abs(front_view.center_y - back_view.center_y),
        )
        height_ratio = _ratio(back_view.foreground_height, front_view.foreground_height)
        area_ratio = _ratio(back_view.foreground_area, front_view.foreground_area)
        semantic = semantic_similarity(front, back)
        duplicate_score, mirror_score = _image_relationship(front, back)

        if aspect_difference > 0.01:
            errors.append("Front/back aspect ratios differ by more than 1%.")
        if center_difference > 0.03:
            warnings.append("Object centers differ by more than 3% of the frame.")
        if not 0.90 <= height_ratio <= 1.10:
            warnings.append("Object heights differ by more than the allowed 10%.")
        if not 0.75 <= area_ratio <= 1.25:
            warnings.append("Foreground areas differ by more than the allowed 25%.")
        if min(front_view.minimum_margin, back_view.minimum_margin) < 0.02:
            errors.append("An object is clipped or has less than a 2% clear border.")
        if semantic < self.semantic_threshold:
            errors.append("DINOv2 similarity is below the production threshold.")
        if duplicate_score < 0.015:
            errors.append("The two references appear to be duplicate views.")
        elif mirror_score + 0.01 < duplicate_score and mirror_score < 0.06:
            errors.append("The back reference appears to be a mirrored front view.")

        outcome = "rejected" if errors else "review" if warnings else "approved"
        return PairValidation(
            outcome=outcome,
            metrics={
                "aspect_difference": aspect_difference,
                "center_difference": center_difference,
                "height_ratio": height_ratio,
                "area_ratio": area_ratio,
                "semantic_similarity": semantic,
                "duplicate_score": duplicate_score,
                "mirror_score": mirror_score,
                "front": front_view.__dict__,
                "back": back_view.__dict__,
            },
            warnings=tuple(warnings),
            errors=tuple(errors),
        )


def _inspect(image_path: Path, mask_path: Path, minimum_size: int) -> ViewMetrics:
    if image_path.suffix.lower() != ".png":
        raise InvalidImageError("Bidirectional production inputs must be PNG files.")
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
    except (OSError, SyntaxError) as exc:
        raise InvalidImageError(f"Could not decode image: {image_path}") from exc
    if width != height:
        raise InvalidImageError("Bidirectional production inputs must be square.")
    if min(width, height) < minimum_size:
        raise InvalidImageError(
            f"Bidirectional inputs must be at least {minimum_size}x{minimum_size}."
        )
    with Image.open(mask_path) as source_mask:
        cutout = source_mask.convert("RGB")
        # trellis.cpp writes a tightly cropped RGB cutout rather than a frame-sized
        # mask. It remains useful as a background-removal integrity check, but its
        # coordinates must never be compared with the source frame.
        extrema = ImageStat.Stat(cutout).extrema
        if all(channel[0] == channel[1] == 0 for channel in extrema):
            raise InvalidImageError("Background removal produced an empty foreground mask.")
    with Image.open(image_path) as source:
        grayscale = frame_foreground_mask(source)
        bbox = grayscale.getbbox()
    if bbox is None:
        raise InvalidImageError("Background removal produced an empty foreground mask.")
    left, top, right, bottom = bbox
    foreground_width = (right - left) / width
    foreground_height = (bottom - top) / height
    foreground_area = ImageStat.Stat(grayscale).sum[0] / (255.0 * width * height)
    minimum_margin = min(
        left / width, top / height, (width - right) / width, (height - bottom) / height
    )
    return ViewMetrics(
        width=width,
        height=height,
        bbox=bbox,
        center_x=((left + right) / 2) / width,
        center_y=((top + bottom) / 2) / height,
        foreground_width=foreground_width,
        foreground_height=foreground_height,
        foreground_area=foreground_area,
        minimum_margin=minimum_margin,
    )


def frame_foreground_mask(source: Image.Image) -> Image.Image:
    """Return a frame-sized mask for alpha or a plain removable background.

    Production references require either transparency or a plain background. The
    TRELLIS BiRefNet cutout is cropped, so source-frame geometry is recovered from
    alpha when available and otherwise from the robust median corner colour.
    """

    rgba = source.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getextrema()[0] < 255:
        return alpha.point(lambda value: 255 if value > 8 else 0)

    rgb = rgba.convert("RGB")
    width, height = rgb.size
    patch = max(2, round(min(width, height) * 0.02))
    pixels: list[tuple[int, int, int]] = []
    for box in (
        (0, 0, patch, patch),
        (width - patch, 0, width, patch),
        (0, height - patch, patch, height),
        (width - patch, height - patch, width, height),
    ):
        pixels.extend(rgb.crop(box).get_flattened_data())
    background = tuple(round(median(channel)) for channel in zip(*pixels, strict=True))
    deviations = sorted(
        max(abs(pixel[index] - background[index]) for index in range(3))
        for pixel in pixels
    )
    p95 = deviations[min(len(deviations) - 1, round(0.95 * (len(deviations) - 1)))]
    # A 24-level floor tolerates antialiasing and gentle gradients in generated
    # plain backgrounds without eating into normally exposed subject pixels.
    threshold = max(24, min(48, p95 * 3 + 4))
    difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, background))
    red, green, blue = difference.split()
    maximum = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    return maximum.point(lambda value: 255 if value > threshold else 0)


def _ratio(value: float, baseline: float) -> float:
    return value / baseline if baseline else float("inf")


def _image_relationship(front: Path, back: Path) -> tuple[float, float]:
    with Image.open(front) as first, Image.open(back) as second:
        a = first.convert("RGB").resize((64, 64))
        b = second.convert("RGB").resize((64, 64))
        direct = ImageStat.Stat(ImageChops.difference(a, b)).mean
        flipped = a.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mirrored = ImageStat.Stat(ImageChops.difference(flipped, b)).mean
    return sum(direct) / (3 * 255), sum(mirrored) / (3 * 255)
