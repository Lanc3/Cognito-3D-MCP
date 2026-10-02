"""Deterministic reference preparation; unknown opaque backgrounds fail closed.

The chroma color is reserved for background, including holes inside a silhouette.
These pixel checks cannot establish object identity or prove that every visible
pixel belongs to the object. Codex must also review the prepared references.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageOps, UnidentifiedImageError

from ..errors import InvalidImageError

_KEY_TOLERANCE = 12
_ALPHA_NOISE = 8


def _fail(reason: str) -> None:
    raise InvalidImageError(
        f"{reason} Repair or regenerate the reference with genuine transparency or a "
        "uniform reserved chroma background and clear margin, then retry preparation. "
        "Review the prepared image for leftover background and object consistency."
    )


def _outside_is_clear(mask: Image.Image, margin: int) -> bool:
    """Inspect every pixel of the exterior band, not just its corners."""
    width, height = mask.size
    boxes = (
        (0, 0, width, margin),
        (0, height - margin, width, height),
        (0, margin, margin, height - margin),
        (width - margin, margin, width, height - margin),
    )
    return all(mask.crop(box).getextrema()[1] == 0 for box in boxes)


def _key_mask(rgb: Image.Image, key_color: tuple[int, int, int]) -> Image.Image:
    difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, key_color))
    red, green, blue = difference.split()
    distance = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    return distance.point(lambda value: 255 if value <= _KEY_TOLERANCE else 0)


def _remove_key(image: Image.Image, key_color: tuple[int, int, int]) -> tuple[Image.Image, int]:
    """Remove reserved key pixels and unmix only supported antialias edge colors.

    A fringe pixel is changed only if its RGB fits a mixture of the declared
    backdrop and nearby interior color. Hard edges and unrelated colors stay
    untouched; this avoids a broad hue-selection filter erasing object colors.
    """
    rgb = image.convert("RGB")
    key_mask = _key_mask(rgb, key_color)
    original_alpha = image.getchannel("A")
    alpha = ImageChops.multiply(original_alpha, ImageOps.invert(key_mask))
    # The nearest 3-pixel band contains likely antialiasing, while candidates
    # beyond it supply opaque interior colors for a conservative mixture fit.
    expanded = key_mask.filter(ImageFilter.MaxFilter(7))
    fringe = ImageChops.subtract(expanded, key_mask)
    output = image.copy()
    output.putalpha(alpha)
    pixels = output.load()
    source_pixels = image.load()
    fringe_pixels = fringe.load()
    expanded_pixels = expanded.load()
    alpha_pixels = original_alpha.load()
    width, height = image.size
    corrected = 0
    for y in range(height):
        for x in range(width):
            if not fringe_pixels[x, y] or alpha_pixels[x, y] <= _ALPHA_NOISE:
                continue
            color = source_pixels[x, y][:3]
            delta = tuple(color[c] - key_color[c] for c in range(3))
            best: tuple[float, float] | None = None
            for radius in (2, 3, 4):
                for dx, dy in (
                    (-radius, 0), (radius, 0), (0, -radius), (0, radius),
                    (-radius, -radius), (-radius, radius),
                    (radius, -radius), (radius, radius),
                ):
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if expanded_pixels[nx, ny] or alpha_pixels[nx, ny] < 250:
                        continue
                    foreground = source_pixels[nx, ny][:3]
                    direction = tuple(foreground[c] - key_color[c] for c in range(3))
                    denominator = sum(component * component for component in direction)
                    if denominator < 1024:
                        continue
                    coverage = sum(delta[c] * direction[c] for c in range(3)) / denominator
                    if not 0.025 < coverage < 0.97:
                        continue
                    residual = max(abs(delta[c] - coverage * direction[c]) for c in range(3))
                    if residual <= 8 and (best is None or residual < best[0]):
                        best = residual, coverage
                if best is not None:
                    break
            if best is not None:
                coverage = best[1]
                recovered = tuple(
                    max(0, min(255, round(key_color[c] + delta[c] / coverage)))
                    for c in range(3)
                )
                pixels[x, y] = (*recovered, round(alpha_pixels[x, y] * coverage))
                corrected += 1
    # Discard hidden backdrop RGB so a later RGB conversion cannot resurrect it.
    output.paste((0, 0, 0, 0), mask=key_mask)
    return output, corrected


def prepare_reference(
    source: Path,
    destination: Path,
    *,
    background_mode: str = "auto",
    key_color: tuple[int, int, int] = (255, 0, 255),
    minimum_size: int = 512,
) -> dict[str, Any]:
    """Write a validated RGBA PNG, alpha-mask PNG and JSON report.

    ``auto`` accepts useful existing alpha or the reserved key across the entire
    outer band. ``transparent`` requires useful alpha. ``chroma`` requires the
    declared key across the band (existing transparent pixels are also allowed).
    No model-based background removal or guessed opaque background is attempted.
    Destinations are written only after all pixel gates pass.
    """
    source, destination = Path(source), Path(destination)
    if background_mode not in {"auto", "transparent", "chroma"}:
        _fail("background_mode must be auto, transparent, or chroma.")
    if (
        not isinstance(key_color, (tuple, list))
        or len(key_color) != 3
        or any(type(value) is not int or not 0 <= value <= 255 for value in key_color)
    ):
        _fail("key_color must contain three integer RGB values from 0 to 255.")
    key_color = tuple(key_color)
    if type(minimum_size) is not int or minimum_size < 1:
        _fail("minimum_size must be a positive integer.")
    if destination.suffix.lower() != ".png":
        _fail("The prepared destination must have a .png extension.")
    mask_path = destination.with_name(f"{destination.stem}.mask.png")
    report_path = destination.with_name(f"{destination.stem}.background.json")
    if source.resolve() in {path.resolve() for path in (destination, mask_path, report_path)}:
        _fail("The prepared destination must differ from the original reference path.")
    try:
        with Image.open(source) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                _fail("Animated or multiframe images are not supported.")
            image = ImageOps.exif_transpose(opened).convert("RGBA")
            image.load()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        _fail(f"Could not decode reference {source.name}: {exc}.")
    width, height = image.size
    if min(width, height) < minimum_size:
        _fail(f"Reference is {width}x{height}; both dimensions must be at least {minimum_size}.")
    margin = max(2, math.ceil(min(width, height) * 0.02))
    if margin * 2 >= min(width, height):
        _fail("Reference is too small to provide a clear exterior margin.")
    alpha = image.getchannel("A")
    useful_alpha = alpha.getextrema()[0] <= _ALPHA_NOISE
    corrected = 0
    if background_mode != "chroma" and useful_alpha:
        method = "existing_alpha"
    elif background_mode == "transparent":
        _fail(
            "No useful alpha transparency exists; an opaque square or painted checkerboard "
            "is not transparency."
        )
    else:
        key_mask = _key_mask(image.convert("RGB"), key_color)
        visible = alpha.point(lambda value: 255 if value > _ALPHA_NOISE else 0)
        unknown = ImageChops.multiply(visible, ImageOps.invert(key_mask))
        if not _outside_is_clear(unknown, margin):
            _fail(
                "The opaque exterior is not a uniform reserved chroma background across "
                "the full margin; corner colors alone are insufficient."
            )
        image, corrected = _remove_key(image, key_color)
        method = "chroma_key"

    alpha = image.getchannel("A").point(lambda value: 0 if value <= _ALPHA_NOISE else value)
    image.putalpha(alpha)
    foreground_mask = alpha.point(lambda value: 255 if value else 0)
    if not _outside_is_clear(foreground_mask, margin):
        _fail(
            f"Foreground or leftover background touches the {margin}-pixel exterior margin."
        )
    histogram = alpha.histogram()
    pixels_count = width * height
    occupancy = sum(value * count for value, count in enumerate(histogram)) / (255 * pixels_count)
    solid_fraction = sum(histogram[224:]) / pixels_count
    bbox = foreground_mask.getbbox()
    if bbox is None or occupancy < 0.01 or solid_fraction < 0.005:
        _fail("Foreground is empty, too small, or almost entirely transparent.")
    if occupancy > 0.85:
        _fail(
            "Foreground occupies over 85% of the image; "
            "a background square or inadequate margin may remain."
        )

    # Normalize hidden RGB for existing alpha too. Partial alpha keeps its color.
    hidden = alpha.point(lambda value: 255 if value == 0 else 0)
    image.paste((0, 0, 0, 0), mask=hidden)
    report: dict[str, Any] = {
        "status": "passed",
        "source_path": str(source.resolve()),
        "prepared_path": str(destination.resolve()),
        "mask_path": str(mask_path.resolve()),
        "report_path": str(report_path.resolve()),
        "background_mode_requested": background_mode,
        "background_method": method,
        "key_color": list(key_color) if method == "chroma_key" else None,
        "width": width,
        "height": height,
        "foreground_bbox": list(bbox),
        "foreground_occupancy": round(occupancy, 6),
        "solid_foreground_fraction": round(solid_fraction, 6),
        "transparent_fraction": round(histogram[0] / pixels_count, 6),
        "required_margin_pixels": margin,
        "perimeter_clean": True,
        "edge_pixels_decontaminated": corrected,
        "semantic_review_required": True,
        "notes": [
            "Pixel gates do not prove all background is absent or all views show the same object.",
            "Codex must inspect prepared RGBA images and masks before generation.",
            "Chroma mode reserves the declared key color exclusively for background.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
    alpha.save(mask_path, format="PNG")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
