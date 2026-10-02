"""Image input validation and staging."""

from __future__ import annotations

import base64
import binascii
import io
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .config import Settings
from .errors import InvalidImageError, InvalidPathError

_DATA_URL_RE = re.compile(r"^data:(image/(?:png|jpeg|jpg|webp));base64,(.+)$", re.IGNORECASE)
_EXTENSIONS = {".png": ".png", ".jpg": ".jpg", ".jpeg": ".jpeg", ".webp": ".webp"}


@dataclass(frozen=True)
class StagedImage:
    path: Path
    cleanup: bool
    source: str


def _validate_image_bytes(data: bytes, settings: Settings) -> None:
    if not data:
        raise InvalidImageError("The image input is empty.")
    if len(data) > settings.max_image_bytes:
        raise InvalidImageError(
            f"Image is {len(data)} bytes; the limit is {settings.max_image_bytes} bytes.",
            code="IMAGE_TOO_LARGE",
        )
    _verify_image(io.BytesIO(data), settings.max_image_pixels)


def _verify_image(source, max_pixels: int) -> None:
    try:
        with Image.open(source) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise InvalidImageError("Unsupported image; provide PNG, JPEG, or WebP.")
            width, height = image.size
            if width * height > max_pixels:
                raise InvalidImageError(
                    f"Image has {width * height} pixels; the limit is {max_pixels}.",
                    code="IMAGE_TOO_LARGE",
                )
            image.verify()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"Invalid image: {exc}") from exc


def validate_image_file(
    path: Path, *, max_bytes: int = 25 * 1024 * 1024, max_pixels: int = 32 * 1024 * 1024
) -> None:
    """Reject invalid or excessively large inputs before starting any ML work."""
    if path.stat().st_size > max_bytes:
        raise InvalidImageError("Image file exceeds the configured size limit.", code="IMAGE_TOO_LARGE")
    _verify_image(path, max_pixels)


def _decode_base64(encoded: str, settings: Settings, field: str) -> bytes:
    if len(encoded) > 4 * ((settings.max_image_bytes + 2) // 3):
        raise InvalidImageError("Encoded image exceeds the configured size limit.", code="IMAGE_TOO_LARGE")
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError(f"{field} contains invalid base64.") from exc


def _suffix_for_mime(mime: str) -> str:
    return (
        ".jpg"
        if mime.lower() in {"image/jpeg", "image/jpg"}
        else "." + mime.rsplit("/", 1)[-1].lower()
    )


def _write_staged(data: bytes, suffix: str, settings: Settings) -> Path:
    staging_dir = settings.output_dir / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="image-", suffix=suffix, dir=staging_dir, delete=False
    ) as handle:
        handle.write(data)
        return Path(handle.name).resolve()


def stage_image_input(
    settings: Settings,
    *,
    image_data_url: str | None = None,
    image_base64: str | None = None,
    image_path: str | None = None,
) -> StagedImage:
    supplied = [value is not None for value in (image_data_url, image_base64, image_path)]
    if sum(supplied) != 1:
        raise InvalidImageError(
            "Provide exactly one of image_data_url, image_base64, or image_path.",
            code="IMAGE_INPUT_COUNT",
        )

    if image_path is not None:
        candidate = Path(image_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise InvalidPathError(f"Image path does not exist or is not a file: {candidate}")
        if not settings.is_allowed_input(candidate):
            raise InvalidPathError(f"Image path is outside allowed input roots: {candidate}")
        if candidate.suffix.lower() not in _EXTENSIONS:
            raise InvalidImageError("Image path must end in .png, .jpg, .jpeg, or .webp.")
        validate_image_file(
            candidate, max_bytes=settings.max_image_bytes, max_pixels=settings.max_image_pixels
        )
        return StagedImage(candidate, cleanup=False, source="path")

    if image_data_url is not None:
        match = _DATA_URL_RE.match(image_data_url.strip())
        if not match:
            raise InvalidImageError("image_data_url must be a base64 PNG, JPEG, or WebP data URL.")
        mime, encoded = match.groups()
        data = _decode_base64(encoded, settings, "image_data_url")
        _validate_image_bytes(data, settings)
        return StagedImage(_write_staged(data, _suffix_for_mime(mime), settings), True, "data_url")

    assert image_base64 is not None
    data = _decode_base64(image_base64.strip(), settings, "image_base64")
    _validate_image_bytes(data, settings)
    return StagedImage(_write_staged(data, ".png", settings), True, "base64")


def cleanup_staged_image(staged: StagedImage) -> None:
    if not staged.cleanup:
        return
    try:
        staged.path.unlink(missing_ok=True)
    except OSError:
        pass
