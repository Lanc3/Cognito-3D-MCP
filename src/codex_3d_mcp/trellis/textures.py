"""Local reference upscaling and final PBR map assembly."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from ..errors import Codex3DError, ModelUnavailableError
from .config import TrellisSettings
from .validation import frame_foreground_mask


class TextureProcessor:
    def __init__(self, settings: TrellisSettings) -> None:
        self.settings = settings

    def upscale(self, source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.settings.test_mode:
            with Image.open(source) as image:
                image.resize(
                    (self.settings.texture_resolution, self.settings.texture_resolution),
                    Image.Resampling.LANCZOS,
                ).save(destination, "PNG")
            return destination
        if not self.settings.realesrgan_exe.is_file():
            raise ModelUnavailableError("Real-ESRGAN is missing; run scripts/setup-trellis.ps1.")
        command = [
            str(self.settings.realesrgan_exe),
            "-i",
            str(source),
            "-o",
            str(destination),
            "-n",
            "realesrgan-x4plus",
            "-s",
            "4",
            "-t",
            "256",
            "-f",
            "png",
        ]
        result = subprocess.run(
            command,
            cwd=self.settings.realesrgan_exe.parent,
            capture_output=True,
            text=True,
            timeout=600,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            check=False,
        )
        if result.returncode != 0 or not destination.is_file():
            detail = (result.stderr or result.stdout)[-2000:]
            raise Codex3DError(f"Real-ESRGAN failed: {detail}", code="UPSCALE_FAILED")
        return destination

    def prepare_reference(self, source: Path, destination: Path) -> Path:
        """Remove the frame background without changing camera registration."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            mask = frame_foreground_mask(image)
            # Projection outside known seed pixels uses a neutral fallback. Keeping
            # the original frame dimensions preserves the canonical camera fit.
            prepared = Image.new("RGB", rgb.size, (24, 24, 28))
            prepared.paste(rgb, mask=mask)
            prepared.save(destination, "PNG", optimize=True)
            mask.save(destination.with_name(f"{destination.stem}-mask.png"), "PNG")
        return destination

    def assemble_maps(self, directory: Path, material_hints: str) -> dict[str, str | None]:
        base_path = directory / "baseColor.png"
        ao_path = directory / "ao.png"
        if not base_path.is_file() or not ao_path.is_file():
            raise Codex3DError(
                "Blender did not produce base color and AO maps.", code="BAKE_FAILED"
            )
        size = (self.settings.texture_resolution, self.settings.texture_resolution)
        with Image.open(base_path) as source:
            base = source.convert("RGBA").resize(size, Image.Resampling.LANCZOS)
            # A small contrast restoration counters diffuse-bake and projection washout.
            rgb = ImageEnhance.Contrast(base.convert("RGB")).enhance(1.04)
            base = Image.merge("RGBA", (*rgb.split(), base.getchannel("A")))
            base.save(base_path, "PNG", optimize=True)
        with Image.open(ao_path) as source_ao:
            ao = source_ao.convert("L").resize(size, Image.Resampling.LANCZOS)

        roughness, metallic = _material_defaults(material_hints)
        orm = Image.merge(
            "RGB",
            (
                ao,
                Image.new("L", size, round(roughness * 255)),
                Image.new("L", size, round(metallic * 255)),
            ),
        )
        orm_path = directory / "orm.png"
        orm.save(orm_path, "PNG", optimize=True)

        alpha = base.getchannel("A")
        opacity_path: Path | None = None
        if alpha.getextrema()[0] < 250:
            opacity_path = directory / "opacity.png"
            alpha.save(opacity_path, "PNG", optimize=True)

        emission_path = _emission_mask(base, material_hints, directory / "emissive.png")
        return {
            "base_color": str(base_path.resolve()),
            "normal": str((directory / "normal.png").resolve()),
            "orm": str(orm_path.resolve()),
            "emissive": str(emission_path.resolve()) if emission_path else None,
            "opacity": str(opacity_path.resolve()) if opacity_path else None,
            "roughness": roughness,
            "metallic": metallic,
        }


def _material_defaults(hints: str) -> tuple[float, float]:
    value = hints.lower()
    metallic = 0.85 if any(word in value for word in ("metal", "steel", "iron", "gold")) else 0.05
    if any(word in value for word in ("rough", "stone", "bark", "cloth", "matte")):
        roughness = 0.78
    elif any(word in value for word in ("polished", "gloss", "glass", "chrome")):
        roughness = 0.22
    else:
        roughness = 0.55
    return roughness, metallic


def _emission_mask(base: Image.Image, hints: str, destination: Path) -> Path | None:
    explicit = any(word in hints.lower() for word in ("glow", "emissive", "lit", "luminous"))
    hsv = base.convert("RGB").convert("HSV")
    saturation = hsv.getchannel("S")
    value = hsv.getchannel("V")
    saturated = saturation.point(lambda pixel: 255 if pixel >= 115 else 0)
    bright = value.point(lambda pixel: 255 if pixel >= (165 if explicit else 225) else 0)
    mask = Image.new("L", base.size, 0)
    mask.paste(bright, mask=saturated)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.0))
    if not explicit and mask.getbbox() is None:
        return None
    black = Image.new("RGB", base.size, (0, 0, 0))
    black.paste(base.convert("RGB"), mask=mask)
    black.save(destination, "PNG", optimize=True)
    return destination
