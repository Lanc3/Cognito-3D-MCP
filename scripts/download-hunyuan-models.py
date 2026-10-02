"""Download only the pinned Hunyuan3D-2mv and Paint files used by this server."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-revision", required=True)
    parser.add_argument("--texture-revision", required=True)
    args = parser.parse_args()
    snapshot_download(
        "tencent/Hunyuan3D-2mv",
        revision=args.shape_revision,
        allow_patterns=[
            "config.json",
            "LICENSE",
            "NOTICE",
            "hunyuan3d-dit-v2-mv/config.yaml",
            # The Windows-safe loader requires normal process-owned checkpoint
            # storage and deliberately never transfers safetensors mappings.
            "hunyuan3d-dit-v2-mv/model.fp16.ckpt",
        ],
    )
    snapshot_download(
        "tencent/Hunyuan3D-2",
        revision=args.texture_revision,
        allow_patterns=[
            "config.json",
            "LICENSE",
            "NOTICE",
            "hunyuan3d-paint-v2-0/**/*.json",
            "hunyuan3d-paint-v2-0/**/*.txt",
            "hunyuan3d-paint-v2-0/**/*.py",
            "hunyuan3d-paint-v2-0/text_encoder/pytorch_model.bin",
            "hunyuan3d-paint-v2-0/unet/diffusion_pytorch_model.safetensors",
            "hunyuan3d-paint-v2-0/vae/diffusion_pytorch_model.safetensors",
            "hunyuan3d-delight-v2-0/**/*.json",
            "hunyuan3d-delight-v2-0/**/*.txt",
            "hunyuan3d-delight-v2-0/**/*.py",
            "hunyuan3d-delight-v2-0/**/*.safetensors",
        ],
    )


if __name__ == "__main__":
    main()
