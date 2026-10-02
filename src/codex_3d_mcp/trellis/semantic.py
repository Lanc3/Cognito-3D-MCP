"""Pinned local DINOv2 semantic comparison for seed pairs."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PIL import Image


class DinoSemanticScorer:
    """Load the small encoder lazily on CPU and return cosine similarity."""

    model_id = "facebook/dinov2-small"
    revision = "ed25f3a31f01632728cabb09d1542f84ab7b0056"

    def __init__(self, cache_dir: Path, *, local_files_only: bool = True) -> None:
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._lock = threading.Lock()
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            os.environ.setdefault("HF_HOME", str(self.cache_dir))
            try:
                from transformers import AutoImageProcessor, AutoModel
            except ImportError as exc:
                raise RuntimeError(
                    "DINOv2 validation dependencies are missing; run setup-trellis.ps1."
                ) from exc
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_id,
                revision=self.revision,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModel.from_pretrained(
                self.model_id,
                revision=self.revision,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            ).eval()

    def similarity(self, first: Path, second: Path) -> float:
        self._load()
        assert self._processor is not None and self._model is not None
        import torch

        images = [Image.open(first).convert("RGB"), Image.open(second).convert("RGB")]
        batch = self._processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            embeddings = self._model(**batch).last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return float((embeddings[0] * embeddings[1]).sum().item())

    def available(self) -> tuple[bool, str | None]:
        try:
            self._load()
        except Exception as exc:
            return False, str(exc)
        return True, None
