from __future__ import annotations

import threading
from pathlib import Path

import pytest

from codex_3d_mcp.errors import Codex3DError, GenerationCancelled
from codex_3d_mcp.trellis.runtime import GpuLease, _multipart


def test_multipart_contains_generation_fields_and_image(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"PNG-CONTENT")
    body, content_type = _multipart(image, {"seed": "42", "resolution": "1024"})
    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="seed"' in body
    assert b"1024" in body
    assert b'name="image"' in body
    assert b"PNG-CONTENT" in body


def test_gpu_lease_is_reusable(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path / "gpu.lock")
    lease.acquire(timeout=1, cancel_event=threading.Event())
    lease.release()
    lease.acquire(timeout=1)
    lease.release()


def test_double_acquire_preserves_owned_lock(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path / "gpu.lock")
    competitor = GpuLease(lease.path)
    lease.acquire(timeout=1)
    try:
        with pytest.raises(Codex3DError, match="already in use"):
            lease.acquire(timeout=0)
        with pytest.raises(Codex3DError, match="Timed out"):
            competitor.acquire(timeout=0)
    finally:
        lease.release()
    competitor.acquire(timeout=1)
    competitor.release()


def test_cancelled_gpu_wait_does_not_leak_file_lock(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path / "gpu.lock")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(GenerationCancelled):
        lease.acquire(timeout=1, cancel_event=cancel)
    lease.acquire(timeout=1)
    lease.release()
