"""Reliably cache SPAR3D's large checkpoint on Windows.

The huggingface-hub version pinned by upstream can stall before the first byte
on Windows, while hf-transfer has a Windows file-handle race. This downloader
uses the same authenticated resolve endpoint with resumable HTTP ranges and
then creates the standard Hugging Face cache pointer.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "stabilityai/stable-point-aware-3d"
WEIGHT_NAME = "model.safetensors"
EXPECTED_LFS_SHA256 = "62673b63fd9dad425e74b213ffe8501262d9621d5174310ac33017747be31f58"
CHUNK_SIZE = 8 * 1024 * 1024
RANGE_SIZE = 64 * 1024 * 1024
PROGRESS_INTERVAL = 256 * 1024 * 1024
MAX_RETRIES = 20


def token_from_env() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing from .env or the process environment.")
    return token


def model_metadata(token: str) -> tuple[str, int, str]:
    info = HfApi(token=token).model_info(REPO_ID, files_metadata=True)
    sibling = next(item for item in info.siblings if item.rfilename == WEIGHT_NAME)
    if sibling.lfs is None or sibling.lfs.sha256 != EXPECTED_LFS_SHA256:
        raise RuntimeError("SPAR3D checkpoint identity differs from the tested release.")
    if sibling.size is None:
        raise RuntimeError("Hugging Face did not report the checkpoint size.")
    return info.sha, sibling.size, sibling.lfs.sha256


def cache_paths(cache_dir: Path, commit: str, digest: str) -> tuple[Path, Path, Path]:
    repo_cache = cache_dir / "models--stabilityai--stable-point-aware-3d"
    blob = repo_cache / "blobs" / digest
    partial = blob.with_suffix(".incomplete")
    pointer = repo_cache / "snapshots" / commit / WEIGHT_NAME
    return blob, partial, pointer


def download(token: str, partial: Path, expected_size: int, *, revision: str = "main") -> None:
    partial.parent.mkdir(parents=True, exist_ok=True)
    resolve_url = f"https://huggingface.co/{REPO_ID}/resolve/{revision}/{WEIGHT_NAME}"
    last_reported = partial.stat().st_size - PROGRESS_INTERVAL if partial.exists() else 0
    failures = 0

    with requests.Session() as session:
        while True:
            start = partial.stat().st_size if partial.exists() else 0
            if start == expected_size:
                return
            if start > expected_size:
                partial.replace(partial.with_suffix(f".oversize-{int(time.time())}"))
                start = 0

            end = min(start + RANGE_SIZE - 1, expected_size - 1)
            headers = {
                "Authorization": f"Bearer {token}",
                "Range": f"bytes={start}-{end}",
            }
            try:
                with session.get(
                    resolve_url,
                    headers=headers,
                    stream=True,
                    timeout=(30, 120),
                ) as response:
                    if response.status_code in {401, 403}:
                        raise requests.HTTPError("Checkpoint access was denied.", response=response)
                    if response.status_code != 206:
                        raise RuntimeError(
                            f"CDN did not honor byte range (HTTP {response.status_code})."
                        )
                    response.raise_for_status()
                    with partial.open("ab") as output:
                        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                            if not chunk:
                                continue
                            output.write(chunk)
                            downloaded = output.tell()
                            if downloaded - last_reported >= PROGRESS_INTERVAL:
                                percent = 100 * downloaded / expected_size
                                print(
                                    f"Downloaded {downloaded / (1024**3):.2f} / "
                                    f"{expected_size / (1024**3):.2f} GiB ({percent:.1f}%)",
                                    flush=True,
                                )
                                last_reported = downloaded
                    if partial.stat().st_size != end + 1:
                        raise RuntimeError("CDN range ended before the requested byte count.")
                failures = 0
            except (OSError, requests.RequestException, RuntimeError) as exc:
                if getattr(getattr(exc, "response", None), "status_code", None) in {401, 403}:
                    raise
                failures += 1
                if failures >= MAX_RETRIES:
                    raise RuntimeError(
                        f"Checkpoint range failed {MAX_RETRIES} times: {exc}"
                    ) from exc
                print(
                    f"Transfer interrupted; resuming range (retry {failures}).",
                    flush=True,
                )
                time.sleep(min(failures * 2, 20))


def verify(path: Path, expected_digest: str) -> None:
    print("Verifying checkpoint SHA-256...", flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_digest:
        raise RuntimeError("Downloaded checkpoint failed SHA-256 verification.")


def create_cache_pointer(blob: Path, pointer: Path) -> None:
    pointer.parent.mkdir(parents=True, exist_ok=True)
    if pointer.exists() or pointer.is_symlink():
        pointer.unlink()
    relative_blob = Path("..") / ".." / "blobs" / blob.name
    try:
        pointer.symlink_to(relative_blob)
    except OSError:
        os.link(blob, pointer)


def remove_quarantined_partials(blob: Path) -> None:
    for partial in blob.parent.glob(f"{blob.name}.*-corrupt"):
        size_gib = partial.stat().st_size / (1024**3)
        partial.unlink()
        print(f"Removed corrupt partial ({size_gib:.2f} GiB): {partial.name}", flush=True)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)
    cache_dir = project_root / ".cache" / "huggingface-spar3d" / "hub"
    token = token_from_env()
    commit, expected_size, expected_digest = model_metadata(token)
    blob, partial, pointer = cache_paths(cache_dir, commit, expected_digest)

    if not blob.exists() or blob.stat().st_size != expected_size:
        download(token, partial, expected_size, revision=commit)
        verify(partial, expected_digest)
        partial.replace(blob)
    else:
        print("SPAR3D checkpoint is already cached.", flush=True)

    create_cache_pointer(blob, pointer)
    remove_quarantined_partials(blob)
    hf_hub_download(
        repo_id=REPO_ID,
        filename="config.yaml",
        cache_dir=cache_dir,
        token=token,
    )
    print(f"SPAR3D checkpoint ready: {pointer.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"SPAR3D checkpoint setup failed: {error}", file=sys.stderr)
        raise
