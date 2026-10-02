"""Guide local Hugging Face authentication and cache legacy 3D model files.

Tokens are entered through the SDK's terminal login, never printed or written to
installer manifests. Existing credentials are reused; rejected credentials are
left in place for the user to correct. No GPU model is loaded by this helper.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODELS = {
    "sf3d": "stabilityai/stable-fast-3d",
    "spar3d": "stabilityai/stable-point-aware-3d",
}
REQUIRED_FILES = ("config.yaml", "model.safetensors")


class ModelPreparationError(RuntimeError):
    """A user-actionable error that never includes an underlying exception/token."""


def select_token_path(cache_root: Path, preferred: Path) -> Path:
    """Reuse a backend's existing credential file if the shared location is absent."""
    if not preferred.is_file() and (cache_root / "token").is_file():
        return cache_root / "token"
    return preferred


def _safe_failure(error: Exception, repo: str) -> ModelPreparationError:
    code = getattr(getattr(error, "response", None), "status_code", None)
    if code == 401:
        detail = (
            "Hugging Face rejected the existing credentials. They were preserved. "
            "Check the token's read permissions and account, then authenticate locally "
            "with huggingface-cli login (or hf auth login) and rerun the installer."
        )
    elif code == 403 or type(error).__name__ == "GatedRepoError":
        detail = (
            "The authenticated account cannot download this gated model. "
            "Open its model card, accept the model terms, request access, and wait "
            "for approval before rerunning the installer. Existing credentials were preserved."
        )
    else:
        detail = (
            "Model preparation could not finish. Check network access, available disk "
            "space, and Hugging Face availability, then rerun using the same model cache."
        )
    return ModelPreparationError(f"{detail} Model card: https://huggingface.co/{repo}")


def authenticate(hub: Any, *, interactive: bool) -> str:
    try:
        token = hub.get_token()
    except Exception:
        raise ModelPreparationError(
            "Could not read Hugging Face credentials. Check HF_TOKEN_PATH and file permissions."
        ) from None
    if token:
        return token
    if not interactive:
        raise ModelPreparationError(
            "No Hugging Face credentials are available in this non-interactive session. "
            "Authenticate locally with huggingface-cli login (or hf auth login), or provide "
            "HF_TOKEN through the process environment, then rerun. Do not paste tokens into chat."
        )
    print("Hugging Face login is required. Enter credentials only in the local terminal.")
    print("Create a read token at https://huggingface.co/settings/tokens if prompted.")
    try:
        hub.login(add_to_git_credential=False)
        token = hub.get_token()
    except Exception:
        raise ModelPreparationError(
            "Hugging Face login failed. Check the account/token locally and rerun. "
            "No model files were downloaded."
        ) from None
    if not token:
        raise ModelPreparationError("Login did not supply a usable Hugging Face credential.")
    return token


def _spar_downloader() -> Any:
    path = Path(__file__).with_name("download-spar3d-model.py")
    spec = importlib.util.spec_from_file_location("spar_checkpoint_download", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_models(
    backend: str,
    cache_root: Path,
    *,
    hub: Any = None,
    interactive: bool = False,
    spar_downloader: Any = None,
) -> dict[str, Any]:
    repo = MODELS[backend]
    print(f"Review the model terms and request access: https://huggingface.co/{repo}")
    if hub is None:
        import huggingface_hub as hub
    token = authenticate(hub, interactive=interactive)
    cache = cache_root / "hub"
    try:
        info = hub.HfApi(token=token).model_info(repo, files_metadata=True)
        if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
            raise ModelPreparationError("Hugging Face returned an invalid model revision.")
        names = {sibling.rfilename for sibling in info.siblings}
        if not set(REQUIRED_FILES).issubset(names):
            raise ModelPreparationError("The model repository is missing required 3D model files.")
        # Verify gated access with the small configuration file before transferring weights.
        hub.hf_hub_download(
            repo_id=repo, filename="config.yaml", revision=info.sha, cache_dir=cache, token=token
        )
        if backend == "spar3d":
            downloader = spar_downloader or _spar_downloader()
            weight = next(item for item in info.siblings if item.rfilename == "model.safetensors")
            lfs = weight.lfs
            digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            if digest != downloader.EXPECTED_LFS_SHA256 or not weight.size:
                raise ModelPreparationError(
                    "SPAR3D checkpoint identity differs from the tested release; "
                    "installation stopped."
                )
            blob, partial, pointer = downloader.cache_paths(cache, info.sha, digest)
            if not blob.is_file() or blob.stat().st_size != weight.size:
                downloader.download(token, partial, weight.size, revision=info.sha)
                downloader.verify(partial, digest)
                partial.replace(blob)
            else:
                downloader.verify(blob, digest)
            downloader.create_cache_pointer(blob, pointer)
        notices = sorted(
            name
            for name in names
            if Path(name).name.lower().startswith(("license", "notice")) or name == "README.md"
        )
        snapshot = Path(
            hub.snapshot_download(
                repo_id=repo,
                revision=info.sha,
                cache_dir=cache,
                token=token,
                allow_patterns=[*REQUIRED_FILES, *notices],
            )
        )
        if not all(
            (snapshot / name).is_file() and (snapshot / name).stat().st_size > 0
            for name in REQUIRED_FILES
        ):
            raise ModelPreparationError(
                "Downloaded model snapshot is incomplete; rerun the installer."
            )
        # The vendor loaders request main. Bind that cache reference to the exact
        # authenticated revision just downloaded, without loading the model/GPU.
        references = cache / ("models--" + repo.replace("/", "--")) / "refs"
        references.mkdir(parents=True, exist_ok=True)
        (references / "main").write_text(info.sha, encoding="utf-8")
        report = {
            "backend": backend,
            "model_id": repo,
            "revision": info.sha,
            "cache_dir": str(cache),
            "snapshot": str(snapshot),
            "required_files": list(REQUIRED_FILES),
            "notice_files": notices,
            "token_path": os.getenv("HF_TOKEN_PATH"),
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "model_files_ready": True,
            "gpu_generation_verified": False,
        }
        (cache_root / "model-installation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    except ModelPreparationError:
        raise
    except Exception as error:
        raise _safe_failure(error, repo) from None
    print(f"{backend.upper()} model files are ready in {cache}.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(MODELS), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--token-path", type=Path, required=True)
    parser.add_argument("--non-interactive", action="store_true")
    args = parser.parse_args()
    # Hub constants are read at import time; establish storage before importing the SDK.
    os.environ["HF_HOME"] = str(args.cache_root)
    os.environ["HF_TOKEN_PATH"] = str(select_token_path(args.cache_root, args.token_path))
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    try:
        prepare_models(
            args.backend,
            args.cache_root,
            interactive=not args.non_interactive and sys.stdin.isatty(),
        )
    except ModelPreparationError as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Model preparation cancelled locally.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
