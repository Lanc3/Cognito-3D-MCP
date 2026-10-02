"""Model preparation tests use mocked Hub calls and tiny local files only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare-models.py"
_SPEC = importlib.util.spec_from_file_location("prepare_models", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
prepare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare)
TOKEN = "mock-private-token"
REVISION = "a" * 40


def fake_hub(cache: Path) -> Mock:
    hub = Mock()
    hub.get_token.return_value = TOKEN
    siblings = [
        SimpleNamespace(rfilename=name)
        for name in ("config.yaml", "model.safetensors", "LICENSE.md", "NOTICE", "README.md")
    ]
    siblings[1].size = 7
    siblings[1].lfs = {"sha256": "expected-test-digest"}
    hub.HfApi.return_value.model_info.return_value = SimpleNamespace(
        sha=REVISION, siblings=siblings
    )

    def snapshot(**kwargs):
        root = (
            cache
            / "hub"
            / ("models--" + kwargs["repo_id"].replace("/", "--"))
            / "snapshots"
            / REVISION
        )
        root.mkdir(parents=True, exist_ok=True)
        for name in kwargs["allow_patterns"]:
            (root / name).write_bytes(b"fixture")
        return str(root)

    hub.snapshot_download.side_effect = snapshot
    return hub


def test_existing_credentials_skip_login_and_download_selected_sf3d_files(tmp_path, capsys):
    hub = fake_hub(tmp_path)
    report = prepare.prepare_models("sf3d", tmp_path, hub=hub, interactive=True)
    hub.login.assert_not_called()
    hub.HfApi.assert_called_once_with(token=TOKEN)
    hub.HfApi.return_value.model_info.assert_called_once_with(
        "stabilityai/stable-fast-3d", files_metadata=True
    )
    request = hub.snapshot_download.call_args.kwargs
    assert request["repo_id"] == "stabilityai/stable-fast-3d"
    assert request["cache_dir"] == tmp_path / "hub"
    assert request["revision"] == REVISION
    assert set(request["allow_patterns"]) == {
        "config.yaml",
        "model.safetensors",
        "LICENSE.md",
        "NOTICE",
        "README.md",
    }
    assert report["model_files_ready"] and not report["gpu_generation_verified"]
    assert (
        tmp_path / "hub" / "models--stabilityai--stable-fast-3d" / "refs" / "main"
    ).read_text() == REVISION
    assert TOKEN not in capsys.readouterr().out
    assert TOKEN not in (tmp_path / "model-installation.json").read_text()


def test_missing_credentials_use_sdk_local_login_once(tmp_path):
    hub = fake_hub(tmp_path)
    hub.get_token.side_effect = [None, TOKEN]
    prepare.prepare_models("sf3d", tmp_path, hub=hub, interactive=True)
    hub.login.assert_called_once_with(add_to_git_credential=False)


def test_noninteractive_missing_credentials_refuse_download(tmp_path):
    hub = fake_hub(tmp_path)
    hub.get_token.return_value = None
    with pytest.raises(prepare.ModelPreparationError, match="non-interactive"):
        prepare.prepare_models("sf3d", tmp_path, hub=hub)
    hub.login.assert_not_called()
    hub.HfApi.assert_not_called()
    hub.snapshot_download.assert_not_called()


@pytest.mark.parametrize("code, expected", [(401, "rejected"), (403, "gated model")])
def test_rejected_access_preserves_credential_and_redacts_sdk_error(
    tmp_path, capsys, code, expected
):
    hub = fake_hub(tmp_path)
    error = RuntimeError(f"HTTP credentials included {TOKEN}")
    error.response = SimpleNamespace(status_code=code)
    hub.hf_hub_download.side_effect = error
    with pytest.raises(prepare.ModelPreparationError, match=expected) as caught:
        prepare.prepare_models("sf3d", tmp_path, hub=hub, interactive=True)
    assert TOKEN not in str(caught.value)
    assert caught.value.__suppress_context__
    assert "https://huggingface.co/stabilityai/stable-fast-3d" in str(caught.value)
    hub.login.assert_not_called()
    hub.snapshot_download.assert_not_called()
    assert TOKEN not in capsys.readouterr().out


def test_login_error_redacts_token_and_stops_before_model_requests(tmp_path):
    hub = fake_hub(tmp_path)
    hub.get_token.return_value = None
    hub.login.side_effect = ValueError(TOKEN)
    with pytest.raises(prepare.ModelPreparationError, match="login failed") as caught:
        prepare.prepare_models("sf3d", tmp_path, hub=hub, interactive=True)
    assert TOKEN not in str(caught.value)
    assert caught.value.__suppress_context__
    hub.HfApi.assert_not_called()


def test_download_failure_redacts_sdk_error_and_writes_no_ready_manifest(tmp_path):
    hub = fake_hub(tmp_path)
    hub.snapshot_download.side_effect = RuntimeError(f"download failed with {TOKEN}")
    with pytest.raises(prepare.ModelPreparationError) as caught:
        prepare.prepare_models("sf3d", tmp_path, hub=hub)
    assert TOKEN not in str(caught.value)
    assert not (tmp_path / "model-installation.json").exists()


def test_spar3d_uses_selected_cache_and_pinned_resumable_checkpoint(tmp_path):
    hub = fake_hub(tmp_path)
    blob = tmp_path / "hub" / "blob"
    partial = tmp_path / "hub" / "blob.incomplete"
    pointer = tmp_path / "hub" / "pointer"
    downloader = Mock(EXPECTED_LFS_SHA256="expected-test-digest")
    downloader.cache_paths.return_value = blob, partial, pointer

    def transfer(token, target, size, *, revision):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")

    downloader.download.side_effect = transfer
    prepare.prepare_models("spar3d", tmp_path, hub=hub, spar_downloader=downloader)
    downloader.cache_paths.assert_called_once_with(
        tmp_path / "hub", REVISION, "expected-test-digest"
    )
    downloader.download.assert_called_once_with(TOKEN, partial, 7, revision=REVISION)
    downloader.verify.assert_called_once_with(partial, "expected-test-digest")
    downloader.create_cache_pointer.assert_called_once_with(blob, pointer)
    assert hub.snapshot_download.call_args.kwargs["repo_id"] == "stabilityai/stable-point-aware-3d"
    assert json.loads((tmp_path / "model-installation.json").read_text())["backend"] == "spar3d"


def test_spar3d_changed_checkpoint_is_refused_before_weight_transfer(tmp_path):
    hub = fake_hub(tmp_path)
    downloader = Mock(EXPECTED_LFS_SHA256="different-digest")
    with pytest.raises(prepare.ModelPreparationError, match="checkpoint identity"):
        prepare.prepare_models("spar3d", tmp_path, hub=hub, spar_downloader=downloader)
    downloader.download.assert_not_called()
    hub.snapshot_download.assert_not_called()


def test_incomplete_snapshot_is_not_marked_ready(tmp_path):
    hub = fake_hub(tmp_path)
    hub.snapshot_download.return_value = str(tmp_path)
    hub.snapshot_download.side_effect = None
    with pytest.raises(prepare.ModelPreparationError, match="incomplete"):
        prepare.prepare_models("sf3d", tmp_path, hub=hub)
    assert not (tmp_path / "model-installation.json").exists()


def test_backend_cached_token_path_is_reused_without_writing_tokens(tmp_path):
    cache = tmp_path / "models"
    cache.mkdir()
    cached = cache / "token"
    cached.write_text(TOKEN)
    shared = tmp_path / "shared" / "token"
    assert prepare.select_token_path(cache, shared) == cached
    assert cached.read_text() == TOKEN
    assert not shared.exists()
    shared.parent.mkdir()
    shared.write_text("other-credential")
    assert prepare.select_token_path(cache, shared) == shared
