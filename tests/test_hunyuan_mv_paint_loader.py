from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from codex_3d_mcp.hunyuan_mv import paint_loader


@pytest.fixture
def tiny_unet(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    module = ModuleType("paint_fixture_unet")
    constructed = []

    class TinyUNet(torch.nn.Module):
        __module__ = module.__name__
        add_untracked_buffer = False

        def __init__(self, base):
            super().__init__()
            self.unet = base
            self.dual = copy.deepcopy(base)
            if self.add_untracked_buffer:
                self.register_buffer("untracked", torch.ones(1), persistent=False)
            constructed.append(all(value.is_meta for value in self.parameters()))

        @staticmethod
        def from_pretrained(path, **kwargs):
            raise AssertionError("The original allocating loader must not run")

    module.UNet2DConditionModel = torch.nn.Linear
    module.UNet2p5DConditionModel = TinyUNet
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(paint_loader, "_record_stage", lambda proof, stage: None)
    (tmp_path / "config.json").write_text(json.dumps({"in_features": 2, "out_features": 2}))
    expected = TinyUNet(torch.nn.Linear(2, 2)).half()
    with torch.no_grad():
        for index, value in enumerate(expected.parameters()):
            value.fill_((index + 1) / 8)
    checkpoint = tmp_path / "diffusion_pytorch_model.bin"
    torch.save(expected.state_dict(), checkpoint)
    return torch, TinyUNet, expected, checkpoint, constructed


def test_owned_meta_loader_preserves_exact_half_weights_with_one_load(tiny_unet, monkeypatch):
    torch, cls, expected, checkpoint, constructed = tiny_unet
    original_descriptor = inspect.getattr_static(cls, "from_pretrained")
    original_load = torch.load
    load_calls = []

    def load(path, **kwargs):
        load_calls.append((path, kwargs))
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    proof = {"checkpoint_loads": 0}
    with paint_loader._patched_unet_loader(cls, checkpoint.parent, proof):
        actual = cls.from_pretrained(checkpoint.parent, torch_dtype=torch.float16)
        with pytest.raises(RuntimeError, match="exactly once"):
            cls.from_pretrained(checkpoint.parent, torch_dtype=torch.float16)
    assert inspect.getattr_static(cls, "from_pretrained") is original_descriptor
    assert constructed == [False, True]
    assert load_calls == [(checkpoint, {
        "map_location": "cpu", "weights_only": True, "mmap": False,
    })]
    assert proof["checkpoint_loads"] == 1
    assert proof["checkpoint_dtypes"] == ["torch.float16"]
    assert proof["remaining_meta_tensors"] == 0 and proof["loader_method_restored"]
    assert proof["strict_assignment"]
    assert all(torch.equal(value, actual.state_dict()[name])
               for name, value in expected.state_dict().items())


@pytest.mark.parametrize("failure", ["missing_weight", "untracked_meta_buffer", "wrong_path"])
def test_safe_loader_fails_closed_and_restores_descriptor(tiny_unet, failure):
    torch, cls, expected, checkpoint, _ = tiny_unet
    original_descriptor = inspect.getattr_static(cls, "from_pretrained")
    if failure == "missing_weight":
        state = expected.state_dict()
        del state["dual.bias"]
        torch.save(state, checkpoint)
        message = "Missing key"
    elif failure == "untracked_meta_buffer":
        cls.add_untracked_buffer = True
        message = "unmaterialized"
    else:
        message = "unpinned checkpoint"
    proof = {"checkpoint_loads": 0}
    with pytest.raises(RuntimeError, match=message):
        with paint_loader._patched_unet_loader(cls, checkpoint.parent, proof):
            path = checkpoint.parent / "other" if failure == "wrong_path" else checkpoint.parent
            cls.from_pretrained(path, torch_dtype=torch.float16)
    assert inspect.getattr_static(cls, "from_pretrained") is original_descriptor
    assert proof["loader_method_restored"]
    if failure == "untracked_meta_buffer":
        assert proof["remaining_meta_tensors"] == 1


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_pipeline_uses_exact_dynamic_class_and_requires_gpu_tensors(tmp_path, monkeypatch, device):
    torch = pytest.importorskip("torch")
    subfolder = "pinned-paint"
    directory = tmp_path / subfolder / "unet"
    directory.mkdir(parents=True)
    for name in ("modules.py", "config.json", "diffusion_pytorch_model.bin"):
        (directory / name).write_bytes(b"fixture")
    events = []

    class PinnedUNet:
        @staticmethod
        def from_pretrained(path, **kwargs):
            raise AssertionError("Original loader called")

    original_descriptor = inspect.getattr_static(PinnedUNet, "from_pretrained")
    tensor = SimpleNamespace(is_meta=False, device=torch.device(device))
    component = SimpleNamespace(
        named_parameters=lambda: iter([("weight", tensor)]), named_buffers=lambda: iter([]),
    )
    components = SimpleNamespace(unet=component, vae=component, text_encoder=component)
    pipeline = SimpleNamespace(models={
        name: SimpleNamespace(pipeline=components) for name in ("delight_model", "multiview_model")
    })

    def resolve(path, **kwargs):
        assert Path(path) == directory
        assert kwargs == {"module_file": "modules.py", "class_name": "UNet2p5DConditionModel"}
        events.append("resolve")
        return PinnedUNet

    def owned_load(cls, path, dtype, proof):
        assert cls is PinnedUNet and path == directory and dtype == torch.float16
        proof.update(checkpoint_loads=1, remaining_meta_tensors=0)
        events.append("owned_load")
        return component

    def from_pretrained(path, **kwargs):
        assert Path(path) == tmp_path and kwargs == {"subfolder": subfolder}
        assert PinnedUNet.from_pretrained(directory, torch_dtype=torch.float16) is component
        return pipeline

    for module_name in (
        "diffusers", "diffusers.utils", "diffusers.utils.dynamic_modules_utils",
        "hy3dgen", "hy3dgen.texgen",
    ):
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    sys.modules["diffusers.utils.dynamic_modules_utils"].get_class_from_dynamic_module = resolve
    sys.modules["hy3dgen.texgen"].Hunyuan3DPaintPipeline = SimpleNamespace(
        from_pretrained=from_pretrained
    )
    monkeypatch.setattr(paint_loader, "_load_unet", owned_load)
    monkeypatch.setattr(paint_loader, "_record_stage", lambda proof, stage: events.append(stage))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("synchronize"))
    if device == "cpu":
        with pytest.raises(RuntimeError, match="not materialized on CUDA"):
            paint_loader.load_paint_pipeline(tmp_path, subfolder=subfolder)
        assert "synchronize" not in events
    else:
        actual, proof = paint_loader.load_paint_pipeline(tmp_path, subfolder=subfolder)
        assert actual is pipeline and proof["ready"]
        assert proof["checkpoint"] == str(directory / "diffusion_pytorch_model.bin")
        assert proof["module_path"] == str(directory / "modules.py")
        assert proof["checkpoint_mmap"] is False and proof["cpu_offload"] is False
        assert proof["devices"] == ["cuda:0"] and proof["device"] == "cuda"
        assert proof["loader_method_restored"] and proof["cuda_synchronized"]
    assert events.count("owned_load") == 1
    assert inspect.getattr_static(PinnedUNet, "from_pretrained") is original_descriptor
