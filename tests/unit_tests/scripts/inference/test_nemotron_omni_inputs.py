# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tensor tests for shared native-processor Omni inference inputs."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]
HELPER = ROOT / "src/megatron/bridge/models/nemotron_omni/inference_inputs.py"
SPEC = importlib.util.spec_from_file_location("omni_native_inputs_under_test", HELPER)
assert SPEC is not None and SPEC.loader is not None
helper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = helper
UTILS_NAME = "megatron.bridge.models.nemotron_omni.nemotron_omni_utils"
UTILS_SPEC = importlib.util.spec_from_file_location(UTILS_NAME, HELPER.with_name("nemotron_omni_utils.py"))
assert UTILS_SPEC is not None and UTILS_SPEC.loader is not None
utils = importlib.util.module_from_spec(UTILS_SPEC)
UTILS_SPEC.loader.exec_module(utils)
with patch.dict(sys.modules, {UTILS_NAME: utils}):
    SPEC.loader.exec_module(helper)
pytestmark = pytest.mark.unit


class Processor:
    image_token = "<image>"
    video_token = "<video>"
    image_token_id = 18
    video_temporal_patch_dim = 2
    image_processor = SimpleNamespace(patch_size=16)

    def __init__(self, frames):
        self.frames = frames
        self.calls = []
        self.content = None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        self.content = messages[0]["content"]
        return f"native:{self.content}"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        video = "videos" in kwargs
        counts = [frame.shape[-2] * frame.shape[-1] // (16 * 16 * 4) for frame in self.frames]
        count = ((len(counts) + 1) // 2) * counts[0] if video else sum(counts)
        ids = torch.tensor([[5, *([18] * count), 7]])
        pixels = torch.stack(self.frames) if len({tuple(frame.shape) for frame in self.frames}) == 1 else self.frames
        self.output = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values_videos" if video else "pixel_values": pixels,
            "imgs_sizes": [list(frame.shape[-2:]) for frame in self.frames],
            "num_tokens": counts,
            "num_patches": [1] * len(self.frames),
        }
        return self.output


def frame(height, width, offset=0):
    return torch.arange(3 * height * width, dtype=torch.float32).reshape(3, height, width) + offset


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (32, 32)])
def test_image_pixels_ids_and_post_resize_grid_are_preserved(shape):
    pixels = frame(*shape)
    processor = Processor([pixels])
    actual = helper.prepare_nemotron_omni_inputs(processor, prompt="describe", images=[object()])
    assert len(processor.calls) == 1
    assert processor.content == "<image>\ndescribe"
    assert actual.hf["input_ids"] is processor.output["input_ids"]
    assert actual.hf["pixel_values"] is processor.output["pixel_values"]
    assert actual.bridge["imgs_sizes"].tolist() == [list(shape)]
    assert actual.bridge["num_frames"].tolist() == [1]
    # Independent spatial unfolding oracle, including orientation/order.
    expected = pixels.unfold(1, 16, 16).unfold(2, 16, 16).permute(1, 2, 0, 3, 4).reshape(1, -1, 768)
    assert torch.equal(actual.bridge["pixel_values"], expected)
    assert "imgs_sizes" not in actual.hf


def test_heterogeneous_images_stay_independent_not_temporal():
    processor = Processor([frame(32, 64), frame(64, 32, 100)])
    actual = helper.prepare_nemotron_omni_inputs(processor, prompt="compare", images=[object(), object()])
    assert actual.bridge["num_frames"].tolist() == [1, 1]
    assert actual.bridge["imgs_sizes"].tolist() == [[32, 64], [64, 32]]
    assert actual.hf["pixel_values"] is processor.frames
    assert processor.content == "<image>\n<image>\ncompare"


def test_native_batch_feature_keeps_heterogeneous_pixels_ragged():
    from transformers.feature_extraction_utils import BatchFeature

    class BatchFeatureProcessor(Processor):
        def __call__(self, **kwargs):
            output = super().__call__(**kwargs)
            output["input_ids"] = output["input_ids"].tolist()
            output["attention_mask"] = output["attention_mask"].tolist()
            return BatchFeature(output, tensor_type=kwargs["return_tensors"])

    processor = BatchFeatureProcessor([frame(32, 64), frame(64, 32)])
    # Demonstrate the actual native conversion failure that the adapter avoids.
    with pytest.raises(ValueError):
        processor(images=[object(), object()], return_tensors="pt")
    actual = helper.prepare_nemotron_omni_inputs(processor, prompt="compare", images=[object(), object()])
    assert processor.calls[-1]["return_tensors"] is None
    assert actual.hf["pixel_values"] is processor.frames
    assert actual.hf["input_ids"].dtype == torch.long
    assert actual.hf["attention_mask"].shape == actual.hf["input_ids"].shape
    assert actual.bridge["imgs_sizes"].tolist() == [[32, 64], [64, 32]]


@pytest.mark.parametrize("count,model_count", [(1, 2), (3, 3), (4, 4)])
def test_video_native_prompt_and_tubelets_with_single_and_odd_frames(count, model_count):
    processor = Processor([frame(32, 64, i * 100) for i in range(count)])
    metadata = SimpleNamespace(fps=30.0, frames_indices=list(range(count)))
    actual = helper.prepare_nemotron_omni_inputs(
        processor, prompt="describe", video_frames=[object()] * count, video_metadata=metadata
    )
    assert len(processor.calls) == 1
    assert processor.calls[0]["videos_kwargs"]["video_metadata"] is metadata
    assert processor.content == "<video>\ndescribe"
    assert actual.hf["pixel_values_videos"].shape[0] == count
    assert actual.bridge["num_frames"].tolist() == [model_count]
    assert actual.bridge["imgs_sizes"].tolist() == [[32, 64]] * model_count
    assert actual.bridge["pixel_values"].shape == (1, model_count * 8, 768)
    if count == 1:
        assert torch.equal(actual.bridge["pixel_values"][:, :8], actual.bridge["pixel_values"][:, 8:])
    assert int((actual.hf["input_ids"] == 18).sum()) == ((count + 1) // 2) * 2


@pytest.mark.parametrize("corruption", ["sizes", "counts", "anchors"])
def test_inconsistent_native_metadata_fails_closed(corruption):
    class BadProcessor(Processor):
        def __call__(self, **kwargs):
            output = super().__call__(**kwargs)
            if corruption == "sizes":
                output["imgs_sizes"] = [[64, 32]]
            elif corruption == "counts":
                output["num_tokens"] = [99]
            else:
                output["input_ids"] = torch.tensor([[5, 18, 7]])
            return output

    with pytest.raises(ValueError):
        helper.prepare_nemotron_omni_inputs(BadProcessor([frame(32, 64)]), prompt="describe", images=[object()])


def test_empty_or_mixed_media_rejected():
    for kwargs in ({}, {"video_frames": []}, {"images": [object()], "video_frames": [object()]}):
        with pytest.raises(ValueError, match="exclusively"):
            helper.prepare_nemotron_omni_inputs(Processor([]), prompt="describe", **kwargs)


@pytest.mark.parametrize(
    "name", ["nemotron_h_omni", "NemotronH_Nano_Omni_Reasoning_V3", "NemotronH_Super_Omni_Reasoning_V3"]
)
def test_registered_omni_family_detection(name):
    assert helper.is_nemotron_omni(SimpleNamespace(model_type=name))
    assert helper.is_nemotron_omni(SimpleNamespace(architectures=[name]))
    assert not helper.is_nemotron_omni(SimpleNamespace(model_type="qwen3_vl"))


def test_reference_metadata_changes_with_media_or_source_not_only_ids():
    inputs = helper.prepare_nemotron_omni_inputs(Processor([frame(32, 64)]), prompt="describe", images=[object()])
    before = helper.nemotron_omni_reference_metadata(inputs, model="model", revision="revision")
    inputs.hf["pixel_values"][0, 0, 0, 0] += 1
    after = helper.nemotron_omni_reference_metadata(inputs, model="model", revision="revision")
    assert before != after
    assert before["tensors"]["input_ids"] == after["tensors"]["input_ids"]
    assert after != helper.nemotron_omni_reference_metadata(inputs, model="model", revision="other")


def test_video_reference_requires_explicit_unpruned_metadata():
    inputs = helper.prepare_nemotron_omni_inputs(
        Processor([frame(32, 64)] * 3), prompt="describe", video_frames=[object()] * 3
    )
    for rate in (None, 0.7):
        with pytest.raises(ValueError, match="unpruned"):
            helper.nemotron_omni_reference_metadata(
                inputs, model="model", revision="revision", video_pruning_rate=rate
            )
    assert (
        helper.nemotron_omni_reference_metadata(inputs, model="model", revision="revision", video_pruning_rate=0.0)[
            "video_pruning_rate"
        ]
        == 0.0
    )


def load_functions(path, names, **globals_):
    """Exercise actual helper bodies without importing distributed GPU dependencies."""
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(selected) == len(names)
    namespace = {"torch": torch, **globals_}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "relative", ["scripts/inference/vlm_generation.py", "examples/conversion/compare_hf_and_megatron/compare.py"]
)
def test_public_iterator_forwards_omni_metadata_without_changing_qwen_fields(relative):
    module = load_functions(ROOT / relative, ["SingleBatchIterator", "vlm_forward_step"])
    ids = torch.tensor([[1, 18, 2]])
    pixels = torch.randn(1, 8, 768)
    sizes, counts = torch.tensor([[32, 64]]), torch.tensor([1])
    seen = {}

    def model(**kwargs):
        seen.update(kwargs)
        return torch.zeros(1, 3, 4)

    iterator = module["SingleBatchIterator"](ids, ids, None, pixel_values=pixels, imgs_sizes=sizes, num_frames=counts)
    module["vlm_forward_step"](iterator, model)
    assert seen["pixel_values"] is pixels
    assert seen["imgs_sizes"] is sizes
    assert seen["num_frames"] is counts
    assert seen["attention_mask"] is None
    seen.clear()
    grid = torch.tensor([[1, 2, 4]])
    iterator = module["SingleBatchIterator"](ids, ids, None, pixel_values=pixels, image_grid_thw=grid)
    module["vlm_forward_step"](iterator, model)
    assert seen["image_grid_thw"] is grid
    assert "imgs_sizes" not in seen and "num_frames" not in seen


def test_video_hf_forward_keeps_outer_vision_model():
    path = ROOT / "examples/conversion/compare_hf_and_megatron/compare.py"
    module = load_functions(path, ["_get_hf_forward_model"], print_rank_0=lambda *args: None)
    model = SimpleNamespace(language_model=torch.nn.Identity())
    assert module["_get_hf_forward_model"](model, None) is model.language_model
    assert module["_get_hf_forward_model"](model, None, torch.zeros(4, 3, 32, 64)) is model


def test_hf_video_forward_receives_native_pixels_not_bridge_patches():
    path = ROOT / "examples/conversion/compare_hf_and_megatron/compare.py"
    module = load_functions(
        path,
        ["_get_hf_forward_model", "_run_hf_inference"],
        _is_rank_0=lambda: True,
        print_rank_0=lambda *args: None,
    )
    seen = {}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.language_model = torch.nn.Identity()

        def forward(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(logits=torch.ones(1, 3, 8))

    pixels = torch.zeros(3, 3, 32, 64)
    module["_run_hf_inference"](
        Model(),
        torch.tensor([[1, 18, 2]]),
        None,
        None,
        SimpleNamespace(decode=lambda ids: "token"),
        pixel_values_videos=pixels,
    )
    assert seen["pixel_values_videos"] is pixels
    assert "pixel_values" not in seen and "imgs_sizes" not in seen and "num_frames" not in seen


def test_saved_reference_rejects_absent_or_changed_media_binding(tmp_path):
    path = ROOT / "examples/conversion/compare_hf_and_megatron/compare.py"
    module = load_functions(path, ["_load_hf_reference_logits"], _is_rank_0=lambda: True)
    ids = torch.tensor([[1, 18, 2]])
    saved = tmp_path / "reference.pt"
    for metadata in (None, {"pixels": "different"}):
        torch.save({"input_ids": ids, "logits": torch.ones(4), "metadata": metadata}, saved)
        with pytest.raises(ValueError, match="native media inputs"):
            module["_load_hf_reference_logits"](saved, ids, None, expected_metadata={"pixels": "expected"})


def test_hf_loading_pruning_override_is_explicit_and_preconstruction():
    from unittest.mock import MagicMock

    path = ROOT / "examples/conversion/compare_hf_and_megatron/compare.py"
    config = SimpleNamespace(video_pruning_rate=0.7)
    auto_config = SimpleNamespace(from_pretrained=MagicMock(return_value=config))
    model_class = SimpleNamespace(__name__="FakeHFModel", from_pretrained=MagicMock())
    module = load_functions(
        path,
        ["_load_hf_model"],
        _is_rank_0=lambda: True,
        print_rank_0=lambda *args: None,
        get_model_class=lambda *args: model_class,
        is_safe_repo=lambda **kwargs: True,
        _hf_revision_kwargs=lambda revision: {"revision": revision},
        AutoConfig=auto_config,
    )
    args = SimpleNamespace(
        model_class=None,
        trust_remote_code=True,
        hf_model_path="model",
        hf_revision="revision",
        hf_device="cpu",
        enable_debug_hooks=False,
        disable_hf_video_pruning=False,
    )
    module["_load_hf_model"](args, True)
    assert "config" not in model_class.from_pretrained.call_args.kwargs
    assert config.video_pruning_rate == 0.7
    args.disable_hf_video_pruning = True
    module["_load_hf_model"](args, True)
    assert model_class.from_pretrained.call_args.kwargs["config"] is config
    assert config.video_pruning_rate == 0.0
