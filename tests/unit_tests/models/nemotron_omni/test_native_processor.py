import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import transformers
from packaging.version import Version
from PIL import Image

from megatron.bridge.models.conversion.auto_bridge import _resolve_pretrained_wrapper_cls
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.nemotron_omni.hf_pretrained import Nemotron35VLPretrained


@pytest.mark.unit
def test_only_super_vl_selects_wrapper():
    assert (
        _resolve_pretrained_wrapper_cls(SimpleNamespace(architectures=["NemotronH_Omni_Reasoning_V3"]))
        is Nemotron35VLPretrained
    )
    for architecture in ("NemotronHForCausalLM", "NemotronH_Nano_Omni_Reasoning_V3"):
        assert _resolve_pretrained_wrapper_cls(SimpleNamespace(architectures=[architecture])) is PreTrainedCausalLM


@pytest.mark.unit
@pytest.mark.parametrize("version", ["5.17.0", "5.19.0", "5.20.0"])
def test_remote_processor_unchanged(version):
    wrapper = Nemotron35VLPretrained("unused", trust_remote_code=True)
    with (
        patch.object(transformers, "__version__", version),
        patch.object(PreTrainedCausalLM, "_load_processor") as load,
    ):
        assert wrapper.processor is load.return_value
        load.assert_called_once_with()


@pytest.mark.unit
def test_native_processor_version_bounds():
    wrapper = Nemotron35VLPretrained("unused", trust_remote_code=False)
    with (
        patch.object(transformers, "__version__", "5.18.0"),
        pytest.raises(ValueError, match="trust_remote_code=True"),
    ):
        wrapper.processor
    with (
        patch.object(transformers, "__version__", "5.20.0"),
        patch.object(transformers.AutoProcessor, "from_pretrained") as load,
    ):
        assert wrapper.processor is load.return_value
        load.assert_called_once_with("unused", trust_remote_code=False)


@pytest.mark.unit
def test_native_audio_config_is_not_discarded():
    wrapper = Nemotron35VLPretrained("unused", trust_remote_code=False, revision="pinned-revision")
    wrapper.config = SimpleNamespace(audio_config={"model_type": "parakeet"})
    with (
        patch.object(transformers, "__version__", "5.19.0"),
        patch.object(
            transformers.AutoProcessor, "from_pretrained", side_effect=ValueError("invalid processor")
        ) as load,
    ):
        with pytest.raises(ValueError, match="invalid processor"):
            wrapper.processor
        load.assert_called_once_with("unused", trust_remote_code=False, revision="pinned-revision")


@pytest.fixture
def processor_source(tmp_path):
    if Version(transformers.__version__).base_version != "5.19.0":
        pytest.skip("Compatibility adapter specifically targets Transformers 5.19.0")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "<image>": 1, "<img>": 2, "</img>": 3}, unk_token="[UNK]")),
        unk_token="[UNK]",
        additional_special_tokens=["<image>", "<img>", "</img>"],
    )
    tokenizer.save_pretrained(tmp_path)
    image_config = {
        "image_processor_type": "NemotronH_Omni_Reasoning_V3ImageProcessor",
        "_downsample_factor": 2,
        "downsample_ratio": 0.5,
        "patch_size": 16,
        "min_num_patches": 16,
        "max_num_patches": 64,
        "video_target_num_patches": 16,
        "norm_mean": [0.48145466, 0.4578275, 0.40821073],
        "norm_std": [0.26862954, 0.26130258, 0.27577711],
    }
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(image_config))
    (tmp_path / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "NemotronH_Omni_Reasoning_V3Processor",
                "video_temporal_patch_dim": 2,
                "image_processor": image_config,
            }
        )
    )
    return tmp_path


@pytest.mark.unit
def test_native_processor_load_save_reload(processor_source, tmp_path):
    wrapper = Nemotron35VLPretrained(processor_source)
    wrapper.config = SimpleNamespace(audio_config=None)
    before = (processor_source / "preprocessor_config.json").read_bytes()
    processor = wrapper.processor
    assert not hasattr(processor, "feature_extractor")
    assert processor.video_processor._downsample_factor == 2
    image = Image.fromarray(np.zeros((48, 80, 3), dtype=np.uint8))
    inputs = processor(text="<image>", images=[image], return_tensors="pt")
    grid = inputs["image_grid_hw"][0]
    assert (inputs["input_ids"] == 1).sum().item() == grid.prod().item() // 4
    output = tmp_path / "export"
    processor.save_pretrained(output)
    assert (output / "processor_config.json").is_file()
    reloaded = Nemotron35VLPretrained(output)
    reloaded.config = SimpleNamespace(audio_config=None)
    result = reloaded.processor(text="<image>", images=[image], return_tensors="pt")
    for key in inputs:
        torch.testing.assert_close(inputs[key], result[key], rtol=0, atol=0)
    video = np.zeros((2, 48, 80, 3), dtype=np.uint8)
    video_inputs = processor(text="<video>", videos=[video], return_tensors="pt", do_sample_frames=False)
    video_result = reloaded.processor(text="<video>", videos=[video], return_tensors="pt", do_sample_frames=False)
    for key in video_inputs:
        torch.testing.assert_close(video_inputs[key], video_result[key], rtol=0, atol=0)
    assert (processor_source / "preprocessor_config.json").read_bytes() == before
    with pytest.raises(ValueError, match="no audio encoder"):
        processor(text="<so_embedding>", audio=np.zeros(1600))
