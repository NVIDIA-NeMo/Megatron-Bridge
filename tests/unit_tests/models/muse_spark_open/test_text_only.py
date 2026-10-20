# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Language-only checkpoint selection must remain lazy, complete, and reversible."""

import json
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from transformers import PretrainedConfig

from megatron.bridge.models.hf_pretrained import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource
from megatron.bridge.models.muse_spark_open.text_only import MuseSparkTextStateSource, select_text_pretrained


pytestmark = pytest.mark.unit


def test_projection_streams_decoder_and_separate_head_without_opening_perception(tmp_path):
    weights = {
        "model.language_model.embed_tokens.weight": torch.randn(8, 4),
        "model.language_model.norm.weight": torch.ones(4),
        "lm_head.weight": torch.randn(8, 4),
    }
    save_file(weights, tmp_path / "language.safetensors")
    weight_map = dict.fromkeys(weights, "language.safetensors")
    weight_map["model.vision_tower.weight"] = "deliberately-absent.safetensors"
    index = json.dumps({"weight_map": weight_map, "metadata": {}})
    (tmp_path / "model.safetensors.index.json").write_text(index)
    source = MuseSparkTextStateSource(tmp_path, revision=None, hub_kwargs={})
    expected = {key.replace("model.language_model.", "model."): value for key, value in weights.items()}
    assert set(source.keys()) == set(expected)
    for key, value in source.load_tensors(list(expected)).items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    with pytest.raises(KeyError, match="Not a selected language tensor"):
        source.load_tensors(["model.vision_tower.weight"])
    output = tmp_path / "export"
    source.save_generator(iter(expected.items()), output, strict=True)
    saved = SafeTensorsStateSource(output)
    assert set(saved.keys()) == set(expected)
    for key, value in expected.items():
        torch.testing.assert_close(saved[key], value, rtol=0, atol=0)
    assert (tmp_path / "model.safetensors.index.json").read_text() == index


def test_projection_preserves_parent_tying_and_artifacts_without_mutating_source(tmp_path):
    source = PreTrainedCausalLM(tmp_path, local_files_only=True)
    source.config = PretrainedConfig(
        text_config=PretrainedConfig(tie_word_embeddings=True), tie_word_embeddings=False, dtype="bfloat16"
    )
    source._tokenizer = Mock()
    source._generation_config = Mock()
    before = source.config.to_dict()
    text = select_text_pretrained(source)
    assert text is not source
    assert text._text_only
    assert not text.config.tie_word_embeddings
    assert text.config.architectures == ["MuseSparkForCausalLM"]
    assert not hasattr(text.config, "text_config")
    assert text._tokenizer is source._tokenizer
    assert text._generation_config is source._generation_config
    assert text._processor is None and text._image_processor is None
    assert text.custom_file_patterns == []
    assert source.config.to_dict() == before
    assert select_text_pretrained(text) is text
