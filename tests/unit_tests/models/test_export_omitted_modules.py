# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict export may exclude declared modules, never missing retained weights."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest
import torch
from safetensors.torch import save_file

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource


@pytest.mark.parametrize("distributed_save", [False, True])
@pytest.mark.parametrize("omit_retained_weight,declare_omitted_module", [(False, True), (True, True), (False, False)])
def test_strict_shared_shard_excludes_only_declared_module(
    tmp_path, distributed_save, omit_retained_weight, declare_omitted_module
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    weights = {"llm.weight": torch.ones(2), "apm.weight": torch.full((2,), 2.0), "tts.weight": torch.full((2,), 3.0)}
    shard = "model-00001-of-00001.safetensors"
    save_file(weights, source_dir / shard)
    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in weights}})
    )
    selected = {key: value for key, value in weights.items() if not key.startswith("tts.")}
    if omit_retained_weight:
        selected.pop("apm.weight")

    class UnderstandingBridge:
        HF_EXPORT_IGNORED_SOURCE_PREFIXES = ("tts.",) if declare_omitted_module else ()

        def stream_weights_megatron_to_hf(self, *args, **kwargs):
            return iter(selected.items())

    bridge = object.__new__(AutoBridge)
    bridge.hf_pretrained = SimpleNamespace(
        config=SimpleNamespace(), state=SimpleNamespace(source=SafeTensorsStateSource(source_dir))
    )
    output = tmp_path / "output"
    with (
        patch.object(AutoBridge, "_model_bridge", new_callable=PropertyMock, return_value=UnderstandingBridge()),
        patch.object(AutoBridge, "_get_model_instance", return_value=SimpleNamespace(config=SimpleNamespace())),
        patch("modelopt.torch.quantization.utils.is_quantized", return_value=False),
    ):
        if omit_retained_weight or not declare_omitted_module:
            with pytest.raises(RuntimeError, match="not written|Missing"):
                bridge.save_hf_weights([Mock()], output, strict=True, distributed_save=distributed_save)
        else:
            bridge.save_hf_weights([Mock()], output, strict=True, distributed_save=distributed_save)
            actual = SafeTensorsStateSource(output).load_tensors(list(selected))
            assert set(actual) == {"llm.weight", "apm.weight"}
            torch.testing.assert_close(actual, selected, rtol=0, atol=0)
            index = json.loads((output / "model.safetensors.index.json").read_text())
            assert set(index["weight_map"]) == set(selected)


@pytest.mark.parametrize("distributed_save", [False, True])
@pytest.mark.parametrize("declare_copies", [True, False])
def test_strict_export_excludes_declared_source_copies(tmp_path, distributed_save, declare_copies):
    """A bridge may name checkpoint tensors it never writes back (copies of shared parameters); the shard that holds
    them is then written with its other tensors. Undeclared, the same copies fail the strict export."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    weights = {
        "model.layers.0.weight": torch.ones(2),
        "model.layers.1.weight": torch.full((2,), 2.0),
        "model.layers.2.weight": torch.full((2,), 2.0),  # the checkpoint's copy of layer 1
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(weights, source_dir / shard)
    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in weights}})
    )
    selected = {key: value for key, value in weights.items() if key != "model.layers.2.weight"}

    class SharedBodyBridge:
        def stream_weights_megatron_to_hf(self, *args, **kwargs):
            return iter(selected.items())

    if declare_copies:

        def hf_export_ignored_source_keys(self, hf_pretrained, source_keys):
            assert set(source_keys) == set(weights)
            return [key for key in source_keys if key == "model.layers.2.weight"]

        SharedBodyBridge.hf_export_ignored_source_keys = hf_export_ignored_source_keys

    bridge = object.__new__(AutoBridge)
    bridge.hf_pretrained = SimpleNamespace(
        config=SimpleNamespace(), state=SimpleNamespace(source=SafeTensorsStateSource(source_dir))
    )
    output = tmp_path / "output"
    with (
        patch.object(AutoBridge, "_model_bridge", new_callable=PropertyMock, return_value=SharedBodyBridge()),
        patch.object(AutoBridge, "_get_model_instance", return_value=SimpleNamespace(config=SimpleNamespace())),
        patch("modelopt.torch.quantization.utils.is_quantized", return_value=False),
    ):
        if not declare_copies:
            with pytest.raises(RuntimeError, match="not written|Missing"):
                bridge.save_hf_weights([Mock()], output, strict=True, distributed_save=distributed_save)
        else:
            bridge.save_hf_weights([Mock()], output, strict=True, distributed_save=distributed_save)
            actual = SafeTensorsStateSource(output).load_tensors(list(selected))
            assert set(actual) == set(selected)
            torch.testing.assert_close(actual, selected, rtol=0, atol=0)
            index = json.loads((output / "model.safetensors.index.json").read_text())
            assert set(index["weight_map"]) == set(selected)


@pytest.mark.parametrize("distributed_save", [False, True])
def test_declared_source_copies_must_not_be_yielded(tmp_path, distributed_save):
    """Declaring a tensor the export does write is a bridge bug: strict saving rejects the tensor as unknown."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    weights = {"model.layers.0.weight": torch.ones(2), "model.layers.1.weight": torch.full((2,), 2.0)}
    shard = "model-00001-of-00001.safetensors"
    save_file(weights, source_dir / shard)
    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in weights}})
    )

    class OverdeclaringBridge:
        def stream_weights_megatron_to_hf(self, *args, **kwargs):
            return iter(weights.items())

        def hf_export_ignored_source_keys(self, hf_pretrained, source_keys):
            return ["model.layers.1.weight"]

    bridge = object.__new__(AutoBridge)
    bridge.hf_pretrained = SimpleNamespace(
        config=SimpleNamespace(), state=SimpleNamespace(source=SafeTensorsStateSource(source_dir))
    )
    with (
        patch.object(AutoBridge, "_model_bridge", new_callable=PropertyMock, return_value=OverdeclaringBridge()),
        patch.object(AutoBridge, "_get_model_instance", return_value=SimpleNamespace(config=SimpleNamespace())),
        patch("modelopt.torch.quantization.utils.is_quantized", return_value=False),
        pytest.raises(KeyError, match="not found in the original model structure"),
    ):
        bridge.save_hf_weights([Mock()], tmp_path / "output", strict=True, distributed_save=distributed_save)
