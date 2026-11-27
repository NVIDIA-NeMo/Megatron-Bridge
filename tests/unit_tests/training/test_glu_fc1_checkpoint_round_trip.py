# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native Bridge checkpoints delegate GLU model/optimizer layout to Core factories."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from megatron.core.dist_checkpointing.glu import (
    glu_checkpoint_metadata,
    validate_glu_checkpoint_layout,
)
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory

from megatron.bridge.training.checkpointing import (
    _generate_model_state_dict,
    _validate_native_glu_backend,
)


def _runtime(tensor, size):
    gate, up = tensor.chunk(2)
    return torch.cat([piece for pair in zip(gate.split(size), up.split(size)) for piece in pair])


class _FactoryModel(torch.nn.Module):
    def __init__(self, canonical, *, size, key):
        super().__init__()
        self.weight = torch.nn.Parameter(_runtime(canonical, size))
        self.size = size
        self.key = key

    def sharded_state_dict(self, metadata=None):
        return {
            self.key: apply_swiglu_sharded_factory(
                ShardedTensor.from_rank_offsets(self.key, self.weight), (), glu_interleave_size=self.size
            )
        }


@pytest.mark.parametrize("component,size", [("experts", 2), ("shared_experts", 4), ("dense", 32)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("virtual_chunks", [1, 2])
def test_native_templates_preserve_model_master_and_moment_layout(component, size, bias, virtual_chunks):
    component_prefix = "" if component == "dense" else component + "."
    key = "decoder.layers.0.mlp." + component_prefix + ("linear_fc1.bias" if bias else "linear_fc1.weight")
    shape = (128,) if bias else (128, 3)
    canonical = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    models = [_FactoryModel(canonical + i * 1000, size=size, key=key) for i in range(virtual_chunks)]
    template = _generate_model_state_dict(models, {}, pg_collection=SimpleNamespace(dp_cp=None))
    for i, model in enumerate(models):
        section = "model" if virtual_chunks == 1 else f"model{i}"
        factory = template[section][key]
        # Core associates optimizer state with the original model parameter identity.
        assert factory.data is model.weight
        for name, offset in (("model", 0), ("master", 100), ("exp_avg", 200), ("exp_avg_sq", 300)):
            state_factory = replace(factory, key=name + "." + key, data=model.weight.detach() + offset)
            parts = state_factory.build()
            torch.testing.assert_close(
                torch.cat([part.data for part in parts]), canonical + i * 1000 + offset, rtol=0, atol=0
            )
            loaded = state_factory.merge_fn([part.data.clone() for part in parts])
            torch.testing.assert_close(loaded, model.weight + offset, rtol=0, atol=0)


@pytest.mark.parametrize("format", ["fsdp_dtensor", "torch_dcp", "torch"])
def test_non_native_interleaved_checkpoint_backends_are_rejected(format):
    config = SimpleNamespace(gated_linear_unit=True, moe_mlp_glu_interleave_size=2)
    with pytest.raises(NotImplementedError, match="Core factories"):
        _validate_native_glu_backend(config, format)
    _validate_native_glu_backend(config, "torch_dist")
    _validate_native_glu_backend(SimpleNamespace(gated_linear_unit=False), format)


@pytest.mark.parametrize("load_optimizer", [False, True])
def test_legacy_interleaved_layout_requires_verified_metadata(load_optimizer):
    config = SimpleNamespace(gated_linear_unit=True, moe_mlp_glu_interleave_size=2)
    with pytest.raises(ValueError, match="unmarked interleaved"):
        validate_glu_checkpoint_layout(
            None, source_config=config, target_config=config, loading_optimizer=load_optimizer
        )
    metadata = glu_checkpoint_metadata(config, optimizer_uses_model_layout=False)
    validate_glu_checkpoint_layout(
        metadata, source_config=config, target_config=config, loading_optimizer=load_optimizer
    )
